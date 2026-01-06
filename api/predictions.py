"""Prediction generation and retrieval logic."""

import sys
import os
from datetime import date, datetime
from typing import List
import pytz

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nba_model import EnsembleNBAPredictor
from nba_api.stats.endpoints import scoreboardv2
import pickle

from api.models import GamePrediction
from api.database import get_db, USE_POSTGRES

# Database configuration for team/player stats
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Determine which database to use for team/player stats
if DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://"):
    TEAM_STATS_USE_POSTGRES = True
    import psycopg2
    # Railway uses postgres://, but psycopg2 needs postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
else:
    TEAM_STATS_USE_POSTGRES = False
    import sqlite3


def get_team_stats_connection():
    """Get connection to team stats database (PostgreSQL or SQLite)."""
    if TEAM_STATS_USE_POSTGRES:
        return psycopg2.connect(DATABASE_URL)
    else:
        return sqlite3.connect('nba_team_stats.db')

# Load model on startup
ENSEMBLE = None
CALIBRATION = None


def load_model():
    """Load ensemble model and calibration."""
    global ENSEMBLE, CALIBRATION
    
    if ENSEMBLE is None:
        print("Loading ensemble model...")
        ENSEMBLE = EnsembleNBAPredictor()
        model_path = os.getenv("MODEL_PATH", "ensemble_model_saved")
        ENSEMBLE.load_ensemble(model_path)
        print("✅ Ensemble loaded")
    
    if CALIBRATION is None:
        cal_path = os.getenv("CALIBRATION_PATH", "calibration_saved/calibration.pkl")
        if os.path.exists(cal_path):
            print("Loading calibration...")
            with open(cal_path, 'rb') as f:
                CALIBRATION = pickle.load(f)
            print("✅ Calibration loaded")


def apply_calibration(raw_prob: float) -> float:
    """Apply calibration to probability."""
    if CALIBRATION is None:
        return raw_prob
    
    import numpy as np
    method = CALIBRATION['method']
    
    if method == 'isotonic':
        calibrated = CALIBRATION['calibrator'].transform([raw_prob])[0]
    elif method == 'platt':
        raw_clipped = np.clip(raw_prob, 1e-7, 1 - 1e-7)
        logit = np.log(raw_clipped / (1 - raw_clipped))
        calibrated = CALIBRATION['calibrator'].predict_proba([[logit]])[0, 1]
    elif method == 'offset':
        calibrated = raw_prob + CALIBRATION['offset']
        calibrated = np.clip(calibrated, 0.0, 1.0)
    else:
        calibrated = raw_prob
    
    # Apply temperature scaling to reduce overconfidence
    # Temperature > 1 makes predictions less extreme (more conservative)
    temperature = 1.5
    calibrated = np.clip(calibrated, 1e-7, 1 - 1e-7)  # Avoid log(0)
    logit = np.log(calibrated / (1 - calibrated))
    scaled_logit = logit / temperature
    calibrated = 1 / (1 + np.exp(-scaled_logit))
    
    # Final clamp to reasonable range
    calibrated = np.clip(calibrated, 0.01, 0.99)
    
    return float(calibrated)


def get_todays_predictions() -> List[GamePrediction]:
    """Generate predictions for today's games."""
    load_model()
    
    # Get today's games from NBA API
    # Use Eastern timezone since NBA schedules are in ET
    eastern = pytz.timezone('America/New_York')
    today = datetime.now(eastern).date()
    board = scoreboardv2.ScoreboardV2(game_date=today.strftime('%m/%d/%Y'))
    game_header_df = board.game_header.get_data_frame()
    line_score_df = board.line_score.get_data_frame()
    
    if game_header_df.empty:
        return []
    
    # Deduplicate game_header (NBA API sometimes returns duplicates)
    game_header_df = game_header_df.drop_duplicates(subset=['GAME_ID'])
    
    predictions = []
    conn = get_team_stats_connection()
    
    for _, game_row in game_header_df.iterrows():
        game_id = game_row['GAME_ID']
        home_team_id = game_row['HOME_TEAM_ID']
        visitor_team_id = game_row['VISITOR_TEAM_ID']
        
        # Get team info from line_score using the correct home/away IDs
        game_teams = line_score_df[line_score_df['GAME_ID'] == game_id]
        
        if len(game_teams) >= 2:
            home_team = game_teams[game_teams['TEAM_ID'] == home_team_id].iloc[0]
            away_team = game_teams[game_teams['TEAM_ID'] == visitor_team_id].iloc[0]
            
            home_id = int(home_team['TEAM_ID'])
            away_id = int(away_team['TEAM_ID'])
            home_abbr = home_team['TEAM_ABBREVIATION']
            away_abbr = away_team['TEAM_ABBREVIATION']
            
            # Get prediction
            pred = ENSEMBLE.predict(home_id, away_id, seasons=['2022-23', '2023-24', '2024-25', '2025-26'])
            
            raw_home_prob = pred['home_win_probability']
            cal_home_prob = apply_calibration(raw_home_prob)
            
            predictions.append(GamePrediction(
                date=today.isoformat(),
                game_id=game_id,
                home_team=home_abbr,
                away_team=away_abbr,
                predicted_home_prob=cal_home_prob,
                predicted_away_prob=1 - cal_home_prob,
                predicted_home_score=pred['predicted_home_points'],
                predicted_away_score=pred['predicted_away_points']
            ))
    
    conn.close()
    return predictions


def get_predictions_for_date(target_date: date) -> List[GamePrediction]:
    """Get predictions from database for a specific date."""
    db = get_db()
    cursor = db.cursor()
    
    from api.database import USE_POSTGRES
    placeholder = "%s" if USE_POSTGRES else "?"
    
    cursor.execute(f"""
        SELECT 
            date, game_id, home_team_abbr, away_team_abbr,
            predicted_home_prob, predicted_away_prob,
            pred_home_points, pred_away_points,
            actual_home_score, actual_away_score, actual_home_win, correct
        FROM predictions
        WHERE date = {placeholder}
        ORDER BY game_id
    """, (target_date.isoformat(),))
    
    rows = cursor.fetchall()
    predictions = []
    
    for row in rows:
        predictions.append(GamePrediction(
            date=row[0],
            game_id=row[1],
            home_team=row[2],
            away_team=row[3],
            predicted_home_prob=row[4],
            predicted_away_prob=row[5],
            predicted_home_score=row[6],
            predicted_away_score=row[7],
            actual_home_score=row[8],
            actual_away_score=row[9],
            actual_home_win=row[10],
            correct=row[11]
        ))
    
    return predictions


def save_predictions_to_db(predictions: List[GamePrediction]) -> None:
    """Save predictions to database."""
    db = get_db()
    cursor = db.cursor()
    
    try:
        for pred in predictions:
            if USE_POSTGRES:
                # PostgreSQL: Use ON CONFLICT and %s placeholders
                cursor.execute("""
                    INSERT INTO predictions (
                        date, game_id, home_team_abbr, away_team_abbr,
                        home_team_id, away_team_id,
                        predicted_home_prob, predicted_away_prob,
                        pred_home_points, pred_away_points
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date, game_id) DO UPDATE SET
                        predicted_home_prob = EXCLUDED.predicted_home_prob,
                        predicted_away_prob = EXCLUDED.predicted_away_prob,
                        pred_home_points = EXCLUDED.pred_home_points,
                        pred_away_points = EXCLUDED.pred_away_points
                """, (
                    pred.date, pred.game_id, pred.home_team, pred.away_team,
                    None, None,
                    pred.predicted_home_prob, pred.predicted_away_prob,
                    pred.predicted_home_score, pred.predicted_away_score
                ))
            else:
                # SQLite: Use INSERT OR REPLACE and ? placeholders
                cursor.execute("""
                    INSERT OR REPLACE INTO predictions (
                        date, game_id, home_team_abbr, away_team_abbr,
                        home_team_id, away_team_id,
                        predicted_home_prob, predicted_away_prob,
                        pred_home_points, pred_away_points
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pred.date, pred.game_id, pred.home_team, pred.away_team,
                    None, None,
                    pred.predicted_home_prob, pred.predicted_away_prob,
                    pred.predicted_home_score, pred.predicted_away_score
                ))
        
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        cursor.close()
