#!/usr/bin/env python3
"""Log predictions vs actual outcomes using NBA Live API for today's games.

Usage:
  python3 log_predictions.py [--ensemble] [--date 2025-12-03]
  python3 log_predictions.py [--ensemble] [--calibrated]  # Use calibration

Defaults to today. Fetches live games and fills in results for finished games.
Writes CSV to `prediction_logs/predictions_<date>_<mode>.csv`.
"""

import argparse
import os
import csv
from datetime import datetime, date
import sqlite3
import pickle
import time

from nba_model import NBAGamePredictor
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.endpoints import boxscoretraditionalv3


def load_ensemble_if_requested(path='ensemble_model_saved'):
    try:
        from nba_model import EnsembleNBAPredictor
        ens = EnsembleNBAPredictor()
        ens.load_ensemble(path)
        return ens
    except Exception as e:
        raise RuntimeError(f"Failed to load ensemble: {e}")


def get_team_id(abbr):
    """Helper to get team ID from abbreviation."""
    conn = sqlite3.connect('nba_team_stats.db')
    cur = conn.execute("SELECT team_id FROM teams WHERE abbreviation = ?", (abbr,))
    result = cur.fetchone()
    conn.close()
    return result[0] if result else None


def get_game_rosters(game_id):
    """Get player roster for a specific game."""
    try:
        boxscore = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
        player_stats = boxscore.player_stats.get_data_frame()
        return player_stats
    except Exception as e:
        return None


def extract_active_roster(player_stats_df, team_abbr):
    """Extract names of active players from boxscore dataframe."""
    if player_stats_df is None:
        return None
    
    # Handle both V2 and V3 column names
    team_col = 'teamTricode' if 'teamTricode' in player_stats_df.columns else 'TEAM_ABBREVIATION'
    player_col = 'nameI' if 'nameI' in player_stats_df.columns else 'PLAYER_NAME'
    min_col = 'minutes' if 'minutes' in player_stats_df.columns else 'MIN'
    
    team_players = player_stats_df[player_stats_df[team_col] == team_abbr]
    
    if team_players.empty:
        return None
    
    # Get players who played (have minutes > 0)
    if min_col in team_players.columns:
        # V3 uses actual minute values (float), V2 uses string format
        if player_stats_df[min_col].dtype == 'object':
            active = team_players[team_players[min_col].notna() & (team_players[min_col] != '0:00')]
        else:
            active = team_players[team_players[min_col].notna() & (team_players[min_col] > 0)]
    else:
        active = team_players
    
    if len(active) > 0:
        return active[player_col].tolist()
    else:
        # If game hasn't started, return all listed players
        return team_players[player_col].tolist() if len(team_players) > 0 else None


def load_calibration(path='calibration_saved/calibration.pkl'):
    """Load calibration model if available."""
    if not os.path.exists(path):
        return None
    
    try:
        with open(path, 'rb') as f:
            cal_data = pickle.load(f)
        return cal_data
    except Exception as e:
        print(f"⚠️  Could not load calibration: {e}")
        return None


def apply_calibration(raw_prob, cal_data):
    """Apply calibration to a probability."""
    if cal_data is None:
        return raw_prob
    
    import numpy as np
    
    method = cal_data['method']
    
    if method == 'isotonic':
        calibrated = cal_data['calibrator'].transform([raw_prob])[0]
    elif method == 'platt':
        raw_clipped = np.clip(raw_prob, 1e-7, 1 - 1e-7)
        logit = np.log(raw_clipped / (1 - raw_clipped))
        calibrated = cal_data['calibrator'].predict_proba([[logit]])[0, 1]
    elif method == 'offset':
        calibrated = raw_prob + cal_data['offset']
        calibrated = np.clip(calibrated, 0.0, 1.0)
    else:
        calibrated = raw_prob
    
    return float(calibrated)


def main():
    parser = argparse.ArgumentParser(description='Log model predictions vs actuals using Live API')
    parser.add_argument('--date', type=str, default=None, help='Date (YYYY-MM-DD); defaults to today')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble instead of single model')
    parser.add_argument('--calibrated', action='store_true', help='Apply calibration to predictions')
    parser.add_argument('--rosters', action='store_true', help='Use live game rosters (default: enabled)')
    parser.add_argument('--output-dir', type=str, default='prediction_logs', help='Directory to write logs')
    parser.add_argument('--ensemble-path', type=str, default='ensemble_model_saved', help='Ensemble model path')
    parser.add_argument('--model-path', type=str, default='nba_model_saved', help='Single model path')
    parser.add_argument('--calibration-path', type=str, default='calibration_saved/calibration.pkl',
                       help='Path to calibration file')
    args = parser.parse_args()
    
    # Enable rosters by default
    if not hasattr(args, 'rosters') or args.rosters is None:
        args.rosters = True

    # Determine date
    if args.date:
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        except Exception as e:
            raise SystemExit(f"Invalid date format: {e}")
    else:
        target_date = date.today()

    date_str = target_date.strftime('%Y-%m-%d')
    os.makedirs(args.output_dir, exist_ok=True)
    mode = 'ensemble' if args.ensemble else 'single'
    if args.calibrated:
        mode += '_calibrated'
    out_path = os.path.join(args.output_dir, f'predictions_{date_str}_{mode}.csv')

    print(f"{'='*80}")
    print(f"LOGGING PREDICTIONS FOR {date_str}")
    print(f"{'='*80}\n")

    # Load calibration if requested
    cal_data = None
    if args.calibrated:
        cal_data = load_calibration(args.calibration_path)
        if cal_data:
            print(f"✅ Loaded calibration (method: {cal_data['method']}, offset: {cal_data['offset']:+.3f})")
        else:
            print("⚠️  Calibration requested but not found. Using raw predictions.")
            print("   Run: python3.11 calibrate_model.py --ensemble --from-logs")

    # Load model(s)
    if args.ensemble:
        print(f"Loading ensemble from {args.ensemble_path}...")
        ensemble = load_ensemble_if_requested(args.ensemble_path)
        predictor = ensemble
    else:
        print(f"Loading single model from {args.model_path}...")
        predictor = NBAGamePredictor(model_path=args.model_path)

    # Use NBA Stats API to get games
    print(f"Fetching games via NBA Stats API for {date_str}...")
    board = scoreboardv2.ScoreboardV2(game_date=target_date.strftime('%m/%d/%Y'))
    
    # Get game header and line score data
    game_header_df = board.game_header.get_data_frame()
    line_score_df = board.line_score.get_data_frame()
    
    if game_header_df.empty:
        print(f"No games found for {date_str}")
        return
    
    print(f"Found {len(game_header_df)} games\n")
    
    # Build games list
    games_list = []
    conn = sqlite3.connect('nba_team_stats.db')
    
    for _, game_row in game_header_df.iterrows():
        game_id = game_row['GAME_ID']
        home_team_id = game_row['HOME_TEAM_ID']
        away_team_id = game_row['VISITOR_TEAM_ID']
        game_status_text = game_row['GAME_STATUS_TEXT']
        
        # Get team abbreviations from database
        home_abbr = conn.execute("SELECT abbreviation FROM teams WHERE team_id = ?", 
                                (home_team_id,)).fetchone()
        away_abbr = conn.execute("SELECT abbreviation FROM teams WHERE team_id = ?", 
                                (away_team_id,)).fetchone()
        
        home_tricode = home_abbr[0] if home_abbr else 'UNK'
        away_tricode = away_abbr[0] if away_abbr else 'UNK'
        
        # Get scores from line_score if game is final
        actual_home_score = None
        actual_away_score = None
        actual_home_win = None
        
        if 'Final' in game_status_text:
            # Find scores in line_score_df
            home_line = line_score_df[(line_score_df['GAME_ID'] == game_id) & 
                                     (line_score_df['TEAM_ID'] == home_team_id)]
            away_line = line_score_df[(line_score_df['GAME_ID'] == game_id) & 
                                     (line_score_df['TEAM_ID'] == away_team_id)]
            
            if not home_line.empty and not away_line.empty:
                actual_home_score = int(home_line.iloc[0]['PTS'])
                actual_away_score = int(away_line.iloc[0]['PTS'])
                actual_home_win = 1 if actual_home_score > actual_away_score else 0
        
        games_list.append({
            'game_id': game_id,
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'home_team_abbr': home_tricode,
            'away_team_abbr': away_tricode,
            'actual_home_score': actual_home_score,
            'actual_away_score': actual_away_score,
            'actual_home_win': actual_home_win
        })
    
    conn.close()
    
    # Make predictions and write CSV
    fieldnames = [
        'date', 'game_id', 'home_team_abbr', 'away_team_abbr',
        'home_team_id', 'away_team_id',
        'predicted_home_prob', 'predicted_away_prob',
        'pred_home_points', 'pred_away_points',
        'actual_home_score', 'actual_away_score', 'actual_home_win', 'correct'
    ]
    
    # Add calibrated prob columns if using calibration
    if args.calibrated:
        fieldnames = [
            'date', 'game_id', 'home_team_abbr', 'away_team_abbr',
            'home_team_id', 'away_team_id',
            'raw_home_prob', 'raw_away_prob',
            'calibrated_home_prob', 'calibrated_away_prob',
            'pred_home_points', 'pred_away_points',
            'actual_home_score', 'actual_away_score', 'actual_home_win', 'correct'
        ]
    
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for game in games_list:
            print(f"Game: {game['away_team_abbr']} @ {game['home_team_abbr']}")
            
            # Try to get rosters if enabled
            home_roster = None
            away_roster = None
            
            if args.rosters:
                try:
                    player_stats = get_game_rosters(game['game_id'])
                    if player_stats is not None:
                        home_roster = extract_active_roster(player_stats, game['home_team_abbr'])
                        away_roster = extract_active_roster(player_stats, game['away_team_abbr'])
                except:
                    pass
                time.sleep(0.6)  # Rate limiting
            
            # Make prediction
            try:
                prediction = predictor.predict(
                    game['home_team_id'],
                    game['away_team_id'],
                    seasons=['2022-23', '2023-24', '2024-25', '2025-26'],
                    home_roster=home_roster,
                    away_roster=away_roster
                )
                
                raw_home_prob = prediction['home_win_probability']
                raw_away_prob = prediction['away_win_probability']
                pred_home_points = prediction['predicted_home_points']
                pred_away_points = prediction['predicted_away_points']
                
                # Apply calibration if available
                if cal_data:
                    cal_home_prob = apply_calibration(raw_home_prob, cal_data)
                    cal_away_prob = 1 - cal_home_prob
                    pred_home_prob = cal_home_prob
                    pred_away_prob = cal_away_prob
                    print(f"  Raw: {game['home_team_abbr']} {raw_home_prob:.1%}")
                    print(f"  Calibrated: {game['home_team_abbr']} {cal_home_prob:.1%} ({pred_home_points:.1f} pts)")
                else:
                    pred_home_prob = raw_home_prob
                    pred_away_prob = raw_away_prob
                    print(f"  Prediction: {game['home_team_abbr']} {pred_home_prob:.1%} ({pred_home_points:.1f} pts)")
                
                # Calculate correctness if actual result available
                correct = None
                if game['actual_home_win'] is not None:
                    predicted_home_win = 1 if pred_home_prob >= 0.5 else 0
                    correct = 1 if predicted_home_win == game['actual_home_win'] else 0
                    print(f"  Actual: {game['home_team_abbr']} {game['actual_home_score']} - {game['away_team_abbr']} {game['actual_away_score']} ({'CORRECT' if correct else 'WRONG'})")
                
                if args.calibrated:
                    writer.writerow({
                        'date': date_str,
                        'game_id': game['game_id'],
                        'home_team_abbr': game['home_team_abbr'],
                        'away_team_abbr': game['away_team_abbr'],
                        'home_team_id': game['home_team_id'],
                        'away_team_id': game['away_team_id'],
                        'raw_home_prob': raw_home_prob,
                        'raw_away_prob': raw_away_prob,
                        'calibrated_home_prob': pred_home_prob,
                        'calibrated_away_prob': pred_away_prob,
                        'pred_home_points': pred_home_points,
                        'pred_away_points': pred_away_points,
                        'actual_home_score': game['actual_home_score'] if game['actual_home_score'] is not None else '',
                        'actual_away_score': game['actual_away_score'] if game['actual_away_score'] is not None else '',
                        'actual_home_win': game['actual_home_win'] if game['actual_home_win'] is not None else '',
                        'correct': correct if correct is not None else ''
                    })
                else:
                    writer.writerow({
                        'date': date_str,
                        'game_id': game['game_id'],
                        'home_team_abbr': game['home_team_abbr'],
                        'away_team_abbr': game['away_team_abbr'],
                        'home_team_id': game['home_team_id'],
                        'away_team_id': game['away_team_id'],
                        'predicted_home_prob': pred_home_prob,
                        'predicted_away_prob': pred_away_prob,
                        'pred_home_points': pred_home_points,
                        'pred_away_points': pred_away_points,
                        'actual_home_score': game['actual_home_score'] if game['actual_home_score'] is not None else '',
                        'actual_away_score': game['actual_away_score'] if game['actual_away_score'] is not None else '',
                        'actual_home_win': game['actual_home_win'] if game['actual_home_win'] is not None else '',
                        'correct': correct if correct is not None else ''
                    })
                
            except Exception as e:
                print(f"  Error predicting: {e}")
            
            print()
    
    print(f"{'='*80}")
    print(f"✅ Saved to: {out_path}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
