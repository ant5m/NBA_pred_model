#!/usr/bin/env python3
"""Calibrate NBA prediction model probabilities to correct systematic bias.

This script:
1. Loads a trained model/ensemble
2. Collects predictions on a validation set
3. Fits isotonic regression to calibrate probabilities
4. Saves calibration model for use in predictions
5. Provides simple offset correction as fallback

Usage:
  python3 calibrate_model.py --ensemble
  python3 calibrate_model.py --single
  python3 calibrate_model.py --from-logs  # Use recent prediction logs
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import sqlite3
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from nba_model import NBAGamePredictor, EnsembleNBAPredictor


def collect_validation_predictions(predictor, seasons=['2023-24'], max_games=500):
    """Collect predictions on validation games."""
    conn = sqlite3.connect('nba_team_stats.db')
    
    # Get validation games
    if isinstance(seasons, str):
        seasons = [seasons]
    
    placeholders = ','.join(['?'] * len(seasons))
    query = f"""
        SELECT DISTINCT
            g1.game_id,
            g1.team_id as home_team_id,
            g2.team_id as away_team_id,
            g1.win_loss as home_win,
            g1.game_date
        FROM game_logs g1
        JOIN game_logs g2 ON g1.game_id = g2.game_id AND g1.team_id != g2.team_id
        WHERE g1.matchup LIKE '%vs%'
            AND g1.season IN ({placeholders})
        ORDER BY g1.game_date DESC
        LIMIT ?
    """
    games_df = pd.read_sql_query(query, conn, params=(*seasons, max_games))
    conn.close()
    
    print(f"Collecting predictions for {len(games_df)} validation games...")
    
    pred_probs = []
    actuals = []
    
    for idx, game in games_df.iterrows():
        if idx % 50 == 0:
            print(f"  {idx}/{len(games_df)}...")
        
        try:
            prediction = predictor.predict(
                game['home_team_id'],
                game['away_team_id'],
                seasons=['2022-23', '2023-24', '2024-25', '2025-26']
            )
            
            pred_probs.append(prediction['home_win_probability'])
            actuals.append(1 if game['home_win'] == 'W' else 0)
            
        except Exception as e:
            print(f"  Error on game {game['game_id']}: {e}")
            continue
    
    return np.array(pred_probs), np.array(actuals)


def collect_from_prediction_logs(log_dir='prediction_logs'):
    """Collect predictions from existing CSV logs."""
    import glob
    import csv
    
    files = sorted(glob.glob(f'{log_dir}/predictions_*.csv'))
    
    if not files:
        print(f"No prediction logs found in {log_dir}/")
        return None, None
    
    print(f"Loading predictions from {len(files)} log files...")
    
    pred_probs = []
    actuals = []
    
    for file in files:
        with open(file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    prob = float(row['predicted_home_prob'])
                    
                    # Only include if we have actual result
                    if row.get('actual_home_score') and row.get('actual_away_score'):
                        home_score = int(row['actual_home_score'])
                        away_score = int(row['actual_away_score'])
                        actual = 1 if home_score > away_score else 0
                        
                        pred_probs.append(prob)
                        actuals.append(actual)
                except (ValueError, KeyError):
                    continue
    
    print(f"Collected {len(pred_probs)} predictions with actual results")
    
    return np.array(pred_probs) if pred_probs else None, np.array(actuals) if actuals else None


def train_isotonic_calibration(pred_probs, actuals):
    """Train isotonic regression calibrator."""
    print("\nTraining isotonic calibration...")
    
    cal = IsotonicRegression(out_of_bounds='clip')
    cal.fit(pred_probs, actuals)
    
    # Evaluate calibration
    calibrated_probs = cal.transform(pred_probs)
    
    # Compute metrics
    from sklearn.metrics import brier_score_loss, log_loss
    
    brier_before = brier_score_loss(actuals, pred_probs)
    brier_after = brier_score_loss(actuals, calibrated_probs)
    
    try:
        logloss_before = log_loss(actuals, pred_probs)
        logloss_after = log_loss(actuals, calibrated_probs)
    except:
        logloss_before = logloss_after = None
    
    print(f"  Brier score before: {brier_before:.4f}")
    print(f"  Brier score after:  {brier_after:.4f}")
    if logloss_before:
        print(f"  Log loss before: {logloss_before:.4f}")
        print(f"  Log loss after:  {logloss_after:.4f}")
    
    return cal


def train_platt_calibration(pred_probs, actuals):
    """Train Platt scaling (logistic regression) calibrator."""
    print("\nTraining Platt scaling calibration...")
    
    # Convert to log-odds
    pred_probs_clipped = np.clip(pred_probs, 1e-7, 1 - 1e-7)
    logits = np.log(pred_probs_clipped / (1 - pred_probs_clipped)).reshape(-1, 1)
    
    cal = LogisticRegression()
    cal.fit(logits, actuals)
    
    # Evaluate
    calibrated_probs = cal.predict_proba(logits)[:, 1]
    
    from sklearn.metrics import brier_score_loss
    brier_before = brier_score_loss(actuals, pred_probs)
    brier_after = brier_score_loss(actuals, calibrated_probs)
    
    print(f"  Brier score before: {brier_before:.4f}")
    print(f"  Brier score after:  {brier_after:.4f}")
    
    return cal


def compute_offset_correction(pred_probs, actuals):
    """Compute simple offset correction."""
    avg_pred = np.mean(pred_probs)
    avg_actual = np.mean(actuals)
    offset = avg_actual - avg_pred
    
    print(f"\nOffset correction:")
    print(f"  Average predicted probability: {avg_pred:.4f}")
    print(f"  Average actual win rate: {avg_actual:.4f}")
    print(f"  Offset: {offset:+.4f}")
    
    return offset


def save_calibration(calibrator, method, offset, output_dir='calibration_saved'):
    """Save calibration model."""
    os.makedirs(output_dir, exist_ok=True)
    
    calibration_data = {
        'method': method,
        'calibrator': calibrator,
        'offset': offset
    }
    
    output_path = os.path.join(output_dir, 'calibration.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(calibration_data, f)
    
    print(f"\n✅ Calibration saved to {output_path}")
    
    return output_path


def apply_calibration(raw_prob, calibration_path='calibration_saved/calibration.pkl'):
    """Apply saved calibration to a probability."""
    with open(calibration_path, 'rb') as f:
        cal_data = pickle.load(f)
    
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
    
    return calibrated


def main():
    parser = argparse.ArgumentParser(description='Calibrate NBA prediction model')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble model')
    parser.add_argument('--single', action='store_true', help='Use single model')
    parser.add_argument('--from-logs', action='store_true', help='Use prediction logs instead of validation set')
    parser.add_argument('--method', type=str, default='isotonic', 
                       choices=['isotonic', 'platt', 'offset', 'all'],
                       help='Calibration method')
    parser.add_argument('--val-seasons', type=str, nargs='+', default=['2023-24'],
                       help='Validation seasons')
    parser.add_argument('--max-games', type=int, default=500,
                       help='Max validation games')
    parser.add_argument('--ensemble-path', type=str, default='ensemble_model_saved')
    parser.add_argument('--model-path', type=str, default='nba_model_saved')
    parser.add_argument('--output-dir', type=str, default='calibration_saved')
    
    args = parser.parse_args()
    
    print("="*80)
    print("NBA MODEL CALIBRATION")
    print("="*80)
    
    # Load model
    if args.from_logs:
        print("\nUsing prediction logs for calibration...")
        pred_probs, actuals = collect_from_prediction_logs()
        
        if pred_probs is None or len(pred_probs) < 10:
            print("ERROR: Not enough predictions in logs. Need at least 10 games with results.")
            return
    else:
        if args.ensemble:
            print(f"\nLoading ensemble from {args.ensemble_path}...")
            predictor = EnsembleNBAPredictor()
            predictor.load_ensemble(args.ensemble_path)
        elif args.single:
            print(f"\nLoading single model from {args.model_path}...")
            predictor = NBAGamePredictor(model_path=args.model_path)
        else:
            print("ERROR: Specify --ensemble or --single")
            return
        
        # Collect validation predictions
        pred_probs, actuals = collect_validation_predictions(
            predictor,
            seasons=args.val_seasons,
            max_games=args.max_games
        )
    
    if len(pred_probs) == 0:
        print("ERROR: No validation data collected")
        return
    
    print(f"\nValidation set: {len(pred_probs)} games")
    print(f"  Home team won: {np.sum(actuals)} ({np.mean(actuals):.1%})")
    print(f"  Avg predicted home prob: {np.mean(pred_probs):.1%}")
    
    # Compute offset
    offset = compute_offset_correction(pred_probs, actuals)
    
    # Train calibrators
    calibrator = None
    
    if args.method == 'isotonic' or args.method == 'all':
        calibrator = train_isotonic_calibration(pred_probs, actuals)
        method = 'isotonic'
    
    if args.method == 'platt' or args.method == 'all':
        platt_cal = train_platt_calibration(pred_probs, actuals)
        if args.method == 'platt':
            calibrator = platt_cal
            method = 'platt'
    
    if args.method == 'offset':
        method = 'offset'
    
    # Save calibration
    if args.method != 'all':
        save_calibration(calibrator, method, offset, args.output_dir)
    else:
        # Save all methods
        save_calibration(calibrator, 'isotonic', offset, 
                        os.path.join(args.output_dir, 'isotonic'))
        save_calibration(platt_cal, 'platt', offset,
                        os.path.join(args.output_dir, 'platt'))
        save_calibration(None, 'offset', offset,
                        os.path.join(args.output_dir, 'offset'))
    
    print("\n" + "="*80)
    print("CALIBRATION COMPLETE")
    print("="*80)
    print("\nTo use calibration in predictions:")
    print("  from calibrate_model import apply_calibration")
    print("  calibrated_prob = apply_calibration(raw_prob)")


if __name__ == '__main__':
    main()
