#!/usr/bin/env python3
"""
Comprehensive Model Testing with Advanced Analytics
Provides detailed metrics: F1, Precision, Recall, MAE, MSE, RMSE, R², Confusion Matrix, etc.
"""

import argparse
import numpy as np
import pickle
import os
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score,
    brier_score_loss, log_loss
)
from nba_model import NBAGamePredictor, EnsembleNBAPredictor
import sys

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def calculate_metrics(y_true, y_pred_proba, y_pred_binary, home_true, away_true, home_pred, away_pred):
    """Calculate comprehensive metrics for binary classification and regression"""
    
    metrics = {}
    
    # ========== CLASSIFICATION METRICS ==========
    metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
    metrics['precision'] = precision_score(y_true, y_pred_binary)
    metrics['recall'] = recall_score(y_true, y_pred_binary)
    metrics['f1_score'] = f1_score(y_true, y_pred_binary)
    metrics['specificity'] = recall_score(y_true, y_pred_binary, pos_label=0)
    
    # AUC-ROC
    try:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
    except:
        metrics['auc_roc'] = None
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives'] = tp
    
    # Additional classification metrics
    metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
    try:
        metrics['log_loss'] = log_loss(y_true, y_pred_proba)
    except:
        metrics['log_loss'] = None
    
    # ========== REGRESSION METRICS (for point predictions) ==========
    # Home team points
    metrics['mae_home'] = mean_absolute_error(home_true, home_pred)
    metrics['mse_home'] = mean_squared_error(home_true, home_pred)
    metrics['rmse_home'] = np.sqrt(metrics['mse_home'])
    metrics['r2_home'] = r2_score(home_true, home_pred)
    
    # Away team points
    metrics['mae_away'] = mean_absolute_error(away_true, away_pred)
    metrics['mse_away'] = mean_squared_error(away_true, away_pred)
    metrics['rmse_away'] = np.sqrt(metrics['mse_away'])
    metrics['r2_away'] = r2_score(away_true, away_pred)
    
    # Combined metrics
    all_true = np.concatenate([home_true, away_true])
    all_pred = np.concatenate([home_pred, away_pred])
    metrics['mae_combined'] = mean_absolute_error(all_true, all_pred)
    metrics['mse_combined'] = mean_squared_error(all_true, all_pred)
    metrics['rmse_combined'] = np.sqrt(metrics['mse_combined'])
    metrics['r2_combined'] = r2_score(all_true, all_pred)
    
    # Point spread metrics
    true_spread = home_true - away_true
    pred_spread = home_pred - away_pred
    metrics['mae_spread'] = mean_absolute_error(true_spread, pred_spread)
    metrics['mse_spread'] = mean_squared_error(true_spread, pred_spread)
    metrics['rmse_spread'] = np.sqrt(metrics['mse_spread'])
    
    return metrics

def print_classification_metrics(metrics):
    """Print classification metrics"""
    print_section("CLASSIFICATION METRICS (Win/Loss Prediction)")
    
    print(f"\n🎯 Accuracy Metrics:")
    print(f"   Overall Accuracy:  {metrics['accuracy']:.1%}")
    print(f"   Precision:         {metrics['precision']:.1%}")
    print(f"   Recall (TPR):      {metrics['recall']:.1%}")
    print(f"   Specificity (TNR): {metrics['specificity']:.1%}")
    print(f"   F1 Score:          {metrics['f1_score']:.4f}")
    
    if metrics['auc_roc']:
        print(f"   AUC-ROC:           {metrics['auc_roc']:.4f}")
    
    print(f"\n📊 Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                Away Win  Home Win")
    print(f"   Actual Away    {metrics['true_negatives']:6d}    {metrics['false_positives']:6d}   (TN & FP)")
    print(f"   Actual Home    {metrics['false_negatives']:6d}    {metrics['true_positives']:6d}   (FN & TP)")
    
    total = sum([metrics['true_negatives'], metrics['false_positives'], 
                 metrics['false_negatives'], metrics['true_positives']])
    print(f"\n   True Negatives:  {metrics['true_negatives']:4d} ({metrics['true_negatives']/total:.1%})")
    print(f"   False Positives: {metrics['false_positives']:4d} ({metrics['false_positives']/total:.1%})")
    print(f"   False Negatives: {metrics['false_negatives']:4d} ({metrics['false_negatives']/total:.1%})")
    print(f"   True Positives:  {metrics['true_positives']:4d} ({metrics['true_positives']/total:.1%})")
    
    print(f"\n🎲 Probability Metrics:")
    print(f"   Brier Score:       {metrics['brier_score']:.4f} (lower is better)")
    if metrics['log_loss']:
        print(f"   Log Loss:          {metrics['log_loss']:.4f} (lower is better)")

def print_regression_metrics(metrics):
    """Print regression metrics"""
    print_section("REGRESSION METRICS (Point Predictions)")
    
    print(f"\n🏠 Home Team Points:")
    print(f"   MAE (Mean Absolute Error):     {metrics['mae_home']:.2f} points")
    print(f"   MSE (Mean Squared Error):      {metrics['mse_home']:.2f}")
    print(f"   RMSE (Root Mean Squared Error):{metrics['rmse_home']:.2f} points")
    print(f"   R² Score:                      {metrics['r2_home']:.4f}")
    
    print(f"\n✈️  Away Team Points:")
    print(f"   MAE (Mean Absolute Error):     {metrics['mae_away']:.2f} points")
    print(f"   MSE (Mean Squared Error):      {metrics['mse_away']:.2f}")
    print(f"   RMSE (Root Mean Squared Error):{metrics['rmse_away']:.2f} points")
    print(f"   R² Score:                      {metrics['r2_away']:.4f}")
    
    print(f"\n📊 Combined (All Predictions):")
    print(f"   MAE (Mean Absolute Error):     {metrics['mae_combined']:.2f} points")
    print(f"   MSE (Mean Squared Error):      {metrics['mse_combined']:.2f}")
    print(f"   RMSE (Root Mean Squared Error):{metrics['rmse_combined']:.2f} points")
    print(f"   R² Score:                      {metrics['r2_combined']:.4f}")
    
    print(f"\n📈 Point Spread:")
    print(f"   MAE (Spread Error):            {metrics['mae_spread']:.2f} points")
    print(f"   MSE (Spread Error):            {metrics['mse_spread']:.2f}")
    print(f"   RMSE (Spread Error):           {metrics['rmse_spread']:.2f} points")

def print_additional_analytics(y_true, y_pred_proba, home_true, away_true, home_pred, away_pred):
    """Print additional analytics and insights"""
    print_section("ADDITIONAL ANALYTICS")
    
    # Home team bias
    actual_home_win_rate = np.mean(y_true)
    predicted_home_win_rate = np.mean(y_pred_proba > 0.5)
    home_bias = predicted_home_win_rate - actual_home_win_rate
    
    print(f"\n🏠 Home Court Advantage:")
    print(f"   Actual home win rate:     {actual_home_win_rate:.1%}")
    print(f"   Predicted home win rate:  {predicted_home_win_rate:.1%}")
    print(f"   Model bias:               {home_bias:+.1%}")
    
    # Confidence analysis
    high_conf = np.sum((y_pred_proba > 0.6) | (y_pred_proba < 0.4))
    low_conf = np.sum((y_pred_proba >= 0.4) & (y_pred_proba <= 0.6))
    
    print(f"\n🎯 Prediction Confidence:")
    print(f"   High confidence (>60% or <40%): {high_conf} games ({high_conf/len(y_pred_proba):.1%})")
    print(f"   Low confidence (40-60%):        {low_conf} games ({low_conf/len(y_pred_proba):.1%})")
    
    # Accuracy by confidence level
    if high_conf > 0:
        high_conf_mask = (y_pred_proba > 0.6) | (y_pred_proba < 0.4)
        y_pred_binary_high = (y_pred_proba[high_conf_mask] > 0.5).astype(int)
        acc_high = accuracy_score(y_true[high_conf_mask], y_pred_binary_high)
        print(f"   Accuracy on high confidence:    {acc_high:.1%}")
    
    if low_conf > 0:
        low_conf_mask = (y_pred_proba >= 0.4) & (y_pred_proba <= 0.6)
        y_pred_binary_low = (y_pred_proba[low_conf_mask] > 0.5).astype(int)
        acc_low = accuracy_score(y_true[low_conf_mask], y_pred_binary_low)
        print(f"   Accuracy on low confidence:     {acc_low:.1%}")
    
    # Score prediction accuracy
    total_margin_error = np.abs((home_pred - away_pred) - (home_true - away_true))
    close_games = np.sum(np.abs(home_true - away_true) <= 5)
    blowouts = np.sum(np.abs(home_true - away_true) > 15)
    
    print(f"\n📊 Game Type Analysis:")
    print(f"   Close games (≤5 pts):     {close_games} ({close_games/len(y_true):.1%})")
    print(f"   Blowouts (>15 pts):       {blowouts} ({blowouts/len(y_true):.1%})")
    print(f"   Avg margin error:         {np.mean(total_margin_error):.2f} points")
    
    # Over/Under analysis
    total_true = home_true + away_true
    total_pred = home_pred + away_pred
    over_under_error = np.abs(total_true - total_pred)
    
    print(f"\n🎲 Total Points (Over/Under):")
    print(f"   Avg actual total:         {np.mean(total_true):.1f} points")
    print(f"   Avg predicted total:      {np.mean(total_pred):.1f} points")
    print(f"   Avg over/under error:     {np.mean(over_under_error):.2f} points")
    print(f"   RMSE total points:        {np.sqrt(mean_squared_error(total_true, total_pred)):.2f} points")

def test_model(model_path, is_ensemble=False, test_size=500):
    """Test model with comprehensive analytics"""
    
    print_section(f"NBA MODEL TESTING - {'ENSEMBLE' if is_ensemble else 'SINGLE MODEL'}")
    print(f"\nModel: {model_path}")
    print(f"Test samples: {test_size}")
    
    # Load model
    print(f"\nLoading model...")
    if is_ensemble:
        predictor = EnsembleNBAPredictor()
        predictor.load_ensemble(model_path)
        print(f"✓ Loaded ensemble with {len(predictor.models)} models")
        # Use first model to load data
        data_predictor = predictor.models[0]
    else:
        predictor = NBAGamePredictor()
        predictor.load_model(model_path)
        print(f"✓ Model loaded")
        data_predictor = predictor
    
    # Load test data using model's method
    print(f"\nPreparing test data...")
    X, y_outcome, y_points, _ = data_predictor.create_training_data(
        seasons=['2023-24', '2024-25', '2025-26']
    )
    
    # Use most recent games as test set
    if len(X) > test_size:
        X = X[-test_size:]
        y_outcome = y_outcome[-test_size:]
        y_points = y_points[-test_size:]
    
    y_home_score = y_points[:, 0]
    y_away_score = y_points[:, 1]
    
    print(f"✓ Prepared {len(X)} test samples")
    
    # Make predictions
    print(f"\nMaking predictions...")
    if is_ensemble:
        # Predict with each model in ensemble
        all_outcome_probs = []
        all_home_scores = []
        all_away_scores = []
        
        for i, model in enumerate(predictor.models):
            preds = model.model.predict(X, verbose=0)
            # preds is [outcome, points] where points is [home, away]
            all_outcome_probs.append(preds[0][:, 0])
            all_home_scores.append(preds[1][:, 0])  # First column is home
            all_away_scores.append(preds[1][:, 1])  # Second column is away
        
        y_pred_proba = np.mean(all_outcome_probs, axis=0)
        y_pred_home = np.mean(all_home_scores, axis=0)
        y_pred_away = np.mean(all_away_scores, axis=0)
    else:
        predictions = predictor.model.predict(X, verbose=0)
        y_pred_proba = predictions[0][:, 0]
        y_pred_home = predictions[1][:, 0]  # First column is home
        y_pred_away = predictions[1][:, 1]  # Second column is away
    
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    print(f"✓ Predictions complete")
    
    # Calculate all metrics
    metrics = calculate_metrics(
        y_outcome, y_pred_proba, y_pred_binary,
        y_home_score, y_away_score,
        y_pred_home, y_pred_away
    )
    
    # Print results
    print_classification_metrics(metrics)
    print_regression_metrics(metrics)
    print_additional_analytics(y_outcome, y_pred_proba, y_home_score, y_away_score, y_pred_home, y_pred_away)
    
    # Summary
    print_section("SUMMARY")
    print(f"\n✅ Model Performance Overview:")
    print(f"   Accuracy:          {metrics['accuracy']:.1%}")
    print(f"   F1 Score:          {metrics['f1_score']:.4f}")
    print(f"   MAE (points):      {metrics['mae_combined']:.2f}")
    print(f"   RMSE (points):     {metrics['rmse_combined']:.2f}")
    print(f"   Brier Score:       {metrics['brier_score']:.4f}")
    
    if metrics['accuracy'] >= 0.65:
        print(f"\n🎉 Excellent performance!")
    elif metrics['accuracy'] >= 0.60:
        print(f"\n👍 Good performance")
    elif metrics['accuracy'] >= 0.55:
        print(f"\n⚠️  Moderate performance - room for improvement")
    else:
        print(f"\n❌ Poor performance - model needs improvement")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Comprehensive model testing with ML analytics')
    parser.add_argument('--model', type=str, default='nba_model_saved',
                       help='Path to model directory')
    parser.add_argument('--ensemble', action='store_true',
                       help='Test ensemble model')
    parser.add_argument('--test-size', type=int, default=500,
                       help='Number of test samples (default: 500)')
    
    args = parser.parse_args()
    
    try:
        metrics = test_model(args.model, args.ensemble, args.test_size)
        
        print("\n" + "="*80)
        print("Testing complete!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
