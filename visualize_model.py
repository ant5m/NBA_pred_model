#!/usr/bin/env python3
"""
Visualize Model Performance with Comprehensive Graphs
Generates plots for all key metrics: ROC Curve, Confusion Matrix, Calibration, etc.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, 
    precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve
from nba_model import NBAGamePredictor, EnsembleNBAPredictor
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def create_output_dir():
    """Create output directory for visualizations"""
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plot confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Away Win', 'Home Win'],
                yticklabels=['Away Win', 'Home Win'])
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.7, f'({pct:.1f}%)', 
                    ha='center', va='top', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: confusion_matrix.png")
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, output_dir):
    """Plot ROC curve with AUC"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', 
              fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: roc_curve.png")
    plt.close()

def plot_precision_recall_curve(y_true, y_pred_proba, output_dir):
    """Plot Precision-Recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: precision_recall_curve.png")
    plt.close()

def plot_calibration_curve(y_true, y_pred_proba, output_dir):
    """Plot calibration curve"""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=10, strategy='uniform'
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
             label='Model', linewidth=2, markersize=8)
    plt.plot([0, 1], [0, 1], "k--", label='Perfectly calibrated', linewidth=2)
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Calibration Curve (Reliability Diagram)', 
              fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_curve.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: calibration_curve.png")
    plt.close()

def plot_probability_distribution(y_true, y_pred_proba, output_dir):
    """Plot probability distribution for wins and losses"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Home wins
    home_wins = y_pred_proba[y_true == 1]
    ax1.hist(home_wins, bins=20, color='green', alpha=0.7, edgecolor='black')
    ax1.axvline(home_wins.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean = {home_wins.mean():.3f}')
    ax1.set_xlabel('Predicted Probability', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Predicted Probabilities for Home Wins', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Away wins
    away_wins = y_pred_proba[y_true == 0]
    ax2.hist(away_wins, bins=20, color='red', alpha=0.7, edgecolor='black')
    ax2.axvline(away_wins.mean(), color='blue', linestyle='--', linewidth=2,
                label=f'Mean = {away_wins.mean():.3f}')
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Predicted Probabilities for Away Wins', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'probability_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: probability_distribution.png")
    plt.close()

def plot_prediction_errors(home_true, away_true, home_pred, away_pred, output_dir):
    """Plot prediction errors for points"""
    home_errors = home_pred - home_true
    away_errors = away_pred - away_true
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Home team errors histogram
    axes[0, 0].hist(home_errors, bins=30, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Prediction Error (Points)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Home Team Point Prediction Errors', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Away team errors histogram
    axes[0, 1].hist(away_errors, bins=30, color='orange', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Prediction Error (Points)', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Away Team Point Prediction Errors', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Actual vs Predicted - Home
    axes[1, 0].scatter(home_true, home_pred, alpha=0.5, s=30)
    axes[1, 0].plot([80, 140], [80, 140], 'r--', linewidth=2, label='Perfect prediction')
    axes[1, 0].set_xlabel('Actual Points', fontsize=11)
    axes[1, 0].set_ylabel('Predicted Points', fontsize=11)
    axes[1, 0].set_title('Home Team: Actual vs Predicted', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Actual vs Predicted - Away
    axes[1, 1].scatter(away_true, away_pred, alpha=0.5, s=30, color='orange')
    axes[1, 1].plot([80, 140], [80, 140], 'r--', linewidth=2, label='Perfect prediction')
    axes[1, 1].set_xlabel('Actual Points', fontsize=11)
    axes[1, 1].set_ylabel('Predicted Points', fontsize=11)
    axes[1, 1].set_title('Away Team: Actual vs Predicted', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_errors.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: prediction_errors.png")
    plt.close()

def plot_metrics_summary(metrics, output_dir):
    """Plot bar chart of key metrics"""
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1'],
        metrics['auc']
    ]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.8, edgecolor='black')
    plt.ylim([0, 1])
    plt.ylabel('Score', fontsize=12)
    plt.title('Classification Metrics Summary', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_summary.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: metrics_summary.png")
    plt.close()

def plot_regression_metrics(home_mae, away_mae, home_rmse, away_rmse, 
                            home_r2, away_r2, output_dir):
    """Plot regression metrics comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # MAE and RMSE
    metrics = ['MAE', 'RMSE']
    home_vals = [home_mae, home_rmse]
    away_vals = [away_mae, away_rmse]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, home_vals, width, label='Home Team', 
                    color='#3498db', edgecolor='black')
    bars2 = ax1.bar(x + width/2, away_vals, width, label='Away Team', 
                    color='#e74c3c', edgecolor='black')
    
    ax1.set_ylabel('Points', fontsize=12)
    ax1.set_title('Point Prediction Errors', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # R² scores
    teams = ['Home Team', 'Away Team']
    r2_vals = [home_r2, away_r2]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax2.bar(teams, r2_vals, color=colors, edgecolor='black')
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('Coefficient of Determination (R²)', fontsize=14, fontweight='bold')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars, r2_vals):
        height = bar.get_height()
        y_pos = height + 0.05 if height > 0 else height - 0.05
        va = 'bottom' if height > 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{value:.3f}', ha='center', va=va, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'regression_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: regression_metrics.png")
    plt.close()

def generate_all_visualizations(predictor, test_size=500):
    """Generate all visualizations"""
    print("="*80)
    print("  NBA MODEL VISUALIZATION")
    print("="*80)
    print(f"\nGenerating visualizations with {test_size} test samples...\n")
    
    # Create output directory
    output_dir = create_output_dir()
    
    # Prepare test data
    print("Preparing test data...")
    X, y_outcome, y_points, _ = predictor.models[0].create_training_data(
        seasons=['2023-24', '2024-25', '2025-26']
    ) if hasattr(predictor, 'models') else predictor.create_training_data(
        seasons=['2023-24', '2024-25', '2025-26']
    )
    
    # Use last test_size samples
    X = X[-test_size:]
    y_outcome = y_outcome[-test_size:]
    y_points = y_points[-test_size:]
    
    print(f"✓ Prepared {len(X)} test samples")
    
    # Make predictions
    print("\nMaking predictions...")
    if hasattr(predictor, 'models'):
        # Ensemble - use each model and average
        all_outcome_probs = []
        all_points_preds = []
        
        for model in predictor.models:
            outcome_pred, points_pred = model.model.predict(
                model.scaler.transform(X), 
                verbose=0
            )
            all_outcome_probs.append(outcome_pred)
            all_points_preds.append(points_pred)
        
        # Average predictions
        y_pred_proba = np.mean(all_outcome_probs, axis=0).flatten()
        points_preds = np.mean(all_points_preds, axis=0)
    else:
        # Single model
        X_scaled = predictor.scaler.transform(X)
        predictions = predictor.model.predict(X_scaled, verbose=0)
        y_pred_proba = predictions[0].flatten()
        points_preds = predictions[1]
    
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    
    home_pred = points_preds[:, 0]
    away_pred = points_preds[:, 1]
    home_true = y_points[:, 0]
    away_true = y_points[:, 1]
    
    print("✓ Predictions complete")
    
    # Calculate metrics for summary
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    metrics = {
        'accuracy': accuracy_score(y_outcome, y_pred_binary),
        'precision': precision_score(y_outcome, y_pred_binary),
        'recall': recall_score(y_outcome, y_pred_binary),
        'f1': f1_score(y_outcome, y_pred_binary),
        'auc': roc_auc_score(y_outcome, y_pred_proba)
    }
    
    home_mae = mean_absolute_error(home_true, home_pred)
    away_mae = mean_absolute_error(away_true, away_pred)
    home_rmse = np.sqrt(mean_squared_error(home_true, home_pred))
    away_rmse = np.sqrt(mean_squared_error(away_true, away_pred))
    home_r2 = r2_score(home_true, home_pred)
    away_r2 = r2_score(away_true, away_pred)
    
    # Generate all plots
    print("\nGenerating visualizations:")
    print("-" * 40)
    
    plot_confusion_matrix(y_outcome, y_pred_binary, output_dir)
    plot_roc_curve(y_outcome, y_pred_proba, output_dir)
    plot_precision_recall_curve(y_outcome, y_pred_proba, output_dir)
    plot_calibration_curve(y_outcome, y_pred_proba, output_dir)
    plot_probability_distribution(y_outcome, y_pred_proba, output_dir)
    plot_prediction_errors(home_true, away_true, home_pred, away_pred, output_dir)
    plot_metrics_summary(metrics, output_dir)
    plot_regression_metrics(home_mae, away_mae, home_rmse, away_rmse, 
                           home_r2, away_r2, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("  VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\n📊 All visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  1. confusion_matrix.png       - Confusion matrix heatmap")
    print("  2. roc_curve.png              - ROC curve with AUC")
    print("  3. precision_recall_curve.png - Precision-Recall curve")
    print("  4. calibration_curve.png      - Calibration/reliability curve")
    print("  5. probability_distribution.png - Win probability distributions")
    print("  6. prediction_errors.png      - Point prediction error analysis")
    print("  7. metrics_summary.png        - Classification metrics bar chart")
    print("  8. regression_metrics.png     - Regression metrics comparison")
    
    print("\n📈 Key Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.1%}")
    print(f"  Precision: {metrics['precision']:.1%}")
    print(f"  Recall:    {metrics['recall']:.1%}")
    print(f"  F1 Score:  {metrics['f1']:.3f}")
    print(f"  AUC-ROC:   {metrics['auc']:.3f}")
    print(f"\n  Home MAE:  {home_mae:.2f} points")
    print(f"  Away MAE:  {away_mae:.2f} points")
    print(f"  Home R²:   {home_r2:.3f}")
    print(f"  Away R²:   {away_r2:.3f}")
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive visualizations for NBA prediction model'
    )
    parser.add_argument('--ensemble', action='store_true',
                       help='Use ensemble model')
    parser.add_argument('--model', type=str, default='ensemble_model_saved',
                       help='Path to model directory')
    parser.add_argument('--test-size', type=int, default=500,
                       help='Number of test samples')
    
    args = parser.parse_args()
    
    # Load model
    if args.ensemble:
        predictor = EnsembleNBAPredictor()
        predictor.load_ensemble(args.model)
        print(f"\n✓ Loaded ensemble with {len(predictor.models)} models\n")
    else:
        predictor = NBAGamePredictor()
        predictor.load_model(args.model)
        print(f"\n✓ Loaded single model\n")
    
    # Generate visualizations
    generate_all_visualizations(predictor, args.test_size)

if __name__ == '__main__':
    main()
