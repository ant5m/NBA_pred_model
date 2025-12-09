#!/usr/bin/env python3
"""Analyze prediction performance across all logs.

Usage:
  python3 analyze_performance.py
"""

import glob
import csv
from collections import defaultdict
from datetime import datetime


def analyze_predictions():
    """Analyze all prediction logs and show performance metrics."""
    
    print("="*80)
    print("NBA PREDICTION MODEL - PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Find all prediction log files
    files = sorted(glob.glob('prediction_logs/predictions_*.csv'))
    
    if not files:
        print("\nNo prediction logs found in prediction_logs/")
        return
    
    print(f"\nFound {len(files)} prediction log file(s)\n")
    
    # Track overall stats
    overall_stats = {
        'total_predictions': 0,
        'total_correct': 0,
        'total_with_results': 0,
        'home_predictions': 0,
        'away_predictions': 0,
        'home_actual_wins': 0,
        'predicted_home_prob_sum': 0,
        'by_date': defaultdict(lambda: {'total': 0, 'correct': 0})
    }
    
    # Process each file
    for filepath in files:
        filename = filepath.split('/')[-1]
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Get prediction probability
                try:
                    if 'calibrated_home_prob' in row and row['calibrated_home_prob']:
                        pred_home_prob = float(row['calibrated_home_prob'])
                    elif 'predicted_home_prob' in row and row['predicted_home_prob']:
                        pred_home_prob = float(row['predicted_home_prob'])
                    else:
                        continue
                except:
                    continue
                
                overall_stats['total_predictions'] += 1
                overall_stats['predicted_home_prob_sum'] += pred_home_prob
                
                # Track home/away predictions
                if pred_home_prob >= 0.5:
                    overall_stats['home_predictions'] += 1
                else:
                    overall_stats['away_predictions'] += 1
                
                # Check actual results
                if row.get('actual_home_score'):
                    try:
                        home_score = int(row['actual_home_score'])
                        away_score = int(row['actual_away_score'])
                        actual_home_win = home_score > away_score
                        
                        overall_stats['total_with_results'] += 1
                        if actual_home_win:
                            overall_stats['home_actual_wins'] += 1
                        
                        # Check correctness
                        predicted_home_win = pred_home_prob >= 0.5
                        if predicted_home_win == actual_home_win:
                            overall_stats['total_correct'] += 1
                            overall_stats['by_date'][row['date']]['correct'] += 1
                        
                        overall_stats['by_date'][row['date']]['total'] += 1
                        
                    except:
                        pass
    
    # Display overall statistics
    print("OVERALL PERFORMANCE")
    print("-" * 80)
    print(f"Total Predictions Made:        {overall_stats['total_predictions']}")
    print(f"Games with Results:            {overall_stats['total_with_results']}")
    
    if overall_stats['total_with_results'] > 0:
        accuracy = overall_stats['total_correct'] / overall_stats['total_with_results'] * 100
        print(f"Correct Predictions:           {overall_stats['total_correct']}")
        print(f"Overall Accuracy:              {accuracy:.1f}%")
        print()
        
        # Prediction bias
        avg_home_prob = overall_stats['predicted_home_prob_sum'] / overall_stats['total_predictions']
        actual_home_rate = overall_stats['home_actual_wins'] / overall_stats['total_with_results']
        bias = avg_home_prob - actual_home_rate
        
        print(f"Home Team Predictions:         {overall_stats['home_predictions']} ({overall_stats['home_predictions']/overall_stats['total_predictions']*100:.1f}%)")
        print(f"Away Team Predictions:         {overall_stats['away_predictions']} ({overall_stats['away_predictions']/overall_stats['total_predictions']*100:.1f}%)")
        print(f"Avg Predicted Home Win Prob:   {avg_home_prob:.1%}")
        print(f"Actual Home Win Rate:          {actual_home_rate:.1%}")
        print(f"Prediction Bias (Home):        {bias:+.1%}")
        
        if abs(bias) > 0.05:
            print(f"\n⚠️  Model shows {abs(bias):.1%} bias toward {'home' if bias > 0 else 'away'} teams")
            print(f"   Consider recalibrating: python3.11 calibrate_model.py --ensemble --from-logs")
    
    print("\n" + "="*80)
    print("PERFORMANCE BY DATE")
    print("="*80)
    print(f"{'Date':<15} {'Games':<8} {'Correct':<10} {'Accuracy':<10}")
    print("-" * 80)
    
    for date in sorted(overall_stats['by_date'].keys()):
        stats = overall_stats['by_date'][date]
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            print(f"{date:<15} {stats['total']:<8} {stats['correct']:<10} {acc:.1f}%")
    
    print("\n" + "="*80)
    
    # Recommendations
    if overall_stats['total_with_results'] > 0:
        print("\n💡 RECOMMENDATIONS")
        print("-" * 80)
        
        if accuracy < 50:
            print("❌ Accuracy below 50% - Model needs retraining")
            print("   Run: python3.11 retrain_and_calibrate.py")
        elif accuracy < 55:
            print("⚠️  Accuracy could be improved")
            print("   1. Update database: python3.11 build_team_stats_db.py")
            print("   2. Retrain model: python3.11 retrain_and_calibrate.py")
        elif accuracy < 60:
            print("✅ Good performance - consider recalibrating for better bias correction")
            print("   Run: python3.11 calibrate_model.py --ensemble --from-logs")
        else:
            print("🎯 Excellent performance!")
            print("   Continue monitoring and update weekly")
        
        if abs(bias) > 0.10:
            print(f"\n⚠️  High bias detected ({bias:+.1%})")
            print("   Recalibration strongly recommended")
        
        if overall_stats['total_with_results'] < 20:
            print(f"\n📊 Limited data ({overall_stats['total_with_results']} games)")
            print("   Collect more predictions before retraining/recalibrating")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    analyze_predictions()
