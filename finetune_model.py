#!/usr/bin/env python3
"""
Fine-tune the NBA prediction model on recent games from the current season.

This allows the model to adapt to current trends without full retraining.
Use this regularly (e.g., weekly) to keep the model updated.

Usage:
    python3 finetune_model.py --recent-games 50 --epochs 10
"""

import argparse
from nba_model import NBAGamePredictor
from sklearn.model_selection import train_test_split
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Fine-tune NBA prediction model on recent games')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to saved model to fine-tune')
    parser.add_argument('--recent-games', type=int, default=50,
                       help='Number of most recent games to use for fine-tuning')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Learning rate for fine-tuning (lower than initial training)')
    parser.add_argument('--seasons', type=str, default='2024-25',
                       help='Season(s) to fine-tune on (comma-separated for multiple)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: overwrites input model)')
    parser.add_argument('--ensemble', action='store_true',
                       help='Fine-tune an ensemble model instead of single model')
    
    args = parser.parse_args()
    
    # Set default model path based on ensemble flag
    if args.model is None:
        args.model = 'ensemble_model_saved' if args.ensemble else 'nba_model_saved'
    
    # Parse seasons
    seasons = [s.strip() for s in args.seasons.split(',')]
    
    print("="*80)
    print("NBA GAME PREDICTION MODEL - FINE-TUNING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Base Model: {args.model}")
    print(f"  Recent Games: {args.recent_games}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Season(s): {', '.join(seasons)}")
    print(f"  Mode: {'Ensemble' if args.ensemble else 'Single Model'}")
    
    # Load existing model
    print("\n" + "="*80)
    print("STEP 1: Loading Base Model")
    print("="*80)
    
    try:
        if args.ensemble:
            from nba_model import EnsembleNBAPredictor
            predictor = EnsembleNBAPredictor()
            predictor.load_ensemble(args.model)
            print(f"✓ Ensemble loaded successfully")
            print(f"  Models in ensemble: {len(predictor.models)}")
            # Use first model to get data
            data_predictor = predictor.models[0]
        else:
            predictor = NBAGamePredictor()
            predictor.load_model(args.model)
            print(f"✓ Model loaded successfully")
            print(f"  Features: {len(predictor.feature_columns)}")
            data_predictor = predictor
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("Please ensure you have a trained model.")
        print("Run: python3 nba_model.py  (or with --ensemble)")
        return
    
    # Create training data from recent games
    print("\n" + "="*80)
    print("STEP 2: Loading Recent Games")
    print("="*80)
    
    X, y_outcome, y_points, _ = data_predictor.create_training_data(seasons=seasons)
    
    if len(X) == 0:
        print(f"\n❌ No training data found for season(s): {', '.join(seasons)}")
        return
    
    # Use only the most recent games
    if len(X) > args.recent_games:
        X = X[-args.recent_games:]
        y_outcome = y_outcome[-args.recent_games:]
        y_points = y_points[-args.recent_games:]
    
    print(f"\n✓ Loaded {len(X)} recent games for fine-tuning")
    
    # Fine-tune
    print("\n" + "="*80)
    print("STEP 3: Fine-Tuning Model")
    print("="*80)
    
    if args.ensemble:
        # Fine-tune each model in the ensemble
        for i, model in enumerate(predictor.models):
            print(f"\nFine-tuning model {i+1}/{len(predictor.models)}...")
            model.fine_tune(
                X, y_outcome, y_points,
                epochs=args.epochs,
                learning_rate=args.learning_rate
            )
    else:
        history = predictor.fine_tune(
            X, y_outcome, y_points,
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )
    
    # Evaluate on the fine-tuning data
    print("\n" + "="*80)
    print("STEP 4: Evaluating Fine-Tuned Model")
    print("="*80)
    
    metrics = predictor.evaluate(X, y_outcome, y_points)
    
    if args.ensemble:
        print("\n📊 Fine-Tuned Ensemble Performance:")
        print(f"  Outcome Accuracy: {metrics['ensemble_outcome_accuracy']:.2%}")
        print(f"  Points MAE (Home): {metrics['ensemble_points_mae_home']:.2f} points")
        print(f"  Points MAE (Away): {metrics['ensemble_points_mae_away']:.2f} points")
        print(f"  Points MAE (Average): {metrics['ensemble_points_mae_avg']:.2f} points")
        
        print(f"\n  Individual Models:")
        for i, m in enumerate(metrics['individual_model_metrics']):
            print(f"    Model {i+1}: {m['outcome_accuracy']:.2%} accuracy, {m['points_mae_avg']:.2f} MAE")
    else:
        print("\n📊 Fine-Tuning Set Performance:")
        print(f"  Outcome Accuracy: {metrics['outcome_accuracy']:.2%}")
        print(f"  Points MAE (Home): {metrics['points_mae_home']:.2f} points")
        print(f"  Points MAE (Away): {metrics['points_mae_away']:.2f} points")
        print(f"  Points MAE (Average): {metrics['points_mae_avg']:.2f} points")
    
    # Save fine-tuned model
    output_path = args.output if args.output else args.model
    
    print("\n" + "="*80)
    print("STEP 5: Saving Fine-Tuned Model")
    print("="*80)
    
    if args.ensemble:
        predictor.save_ensemble(output_path)
    else:
        predictor.save_model(output_path)
    
    print("\n" + "="*80)
    print("✅ FINE-TUNING COMPLETE!")
    print("="*80)
    
    if args.output:
        print(f"\nFine-tuned model saved to: {args.output}")
        print(f"Original model unchanged: {args.model}")
    else:
        print(f"\nModel updated: {args.model}")
    
    print("\n💡 Recommendations:")
    print("  - Fine-tune weekly/monthly to keep model current")
    print("  - Use --recent-games to control how many games to learn from")
    print("  - Keep learning rate low (0.0001) to avoid catastrophic forgetting")
    print("  - Use --seasons to specify which season(s) to fine-tune on")
    print("\n📋 Next steps:")
    print(f"  - Make predictions: python3 predict_game.py --model {output_path}")
    
    print("\n📖 Examples:")
    print("  # Fine-tune single model on latest 100 games:")
    print("  python3 finetune_model.py --recent-games 100")
    print("\n  # Fine-tune ensemble on specific season:")
    print("  python3 finetune_model.py --ensemble --model ensemble_model_saved --seasons 2024-25")
    print("\n  # Fine-tune on multiple seasons with custom output:")
    print("  python3 finetune_model.py --seasons 2023-24,2024-25 --output nba_model_finetuned")


if __name__ == '__main__':
    main()
