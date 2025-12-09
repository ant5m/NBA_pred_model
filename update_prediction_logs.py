#!/usr/bin/env python3
"""Update prediction logs CSV files with actual game results from NBA Live API.

Reads existing CSV logs, fetches live/final scores from NBA Live API,
and updates the actual_home_score, actual_away_score, actual_home_win, and correct columns.

Usage:
  python3 update_prediction_logs.py [--logs-dir prediction_logs]
"""

import argparse
import os
import csv
from datetime import datetime

from nba_api.live.nba.endpoints import scoreboard as live_scoreboard


def get_game_results_from_live_api():
    """Fetch game results from NBA Live API (gets current/finished games).
    
    Returns:
        Dict mapping game_id to {'home_score': int, 'away_score': int, 'home_win': bool,
                                  'home_team_id': int, 'away_team_id': int, 'status': str}
    """
    try:
        print(f"  Fetching from NBA Live API...")
        board = live_scoreboard.ScoreBoard()
        games_data = board.games.get_dict()
        
        if not games_data:
            print(f"  No games found")
            return {}
        
        results = {}
        
        for game in games_data:
            game_id = game['gameId']
            home_team = game['homeTeam']
            away_team = game['awayTeam']
            game_status = game['gameStatus']  # 1=scheduled, 2=live, 3=final
            
            # Get scores for games that have started (live or finished)
            if game_status in [2, 3]:
                home_score = home_team['score']
                away_score = away_team['score']
                
                results[game_id] = {
                    'home_score': home_score,
                    'away_score': away_score,
                    'home_win': home_score > away_score if game_status == 3 else None,  # Only mark winner if final
                    'home_team_id': home_team['teamId'],
                    'away_team_id': away_team['teamId'],
                    'status': 'Final' if game_status == 3 else 'Live'
                }
        
        print(f"  Found {len(results)} games with scores (live or finished)")
        return results
        
    except Exception as e:
        print(f"  Error fetching results from Live API: {e}")
        import traceback
        traceback.print_exc()
        return {}


def update_csv_file(csv_path):
    """Update a CSV file with actual game results."""
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    
    # Read existing CSV
    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    if not rows:
        print(f"No data in {csv_path}")
        return
    
    # Fetch live/finished game results
    print(f"Fetching live game scores...")
    all_results = get_game_results_from_live_api()
    
    # Update rows with actual results
    updated_count = 0
    for row in rows:
        game_id = row['game_id']
        if game_id in all_results:
            result = all_results[game_id]
            
            # Verify team IDs match (optional sanity check)
            csv_home_id = int(row['home_team_id']) if row['home_team_id'] else None
            csv_away_id = int(row['away_team_id']) if row['away_team_id'] else None
            
            if csv_home_id and csv_home_id != result['home_team_id']:
                print(f"  Warning: Home team ID mismatch for {game_id} (CSV: {csv_home_id}, API: {result['home_team_id']})")
            if csv_away_id and csv_away_id != result['away_team_id']:
                print(f"  Warning: Away team ID mismatch for {game_id} (CSV: {csv_away_id}, API: {result['away_team_id']})")
            
            # Update actual scores
            row['actual_home_score'] = result['home_score']
            row['actual_away_score'] = result['away_score']
            
            # Only mark winner and correctness if game is final
            if result['home_win'] is not None:
                row['actual_home_win'] = 1 if result['home_win'] else 0
                
                # Determine if prediction was correct
                # Check for calibrated predictions first, fall back to raw predictions
                if 'calibrated_home_prob' in row and row['calibrated_home_prob']:
                    pred_home_prob = float(row['calibrated_home_prob'])
                elif 'predicted_home_prob' in row and row['predicted_home_prob']:
                    pred_home_prob = float(row['predicted_home_prob'])
                else:
                    pred_home_prob = 0.5
                
                predicted_home_win = pred_home_prob >= 0.5
                actual_home_win = result['home_win']
                
                row['correct'] = 1 if predicted_home_win == actual_home_win else 0
                status_msg = 'CORRECT' if row['correct'] == 1 else 'WRONG'
            else:
                # Game is live, don't mark winner yet
                status_msg = 'LIVE'
            
            print(f"  ✓ {row['home_team_abbr']} vs {row['away_team_abbr']}: {result['home_score']}-{result['away_score']} [{result['status']}] ({status_msg})")
            updated_count += 1
    
    # Write updated CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Updated {updated_count} games in {csv_path}")
    
    # Calculate and display accuracy
    correct = sum(1 for row in rows if row['correct'] == '1')
    total = sum(1 for row in rows if row['correct'] in ['0', '1'])
    if total > 0:
        print(f"  Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Update prediction logs with actual game results')
    parser.add_argument('--logs-dir', type=str, default='prediction_logs', 
                       help='Directory containing prediction log CSV files')
    args = parser.parse_args()
    
    if not os.path.exists(args.logs_dir):
        print(f"Logs directory not found: {args.logs_dir}")
        return
    
    # Find all CSV files in logs directory
    csv_files = [f for f in os.listdir(args.logs_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {args.logs_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) to update")
    print("="*80)
    
    for csv_file in sorted(csv_files):
        csv_path = os.path.join(args.logs_dir, csv_file)
        print(f"\nProcessing: {csv_file}")
        update_csv_file(csv_path)
    
    print("\n" + "="*80)
    print("Update complete!")


if __name__ == '__main__':
    main()
