#!/usr/bin/env python3
"""Migrate prediction logs from CSV files to PostgreSQL/SQLite database."""

import os
import sys
import csv
from datetime import datetime
from api.database import get_db, init_db

def migrate_csv_to_db(csv_dir='prediction_logs'):
    """Migrate all CSV files to database."""
    print("Initializing database...")
    init_db()
    
    db = get_db()
    cursor = db.cursor()
    
    if not os.path.exists(csv_dir):
        print(f"Directory {csv_dir} not found")
        return
    
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to migrate\n")
    
    total_migrated = 0
    
    for csv_file in sorted(csv_files):
        csv_path = os.path.join(csv_dir, csv_file)
        print(f"Processing {csv_file}...")
        
        try:
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if not rows:
                    print(f"  No data in {csv_file}")
                    continue
                
                # Skip backfill files without game_id
                if 'game_id' not in reader.fieldnames:
                    print(f"  Skipping backfill file (no game_id)")
                    continue
                
                migrated = 0
                
                for row in rows:
                    # Handle both column name formats
                    date_val = row.get('date')
                    game_id = row.get('game_id')
                    home_team = row.get('home_team_abbr') or row.get('home_team')
                    away_team = row.get('away_team_abbr') or row.get('away_team')
                    
                    # Prediction columns
                    pred_home_prob = row.get('predicted_home_prob') or row.get('calibrated_home_prob') or row.get('raw_home_prob')
                    pred_away_prob = row.get('predicted_away_prob') or row.get('calibrated_away_prob') or row.get('raw_away_prob')
                    pred_home_pts = row.get('pred_home_points') or row.get('predicted_home_points')
                    pred_away_pts = row.get('pred_away_points') or row.get('predicted_away_points')
                    
                    # Actual columns
                    actual_home_score = row.get('actual_home_score')
                    actual_away_score = row.get('actual_away_score')
                    actual_home_win = row.get('actual_home_win')
                    correct = row.get('correct')
                    
                    # Convert empty strings to None
                    actual_home_score = int(actual_home_score) if actual_home_score and actual_home_score != '' else None
                    actual_away_score = int(actual_away_score) if actual_away_score and actual_away_score != '' else None
                    actual_home_win = int(actual_home_win) if actual_home_win and actual_home_win != '' else None
                    correct = int(correct) if correct and correct != '' else None
                    
                    try:
                        cursor.execute("""
                            INSERT OR REPLACE INTO predictions (
                                date, game_id, home_team_abbr, away_team_abbr,
                                predicted_home_prob, predicted_away_prob,
                                pred_home_points, pred_away_points,
                                actual_home_score, actual_away_score, actual_home_win, correct
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            date_val, game_id, home_team, away_team,
                            float(pred_home_prob), float(pred_away_prob),
                            float(pred_home_pts), float(pred_away_pts),
                            actual_home_score, actual_away_score, actual_home_win, correct
                        ))
                        migrated += 1
                    except Exception as e:
                        print(f"    Error inserting row: {e}")
                        continue
                
                print(f"  ✓ Migrated {migrated} predictions")
                total_migrated += migrated
                
        except Exception as e:
            print(f"  ✗ Error processing {csv_file}: {e}")
            continue
    
    db.commit()
    cursor.close()
    db.close()
    
    print(f"\n{'='*60}")
    print(f"✅ Migration complete!")
    print(f"Total predictions migrated: {total_migrated}")
    print(f"{'='*60}")


if __name__ == "__main__":
    migrate_csv_to_db()
