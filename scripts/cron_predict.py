#!/usr/bin/env python3
"""Cron job: Generate and log today's predictions."""

import os
import sys
from datetime import date

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.predictions import get_todays_predictions
from api.database import get_db, USE_POSTGRES

def main():
    """Generate predictions and save to database."""
    print(f"[{date.today()}] Generating predictions...")
    
    try:
        predictions = get_todays_predictions()
        
        if not predictions:
            print("No games scheduled today")
            return
        
        print(f"Found {len(predictions)} games")
        
        # Save to database
        db = get_db()
        cursor = db.cursor()
        
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
            
            print(f"  ✓ {pred.away_team} @ {pred.home_team}: {pred.predicted_home_prob:.1%}")
        
        db.commit()
        cursor.close()
        db.close()
        
        print(f"✅ Saved {len(predictions)} predictions")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
