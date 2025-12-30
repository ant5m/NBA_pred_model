#!/usr/bin/env python3
"""Cron job: Update predictions with actual results."""

import os
import sys
from datetime import date, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
from api.database import get_db, USE_POSTGRES

def main():
    """Update predictions with actual game results."""
    print(f"[{date.today()}] Updating predictions with results...")
    
    try:
        # Get live/finished games
        board = live_scoreboard.ScoreBoard()
        games_data = board.games.get_dict()
        
        if not games_data:
            print("No games found")
            return
        
        db = get_db()
        cursor = db.cursor()
        updated = 0
        
        for game in games_data:
            game_id = game['gameId']
            status = game['gameStatus']  # 1=scheduled, 2=live, 3=final
            
            if status in [2, 3]:  # Live or Final
                home_score = game['homeTeam']['score']
                away_score = game['awayTeam']['score']
                home_win = 1 if home_score > away_score else 0
                
                # Get prediction to check correctness
                if USE_POSTGRES:
                    cursor.execute("""
                        SELECT predicted_home_prob
                        FROM predictions
                        WHERE game_id = %s
                    """, (game_id,))
                else:
                    cursor.execute("""
                        SELECT predicted_home_prob
                        FROM predictions
                        WHERE game_id = ?
                    """, (game_id,))
                
                row = cursor.fetchone()
                if row:
                    pred_home_prob = row[0]
                    predicted_home_win = 1 if pred_home_prob >= 0.5 else 0
                    correct = 1 if predicted_home_win == home_win else 0
                    
                    # Update database
                    if USE_POSTGRES:
                        cursor.execute("""
                            UPDATE predictions
                            SET actual_home_score = %s,
                                actual_away_score = %s,
                                actual_home_win = %s,
                                correct = %s
                            WHERE game_id = %s
                        """, (home_score, away_score, home_win, correct, game_id))
                    else:
                        cursor.execute("""
                            UPDATE predictions
                            SET actual_home_score = ?,
                                actual_away_score = ?,
                                actual_home_win = ?,
                                correct = ?
                            WHERE game_id = ?
                        """, (home_score, away_score, home_win, correct, game_id))
                    
                    updated += 1
                    status_text = "✓" if correct else "✗"
                    print(f"  {status_text} Game {game_id}: {home_score}-{away_score}")
        
        db.commit()
        cursor.close()
        db.close()
        
        print(f"✅ Updated {updated} games")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
