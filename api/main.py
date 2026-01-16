"""FastAPI backend for NBA prediction model.

Endpoints:
- GET /predictions/today - Today's game predictions
- GET /predictions/date/{date} - Predictions for specific date
- GET /predictions/history - All past predictions
- GET /accuracy/monthly - Monthly accuracy stats
- GET /accuracy/overall - Overall performance metrics
- GET /health - Health check
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import date, datetime, timedelta
from typing import List, Optional
import os
import sys
import pytz

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.database import get_db, init_db
from api.models import (
    PredictionResponse, 
    HistoryResponse, 
    MonthlyAccuracy, 
    OverallAccuracy,
    GamePrediction
)
from api.predictions import get_todays_predictions, get_predictions_for_date, save_predictions_to_db
from api.analytics import get_monthly_accuracy, get_overall_accuracy
from api.boxscore import get_box_score

# Import for updating results
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard

app = FastAPI(
    title="NBA Prediction API",
    description="ML-powered NBA game predictions",
    version="1.0.0"
)

# CORS configuration - update origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        os.getenv("FRONTEND_URL", ""),
    ] if os.getenv("FRONTEND_URL") else ["*"],  # Allow all origins if FRONTEND_URL not set
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    init_db()


@app.get("/")
def read_root():
    """Root endpoint."""
    return {
        "name": "NBA Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predictions_today": "/predictions/today",
            "predictions_date": "/predictions/date/{date}",
            "predictions_history": "/predictions/history",
            "accuracy_monthly": "/accuracy/monthly",
            "accuracy_overall": "/accuracy/overall",
            "health": "/health"
        }
    }

@app.get("/boxscore/{game_id}")
def get_game_boxscore(game_id: str):
    """Get box score for a specific game.
    
    Args:
        game_id: NBA game ID (e.g., '0022500123')
    
    Returns:
        Box score data with team stats and player stats
    """
    boxscore = get_box_score(game_id)
    if boxscore is None:
        raise HTTPException(
            status_code=404,
            detail="Box score not available. Game may not have started yet."
        )
    return boxscore

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/predictions/today", response_model=PredictionResponse)
def get_today():
    """Get predictions for today's games."""
    try:
        predictions = get_todays_predictions()
        
        # Auto-save predictions to database
        if predictions:
            save_predictions_to_db(predictions)
        
        # Use Eastern timezone for consistent date handling
        eastern = pytz.timezone('America/New_York')
        today_et = datetime.now(eastern).date()
        
        return PredictionResponse(
            date=today_et.isoformat(),
            games=predictions,
            count=len(predictions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching predictions: {str(e)}")


@app.get("/predictions/date/{date_str}", response_model=PredictionResponse)
def get_by_date(date_str: str):
    """Get predictions for a specific date (YYYY-MM-DD)."""
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        predictions = get_predictions_for_date(target_date)
        return PredictionResponse(
            date=date_str,
            games=predictions,
            count=len(predictions)
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching predictions: {str(e)}")


@app.get("/predictions/history", response_model=HistoryResponse)
def get_history(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to retrieve"),
    skip: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum records to return")
):
    """Get historical predictions with pagination."""
    try:
        db = get_db()
        cursor = db.cursor()
        
        # Get predictions from last N days (use Eastern timezone)
        eastern = pytz.timezone('America/New_York')
        today_et = datetime.now(eastern).date()
        start_date = (today_et - timedelta(days=days)).isoformat()
        
        # Use appropriate placeholders for database type
        from api.database import USE_POSTGRES
        placeholder = "%s" if USE_POSTGRES else "?"
        
        cursor.execute(f"""
            SELECT COUNT(*) FROM predictions 
            WHERE date >= {placeholder}
        """, (start_date,))
        total = cursor.fetchone()[0]
        
        cursor.execute(f"""
            SELECT 
                date, game_id, home_team_abbr, away_team_abbr,
                predicted_home_prob, predicted_away_prob,
                pred_home_points, pred_away_points,
                actual_home_score, actual_away_score, actual_home_win, correct
            FROM predictions
            WHERE date >= {placeholder}
            ORDER BY date DESC, game_id
            LIMIT {placeholder} OFFSET {placeholder}
        """, (start_date, limit, skip))
        
        rows = cursor.fetchall()
        predictions = []
        
        for row in rows:
            predictions.append(GamePrediction(
                date=str(row[0]) if row[0] else "",  # Convert date to string
                game_id=row[1],
                home_team=row[2],
                away_team=row[3],
                predicted_home_prob=row[4],
                predicted_away_prob=row[5],
                predicted_home_score=row[6],
                predicted_away_score=row[7],
                actual_home_score=row[8],
                actual_away_score=row[9],
                actual_home_win=row[10],
                correct=row[11]
            ))
        
        return HistoryResponse(
            predictions=predictions,
            total=total,
            skip=skip,
            limit=limit
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")


@app.get("/accuracy/monthly", response_model=List[MonthlyAccuracy])
def get_monthly():
    """Get accuracy statistics by month."""
    try:
        return get_monthly_accuracy()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating monthly accuracy: {str(e)}")


@app.get("/accuracy/overall", response_model=OverallAccuracy)
def get_overall():
    """Get overall accuracy statistics."""
    try:
        return get_overall_accuracy()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating overall accuracy: {str(e)}")


@app.post("/admin/update-results")
def update_results():
    """Manually trigger update of predictions with actual game results."""
    try:
        from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
        
        board = live_scoreboard.ScoreBoard()
        games_data = board.games.get_dict()
        
        if not games_data:
            return {"status": "success", "message": "No games found", "updated": 0}
        
        db = get_db()
        cursor = db.cursor()
        from api.database import USE_POSTGRES
        placeholder = "%s" if USE_POSTGRES else "?"
        updated = 0
        results = []
        
        for game in games_data:
            game_id = game['gameId']
            status = game['gameStatus']  # 1=scheduled, 2=live, 3=final
            
            if status in [2, 3]:  # Live or Final
                home_score = game['homeTeam']['score']
                away_score = game['awayTeam']['score']
                home_win = 1 if home_score > away_score else 0
                
                # Get prediction to check correctness
                cursor.execute(f"""
                    SELECT predicted_home_prob
                    FROM predictions
                    WHERE game_id = {placeholder}
                """, (game_id,))
                
                row = cursor.fetchone()
                if row:
                    pred_home_prob = row[0]
                    predicted_home_win = 1 if pred_home_prob >= 0.5 else 0
                    
                    # Only mark correct for final games
                    if status == 3:  # Final
                        correct = 1 if predicted_home_win == home_win else 0
                        
                        cursor.execute(f"""
                            UPDATE predictions
                            SET actual_home_score = {placeholder},
                                actual_away_score = {placeholder},
                                actual_home_win = {placeholder},
                                correct = {placeholder}
                            WHERE game_id = {placeholder}
                        """, (home_score, away_score, home_win, correct, game_id))
                        
                        updated += 1
                        results.append({
                            "game_id": game_id,
                            "score": f"{home_score}-{away_score}",
                            "correct": bool(correct),
                            "status": "final"
                        })
                    else:  # Live
                        cursor.execute(f"""
                            UPDATE predictions
                            SET actual_home_score = {placeholder},
                                actual_away_score = {placeholder}
                            WHERE game_id = {placeholder}
                        """, (home_score, away_score, game_id))
                        
                        updated += 1
                        results.append({
                            "game_id": game_id,
                            "score": f"{home_score}-{away_score}",
                            "status": "live"
                        })
        
        db.commit()
        cursor.close()
        
        return {
            "status": "success",
            "message": f"Updated {updated} games",
            "updated": updated,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating results: {str(e)}")


@app.post("/predictions/update-results")
def update_results():
    """Update predictions with actual game results from NBA Live API."""
    try:
        # Get live/finished games
        board = live_scoreboard.ScoreBoard()
        games_data = board.games.get_dict()
        
        if not games_data:
            return {"message": "No games found", "updated": 0}
        
        db = get_db()
        cursor = db.cursor()
        updated = 0
        results = []
        
        from api.database import USE_POSTGRES
        placeholder = "%s" if USE_POSTGRES else "?"
        
        for game in games_data:
            game_id = game['gameId']
            status = game['gameStatus']  # 1=scheduled, 2=live, 3=final
            
            if status in [2, 3]:  # Live or Final
                home_score = game['homeTeam']['score']
                away_score = game['awayTeam']['score']
                home_win = 1 if home_score > away_score else 0
                
                # Get prediction to check correctness
                cursor.execute(f"""
                    SELECT predicted_home_prob, home_team_abbr, away_team_abbr
                    FROM predictions
                    WHERE game_id = {placeholder}
                """, (game_id,))
                
                row = cursor.fetchone()
                if row:
                    pred_home_prob, home_team, away_team = row
                    predicted_home_win = 1 if pred_home_prob >= 0.5 else 0
                    correct = 1 if predicted_home_win == home_win else 0
                    
                    # Update database - only mark correct for final games
                    if status == 3:  # Final only
                        cursor.execute(f"""
                            UPDATE predictions
                            SET actual_home_score = {placeholder},
                                actual_away_score = {placeholder},
                                actual_home_win = {placeholder},
                                correct = {placeholder}
                            WHERE game_id = {placeholder}
                        """, (home_score, away_score, home_win, correct, game_id))
                    else:  # Live - update scores but not correctness yet
                        cursor.execute(f"""
                            UPDATE predictions
                            SET actual_home_score = {placeholder},
                                actual_away_score = {placeholder}
                            WHERE game_id = {placeholder}
                        """, (home_score, away_score, game_id))
                    
                    updated += 1
                    results.append({
                        "game_id": game_id,
                        "matchup": f"{away_team} @ {home_team}",
                        "score": f"{home_score}-{away_score}",
                        "status": "Final" if status == 3 else "Live",
                        "correct": correct if status == 3 else None
                    })
        
        db.commit()
        cursor.close()
        
        return {
            "message": f"Updated {updated} games",
            "updated": updated,
            "games": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating results: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
