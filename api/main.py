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
from api.predictions import get_todays_predictions, get_predictions_for_date
from api.analytics import get_monthly_accuracy, get_overall_accuracy
from api.boxscore import get_box_score

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
        return PredictionResponse(
            date=date.today().isoformat(),
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
        
        # Get predictions from last N days
        start_date = (date.today() - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            SELECT COUNT(*) FROM predictions 
            WHERE date >= ?
        """, (start_date,))
        total = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT 
                date, game_id, home_team_abbr, away_team_abbr,
                predicted_home_prob, predicted_away_prob,
                pred_home_points, pred_away_points,
                actual_home_score, actual_away_score, actual_home_win, correct
            FROM predictions
            WHERE date >= ?
            ORDER BY date DESC, game_id
            LIMIT ? OFFSET ?
        """, (start_date, limit, skip))
        
        rows = cursor.fetchall()
        predictions = []
        
        for row in rows:
            predictions.append(GamePrediction(
                date=row[0],
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


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
