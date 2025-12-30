"""Pydantic models for API request/response."""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date


class GamePrediction(BaseModel):
    """Single game prediction."""
    date: str
    game_id: str
    home_team: str
    away_team: str
    predicted_home_prob: float = Field(..., ge=0, le=1)
    predicted_away_prob: float = Field(..., ge=0, le=1)
    predicted_home_score: float
    predicted_away_score: float
    actual_home_score: Optional[int] = None
    actual_away_score: Optional[int] = None
    actual_home_win: Optional[int] = None
    correct: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "date": "2025-12-30",
                "game_id": "0022501234",
                "home_team": "LAL",
                "away_team": "GSW",
                "predicted_home_prob": 0.65,
                "predicted_away_prob": 0.35,
                "predicted_home_score": 112.5,
                "predicted_away_score": 108.2,
                "actual_home_score": 115,
                "actual_away_score": 110,
                "actual_home_win": 1,
                "correct": 1
            }
        }


class PredictionResponse(BaseModel):
    """Response for predictions endpoint."""
    date: str
    games: List[GamePrediction]
    count: int


class HistoryResponse(BaseModel):
    """Paginated history response."""
    predictions: List[GamePrediction]
    total: int
    skip: int
    limit: int


class MonthlyAccuracy(BaseModel):
    """Monthly accuracy statistics."""
    month: str  # YYYY-MM format
    correct: int
    total: int
    accuracy: float = Field(..., ge=0, le=1)
    avg_confidence: float = Field(..., ge=0, le=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "month": "2025-12",
                "correct": 45,
                "total": 60,
                "accuracy": 0.75,
                "avg_confidence": 0.68
            }
        }


class OverallAccuracy(BaseModel):
    """Overall accuracy statistics."""
    total_predictions: int
    total_correct: int
    overall_accuracy: float = Field(..., ge=0, le=1)
    avg_confidence: float = Field(..., ge=0, le=1)
    best_month: Optional[str] = None
    best_month_accuracy: Optional[float] = None
    recent_streak: int = 0
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_predictions": 250,
                "total_correct": 185,
                "overall_accuracy": 0.74,
                "avg_confidence": 0.67,
                "best_month": "2025-11",
                "best_month_accuracy": 0.82,
                "recent_streak": 5
            }
        }
