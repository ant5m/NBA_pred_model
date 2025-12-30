"""Analytics and accuracy calculations."""

from typing import List
from api.models import MonthlyAccuracy, OverallAccuracy
from api.database import get_db


def get_monthly_accuracy() -> List[MonthlyAccuracy]:
    """Calculate accuracy by month."""
    db = get_db()
    cursor = db.cursor()
    
    cursor.execute("""
        SELECT 
            strftime('%Y-%m', date) as month,
            COUNT(*) as total,
            SUM(correct) as correct,
            AVG(predicted_home_prob) as avg_conf
        FROM predictions
        WHERE correct IS NOT NULL
        GROUP BY month
        ORDER BY month DESC
    """)
    
    rows = cursor.fetchall()
    results = []
    
    for row in rows:
        month, total, correct, avg_conf = row
        correct = correct or 0
        
        results.append(MonthlyAccuracy(
            month=month,
            total=total,
            correct=correct,
            accuracy=correct / total if total > 0 else 0,
            avg_confidence=avg_conf or 0.5
        ))
    
    return results


def get_overall_accuracy() -> OverallAccuracy:
    """Calculate overall accuracy statistics."""
    db = get_db()
    cursor = db.cursor()
    
    # Overall stats
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(correct) as correct,
            AVG(predicted_home_prob) as avg_conf
        FROM predictions
        WHERE correct IS NOT NULL
    """)
    
    row = cursor.fetchone()
    total, correct, avg_conf = row
    correct = correct or 0
    
    # Best month
    cursor.execute("""
        SELECT 
            strftime('%Y-%m', date) as month,
            COUNT(*) as total,
            SUM(correct) as correct
        FROM predictions
        WHERE correct IS NOT NULL
        GROUP BY month
        HAVING COUNT(*) >= 10
        ORDER BY (CAST(SUM(correct) AS REAL) / COUNT(*)) DESC
        LIMIT 1
    """)
    
    best_month_row = cursor.fetchone()
    best_month = None
    best_month_accuracy = None
    
    if best_month_row:
        best_month, month_total, month_correct = best_month_row
        month_correct = month_correct or 0
        best_month_accuracy = month_correct / month_total if month_total > 0 else 0
    
    # Recent streak
    cursor.execute("""
        SELECT correct
        FROM predictions
        WHERE correct IS NOT NULL
        ORDER BY date DESC, game_id DESC
        LIMIT 20
    """)
    
    recent_results = [row[0] for row in cursor.fetchall()]
    streak = 0
    for result in recent_results:
        if result == 1:
            streak += 1
        else:
            break
    
    return OverallAccuracy(
        total_predictions=total,
        total_correct=correct,
        overall_accuracy=correct / total if total > 0 else 0,
        avg_confidence=avg_conf or 0.5,
        best_month=best_month,
        best_month_accuracy=best_month_accuracy,
        recent_streak=streak
    )
