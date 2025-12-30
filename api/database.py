"""Database connection and initialization for PostgreSQL/SQLite."""

import os
import sqlite3
from contextlib import contextmanager

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///nba_predictions.db")

# Parse database URL
if DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://"):
    USE_POSTGRES = True
    # Railway uses postgres://, but psycopg2 needs postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
else:
    USE_POSTGRES = False
    # Extract SQLite path
    if DATABASE_URL.startswith("sqlite:///"):
        DB_PATH = DATABASE_URL.replace("sqlite:///", "")
    else:
        DB_PATH = "nba_predictions.db"


if USE_POSTGRES:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    
    def get_db():
        """Get PostgreSQL database connection."""
        return psycopg2.connect(DATABASE_URL)
    
    def init_db():
        """Initialize PostgreSQL database schema."""
        conn = get_db()
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                game_id VARCHAR(20) NOT NULL,
                home_team_abbr VARCHAR(3) NOT NULL,
                away_team_abbr VARCHAR(3) NOT NULL,
                home_team_id INTEGER,
                away_team_id INTEGER,
                predicted_home_prob REAL NOT NULL,
                predicted_away_prob REAL NOT NULL,
                pred_home_points REAL NOT NULL,
                pred_away_points REAL NOT NULL,
                actual_home_score INTEGER,
                actual_away_score INTEGER,
                actual_home_win INTEGER,
                correct INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, game_id)
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_date 
            ON predictions(date DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_created 
            ON predictions(created_at DESC)
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
else:
    def get_db():
        """Get SQLite database connection."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_db():
        """Initialize SQLite database schema."""
        conn = get_db()
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                game_id TEXT NOT NULL,
                home_team_abbr TEXT NOT NULL,
                away_team_abbr TEXT NOT NULL,
                home_team_id INTEGER,
                away_team_id INTEGER,
                predicted_home_prob REAL NOT NULL,
                predicted_away_prob REAL NOT NULL,
                pred_home_points REAL NOT NULL,
                pred_away_points REAL NOT NULL,
                actual_home_score INTEGER,
                actual_away_score INTEGER,
                actual_home_win INTEGER,
                correct INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, game_id)
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_date 
            ON predictions(date DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_created 
            ON predictions(created_at DESC)
        """)
        
        conn.commit()
        cursor.close()
        conn.close()


@contextmanager
def get_db_context():
    """Context manager for database connections."""
    conn = get_db()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
