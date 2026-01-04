#!/usr/bin/env python3
"""
Build and update comprehensive team stats database.
Covers seasons 2022-23, 2023-24, 2024-25, 2025-26.

Usage:
    python3 build_team_stats_db.py --initial  # First time build
    python3 build_team_stats_db.py --update   # Update with latest data
    python3 build_team_stats_db.py --stats    # Show database stats
"""

import os
import sqlite3
import pandas as pd
import time
import argparse
from nba_api.stats.endpoints import leaguedashteamstats, teamgamelog
from nba_api.stats.static import teams

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///nba_team_stats.db")

# Parse database URL
if DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://"):
    USE_POSTGRES = True
    import psycopg2
    from psycopg2.extras import RealDictCursor
    # Railway uses postgres://, but psycopg2 needs postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
else:
    USE_POSTGRES = False
    # Extract SQLite path
    if DATABASE_URL.startswith("sqlite:///"):
        DB_NAME = DATABASE_URL.replace("sqlite:///", "")
    else:
        DB_NAME = "nba_team_stats.db"

# Configuration
SEASONS = ['2022-23', '2023-24', '2024-25', '2025-26']
CURRENT_SEASON = '2025-26'

def get_connection():
    """Get database connection."""
    if USE_POSTGRES:
        return psycopg2.connect(DATABASE_URL)
    else:
        return sqlite3.connect(DB_NAME)

def create_tables():
    """Create database tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Teams table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY,
            full_name TEXT NOT NULL,
            abbreviation TEXT NOT NULL,
            nickname TEXT,
            city TEXT,
            state TEXT,
            year_founded INTEGER
        )
    ''')
    
    # Season stats table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS team_season_stats (
            team_id INTEGER,
            season TEXT,
            games_played INTEGER,
            wins INTEGER,
            losses INTEGER,
            win_pct REAL,
            minutes_played REAL,
            points REAL,
            field_goals_made REAL,
            field_goals_attempted REAL,
            field_goal_pct REAL,
            three_pointers_made REAL,
            three_pointers_attempted REAL,
            three_point_pct REAL,
            free_throws_made REAL,
            free_throws_attempted REAL,
            free_throw_pct REAL,
            offensive_rebounds REAL,
            defensive_rebounds REAL,
            total_rebounds REAL,
            assists REAL,
            turnovers REAL,
            steals REAL,
            blocks REAL,
            personal_fouls REAL,
            plus_minus REAL,
            PRIMARY KEY (team_id, season),
            FOREIGN KEY (team_id) REFERENCES teams(team_id)
        )
    ''')
    
    # Game logs table (for current season)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS team_game_logs (
            game_id TEXT,
            team_id INTEGER,
            season TEXT,
            game_date TEXT,
            matchup TEXT,
            win_loss TEXT,
            minutes INTEGER,
            points INTEGER,
            field_goals_made INTEGER,
            field_goals_attempted INTEGER,
            field_goal_pct REAL,
            three_pointers_made INTEGER,
            three_pointers_attempted INTEGER,
            three_point_pct REAL,
            free_throws_made INTEGER,
            free_throws_attempted INTEGER,
            free_throw_pct REAL,
            offensive_rebounds INTEGER,
            defensive_rebounds INTEGER,
            total_rebounds INTEGER,
            assists INTEGER,
            turnovers INTEGER,
            steals INTEGER,
            blocks INTEGER,
            personal_fouls INTEGER,
            plus_minus INTEGER,
            PRIMARY KEY (game_id, team_id),
            FOREIGN KEY (team_id) REFERENCES teams(team_id)
        )
    ''')
    
    conn.commit()
    conn.close()
    db_type = "PostgreSQL" if USE_POSTGRES else f"{DB_NAME}"
    print(f"✓ Database tables created/verified in {db_type}")

def get_all_teams():
    """Get all NBA teams."""
    all_teams = teams.get_teams()
    print(f"✓ Found {len(all_teams)} NBA teams")
    return all_teams

def update_team_info(all_teams):
    """Insert or update team information."""
    conn = get_connection()
    cursor = conn.cursor()
    
    for team in all_teams:
        if USE_POSTGRES:
            cursor.execute('''
                INSERT INTO teams 
                (team_id, full_name, abbreviation, nickname, city, state, year_founded)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (team_id) DO UPDATE SET
                    full_name = EXCLUDED.full_name,
                    abbreviation = EXCLUDED.abbreviation,
                    nickname = EXCLUDED.nickname,
                    city = EXCLUDED.city,
                    state = EXCLUDED.state,
                    year_founded = EXCLUDED.year_founded
            ''', (
                team['id'],
                team['full_name'],
                team['abbreviation'],
                team['nickname'],
                team['city'],
                team['state'],
                team['year_founded']
            ))
        else:
            cursor.execute('''
                INSERT OR REPLACE INTO teams 
                (team_id, full_name, abbreviation, nickname, city, state, year_founded)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                team['id'],
                team['full_name'],
                team['abbreviation'],
                team['nickname'],
                team['city'],
                team['state'],
                team['year_founded']
            ))
    
    conn.commit()
    conn.close()
    print(f"✓ Updated {len(all_teams)} team records")

def fetch_season_stats(season):
    """Fetch team stats for a specific season."""
    print(f"Fetching stats for {season}...")
    
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            per_mode_detailed='PerGame'
        )
        df = stats.get_data_frames()[0]
        time.sleep(0.6)  # Rate limiting
        return df
    except Exception as e:
        print(f"  ⚠ Error fetching {season}: {e}")
        return None

def update_season_stats(seasons=None):
    """Update season stats for specified seasons."""
    if seasons is None:
        seasons = SEASONS
    
    conn = get_connection()
    cursor = conn.cursor()
    
    total_records = 0
    for season in seasons:
        df = fetch_season_stats(season)
        
        if df is not None and not df.empty:
            # Select and rename relevant columns
            stats_df = pd.DataFrame({
                'team_id': df['TEAM_ID'],
                'season': season,
                'games_played': df['GP'],
                'wins': df['W'],
                'losses': df['L'],
                'win_pct': df['W_PCT'],
                'minutes_played': df['MIN'],
                'points': df['PTS'],
                'field_goals_made': df['FGM'],
                'field_goals_attempted': df['FGA'],
                'field_goal_pct': df['FG_PCT'],
                'three_pointers_made': df['FG3M'],
                'three_pointers_attempted': df['FG3A'],
                'three_point_pct': df['FG3_PCT'],
                'free_throws_made': df['FTM'],
                'free_throws_attempted': df['FTA'],
                'free_throw_pct': df['FT_PCT'],
                'offensive_rebounds': df['OREB'],
                'defensive_rebounds': df['DREB'],
                'total_rebounds': df['REB'],
                'assists': df['AST'],
                'turnovers': df['TOV'],
                'steals': df['STL'],
                'blocks': df['BLK'],
                'personal_fouls': df['PF'],
                'plus_minus': df['PLUS_MINUS']
            })
            
            # Insert or update season stats
            for _, row in stats_df.iterrows():
                if USE_POSTGRES:
                    cursor.execute('''
                        INSERT INTO team_season_stats
                        (team_id, season, games_played, wins, losses, win_pct,
                         minutes_played, points, field_goals_made, field_goals_attempted,
                         field_goal_pct, three_pointers_made, three_pointers_attempted,
                         three_point_pct, free_throws_made, free_throws_attempted,
                         free_throw_pct, offensive_rebounds, defensive_rebounds,
                         total_rebounds, assists, turnovers, steals, blocks,
                         personal_fouls, plus_minus)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (team_id, season) DO UPDATE SET
                            games_played = EXCLUDED.games_played,
                            wins = EXCLUDED.wins,
                            losses = EXCLUDED.losses,
                            win_pct = EXCLUDED.win_pct,
                            minutes_played = EXCLUDED.minutes_played,
                            points = EXCLUDED.points,
                            field_goals_made = EXCLUDED.field_goals_made,
                            field_goals_attempted = EXCLUDED.field_goals_attempted,
                            field_goal_pct = EXCLUDED.field_goal_pct,
                            three_pointers_made = EXCLUDED.three_pointers_made,
                            three_pointers_attempted = EXCLUDED.three_pointers_attempted,
                            three_point_pct = EXCLUDED.three_point_pct,
                            free_throws_made = EXCLUDED.free_throws_made,
                            free_throws_attempted = EXCLUDED.free_throws_attempted,
                            free_throw_pct = EXCLUDED.free_throw_pct,
                            offensive_rebounds = EXCLUDED.offensive_rebounds,
                            defensive_rebounds = EXCLUDED.defensive_rebounds,
                            total_rebounds = EXCLUDED.total_rebounds,
                            assists = EXCLUDED.assists,
                            turnovers = EXCLUDED.turnovers,
                            steals = EXCLUDED.steals,
                            blocks = EXCLUDED.blocks,
                            personal_fouls = EXCLUDED.personal_fouls,
                            plus_minus = EXCLUDED.plus_minus
                    ''', (
                        row['team_id'], row['season'], row['games_played'],
                        row['wins'], row['losses'], row['win_pct'],
                        row['minutes_played'], row['points'],
                        row['field_goals_made'], row['field_goals_attempted'],
                        row['field_goal_pct'], row['three_pointers_made'],
                        row['three_pointers_attempted'], row['three_point_pct'],
                        row['free_throws_made'], row['free_throws_attempted'],
                        row['free_throw_pct'], row['offensive_rebounds'],
                        row['defensive_rebounds'], row['total_rebounds'],
                        row['assists'], row['turnovers'], row['steals'],
                        row['blocks'], row['personal_fouls'], row['plus_minus']
                    ))
                else:
                    cursor.execute('''
                        INSERT OR REPLACE INTO season_stats
                        (team_id, season, games_played, wins, losses, win_pct,
                         minutes_played, points, field_goals_made, field_goals_attempted,
                         field_goal_pct, three_pointers_made, three_pointers_attempted,
                         three_point_pct, free_throws_made, free_throws_attempted,
                         free_throw_pct, offensive_rebounds, defensive_rebounds,
                         total_rebounds, assists, turnovers, steals, blocks,
                         personal_fouls, plus_minus)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row['team_id'], row['season'], row['games_played'],
                        row['wins'], row['losses'], row['win_pct'],
                        row['minutes_played'], row['points'],
                        row['field_goals_made'], row['field_goals_attempted'],
                        row['field_goal_pct'], row['three_pointers_made'],
                        row['three_pointers_attempted'], row['three_point_pct'],
                        row['free_throws_made'], row['free_throws_attempted'],
                        row['free_throw_pct'], row['offensive_rebounds'],
                        row['defensive_rebounds'], row['total_rebounds'],
                        row['assists'], row['turnovers'], row['steals'],
                        row['blocks'], row['personal_fouls'], row['plus_minus']
                    ))
            
            conn.commit()
            total_records += len(stats_df)
            print(f"  ✓ Saved {len(stats_df)} team season records for {season}")
    
    conn.close()
    print(f"✓ Total season stats records: {total_records}")

def update_current_season_game_logs(limit=None):
    """Fetch game logs for all teams in current season."""
    print(f"\nFetching game logs for {CURRENT_SEASON}...")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get all team IDs
    cursor.execute("SELECT team_id, abbreviation FROM teams")
    all_teams = cursor.fetchall()
    
    if limit:
        all_teams = all_teams[:limit]
    
    total_games = 0
    for idx, (team_id, abbr) in enumerate(all_teams, 1):
        try:
            # Fetch game log for this team
            gamelog = teamgamelog.TeamGameLog(
                team_id=team_id,
                season=CURRENT_SEASON
            )
            df = gamelog.get_data_frames()[0]
            
            if not df.empty:
                # Prepare game log data
                games_df = pd.DataFrame({
                    'game_id': df['Game_ID'],
                    'team_id': team_id,
                    'season': CURRENT_SEASON,
                    'game_date': df['GAME_DATE'],
                    'matchup': df['MATCHUP'],
                    'win_loss': df['WL'],
                    'minutes': df['MIN'],
                    'points': df['PTS'],
                    'field_goals_made': df['FGM'],
                    'field_goals_attempted': df['FGA'],
                    'field_goal_pct': df['FG_PCT'],
                    'three_pointers_made': df['FG3M'],
                    'three_pointers_attempted': df['FG3A'],
                    'three_point_pct': df['FG3_PCT'],
                    'free_throws_made': df['FTM'],
                    'free_throws_attempted': df['FTA'],
                    'free_throw_pct': df['FT_PCT'],
                    'offensive_rebounds': df['OREB'],
                    'defensive_rebounds': df['DREB'],
                    'total_rebounds': df['REB'],
                    'assists': df['AST'],
                    'turnovers': df['TOV'],
                    'steals': df['STL'],
                    'blocks': df['BLK'],
                    'personal_fouls': df['PF'],
                    'plus_minus': df.get('PLUS_MINUS', 0)  # Use get() with default
                })
                
                # Insert game logs
                if USE_POSTGRES:
                    # Manually insert for PostgreSQL
                    for _, game in games_df.iterrows():
                        cursor.execute('''
                            INSERT INTO team_game_logs
                            (game_id, team_id, season, game_date, matchup, win_loss,
                             minutes, points, field_goals_made, field_goals_attempted,
                             field_goal_pct, three_pointers_made, three_pointers_attempted,
                             three_point_pct, free_throws_made, free_throws_attempted,
                             free_throw_pct, offensive_rebounds, defensive_rebounds,
                             total_rebounds, assists, turnovers, steals, blocks,
                             personal_fouls, plus_minus)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (game_id, team_id) DO UPDATE SET
                                game_date = EXCLUDED.game_date,
                                matchup = EXCLUDED.matchup,
                                win_loss = EXCLUDED.win_loss,
                                minutes = EXCLUDED.minutes,
                                points = EXCLUDED.points,
                                field_goals_made = EXCLUDED.field_goals_made,
                                field_goals_attempted = EXCLUDED.field_goals_attempted,
                                field_goal_pct = EXCLUDED.field_goal_pct,
                                three_pointers_made = EXCLUDED.three_pointers_made,
                                three_pointers_attempted = EXCLUDED.three_pointers_attempted,
                                three_point_pct = EXCLUDED.three_point_pct,
                                free_throws_made = EXCLUDED.free_throws_made,
                                free_throws_attempted = EXCLUDED.free_throws_attempted,
                                free_throw_pct = EXCLUDED.free_throw_pct,
                                offensive_rebounds = EXCLUDED.offensive_rebounds,
                                defensive_rebounds = EXCLUDED.defensive_rebounds,
                                total_rebounds = EXCLUDED.total_rebounds,
                                assists = EXCLUDED.assists,
                                turnovers = EXCLUDED.turnovers,
                                steals = EXCLUDED.steals,
                                blocks = EXCLUDED.blocks,
                                personal_fouls = EXCLUDED.personal_fouls,
                                plus_minus = EXCLUDED.plus_minus
                        ''', (
                            game['game_id'], game['team_id'], game['season'],
                            game['game_date'], game['matchup'], game['win_loss'],
                            game['minutes'], game['points'], game['field_goals_made'],
                            game['field_goals_attempted'], game['field_goal_pct'],
                            game['three_pointers_made'], game['three_pointers_attempted'],
                            game['three_point_pct'], game['free_throws_made'],
                            game['free_throws_attempted'], game['free_throw_pct'],
                            game['offensive_rebounds'], game['defensive_rebounds'],
                            game['total_rebounds'], game['assists'], game['turnovers'],
                            game['steals'], game['blocks'], game['personal_fouls'],
                            game['plus_minus']
                        ))
                else:
                    # Use pandas for SQLite
                    games_df.to_sql('game_logs', conn, if_exists='append', index=False)
                    
                    # Remove duplicates for SQLite
                    cursor.execute('''
                        DELETE FROM team_game_logs
                        WHERE rowid NOT IN (
                            SELECT MAX(rowid)
                            FROM team_game_logs
                            GROUP BY game_id, team_id
                        )
                    ''')
                
                conn.commit()
                conn.commit()
                
                total_games += len(games_df)
                print(f"  Progress: {idx}/{len(all_teams)} teams ({abbr}: {len(games_df)} games)")
            
            time.sleep(0.6)  # Rate limiting
            
        except Exception as e:
            print(f"  ⚠ Error fetching game log for {abbr}: {e}")
            continue
    
    conn.close()
    print(f"✓ Total game logs: {total_games} games")

def initial_build(limit=None):
    """Perform initial database build."""
    print("=" * 60)
    print("INITIAL BUILD - NBA Team Stats Database")
    print("=" * 60)
    print(f"USE_POSTGRES: {USE_POSTGRES}")
    print(f"DATABASE_URL: {DATABASE_URL[:50]}..." if len(DATABASE_URL) > 50 else f"DATABASE_URL: {DATABASE_URL}")
    
    # Step 0: FORCE DROP and recreate tables for PostgreSQL
    if USE_POSTGRES:
        print("\n" + "=" * 60)
        print("FORCING COMPLETE REBUILD - Dropping all tables...")
        print("=" * 60)
        try:
            conn = get_connection()
            print(f"✓ Connected to PostgreSQL")
            cursor = conn.cursor()
            
            cursor.execute('DROP TABLE IF EXISTS team_game_logs CASCADE')
            print("  ✓ Dropped team_game_logs")
            
            cursor.execute('DROP TABLE IF EXISTS team_season_stats CASCADE')
            print("  ✓ Dropped team_season_stats")
            
            cursor.execute('DROP TABLE IF EXISTS teams CASCADE')
            print("  ✓ Dropped teams")
            
            conn.commit()
            print("  ✓ Changes committed")
            conn.close()
            print("=" * 60)
            print("✓ ALL TABLES DROPPED SUCCESSFULLY")
            print("=" * 60 + "\n")
        except Exception as e:
            print(f"ERROR dropping tables: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # Step 1: Create tables
    create_tables()
    
    # Step 2: Get and store team info
    all_teams = get_all_teams()
    update_team_info(all_teams)
    
    # Step 3: Fetch season stats for all seasons
    update_season_stats(SEASONS)
    
    # Step 4: Fetch game logs for current season
    update_current_season_game_logs(limit=limit)
    
    print("\n" + "=" * 60)
    print("✓ INITIAL BUILD COMPLETE")
    print("=" * 60)
    show_stats()

def update_database():
    """Update database with latest data (current season only)."""
    print("=" * 60)
    print(f"UPDATE - NBA Team Stats Database ({CURRENT_SEASON})")
    print("=" * 60)
    
    # Update ONLY current season stats
    print(f"\nUpdating season stats for {CURRENT_SEASON} only...")
    update_season_stats([CURRENT_SEASON])
    
    # Update current season game logs
    print(f"\nUpdating game logs for {CURRENT_SEASON}...")
    update_current_season_game_logs()
    
    print("\n" + "=" * 60)
    print(f"✓ UPDATE COMPLETE ({CURRENT_SEASON} only)")
    print("=" * 60)
    show_stats()

def show_stats():
    """Display database statistics."""
    conn = get_connection()
    cursor = conn.cursor()
    
    print("\n" + "=" * 60)
    print("DATABASE STATISTICS")
    print("=" * 60)
    
    # Teams count
    cursor.execute("SELECT COUNT(*) FROM teams")
    team_count = cursor.fetchone()[0]
    print(f"Teams: {team_count}")
    
    # Season stats count
    cursor.execute("SELECT COUNT(*) FROM team_season_stats")
    season_count = cursor.fetchone()[0]
    print(f"Season stats records: {season_count}")
    
    # Season breakdown
    cursor.execute("""
        SELECT season, COUNT(*) as count
        FROM team_season_stats
        GROUP BY season
        ORDER BY season
    """)
    print("\nSeason breakdown:")
    for season, count in cursor.fetchall():
        print(f"  {season}: {count} teams")
    
    # Game logs count
    cursor.execute("SELECT COUNT(*) FROM team_game_logs")
    game_count = cursor.fetchone()[0]
    print(f"\nGame logs: {game_count} total games")
    
    # Current season game count
    if USE_POSTGRES:
        cursor.execute("""
            SELECT COUNT(DISTINCT game_id) as games
            FROM team_game_logs
            WHERE season = %s
        """, (CURRENT_SEASON,))
    else:
        cursor.execute("""
            SELECT COUNT(DISTINCT game_id) as games
            FROM team_game_logs
            WHERE season = ?
        """, (CURRENT_SEASON,))
    current_games = cursor.fetchone()[0]
    print(f"  {CURRENT_SEASON} season: {current_games} games logged")
    
    conn.close()
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description='Build and update NBA team stats database'
    )
    parser.add_argument(
        '--initial',
        action='store_true',
        help='Perform initial database build (all seasons)'
    )
    parser.add_argument(
        '--update',
        action='store_true',
        help='Update database with latest data (current season only)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show database statistics'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of teams (for testing)'
    )
    
    args = parser.parse_args()
    
    if args.initial:
        initial_build(limit=args.limit)
    elif args.update:
        update_database()
    elif args.stats:
        show_stats()
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python3 build_team_stats_db.py --initial  # First time")
        print("  python3 build_team_stats_db.py --update   # Daily updates")
        print("  python3 build_team_stats_db.py --stats    # View stats")

if __name__ == '__main__':
    main()
