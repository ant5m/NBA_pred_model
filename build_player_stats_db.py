"""Build and update a comprehensive player stats database.

This script creates/updates a SQLite database with:
- Player career info
- Season stats (last 3 seasons + current)
- Game-by-game logs (current season)
- Easy updates as new games are played

Run:
  python3 build_player_stats_db.py --initial    # First time setup
  python3 build_player_stats_db.py --update     # Update with latest games
"""

import os
import sqlite3
import argparse
from datetime import datetime
from nba_api.stats.endpoints import playercareerstats, playergamelog, leaguedashplayerstats
from nba_api.stats.static import players
import pandas as pd
import time

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Determine which database to use
if DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://"):
    USE_POSTGRES = True
    import psycopg2
    # Railway uses postgres://, but psycopg2 needs postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
else:
    USE_POSTGRES = False
    DB_PATH = 'nba_player_stats.db'

# Seasons to include (last 3 + current)
SEASONS = ['2023-24', '2024-25', '2025-26']


def get_connection():
    """Get database connection (PostgreSQL or SQLite)."""
    if USE_POSTGRES:
        return psycopg2.connect(DATABASE_URL)
    else:
        return sqlite3.connect(DB_PATH)
CURRENT_SEASON = '2025-26'


def create_tables(conn):
    """Create database tables if they don't exist."""
    cursor = conn.cursor()
    
    # Player info table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY,
            full_name TEXT,
            first_name TEXT,
            last_name TEXT,
            is_active INTEGER,
            last_updated TIMESTAMP
        )
    ''')
    
    # Season stats table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS season_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER,
            player_name TEXT,
            season TEXT,
            team_abbreviation TEXT,
            age INTEGER,
            games_played INTEGER,
            games_started INTEGER,
            minutes_played REAL,
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
            steals REAL,
            blocks REAL,
            turnovers REAL,
            personal_fouls REAL,
            points REAL,
            last_updated TIMESTAMP,
            UNIQUE(player_id, season, team_abbreviation)
        )
    ''')
    
    # Game log table (detailed game-by-game for current season)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS game_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER,
            player_name TEXT,
            season TEXT,
            game_id TEXT,
            game_date DATE,
            matchup TEXT,
            wl TEXT,
            minutes INTEGER,
            fgm INTEGER,
            fga INTEGER,
            fg_pct REAL,
            fg3m INTEGER,
            fg3a INTEGER,
            fg3_pct REAL,
            ftm INTEGER,
            fta INTEGER,
            ft_pct REAL,
            oreb INTEGER,
            dreb INTEGER,
            reb INTEGER,
            ast INTEGER,
            stl INTEGER,
            blk INTEGER,
            tov INTEGER,
            pf INTEGER,
            pts INTEGER,
            plus_minus INTEGER,
            last_updated TIMESTAMP,
            UNIQUE(player_id, game_id)
        )
    ''')
    
    conn.commit()
    print("✓ Database tables created/verified")


def get_all_active_players():
    """Get list of all active NBA players."""
    all_players = players.get_active_players()
    print(f"✓ Found {len(all_players)} active players")
    return all_players


def update_player_info(conn, player_list):
    """Update player basic info."""
    cursor = conn.cursor()
    now = datetime.now()
    
    for player in player_list:
        if USE_POSTGRES:
            cursor.execute('''
                INSERT INTO players 
                (player_id, full_name, first_name, last_name, is_active, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (player_id) DO UPDATE SET
                    full_name = EXCLUDED.full_name,
                    first_name = EXCLUDED.first_name,
                    last_name = EXCLUDED.last_name,
                    is_active = EXCLUDED.is_active,
                    last_updated = EXCLUDED.last_updated
            ''', (player['id'], player['full_name'], player['first_name'], 
                  player['last_name'], player['is_active'], now))
        else:
            cursor.execute('''
                INSERT OR REPLACE INTO players 
                (player_id, full_name, first_name, last_name, is_active, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (player['id'], player['full_name'], player['first_name'], 
                  player['last_name'], player['is_active'], now))
    
    conn.commit()
    print(f"✓ Updated {len(player_list)} player records")


def fetch_season_stats(season):
    """Fetch all player stats for a given season."""
    print(f"  Fetching stats for {season}...")
    
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed='PerGame'
        )
        df = stats.get_data_frames()[0]
        time.sleep(0.6)  # Rate limiting
        return df
    except Exception as e:
        print(f"  Error fetching {season} stats: {e}")
        return None


def update_season_stats(conn, seasons=SEASONS):
    """Update season statistics for specified seasons."""
    cursor = conn.cursor()
    now = datetime.now()
    
    total_records = 0
    
    for season in seasons:
        df = fetch_season_stats(season)
        
        if df is None:
            continue
        
        for _, row in df.iterrows():
            if USE_POSTGRES:
                cursor.execute('''
                    INSERT INTO season_stats
                    (player_id, player_name, season, team_abbreviation, age,
                     games_played, games_started, minutes_played,
                     field_goals_made, field_goals_attempted, field_goal_pct,
                     three_pointers_made, three_pointers_attempted, three_point_pct,
                     free_throws_made, free_throws_attempted, free_throw_pct,
                     offensive_rebounds, defensive_rebounds, total_rebounds,
                     assists, steals, blocks, turnovers, personal_fouls, points,
                     last_updated)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (player_id, season) DO UPDATE SET
                        player_name = EXCLUDED.player_name,
                        team_abbreviation = EXCLUDED.team_abbreviation,
                        age = EXCLUDED.age,
                        games_played = EXCLUDED.games_played,
                        games_started = EXCLUDED.games_started,
                        minutes_played = EXCLUDED.minutes_played,
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
                        steals = EXCLUDED.steals,
                        blocks = EXCLUDED.blocks,
                        turnovers = EXCLUDED.turnovers,
                        personal_fouls = EXCLUDED.personal_fouls,
                        points = EXCLUDED.points,
                        last_updated = EXCLUDED.last_updated
                ''', (
                    row['PLAYER_ID'], row['PLAYER_NAME'], season, row['TEAM_ABBREVIATION'],
                    row.get('AGE'),
                    row['GP'], row.get('GS', 0), row['MIN'],
                    row['FGM'], row['FGA'], row['FG_PCT'],
                    row['FG3M'], row['FG3A'], row['FG3_PCT'],
                    row['FTM'], row['FTA'], row['FT_PCT'],
                    row['OREB'], row['DREB'], row['REB'],
                    row['AST'], row['STL'], row['BLK'], row['TOV'], row['PF'], row['PTS'],
                    now
                ))
            else:
                cursor.execute('''
                    INSERT OR REPLACE INTO season_stats
                    (player_id, player_name, season, team_abbreviation, age,
                     games_played, games_started, minutes_played,
                     field_goals_made, field_goals_attempted, field_goal_pct,
                     three_pointers_made, three_pointers_attempted, three_point_pct,
                     free_throws_made, free_throws_attempted, free_throw_pct,
                     offensive_rebounds, defensive_rebounds, total_rebounds,
                     assists, steals, blocks, turnovers, personal_fouls, points,
                     last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['PLAYER_ID'], row['PLAYER_NAME'], season, row['TEAM_ABBREVIATION'],
                    row.get('AGE'),
                    row['GP'], row.get('GS', 0), row['MIN'],
                    row['FGM'], row['FGA'], row['FG_PCT'],
                    row['FG3M'], row['FG3A'], row['FG3_PCT'],
                    row['FTM'], row['FTA'], row['FT_PCT'],
                    row['OREB'], row['DREB'], row['REB'],
                    row['AST'], row['STL'], row['BLK'], row['TOV'], row['PF'], row['PTS'],
                    now
                ))
        
        total_records += len(df)
        conn.commit()
        print(f"  ✓ Saved {len(df)} player season records for {season}")
    
    print(f"✓ Total season stats records: {total_records}")


def update_current_season_game_logs(conn, player_list=None, limit=None):
    """Update game-by-game logs for current season."""
    cursor = conn.cursor()
    now = datetime.now()
    
    # If no player list provided, get active players from DB
    if player_list is None:
        cursor.execute("SELECT player_id, full_name FROM players WHERE is_active = 1")
        player_list = [{'id': row[0], 'full_name': row[1]} for row in cursor.fetchall()]
    
    if limit:
        player_list = player_list[:limit]
    
    print(f"  Fetching game logs for {len(player_list)} players in {CURRENT_SEASON}...")
    
    total_games = 0
    errors = 0
    
    for i, player in enumerate(player_list):
        try:
            gamelog = playergamelog.PlayerGameLog(
                player_id=player['id'],
                season=CURRENT_SEASON
            )
            df = gamelog.get_data_frames()[0]
            
            if len(df) > 0:
                for _, game in df.iterrows():
                    if USE_POSTGRES:
                        cursor.execute('''
                            INSERT INTO game_logs
                            (player_id, player_name, season, game_id, game_date, matchup, wl,
                             minutes, fgm, fga, fg_pct, fg3m, fg3a, fg3_pct,
                             ftm, fta, ft_pct, oreb, dreb, reb, ast, stl, blk, tov, pf, pts,
                             plus_minus, last_updated)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (player_id, game_id) DO UPDATE SET
                                player_name = EXCLUDED.player_name,
                                season = EXCLUDED.season,
                                game_date = EXCLUDED.game_date,
                                matchup = EXCLUDED.matchup,
                                wl = EXCLUDED.wl,
                                minutes = EXCLUDED.minutes,
                                fgm = EXCLUDED.fgm,
                                fga = EXCLUDED.fga,
                                fg_pct = EXCLUDED.fg_pct,
                                fg3m = EXCLUDED.fg3m,
                                fg3a = EXCLUDED.fg3a,
                                fg3_pct = EXCLUDED.fg3_pct,
                                ftm = EXCLUDED.ftm,
                                fta = EXCLUDED.fta,
                                ft_pct = EXCLUDED.ft_pct,
                                oreb = EXCLUDED.oreb,
                                dreb = EXCLUDED.dreb,
                                reb = EXCLUDED.reb,
                                ast = EXCLUDED.ast,
                                stl = EXCLUDED.stl,
                                blk = EXCLUDED.blk,
                                tov = EXCLUDED.tov,
                                pf = EXCLUDED.pf,
                                pts = EXCLUDED.pts,
                                plus_minus = EXCLUDED.plus_minus,
                                last_updated = EXCLUDED.last_updated
                        ''', (
                            player['id'], player['full_name'], CURRENT_SEASON,
                            game['Game_ID'], game['GAME_DATE'], game['MATCHUP'], game['WL'],
                            game.get('MIN'), game['FGM'], game['FGA'], game['FG_PCT'],
                            game['FG3M'], game['FG3A'], game['FG3_PCT'],
                            game['FTM'], game['FTA'], game['FT_PCT'],
                            game['OREB'], game['DREB'], game['REB'],
                            game['AST'], game['STL'], game['BLK'], game['TOV'], game['PF'], game['PTS'],
                            game.get('PLUS_MINUS'),
                            now
                        ))
                    else:
                        cursor.execute('''
                            INSERT OR REPLACE INTO game_logs
                            (player_id, player_name, season, game_id, game_date, matchup, wl,
                             minutes, fgm, fga, fg_pct, fg3m, fg3a, fg3_pct,
                             ftm, fta, ft_pct, oreb, dreb, reb, ast, stl, blk, tov, pf, pts,
                             plus_minus, last_updated)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            player['id'], player['full_name'], CURRENT_SEASON,
                            game['Game_ID'], game['GAME_DATE'], game['MATCHUP'], game['WL'],
                            game.get('MIN'), game['FGM'], game['FGA'], game['FG_PCT'],
                            game['FG3M'], game['FG3A'], game['FG3_PCT'],
                            game['FTM'], game['FTA'], game['FT_PCT'],
                            game['OREB'], game['DREB'], game['REB'],
                            game['AST'], game['STL'], game['BLK'], game['TOV'], game['PF'], game['PTS'],
                            game.get('PLUS_MINUS'),
                            now
                        ))
                
                total_games += len(df)
            
            if (i + 1) % 50 == 0:
                conn.commit()
                print(f"    Progress: {i + 1}/{len(player_list)} players ({total_games} games)")
            
            time.sleep(0.6)  # Rate limiting
            
        except Exception as e:
            errors += 1
            if errors < 5:  # Only print first few errors
                print(f"    Error for {player['full_name']}: {e}")
    
    conn.commit()
    print(f"  ✓ Saved {total_games} game log entries (errors: {errors})")


def initial_build(limit_players=None):
    
    conn = sqlite3.connect(DB_PATH)
    

    create_tables(conn)
    

    player_list = get_all_active_players()
    if limit_players:
        player_list = player_list[:limit_players]
        print(f"  (Limited to {limit_players} players for testing)")
    update_player_info(conn, player_list)
    

    update_season_stats(conn, SEASONS)
    

    update_current_season_game_logs(conn, player_list)
    
    conn.close()
    
   


def update_database():
    """Update database with latest data (current season only)."""
    print("=" * 80)
    print(f"UPDATE - NBA Player Stats Database ({CURRENT_SEASON})")
    print("=" * 80)
    
    conn = sqlite3.connect(DB_PATH)
    
    # Update ONLY current season stats
    print(f"\nUpdating season stats for {CURRENT_SEASON} only...")
    update_season_stats(conn, [CURRENT_SEASON])
    
    # Update current season game logs
    print(f"\nUpdating game logs for {CURRENT_SEASON}...")
    update_current_season_game_logs(conn)
    
    conn.close()
    
    print("\n" + "=" * 80)
    print(f"✓ UPDATE COMPLETE ({CURRENT_SEASON} only)")
    print("=" * 80)

def show_stats():
    """Show database statistics."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("\n" + "="*80)
    print("DATABASE STATISTICS")
    print("="*80 + "\n")
    
    cursor.execute("SELECT COUNT(*) FROM players")
    print(f"Total players: {cursor.fetchone()[0]:,}")
    
    cursor.execute("SELECT COUNT(*) FROM season_stats")
    print(f"Season stats records: {cursor.fetchone()[0]:,}")
    
    cursor.execute("SELECT COUNT(*) FROM game_logs")
    print(f"Game log entries: {cursor.fetchone()[0]:,}")
    
    cursor.execute("SELECT season, COUNT(*) FROM season_stats GROUP BY season")
    print("\nRecords per season:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]:,} records")
    
    cursor.execute("SELECT last_updated FROM season_stats ORDER BY last_updated DESC LIMIT 1")
    last_update = cursor.fetchone()
    if last_update:
        print(f"\nLast updated: {last_update[0]}")
    
    conn.close()
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build and update NBA player stats database')
    parser.add_argument('--initial', action='store_true', help='Initial database build')
    parser.add_argument('--update', action='store_true', help='Update existing database')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    parser.add_argument('--limit', type=int, help='Limit number of players (for testing)')
    
    args = parser.parse_args()
    
    if args.initial:
        initial_build(limit_players=args.limit)
        show_stats()
    elif args.update:
        update_database()
        show_stats()
    elif args.stats:
        show_stats()
    else:
        print("Usage:")
        print("  Initial build:  python3 build_player_stats_db.py --initial")
        print("  Update:         python3 build_player_stats_db.py --update")
        print("  Show stats:     python3 build_player_stats_db.py --stats")
        print("\nOptions:")
        print("  --limit N       Limit to N players (for testing)")
