#!/usr/bin/env python3
"""
Query helper functions for team stats database.

Example usage:
    from query_team_stats import *
    
    # Get team's season stats
    get_team_season_stats('Lakers')
    
    # Get recent games
    get_recent_games('LAL', 10)
    
    # Compare teams
    compare_teams(['LAL', 'GSW', 'BOS'], '2025-26')
    
    # Get standings
    get_standings('2025-26')
"""

import sqlite3
import pandas as pd

DB_NAME = 'nba_team_stats.db'

def get_team_season_stats(team_name, seasons=None):
    """
    Get season stats for a team.
    
    Args:
        team_name: Team name or abbreviation (e.g., 'Lakers' or 'LAL')
        seasons: List of seasons or single season (default: all available)
    
    Returns:
        DataFrame with season stats
    """
    conn = sqlite3.connect(DB_NAME)
    
    if seasons is None:
        season_filter = ""
    elif isinstance(seasons, str):
        season_filter = f"AND s.season = '{seasons}'"
    else:
        seasons_str = "','".join(seasons)
        season_filter = f"AND s.season IN ('{seasons_str}')"
    
    query = f'''
        SELECT 
            t.full_name as Team,
            t.abbreviation as Abbr,
            s.season as Season,
            s.games_played as GP,
            s.wins as W,
            s.losses as L,
            ROUND(s.win_pct, 3) as "Win%",
            ROUND(s.points, 1) as PPG,
            ROUND(s.total_rebounds, 1) as RPG,
            ROUND(s.assists, 1) as APG,
            ROUND(s.field_goal_pct, 3) as "FG%",
            ROUND(s.three_point_pct, 3) as "3P%",
            ROUND(s.plus_minus, 1) as "+/-"
        FROM season_stats s
        JOIN teams t ON s.team_id = t.team_id
        WHERE (t.full_name LIKE '%{team_name}%' 
               OR t.abbreviation LIKE '%{team_name}%'
               OR t.nickname LIKE '%{team_name}%')
        {season_filter}
        ORDER BY s.season DESC
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print(f"No stats found for team: {team_name}")
        return df
    
    print(f"\n{df.iloc[0]['Team']} - Season Stats:")
    print("=" * 80)
    print(df.to_string(index=False))
    return df

def get_recent_games(team_abbr, num_games=10):
    """
    Get recent game logs for a team.
    
    Args:
        team_abbr: Team abbreviation (e.g., 'LAL')
        num_games: Number of recent games to retrieve
    
    Returns:
        DataFrame with recent games
    """
    conn = sqlite3.connect(DB_NAME)
    
    query = f'''
        SELECT 
            g.game_date as Date,
            g.matchup as Matchup,
            g.win_loss as "W/L",
            g.points as PTS,
            g.total_rebounds as REB,
            g.assists as AST,
            g.field_goal_pct as "FG%",
            g.three_point_pct as "3P%",
            g.plus_minus as "+/-"
        FROM game_logs g
        JOIN teams t ON g.team_id = t.team_id
        WHERE t.abbreviation = '{team_abbr}'
        ORDER BY g.game_date DESC
        LIMIT {num_games}
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print(f"No recent games found for team: {team_abbr}")
        return df
    
    print(f"\n{team_abbr} - Last {num_games} Games:")
    print("=" * 80)
    print(df.to_string(index=False))
    return df

def get_standings(season='2025-26', conference=None):
    """
    Get standings for a season.
    
    Args:
        season: Season (e.g., '2025-26')
        conference: 'East' or 'West' (default: both)
    
    Returns:
        DataFrame with standings
    """
    conn = sqlite3.connect(DB_NAME)
    
    # Note: conference info not in current schema, showing all teams
    query = f'''
        SELECT 
            ROW_NUMBER() OVER (ORDER BY s.win_pct DESC) as Rank,
            t.full_name as Team,
            s.wins as W,
            s.losses as L,
            ROUND(s.win_pct, 3) as "Win%",
            ROUND(s.points, 1) as PPG,
            ROUND(s.total_rebounds, 1) as RPG,
            ROUND(s.assists, 1) as APG
        FROM season_stats s
        JOIN teams t ON s.team_id = t.team_id
        WHERE s.season = '{season}'
        ORDER BY s.win_pct DESC, s.wins DESC
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print(f"No standings found for season: {season}")
        return df
    
    print(f"\nNBA Standings - {season} Season:")
    print("=" * 80)
    print(df.to_string(index=False))
    return df

def compare_teams(team_abbrs, season='2025-26'):
    """
    Compare multiple teams for a season.
    
    Args:
        team_abbrs: List of team abbreviations (e.g., ['LAL', 'GSW'])
        season: Season to compare
    
    Returns:
        DataFrame with comparison
    """
    conn = sqlite3.connect(DB_NAME)
    
    abbrs_str = "','".join(team_abbrs)
    
    query = f'''
        SELECT 
            t.abbreviation as Team,
            s.wins as W,
            s.losses as L,
            ROUND(s.win_pct, 3) as "Win%",
            ROUND(s.points, 1) as PPG,
            ROUND(s.total_rebounds, 1) as RPG,
            ROUND(s.assists, 1) as APG,
            ROUND(s.field_goal_pct, 3) as "FG%",
            ROUND(s.three_point_pct, 3) as "3P%",
            ROUND(s.steals, 1) as SPG,
            ROUND(s.blocks, 1) as BPG
        FROM season_stats s
        JOIN teams t ON s.team_id = t.team_id
        WHERE t.abbreviation IN ('{abbrs_str}')
        AND s.season = '{season}'
        ORDER BY s.win_pct DESC
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print(f"No stats found for teams: {team_abbrs}")
        return df
    
    print(f"\nTeam Comparison - {season} Season:")
    print("=" * 80)
    print(df.to_string(index=False))
    return df

def get_team_avg_last_n_games(team_abbr, n_games=5):
    """
    Get team's average stats over last N games.
    
    Args:
        team_abbr: Team abbreviation
        n_games: Number of recent games to average
    
    Returns:
        DataFrame with averages
    """
    conn = sqlite3.connect(DB_NAME)
    
    query = f'''
        SELECT 
            t.full_name as Team,
            COUNT(*) as Games,
            ROUND(AVG(g.points), 1) as PPG,
            ROUND(AVG(g.total_rebounds), 1) as RPG,
            ROUND(AVG(g.assists), 1) as APG,
            ROUND(AVG(g.field_goal_pct), 3) as "FG%",
            ROUND(AVG(g.three_point_pct), 3) as "3P%",
            ROUND(AVG(g.plus_minus), 1) as "+/-"
        FROM (
            SELECT * FROM game_logs
            WHERE team_id = (SELECT team_id FROM teams WHERE abbreviation = '{team_abbr}')
            ORDER BY game_date DESC
            LIMIT {n_games}
        ) g
        JOIN teams t ON g.team_id = t.team_id
        GROUP BY t.full_name
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print(f"No recent games found for team: {team_abbr}")
        return df
    
    print(f"\n{team_abbr} - Last {n_games} Games Average:")
    print("=" * 80)
    print(df.to_string(index=False))
    return df

def get_top_offensive_teams(season='2025-26', min_games=10):
    """
    Get top offensive teams by PPG.
    
    Args:
        season: Season
        min_games: Minimum games played
    
    Returns:
        DataFrame with top scorers
    """
    conn = sqlite3.connect(DB_NAME)
    
    query = f'''
        SELECT 
            ROW_NUMBER() OVER (ORDER BY s.points DESC) as Rank,
            t.full_name as Team,
            s.games_played as GP,
            ROUND(s.points, 1) as PPG,
            ROUND(s.field_goal_pct, 3) as "FG%",
            ROUND(s.three_point_pct, 3) as "3P%",
            ROUND(s.assists, 1) as APG
        FROM season_stats s
        JOIN teams t ON s.team_id = t.team_id
        WHERE s.season = '{season}'
        AND s.games_played >= {min_games}
        ORDER BY s.points DESC
        LIMIT 10
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"\nTop Offensive Teams - {season} Season:")
    print("=" * 80)
    print(df.to_string(index=False))
    return df

def get_top_defensive_teams(season='2025-26', min_games=10):
    """
    Get top defensive teams by blocks and steals.
    
    Args:
        season: Season
        min_games: Minimum games played
    
    Returns:
        DataFrame with top defensive teams
    """
    conn = sqlite3.connect(DB_NAME)
    
    query = f'''
        SELECT 
            t.full_name as Team,
            s.games_played as GP,
            ROUND(s.steals, 1) as SPG,
            ROUND(s.blocks, 1) as BPG,
            ROUND(s.defensive_rebounds, 1) as DRPG,
            ROUND(s.steals + s.blocks, 1) as "STL+BLK"
        FROM season_stats s
        JOIN teams t ON s.team_id = t.team_id
        WHERE s.season = '{season}'
        AND s.games_played >= {min_games}
        ORDER BY (s.steals + s.blocks) DESC
        LIMIT 10
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"\nTop Defensive Teams - {season} Season:")
    print("=" * 80)
    print(df.to_string(index=False))
    return df

# Example usage
if __name__ == '__main__':
    print("Team Stats Query Examples")
    print("=" * 80)
    
    
    print("Compare Lakers, Warriors, Celtics:")
    compare_teams(['LAL', 'GSW', 'BOS'], '2025-26')
    
    print("Current standings:")
    get_standings('2025-26')
    
    print("Top scoring teams:")
    get_top_offensive_teams('2025-26')
