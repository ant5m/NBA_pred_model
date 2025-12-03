#!/usr/bin/env python3
"""
Fetch historical game logs from previous seasons to build a larger training dataset.
"""

import sqlite3
import time
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams

DB_NAME = 'nba_team_stats.db'

def fetch_historical_game_logs(seasons=['2022-23', '2023-24', '2024-25']):
    """Fetch game logs from multiple previous seasons."""
    
    print("="*80)
    print("FETCHING HISTORICAL GAME LOGS")
    print("="*80)
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    all_teams = teams.get_teams()
    total_games = 0
    
    for season in seasons:
        print(f"\n{'='*80}")
        print(f"Season: {season}")
        print('='*80)
        
        season_games = 0
        
        for idx, team in enumerate(all_teams, 1):
            team_id = team['id']
            team_abbr = team['abbreviation']
            
            try:
                # Fetch game logs for this team and season
                gamelog = teamgamelog.TeamGameLog(
                    team_id=team_id,
                    season=season,
                    season_type_all_star='Regular Season'
                )
                
                time.sleep(0.6)  # Rate limiting
                
                df = gamelog.get_data_frames()[0]
                
                if df.empty:
                    print(f"  {idx}/30 {team_abbr}: No games found")
                    continue
                
                # Prepare data for insertion
                for _, game in df.iterrows():
                    # Handle PLUS_MINUS which may not exist in older data
                    plus_minus = game.get('PLUS_MINUS', 0) if 'PLUS_MINUS' in game else 0
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO game_logs (
                            game_id, team_id, season, game_date, matchup, win_loss,
                            minutes, points, field_goals_made, field_goals_attempted, field_goal_pct,
                            three_pointers_made, three_pointers_attempted, three_point_pct,
                            free_throws_made, free_throws_attempted, free_throw_pct,
                            offensive_rebounds, defensive_rebounds, total_rebounds,
                            assists, turnovers, steals, blocks, personal_fouls, plus_minus
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        game['Game_ID'], team_id, season, game['GAME_DATE'], game['MATCHUP'], game['WL'],
                        game.get('MIN', 0), game['PTS'], game['FGM'], game['FGA'], game['FG_PCT'],
                        game['FG3M'], game['FG3A'], game['FG3_PCT'],
                        game['FTM'], game['FTA'], game['FT_PCT'],
                        game['OREB'], game['DREB'], game['REB'],
                        game['AST'], game['TOV'], game['STL'], game['BLK'], game['PF'], plus_minus
                    ))
                
                games_count = len(df)
                season_games += games_count
                print(f"  {idx}/30 {team_abbr}: {games_count} games")
                
            except Exception as e:
                print(f"  {idx}/30 {team_abbr}: Error - {e}")
                continue
        
        conn.commit()
        total_games += season_games
        print(f"\n✓ Season {season}: {season_games} total game logs")
    
    conn.close()
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETE: {total_games} historical game logs fetched")
    print('='*80)


def show_database_stats():
    """Show statistics about the game logs database."""
    conn = sqlite3.connect(DB_NAME)
    
    print("\n" + "="*80)
    print("DATABASE STATISTICS")
    print("="*80)
    
    # Games per season
    query = """
        SELECT season, COUNT(DISTINCT game_id) as games
        FROM game_logs
        GROUP BY season
        ORDER BY season
    """
    
    df = pd.read_sql_query(query, conn)
    
    print("\nGames per season:")
    for _, row in df.iterrows():
        print(f"  {row['season']}: {row['games']} games")
    
    print(f"\nTotal unique games: {df['games'].sum()}")
    
    # Average points per season
    query = """
        SELECT season, AVG(points) as avg_points
        FROM game_logs
        GROUP BY season
        ORDER BY season
    """
    
    df = pd.read_sql_query(query, conn)
    
    print("\nAverage points per game (per team):")
    for _, row in df.iterrows():
        print(f"  {row['season']}: {row['avg_points']:.1f} ppg")
    
    conn.close()


if __name__ == '__main__':
    import pandas as pd
    
    print("This will fetch game logs from 2022-23, 2023-24, and 2024-25 seasons")
    print("This may take 10-15 minutes due to API rate limiting...")
    print("\nProceed? (yes/no): ", end='')
    
    response = input().strip().lower()
    
    if response in ['yes', 'y']:
        fetch_historical_game_logs()
        show_database_stats()
        
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Retrain model with more data:")
        print("   python3.11 nba_model.py")
        print("\n2. This will give you ~3,500+ games instead of 286")
        print("   Much better training data for accurate predictions!")
    else:
        print("\nCancelled.")
