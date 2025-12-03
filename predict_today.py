#!/usr/bin/env python3
"""
Predict today's NBA games with optional roster support
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
from nba_model import NBAGamePredictor, EnsembleNBAPredictor
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints import boxscoretraditionalv2
from datetime import datetime
import sqlite3
import time

def get_team_id_from_abbr(abbr):
    """Get team ID from abbreviation."""
    conn = sqlite3.connect('nba_team_stats.db')
    query = "SELECT team_id FROM teams WHERE abbreviation = ?"
    cursor = conn.execute(query, (abbr,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def get_game_rosters(game_id):
    """Get player roster for a specific game."""
    try:
        boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        player_stats = boxscore.player_stats.get_data_frame()
        return player_stats
    except Exception as e:
        return None

def extract_active_roster(player_stats_df, team_abbr):
    """Extract names of active players from boxscore dataframe."""
    if player_stats_df is None:
        return None
    
    team_players = player_stats_df[player_stats_df['TEAM_ABBREVIATION'] == team_abbr]
    
    # Get players who played (have minutes > 0)
    active = team_players[team_players['MIN'].notna() & (team_players['MIN'] != '0:00')]
    
    if len(active) > 0:
        return active['PLAYER_NAME'].tolist()
    else:
        # If game hasn't started, return all listed players
        return team_players['PLAYER_NAME'].tolist() if len(team_players) > 0 else None

def predict_todays_games(use_ensemble=False, model_path=None, use_rosters=False):
    """Predict all games scheduled for today."""
    print("="*80)
    print(f"NBA GAME PREDICTIONS - {datetime.now().strftime('%B %d, %Y')}")
    if use_rosters:
        print("🎯 ROSTER MODE: Using live game rosters when available")
    print("="*80)
    
    # Load the trained model
    print(f"\nLoading trained {'ensemble ' if use_ensemble else ''}model...")
    
    if use_ensemble:
        default_path = 'ensemble_model_saved'
        predictor = EnsembleNBAPredictor()
        predictor.load_ensemble(model_path or default_path)
        print(f"✓ Ensemble loaded ({len(predictor.models)} models)")
    else:
        default_path = 'nba_model_saved'
        predictor = NBAGamePredictor(model_path=model_path or default_path)
        print("✓ Model loaded")
    
    # Get today's games
    print("\nFetching today's games...")
    try:
        board = scoreboard.ScoreBoard()
        games = board.games.get_dict()
        
        if not games:
            print("No games scheduled for today.")
            return
        
        print(f"✓ Found {len(games)} game(s) today\n")
        
        # Make predictions for each game
        for i, game in enumerate(games, 1):
            home_team = game['homeTeam']['teamTricode']
            away_team = game['awayTeam']['teamTricode']
            game_status = game['gameStatusText']
            
            print(f"\nGame {i}: {away_team} @ {home_team}")
            print(f"Status: {game_status}")
            print("-" * 80)
            
            # Get team IDs
            home_id = get_team_id_from_abbr(home_team)
            away_id = get_team_id_from_abbr(away_team)
            
            if not home_id or not away_id:
                print(f"⚠️  Could not find team IDs for {home_team} or {away_team}")
                continue
            
            # Try to get rosters if roster mode is enabled
            home_roster = None
            away_roster = None
            
            if use_rosters and game['gameStatus'] >= 2:  # Game started or finished
                try:
                    player_stats = get_game_rosters(game['gameId'])
                    if player_stats is not None:
                        home_roster = extract_active_roster(player_stats, home_team)
                        away_roster = extract_active_roster(player_stats, away_team)
                        
                        if home_roster and away_roster:
                            print(f"\n📋 Active Rosters:")
                            print(f"   {home_team}: {len(home_roster)} players")
                            print(f"   {away_team}: {len(away_roster)} players")
                except Exception as e:
                    print(f"\n⚠️  Could not fetch rosters: {e}")
                
                # Rate limiting for NBA API
                time.sleep(0.6)
            
            # Make prediction
            try:
                prediction = predictor.predict(home_id, away_id, 
                                              seasons=['2022-23', '2023-24', '2024-25', '2025-26'],
                                              home_roster=home_roster,
                                              away_roster=away_roster)
                
                print(f"\n🏀 PREDICTION:")
                print(f"   Winner: {prediction['predicted_winner']}")
                print(f"\n   Win Probability:")
                print(f"      {home_team}: {prediction['home_win_probability']:.1%}")
                print(f"      {away_team}: {prediction['away_win_probability']:.1%}")
                print(f"\n   Predicted Score:")
                print(f"      {home_team}: {prediction['predicted_home_points']:.1f}")
                print(f"      {away_team}: {prediction['predicted_away_points']:.1f}")
                
                # Show individual model predictions if using ensemble
                if use_ensemble and 'individual_predictions' in prediction:
                    print(f"\n   📊 Individual Model Predictions:")
                    for idx, pred in enumerate(prediction['individual_predictions'], 1):
                        print(f"      Model {idx}: {pred['predicted_winner']} "
                              f"({pred['predicted_home_points']:.1f} - {pred['predicted_away_points']:.1f})")
                
                # Indicate if using roster-based prediction
                if use_rosters:
                    if home_roster and away_roster:
                        print(f"\n   ✅ Using live roster data")
                    else:
                        print(f"\n   ℹ️  Using default team stats (roster not available)")
                
                # Show actual score if game has started or finished
                if game['gameStatus'] != 1:  # Not "before game"
                    home_score = game['homeTeam']['score']
                    away_score = game['awayTeam']['score']
                    print(f"\n   📊 Current/Final Score:")
                    print(f"      {home_team}: {home_score}")
                    print(f"      {away_team}: {away_score}")
                    
            except Exception as e:
                print(f"❌ Error making prediction: {e}")
        
        print("\n" + "="*80)
        print("PREDICTIONS COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error fetching games: {e}")
        print("\nNote: This may happen if the NBA API is down or there are no games today.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict today\'s NBA games')
    parser.add_argument('--ensemble', action='store_true',
                       help='Use ensemble model instead of single model')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model (default: nba_model_saved or ensemble_model_saved)')
    parser.add_argument('--rosters', action='store_true',
                       help='Use live game rosters when available (factors in injuries, rotations)')
    
    args = parser.parse_args()
    predict_todays_games(use_ensemble=args.ensemble, model_path=args.model, use_rosters=args.rosters)
