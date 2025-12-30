#!/usr/bin/env python3
"""Predict today's games with calibrated probabilities.

This is a wrapper around predict_today.py that applies probability calibration
to reduce home-team bias and improve prediction accuracy.

Usage:
  python3 predict_today_calibrated.py --ensemble
  python3 predict_today_calibrated.py --single
"""

import argparse
import os
import sys
from datetime import datetime, date
import sqlite3
import time

from nba_model import NBAGamePredictor, EnsembleNBAPredictor
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.endpoints import boxscoretraditionalv3

# NBA team mapping for when API doesn't provide team info
NBA_TEAMS = {
    1610612737: 'ATL', 1610612738: 'BOS', 1610612751: 'BKN', 1610612766: 'CHA',
    1610612741: 'CHI', 1610612739: 'CLE', 1610612742: 'DAL', 1610612743: 'DEN',
    1610612765: 'DET', 1610612744: 'GSW', 1610612745: 'HOU', 1610612754: 'IND',
    1610612746: 'LAC', 1610612747: 'LAL', 1610612763: 'MEM', 1610612748: 'MIA',
    1610612749: 'MIL', 1610612750: 'MIN', 1610612740: 'NOP', 1610612752: 'NYK',
    1610612760: 'OKC', 1610612753: 'ORL', 1610612755: 'PHI', 1610612756: 'PHX',
    1610612757: 'POR', 1610612758: 'SAC', 1610612759: 'SAS', 1610612761: 'TOR',
    1610612762: 'UTA', 1610612764: 'WAS'
}


def load_calibration(path='calibration_saved/calibration.pkl'):
    """Load calibration model if available."""
    if not os.path.exists(path):
        print(f"⚠️  No calibration found at {path}")
        print("   Run: python3 calibrate_model.py --ensemble --from-logs")
        return None
    
    try:
        from calibrate_model import apply_calibration
        import pickle
        
        with open(path, 'rb') as f:
            cal_data = pickle.load(f)
        
        print(f"✅ Loaded calibration (method: {cal_data['method']}, offset: {cal_data['offset']:+.3f})")
        return cal_data
    except Exception as e:
        print(f"⚠️  Could not load calibration: {e}")
        return None


def apply_cal(raw_prob, cal_data):
    """Apply calibration to a probability."""
    if cal_data is None:
        return raw_prob
    
    import numpy as np
    
    method = cal_data['method']
    
    if method == 'isotonic':
        calibrated = cal_data['calibrator'].transform([raw_prob])[0]
    elif method == 'platt':
        raw_clipped = np.clip(raw_prob, 1e-7, 1 - 1e-7)
        logit = np.log(raw_clipped / (1 - raw_clipped))
        calibrated = cal_data['calibrator'].predict_proba([[logit]])[0, 1]
    elif method == 'offset':
        calibrated = raw_prob + cal_data['offset']
        calibrated = np.clip(calibrated, 0.0, 1.0)
    else:
        calibrated = raw_prob
    
    return float(calibrated)


def get_game_rosters(game_id):
    """Get player roster for a specific game."""
    try:
        boxscore = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
        player_stats = boxscore.player_stats.get_data_frame()
        return player_stats
    except Exception as e:
        return None


def extract_active_roster(player_stats_df, team_abbr):
    """Extract names of active players from boxscore dataframe."""
    if player_stats_df is None:
        return None
    
    # Handle both V2 and V3 column names
    team_col = 'teamTricode' if 'teamTricode' in player_stats_df.columns else 'TEAM_ABBREVIATION'
    player_col = 'nameI' if 'nameI' in player_stats_df.columns else 'PLAYER_NAME'
    min_col = 'minutes' if 'minutes' in player_stats_df.columns else 'MIN'
    
    team_players = player_stats_df[player_stats_df[team_col] == team_abbr]
    
    if team_players.empty:
        return None
    
    # Get players who played (have minutes > 0)
    if min_col in team_players.columns:
        # V3 uses actual minute values (float), V2 uses string format
        if player_stats_df[min_col].dtype == 'object':
            active = team_players[team_players[min_col].notna() & (team_players[min_col] != '0:00')]
        else:
            active = team_players[team_players[min_col].notna() & (team_players[min_col] > 0)]
    else:
        active = team_players
    
    if len(active) > 0:
        return active[player_col].tolist()
    else:
        # If game hasn't started, return all listed players
        return team_players[player_col].tolist() if len(team_players) > 0 else None


def main():
    parser = argparse.ArgumentParser(description='Predict NBA games with calibration')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble model')
    parser.add_argument('--single', action='store_true', help='Use single model')
    parser.add_argument('--no-calibration', action='store_true', help='Skip calibration')
    parser.add_argument('--rosters', action='store_true', 
                       help='Use live game rosters when available (factors in injuries, rotations)')
    parser.add_argument('--calibration-path', type=str, 
                       default='calibration_saved/calibration.pkl',
                       help='Path to calibration file')
    parser.add_argument('--ensemble-path', type=str, default='ensemble_model_saved')
    parser.add_argument('--model-path', type=str, default='nba_model_saved')
    
    args = parser.parse_args()
    
    print("="*80)
    print("NBA PREDICTIONS WITH CALIBRATION")
    if args.rosters:
        print("🎯 ROSTER MODE: Using live game rosters when available")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n")
    
    # Load calibration
    cal_data = None if args.no_calibration else load_calibration(args.calibration_path)
    
    # Load model
    if args.ensemble:
        print(f"\nLoading ensemble from {args.ensemble_path}...")
        predictor = EnsembleNBAPredictor()
        predictor.load_ensemble(args.ensemble_path)
        mode = "ensemble"
    elif args.single:
        print(f"\nLoading model from {args.model_path}...")
        predictor = NBAGamePredictor(model_path=args.model_path)
        mode = "single"
    else:
        print("ERROR: Specify --ensemble or --single")
        return
    
    # Get today's games
    print("\nFetching today's games...")
    today = date.today()
    board = scoreboardv2.ScoreboardV2(game_date=today.strftime('%m/%d/%Y'))
    
    # Get game header data
    game_header_df = board.game_header.get_data_frame()
    
    if game_header_df.empty:
        print("No games scheduled today")
        return
    
    # Try to get team info from line_score first
    line_score_df = board.line_score.get_data_frame()
    
    games = []
    for _, game_row in game_header_df.iterrows():
        game_id = game_row['GAME_ID']
        
        # Try to get team IDs from line_score for this game (works if game started)
        game_teams = line_score_df[line_score_df['GAME_ID'] == game_id]
        
        if len(game_teams) >= 2:
            # Line score has both teams (game started)
            home_team = game_teams.iloc[0]
            away_team = game_teams.iloc[1]
            
            games.append({
                'gameId': game_id,
                'homeTeam': {
                    'teamId': int(home_team['TEAM_ID']),
                    'teamTricode': home_team['TEAM_ABBREVIATION']
                },
                'awayTeam': {
                    'teamId': int(away_team['TEAM_ID']),
                    'teamTricode': away_team['TEAM_ABBREVIATION']
                },
                'gameStatus': game_row['GAME_STATUS_TEXT']
            })
        else:
            # Game hasn't started - try to get from boxscore
            try:
                time.sleep(0.6)  # Rate limit
                boxscore = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
                team_stats = boxscore.team_stats.get_data_frame()
                
                if len(team_stats) >= 2:
                    # Get home/away teams (need to determine which is which)
                    team1_id = int(team_stats.iloc[0]['teamId'])
                    team2_id = int(team_stats.iloc[1]['teamId'])
                    team1_abbr = team_stats.iloc[0]['teamTricode']
                    team2_abbr = team_stats.iloc[1]['teamTricode']
                    
                    # Boxscore doesn't always indicate home/away clearly
                    # We'll use the first team as home by convention
                    games.append({
                        'gameId': game_id,
                        'homeTeam': {
                            'teamId': team1_id,
                            'teamTricode': team1_abbr
                        },
                        'awayTeam': {
                            'teamId': team2_id,
                            'teamTricode': team2_abbr
                        },
                        'gameStatus': game_row['GAME_STATUS_TEXT']
                    })
                    continue
            except:
                pass
            
            # Last resort: skip this game
            print(f"⚠️  Skipping game {game_id} - unable to determine teams")
            print("    (NBA API no longer provides team IDs for future games)")
    
    print(f"Found {len(games)} games\n")
    print("="*80)
    
    # Predict each game
    for i, game in enumerate(games, 1):
        home = game['homeTeam']
        away = game['awayTeam']
        
        home_id = home['teamId']
        away_id = away['teamId']
        home_name = home['teamTricode']
        away_name = away['teamTricode']
        
        # Game status
        status_text = game['gameStatus']
        
        print(f"\nGame {i}: {away_name} @ {home_name}")
        print(f"Status: {status_text}")
        
        # Try to get rosters if roster mode is enabled
        home_roster = None
        away_roster = None
        
        if args.rosters:
            try:
                player_stats = get_game_rosters(game['gameId'])
                if player_stats is not None:
                    home_roster = extract_active_roster(player_stats, home_name)
                    away_roster = extract_active_roster(player_stats, away_name)
                    
                    if home_roster and away_roster:
                        print(f"\n📋 Active Rosters:")
                        print(f"   {home_name}: {len(home_roster)} players")
                        print(f"   {away_name}: {len(away_roster)} players")
            except Exception as e:
                print(f"\n⚠️  Could not fetch rosters: {e}")
            
            # Rate limiting for NBA API
            time.sleep(0.6)
        
        try:
            # Get prediction
            pred = predictor.predict(home_id, away_id,
                                   seasons=['2022-23', '2023-24', '2024-25', '2025-26'],
                                   home_roster=home_roster,
                                   away_roster=away_roster)
            
            raw_home_prob = pred['home_win_probability']
            raw_away_prob = pred['away_win_probability']
            
            # Apply calibration
            if cal_data:
                cal_home_prob = apply_cal(raw_home_prob, cal_data)
                cal_away_prob = 1 - cal_home_prob
                
                print(f"\nRaw Prediction:")
                print(f"  {home_name}: {raw_home_prob:.1%} | {away_name}: {raw_away_prob:.1%}")
                print(f"\nCalibrated Prediction:")
                print(f"  {home_name}: {cal_home_prob:.1%} | {away_name}: {cal_away_prob:.1%}")
                
                winner = home_name if cal_home_prob > 0.5 else away_name
                confidence = max(cal_home_prob, cal_away_prob)
            else:
                cal_home_prob = raw_home_prob
                cal_away_prob = raw_away_prob
                
                print(f"\nPrediction:")
                print(f"  {home_name}: {raw_home_prob:.1%} | {away_name}: {raw_away_prob:.1%}")
                
                winner = home_name if raw_home_prob > 0.5 else away_name
                confidence = max(raw_home_prob, raw_away_prob)
            
            print(f"\n🏆 Predicted Winner: {winner} ({confidence:.1%})")
            print(f"📊 Predicted Score: {home_name} {pred['predicted_home_points']:.1f} - "
                  f"{away_name} {pred['predicted_away_points']:.1f}")
            
            # Indicate if using roster-based prediction
            if args.rosters:
                if home_roster and away_roster:
                    print(f"   ✅ Using live roster data")
                else:
                    print(f"   ℹ️  Using default team stats (roster not available)")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 80)
    
    print("\n" + "="*80)
    print("PREDICTIONS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
