"""Get active rosters for games (who's playing vs. on bench).

Run:
  python3 game_rosters.py

This fetches today's games and shows which players are active/inactive.
"""

from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints import boxscoretraditionalv2
import time

def get_todays_games():
    """Get today's scoreboard."""
    games = scoreboard.ScoreBoard()
    games_dict = games.get_dict()
    return games_dict.get('scoreboard', {}).get('games', [])

def get_game_rosters(game_id):
    """Get detailed roster info for a specific game (active players + bench)."""
    try:
        boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        
        # Get player stats (includes everyone who played or is on bench)
        player_stats = boxscore.player_stats.get_data_frame()
        
        # Get team stats for team names
        team_stats = boxscore.team_stats.get_data_frame()
        
        return player_stats, team_stats
    except Exception as e:
        print(f"Error fetching boxscore for game {game_id}: {e}")
        return None, None

def display_game_rosters(game_id, home_team, away_team):
    """Display active roster for a game."""
    player_stats, team_stats = get_game_rosters(game_id)
    
    if player_stats is None:
        print(f"Could not fetch roster for {away_team} @ {home_team}")
        return
    
    print(f"\n{'='*80}")
    print(f"GAME ROSTER: {away_team} @ {home_team}")
    print(f"Game ID: {game_id}")
    print(f"{'='*80}")
    
    # Split by team
    for team_id in player_stats['TEAM_ID'].unique():
        team_players = player_stats[player_stats['TEAM_ID'] == team_id]
        team_name = team_players.iloc[0]['TEAM_ABBREVIATION']
        
        print(f"\n{team_name} ROSTER:")
        print("-" * 80)
        
        # Active players (those who played - have minutes > 0)
        active = team_players[team_players['MIN'].notna() & (team_players['MIN'] != '0:00')]
        inactive = team_players[team_players['MIN'].isna() | (team_players['MIN'] == '0:00')]
        
        if len(active) > 0:
            print(f"\nACTIVE PLAYERS ({len(active)}):")
            for idx, player in active.iterrows():
                mins = player['MIN'] if player['MIN'] else '0:00'
                pts = player['PTS'] if player['PTS'] else 0
                reb = player['REB'] if player['REB'] else 0
                ast = player['AST'] if player['AST'] else 0
                print(f"  {player['PLAYER_NAME']:25s} {player['START_POSITION']:5s} "
                      f"MIN: {mins:5s}  PTS: {pts:2.0f}  REB: {reb:2.0f}  AST: {ast:2.0f}")
        
        if len(inactive) > 0:
            print(f"\nBENCH/INACTIVE ({len(inactive)}):")
            for idx, player in inactive.iterrows():
                print(f"  {player['PLAYER_NAME']:25s} {player['START_POSITION'] or '':5s} DNP")

def get_live_game_rosters():
    """Get rosters for all games today."""
    games = get_todays_games()
    
    if not games:
        print("No games found for today.")
        return
    
    print(f"\nFound {len(games)} game(s) today:\n")
    
    for game in games:
        game_id = game['gameId']
        home_team = game['homeTeam']['teamTricode']
        away_team = game['awayTeam']['teamTricode']
        game_status = game['gameStatusText']
        
        print(f"{away_team} @ {home_team} - {game_status}")
        
        # Only fetch detailed rosters for games that have started or finished
        if game['gameStatus'] >= 2:  # 2 = live, 3 = final
            display_game_rosters(game_id, home_team, away_team)
            time.sleep(0.6)  # Rate limiting
        else:
            print(f"  (Game hasn't started yet - roster not available)")
    
    print(f"\n{'='*80}")
    print("Done!")

def get_specific_game_roster(game_id):
    """Get roster for a specific game by game_id."""
    try:
        player_stats, team_stats = get_game_rosters(game_id)
        
        if player_stats is None:
            print(f"Could not fetch roster for game {game_id}")
            return None
        
        # Get team names
        home_team = player_stats[player_stats['TEAM_ID'] == player_stats.iloc[0]['TEAM_ID']].iloc[0]['TEAM_ABBREVIATION']
        away_team = player_stats[player_stats['TEAM_ID'] != player_stats.iloc[0]['TEAM_ID']].iloc[0]['TEAM_ABBREVIATION']
        
        display_game_rosters(game_id, home_team, away_team)
        return player_stats
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == '__main__':
    # Example 1: Get rosters for today's games
    print("Fetching rosters for today's games...")
    print("=" * 80)
    get_live_game_rosters()
    
    # Example 2: Get roster for a specific game (uncomment and use a valid game_id)
    # print("\n\nFetching roster for specific game...")
    # get_specific_game_roster('0022400123')
