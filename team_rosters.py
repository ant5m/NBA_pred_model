"""Get current rosters for all NBA teams.

Run:
  python3 team_rosters.py

This will fetch and display the current roster for each team.
"""

from nba_api.stats.static import teams
from nba_api.stats.endpoints import commonteamroster
import time

def get_all_team_rosters():
    """Fetch rosters for all NBA teams."""
    # Get all teams
    nba_teams = teams.get_teams()
    
    print(f"Fetching rosters for {len(nba_teams)} teams...\n")
    
    for team in nba_teams:
        team_id = team['id']
        team_name = team['full_name']
        team_abbr = team['abbreviation']
        
        try:
            # Fetch roster for this team
            roster = commonteamroster.CommonTeamRoster(team_id=team_id)
            roster_df = roster.get_data_frames()[0]
            
            print(f"\n{team_name} ({team_abbr}) - {len(roster_df)} players:")
            print("-" * 80)
            
            # Display roster
            for idx, player in roster_df.iterrows():
                print(f"  #{player['NUM']:3s} {player['PLAYER']:30s} {player['POSITION']:5s} "
                      f"Exp: {player['EXP']:2s}  Age: {player['AGE']}")
            
            # Small delay to avoid rate limiting
            time.sleep(0.6)
            
        except Exception as e:
            print(f"  Error fetching roster for {team_name}: {e}")
    
    print("\n" + "=" * 80)
    print("Done!")


def get_single_team_roster(team_abbreviation):
    """Fetch roster for a specific team by abbreviation (e.g., 'LAL', 'GSW')."""
    nba_teams = teams.get_teams()
    team = [t for t in nba_teams if t['abbreviation'] == team_abbreviation.upper()]
    
    if not team:
        print(f"Team '{team_abbreviation}' not found!")
        return
    
    team = team[0]
    team_id = team['id']
    team_name = team['full_name']
    
    try:
        roster = commonteamroster.CommonTeamRoster(team_id=team_id)
        roster_df = roster.get_data_frames()[0]
        
        print(f"\n{team_name} ({team_abbreviation.upper()}) - {len(roster_df)} players:")
        print("=" * 80)
        
        for idx, player in roster_df.iterrows():
            print(f"  #{player['NUM']:3s} {player['PLAYER']:30s} {player['POSITION']:5s} "
                  f"Height: {player['HEIGHT']:5s}  Weight: {player['WEIGHT']:3s}  "
                  f"Exp: {player['EXP']:2s}  Age: {player['AGE']}")
        
        return roster_df
        
    except Exception as e:
        print(f"Error fetching roster: {e}")
        return None


if __name__ == '__main__':
    # Example 1: Get a single team's roster (Lakers)
    print("Example 1: Get Lakers roster")
    print("=" * 80)
    get_single_team_roster('MIA')
    
    
