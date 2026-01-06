"""Box score data retrieval for NBA games."""

from typing import Dict, List, Optional
from nba_api.stats.endpoints import boxscoretraditionalv3

def get_box_score(game_id: str) -> Optional[Dict]:
    """Fetch box score data for a specific game.
    
    Args:
        game_id: NBA game ID (format: '0022500123')
        
    Returns:
        Dictionary containing team stats and player stats, or None if unavailable
    """
    try:
        boxscore = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
        
        # Get team stats
        team_stats_df = boxscore.team_stats.get_data_frame()
        # Get player stats
        player_stats_df = boxscore.player_stats.get_data_frame()
        
        if team_stats_df.empty:
            return None
            
        # Format team stats
        teams = []
        for _, row in team_stats_df.iterrows():
            teams.append({
                'team_id': int(row['teamId']),
                'team_name': row['teamName'],
                'team_abbreviation': row['teamTricode'],
                'points': int(row['points']),
                'field_goals_made': int(row['fieldGoalsMade']),
                'field_goals_attempted': int(row['fieldGoalsAttempted']),
                'field_goal_pct': float(row['fieldGoalsPercentage']),
                'three_pointers_made': int(row['threePointersMade']),
                'three_pointers_attempted': int(row['threePointersAttempted']),
                'three_point_pct': float(row['threePointersPercentage']),
                'free_throws_made': int(row['freeThrowsMade']),
                'free_throws_attempted': int(row['freeThrowsAttempted']),
                'free_throw_pct': float(row['freeThrowsPercentage']),
                'rebounds_offensive': int(row['reboundsOffensive']),
                'rebounds_defensive': int(row['reboundsDefensive']),
                'rebounds_total': int(row['reboundsTotal']),
                'assists': int(row['assists']),
                'steals': int(row['steals']),
                'blocks': int(row['blocks']),
                'turnovers': int(row['turnovers']),
                'fouls_personal': int(row['foulsPersonal']),
            })
        
        # Format player stats
        players = []
        for _, row in player_stats_df.iterrows():
            if row['minutes'] is None or row['minutes'] == '' or row['minutes'] == 'None':
                continue  # Skip players who didn't play
                
            players.append({
                'player_id': int(row['personId']),
                'player_name': row['nameI'],
                'team_abbreviation': row['teamTricode'],
                'position': row['position'] if row['position'] else '',
                'minutes': str(row['minutes']),
                'points': int(row['points']),
                'field_goals_made': int(row['fieldGoalsMade']),
                'field_goals_attempted': int(row['fieldGoalsAttempted']),
                'field_goal_pct': float(row['fieldGoalsPercentage']) if row['fieldGoalsPercentage'] else 0.0,
                'three_pointers_made': int(row['threePointersMade']),
                'three_pointers_attempted': int(row['threePointersAttempted']),
                'three_point_pct': float(row['threePointersPercentage']) if row['threePointersPercentage'] else 0.0,
                'free_throws_made': int(row['freeThrowsMade']),
                'free_throws_attempted': int(row['freeThrowsAttempted']),
                'free_throw_pct': float(row['freeThrowsPercentage']) if row['freeThrowsPercentage'] else 0.0,
                'rebounds_offensive': int(row['reboundsOffensive']),
                'rebounds_defensive': int(row['reboundsDefensive']),
                'rebounds_total': int(row['reboundsTotal']),
                'assists': int(row['assists']),
                'steals': int(row['steals']),
                'blocks': int(row['blocks']),
                'turnovers': int(row['turnovers']),
                'fouls_personal': int(row['foulsPersonal']),
                'plus_minus': int(row['plusMinusPoints']) if row['plusMinusPoints'] else 0,
            })
        
        return {
            'game_id': game_id,
            'teams': teams,
            'players': players
        }
        
    except Exception as e:
        print(f"Error fetching box score for game {game_id}: {e}")
        return None
