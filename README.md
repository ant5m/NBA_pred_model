# NBA Prediction Model

A comprehensive NBA data collection and analysis toolkit for building prediction models.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**

   ```bash
   cd NBA_pred_model
   ```
2. **Set up virtual environment (recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or on Windows: venv\Scripts\activate
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## 📊 Database Setup

### Player Stats Database

Build the player stats database with the last 3-4 seasons:

```bash
python3 build_player_stats_db.py --initial
```

This will:

- Create `nba_player_stats.db` with 530+ active players
- Load season stats for 2022-23, 2023-24, 2024-25, 2025-26
- Fetch game-by-game logs for current season (2025-26)
- **Takes 15-30 minutes on first run**

### Team Stats Database

Build the team stats database with the same seasons:

```bash
python3 build_team_stats_db.py --initial
```

This will:

- Create `nba_team_stats.db` with all 30 NBA teams
- Load season stats for 2022-23, 2023-24, 2024-25, 2025-26
- Fetch game-by-game logs for current season (2025-26)
- **Takes 5-10 minutes on first run**

### Daily Updates

Update both databases with latest games as the season progresses:

```bash
python3 build_player_stats_db.py --update
python3 build_team_stats_db.py --update
```

### Check Database Stats

```bash
python3 build_player_stats_db.py --stats
python3 build_team_stats_db.py --stats
```

## 📁 Available Databases

### 1. Player Stats (`nba_player_stats.db`)

Player stats database with recent seasons:

- Season averages (2022-23 to 2025-26)
- Game-by-game logs for current season
- 530+ active players
- Easy to update as games are played

### 2. Team Stats (`nba_team_stats.db`)

Team stats database with recent seasons:

- Season averages for all 30 NBA teams (2022-23 to 2025-26)
- Game-by-game logs for current season
- Win/loss records, offensive/defensive stats
- Easy to update as games are played

## 🔍 Data Exploration

### View Database Info

```bash
python3 main.py
```

Shows available tables, sample queries, and key parameters.

### Query Player Stats

```bash
python3 query_player_stats.py      # See example queries
python3 print_player_sample.py     # View player stats across seasons
```

### Query Team Stats

```bash
python3 query_team_stats.py        # See example queries and standings
```

## 📡 Live Data & Rosters

### Today's Games & Scores

```bash
python3 nba_live.py
```

### Team Rosters

```bash
python3 team_rosters.py            # Current rosters for all teams
```

### Game-Day Rosters (Active/Inactive)

```bash
python3 game_rosters.py            # Who's playing vs. on bench
python3 game_roster_example.py     # See example from recent game
```

## 🛠️ Key Scripts

| Script                       | Purpose                                |
| ---------------------------- | -------------------------------------- |
| `build_player_stats_db.py` | Build/update player stats database     |
| `build_team_stats_db.py`   | Build/update team stats database       |
| `query_player_stats.py`    | Query player stats (examples included) |
| `query_team_stats.py`      | Query team stats (examples included)   |
| `main.py`                  | Database info and query examples       |
| `nba_live.py`              | Today's live scores                    |
| `team_rosters.py`          | Get team rosters                       |
| `game_rosters.py`          | Game-day active rosters                |

## 💾 Database Tables

### `nba_player_stats.db`

**players**

- player_id, full_name, is_active

**season_stats**

- Season averages: PPG, RPG, APG, FG%, 3P%, etc.
- Covers 2022-23 to 2025-26

**game_logs**

- Game-by-game stats for current season
- Minutes, points, rebounds, assists, +/-, etc.

### `nba_team_stats.db`

**teams**

- team_id, full_name, abbreviation, nickname, city, state, year_founded

**season_stats**

- Season averages: W, L, Win%, PPG, RPG, APG, FG%, 3P%, etc.
- Covers 2022-23 to 2025-26

**game_logs**

- Game-by-game stats for current season
- Game date, matchup, W/L, points, rebounds, assists, +/-, etc.

## 🔄 Updating Data

### Update Player Stats

```bash
python3 build_player_stats_db.py --update
```

### Update Team Stats

```bash
python3 build_team_stats_db.py --update
```

### Update Both Databases

```bash
python3 build_player_stats_db.py --update && python3 build_team_stats_db.py --update
```

## 📝 Common Workflows

### 1. Daily Prediction Model Update

```bash
# Morning: Get latest stats
python3 build_player_stats_db.py --update
python3 build_team_stats_db.py --update

# Check today's games
python3 nba_live.py

# Get active rosters for today's games
python3 game_rosters.py
```

### 2. Team Analysis

```bash
# Get team stats and standings
python3 query_team_stats.py

# Get team roster
python3 team_rosters.py

# Get individual player stats
python3 query_player_stats.py
```

### 3. Compare Teams

```python
from query_team_stats import *

# Current standings
get_standings('2025-26')

# Compare specific teams
compare_teams(['LAL', 'GSW', 'BOS'], '2025-26')

# Top offensive teams
get_top_offensive_teams('2025-26')
```

## 📖 Example Queries

### Player Queries

```python
from query_player_stats import *

# Get player's season stats
get_player_season_stats('LeBron James')

# Last 10 games
get_recent_games('LeBron James', 10)

# Compare players
compare_players(['LeBron', 'Curry', 'Durant'], '2025-26')
```

### Team Queries

```python
from query_team_stats import *

# Get team's season stats
get_team_season_stats('Lakers')

# Last 10 games
get_recent_games('LAL', 10)

# Current standings
get_standings('2025-26')

# Compare teams
compare_teams(['LAL', 'GSW', 'BOS'], '2025-26')
```

## 🐛 Troubleshooting

**Database not found:**

```bash
python3 build_player_stats_db.py --initial  # Build player stats DB
python3 build_team_stats_db.py --initial    # Build team stats DB
```

## 📚 Dependencies

See `requirements.txt` for full list. Main packages:

- `nba-api` - NBA stats and live data
- `pandas` - Data manipulation
- `sqlite3` - Database (built-in)

1. **Build your databases:**
   ```bash
   python3 build_player_stats_db.py --initial
   python3 build_team_stats_db.py --initial
   ```
2. **Explore the data:**
   ```bash
   python3 query_player_stats.py
   python3 query_team_stats.py
   ```
3. **Check today's games:**
   ```bash
   python3 nba_live.py
   ```
4. 📄 License

This project uses the unofficial NBA API. Please use responsibly and respect rate limits.
