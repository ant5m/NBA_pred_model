# NBA Game Prediction Model

Neural network-based prediction system for NBA games using TensorFlow and ensemble learning methods.

## Features

- **Dual Model Architecture**: Single model and ensemble (3 models) prediction options
- **Multi-Output Predictions**: Game outcome probability and point totals for both teams
- **Temporal Validation**: Train on current season, validate on previous seasons
- **Fine-tuning Support**: Adapt models to recent game trends
- **Roster Integration**: Optional player-specific predictions accounting for injuries and lineups
- **Live Game Predictions**: Fetch and predict today's NBA games

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

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

## Quick Start

### 1. Build Databases

Build player and team statistics databases:

```bash
python3.11 build_player_stats_db.py --initial  # Takes 15-30 minutes
python3.11 build_team_stats_db.py --initial    # Takes 5-10 minutes
```

This creates:
- `nba_player_stats.db` - 530+ active players, seasons 2022-23 to 2025-26
- `nba_team_stats.db` - All 30 NBA teams, seasons 2022-23 to 2025-26

### 2. Train the Model

Train a single model:
```bash
python3.11 nba_model.py
```

Train an ensemble (3 models, better accuracy):
```bash
python3.11 nba_model.py --ensemble
```

Training details:
- Trains on 2024-25 season data
- Validates on 2022-23 and 2023-24 seasons
- Takes 10-20 minutes per model
- Saves to `nba_model_saved/` or `ensemble_model_saved/`

### 3. Predict Today's Games

Basic prediction:
```bash
python3.11 predict_today.py
```

Ensemble prediction (recommended):
```bash
python3.11 predict_today.py --ensemble
```

With live roster data (accounts for injuries):
```bash
python3.11 predict_today.py --ensemble --rosters
```

## Core Scripts

| Script | Purpose |
|--------|---------|
| `nba_model.py` | Main model: training, evaluation, prediction |
| `predict_today.py` | Predict today's NBA games |
| `finetune_model.py` | Fine-tune models on recent games |
| `build_team_stats_db.py` | Build/update team statistics database |
| `build_player_stats_db.py` | Build/update player statistics database |
| `fetch_historical_data.py` | Fetch historical NBA data |

## Model Architecture

### Neural Network Design
- **Input**: 77 features per game (team stats, player stats, home advantage)
- **Architecture**: Dense layers (256 -> 128 -> 64 -> 32) with batch normalization and dropout
- **Outputs**: 
  - Binary classification (win probability)
  - Regression (predicted points for both teams)

### Training Strategy
- **Train Set**: Current season (2024-25)
- **Test Set**: Previous seasons (2022-23, 2023-24)
- **Temporal Validation**: Prevents data leakage, realistic evaluation

### Ensemble Method
- 3 models with different random initializations
- Predictions averaged for improved stability
- Typically 1-9% accuracy improvement over single model

## Performance

Expected metrics (on historical data):
- **Outcome Accuracy**: 60-65% (predicting winner)
- **Points MAE**: 9-13 points per team
- **Ensemble Improvement**: +3-5% accuracy over single model

After fine-tuning on recent games:
- **Outcome Accuracy**: 65-68%
- **Points MAE**: 9-10 points per team

## Advanced Usage

### Fine-tuning Models

Fine-tune on recent games to adapt to current season trends:

```bash
# Fine-tune single model
python3.11 finetune_model.py --recent-games 50

# Fine-tune ensemble
python3.11 finetune_model.py --ensemble --recent-games 50
```

Fine-tuning options:
- `--recent-games N`: Use last N games (default: 50)
- `--epochs N`: Training epochs (default: 10)
- `--learning-rate X`: Learning rate (default: 0.0001)
- `--ensemble`: Fine-tune ensemble models

### Update Databases

Keep data current by updating daily:

```bash
python3.11 build_player_stats_db.py --update
python3.11 build_team_stats_db.py --update
```

Check database statistics:
```bash
python3.11 build_player_stats_db.py --stats
python3.11 build_team_stats_db.py --stats
```

### Database Schema

**nba_team_stats.db**
- `teams`: Team info (ID, name, abbreviation)
- `season_stats`: Season averages (win%, points, FG%, etc.)
- `game_logs`: Game-by-game results

**nba_player_stats.db**
- `players`: Player info (ID, name, active status)
- `season_stats`: Season averages (PPG, RPG, APG, etc.)
- `game_logs`: Game-by-game performance

## Prediction Workflow

### Daily Workflow
1. Update databases with latest games:
   ```bash
   python3.11 build_team_stats_db.py --update
   python3.11 build_player_stats_db.py --update
   ```

2. (Optional) Fine-tune model weekly:
   ```bash
   python3.11 finetune_model.py --ensemble --recent-games 50
   ```

3. Get today's predictions:
   ```bash
   python3.11 predict_today.py --ensemble --rosters
   ```

### Weekly Workflow
- Monday: Update databases, review previous week's accuracy
- Wednesday: Fine-tune model if needed
- Daily: Run predictions for upcoming games

## Command Line Options

### Training
```bash
python3.11 nba_model.py [--ensemble]
```
- `--ensemble`: Train ensemble of 3 models (recommended)

### Prediction
```bash
python3.11 predict_today.py [--ensemble] [--rosters] [--model PATH]
```
- `--ensemble`: Use ensemble model
- `--rosters`: Use live game rosters (accounts for injuries)
- `--model PATH`: Custom model path

### Fine-tuning
```bash
python3.11 finetune_model.py [--ensemble] [--recent-games N] [--epochs N] [--learning-rate X]
```
- `--ensemble`: Fine-tune ensemble models
- `--recent-games N`: Number of recent games to use
- `--epochs N`: Training epochs
- `--learning-rate X`: Learning rate (lower = more conservative)

## Troubleshooting

**ModuleNotFoundError: No module named 'tensorflow'**
```bash
pip install tensorflow
# For Apple Silicon (M1/M2):
pip install tensorflow-macos tensorflow-metal
```

**Database not found**
```bash
python3.11 build_team_stats_db.py --initial
python3.11 build_player_stats_db.py --initial
```

**Model not found**
```bash
python3.11 nba_model.py --ensemble
```

**Poor accuracy**
- Fine-tune on recent games
- Retrain with more recent data
- Use ensemble instead of single model

## Dependencies

Core packages:
- `tensorflow>=2.13.0` - Neural network framework
- `scikit-learn>=1.2.0` - Data preprocessing, metrics
- `nba_api>=1.1.0` - NBA statistics API
- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.23.0` - Numerical operations

See `requirements.txt` for complete list.

## Model Limitations

- Does not account for injuries (unless using `--rosters` flag)
- Does not factor in trades, coaching changes
- Does not consider back-to-back games or rest days
- Performance depends on quality of historical data
- Predictions should be one input among many factors

## License

This project uses the unofficial NBA API. Please use responsibly and respect API rate limits.
