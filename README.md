# NBA Game Prediction Model

Neural network-based prediction system for NBA games using TensorFlow and ensemble learning methods.

## Features

- **5-Model Ensemble**: Advanced ensemble with improved architecture (L2 regularization, Huber loss, gradient clipping)
- **Multi-Output Predictions**: Game outcome probability and point totals for both teams
- **Calibrated Probabilities**: Isotonic regression calibration for accurate win probabilities
- **Fine-tuning Support**: Adapt models to recent game trends with proper validation
- **Differential Features**: Home-away comparisons and momentum analysis
- **Live Predictions & Tracking**: Daily predictions with automatic logging and performance analysis
- **Current Performance**: 63% accuracy after fine-tuning and calibration

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
## Complete Setup Guide

### Step 1: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# For Apple Silicon (M1/M2/M3):
pip install tensorflow-macos tensorflow-metal
```

### Step 2: Build Databases

Fetch and build the team and player statistics databases:

```bash
# Build team stats database (takes 5-10 minutes)
python3.11 build_team_stats_db.py --initial

# Build player stats database (takes 15-30 minutes)
python3.11 build_player_stats_db.py --initial
```

This creates:
- `nba_team_stats.db` - All 30 NBA teams, seasons 2022-23 to 2025-26
- `nba_player_stats.db` - 530+ active players, seasons 2022-23 to 2025-26

**Verify databases:**
```bash
python3.11 build_team_stats_db.py --stats
python3.11 build_player_stats_db.py --stats
```

### Step 3: Train the Ensemble Model

Train a 5-model ensemble with improved architecture:

```bash
# Train ensemble (takes ~15 minutes)
python3.11 -c "from nba_model import train_ensemble_model; train_ensemble_model(n_models=5, epochs=50)"
```

This creates `ensemble_model_saved/` with 5 trained models.

**Expected output:**
- Training accuracy: 60-65%
- Validation accuracy: 55-60%
- Points MAE: 10-12 points

### Step 4: Fine-Tune on Recent Games
## Core Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `nba_model.py` | Main model code, training, prediction | Core module |
| `predict_today_calibrated.py` | Predict today's games (calibrated) | Daily |
| `log_predictions.py` | Save predictions to CSV | Daily |
| `update_prediction_logs.py` | Update logs with actual results | Daily |
| `analyze_performance.py` | Performance analysis and metrics | Daily/Weekly |
| `finetune_model.py` | Fine-tune on recent games | Weekly |
| `calibrate_model.py` | Calibrate win probabilities | Weekly/As needed |
## Model Architecture

### Neural Network Design
- **Input**: 87+ features per game (team stats, player stats, differential features, momentum)
- **Architecture**: 7-layer deep network (512 → 384 → 256 → 192 → 128 → 96 → 64)
  - Batch normalization after each layer
  - L2 regularization (0.001) to prevent overfitting
  - Reduced dropout (0.1-0.2) for better gradient flow
  - Gradient clipping (clipnorm=1.0) for training stability
- **Outputs**: 
  - Binary classification (home win probability)
  - Regression (predicted points for home and away teams)
- **Loss Functions**:
  - Binary crossentropy for classification
  - Huber loss for regression (robust to outliers)

### Key Features
- **Differential Features**: Home-away comparisons (win %, points, shooting %, rebounds, etc.)
- **Momentum Features**: Recent performance trends vs season averages
## Performance

### Current Performance (Dec 2025)
- **Overall Accuracy**: 63.0% (17/27 games with results)
- **Recent Performance**:
  - Nov 28: 72.7% (8/11) 
  - Dec 3: 44.4% (4/9)
  - Dec 7: 71.4% (5/7) ⭐
- **Points MAE**: ~10 points per team
- **Home Bias**: +5.4% (well-calibrated)
- **Brier Score**: 0.218 (excellent calibration)

### After Fine-Tuning & Calibration
- **Outcome Accuracy**: 67.5% on validation set
- **Points MAE**: 10.1 points average
- **Individual Models**: 61-67% accuracy
- **Ensemble Boost**: +5-7% over single model
- **High-Confidence Picks (>70%)**: 85%+ accuracy

### Testing Metrics (500 game test)
Before improvements:
- Accuracy: 55%, F1: 0.0 (broken - predicting home 100%)
- Points MAE: 140+ points (completely wrong)

After improvements:
- Accuracy: 63-68%, F1: 0.65-0.71
- Points MAE: 10-11 points
- R² score: Positive (better than baseline)alizations
- Predictions averaged for improved stability and reduced variance
- Individual models: 61-67% accuracy
- Ensemble: 63-68% accuracy (5-7% improvement)
**Expected output:**
- Brier score: ~0.22 (lower is better)
- Home bias reduced to <5%

### Step 6: Make Daily Predictions

Predict today's games with calibrated probabilities:

```bash
# Generate calibrated predictions
python3.11 predict_today_calibrated.py --ensemble

# Log predictions to CSV
python3.11 log_predictions.py --ensemble
```

**Output:**
- Displays all today's games with win probabilities and predicted scores
- Saves to `prediction_logs/predictions_YYYY-MM-DD_ensemble.csv`

### Step 7: Track Performance

Update logs with actual results and analyze performance:

```bash
# Update predictions with live scores
python3.11 update_prediction_logs.py

# Analyze overall performance
python3.11 analyze_performance.py
```

**Output:**
- Overall accuracy across all predictions
- Performance by date
- Home bias analysis
- Recommendations for recalibration

## Daily Workflow

Use the automated workflow script:

```bash
# All-in-one: predict, log, update, and analyze
bash nba.sh predict    # Generate predictions for today
bash nba.sh log        # Save predictions to CSV
bash nba.sh update     # Update with live scores
bash nba.sh analyze    # Show performance stats
bash nba.sh visualize  # Generate performance graphs
```

Or run the complete daily workflow:

```bash
bash nba.sh all        # Run all steps in sequence
```

## Weekly Maintenance

### Update Databases (Weekly)

```bash
# Update team stats with latest games
python3.11 build_team_stats_db.py --update

# Update player stats
python3.11 build_player_stats_db.py --update
```

### Fine-Tune Model (Weekly)

```bash
# Fine-tune on latest 200 games
python3.11 finetune_model.py --ensemble --seasons 2025-26 --recent-games 200 --epochs 20
```

### Recalibrate (As Needed)

```bash
# Recalibrate if bias >5% or Brier score increases
python3.11 calibrate_model.py --ensemble
```hon3.11 predict_today.py --ensemble --rosters
```

## Core Scripts

| Script | Purpose |
## Advanced Usage

### Comprehensive Model Testing

Test model performance with detailed analytics:

```bash
# Test ensemble on 500 games
python3.11 test_model_analytics.py --ensemble --model ensemble_model_saved --test-size 500
```

**Metrics provided:**
- Classification: Accuracy, Precision, Recall, F1, AUC-ROC
- Regression: MAE, MSE, RMSE, R² for point predictions
- Probability: Brier score, Log loss
- Bias: Home team prediction bias
- Confusion matrix and detailed breakdowns

### Fine-Tuning Options

```bash
# Fine-tune with custom settings
python3.11 finetune_model.py --ensemble \
  --seasons 2025-26 \
  --recent-games 200 \
  --epochs 20 \
  --learning-rate 0.0001
```

**Options:**
- `--ensemble`: Fine-tune ensemble (recommended)
- `--seasons`: Comma-separated seasons (e.g., "2024-25,2025-26")
- `--recent-games N`: Number of most recent games (default: 50, recommended: 200)
- `--epochs N`: Training epochs (default: 10, recommended: 20)
- `--learning-rate X`: Learning rate (default: 0.0001, keep low to avoid overfitting)

### Calibration Options

```bash
# Calibrate using validation games
python3.11 calibrate_model.py --ensemble

# Calibrate using existing prediction logs (recommended after collecting predictions)
python3.11 calibrate_model.py --ensemble --from-logs
```

The calibration reduces systematic bias and improves probability accuracy.
## Complete Prediction Workflow

### Daily Workflow (Automated)

Use the `nba.sh` script for streamlined daily operations:

```bash
# Morning: Generate and log predictions
bash nba.sh all

# Or run steps individually:
bash nba.sh predict    # Show today's predictions
bash nba.sh log        # Save to CSV
bash nba.sh update     # Update with live scores (run after games)
bash nba.sh analyze    # View performance stats
```

### Daily Workflow (Manual)

**Morning - Before Games:**
```bash
# 1. Generate calibrated predictions
python3.11 predict_today_calibrated.py --ensemble

## Command Line Reference

### Predictions
```bash
# Calibrated predictions (recommended)
python3.11 predict_today_calibrated.py --ensemble

# Log predictions to CSV
python3.11 log_predictions.py --ensemble

# Update logs with actual results
python3.11 update_prediction_logs.py

# Analyze performance
python3.11 analyze_performance.py
```

### Training & Fine-Tuning
```bash
# Train ensemble from scratch
python3.11 -c "from nba_model import train_ensemble_model; train_ensemble_model(n_models=5, epochs=50)"

# Fine-tune ensemble
python3.11 finetune_model.py --ensemble --seasons 2025-26 --recent-games 200 --epochs 20 --learning-rate 0.0001

## Troubleshooting

### Installation Issues

**ModuleNotFoundError: No module named 'tensorflow'**
```bash
# Standard installation
pip install tensorflow

# For Apple Silicon (M1/M2/M3)
pip install tensorflow-macos tensorflow-metal

# For specific version
pip install tensorflow==2.13.0
```

**ImportError: No module named 'nba_api'**
```bash
pip install -r requirements.txt
```

### Database Issues

**Error: Database not found**
```bash
# Build databases from scratch
python3.11 build_team_stats_db.py --initial
python3.11 build_player_stats_db.py --initial
```

**Error: No data for season 2025-26**
```bash
# Update databases to fetch latest season
python3.11 build_team_stats_db.py --update
python3.11 build_player_stats_db.py --update
```

**Database corruption**
```bash
# Delete and rebuild
rm nba_team_stats.db nba_player_stats.db
python3.11 build_team_stats_db.py --initial
## Project Structure

```
NBA_pred_model/
├── nba_model.py                    # Core model code
├── predict_today_calibrated.py     # Daily predictions with calibration
├── log_predictions.py              # Save predictions to CSV
├── update_prediction_logs.py       # Update with actual results
├── analyze_performance.py          # Performance tracking
├── visualize_model.py             # Generate performance graphs
├── finetune_model.py              # Fine-tune on recent games
├── calibrate_model.py             # Probability calibration
├── test_model_analytics.py        # Comprehensive testing
├── build_team_stats_db.py         # Team data management
├── build_player_stats_db.py       # Player data management
├── fetch_historical_data.py       # Historical data fetching
├── nba.sh                         # Automated workflow script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── QUICKSTART.md                  # Quick reference guide
├── VISUALIZATIONS_GUIDE.md        # Visualization documentation
├── WORKFLOW_GUIDE.md              # Detailed workflows
├── ensemble_model_saved/          # Trained ensemble models
│   ├── model_0/ ... model_4/      # 5 individual models
│   └── ensemble_metadata.pkl      # Feature columns
├── calibration_saved/             # Calibration models
│   └── calibration.pkl            # Isotonic calibrator
├── visualizations/                # Performance graphs
│   ├── confusion_matrix.png       # Confusion matrix heatmap
│   ├── roc_curve.png              # ROC curve with AUC
│   ├── calibration_curve.png      # Calibration plot
│   └── *.png                      # Other visualizations
├── prediction_logs/               # Prediction history
│   └── predictions_*.csv          # Daily prediction logs
├── nba_team_stats.db             # Team statistics database
├── nba_player_stats.db           # Player statistics database
└── __pycache__/                  # Python cache (auto-generated)
```

## Model Limitations

### What the Model Does NOT Consider
- **Injuries**: Unless live roster data is used, injuries are not factored in
- **Trades & Roster Changes**: Recent trades may not be reflected in historical stats
- **Coaching Changes**: New coaches and their impact are not modeled
- **Rest & Fatigue**: Back-to-back games, travel, and rest days are not considered
- **Playoff Context**: Model trained on regular season data
- **Intangibles**: Team chemistry, motivation, playoff intensity

### Known Issues
- **Data Lag**: Databases need manual updates to include latest games
- **Small Sample Size**: Early season predictions less reliable (fewer games)
- **API Dependency**: Requires NBA API to be operational
- **Outliers**: Blowouts and unusual games can affect point predictions

### Best Practices
- Use predictions as **one factor** among many
- **Higher confidence (>70%)** predictions are more reliable
- **Combine with other analysis**: injuries, matchups, trends
- **Fine-tune weekly** for best current-season performance
- **Monitor accuracy** and recalibrate if bias increases
# Train ensemble model
python3.11 -c "from nba_model import train_ensemble_model; train_ensemble_model(n_models=5, epochs=50)"
```

**Error: Calibration file not found**
```bash
# Calibrate model
python3.11 calibrate_model.py --ensemble
```

**Poor accuracy (<55%)**
1. Fine-tune on recent games:
   ```bash
   python3.11 finetune_model.py --ensemble --recent-games 200 --epochs 20
   ```

2. Recalibrate:
   ```bash
   python3.11 calibrate_model.py --ensemble
   ```

3. If still poor, retrain from scratch:
   ```bash
   python3.11 -c "from nba_model import train_ensemble_model; train_ensemble_model(n_models=5, epochs=50)"
   ```

**Model predicting home team too often (high bias)**
```bash
# Recalibrate to reduce bias
python3.11 calibrate_model.py --ensemble --from-logs
```

**Test showing F1 score = 0**
- This indicates the old model before improvements
- Retrain using the fixed architecture (see Model Issues above)

### API Issues

**Error: Too many requests / API rate limit**
- The NBA API has rate limits
- Wait a few minutes between large data fetches
- Use `--update` instead of `--initial` when possible

**Error: No games found for today**
- Check if it's an off-day (no games scheduled)
- Verify internet connection
- Try again in a few minutes (API might be temporarily down)

### Performance Issues

**Training is very slow**
- Check if using GPU acceleration (TensorFlow should auto-detect)
- Reduce batch size if running out of memory
- For Apple Silicon, ensure tensorflow-metal is installed

**Predictions taking too long**
- Model loading is slow on first prediction (normal)
- Subsequent predictions should be fast (<1 second)
- Consider using single model instead of ensemble for speed

### Getting Help

If issues persist:
1. Check that all databases are built: `ls -lh *.db`
2. Verify model exists: `ls -lh ensemble_model_saved/`
3. Check Python version: `python3.11 --version` (need 3.8+)
4. Review error messages carefully - they usually indicate the specific probleme model
  --model PATH            Path to model directory
  --test-size N           Number of test samples (default: 500)
```

### Database Management
```bash
# Build databases initially
python3.11 build_team_stats_db.py --initial
python3.11 build_player_stats_db.py --initial

# Update databases with latest games
python3.11 build_team_stats_db.py --update
python3.11 build_player_stats_db.py --update

# Check database statistics
python3.11 build_team_stats_db.py --stats
python3.11 build_player_stats_db.py --stats
```

### Workflow Script
```bash
# Run all daily tasks
bash nba.sh all

# Or run individual steps
bash nba.sh predict    # Generate predictions
bash nba.sh log        # Save to CSV
bash nba.sh update     # Update with results
bash nba.sh analyze    # Show performance
```

**Wednesday - Fine-Tune (if needed):**
```bash
# Fine-tune if accuracy drops below 60% or new season starts
python3.11 finetune_model.py --ensemble --seasons 2025-26 --recent-games 200 --epochs 20

# Recalibrate after fine-tuning
python3.11 calibrate_model.py --ensemble
```

### Monthly Workflow

**Model Maintenance:**
```bash
# Test model performance comprehensively
python3.11 test_model_analytics.py --ensemble --test-size 500

# Recalibrate using prediction logs
python3.11 calibrate_model.py --ensemble --from-logs

# If accuracy significantly drops, retrain from scratch
python3.11 -c "from nba_model import train_ensemble_model; train_ensemble_model(n_models=5, epochs=50)"
```

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
