"""Compact NBA game prediction model.
Minimal, focused implementation of the NBAGamePredictor class.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Only show errors, not warnings
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import pickle

# Database paths
TEAM_DB = 'nba_team_stats.db'
PLAYER_DB = 'nba_player_stats.db'


class NBAGamePredictor:
    """Neural network for outcome and score prediction."""
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def get_team_features(self, team_id, seasons=['2022-23', '2023-24', '2024-25', '2025-26'], as_of_date=None):
        """Return season + recent averages for a team."""
        conn = sqlite3.connect(TEAM_DB)
        
        # Handle single season string or list of seasons
        if isinstance(seasons, str):
            seasons = [seasons]
        
        # Get season averages (most recent season with data)
        placeholders = ','.join(['?'] * len(seasons))
        query = f"""
            SELECT 
                win_pct, points, field_goal_pct, three_point_pct, free_throw_pct,
                total_rebounds, assists, steals, blocks, turnovers, personal_fouls,
                offensive_rebounds, defensive_rebounds, plus_minus
            FROM season_stats
            WHERE team_id = ? AND season IN ({placeholders})
            ORDER BY season DESC
            LIMIT 1
        """
        season_df = pd.read_sql_query(query, conn, params=(team_id, *seasons))
        
        # Get recent game performance (last 10 games across all seasons)
        games_query = f"""
            SELECT 
                points, field_goal_pct, three_point_pct, free_throw_pct,
                total_rebounds, assists, steals, blocks, turnovers,
                plus_minus, win_loss
            FROM game_logs
            WHERE team_id = ? AND season IN ({placeholders})
            ORDER BY game_date DESC
            LIMIT 10
        """
        games_df = pd.read_sql_query(games_query, conn, params=(team_id, *seasons))
        conn.close()
        
        features = {}
        
        # Season stats
        if not season_df.empty:
            for col in season_df.columns:
                features[f'team_season_{col}'] = season_df[col].values[0]
        else:
            # Fill with defaults if no season stats
            for col in ['win_pct', 'points', 'field_goal_pct', 'three_point_pct', 
                       'free_throw_pct', 'total_rebounds', 'assists', 'steals', 
                       'blocks', 'turnovers', 'personal_fouls', 'offensive_rebounds',
                       'defensive_rebounds', 'plus_minus']:
                features[f'team_season_{col}'] = 0.0
        
        # Recent performance (last 10 games averages) and momentum (last 5 games)
        if not games_df.empty:
            # Win percentage in last 10
            features['team_recent_win_pct'] = (games_df['win_loss'] == 'W').mean()

            # Averages from recent games (last 10)
            for col in ['points', 'field_goal_pct', 'three_point_pct', 'free_throw_pct',
                       'total_rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'plus_minus']:
                features[f'team_recent_{col}'] = games_df[col].mean()

            # Momentum: last 5 games (more responsive indicator)
            last5 = games_df.head(5)
            features['team_momentum_win_pct_5'] = (last5['win_loss'] == 'W').mean()
            # Use plus_minus as point differential per game if available
            if 'plus_minus' in last5.columns:
                features['team_momentum_point_diff_5'] = last5['plus_minus'].mean()
            else:
                # Fallback to points - (opponent points) not available here; use recent points as proxy
                features['team_momentum_point_diff_5'] = last5['points'].mean() - games_df['points'].mean()
            
            # Volatility features (help predict blowouts vs close games)
            features['team_score_volatility'] = games_df['points'].std() if len(games_df) > 1 else 0.0
            features['team_margin_volatility'] = games_df['plus_minus'].std() if len(games_df) > 1 else 0.0
            features['team_blowout_rate'] = (games_df['plus_minus'].abs() > 15).mean()
        else:
            # Fill with defaults
            features['team_recent_win_pct'] = 0.5
            for col in ['points', 'field_goal_pct', 'three_point_pct', 'free_throw_pct',
                       'total_rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'plus_minus']:
                features[f'team_recent_{col}'] = 0.0
            # Default momentum
            features['team_momentum_win_pct_5'] = 0.5
            features['team_momentum_point_diff_5'] = 0.0
            # Default volatility
            features['team_score_volatility'] = 0.0
            features['team_margin_volatility'] = 0.0
            features['team_blowout_rate'] = 0.0
        
        return features
    
    def get_top_players_features(self, team_abbr, seasons=['2022-23', '2023-24', '2024-25', '2025-26'], n_players=5):
        """Return aggregated stats for top N players."""
        conn = sqlite3.connect(PLAYER_DB)
        
        # Handle single season string or list of seasons
        if isinstance(seasons, str):
            seasons = [seasons]
        
        # Get top players by minutes played (most recent season with data)
        placeholders = ','.join(['?'] * len(seasons))
        query = f"""
            SELECT 
                points, assists, total_rebounds, steals, blocks, turnovers,
                field_goal_pct, three_point_pct, free_throw_pct, minutes_played
            FROM season_stats
            WHERE team_abbreviation = ? AND season IN ({placeholders})
            ORDER BY season DESC, minutes_played DESC
            LIMIT ?
        """
        players_df = pd.read_sql_query(query, conn, params=(team_abbr, *seasons, n_players))
        conn.close()
        
        features = {}
        
        if not players_df.empty:
            # Aggregate top players' stats
            features['top_players_avg_points'] = players_df['points'].mean()
            features['top_players_avg_assists'] = players_df['assists'].mean()
            features['top_players_avg_rebounds'] = players_df['total_rebounds'].mean()
            features['top_players_avg_steals'] = players_df['steals'].mean()
            features['top_players_avg_blocks'] = players_df['blocks'].mean()
            features['top_players_avg_turnovers'] = players_df['turnovers'].mean()
            features['top_players_avg_fg_pct'] = players_df['field_goal_pct'].mean()
            features['top_players_avg_3pt_pct'] = players_df['three_point_pct'].mean()
            features['top_players_avg_ft_pct'] = players_df['free_throw_pct'].mean()
            features['top_players_total_minutes'] = players_df['minutes_played'].sum()
            
            # Best player stats
            features['best_player_points'] = players_df['points'].iloc[0]
            features['best_player_assists'] = players_df['assists'].iloc[0]
            features['best_player_rebounds'] = players_df['total_rebounds'].iloc[0]
        else:
            # Fill with defaults
            for stat in ['points', 'assists', 'rebounds', 'steals', 'blocks', 
                        'turnovers', 'fg_pct', '3pt_pct', 'ft_pct']:
                features[f'top_players_avg_{stat}'] = 0.0
            features['top_players_total_minutes'] = 0.0
            for stat in ['points', 'assists', 'rebounds']:
                features[f'best_player_{stat}'] = 0.0
        
        return features
    
    def get_team_abbreviation(self, team_id):
        """Lookup team abbreviation from `teams` table."""
        conn = sqlite3.connect(TEAM_DB)
        query = "SELECT abbreviation FROM teams WHERE team_id = ?"
        result = pd.read_sql_query(query, conn, params=(team_id,))
        conn.close()
        
        if not result.empty:
            return result['abbreviation'].values[0]
        return None
    
    def prepare_game_features(self, home_team_id, away_team_id, seasons=['2022-23', '2023-24', '2024-25', '2025-26']):
        """Combine team + player features for a matchup."""
        # Get team abbreviations
        home_abbr = self.get_team_abbreviation(home_team_id)
        away_abbr = self.get_team_abbreviation(away_team_id)
        
        # Home team features
        home_team_features = self.get_team_features(home_team_id, seasons)
        home_player_features = self.get_top_players_features(home_abbr, seasons)
        
        # Away team features
        away_team_features = self.get_team_features(away_team_id, seasons)
        away_player_features = self.get_top_players_features(away_abbr, seasons)
        
        # Combine all features with prefixes
        features = {}
        for k, v in home_team_features.items():
            features[f'home_{k}'] = v
        for k, v in home_player_features.items():
            features[f'home_{k}'] = v
        for k, v in away_team_features.items():
            features[f'away_{k}'] = v
        for k, v in away_player_features.items():
            features[f'away_{k}'] = v
        
        # Add differential features (home - away comparisons)
        key_stats = ['win_pct', 'points', 'field_goal_pct', 'three_point_pct', 
                     'assists', 'total_rebounds', 'plus_minus', 'recent_win_pct',
                     'top_player_avg_points', 'top_player_avg_assists', 'top_player_avg_rebounds']
        
        for stat in key_stats:
            home_key = f'home_{stat}'
            away_key = f'away_{stat}'
            if home_key in features and away_key in features:
                home_val = features[home_key] if features[home_key] is not None else 0
                away_val = features[away_key] if features[away_key] is not None else 0
                features[f'{stat}_diff'] = home_val - away_val
        
        # Add momentum features (recent performance trend)
        if 'home_recent_win_pct' in features and 'home_win_pct' in features:
            home_recent = features['home_recent_win_pct'] if features['home_recent_win_pct'] is not None else 0
            home_overall = features['home_win_pct'] if features['home_win_pct'] is not None else 0
            features['home_momentum'] = home_recent - home_overall
        
        if 'away_recent_win_pct' in features and 'away_win_pct' in features:
            away_recent = features['away_recent_win_pct'] if features['away_recent_win_pct'] is not None else 0
            away_overall = features['away_win_pct'] if features['away_win_pct'] is not None else 0
            features['away_momentum'] = away_recent - away_overall
        
        # Momentum differential
        if 'home_momentum' in features and 'away_momentum' in features:
            features['momentum_diff'] = features['home_momentum'] - features['away_momentum']
        
        # Add home court advantage feature
        features['home_court_advantage'] = 1.0
        
        return features
    
    def create_training_data(self, seasons=['2022-23', '2023-24', '2024-25', '2025-26'], min_games=10):
        """Build X, y arrays from `game_logs` for a season."""
        conn = sqlite3.connect(TEAM_DB)
        
        # Handle single season string or list of seasons
        if isinstance(seasons, str):
            seasons = [seasons]
        
        # Define temporal weights for each season (more recent = higher weight)
        season_weights = {
            '2025-26': 3.0,
            '2024-25': 2.5,
            '2023-24': 1.5,
            '2022-23': 1.0
        }
        
        # Get all games with both teams' data
        placeholders = ','.join(['?'] * len(seasons))
        query = f"""
            SELECT DISTINCT
                g1.game_id,
                g1.team_id as home_team_id,
                g2.team_id as away_team_id,
                g1.win_loss as home_win,
                g1.points as home_points,
                g2.points as away_points,
                g1.game_date,
                g1.season
            FROM game_logs g1
            JOIN game_logs g2 ON g1.game_id = g2.game_id AND g1.team_id != g2.team_id
            WHERE g1.matchup LIKE '%vs%'
                AND g1.season IN ({placeholders})
            ORDER BY g1.game_date
        """
        games_df = pd.read_sql_query(query, conn, params=seasons)
        conn.close()
        
        print(f"Found {len(games_df)} games for training")
        
        X_data = []
        y_outcome = []
        y_home_points = []
        y_away_points = []
        sample_weights = []
        
        for idx, game in games_df.iterrows():
            if idx % 100 == 0:
                print(f"Processing game {idx}/{len(games_df)}...")
            
            try:
                features = self.prepare_game_features(
                    game['home_team_id'],
                    game['away_team_id'],
                    seasons
                )
                
                X_data.append(features)
                
                # Outcome: 1 if home team wins, 0 if away team wins
                y_outcome.append(1 if game['home_win'] == 'W' else 0)
                
                # Points scored
                y_home_points.append(game['home_points'])
                y_away_points.append(game['away_points'])
                
                # Add sample weight based on season recency
                weight = season_weights.get(game['season'], 1.0)
                sample_weights.append(weight)
                
            except Exception as e:
                print(f"Error processing game {game['game_id']}: {e}")
                continue
        
        # Convert to DataFrame then numpy array
        X_df = pd.DataFrame(X_data)
        
        # Store feature columns for later use
        self.feature_columns = X_df.columns.tolist()
        
        # Fill any missing values
        X_df = X_df.fillna(0)
        
        X = X_df.values
        y_outcome = np.array(y_outcome)
        y_points = np.column_stack([y_home_points, y_away_points])
        sample_weights = np.array(sample_weights)
        
        # Print weight distribution
        print(f"Created dataset with {X.shape[0]} samples and {X.shape[1]} features")
        for season in seasons:
            season_count = np.sum(games_df['season'] == season)
            weight = season_weights.get(season, 1.0)
            print(f"  {season}: {season_count} games (weight: {weight}x)")
        
        return X, y_outcome, y_points, sample_weights
    
    def build_model(self, input_dim):
        """Create Keras multi-output model with deep architecture, regularization, and residual connections."""
        # Input layer with normalization
        inputs = layers.Input(shape=(input_dim,))
        x = layers.BatchNormalization()(inputs)
        
        # Deep shared layers with L2 regularization and reduced dropout
        x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(384, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(192, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.15)(x)
        
        x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.15)(x)
        
        x = layers.Dense(96, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        # Branch 1: Game outcome prediction (binary classification)
        outcome_branch = layers.Dense(48, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        outcome_branch = layers.BatchNormalization()(outcome_branch)
        outcome_branch = layers.Dropout(0.1)(outcome_branch)
        outcome_branch = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(outcome_branch)
        outcome_branch = layers.Dropout(0.1)(outcome_branch)
        outcome_output = layers.Dense(1, activation='sigmoid', name='outcome')(outcome_branch)
        
        # Branch 2: Points prediction (regression)
        points_branch = layers.Dense(48, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        points_branch = layers.BatchNormalization()(points_branch)
        points_branch = layers.Dropout(0.1)(points_branch)
        points_branch = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(points_branch)
        points_branch = layers.Dropout(0.1)(points_branch)
        points_output = layers.Dense(2, activation='linear', name='points')(points_branch)
        
        # Create model with multiple outputs
        model = models.Model(
            inputs=inputs,
            outputs=[outcome_output, points_output]
        )
        
        # Compile with Huber loss for robustness and gradient clipping
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
            loss={
                'outcome': 'binary_crossentropy',
                'points': 'huber'
            },
            loss_weights={
                'outcome': 1.0,
                'points': 0.3
            },
            metrics={
                'outcome': ['accuracy'],
                'points': ['mae']
            }
        )
        
        return model
    
    def train(self, X, y_outcome, y_points, sample_weights=None, epochs=50, batch_size=32, validation_split=0.2):
        """Fit model on training data with optional sample weighting."""
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Build model
        self.model = self.build_model(X.shape[1])
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=0.00001
        )
        
        # Handle sample weights by duplicating samples (workaround for Keras multi-output bug)
        if sample_weights is not None:
            # Normalize weights to integers by scaling
            weights_scaled = (sample_weights / sample_weights.min()).astype(int)
            
            # Duplicate samples based on weights
            indices = []
            for i, weight in enumerate(weights_scaled):
                indices.extend([i] * weight)
            
            indices = np.array(indices)
            X_scaled = X_scaled[indices]
            y_outcome = y_outcome[indices]
            y_points = y_points[indices]
            
            print(f"Applied temporal weighting: {len(X_scaled)} effective samples (from {len(sample_weights)} original)")
        
        # Train model with standard validation split
        history = self.model.fit(
            X_scaled,
            {'outcome': y_outcome, 'points': y_points},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def fine_tune(self, X, y_outcome, y_points, epochs=10, batch_size=32, learning_rate=0.0001, validation_split=0.15):
        """Fine-tune existing model on new data with proper validation and weight preservation."""
        if self.model is None:
            raise ValueError("No model to fine-tune")
        
        # Split data BEFORE scaling to prevent leakage
        if validation_split > 0:
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_out_train, y_out_val = y_outcome[:split_idx], y_outcome[split_idx:]
            y_pts_train, y_pts_val = y_points[:split_idx], y_points[split_idx:]
            
            X_train_scaled = self.scaler.transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, {'outcome': y_out_val, 'points': y_pts_val})
        else:
            X_train_scaled = self.scaler.transform(X)
            y_out_train, y_pts_train = y_outcome, y_points
            validation_data = None
        
        # Recompile with lower learning rate for fine-tuning
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss={'outcome': 'binary_crossentropy', 'points': 'huber'},
            loss_weights={'outcome': 1.0, 'points': 0.3},
            metrics={'outcome': ['accuracy', keras.metrics.AUC(name='auc')], 'points': ['mae', 'mse']}
        )
        
        # Use callbacks to prevent overfitting during fine-tuning
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=5,
            restore_best_weights=True,
            min_delta=0.0001
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if validation_data else 'loss',
            factor=0.7,
            patience=3,
            min_lr=learning_rate * 0.1,
            verbose=1
        )
        
        callbacks = [early_stopping, reduce_lr] if validation_data else [reduce_lr]
        
        history = self.model.fit(
            X_train_scaled,
            {'outcome': y_out_train, 'points': y_pts_train},
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def match_player_names(self, abbreviated_names, team_abbr, seasons):
        """Convert abbreviated names (e.g., 'J. Tatum') to full database names (e.g., 'Jayson Tatum')."""
        if not abbreviated_names:
            return []
        
        # Handle single season string or list of seasons
        if isinstance(seasons, str):
            seasons = [seasons]
        
        conn = sqlite3.connect(PLAYER_DB)
        placeholders_seasons = ','.join(['?' for _ in seasons])
        
        # Get all players for this team from the database
        query = f"""
            SELECT DISTINCT player_name, season, minutes_played
            FROM season_stats
            WHERE team_abbreviation = ? 
                AND season IN ({placeholders_seasons})
            ORDER BY season DESC, minutes_played DESC
        """
        params = [team_abbr] + list(seasons)
        cursor = conn.cursor()
        cursor.execute(query, params)
        db_players = cursor.fetchall()
        conn.close()
        
        matched_names = []
        for abbr_name in abbreviated_names:
            # Try exact match first
            exact_match = [p[0] for p in db_players if p[0] == abbr_name]
            if exact_match:
                matched_names.append(exact_match[0])
                continue
            
            # Parse abbreviated name (e.g., "J. Tatum" -> initial="J", last="Tatum")
            parts = abbr_name.split()
            if len(parts) >= 2 and parts[0].endswith('.'):
                first_initial = parts[0][0].upper()
                last_name = ' '.join(parts[1:])  # Handle multi-part last names
                
                # Find matching players (same initial + last name)
                matches = [p[0] for p in db_players 
                          if p[0].split()[0][0].upper() == first_initial 
                          and p[0].split()[-1].lower() == last_name.lower()]
                
                if matches:
                    # If multiple matches, prefer the one with most recent/most minutes
                    matched_names.append(matches[0])
                    continue
            
            # If no match found, try fuzzy matching on last name only
            if parts:
                last_name = parts[-1].replace('.', '')
                matches = [p[0] for p in db_players if p[0].split()[-1].lower() == last_name.lower()]
                if matches:
                    matched_names.append(matches[0])
        
        return matched_names
    
    def get_roster_player_stats(self, player_names, team_abbr, seasons=['2022-23', '2023-24', '2024-25', '2025-26']):
        """Get aggregated stats for a given roster (fallback to top players)."""
        if not player_names:
            return self.get_top_players_features(team_abbr, seasons, n_players=5)
        
        # Handle single season string or list of seasons
        if isinstance(seasons, str):
            seasons = [seasons]
        
        # Convert abbreviated names to full database names
        matched_names = self.match_player_names(player_names, team_abbr, seasons)
        
        if not matched_names:
            # No matches found, fall back to top players
            return self.get_top_players_features(team_abbr, seasons, n_players=5)
        
        conn = sqlite3.connect(PLAYER_DB)
        
        # Query for specific players across all seasons
        placeholders_players = ','.join(['?' for _ in matched_names])
        placeholders_seasons = ','.join(['?' for _ in seasons])
        query = f"""
            SELECT 
                points, assists, total_rebounds, steals, blocks, turnovers,
                field_goal_pct, three_point_pct, free_throw_pct, minutes_played,
                player_name
            FROM season_stats
            WHERE player_name IN ({placeholders_players}) 
                AND team_abbreviation = ? 
                AND season IN ({placeholders_seasons})
            ORDER BY season DESC, minutes_played DESC
            LIMIT 10
        """
        
        params = list(matched_names) + [team_abbr] + list(seasons)
        players_df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        features = {}
        
        if not players_df.empty and len(players_df) >= 3:
            # Aggregate roster stats
            features['top_players_avg_points'] = players_df['points'].mean()
            features['top_players_avg_assists'] = players_df['assists'].mean()
            features['top_players_avg_rebounds'] = players_df['total_rebounds'].mean()
            features['top_players_avg_steals'] = players_df['steals'].mean()
            features['top_players_avg_blocks'] = players_df['blocks'].mean()
            features['top_players_avg_turnovers'] = players_df['turnovers'].mean()
            features['top_players_avg_fg_pct'] = players_df['field_goal_pct'].mean()
            features['top_players_avg_3pt_pct'] = players_df['three_point_pct'].mean()
            features['top_players_avg_ft_pct'] = players_df['free_throw_pct'].mean()
            features['top_players_total_minutes'] = players_df['minutes_played'].sum()
            
            # Best player stats
            features['best_player_points'] = players_df['points'].iloc[0]
            features['best_player_assists'] = players_df['assists'].iloc[0]
            features['best_player_rebounds'] = players_df['total_rebounds'].iloc[0]
        else:
            # Fall back to default top players if roster data incomplete
            return self.get_top_players_features(team_abbr, seasons, n_players=5)
        
        return features
    
    def predict(self, home_team_id, away_team_id, seasons=['2022-23', '2023-24', '2024-25', '2025-26'], home_roster=None, away_roster=None):
        """Predict outcome & scores for a given matchup."""
        if self.model is None:
            raise ValueError("Model not available")
        home_abbr = self.get_team_abbreviation(home_team_id)
        away_abbr = self.get_team_abbreviation(away_team_id)
        if home_roster or away_roster:
            features = {}
            hteam = self.get_team_features(home_team_id, seasons)
            ateam = self.get_team_features(away_team_id, seasons)
            hplayers = self.get_roster_player_stats(home_roster, home_abbr, seasons) if home_roster else self.get_top_players_features(home_abbr, seasons)
            aplayers = self.get_roster_player_stats(away_roster, away_abbr, seasons) if away_roster else self.get_top_players_features(away_abbr, seasons)
            for k, v in hteam.items(): features[f'home_{k}'] = v
            for k, v in hplayers.items(): features[f'home_{k}'] = v
            for k, v in ateam.items(): features[f'away_{k}'] = v
            for k, v in aplayers.items(): features[f'away_{k}'] = v
            features['home_court_advantage'] = 1.0
        else:
            features = self.prepare_game_features(home_team_id, away_team_id, seasons)
        X_df = pd.DataFrame([features])
        for col in self.feature_columns:
            if col not in X_df.columns: X_df[col] = 0
        X = self.scaler.transform(X_df[self.feature_columns].values)
        outcome_prob, points_pred = self.model.predict(X, verbose=0)
        return {
            'home_team': home_abbr,
            'away_team': away_abbr,
            'home_win_probability': float(outcome_prob[0][0]),
            'away_win_probability': float(1 - outcome_prob[0][0]),
            'predicted_home_points': float(points_pred[0][0]),
            'predicted_away_points': float(points_pred[0][1]),
            'predicted_winner': home_abbr if outcome_prob[0][0] > 0.5 else away_abbr,
        }
    
    def evaluate(self, X, y_outcome, y_points):
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y_outcome: True outcomes
            y_points: True points
        
        Returns:
            Evaluation metrics
        """
        X_scaled = self.scaler.transform(X)
        
        results = self.model.evaluate(
            X_scaled,
            {'outcome': y_outcome, 'points': y_points},
            verbose=0
        )
        
        # Predictions for detailed metrics
        outcome_pred, points_pred = self.model.predict(X_scaled, verbose=0)
        
        # Accuracy
        accuracy = np.mean((outcome_pred > 0.5).astype(int).flatten() == y_outcome)
        
        # MAE for points
        mae_home = np.mean(np.abs(points_pred[:, 0] - y_points[:, 0]))
        mae_away = np.mean(np.abs(points_pred[:, 1] - y_points[:, 1]))
        
        return {
            'outcome_accuracy': accuracy,
            'points_mae_home': mae_home,
            'points_mae_away': mae_away,
            'points_mae_avg': (mae_home + mae_away) / 2
        }
    
    def save_model(self, path='nba_model_saved'):
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, 'model.keras'))
        with open(os.path.join(path, 'scaler.pkl'), 'wb') as f: pickle.dump(self.scaler, f)
        with open(os.path.join(path, 'features.pkl'), 'wb') as f: pickle.dump(self.feature_columns, f)
    
    def load_model(self, path='nba_model_saved'):
        self.model = keras.models.load_model(os.path.join(path, 'model.keras'))
        with open(os.path.join(path, 'scaler.pkl'), 'rb') as f: self.scaler = pickle.load(f)
        with open(os.path.join(path, 'features.pkl'), 'rb') as f: self.feature_columns = pickle.load(f)


class EnsembleNBAPredictor:
    """Ensemble of multiple NBA prediction models."""
    
    def __init__(self, n_models=3, model_path=None):
        """
        Initialize ensemble predictor.
        
        Args:
            n_models: Number of models in the ensemble
            model_path: Path to load saved ensemble from
        """
        self.n_models = n_models
        self.models = []
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load_ensemble(model_path)
    
    def train_ensemble(self, X, y_outcome, y_points, sample_weights=None, feature_columns=None, epochs=25, batch_size=32, validation_split=0.2):
        """
        Train multiple models with different random initializations.
        
        Args:
            X: Feature matrix
            y_outcome: Outcome labels
            y_points: Points labels
            sample_weights: Optional sample weights for temporal weighting
            feature_columns: List of feature column names (required for proper model saving/loading)
            epochs: Training epochs per model
            batch_size: Batch size
            validation_split: Validation split ratio
        """
        print(f"\nTraining ensemble of {self.n_models} models...")
        print("="*80)
        
        self.models = []
        
        for i in range(self.n_models):
            print(f"\nTraining model {i+1}/{self.n_models}...")
            print("-"*80)
            
            # Create new predictor with different random seed
            predictor = NBAGamePredictor()
            
            # Set feature columns if provided
            if feature_columns is not None:
                predictor.feature_columns = feature_columns
            
            # Set random seed for reproducibility
            np.random.seed(42 + i)
            tf.random.set_seed(42 + i)
            
            # Train the model with sample weights
            history = predictor.train(
                X, y_outcome, y_points,
                sample_weights=sample_weights,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split
            )
            
            self.models.append(predictor)
            
            print(f"\nModel {i+1} training complete!")
        
        print("\n" + "="*80)
        print(f"Ensemble training complete! Trained {len(self.models)} models.")
        print("="*80)
    
    def predict(self, home_team_id, away_team_id, seasons=['2022-23', '2023-24', '2024-25', '2025-26'], 
                home_roster=None, away_roster=None):
        """
        Make prediction using ensemble average.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            seasons: Seasons to use for features
            home_roster: Optional home team roster
            away_roster: Optional away team roster
        
        Returns:
            Averaged prediction from all models
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(home_team_id, away_team_id, seasons=seasons, 
                               home_roster=home_roster, away_roster=away_roster)
            predictions.append(pred)
        
        # Average predictions
        avg_home_win_prob = np.mean([p['home_win_probability'] for p in predictions])
        avg_home_points = np.mean([p['predicted_home_points'] for p in predictions])
        avg_away_points = np.mean([p['predicted_away_points'] for p in predictions])
        
        # Get team abbreviations from first prediction
        home_team = predictions[0]['home_team']
        away_team = predictions[0]['away_team']
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_win_probability': float(avg_home_win_prob),
            'away_win_probability': float(1 - avg_home_win_prob),
            'predicted_home_points': float(avg_home_points),
            'predicted_away_points': float(avg_away_points),
            'predicted_winner': home_team if avg_home_win_prob > 0.5 else away_team,
            'individual_predictions': predictions  # Include individual model predictions
        }
    
    def evaluate(self, X, y_outcome, y_points):
        """
        Evaluate ensemble performance.
        
        Args:
            X: Feature matrix
            y_outcome: True outcomes
            y_points: True points
        
        Returns:
            Evaluation metrics
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get predictions from all models
        all_outcome_probs = []
        all_points_preds = []
        
        for model in self.models:
            # Simply use the model's evaluate method which already handles scaling
            outcome_pred, points_pred = model.model.predict(
                model.scaler.transform(X), 
                verbose=0
            )
            all_outcome_probs.append(outcome_pred)
            all_points_preds.append(points_pred)
        
        # Average predictions
        avg_outcome_probs = np.mean(all_outcome_probs, axis=0)
        avg_points_preds = np.mean(all_points_preds, axis=0)
        
        # Calculate metrics
        accuracy = np.mean((avg_outcome_probs > 0.5).astype(int).flatten() == y_outcome)
        mae_home = np.mean(np.abs(avg_points_preds[:, 0] - y_points[:, 0]))
        mae_away = np.mean(np.abs(avg_points_preds[:, 1] - y_points[:, 1]))
        
        # Also calculate individual model metrics
        individual_metrics = []
        for i, model in enumerate(self.models):
            metrics = model.evaluate(X, y_outcome, y_points)
            individual_metrics.append(metrics)
        
        return {
            'ensemble_outcome_accuracy': accuracy,
            'ensemble_points_mae_home': mae_home,
            'ensemble_points_mae_away': mae_away,
            'ensemble_points_mae_avg': (mae_home + mae_away) / 2,
            'individual_model_metrics': individual_metrics
        }
    
    def save_ensemble(self, path='ensemble_model_saved'):
        """Save all models in the ensemble."""
        os.makedirs(path, exist_ok=True)
        
        for i, model in enumerate(self.models):
            model_dir = os.path.join(path, f'model_{i}')
            model.save_model(model_dir)
        
        # Save ensemble metadata
        metadata = {
            'n_models': self.n_models,
            'model_count': len(self.models)
        }
        with open(os.path.join(path, 'ensemble_metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\nEnsemble saved to {path}/")
    
    def load_ensemble(self, path='ensemble_model_saved'):
        """Load all models in the ensemble."""
        # Load metadata
        with open(os.path.join(path, 'ensemble_metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        self.n_models = metadata['n_models']
        self.models = []
        
        # Load each model
        for i in range(metadata['model_count']):
            model_dir = os.path.join(path, f'model_{i}')
            if os.path.exists(model_dir):
                predictor = NBAGamePredictor(model_path=model_dir)
                self.models.append(predictor)
        
        print(f"\nLoaded ensemble with {len(self.models)} models from {path}/")


def train_single_model():
    """Train a single NBA prediction model."""
    print("="*80)
    print("NBA GAME PREDICTION MODEL - SINGLE MODEL")
    print("="*80)
    
    # Initialize predictor
    predictor = NBAGamePredictor()
    
    # Create training data from multiple seasons with temporal weighting
    print("\n1. Creating training dataset from 2022-23 to 2025-26 seasons...")
    X_train, y_out_train, y_pts_train, weights_train = predictor.create_training_data(
        seasons=['2022-23', '2023-24', '2024-25', '2025-26']
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    
    # Create test data from 2023-24 season for validation
    print("\n2. Creating test dataset from 2023-24 season...")
    X_test, y_out_test, y_pts_test, _ = predictor.create_training_data(seasons=['2023-24'])
    
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model with temporal weighting
    print("\n3. Training deep neural network with temporal weighting...")
    history = predictor.train(X_train, y_out_train, y_pts_train, sample_weights=weights_train, epochs=50)
    
    # Evaluate
    print("\n4. Evaluating model on previous seasons...")
    metrics = predictor.evaluate(X_test, y_out_test, y_pts_test)
    
    print("\nTest Set Performance (2022-23 & 2023-24 seasons):")
    print(f"  Outcome Accuracy: {metrics['outcome_accuracy']:.2%}")
    print(f"  Points MAE (Home): {metrics['points_mae_home']:.2f}")
    print(f"  Points MAE (Away): {metrics['points_mae_away']:.2f}")
    print(f"  Points MAE (Avg): {metrics['points_mae_avg']:.2f}")
    
    # Save model
    print("\n5. Saving model...")
    predictor.save_model()
    
    # Example prediction
    print("\n6. Example Prediction:")
    print("-" * 80)
    
    # Get Lakers (1610612747) vs Warriors (1610612744) prediction as example
    try:
        prediction = predictor.predict(1610612747, 1610612744)
        
        print(f"\nMatchup: {prediction['home_team']} (Home) vs {prediction['away_team']} (Away)")
        print(f"\nWin Probability:")
        print(f"  {prediction['home_team']}: {prediction['home_win_probability']:.1%}")
        print(f"  {prediction['away_team']}: {prediction['away_win_probability']:.1%}")
        print(f"\nPredicted Score:")
        print(f"  {prediction['home_team']}: {prediction['predicted_home_points']:.1f}")
        print(f"  {prediction['away_team']}: {prediction['predicted_away_points']:.1f}")
        print(f"\nPredicted Winner: {prediction['predicted_winner']}")
        
    except Exception as e:
        print(f"Could not make example prediction: {e}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)


def train_ensemble_model(n_models=3, epochs=50):
    """Train an ensemble of NBA prediction models."""
    print("="*80)
    print("NBA GAME PREDICTION MODEL - ENSEMBLE")
    print("="*80)
    
    # Initialize ensemble
    ensemble = EnsembleNBAPredictor(n_models=n_models)
    
    # Create base predictor to get data
    print("\n1. Creating training dataset from 2022-23 to 2025-26 seasons...")
    base_predictor = NBAGamePredictor()
    X_train, y_out_train, y_pts_train, weights_train = base_predictor.create_training_data(
        seasons=['2022-23', '2023-24', '2024-25', '2025-26']
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    
    # Create test data from 2023-24 season for validation
    print("\n2. Creating test dataset from 2023-24 season...")
    X_test, y_out_test, y_pts_test, _ = base_predictor.create_training_data(seasons=['2023-24'])
    
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train ensemble with temporal weighting
    print(f"\n3. Training ensemble of {n_models} deep models with temporal weighting...")
    ensemble.train_ensemble(X_train, y_out_train, y_pts_train, 
                           sample_weights=weights_train,
                           feature_columns=base_predictor.feature_columns, 
                           epochs=epochs)
    
    # Evaluate ensemble
    print("\n4. Evaluating ensemble on previous seasons...")
    metrics = ensemble.evaluate(X_test, y_out_test, y_pts_test)
    
    print("\n" + "="*80)
    print("ENSEMBLE PERFORMANCE (2022-23 & 2023-24 seasons)")
    print("="*80)
    print(f"\nEnsemble Metrics:")
    print(f"  Outcome Accuracy: {metrics['ensemble_outcome_accuracy']:.2%}")
    print(f"  Points MAE (Home): {metrics['ensemble_points_mae_home']:.2f}")
    print(f"  Points MAE (Away): {metrics['ensemble_points_mae_away']:.2f}")
    print(f"  Points MAE (Avg): {metrics['ensemble_points_mae_avg']:.2f}")
    
    print(f"\nIndividual Model Performance:")
    for i, model_metrics in enumerate(metrics['individual_model_metrics']):
        print(f"\n  Model {i+1}:")
        print(f"    Accuracy: {model_metrics['outcome_accuracy']:.2%}")
        print(f"    Points MAE: {model_metrics['points_mae_avg']:.2f}")
    
    # Save ensemble
    print("\n5. Saving ensemble...")
    ensemble.save_ensemble()
    
    # Example prediction
    print("\n6. Example Prediction:")
    print("-" * 80)
    
    try:
        prediction = ensemble.predict(1610612747, 1610612744)  # Lakers vs Warriors
        
        print(f"\nMatchup: {prediction['home_team']} (Home) vs {prediction['away_team']} (Away)")
        print(f"\nEnsemble Prediction:")
        print(f"  Win Probability:")
        print(f"    {prediction['home_team']}: {prediction['home_win_probability']:.1%}")
        print(f"    {prediction['away_team']}: {prediction['away_win_probability']:.1%}")
        print(f"  Predicted Score:")
        print(f"    {prediction['home_team']}: {prediction['predicted_home_points']:.1f}")
        print(f"    {prediction['away_team']}: {prediction['predicted_away_points']:.1f}")
        print(f"  Predicted Winner: {prediction['predicted_winner']}")
        
        print(f"\nIndividual Model Predictions:")
        for i, pred in enumerate(prediction['individual_predictions']):
            print(f"  Model {i+1}: {pred['predicted_winner']} "
                  f"({pred['predicted_home_points']:.1f} - {pred['predicted_away_points']:.1f})")
        
    except Exception as e:
        print(f"Could not make example prediction: {e}")
    
    print("\n" + "="*80)
    print("ENSEMBLE TRAINING COMPLETE!")
    print("="*80)


def main():
    """Main entry point - choose between single model or ensemble training."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--ensemble':
        # Train ensemble model
        n_models = 3
        if len(sys.argv) > 2:
            try:
                n_models = int(sys.argv[2])
            except ValueError:
                print("Invalid number of models. Using default: 3")
        train_ensemble_model(n_models=n_models)
    else:
        # Train single model (default)
        train_single_model()


if __name__ == '__main__':
    main()
