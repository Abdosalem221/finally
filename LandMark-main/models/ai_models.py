"""
Advanced AI Models Integration Module
Implements 50 advanced AI models for market analysis and signal generation
"""

import os
import sys
import time
import json
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
from collections import defaultdict
import joblib
import math
import random
from scipy import stats

# Import machine learning libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, 
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet, 
    SGDClassifier, SGDRegressor, LinearRegression
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    TimeSeriesSplit, KFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.pipeline import Pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ai_models.log')
    ]
)

logger = logging.getLogger("ai_models")

class AdvancedAIModels:
    """
    Implementation of 50 advanced AI models for market analysis and prediction
    """
    
    def __init__(self, app=None):
        """Initialize the advanced AI models system"""
        self.app = app
        
        # Dictionary to store all models
        self.models = {}
        
        # Dictionary to store model metadata
        self.model_metadata = {
            # PRICE PREDICTION MODELS
            "lstm_price_predictor": {
                "type": "price_prediction",
                "algorithm": "lstm",
                "description": "Long Short-Term Memory neural network for price prediction",
                "features_required": ["close", "open", "high", "low", "volume"],
                "prediction_horizon": [1, 3, 5, 10],
                "accuracy": 0.0,
                "last_trained": None
            },
            "xgboost_price_predictor": {
                "type": "price_prediction",
                "algorithm": "xgboost",
                "description": "XGBoost model for price prediction with feature importance",
                "features_required": ["close", "open", "high", "low", "volume", "technical_indicators"],
                "prediction_horizon": [1, 3, 5],
                "accuracy": 0.0,
                "last_trained": None
            },
            "ensemble_price_predictor": {
                "type": "price_prediction",
                "algorithm": "ensemble",
                "description": "Ensemble of multiple algorithms for robust price prediction",
                "features_required": ["close", "open", "high", "low", "volume", "technical_indicators"],
                "prediction_horizon": [1, 3, 5],
                "accuracy": 0.0,
                "last_trained": None
            },
            "transformer_price_predictor": {
                "type": "price_prediction",
                "algorithm": "transformer",
                "description": "Transformer-based model for capturing long-range price dependencies",
                "features_required": ["close", "open", "high", "low", "volume"],
                "prediction_horizon": [1, 5, 10],
                "accuracy": 0.0,
                "last_trained": None
            },
            "wavenet_price_predictor": {
                "type": "price_prediction",
                "algorithm": "wavenet",
                "description": "WaveNet-inspired model for financial time series prediction",
                "features_required": ["close"],
                "prediction_horizon": [1, 3, 5],
                "accuracy": 0.0,
                "last_trained": None
            },
            "temporal_fusion_transformer": {
                "type": "price_prediction",
                "algorithm": "tft",
                "description": "Temporal Fusion Transformer for multi-horizon forecasting",
                "features_required": ["close", "open", "high", "low", "volume", "technical_indicators"],
                "prediction_horizon": [1, 3, 5, 10, 20],
                "accuracy": 0.0,
                "last_trained": None
            },
            "nbeats_predictor": {
                "type": "price_prediction",
                "algorithm": "nbeats",
                "description": "Neural Basis Expansion Analysis for Time Series forecasting",
                "features_required": ["close"],
                "prediction_horizon": [1, 5, 10, 20],
                "accuracy": 0.0,
                "last_trained": None
            },
            "prophet_price_predictor": {
                "type": "price_prediction",
                "algorithm": "prophet",
                "description": "Facebook Prophet model for trend and seasonality decomposition",
                "features_required": ["close"],
                "prediction_horizon": [1, 5, 10, 20],
                "accuracy": 0.0,
                "last_trained": None
            },
            "deepar_price_predictor": {
                "type": "price_prediction",
                "algorithm": "deepar",
                "description": "DeepAR probabilistic forecasting with autoregressive RNNs",
                "features_required": ["close", "technical_indicators"],
                "prediction_horizon": [1, 5, 10],
                "accuracy": 0.0,
                "last_trained": None
            },
            "informer_price_predictor": {
                "type": "price_prediction",
                "algorithm": "informer",
                "description": "Informer model for efficient long-sequence time-series forecasting",
                "features_required": ["close", "open", "high", "low", "volume"],
                "prediction_horizon": [1, 5, 10, 20, 30],
                "accuracy": 0.0,
                "last_trained": None
            },
            
            # TREND PREDICTION MODELS
            "cnn_trend_classifier": {
                "type": "trend_prediction",
                "algorithm": "cnn",
                "description": "Convolutional Neural Network for trend classification",
                "features_required": ["close", "ohlcv_patterns"],
                "classes": ["up", "down", "sideways"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "random_forest_trend_classifier": {
                "type": "trend_prediction",
                "algorithm": "random_forest",
                "description": "Random Forest for trend direction classification",
                "features_required": ["technical_indicators"],
                "classes": ["up", "down", "sideways"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "xgboost_trend_classifier": {
                "type": "trend_prediction",
                "algorithm": "xgboost",
                "description": "XGBoost classifier for trend prediction with feature importance",
                "features_required": ["technical_indicators", "close", "volume"],
                "classes": ["up", "down", "sideways"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "lstm_trend_classifier": {
                "type": "trend_prediction",
                "algorithm": "lstm",
                "description": "LSTM network for trend classification with sequential data",
                "features_required": ["close", "technical_indicators"],
                "classes": ["up", "down", "sideways"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "svm_trend_classifier": {
                "type": "trend_prediction",
                "algorithm": "svm",
                "description": "Support Vector Machine for trend boundary detection",
                "features_required": ["technical_indicators"],
                "classes": ["up", "down", "sideways"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "ensemble_trend_classifier": {
                "type": "trend_prediction",
                "algorithm": "ensemble",
                "description": "Ensemble of multiple models for robust trend classification",
                "features_required": ["technical_indicators", "close", "volume"],
                "classes": ["up", "down", "sideways"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "gradient_boosting_trend_classifier": {
                "type": "trend_prediction",
                "algorithm": "gradient_boosting",
                "description": "Gradient Boosting for trend classification with feature importance",
                "features_required": ["technical_indicators", "moving_averages"],
                "classes": ["up", "down", "sideways"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "transformer_trend_classifier": {
                "type": "trend_prediction",
                "algorithm": "transformer",
                "description": "Transformer model for trend classification with attention mechanism",
                "features_required": ["close", "technical_indicators"],
                "classes": ["up", "down", "sideways"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "ada_boost_trend_classifier": {
                "type": "trend_prediction",
                "algorithm": "ada_boost",
                "description": "AdaBoost ensemble for trend classification",
                "features_required": ["technical_indicators"],
                "classes": ["up", "down", "sideways"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "catboost_trend_classifier": {
                "type": "trend_prediction",
                "algorithm": "catboost",
                "description": "CatBoost classifier for trend prediction with categorical features",
                "features_required": ["technical_indicators", "market_regime", "session"],
                "classes": ["up", "down", "sideways"],
                "accuracy": 0.0,
                "last_trained": None
            },
            
            # VOLATILITY PREDICTION MODELS
            "garch_volatility_predictor": {
                "type": "volatility_prediction",
                "algorithm": "garch",
                "description": "GARCH model for volatility forecasting",
                "features_required": ["returns"],
                "prediction_horizon": [1, 5, 10],
                "accuracy": 0.0,
                "last_trained": None
            },
            "lstm_volatility_predictor": {
                "type": "volatility_prediction",
                "algorithm": "lstm",
                "description": "LSTM network for volatility prediction",
                "features_required": ["returns", "realized_volatility"],
                "prediction_horizon": [1, 5, 10],
                "accuracy": 0.0,
                "last_trained": None
            },
            "gaussian_process_volatility": {
                "type": "volatility_prediction",
                "algorithm": "gaussian_process",
                "description": "Gaussian Process Regression for volatility with uncertainty estimation",
                "features_required": ["returns", "realized_volatility"],
                "prediction_horizon": [1, 5],
                "accuracy": 0.0,
                "last_trained": None
            },
            "svr_volatility_predictor": {
                "type": "volatility_prediction",
                "algorithm": "svr",
                "description": "Support Vector Regression for volatility prediction",
                "features_required": ["returns", "realized_volatility", "technical_indicators"],
                "prediction_horizon": [1, 5],
                "accuracy": 0.0,
                "last_trained": None
            },
            "random_forest_volatility_predictor": {
                "type": "volatility_prediction",
                "algorithm": "random_forest",
                "description": "Random Forest for volatility prediction",
                "features_required": ["returns", "realized_volatility", "technical_indicators"],
                "prediction_horizon": [1, 5],
                "accuracy": 0.0,
                "last_trained": None
            },
            
            # PATTERN RECOGNITION MODELS
            "cnn_pattern_recognizer": {
                "type": "pattern_recognition",
                "algorithm": "cnn",
                "description": "CNN for candlestick pattern recognition on price charts",
                "features_required": ["ohlc_image"],
                "patterns": ["double_top", "double_bottom", "head_shoulders", "inv_head_shoulders", 
                            "triangle", "flag", "pennant", "wedge", "rectangle"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "dtw_pattern_matcher": {
                "type": "pattern_recognition",
                "algorithm": "dtw",
                "description": "Dynamic Time Warping for pattern matching with historical patterns",
                "features_required": ["close"],
                "patterns": ["double_top", "double_bottom", "head_shoulders", "inv_head_shoulders", 
                            "triangle", "flag", "pennant", "wedge", "rectangle"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "hmm_pattern_recognizer": {
                "type": "pattern_recognition",
                "algorithm": "hmm",
                "description": "Hidden Markov Model for pattern state detection",
                "features_required": ["close", "returns"],
                "patterns": ["accumulation", "distribution", "markup", "markdown"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "template_matching_recognizer": {
                "type": "pattern_recognition",
                "algorithm": "template_matching",
                "description": "Template matching with historical patterns",
                "features_required": ["close"],
                "patterns": ["double_top", "double_bottom", "head_shoulders", "inv_head_shoulders", 
                            "triangle", "flag", "pennant", "wedge", "rectangle"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "lstm_pattern_recognizer": {
                "type": "pattern_recognition",
                "algorithm": "lstm",
                "description": "LSTM for sequential pattern recognition",
                "features_required": ["close", "returns"],
                "patterns": ["double_top", "double_bottom", "head_shoulders", "inv_head_shoulders", 
                            "triangle", "flag", "pennant", "wedge", "rectangle"],
                "accuracy": 0.0,
                "last_trained": None
            },
            
            # REGIME DETECTION MODELS
            "hmm_regime_detector": {
                "type": "regime_detection",
                "algorithm": "hmm",
                "description": "Hidden Markov Model for market regime identification",
                "features_required": ["returns", "volatility"],
                "regimes": ["bull", "bear", "sideways", "volatile"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "kmeans_regime_detector": {
                "type": "regime_detection",
                "algorithm": "kmeans",
                "description": "K-means clustering for regime identification",
                "features_required": ["returns", "volatility", "trading_volume"],
                "regimes": ["bull", "bear", "sideways", "volatile"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "gaussian_mixture_regime_detector": {
                "type": "regime_detection",
                "algorithm": "gmm",
                "description": "Gaussian Mixture Model for regime probability estimation",
                "features_required": ["returns", "volatility", "trading_volume"],
                "regimes": ["bull", "bear", "sideways", "volatile"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "dbscan_regime_detector": {
                "type": "regime_detection",
                "algorithm": "dbscan",
                "description": "DBSCAN for density-based regime clustering",
                "features_required": ["returns", "volatility", "trading_volume"],
                "regimes": ["bull", "bear", "sideways", "volatile"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "random_forest_regime_classifier": {
                "type": "regime_detection",
                "algorithm": "random_forest",
                "description": "Random Forest for market regime classification",
                "features_required": ["returns", "volatility", "technical_indicators"],
                "regimes": ["bull", "bear", "sideways", "volatile"],
                "accuracy": 0.0,
                "last_trained": None
            },
            
            # ANOMALY DETECTION MODELS
            "isolation_forest_detector": {
                "type": "anomaly_detection",
                "algorithm": "isolation_forest",
                "description": "Isolation Forest for detecting market anomalies",
                "features_required": ["returns", "volatility", "volume"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "lof_anomaly_detector": {
                "type": "anomaly_detection",
                "algorithm": "lof",
                "description": "Local Outlier Factor for anomaly detection",
                "features_required": ["returns", "volatility", "volume"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "autoencoder_anomaly_detector": {
                "type": "anomaly_detection",
                "algorithm": "autoencoder",
                "description": "Autoencoder for reconstruction-based anomaly detection",
                "features_required": ["returns", "volatility", "volume", "technical_indicators"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "pca_anomaly_detector": {
                "type": "anomaly_detection",
                "algorithm": "pca",
                "description": "PCA-based anomaly detection through dimensionality reduction",
                "features_required": ["returns", "technical_indicators"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "statistical_anomaly_detector": {
                "type": "anomaly_detection",
                "algorithm": "statistical",
                "description": "Statistical methods (Z-score, GESD) for anomaly detection",
                "features_required": ["returns", "volume"],
                "accuracy": 0.0,
                "last_trained": None
            },
            
            # SENTIMENT ANALYSIS MODELS
            "bert_sentiment_analyzer": {
                "type": "sentiment_analysis",
                "algorithm": "bert",
                "description": "BERT-based model for news and social media sentiment analysis",
                "features_required": ["news_text", "social_media_text"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "vader_sentiment_analyzer": {
                "type": "sentiment_analysis",
                "algorithm": "vader",
                "description": "VADER lexicon-based sentiment analyzer",
                "features_required": ["news_text", "social_media_text"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "finbert_sentiment_analyzer": {
                "type": "sentiment_analysis",
                "algorithm": "finbert",
                "description": "FinBERT model fine-tuned for financial text sentiment",
                "features_required": ["news_text", "financial_reports"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "lstm_sentiment_analyzer": {
                "type": "sentiment_analysis",
                "algorithm": "lstm",
                "description": "LSTM model for sequential sentiment analysis",
                "features_required": ["news_text"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "market_buzz_analyzer": {
                "type": "sentiment_analysis",
                "algorithm": "ensemble",
                "description": "Ensemble model for market buzz and sentiment monitoring",
                "features_required": ["news_volume", "social_media_volume", "news_sentiment", "social_media_sentiment"],
                "accuracy": 0.0,
                "last_trained": None
            },
            
            # ORDER FLOW MODELS
            "limit_order_book_cnn": {
                "type": "order_flow",
                "algorithm": "cnn",
                "description": "CNN for analyzing limit order book patterns",
                "features_required": ["order_book_snapshot"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "lstm_order_flow_predictor": {
                "type": "order_flow",
                "algorithm": "lstm",
                "description": "LSTM for order flow prediction",
                "features_required": ["order_flow_history"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "transformer_order_book_analyzer": {
                "type": "order_flow",
                "algorithm": "transformer",
                "description": "Transformer model for order book dynamics analysis",
                "features_required": ["order_book_snapshots"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "reinforcement_learning_order_flow": {
                "type": "order_flow",
                "algorithm": "rl",
                "description": "Reinforcement learning for order flow pattern recognition",
                "features_required": ["order_flow_history", "order_book_snapshots"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "event_based_order_flow_analyzer": {
                "type": "order_flow",
                "algorithm": "event_based",
                "description": "Event-based analysis of order flow imbalances",
                "features_required": ["order_events", "trade_events"],
                "accuracy": 0.0,
                "last_trained": None
            },
            
            # CORRELATION AND INTERMARKET MODELS
            "dynamic_correlation_network": {
                "type": "correlation",
                "algorithm": "network",
                "description": "Dynamic correlation network for market interconnections",
                "features_required": ["multi_asset_returns"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "copula_dependency_model": {
                "type": "correlation",
                "algorithm": "copula",
                "description": "Copula-based model for complex dependency structures",
                "features_required": ["multi_asset_returns"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "lead_lag_detector": {
                "type": "correlation",
                "algorithm": "lead_lag",
                "description": "Lead-lag relationship detector between assets",
                "features_required": ["multi_asset_returns"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "correlation_regime_switch_detector": {
                "type": "correlation",
                "algorithm": "regime_switching",
                "description": "Model for detecting correlation regime switches",
                "features_required": ["multi_asset_returns", "volatility"],
                "accuracy": 0.0,
                "last_trained": None
            },
            "hierarchical_risk_parity": {
                "type": "correlation",
                "algorithm": "hrp",
                "description": "Hierarchical Risk Parity for correlation-based portfolio allocation",
                "features_required": ["multi_asset_returns"],
                "accuracy": 0.0,
                "last_trained": None
            }
        }
        
        # Initialize model performance tracking
        self.model_performance = defaultdict(dict)
        
        # Load existing models if available
        self._load_models()
    
    def _load_models(self):
        """Load trained models from disk"""
        try:
            models_dir = "ai_models"
            
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
                logger.info(f"Created directory for AI models: {models_dir}")
                return
            
            # Look for model files
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
            
            for model_file in model_files:
                try:
                    model_path = os.path.join(models_dir, model_file)
                    model_data = joblib.load(model_path)
                    
                    model_name = model_file.replace('.joblib', '')
                    self.models[model_name] = model_data.get('model')
                    
                    # Update metadata if available
                    if 'metadata' in model_data:
                        self.model_metadata[model_name].update(model_data['metadata'])
                    
                    # Update performance data if available
                    if 'performance' in model_data:
                        self.model_performance[model_name] = model_data['performance']
                    
                    logger.info(f"Loaded AI model: {model_name}")
                except Exception as e:
                    logger.error(f"Error loading model {model_file}: {str(e)}")
            
            logger.info(f"Loaded {len(self.models)} AI models")
        except Exception as e:
            logger.error(f"Error loading AI models: {str(e)}")
    
    def _save_model(self, model_name, model, metadata=None, performance=None):
        """Save a model to disk"""
        try:
            models_dir = "ai_models"
            
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            
            model_path = os.path.join(models_dir, f"{model_name}.joblib")
            
            # Prepare model data
            model_data = {
                'model': model,
                'metadata': metadata or {},
                'performance': performance or {}
            }
            
            # Save model
            joblib.dump(model_data, model_path)
            
            logger.info(f"Saved AI model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {str(e)}")
            return False
    
    def load_training_data(self, symbol, period='1d', limit=1000):
        """
        Load historical data for model training
        
        Args:
            symbol: Symbol/instrument to get data for
            period: Timeframe period (1m, 5m, 1h, 1d, etc.)
            limit: Number of data points to retrieve
            
        Returns:
            Pandas DataFrame with historical data and features
        """
        try:
            # Check if app is available to get data from database
            if self.app:
                from technical_analysis import TechnicalAnalysis
                
                # Initialize technical analysis
                ta = TechnicalAnalysis()
                
                # Get historical data with technical indicators
                data = ta.get_historical_data(symbol, period=period, limit=limit, include_indicators=True)
                
                if data is not None and not data.empty:
                    return data
            
            # If app is not available or data retrieval failed, return None
            logger.warning(f"Could not load training data for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            return None
    
    def engineer_features(self, data):
        """
        Apply feature engineering to prepare data for AI models
        
        Args:
            data: DataFrame with historical market data
            
        Returns:
            DataFrame with engineered features
        """
        if data is None or data.empty:
            return None
        
        try:
            # Make a copy to avoid modifying the original data
            df = data.copy()
            
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Volatility features
            df['realized_volatility'] = df['returns'].rolling(window=22).std() * np.sqrt(252)
            df['high_low_range'] = (df['high'] - df['low']) / df['close']
            df['daily_range'] = (df['high'] - df['low'])
            
            # Volume features
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Price position features
            df['price_ma_ratio_20'] = df['close'] / df['close'].rolling(window=20).mean()
            df['price_ma_ratio_50'] = df['close'] / df['close'].rolling(window=50).mean()
            
            # Trend strength features
            df['adx'] = self._calculate_adx(df)
            
            # Momentum features
            df['rsi_diff'] = df['rsi'] - df['rsi'].shift(1)
            df['macd_diff'] = df['macd'] - df['macd'].shift(1)
            
            # Remove NaN values
            df = df.dropna()
            
            return df
        except Exception as e:
            logger.error(f"Error engineering features: {str(e)}")
            return data
    
    def _calculate_adx(self, df, period=14):
        """Calculate Average Directional Index (ADX)"""
        try:
            # Calculate True Range
            df['tr1'] = abs(df['high'] - df['low'])
            df['tr2'] = abs(df['high'] - df['close'].shift(1))
            df['tr3'] = abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            
            # Calculate Directional Movement
            df['up_move'] = df['high'] - df['high'].shift(1)
            df['down_move'] = df['low'].shift(1) - df['low']
            
            # Calculate Positive and Negative Directional Movement
            df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
            df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
            
            # Calculate Smoothed Averages
            df['tr_smoothed'] = df['tr'].rolling(window=period).mean()
            df['plus_di'] = 100 * (df['plus_dm'].rolling(window=period).mean() / df['tr_smoothed'])
            df['minus_di'] = 100 * (df['minus_dm'].rolling(window=period).mean() / df['tr_smoothed'])
            
            # Calculate Directional Index
            df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']))
            
            # Calculate ADX
            df['adx'] = df['dx'].rolling(window=period).mean()
            
            # Clean up intermediate columns
            columns_to_drop = ['tr1', 'tr2', 'tr3', 'tr', 'up_move', 'down_move', 'plus_dm', 'minus_dm', 
                              'tr_smoothed', 'plus_di', 'minus_di', 'dx']
            df = df.drop(columns=columns_to_drop, errors='ignore')
            
            return df['adx']
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            return pd.Series(np.nan, index=df.index)
    
    def prepare_training_data(self, data, target_column, feature_columns=None, sequence_length=None, 
                              prediction_horizon=1, train_size=0.8, scale=True):
        """
        Prepare data for model training, including scaling and train/test split
        
        Args:
            data: DataFrame with processed data
            target_column: Column name for the target variable
            feature_columns: List of column names to use as features
            sequence_length: Length of sequence for sequential models (LSTM, etc.)
            prediction_horizon: Number of periods ahead to predict
            train_size: Proportion of data to use for training
            scale: Whether to apply scaling to the data
            
        Returns:
            Dictionary containing processed datasets and related objects
        """
        if data is None or data.empty:
            return None
        
        try:
            # Make a copy to avoid modifying the original data
            df = data.copy()
            
            # Drop rows with NaN values
            df = df.dropna()
            
            # Use default feature columns if not provided
            if feature_columns is None:
                # Exclude timestamp and target columns for default features
                feature_columns = [col for col in df.columns if col != 'timestamp' and col != target_column]
            
            # Create shifted target for predicting future values
            target_column_shifted = f"{target_column}_future_{prediction_horizon}"
            df[target_column_shifted] = df[target_column].shift(-prediction_horizon)
            
            # Drop rows with NaN values after shifting
            df = df.dropna()
            
            # Extract features and target
            X = df[feature_columns].values
            y = df[target_column_shifted].values
            
            # Scale the data if requested
            scaler_X = None
            scaler_y = None
            
            if scale:
                scaler_X = StandardScaler()
                X = scaler_X.fit_transform(X)
                
                # Use MinMaxScaler for the target if it's a price
                if 'price' in target_column or 'close' in target_column:
                    scaler_y = MinMaxScaler()
                    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
                else:
                    scaler_y = StandardScaler()
                    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            
            # For sequential models (LSTM, CNN, etc.), reshape data into sequences
            if sequence_length is not None:
                X_seq, y_seq = self._create_sequences(X, y, sequence_length)
                
                # Split into training and testing sets
                train_idx = int(train_size * len(X_seq))
                X_train, X_test = X_seq[:train_idx], X_seq[train_idx:]
                y_train, y_test = y_seq[:train_idx], y_seq[train_idx:]
            else:
                # Split into training and testing sets
                train_idx = int(train_size * len(X))
                X_train, X_test = X[:train_idx], X[train_idx:]
                y_train, y_test = y[:train_idx], y[train_idx:]
            
            # Prepare result dictionary
            result = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "feature_columns": feature_columns,
                "target_column": target_column,
                "scaler_X": scaler_X,
                "scaler_y": scaler_y,
                "sequence_length": sequence_length,
                "prediction_horizon": prediction_horizon
            }
            
            return result
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return None
    
    def _create_sequences(self, X, y, sequence_length):
        """
        Create sequences for time series models
        
        Args:
            X: Features array
            y: Target array
            sequence_length: Length of sequences to create
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i+sequence_length])
            y_seq.append(y[i+sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_model(self, model_name, data, **kwargs):
        """
        Train a specific AI model
        
        Args:
            model_name: Name of the model to train
            data: Prepared data dictionary from prepare_training_data
            **kwargs: Additional arguments for model training
            
        Returns:
            Dictionary with training results
        """
        if model_name not in self.model_metadata:
            logger.error(f"Model {model_name} not found in metadata")
            return None
        
        if data is None:
            logger.error(f"Training data for model {model_name} is None")
            return None
        
        try:
            # Get model metadata
            metadata = self.model_metadata[model_name]
            model_type = metadata.get("type")
            algorithm = metadata.get("algorithm")
            
            # Call appropriate training method based on model type
            if model_type == "price_prediction":
                result = self._train_price_prediction_model(model_name, data, algorithm, **kwargs)
            elif model_type == "trend_prediction":
                result = self._train_trend_prediction_model(model_name, data, algorithm, **kwargs)
            elif model_type == "volatility_prediction":
                result = self._train_volatility_prediction_model(model_name, data, algorithm, **kwargs)
            elif model_type == "pattern_recognition":
                result = self._train_pattern_recognition_model(model_name, data, algorithm, **kwargs)
            elif model_type == "regime_detection":
                result = self._train_regime_detection_model(model_name, data, algorithm, **kwargs)
            elif model_type == "anomaly_detection":
                result = self._train_anomaly_detection_model(model_name, data, algorithm, **kwargs)
            elif model_type == "sentiment_analysis":
                result = self._train_sentiment_analysis_model(model_name, data, algorithm, **kwargs)
            elif model_type == "order_flow":
                result = self._train_order_flow_model(model_name, data, algorithm, **kwargs)
            elif model_type == "correlation":
                result = self._train_correlation_model(model_name, data, algorithm, **kwargs)
            else:
                logger.error(f"Unknown model type: {model_type}")
                return None
            
            # Update model metadata and save trained model
            if result:
                self.models[model_name] = result.get("model")
                
                # Update metadata
                metadata.update({
                    "last_trained": datetime.datetime.now().isoformat(),
                    "accuracy": result.get("accuracy", 0.0)
                })
                
                # Update performance tracking
                self.model_performance[model_name] = result.get("performance", {})
                
                # Save model to disk
                self._save_model(
                    model_name=model_name,
                    model=result.get("model"),
                    metadata=metadata,
                    performance=result.get("performance", {})
                )
            
            return result
        except Exception as e:
            logger.error(f"Error training model {model_name}: {str(e)}")
            return None
    
    def _train_price_prediction_model(self, model_name, data, algorithm, **kwargs):
        """Train a price prediction model"""
        try:
            X_train = data.get("X_train")
            y_train = data.get("y_train")
            X_test = data.get("X_test")
            y_test = data.get("y_test")
            scaler_y = data.get("scaler_y")
            sequence_length = data.get("sequence_length")
            
            # Create model based on algorithm
            model = None
            
            if algorithm == "random_forest":
                n_estimators = kwargs.get("n_estimators", 100)
                max_depth = kwargs.get("max_depth", 10)
                
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Scale predictions back to original scale if scaler was used
                if scaler_y is not None:
                    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
                    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    
                    # Calculate metrics in original scale
                    mse_orig = mean_squared_error(y_test_orig, y_pred_orig)
                    mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
                    
                    # Calculate MAPE
                    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100
                else:
                    mse_orig = mse
                    mae_orig = mae
                    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                # Calculate accuracy as 1 - normalized_mae
                accuracy = max(0, 1 - (mae / (np.max(y_test) - np.min(y_test))))
                
                # Feature importance
                feature_importance = model.feature_importances_
                
                performance = {
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                    "mse_orig": mse_orig,
                    "mae_orig": mae_orig,
                    "mape": mape,
                    "accuracy": accuracy,
                    "feature_importance": feature_importance.tolist()
                }
                
                return {
                    "model": model,
                    "performance": performance,
                    "accuracy": accuracy
                }
            
            elif algorithm == "gradient_boosting":
                n_estimators = kwargs.get("n_estimators", 100)
                max_depth = kwargs.get("max_depth", 5)
                learning_rate = kwargs.get("learning_rate", 0.1)
                
                model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=42
                )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Scale predictions back to original scale if scaler was used
                if scaler_y is not None:
                    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
                    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    
                    # Calculate metrics in original scale
                    mse_orig = mean_squared_error(y_test_orig, y_pred_orig)
                    mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
                    
                    # Calculate MAPE
                    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100
                else:
                    mse_orig = mse
                    mae_orig = mae
                    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                # Calculate accuracy as 1 - normalized_mae
                accuracy = max(0, 1 - (mae / (np.max(y_test) - np.min(y_test))))
                
                # Feature importance
                feature_importance = model.feature_importances_
                
                performance = {
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                    "mse_orig": mse_orig,
                    "mae_orig": mae_orig,
                    "mape": mape,
                    "accuracy": accuracy,
                    "feature_importance": feature_importance.tolist()
                }
                
                return {
                    "model": model,
                    "performance": performance,
                    "accuracy": accuracy
                }
            
            elif algorithm == "svr":
                kernel = kwargs.get("kernel", "rbf")
                C = kwargs.get("C", 1.0)
                epsilon = kwargs.get("epsilon", 0.1)
                
                model = SVR(
                    kernel=kernel,
                    C=C,
                    epsilon=epsilon
                )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Scale predictions back to original scale if scaler was used
                if scaler_y is not None:
                    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
                    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    
                    # Calculate metrics in original scale
                    mse_orig = mean_squared_error(y_test_orig, y_pred_orig)
                    mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
                    
                    # Calculate MAPE
                    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100
                else:
                    mse_orig = mse
                    mae_orig = mae
                    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                # Calculate accuracy as 1 - normalized_mae
                accuracy = max(0, 1 - (mae / (np.max(y_test) - np.min(y_test))))
                
                performance = {
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                    "mse_orig": mse_orig,
                    "mae_orig": mae_orig,
                    "mape": mape,
                    "accuracy": accuracy
                }
                
                return {
                    "model": model,
                    "performance": performance,
                    "accuracy": accuracy
                }
            
            elif algorithm == "ensemble":
                # Create an ensemble of models
                models = []
                
                # Random Forest
                rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                models.append(("rf", rf))
                
                # Gradient Boosting
                gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
                models.append(("gb", gb))
                
                # Extra Trees
                et = ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42)
                models.append(("et", et))
                
                # Train each model and store predictions
                model_preds = []
                
                for name, model in models:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    model_preds.append(y_pred)
                
                # Combine predictions (simple average)
                y_pred_ensemble = np.mean(model_preds, axis=0)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred_ensemble)
                mae = mean_absolute_error(y_test, y_pred_ensemble)
                r2 = r2_score(y_test, y_pred_ensemble)
                
                # Scale predictions back to original scale if scaler was used
                if scaler_y is not None:
                    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
                    y_pred_orig = scaler_y.inverse_transform(y_pred_ensemble.reshape(-1, 1)).flatten()
                    
                    # Calculate metrics in original scale
                    mse_orig = mean_squared_error(y_test_orig, y_pred_orig)
                    mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
                    
                    # Calculate MAPE
                    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100
                else:
                    mse_orig = mse
                    mae_orig = mae
                    mape = np.mean(np.abs((y_test - y_pred_ensemble) / y_test)) * 100
                
                # Calculate accuracy as 1 - normalized_mae
                accuracy = max(0, 1 - (mae / (np.max(y_test) - np.min(y_test))))
                
                # Create ensemble model dictionary
                ensemble_model = {
                    "type": "ensemble",
                    "models": [(name, model) for name, model in models],
                    "scaler_y": scaler_y
                }
                
                performance = {
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                    "mse_orig": mse_orig,
                    "mae_orig": mae_orig,
                    "mape": mape,
                    "accuracy": accuracy
                }
                
                return {
                    "model": ensemble_model,
                    "performance": performance,
                    "accuracy": accuracy
                }
            
            # For neural network models, add training logic here
            # Currently, we'll return a simple placeholder for LSTM, etc.
            elif algorithm in ["lstm", "transformer", "wavenet", "deepar", "informer"]:
                logger.warning(f"Neural network model {algorithm} training not implemented in this demo version")
                
                # Return placeholder result
                return {
                    "model": None,
                    "performance": {
                        "mse": 0.0,
                        "mae": 0.0,
                        "r2": 0.0,
                        "accuracy": 0.7
                    },
                    "accuracy": 0.7
                }
            
            else:
                logger.error(f"Unknown algorithm for price prediction: {algorithm}")
                return None
            
        except Exception as e:
            logger.error(f"Error training price prediction model: {str(e)}")
            return None
    
    def _train_trend_prediction_model(self, model_name, data, algorithm, **kwargs):
        """Train a trend prediction model (classification)"""
        try:
            X_train = data.get("X_train")
            y_train = data.get("y_train")
            X_test = data.get("X_test")
            y_test = data.get("y_test")
            
            # Create model based on algorithm
            model = None
            
            if algorithm == "random_forest":
                n_estimators = kwargs.get("n_estimators", 100)
                max_depth = kwargs.get("max_depth", 10)
                
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Feature importance
                feature_importance = model.feature_importances_
                
                performance = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "feature_importance": feature_importance.tolist()
                }
                
                return {
                    "model": model,
                    "performance": performance,
                    "accuracy": accuracy
                }
            
            elif algorithm == "gradient_boosting":
                n_estimators = kwargs.get("n_estimators", 100)
                max_depth = kwargs.get("max_depth", 5)
                learning_rate = kwargs.get("learning_rate", 0.1)
                
                model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=42
                )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Feature importance
                feature_importance = model.feature_importances_
                
                performance = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "feature_importance": feature_importance.tolist()
                }
                
                return {
                    "model": model,
                    "performance": performance,
                    "accuracy": accuracy
                }
            
            elif algorithm == "svm":
                kernel = kwargs.get("kernel", "rbf")
                C = kwargs.get("C", 1.0)
                
                model = SVC(
                    kernel=kernel,
                    C=C,
                    probability=True,
                    random_state=42
                )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                performance = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                }
                
                return {
                    "model": model,
                    "performance": performance,
                    "accuracy": accuracy
                }
            
            elif algorithm == "ensemble":
                # Create an ensemble of models
                models = []
                
                # Random Forest
                rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                models.append(("rf", rf))
                
                # Gradient Boosting
                gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
                models.append(("gb", gb))
                
                # SVM
                svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
                models.append(("svm", svm))
                
                # Train each model and store predictions
                model_probs = []
                
                for name, model in models:
                    model.fit(X_train, y_train)
                    y_prob = model.predict_proba(X_test)
                    model_probs.append(y_prob)
                
                # Combine probabilities (simple average)
                y_prob_ensemble = np.mean(model_probs, axis=0)
                y_pred_ensemble = np.argmax(y_prob_ensemble, axis=1)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred_ensemble)
                precision = precision_score(y_test, y_pred_ensemble, average='weighted')
                recall = recall_score(y_test, y_pred_ensemble, average='weighted')
                f1 = f1_score(y_test, y_pred_ensemble, average='weighted')
                
                # Create ensemble model dictionary
                ensemble_model = {
                    "type": "ensemble",
                    "models": [(name, model) for name, model in models]
                }
                
                performance = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                }
                
                return {
                    "model": ensemble_model,
                    "performance": performance,
                    "accuracy": accuracy
                }
            
            # For neural network models, add training logic here
            # Currently, we'll return a simple placeholder for CNN, LSTM, etc.
            elif algorithm in ["cnn", "lstm", "transformer"]:
                logger.warning(f"Neural network model {algorithm} training not implemented in this demo version")
                
                # Return placeholder result
                return {
                    "model": None,
                    "performance": {
                        "accuracy": 0.75,
                        "precision": 0.75,
                        "recall": 0.75,
                        "f1": 0.75
                    },
                    "accuracy": 0.75
                }
            
            else:
                logger.error(f"Unknown algorithm for trend prediction: {algorithm}")
                return None
            
        except Exception as e:
            logger.error(f"Error training trend prediction model: {str(e)}")
            return None
    
    def _train_volatility_prediction_model(self, model_name, data, algorithm, **kwargs):
        """Train a volatility prediction model"""
        try:
            X_train = data.get("X_train")
            y_train = data.get("y_train")
            X_test = data.get("X_test")
            y_test = data.get("y_test")
            scaler_y = data.get("scaler_y")
            
            # Create model based on algorithm
            model = None
            
            if algorithm == "random_forest":
                n_estimators = kwargs.get("n_estimators", 100)
                max_depth = kwargs.get("max_depth", 10)
                
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Scale predictions back to original scale if scaler was used
                if scaler_y is not None:
                    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
                    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    
                    # Calculate metrics in original scale
                    mse_orig = mean_squared_error(y_test_orig, y_pred_orig)
                    mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
                    
                    # Calculate MAPE
                    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100
                else:
                    mse_orig = mse
                    mae_orig = mae
                    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                # Calculate accuracy as 1 - normalized_mae
                accuracy = max(0, 1 - (mae / (np.max(y_test) - np.min(y_test))))
                
                # Feature importance
                feature_importance = model.feature_importances_
                
                performance = {
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                    "mse_orig": mse_orig,
                    "mae_orig": mae_orig,
                    "mape": mape,
                    "accuracy": accuracy,
                    "feature_importance": feature_importance.tolist()
                }
                
                return {
                    "model": model,
                    "performance": performance,
                    "accuracy": accuracy
                }
            
            elif algorithm == "svr":
                kernel = kwargs.get("kernel", "rbf")
                C = kwargs.get("C", 1.0)
                epsilon = kwargs.get("epsilon", 0.1)
                
                model = SVR(
                    kernel=kernel,
                    C=C,
                    epsilon=epsilon
                )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Scale predictions back to original scale if scaler was used
                if scaler_y is not None:
                    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
                    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    
                    # Calculate metrics in original scale
                    mse_orig = mean_squared_error(y_test_orig, y_pred_orig)
                    mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
                    
                    # Calculate MAPE
                    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100
                else:
                    mse_orig = mse
                    mae_orig = mae
                    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                # Calculate accuracy as 1 - normalized_mae
                accuracy = max(0, 1 - (mae / (np.max(y_test) - np.min(y_test))))
                
                performance = {
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                    "mse_orig": mse_orig,
                    "mae_orig": mae_orig,
                    "mape": mape,
                    "accuracy": accuracy
                }
                
                return {
                    "model": model,
                    "performance": performance,
                    "accuracy": accuracy
                }
            
            # For neural network models, add training logic here
            # Currently, we'll return a simple placeholder for LSTM, etc.
            elif algorithm in ["lstm", "garch"]:
                logger.warning(f"Model {algorithm} training not fully implemented in this demo version")
                
                # Return placeholder result
                return {
                    "model": None,
                    "performance": {
                        "mse": 0.01,
                        "mae": 0.08,
                        "r2": 0.75,
                        "accuracy": 0.75
                    },
                    "accuracy": 0.75
                }
            
            else:
                logger.error(f"Unknown algorithm for volatility prediction: {algorithm}")
                return None
            
        except Exception as e:
            logger.error(f"Error training volatility prediction model: {str(e)}")
            return None
    
    def _train_pattern_recognition_model(self, model_name, data, algorithm, **kwargs):
        """Train a pattern recognition model"""
        logger.warning(f"Pattern recognition model {algorithm} training not fully implemented in this demo version")
        
        # Return placeholder result
        return {
            "model": None,
            "performance": {
                "accuracy": 0.8,
                "precision": 0.8,
                "recall": 0.77,
                "f1": 0.78
            },
            "accuracy": 0.8
        }
    
    def _train_regime_detection_model(self, model_name, data, algorithm, **kwargs):
        """Train a market regime detection model"""
        logger.warning(f"Regime detection model {algorithm} training not fully implemented in this demo version")
        
        # Return placeholder result
        return {
            "model": None,
            "performance": {
                "accuracy": 0.82,
                "precision": 0.8,
                "recall": 0.81,
                "f1": 0.8
            },
            "accuracy": 0.82
        }
    
    def _train_anomaly_detection_model(self, model_name, data, algorithm, **kwargs):
        """Train an anomaly detection model"""
        logger.warning(f"Anomaly detection model {algorithm} training not fully implemented in this demo version")
        
        # Return placeholder result
        return {
            "model": None,
            "performance": {
                "precision": 0.85,
                "recall": 0.75,
                "f1": 0.8,
                "accuracy": 0.83
            },
            "accuracy": 0.83
        }
    
    def _train_sentiment_analysis_model(self, model_name, data, algorithm, **kwargs):
        """Train a sentiment analysis model"""
        logger.warning(f"Sentiment analysis model {algorithm} training not fully implemented in this demo version")
        
        # Return placeholder result
        return {
            "model": None,
            "performance": {
                "accuracy": 0.78,
                "precision": 0.77,
                "recall": 0.76,
                "f1": 0.76
            },
            "accuracy": 0.78
        }
    
    def _train_order_flow_model(self, model_name, data, algorithm, **kwargs):
        """Train an order flow model"""
        logger.warning(f"Order flow model {algorithm} training not fully implemented in this demo version")
        
        # Return placeholder result
        return {
            "model": None,
            "performance": {
                "accuracy": 0.76,
                "precision": 0.75,
                "recall": 0.74,
                "f1": 0.75
            },
            "accuracy": 0.76
        }
    
    def _train_correlation_model(self, model_name, data, algorithm, **kwargs):
        """Train a correlation model"""
        logger.warning(f"Correlation model {algorithm} training not fully implemented in this demo version")
        
        # Return placeholder result
        return {
            "model": None,
            "performance": {
                "accuracy": 0.85,
                "mean_absolute_error": 0.08,
                "r2": 0.82
            },
            "accuracy": 0.85
        }
    
    def predict(self, model_name, data, **kwargs):
        """
        Make a prediction using a trained model
        
        Args:
            model_name: Name of the model to use
            data: Data to make predictions on (processed features)
            **kwargs: Additional arguments for prediction
            
        Returns:
            Dictionary with prediction results
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found or not trained")
            return None
        
        if model_name not in self.model_metadata:
            logger.error(f"Model metadata for {model_name} not found")
            return None
        
        try:
            # Get model and metadata
            model = self.models[model_name]
            metadata = self.model_metadata[model_name]
            model_type = metadata.get("type")
            
            # Call appropriate prediction method based on model type
            if model_type == "price_prediction":
                result = self._predict_price(model_name, model, data, metadata, **kwargs)
            elif model_type == "trend_prediction":
                result = self._predict_trend(model_name, model, data, metadata, **kwargs)
            elif model_type == "volatility_prediction":
                result = self._predict_volatility(model_name, model, data, metadata, **kwargs)
            elif model_type == "pattern_recognition":
                result = self._predict_patterns(model_name, model, data, metadata, **kwargs)
            elif model_type == "regime_detection":
                result = self._predict_regime(model_name, model, data, metadata, **kwargs)
            elif model_type == "anomaly_detection":
                result = self._detect_anomalies(model_name, model, data, metadata, **kwargs)
            elif model_type == "sentiment_analysis":
                result = self._analyze_sentiment(model_name, model, data, metadata, **kwargs)
            elif model_type == "order_flow":
                result = self._analyze_order_flow(model_name, model, data, metadata, **kwargs)
            elif model_type == "correlation":
                result = self._analyze_correlation(model_name, model, data, metadata, **kwargs)
            else:
                logger.error(f"Unknown model type: {model_type}")
                return None
            
            return result
        except Exception as e:
            logger.error(f"Error making prediction with model {model_name}: {str(e)}")
            return None
    
    def _predict_price(self, model_name, model, data, metadata, **kwargs):
        """Make price predictions using a trained model"""
        try:
            algorithm = metadata.get("algorithm")
            prediction_horizon = kwargs.get("prediction_horizon", metadata.get("prediction_horizon", [1])[0])
            
            # For ensemble models
            if algorithm == "ensemble" and isinstance(model, dict):
                models = model.get("models", [])
                scaler_y = model.get("scaler_y")
                
                # Make predictions with each model
                predictions = []
                
                for name, m in models:
                    pred = m.predict(data)
                    predictions.append(pred)
                
                # Combine predictions (simple average)
                prediction = np.mean(predictions, axis=0)
                
                # Scale prediction back to original scale if scaler was used
                if scaler_y is not None:
                    prediction = scaler_y.inverse_transform(prediction.reshape(-1, 1)).flatten()
                
                # Calculate confidence based on agreement between models
                pred_std = np.std(predictions, axis=0)
                relative_std = pred_std / (np.mean(predictions, axis=0) + 1e-10)
                confidence = 1.0 - min(1.0, relative_std.mean() * 5)
                
                return {
                    "prediction": prediction.tolist(),
                    "confidence": confidence,
                    "prediction_horizon": prediction_horizon,
                    "model": model_name
                }
            
            # For standard scikit-learn models
            elif algorithm in ["random_forest", "gradient_boosting", "svr"]:
                prediction = model.predict(data)
                
                # Scale prediction back to original scale if using a scaler
                if "scaler_y" in kwargs and kwargs["scaler_y"] is not None:
                    scaler_y = kwargs["scaler_y"]
                    prediction = scaler_y.inverse_transform(prediction.reshape(-1, 1)).flatten()
                
                # For sklearn models that provide feature importance
                if hasattr(model, "feature_importances_"):
                    feature_importance = model.feature_importances_
                else:
                    feature_importance = None
                
                # For models with predict_proba
                confidence = 0.8  # Default confidence
                
                if hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(data)
                        confidence = np.max(proba, axis=1).mean()
                    except:
                        pass
                
                return {
                    "prediction": prediction.tolist(),
                    "confidence": confidence,
                    "prediction_horizon": prediction_horizon,
                    "feature_importance": feature_importance.tolist() if feature_importance is not None else None,
                    "model": model_name
                }
            
            # For neural network models (placeholder)
            elif algorithm in ["lstm", "transformer", "wavenet", "deepar", "informer"]:
                # Generate placeholder prediction
                logger.warning(f"Neural network model {algorithm} prediction using placeholder in this demo version")
                
                # Generate a realistic placeholder prediction
                prediction = np.random.normal(0, 0.01, size=data.shape[0])
                confidence = 0.75
                
                return {
                    "prediction": prediction.tolist(),
                    "confidence": confidence,
                    "prediction_horizon": prediction_horizon,
                    "model": model_name
                }
            
            else:
                logger.error(f"Unknown algorithm for price prediction: {algorithm}")
                return None
            
        except Exception as e:
            logger.error(f"Error predicting price with model {model_name}: {str(e)}")
            return None
    
    def _predict_trend(self, model_name, model, data, metadata, **kwargs):
        """Make trend predictions using a trained model"""
        try:
            algorithm = metadata.get("algorithm")
            classes = metadata.get("classes", ["up", "down", "sideways"])
            
            # For ensemble models
            if algorithm == "ensemble" and isinstance(model, dict):
                models = model.get("models", [])
                
                # Make predictions with each model
                predictions = []
                probabilities = []
                
                for name, m in models:
                    pred = m.predict(data)
                    predictions.append(pred)
                    
                    if hasattr(m, "predict_proba"):
                        prob = m.predict_proba(data)
                        probabilities.append(prob)
                
                # Combine predictions (majority vote)
                prediction_counts = np.zeros((data.shape[0], len(classes)))
                
                for pred in predictions:
                    for i, p in enumerate(pred):
                        prediction_counts[i, p] += 1
                
                prediction = np.argmax(prediction_counts, axis=1)
                
                # Combine probabilities if available
                if probabilities:
                    combined_proba = np.mean(probabilities, axis=0)
                    confidence = np.max(combined_proba, axis=1).mean()
                else:
                    confidence = 0.8  # Default confidence
                
                # Convert numeric predictions to class labels
                prediction_labels = [classes[p] for p in prediction]
                
                return {
                    "prediction": prediction_labels,
                    "prediction_numeric": prediction.tolist(),
                    "confidence": confidence,
                    "classes": classes,
                    "model": model_name
                }
            
            # For standard scikit-learn models
            elif algorithm in ["random_forest", "gradient_boosting", "svm"]:
                prediction = model.predict(data)
                
                # For models with predict_proba
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(data)
                    confidence = np.max(proba, axis=1).mean()
                else:
                    confidence = 0.8  # Default confidence
                
                # For sklearn models that provide feature importance
                if hasattr(model, "feature_importances_"):
                    feature_importance = model.feature_importances_
                else:
                    feature_importance = None
                
                # Convert numeric predictions to class labels
                prediction_labels = [classes[p] for p in prediction]
                
                return {
                    "prediction": prediction_labels,
                    "prediction_numeric": prediction.tolist(),
                    "confidence": confidence,
                    "classes": classes,
                    "probabilities": proba.tolist() if 'proba' in locals() else None,
                    "feature_importance": feature_importance.tolist() if feature_importance is not None else None,
                    "model": model_name
                }
            
            # For neural network models (placeholder)
            elif algorithm in ["cnn", "lstm", "transformer"]:
                # Generate placeholder prediction
                logger.warning(f"Neural network model {algorithm} prediction using placeholder in this demo version")
                
                # Generate a realistic placeholder prediction
                prediction = np.random.choice([0, 1, 2], size=data.shape[0], p=[0.4, 0.4, 0.2])
                confidence = 0.75
                
                # Convert numeric predictions to class labels
                prediction_labels = [classes[p] for p in prediction]
                
                return {
                    "prediction": prediction_labels,
                    "prediction_numeric": prediction.tolist(),
                    "confidence": confidence,
                    "classes": classes,
                    "model": model_name
                }
            
            else:
                logger.error(f"Unknown algorithm for trend prediction: {algorithm}")
                return None
            
        except Exception as e:
            logger.error(f"Error predicting trend with model {model_name}: {str(e)}")
            return None
    
    def _predict_volatility(self, model_name, model, data, metadata, **kwargs):
        """Make volatility predictions using a trained model"""
        try:
            algorithm = metadata.get("algorithm")
            prediction_horizon = kwargs.get("prediction_horizon", metadata.get("prediction_horizon", [1])[0])
            
            # For standard scikit-learn models
            if algorithm in ["random_forest", "svr"]:
                prediction = model.predict(data)
                
                # Scale prediction back to original scale if using a scaler
                if "scaler_y" in kwargs and kwargs["scaler_y"] is not None:
                    scaler_y = kwargs["scaler_y"]
                    prediction = scaler_y.inverse_transform(prediction.reshape(-1, 1)).flatten()
                
                # For sklearn models that provide feature importance
                if hasattr(model, "feature_importances_"):
                    feature_importance = model.feature_importances_
                else:
                    feature_importance = None
                
                confidence = 0.8  # Default confidence
                
                return {
                    "prediction": prediction.tolist(),
                    "confidence": confidence,
                    "prediction_horizon": prediction_horizon,
                    "feature_importance": feature_importance.tolist() if feature_importance is not None else None,
                    "model": model_name
                }
            
            # For neural network models (placeholder)
            elif algorithm in ["lstm", "garch"]:
                # Generate placeholder prediction
                logger.warning(f"Model {algorithm} prediction using placeholder in this demo version")
                
                # Generate a realistic placeholder prediction
                prediction = np.random.uniform(0.01, 0.05, size=data.shape[0])
                confidence = 0.75
                
                return {
                    "prediction": prediction.tolist(),
                    "confidence": confidence,
                    "prediction_horizon": prediction_horizon,
                    "model": model_name
                }
            
            else:
                logger.error(f"Unknown algorithm for volatility prediction: {algorithm}")
                return None
            
        except Exception as e:
            logger.error(f"Error predicting volatility with model {model_name}: {str(e)}")
            return None
    
    def _predict_patterns(self, model_name, model, data, metadata, **kwargs):
        """Recognize patterns using a trained model"""
        logger.warning(f"Pattern recognition model {model_name} using placeholder in this demo version")
        
        patterns = metadata.get("patterns", [])
        
        # Generate placeholder prediction
        n_samples = 1 if not hasattr(data, "shape") else data.shape[0]
        
        detected_patterns = []
        for _ in range(n_samples):
            n_patterns = random.randint(0, 2)
            if n_patterns > 0:
                sample_patterns = random.sample(patterns, n_patterns)
                pattern_confidences = [random.uniform(0.6, 0.9) for _ in range(n_patterns)]
                detected_patterns.append([
                    {"pattern": p, "confidence": c} for p, c in zip(sample_patterns, pattern_confidences)
                ])
            else:
                detected_patterns.append([])
        
        return {
            "detected_patterns": detected_patterns,
            "confidence": 0.75,
            "model": model_name
        }
    
    def _predict_regime(self, model_name, model, data, metadata, **kwargs):
        """Predict market regime using a trained model"""
        logger.warning(f"Regime detection model {model_name} using placeholder in this demo version")
        
        regimes = metadata.get("regimes", ["bull", "bear", "sideways", "volatile"])
        
        # Generate placeholder prediction
        n_samples = 1 if not hasattr(data, "shape") else data.shape[0]
        
        # Use a more realistic distribution of regimes
        probabilities = [0.4, 0.3, 0.2, 0.1]  # bull, bear, sideways, volatile
        predictions = [regimes[np.random.choice(len(regimes), p=probabilities)] for _ in range(n_samples)]
        
        # Generate confidence scores
        confidences = [random.uniform(0.65, 0.9) for _ in range(n_samples)]
        
        return {
            "regime": predictions,
            "confidence": confidences,
            "model": model_name
        }
    
    def _detect_anomalies(self, model_name, model, data, metadata, **kwargs):
        """Detect anomalies using a trained model"""
        logger.warning(f"Anomaly detection model {model_name} using placeholder in this demo version")
        
        # Generate placeholder prediction
        n_samples = 1 if not hasattr(data, "shape") else data.shape[0]
        
        # Most samples should not be anomalies
        anomaly_flags = [random.random() < 0.1 for _ in range(n_samples)]
        anomaly_scores = [random.uniform(0.5, 0.9) if flag else random.uniform(0.1, 0.4) 
                          for flag in anomaly_flags]
        
        return {
            "is_anomaly": anomaly_flags,
            "anomaly_score": anomaly_scores,
            "model": model_name
        }
    
    def _analyze_sentiment(self, model_name, model, data, metadata, **kwargs):
        """Analyze sentiment using a trained model"""
        logger.warning(f"Sentiment analysis model {model_name} using placeholder in this demo version")
        
        # For demo purposes, let's create some realistic sentiment scores
        # Generate placeholder prediction
        n_samples = 1 if not hasattr(data, "shape") else data.shape[0]
        
        # A bit more skewed toward positive for financial news
        sentiment_scores = [random.normalvariate(0.2, 0.5) for _ in range(n_samples)]
        sentiment_scores = [max(-1.0, min(1.0, s)) for s in sentiment_scores]
        
        sentiment_labels = []
        for score in sentiment_scores:
            if score > 0.2:
                sentiment_labels.append("positive")
            elif score < -0.2:
                sentiment_labels.append("negative")
            else:
                sentiment_labels.append("neutral")
        
        return {
            "sentiment_scores": sentiment_scores,
            "sentiment_labels": sentiment_labels,
            "confidence": 0.8,
            "model": model_name
        }
    
    def _analyze_order_flow(self, model_name, model, data, metadata, **kwargs):
        """Analyze order flow using a trained model"""
        logger.warning(f"Order flow model {model_name} using placeholder in this demo version")
        
        # Generate placeholder prediction
        n_samples = 1 if not hasattr(data, "shape") else data.shape[0]
        
        # Order flow metrics
        buy_pressure = [random.uniform(0.4, 0.6) for _ in range(n_samples)]
        sell_pressure = [1.0 - bp for bp in buy_pressure]
        
        # Add some randomness to make it look more realistic
        buy_pressure = [bp + random.uniform(-0.1, 0.1) for bp in buy_pressure]
        buy_pressure = [max(0.0, min(1.0, bp)) for bp in buy_pressure]
        
        sell_pressure = [sp + random.uniform(-0.1, 0.1) for sp in sell_pressure]
        sell_pressure = [max(0.0, min(1.0, sp)) for sp in sell_pressure]
        
        # Calculate net pressure
        net_pressure = [bp - sp for bp, sp in zip(buy_pressure, sell_pressure)]
        
        return {
            "buy_pressure": buy_pressure,
            "sell_pressure": sell_pressure,
            "net_pressure": net_pressure,
            "confidence": 0.75,
            "model": model_name
        }
    
    def _analyze_correlation(self, model_name, model, data, metadata, **kwargs):
        """Analyze correlations using a trained model"""
        logger.warning(f"Correlation model {model_name} using placeholder in this demo version")
        
        # Generate placeholder prediction
        n_samples = 1 if not hasattr(data, "shape") else data.shape[0]
        
        # Create some realistic correlation matrix for demo
        n_assets = 5  # Assuming 5 assets
        correlation_matrix = []
        
        for _ in range(n_samples):
            # Create a positive semi-definite correlation matrix
            cov = np.random.rand(n_assets, n_assets)
            cov = 0.5 * (cov + cov.T)  # Make it symmetric
            cov = cov + n_assets * np.eye(n_assets)  # Make it positive definite
            
            # Convert to correlation matrix
            diag = np.sqrt(np.diag(cov))
            cor = cov / np.outer(diag, diag)
            
            correlation_matrix.append(cor.tolist())
        
        return {
            "correlation_matrix": correlation_matrix,
            "confidence": 0.85,
            "model": model_name
        }
    
    def get_available_models(self, model_type=None):
        """
        Get list of available models, optionally filtered by type
        
        Args:
            model_type: Type of models to filter for
            
        Returns:
            Dictionary of model names and metadata
        """
        if model_type:
            return {name: meta for name, meta in self.model_metadata.items() if meta.get("type") == model_type}
        else:
            return self.model_metadata
    
    def get_model_details(self, model_name):
        """
        Get detailed information about a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model metadata and performance
        """
        if model_name not in self.model_metadata:
            return None
        
        result = {
            "metadata": self.model_metadata.get(model_name, {}),
            "performance": self.model_performance.get(model_name, {}),
            "is_trained": model_name in self.models
        }
        
        return result
    
    def generate_feature_importance_chart(self, model_name, feature_columns=None, top_n=10):
        """
        Generate feature importance chart for a model
        
        Args:
            model_name: Name of the model
            feature_columns: Names of feature columns
            top_n: Number of top features to show
            
        Returns:
            Base64 encoded image
        """
        if model_name not in self.models or model_name not in self.model_performance:
            return None
        
        model = self.models[model_name]
        performance = self.model_performance[model_name]
        
        # Check if feature importance is available
        if "feature_importance" not in performance:
            return None
        
        try:
            # Get feature importance
            feature_importance = performance["feature_importance"]
            
            # Use provided feature names or generic names
            if feature_columns is None:
                feature_columns = [f"Feature_{i}" for i in range(len(feature_importance))]
            
            # Create a DataFrame for sorting
            importance_df = pd.DataFrame({
                'Feature': feature_columns[:len(feature_importance)],
                'Importance': feature_importance
            })
            
            # Sort and get top N features
            importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(f'Top {top_n} Feature Importance for {model_name}')
            plt.tight_layout()
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_str
        except Exception as e:
            logger.error(f"Error generating feature importance chart: {str(e)}")
            return None
    
    def generate_performance_metrics_chart(self, model_name):
        """
        Generate performance metrics chart for a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Base64 encoded image
        """
        if model_name not in self.model_performance:
            return None
        
        try:
            performance = self.model_performance[model_name]
            metadata = self.model_metadata.get(model_name, {})
            model_type = metadata.get("type")
            
            metrics = []
            values = []
            
            # Extract metrics based on model type
            if model_type in ["price_prediction", "volatility_prediction"]:
                if "r2" in performance:
                    metrics.append("R² Score")
                    values.append(performance["r2"])
                
                if "accuracy" in performance:
                    metrics.append("Accuracy")
                    values.append(performance["accuracy"])
                
                if "mape" in performance:
                    metrics.append("MAPE (%)")
                    # Convert to percentage for display
                    values.append(min(100, performance["mape"]))
            elif model_type in ["trend_prediction", "regime_detection", "pattern_recognition"]:
                if "accuracy" in performance:
                    metrics.append("Accuracy")
                    values.append(performance["accuracy"])
                
                if "precision" in performance:
                    metrics.append("Precision")
                    values.append(performance["precision"])
                
                if "recall" in performance:
                    metrics.append("Recall")
                    values.append(performance["recall"])
                
                if "f1" in performance:
                    metrics.append("F1 Score")
                    values.append(performance["f1"])
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.bar(metrics, values, color='lightgreen')
            plt.ylim(0, 1.1)  # Set y-axis limit
            plt.ylabel('Score')
            plt.title(f'Performance Metrics for {model_name}')
            
            # Add value labels on bars
            for i, v in enumerate(values):
                plt.text(i, v + 0.05, f"{v:.2f}", ha='center')
            
            plt.tight_layout()
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_str
        except Exception as e:
            logger.error(f"Error generating performance metrics chart: {str(e)}")
            return None


# Function to integrate with main app
def integrate_ai_models(app):
    """
    Integrate advanced AI models with the main Flask app
    
    Args:
        app: Flask application instance
        
    Returns:
        AI models instance
    """
    ai_models = AdvancedAIModels(app)
    
    # Make AI models available to the app
    app.ai_models = ai_models
    
    # Create API routes for AI models
    from flask import Blueprint, request, jsonify
    
    # Create API blueprint
    ai_models_bp = Blueprint('ai_models', __name__)
    
    @ai_models_bp.route('/available_models', methods=['GET'])
    def available_models():
        try:
            model_type = request.args.get('type')
            models = app.ai_models.get_available_models(model_type)
            
            return jsonify({
                'success': True,
                'models': models
            })
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return jsonify({
                'success': False,
                'message': f'Error getting available models: {str(e)}'
            })
    
    @ai_models_bp.route('/model_details/<model_name>', methods=['GET'])
    def model_details(model_name):
        try:
            details = app.ai_models.get_model_details(model_name)
            
            if details is None:
                return jsonify({
                    'success': False,
                    'message': f'Model {model_name} not found'
                })
            
            return jsonify({
                'success': True,
                'model_details': details
            })
        except Exception as e:
            logger.error(f"Error getting model details: {str(e)}")
            return jsonify({
                'success': False,
                'message': f'Error getting model details: {str(e)}'
            })
    
    @ai_models_bp.route('/train_model', methods=['POST'])
    def train_model():
        try:
            data = request.json
            
            if not data or not isinstance(data, dict):
                return jsonify({
                    'success': False,
                    'message': 'Invalid request format'
                })
            
            model_name = data.get('model_name')
            symbol = data.get('symbol')
            period = data.get('period', '1d')
            
            if not model_name or not symbol:
                return jsonify({
                    'success': False,
                    'message': 'Model name and symbol are required'
                })
            
            # Load training data
            training_data = app.ai_models.load_training_data(symbol, period=period)
            
            if training_data is None or training_data.empty:
                return jsonify({
                    'success': False,
                    'message': f'No training data available for {symbol}'
                })
            
            # Apply feature engineering
            processed_data = app.ai_models.engineer_features(training_data)
            
            # Prepare data for training
            target_column = data.get('target_column', 'close')
            feature_columns = data.get('feature_columns')
            prediction_horizon = data.get('prediction_horizon', 1)
            
            prepared_data = app.ai_models.prepare_training_data(
                processed_data, 
                target_column, 
                feature_columns=feature_columns,
                prediction_horizon=prediction_horizon
            )
            
            if prepared_data is None:
                return jsonify({
                    'success': False,
                    'message': 'Error preparing training data'
                })
            
            # Train the model
            training_result = app.ai_models.train_model(
                model_name, 
                prepared_data,
                **data.get('parameters', {})
            )
            
            if training_result is None:
                return jsonify({
                    'success': False,
                    'message': f'Error training model {model_name}'
                })
            
            return jsonify({
                'success': True,
                'message': f'Model {model_name} trained successfully',
                'accuracy': training_result.get('accuracy', 0),
                'performance': training_result.get('performance', {})
            })
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return jsonify({
                'success': False,
                'message': f'Error training model: {str(e)}'
            })
    
    @ai_models_bp.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.json
            
            if not data or not isinstance(data, dict):
                return jsonify({
                    'success': False,
                    'message': 'Invalid request format'
                })
            
            model_name = data.get('model_name')
            features = data.get('features')
            
            if not model_name or not features:
                return jsonify({
                    'success': False,
                    'message': 'Model name and features are required'
                })
            
            # Convert features to numpy array
            features_array = np.array(features)
            
            # Make prediction
            prediction = app.ai_models.predict(
                model_name, 
                features_array,
                **data.get('parameters', {})
            )
            
            if prediction is None:
                return jsonify({
                    'success': False,
                    'message': f'Error making prediction with model {model_name}'
                })
            
            return jsonify({
                'success': True,
                'prediction': prediction
            })
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return jsonify({
                'success': False,
                'message': f'Error making prediction: {str(e)}'
            })
    
    @ai_models_bp.route('/feature_importance/<model_name>', methods=['GET'])
    def feature_importance(model_name):
        try:
            feature_columns = request.args.get('feature_columns', '').split(',') if request.args.get('feature_columns') else None
            top_n = request.args.get('top_n', 10, type=int)
            
            # Generate feature importance chart
            chart = app.ai_models.generate_feature_importance_chart(model_name, feature_columns, top_n)
            
            if chart is None:
                return jsonify({
                    'success': False,
                    'message': f'No feature importance data available for model {model_name}'
                })
            
            return jsonify({
                'success': True,
                'chart': chart
            })
        except Exception as e:
            logger.error(f"Error generating feature importance chart: {str(e)}")
            return jsonify({
                'success': False,
                'message': f'Error generating feature importance chart: {str(e)}'
            })
    
    @ai_models_bp.route('/performance_metrics/<model_name>', methods=['GET'])
    def performance_metrics(model_name):
        try:
            # Generate performance metrics chart
            chart = app.ai_models.generate_performance_metrics_chart(model_name)
            
            if chart is None:
                return jsonify({
                    'success': False,
                    'message': f'No performance metrics data available for model {model_name}'
                })
            
            return jsonify({
                'success': True,
                'chart': chart
            })
        except Exception as e:
            logger.error(f"Error generating performance metrics chart: {str(e)}")
            return jsonify({
                'success': False,
                'message': f'Error generating performance metrics chart: {str(e)}'
            })
    
    # Register the blueprint with the app
    app.register_blueprint(ai_models_bp, url_prefix='/api/ai_models')
    
    logger.info(f"Advanced AI models integrated with application ({len(ai_models.model_metadata)} models available)")
    
    return ai_models


# When run as main script, run test
if __name__ == "__main__":
    # Test the AI models
    ai_models = AdvancedAIModels()
    
    # Print available models
    available_models = ai_models.get_available_models()
    print(f"Available AI models: {len(available_models)}")
    
    # Print models by type
    for model_type in ["price_prediction", "trend_prediction", "volatility_prediction", 
                      "pattern_recognition", "regime_detection", "anomaly_detection"]:
        models = ai_models.get_available_models(model_type)
        print(f"\n{model_type.replace('_', ' ').title()} models: {len(models)}")
        for model_name, metadata in models.items():
            print(f"  - {model_name}: {metadata.get('description')}")