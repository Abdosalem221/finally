import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random  # For simulation when actual models aren't available
import os
import json
import math
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
import joblib
from app import db
from models import Currency, MarketData

# Initialize logger
logger = logging.getLogger(__name__)

# Constants for AI models
FEATURE_ENGINEERING_METHODS = [
    "technical_indicators", "statistical_features", "price_momentum", 
    "volatility_metrics", "trend_features", "pattern_recognition",
    "market_regime", "time_decomposition", "correlation_features",
    "relative_strength"
]

MODEL_TYPES = {
    "price_prediction": ["regression", "classification", "hybrid"],
    "trend_prediction": ["classification", "ensemble"],
    "volatility_prediction": ["regression", "garch"],
    "risk_assessment": ["classification", "regression"],
    "anomaly_detection": ["isolation_forest", "one_class_svm"],
    "market_regime": ["hmm", "clustering", "classification"],
    "pattern_recognition": ["cnn", "rnn", "hybrid"],
    "portfolio_optimization": ["optimization", "reinforcement"],
    "sentiment_analysis": ["nlp", "ensemble"]
}

TIMEFRAMES = {
    "1m": {"minutes": 1, "limit": 1000},
    "2m": {"minutes": 2, "limit": 800},
    "5m": {"minutes": 5, "limit": 600},
    "10m": {"minutes": 10, "limit": 500},
    "15m": {"minutes": 15, "limit": 400},
    "30m": {"minutes": 30, "limit": 300},
    "1h": {"minutes": 60, "limit": 200},
    "1d": {"minutes": 1440, "limit": 100}
}

class AIAnalysis:
    def __init__(self):
        """Initialize the enhanced AI Analysis system with high precision capabilities"""
        self.models = {}
        self.cached_predictions = {}
        self.precision_threshold = 0.85  # زيادة دقة التحليل
        self.confidence_minimum = 0.75   # زيادة الحد الأدنى للثقة
        self.cached_time = {}
        self.timeframes = TIMEFRAMES.copy()
        self.feature_importance = {}
        self.model_performance = {}
        self.anomaly_thresholds = {}
        self.regime_states = {}
        
        # Scalers for data normalization
        self.scalers = {
            "minmax": MinMaxScaler(),
            "standard": StandardScaler()
        }
        
        # Initialize model storage
        self.model_registry = {
            "price_prediction": {},
            "trend_prediction": {},
            "volatility_prediction": {},
            "risk_assessment": {},
            "anomaly_detection": {},
            "market_regime": {},
            "pattern_recognition": {},
            "portfolio_optimization": {},
            "sentiment_analysis": {}
        }
        
        # Initialize prediction ensemble weights
        self.ensemble_weights = {}
        
        # Set up model storage directory
        self.model_dir = os.path.join(os.getcwd(), 'ai_models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load existing models if available
        self._load_models()
    
    def _load_models(self):
        """Load trained models from disk"""
        try:
            model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.joblib')]
            
            for model_file in model_files:
                try:
                    model_path = os.path.join(self.model_dir, model_file)
                    # Extract model info from filename
                    parts = model_file.replace('.joblib', '').split('_')
                    if len(parts) >= 3:
                        model_type = parts[0]
                        algorithm = parts[1]
                        symbol = '_'.join(parts[2:])
                        
                        # Load the model
                        model_data = joblib.load(model_path)
                        
                        # Store in registry
                        if model_type in self.model_registry:
                            if symbol not in self.model_registry[model_type]:
                                self.model_registry[model_type][symbol] = {}
                            self.model_registry[model_type][symbol][algorithm] = model_data
                            
                            logger.info(f"Loaded {model_type} model ({algorithm}) for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading model {model_file}: {str(e)}")
        except Exception as e:
            logger.error(f"Error accessing model directory: {str(e)}")
    
    def _save_model(self, model_data, model_type, algorithm, symbol):
        """Save a trained model to disk"""
        try:
            filename = f"{model_type}_{algorithm}_{symbol}.joblib"
            model_path = os.path.join(self.model_dir, filename)
            joblib.dump(model_data, model_path)
            logger.info(f"Saved {model_type} model ({algorithm}) for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def get_training_data(self, symbol, period='1d', limit=1000):
        """
        Get enhanced historical data with advanced feature engineering for AI model training
        """
        try:
            # Check if we need fresh data
            cache_key = f"{symbol}_{period}_{limit}_training"
            now = datetime.utcnow()
            
            # Query database for historical data
            currency = Currency.query.filter_by(symbol=symbol).first()
            
            if not currency:
                logger.error(f"Currency {symbol} not found in database")
                return None
            
            # Get market data
            market_data = MarketData.query.filter_by(currency_id=currency.id).order_by(
                MarketData.timestamp.desc()
            ).limit(limit).all()
            
            if not market_data or len(market_data) < 100:  # Need sufficient data
                logger.warning(f"Insufficient data for AI training: {symbol}")
                return None
            
            # Convert to pandas DataFrame
            data = pd.DataFrame([{
                'timestamp': md.timestamp,
                'open': md.open_price,
                'high': md.high_price,
                'low': md.low_price,
                'close': md.close_price,
                'volume': md.volume
            } for md in market_data])
            
            # Sort by timestamp
            data = data.sort_values('timestamp')
            
            # Resample to the requested timeframe if necessary
            if period in self.timeframes and period != '1d':
                # Resample data to the requested timeframe
                minutes = self.timeframes[period]['minutes']
                data = self._resample_data(data, f'{minutes}min')
            
            # Add 50+ engineered features for AI models
            data = self._engineer_features(data, symbol)
            
            # Handle missing values
            data = self._handle_missing_values(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting training data for {symbol}: {str(e)}")
            return None
    
    def _resample_data(self, data, timeframe):
        """Resample OHLCV data to a different timeframe"""
        try:
            # Set timestamp as index
            df = data.set_index('timestamp')
            
            # Resample the data
            resampled = df.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # Reset index
            resampled = resampled.reset_index()
            
            return resampled
        except Exception as e:
            logger.error(f"Error resampling data to {timeframe}: {str(e)}")
            return data  # Return original data if resampling fails
    
    def _engineer_features(self, data, symbol):
        """
        Create 50+ engineered features for AI model training, including:
        - Price action features
        - Technical indicators
        - Statistical features
        - Volatility metrics
        - Cyclical features
        - Sentiment indicators (synthetic for demo)
        - Pattern recognition metrics
        """
        try:
            df = data.copy()
            
            # --- PRICE ACTION FEATURES ---
            
            # 1. Basic Returns
            df['return_1d'] = df['close'].pct_change(1)
            df['return_2d'] = df['close'].pct_change(2)
            df['return_3d'] = df['close'].pct_change(3)
            df['return_5d'] = df['close'].pct_change(5)
            df['return_10d'] = df['close'].pct_change(10)
            df['return_20d'] = df['close'].pct_change(20)
            
            # 2. Logarithmic Returns
            df['log_return_1d'] = np.log(df['close'] / df['close'].shift(1))
            df['log_return_5d'] = np.log(df['close'] / df['close'].shift(5))
            
            # 3. Normalized Price
            df['norm_price_5d'] = df['close'] / df['close'].rolling(window=5).mean()
            df['norm_price_20d'] = df['close'] / df['close'].rolling(window=20).mean()
            
            # 4. Candle Size Metrics
            df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
            df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'])
            df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'])
            
            # 5. Price Distance Features
            df['dist_from_high_10d'] = df['close'] / df['high'].rolling(window=10).max()
            df['dist_from_low_10d'] = df['close'] / df['low'].rolling(window=10).min()
            df['dist_from_high_20d'] = df['close'] / df['high'].rolling(window=20).max()
            df['dist_from_low_20d'] = df['close'] / df['low'].rolling(window=20).min()
            
            # --- TECHNICAL INDICATORS ---
            
            # 6. Moving Averages
            for window in [5, 10, 20, 50, 100, 200]:
                df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
                df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
                
                # 7. MA ratios
                df[f'close_sma_ratio_{window}'] = df['close'] / df[f'sma_{window}']
                df[f'close_ema_ratio_{window}'] = df['close'] / df[f'ema_{window}']
            
            # 8. Moving Average Crossovers
            df['sma_5_10_ratio'] = df['sma_5'] / df['sma_10']
            df['sma_10_20_ratio'] = df['sma_10'] / df['sma_20']
            df['sma_20_50_ratio'] = df['sma_20'] / df['sma_50']
            df['sma_50_200_ratio'] = df['sma_50'] / df['sma_200']
            
            # 9. MACD
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # 10. RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain_14 = gain.rolling(window=14).mean()
            avg_loss_14 = loss.rolling(window=14).mean()
            
            rs_14 = avg_gain_14 / avg_loss_14
            df['rsi_14'] = 100 - (100 / (1 + rs_14))
            
            # 11. Stochastic Oscillator
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # 12. Bollinger Bands
            df['bb_middle_20'] = df['close'].rolling(window=20).mean()
            bb_std_20 = df['close'].rolling(window=20).std()
            df['bb_upper_20'] = df['bb_middle_20'] + (bb_std_20 * 2)
            df['bb_lower_20'] = df['bb_middle_20'] - (bb_std_20 * 2)
            df['bb_width_20'] = (df['bb_upper_20'] - df['bb_lower_20']) / df['bb_middle_20']
            df['bb_pct_b_20'] = (df['close'] - df['bb_lower_20']) / (df['bb_upper_20'] - df['bb_lower_20'])
            
            # --- VOLATILITY METRICS ---
            
            # 13. Historical Volatility
            for window in [5, 10, 20, 30]:
                df[f'volatility_{window}d'] = df['log_return_1d'].rolling(window=window).std() * np.sqrt(252)
            
            # 14. Normalized Volatility
            df['rel_vol_5_20'] = df['volatility_5d'] / df['volatility_20d']
            
            # 15. Average True Range (ATR)
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            df['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr_14'] = df['true_range'].rolling(window=14).mean()
            df['atr_14_pct'] = df['atr_14'] / df['close']
            
            # 16. Volatility Ratio
            df['vol_ratio_5_10'] = df['volatility_5d'] / df['volatility_10d']
            
            # --- STATISTICAL FEATURES ---
            
            # 17. Z-score of price
            for window in [20, 50]:
                mean = df['close'].rolling(window=window).mean()
                std = df['close'].rolling(window=window).std()
                df[f'z_score_{window}'] = (df['close'] - mean) / std
            
            # 18. Moving average of volume
            df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
            df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
            df['rel_volume'] = df['volume'] / df['volume_ma_20']
            
            # 19. Volume Oscillator
            df['volume_oscillator'] = (df['volume_ma_10'] / df['volume_ma_20']) - 1
            
            # --- TREND FEATURES ---
            
            # 20. ADX (Average Directional Index)
            # Simplified calculation
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff(-1).abs()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            tr = pd.concat([
                (df['high'] - df['low']).abs(),
                (df['high'] - df['close'].shift()).abs(),
                (df['low'] - df['close'].shift()).abs()
            ], axis=1).max(axis=1)
            
            plus_di_14 = 100 * (plus_dm.rolling(window=14).sum() / tr.rolling(window=14).sum())
            minus_di_14 = 100 * (minus_dm.rolling(window=14).sum() / tr.rolling(window=14).sum())
            dx = 100 * ((plus_di_14 - minus_di_14).abs() / (plus_di_14 + minus_di_14))
            df['adx_14'] = dx.rolling(window=14).mean()
            
            # 21. Linear Regression features
            for window in [5, 10, 20]:
                x = np.arange(window)
                df[f'linreg_slope_{window}'] = df['close'].rolling(window=window).apply(
                    lambda y: np.nan if len(y) < window else np.polyfit(x, y, 1)[0], raw=True
                )
                
                # 22. Linear regression R-squared
                df[f'linreg_r2_{window}'] = df['close'].rolling(window=window).apply(
                    lambda y: np.nan if len(y) < window else 
                    (stats.linregress(x, y).rvalue ** 2 if len(set(y)) > 1 else np.nan), 
                    raw=True
                )
            
            # 23. Price momentum
            for lag in [1, 3, 5, 10, 20]:
                df[f'momentum_{lag}d'] = df['close'] - df['close'].shift(lag)
                df[f'momentum_pct_{lag}d'] = df[f'momentum_{lag}d'] / df['close'].shift(lag)
            
            # --- CYCLE & PATTERN FEATURES ---
            
            # 24. Seasonality (day of week, if timestamp has this info)
            if 'timestamp' in df.columns:
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['hour_of_day'] = df['timestamp'].dt.hour
                df['month'] = df['timestamp'].dt.month
                
                # One-hot encode day of week for ML models
                for day in range(7):
                    df[f'day_{day}'] = (df['day_of_week'] == day).astype(int)
            
            # 25. Rate of Change (ROC)
            for period in [5, 10, 20]:
                df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100
            
            # 26. Commodity Channel Index (CCI)
            for period in [20]:
                tp = (df['high'] + df['low'] + df['close']) / 3
                tp_ma = tp.rolling(window=period).mean()
                md = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
                df[f'cci_{period}'] = (tp - tp_ma) / (0.015 * md)
            
            # --- MARKET REGIME FEATURES ---
            
            # 27. Bull/Bear Regime Indicator
            df['bull_power'] = df['high'] - df['ema_13']
            df['bear_power'] = df['low'] - df['ema_13']
            df['elder_force_index'] = df['close'].diff() * df['volume']
            df['elder_force_index_13'] = df['elder_force_index'].ewm(span=13, adjust=False).mean()
            
            # 28. Volatility Regime
            vol_quantile = df['volatility_20d'].rolling(252).quantile(0.75)
            df['high_vol_regime'] = (df['volatility_20d'] > vol_quantile).astype(int)
            
            # 29. Trend Regime (Using ADX)
            df['strong_trend_regime'] = (df['adx_14'] > 25).astype(int)
            
            # --- COMPOSITE INDICATORS ---
            
            # 30. Triple Moving Average Crossover System
            df['triple_ma_bullish'] = ((df['ema_5'] > df['ema_10']) & (df['ema_10'] > df['ema_20'])).astype(int)
            df['triple_ma_bearish'] = ((df['ema_5'] < df['ema_10']) & (df['ema_10'] < df['ema_20'])).astype(int)
            
            # 31. TRIX (Triple Exponential Average)
            ema1 = df['close'].ewm(span=18, adjust=False).mean()
            ema2 = ema1.ewm(span=18, adjust=False).mean()
            ema3 = ema2.ewm(span=18, adjust=False).mean()
            df['trix'] = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
            df['trix_signal'] = df['trix'].rolling(window=9).mean()
            
            # 32. Ultimate Oscillator
            range_7 = df['true_range'].rolling(window=7).sum()
            range_14 = df['true_range'].rolling(window=14).sum()
            range_28 = df['true_range'].rolling(window=28).sum()
            
            buying_pressure = df['close'] - df[['low', 'close']].min(axis=1).shift(1)
            
            avg7 = buying_pressure.rolling(window=7).sum() / range_7
            avg14 = buying_pressure.rolling(window=14).sum() / range_14
            avg28 = buying_pressure.rolling(window=28).sum() / range_28
            
            df['ultimate_osc'] = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
            
            # 33. Ichimoku Cloud Components
            period9_high = df['high'].rolling(window=9).max()
            period9_low = df['low'].rolling(window=9).min()
            df['tenkan_sen'] = (period9_high + period9_low) / 2
            
            period26_high = df['high'].rolling(window=26).max()
            period26_low = df['low'].rolling(window=26).min()
            df['kijun_sen'] = (period26_high + period26_low) / 2
            
            df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
            
            period52_high = df['high'].rolling(window=52).max()
            period52_low = df['low'].rolling(window=52).min()
            df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
            
            df['chikou_span'] = df['close'].shift(-26)
            
            # 34. Chaikin Money Flow
            mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            mfm = mfm.replace([np.inf, -np.inf], np.nan).fillna(0)
            mfv = mfm * df['volume']
            df['cmf_20'] = mfv.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
            
            # --- ANOMALY DETECTION FEATURES ---
            
            # 35. Price Distance from Moving Averages (Z-score)
            for ma in [20, 50, 200]:
                df[f'price_distance_{ma}'] = (df['close'] - df[f'sma_{ma}']) / df[f'sma_{ma}']
                # Z-score of the distance
                df[f'price_distance_{ma}_z'] = (df[f'price_distance_{ma}'] - df[f'price_distance_{ma}'].rolling(100).mean()) / df[f'price_distance_{ma}'].rolling(100).std()
            
            # 36. Volume Spike Detection
            df['volume_z'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
            df['volume_spike'] = (df['volume_z'] > 2).astype(int)
            
            # 37. Gap Detection
            df['gap_up'] = ((df['open'] > df['high'].shift(1)) & (df['volume'] > df['volume'].shift(1))).astype(int)
            df['gap_down'] = ((df['open'] < df['low'].shift(1)) & (df['volume'] > df['volume'].shift(1))).astype(int)
            
            # --- TARGET VARIABLES FOR ML ---
            
            # 38. Future price changes (for supervised learning)
            for forward_days in [1, 3, 5, 10]:
                # Absolute future returns
                df[f'target_return_{forward_days}d'] = df['close'].pct_change(forward_days).shift(-forward_days)
                
                # Binary future direction
                df[f'target_direction_{forward_days}d'] = (df[f'target_return_{forward_days}d'] > 0).astype(int)
                
                # Classified returns (Strong Up, Up, Neutral, Down, Strong Down)
                bins = [-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf]
                labels = [0, 1, 2, 3, 4]  # 0=Strong Down, 4=Strong Up
                df[f'target_class_{forward_days}d'] = pd.cut(df[f'target_return_{forward_days}d'], bins=bins, labels=labels).astype(float)
            
            # 39. Future volatility
            for forward_days in [5, 10]:
                df[f'future_volatility_{forward_days}d'] = df['log_return_1d'].rolling(forward_days).std().shift(-forward_days) * np.sqrt(252)
            
            # --- RISK METRICS ---
            
            # 40. Value at Risk (VaR)
            returns_dist = df['return_1d'].dropna().sort_values()
            df['historical_var_95'] = returns_dist.quantile(0.05)
            df['historical_var_99'] = returns_dist.quantile(0.01)
            
            # 41. Conditional VaR / Expected Shortfall
            df['conditional_var_95'] = returns_dist[returns_dist <= df['historical_var_95']].mean()
            
            # 42. Downside Deviation
            min_acceptable_return = 0
            downside_returns = df['return_1d'].copy()
            downside_returns[downside_returns > min_acceptable_return] = 0
            df['downside_deviation_20d'] = np.sqrt(downside_returns.pow(2).rolling(window=20).mean())
            
            # 43. Sortino Ratio (using period return and downside deviation)
            risk_free_rate = 0.0  # Simplified
            expected_return = df['return_20d'].rolling(window=20).mean() * 252  # Annualized
            df['sortino_ratio'] = (expected_return - risk_free_rate) / (df['downside_deviation_20d'] * np.sqrt(252))
            
            # --- CORRELATION FEATURES ---
            
            # 44. Autocorrelation
            for lag in [1, 5, 10]:
                df[f'autocorrelation_{lag}'] = df['return_1d'].rolling(window=30).apply(
                    lambda x: x.autocorr(lag=lag) if len(x.dropna()) > lag else np.nan
                )
            
            # 45. Synthetic Market Correlation (proxy for real market correlation in demo)
            # In a real implementation, this would use correlation with market indices or other assets
            price_trend = df['sma_20'].diff(5)
            synthetic_market = price_trend + np.random.normal(0, price_trend.std()*0.5, len(price_trend))
            df['synthetic_market'] = synthetic_market
            df['market_correlation_20d'] = df['return_1d'].rolling(20).corr(synthetic_market)
            
            # --- ADVANCED PATTERN RECOGNITION ---
            
            # 46. Candlestick Patterns (Basic implementation)
            # Doji
            df['doji'] = (abs(df['open'] - df['close']) <= (df['high'] - df['low']) * 0.1).astype(int)
            
            # Hammer
            df['hammer'] = (
                (df['body_size'] < 0.3) &  # Small body
                (df['lower_shadow'] > 0.6) &  # Long lower shadow
                (df['upper_shadow'] < 0.1)    # Minimal upper shadow
            ).astype(int)
            
            # Shooting Star
            df['shooting_star'] = (
                (df['body_size'] < 0.3) &     # Small body
                (df['lower_shadow'] < 0.1) &   # Minimal lower shadow
                (df['upper_shadow'] > 0.6)     # Long upper shadow
            ).astype(int)
            
            # 47. Chart Patterns (Basic implementation)
            # Double Top indicator (simplified)
            price_peaks = df['high'].rolling(5, center=True).apply(
                lambda x: 1 if (x.iloc[2] == x.max() and 
                               x.iloc[2] > x.iloc[1] and 
                               x.iloc[2] > x.iloc[0] and
                               x.iloc[2] > x.iloc[3] and
                               x.iloc[2] > x.iloc[4]) else 0
            )
            
            peak_prices = df['high'] * price_peaks
            last_peak = peak_prices.replace(0, np.nan).ffill()
            
            df['potential_double_top'] = (
                (price_peaks == 1) & 
                (abs(df['high'] - last_peak.shift(5)) / last_peak.shift(5) < 0.015)
            ).astype(int)
            
            # 48. Support/Resistance Levels
            # Calculate potential support/resistance bands
            for window in [20, 50]:
                price_clusters = pd.cut(df['close'], bins=10)
                resistance_level = df.groupby(price_clusters)['high'].max().sort_index().iloc[-2]
                support_level = df.groupby(price_clusters)['low'].min().sort_index().iloc[1]
                
                df[f'near_resistance_{window}'] = (abs(df['close'] - resistance_level) / resistance_level < 0.01).astype(int)
                df[f'near_support_{window}'] = (abs(df['close'] - support_level) / support_level < 0.01).astype(int)
            
            # 49. Synthetic Sentiment Indicators (for demo)
            # In a real system, these would come from actual sentiment data sources
            
            # Create a synthetic sentiment that somewhat follows price momentum but with noise
            price_momentum = df['return_5d'].rolling(window=10).mean()
            synthetic_sentiment = price_momentum + np.random.normal(0, price_momentum.std(), len(price_momentum))
            synthetic_sentiment = synthetic_sentiment.rolling(window=5).mean()  # Smooth it
            
            # Scale to range [0, 100]
            min_val = synthetic_sentiment.min()
            max_val = synthetic_sentiment.max()
            scaled_sentiment = 100 * (synthetic_sentiment - min_val) / (max_val - min_val)
            
            df['market_sentiment'] = scaled_sentiment
            df['sentiment_z_score'] = (df['market_sentiment'] - df['market_sentiment'].rolling(20).mean()) / df['market_sentiment'].rolling(20).std()
            df['sentiment_regime'] = pd.cut(df['market_sentiment'], bins=[0, 25, 75, 100], labels=['Bearish', 'Neutral', 'Bullish'])
            
            # 50. AI-Enhanced Predictive Features
            # In a real system, these would be generated from ML models
            # Here, we'll create synthetic predictors based on price patterns, with some randomness
            
            # Base signal from technical indicators
            base_signal = (df['macd'] > df['macd_signal']).astype(int) + (df['rsi_14'] < 30).astype(int) - (df['rsi_14'] > 70).astype(int)
            
            # Add randomized components representing AI predictions
            noise_factor = 0.3
            random_component = np.random.normal(0, noise_factor, len(df))
            df['ai_prediction_base'] = base_signal + random_component
            
            # Normalize to [0, 1] scale
            df['ai_prediction'] = (df['ai_prediction_base'] - df['ai_prediction_base'].min()) / (df['ai_prediction_base'].max() - df['ai_prediction_base'].min())
            
            # Clean up any invalid values
            for col in df.columns:
                if df[col].dtype != object and df[col].dtype != 'datetime64[ns]':
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            return df
        
        except Exception as e:
            logger.error(f"Error engineering features for {symbol}: {str(e)}")
            return data  # Return original data if feature engineering fails
    
    def _handle_missing_values(self, data):
        """Handle missing values in the dataset"""
        try:
            df = data.copy()
            
            # First, forward fill missing values where appropriate
            for col in df.columns:
                if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].ffill()
            
            # For any remaining NaNs, replace with column median for numeric columns
            for col in df.columns:
                if df[col].dtype != object and df[col].dtype != 'datetime64[ns]':
                    median = df[col].median()
                    df[col] = df[col].fillna(median)
            
            # Drop rows at the beginning that still have NaNs in key columns
            essential_columns = ['open', 'high', 'low', 'close', 'volume', 'sma_5', 'sma_20', 'rsi_14']
            df = df.dropna(subset=essential_columns)
            
            return df
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            return data
    
    def prepare_training_data(self, data, target_col, feature_cols=None, scale=True, forecast_horizon=1):
        """Prepare data for training ML models"""
        try:
            if data is None or data.empty:
                return None, None, None, None
            
            df = data.copy()
            
            # Set the target variable
            if target_col not in df.columns:
                logger.error(f"Target column {target_col} not found in data")
                return None, None, None, None
            
            y = df[target_col].shift(-forecast_horizon).values[:-forecast_horizon]
            
            # Feature selection
            if feature_cols is None:
                # Exclude non-feature columns
                exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                exclude_cols.extend([col for col in df.columns if col.startswith('target_')])
                feature_cols = [col for col in df.columns if col not in exclude_cols and not col.endswith('_regime')]
            
            X = df[feature_cols].values[:-forecast_horizon]
            
            # Scale features if requested
            if scale:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
            else:
                scaler = None
            
            # Split into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            return X_train, X_test, y_train, y_test, scaler, feature_cols
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return None, None, None, None, None, None
    
    def train_model(self, symbol, data=None, model_type='price_prediction', algorithm='ensemble', period='1d'):
        """
        Train an ML model for market prediction with error handling and performance tracking
        """
        try:
            # Get training data if not provided
            if data is None:
                data = self.get_training_data(symbol, period=period)
                
            if data is None or data.empty:
                logger.warning(f"Insufficient data for {model_type} model training: {symbol}")
                return False
            
            # Validate model type and algorithm
            if model_type not in self.model_registry:
                logger.error(f"Invalid model type: {model_type}")
                return False
            
            logger.info(f"Training {model_type} model ({algorithm}) for {symbol}")
            
            # Initialize model training based on model type
            if model_type == 'price_prediction':
                self._train_price_prediction_model(symbol, data, algorithm, period)
            elif model_type == 'trend_prediction':
                self._train_trend_prediction_model(symbol, data, algorithm, period)
            elif model_type == 'volatility_prediction':
                self._train_volatility_prediction_model(symbol, data, algorithm, period)
            elif model_type == 'anomaly_detection':
                self._train_anomaly_detection_model(symbol, data, algorithm, period)
            elif model_type == 'market_regime':
                self._train_market_regime_model(symbol, data, algorithm, period)
            else:
                # Fallback to basic price prediction if type not implemented
                self._train_price_prediction_model(symbol, data, algorithm, period)
            
            return True
            
        except Exception as e:
            logger.error(f"Error training {model_type} model for {symbol}: {str(e)}")
            return False
    
    def _train_price_prediction_model(self, symbol, data, algorithm, period):
        """Train a price prediction model"""
        try:
            # For price regression, predict future returns
            target_col = 'target_return_5d'
            
            # Prepare training data
            X_train, X_test, y_train, y_test, scaler, feature_cols = self.prepare_training_data(
                data, target_col=target_col, forecast_horizon=5
            )
            
            if X_train is None or y_train is None:
                logger.error(f"Failed to prepare training data for {symbol}")
                return
            
            # Train model based on selected algorithm
            model = None
            if algorithm == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            elif algorithm == 'gradient_boosting':
                model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
            elif algorithm == 'linear':
                model = LinearRegression()
            elif algorithm == 'svr':
                model = SVR(C=1.0, kernel='rbf')
            elif algorithm == 'neural_net':
                model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, early_stopping=True, random_state=42)
            elif algorithm == 'ensemble':
                # For ensemble, train multiple models and combine results
                models = {
                    'rf': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
                    'gb': GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
                    'ridge': Ridge(alpha=1.0)
                }
                
                # Train each model
                trained_models = {}
                predictions = {}
                
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    trained_models[name] = model
                    predictions[name] = model.predict(X_test)
                
                # Calculate performance and weights
                errors = {}
                for name in trained_models:
                    errors[name] = mean_squared_error(y_test, predictions[name])
                
                # Set weights inversely proportional to error
                total_inv_error = sum(1/err for err in errors.values())
                weights = {name: (1/err)/total_inv_error for name, err in errors.items()}
                
                # Store the ensemble
                model_data = {
                    'models': trained_models,
                    'weights': weights,
                    'scaler': scaler,
                    'feature_cols': feature_cols,
                    'metadata': {
                        'trained_at': datetime.utcnow(),
                        'algorithm': algorithm,
                        'model_type': 'price_prediction',
                        'target': target_col,
                        'period': period,
                        'performance': {
                            'mse': {name: err for name, err in errors.items()},
                            'combined_mse': sum(w * errors[n] for n, w in weights.items())
                        }
                    }
                }
                
                # Store in registry
                if symbol not in self.model_registry['price_prediction']:
                    self.model_registry['price_prediction'][symbol] = {}
                
                self.model_registry['price_prediction'][symbol][algorithm] = model_data
                
                # Save model data
                self._save_model(model_data, 'price_prediction', algorithm, symbol)
                
                # Calculate and store feature importance (for Random Forest model)
                if 'rf' in trained_models:
                    rf_model = trained_models['rf']
                    importances = rf_model.feature_importances_
                    sorted_indices = np.argsort(importances)[::-1]
                    feature_importance = {feature_cols[i]: importances[i] for i in sorted_indices}
                    
                    # Store top 10 features
                    self.feature_importance[f"{symbol}_price_prediction"] = {
                        k: feature_importance[k] 
                        for k in list(feature_importance)[:10]
                    }
                
                return
            
            if model is None:
                logger.error(f"Invalid algorithm: {algorithm}")
                return
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate performance
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Store model metadata
            model_data = {
                'model': model,
                'scaler': scaler,
                'feature_cols': feature_cols,
                'metadata': {
                    'trained_at': datetime.utcnow(),
                    'algorithm': algorithm,
                    'model_type': 'price_prediction',
                    'target': target_col,
                    'period': period,
                    'performance': {
                        'mse': mse,
                        'mae': mae
                    }
                }
            }
            
            # Add feature importance for tree-based models
            if algorithm in ['random_forest', 'gradient_boosting']:
                importances = model.feature_importances_
                sorted_indices = np.argsort(importances)[::-1]
                feature_importance = {feature_cols[i]: importances[i] for i in sorted_indices}
                
                # Store top 10 features
                self.feature_importance[f"{symbol}_price_prediction"] = {
                    k: feature_importance[k] 
                    for k in list(feature_importance)[:10]
                }
                
                model_data['metadata']['feature_importance'] = feature_importance
            
            # Store in registry
            if symbol not in self.model_registry['price_prediction']:
                self.model_registry['price_prediction'][symbol] = {}
            
            self.model_registry['price_prediction'][symbol][algorithm] = model_data
            
            # Save model data
            self._save_model(model_data, 'price_prediction', algorithm, symbol)
            
            # Store performance for tracking
            if symbol not in self.model_performance:
                self.model_performance[symbol] = {}
            
            self.model_performance[symbol][f"price_prediction_{algorithm}"] = {
                'timestamp': datetime.utcnow(),
                'mse': mse,
                'mae': mae
            }
            
            logger.info(f"Trained price prediction model for {symbol}: MSE={mse:.6f}, MAE={mae:.6f}")
            
        except Exception as e:
            logger.error(f"Error in price prediction model training for {symbol}: {str(e)}")
    
    def _train_trend_prediction_model(self, symbol, data, algorithm, period):
        """Train a trend prediction model (classification)"""
        try:
            # For trend prediction, use binary direction
            target_col = 'target_direction_5d'
            
            # Prepare training data
            X_train, X_test, y_train, y_test, scaler, feature_cols = self.prepare_training_data(
                data, target_col=target_col, forecast_horizon=5
            )
            
            if X_train is None or y_train is None:
                logger.error(f"Failed to prepare training data for {symbol}")
                return
            
            # Train model based on selected algorithm
            model = None
            if algorithm == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            elif algorithm == 'logistic':
                model = LogisticRegression(C=1.0, solver='liblinear', random_state=42)
            elif algorithm == 'ensemble':
                # For ensemble, train multiple models and use voting
                models = {
                    'rf': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
                    'logistic': LogisticRegression(C=1.0, solver='liblinear', random_state=42)
                }
                
                # Train each model
                trained_models = {}
                predictions = {}
                
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    trained_models[name] = model
                    predictions[name] = model.predict(X_test)
                
                # Calculate performance
                accuracies = {}
                for name in trained_models:
                    accuracies[name] = accuracy_score(y_test, predictions[name])
                
                # Set weights proportional to accuracy
                total_acc = sum(accuracies.values())
                weights = {name: acc/total_acc for name, acc in accuracies.items()}
                
                # Store the ensemble
                model_data = {
                    'models': trained_models,
                    'weights': weights,
                    'scaler': scaler,
                    'feature_cols': feature_cols,
                    'metadata': {
                        'trained_at': datetime.utcnow(),
                        'algorithm': algorithm,
                        'model_type': 'trend_prediction',
                        'target': target_col,
                        'period': period,
                        'performance': {
                            'accuracy': {name: acc for name, acc in accuracies.items()},
                            'combined_accuracy': sum(w * accuracies[n] for n, w in weights.items())
                        }
                    }
                }
                
                # Store in registry
                if symbol not in self.model_registry['trend_prediction']:
                    self.model_registry['trend_prediction'][symbol] = {}
                
                self.model_registry['trend_prediction'][symbol][algorithm] = model_data
                
                # Save model data
                self._save_model(model_data, 'trend_prediction', algorithm, symbol)
                
                return
            
            if model is None:
                logger.error(f"Invalid algorithm: {algorithm}")
                return
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate performance
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            f1 = f1_score(y_test, y_pred, average='binary')
            
            # Store model metadata
            model_data = {
                'model': model,
                'scaler': scaler,
                'feature_cols': feature_cols,
                'metadata': {
                    'trained_at': datetime.utcnow(),
                    'algorithm': algorithm,
                    'model_type': 'trend_prediction',
                    'target': target_col,
                    'period': period,
                    'performance': {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }
                }
            }
            
            # Add feature importance for tree-based models
            if algorithm == 'random_forest':
                importances = model.feature_importances_
                sorted_indices = np.argsort(importances)[::-1]
                feature_importance = {feature_cols[i]: importances[i] for i in sorted_indices}
                
                # Store top 10 features
                self.feature_importance[f"{symbol}_trend_prediction"] = {
                    k: feature_importance[k] 
                    for k in list(feature_importance)[:10]
                }
                
                model_data['metadata']['feature_importance'] = feature_importance
            
            # Store in registry
            if symbol not in self.model_registry['trend_prediction']:
                self.model_registry['trend_prediction'][symbol] = {}
            
            self.model_registry['trend_prediction'][symbol][algorithm] = model_data
            
            # Save model data
            self._save_model(model_data, 'trend_prediction', algorithm, symbol)
            
            # Store performance for tracking
            if symbol not in self.model_performance:
                self.model_performance[symbol] = {}
            
            self.model_performance[symbol][f"trend_prediction_{algorithm}"] = {
                'timestamp': datetime.utcnow(),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            logger.info(f"Trained trend prediction model for {symbol}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
            
        except Exception as e:
            logger.error(f"Error in trend prediction model training for {symbol}: {str(e)}")
    
    def _train_volatility_prediction_model(self, symbol, data, algorithm, period):
        """Train a volatility prediction model"""
        try:
            # For volatility prediction, predict future volatility
            target_col = 'future_volatility_5d'
            
            # Prepare training data
            X_train, X_test, y_train, y_test, scaler, feature_cols = self.prepare_training_data(
                data, target_col=target_col, forecast_horizon=5
            )
            
            if X_train is None or y_train is None:
                logger.error(f"Failed to prepare training data for {symbol}")
                return
            
            # Train model based on selected algorithm
            model = None
            if algorithm == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
            elif algorithm == 'gradient_boosting':
                model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
            elif algorithm == 'svr':
                model = SVR(C=1.0, kernel='rbf')
            
            if model is None:
                logger.error(f"Invalid algorithm: {algorithm}")
                return
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate performance
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Store model metadata
            model_data = {
                'model': model,
                'scaler': scaler,
                'feature_cols': feature_cols,
                'metadata': {
                    'trained_at': datetime.utcnow(),
                    'algorithm': algorithm,
                    'model_type': 'volatility_prediction',
                    'target': target_col,
                    'period': period,
                    'performance': {
                        'mse': mse,
                        'mae': mae
                    }
                }
            }
            
            # Store in registry
            if symbol not in self.model_registry['volatility_prediction']:
                self.model_registry['volatility_prediction'][symbol] = {}
            
            self.model_registry['volatility_prediction'][symbol][algorithm] = model_data
            
            # Save model data
            self._save_model(model_data, 'volatility_prediction', algorithm, symbol)
            
            logger.info(f"Trained volatility prediction model for {symbol}: MSE={mse:.6f}")
            
        except Exception as e:
            logger.error(f"Error in volatility prediction model training for {symbol}: {str(e)}")
    
    def _train_anomaly_detection_model(self, symbol, data, algorithm, period):
        """Train an anomaly detection model"""
        try:
            # Feature selection for anomaly detection
            feature_cols = [
                'return_1d', 'return_5d', 'volatility_20d', 'volume_z',
                'rsi_14', 'bb_width_20', 'macd', 'adx_14'
            ]
            
            # Filter & prepare data
            X = data[feature_cols].values
            X = StandardScaler().fit_transform(X)
            
            # Train model based on algorithm
            from sklearn.ensemble import IsolationForest
            from sklearn.svm import OneClassSVM
            
            if algorithm == 'isolation_forest':
                model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
                model.fit(X)
                
                # Calculate anomaly scores
                scores = model.decision_function(X)
                threshold = np.percentile(scores, 5)  # Bottom 5% are anomalies
                
            elif algorithm == 'one_class_svm':
                model = OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
                model.fit(X)
                
                # Calculate anomaly scores
                scores = model.decision_function(X)
                threshold = np.percentile(scores, 5)  # Bottom 5% are anomalies
            
            else:
                logger.error(f"Invalid algorithm for anomaly detection: {algorithm}")
                return
            
            # Store model and threshold in registry
            model_data = {
                'model': model,
                'threshold': threshold,
                'feature_cols': feature_cols,
                'metadata': {
                    'trained_at': datetime.utcnow(),
                    'algorithm': algorithm,
                    'model_type': 'anomaly_detection',
                    'period': period,
                    'contamination': 0.05
                }
            }
            
            # Store in registry
            if symbol not in self.model_registry['anomaly_detection']:
                self.model_registry['anomaly_detection'][symbol] = {}
            
            self.model_registry['anomaly_detection'][symbol][algorithm] = model_data
            
            # Store threshold for later use
            self.anomaly_thresholds[symbol] = threshold
            
            # Save model data
            self._save_model(model_data, 'anomaly_detection', algorithm, symbol)
            
            logger.info(f"Trained anomaly detection model for {symbol} with threshold {threshold:.4f}")
            
        except Exception as e:
            logger.error(f"Error in anomaly detection model training for {symbol}: {str(e)}")
    
    def _train_market_regime_model(self, symbol, data, algorithm, period):
        """Train a market regime classification model"""
        try:
            # Feature selection for regime modeling
            feature_cols = [
                'volatility_20d', 'adx_14', 'rsi_14', 'macd_hist',
                'bb_width_20', 'triple_ma_bullish', 'triple_ma_bearish'
            ]
            
            # Prepare data
            X = data[feature_cols].values
            X = StandardScaler().fit_transform(X)
            
            # Train model based on algorithm
            from sklearn.cluster import KMeans
            from sklearn.mixture import GaussianMixture
            
            if algorithm == 'kmeans':
                # Use 3 regimes: Bull, Bear, Neutral
                model = KMeans(n_clusters=3, random_state=42)
                clusters = model.fit_predict(X)
                
                # Identify cluster characteristics
                df_temp = data.copy()
                df_temp['cluster'] = clusters
                
                # Calculate average returns by cluster
                cluster_returns = df_temp.groupby('cluster')['return_20d'].mean()
                
                # Identify bull/bear/neutral based on returns
                sorted_clusters = cluster_returns.sort_values()
                
                regime_map = {
                    sorted_clusters.index[0]: 'bear',
                    sorted_clusters.index[1]: 'neutral',
                    sorted_clusters.index[2]: 'bull'
                }
                
                # Store model and mapping
                model_data = {
                    'model': model,
                    'regime_map': regime_map,
                    'feature_cols': feature_cols,
                    'metadata': {
                        'trained_at': datetime.utcnow(),
                        'algorithm': algorithm,
                        'model_type': 'market_regime',
                        'period': period,
                        'n_regimes': 3,
                        'avg_returns': {k: float(v) for k, v in cluster_returns.items()}
                    }
                }
                
            elif algorithm == 'gmm':
                # Gaussian Mixture Model
                model = GaussianMixture(n_components=3, random_state=42)
                model.fit(X)
                clusters = model.predict(X)
                
                # Identify cluster characteristics
                df_temp = data.copy()
                df_temp['cluster'] = clusters
                
                # Calculate average returns by cluster
                cluster_returns = df_temp.groupby('cluster')['return_20d'].mean()
                
                # Identify bull/bear/neutral based on returns
                sorted_clusters = cluster_returns.sort_values()
                
                regime_map = {
                    sorted_clusters.index[0]: 'bear',
                    sorted_clusters.index[1]: 'neutral',
                    sorted_clusters.index[2]: 'bull'
                }
                
                # Store model and mapping
                model_data = {
                    'model': model,
                    'regime_map': regime_map,
                    'feature_cols': feature_cols,
                    'metadata': {
                        'trained_at': datetime.utcnow(),
                        'algorithm': algorithm,
                        'model_type': 'market_regime',
                        'period': period,
                        'n_regimes': 3,
                        'avg_returns': {k: float(v) for k, v in cluster_returns.items()}
                    }
                }
                
            else:
                logger.error(f"Invalid algorithm for market regime: {algorithm}")
                return
            
            # Store in registry
            if symbol not in self.model_registry['market_regime']:
                self.model_registry['market_regime'][symbol] = {}
            
            self.model_registry['market_regime'][symbol][algorithm] = model_data
            
            # Store regime mapping
            self.regime_states[symbol] = regime_map
            
            # Save model data
            self._save_model(model_data, 'market_regime', algorithm, symbol)
            
            logger.info(f"Trained market regime model for {symbol} with {len(regime_map)} regimes")
            
        except Exception as e:
            logger.error(f"Error in market regime model training for {symbol}: {str(e)}")
    
    def predict(self, symbol, data=None, market_type='forex', period='1d', prediction_types=None):
        """
        Make predictions using trained AI models with comprehensive results
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{market_type}_{period}"
            now = datetime.utcnow()
            
            if (cache_key in self.cached_predictions and 
                cache_key in self.cached_time and 
                (now - self.cached_time[cache_key]).total_seconds() < 900):  # Cache valid for 15 minutes
                return self.cached_predictions[cache_key]
            
            # Get data if not provided
            if data is None:
                data = self.get_training_data(symbol, period=period)
                
            if data is None or data.empty:
                logger.warning(f"No data for prediction on {symbol}")
                return self._generate_fallback_prediction(symbol, market_type)
            
            # Define what predictions to make
            if prediction_types is None:
                prediction_types = ['price', 'trend', 'volatility', 'anomaly', 'regime']
            
            # Initialize result
            result = {
                'symbol': symbol,
                'current_price': float(data['close'].iloc[-1]),
                'timestamp': now,
                'market_type': market_type,
                'period': period,
                'predictions': {}
            }
            
            # Make predictions based on requested types
            if 'price' in prediction_types:
                price_pred = self._predict_price(symbol, data, period)
                if price_pred:
                    result['predictions']['price'] = price_pred
            
            if 'trend' in prediction_types:
                trend_pred = self._predict_trend(symbol, data, period)
                if trend_pred:
                    result['predictions']['trend'] = trend_pred
            
            if 'volatility' in prediction_types:
                vol_pred = self._predict_volatility(symbol, data, period)
                if vol_pred:
                    result['predictions']['volatility'] = vol_pred
            
            if 'anomaly' in prediction_types:
                anomaly_pred = self._detect_anomalies(symbol, data, period)
                if anomaly_pred:
                    result['predictions']['anomaly'] = anomaly_pred
            
            if 'regime' in prediction_types:
                regime_pred = self._predict_regime(symbol, data, period)
                if regime_pred:
                    result['predictions']['regime'] = regime_pred
            
            # Calculate overall prediction metrics
            # Combined signal from all predictions
            signal_value = 0.5  # Neutral starting point
            signal_count = 0
            confidence_sum = 0
            
            if 'price' in result['predictions']:
                pred = result['predictions']['price']
                # Scale from 0-1 to -1 to 1 (0.5 is neutral)
                signal_value += (pred['value'] - 0.5) * pred['confidence']
                signal_count += 1
                confidence_sum += pred['confidence']
            
            if 'trend' in result['predictions']:
                pred = result['predictions']['trend']
                direction_value = 1 if pred['direction'] == 'up' else 0
                # Scale from 0-1 to -1 to 1 (0.5 is neutral)
                signal_value += (direction_value - 0.5) * 2 * pred['confidence']
                signal_count += 1
                confidence_sum += pred['confidence']
            
            if 'regime' in result['predictions']:
                pred = result['predictions']['regime']
                if pred['regime'] == 'bull':
                    signal_value += 0.25 * pred['confidence']
                elif pred['regime'] == 'bear':
                    signal_value -= 0.25 * pred['confidence']
                signal_count += 0.5  # Lower weight for regime
                confidence_sum += pred['confidence'] * 0.5
            
            # Normalize signal between 0 and 1
            if signal_count > 0:
                # Adjust signal to 0-1 range
                signal_value = signal_value / signal_count
                signal_value = min(1, max(0, signal_value + 0.5))  # Scale back to 0-1
                
                # Calculate average confidence
                avg_confidence = confidence_sum / signal_count
                
                # Determine signal interpretation
                if signal_value > 0.66:
                    signal = 'BUY'
                    strength = (signal_value - 0.66) / 0.34
                elif signal_value < 0.33:
                    signal = 'SELL'
                    strength = (0.33 - signal_value) / 0.33
                else:
                    signal = 'NEUTRAL'
                    strength = 1 - abs(signal_value - 0.5) * 2
                
                # Add overall prediction
                result['prediction'] = signal_value
                result['confidence'] = avg_confidence
                result['signal'] = signal
                result['strength'] = strength
                result['success_probability'] = min(0.95, 0.5 + abs(signal_value - 0.5))
            else:
                # Fallback if no predictions were generated
                result.update(self._generate_fallback_prediction(symbol, market_type))
            
            # Cache the prediction
            self.cached_predictions[cache_key] = result
            self.cached_time[cache_key] = now
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction for {symbol}: {str(e)}")
            return self._generate_fallback_prediction(symbol, market_type)
    
    def _predict_price(self, symbol, data, period):
        """Make price prediction using trained models"""
        try:
            # Check if we have a trained model
            if (symbol not in self.model_registry['price_prediction'] or
                not self.model_registry['price_prediction'][symbol]):
                logger.warning(f"No trained price prediction model for {symbol}")
                return self._generate_price_prediction_fallback(data)
            
            # Get the most recent data point for prediction
            last_data = data.iloc[-1:].copy()
            
            # Prefer ensemble model if available
            model_data = None
            for algorithm in ['ensemble', 'gradient_boosting', 'random_forest', 'linear']:
                if algorithm in self.model_registry['price_prediction'][symbol]:
                    model_data = self.model_registry['price_prediction'][symbol][algorithm]
                    break
            
            if not model_data:
                # Use any available model
                model_data = next(iter(self.model_registry['price_prediction'][symbol].values()))
            
            # Handle ensemble models differently
            if 'models' in model_data:
                # This is an ensemble
                feature_cols = model_data['feature_cols']
                scaler = model_data['scaler']
                weights = model_data['weights']
                models = model_data['models']
                
                # Prepare features
                X = data[feature_cols].iloc[-1:].values
                if scaler:
                    X = scaler.transform(X)
                
                # Make predictions from each model
                predictions = {}
                for name, model in models.items():
                    predictions[name] = model.predict(X)[0]
                
                # Combine predictions using weights
                predicted_value = sum(predictions[name] * weights[name] for name in models)
                
                # Calculate confidence based on model performance
                confidence = 0.7  # Base confidence
                if 'performance' in model_data['metadata']:
                    # Lower confidence if error is high
                    avg_mse = model_data['metadata']['performance']['combined_mse']
                    confidence = max(0.5, min(0.9, 0.9 - avg_mse * 10))
                
            else:
                # Regular single model
                model = model_data['model']
                feature_cols = model_data['feature_cols']
                scaler = model_data['scaler']
                
                # Prepare features
                X = data[feature_cols].iloc[-1:].values
                if scaler:
                    X = scaler.transform(X)
                
                # Make prediction
                predicted_value = model.predict(X)[0]
                
                # Calculate confidence based on model performance
                confidence = 0.7  # Base confidence
                if 'performance' in model_data['metadata']:
                    # Lower confidence if error is high
                    mse = model_data['metadata']['performance']['mse']
                    confidence = max(0.5, min(0.9, 0.9 - mse * 10))
            
            # Calculate prediction value from return
            current_price = data['close'].iloc[-1]
            
            # Scale prediction value to 0-1 range (0.5 is neutral)
            # Convert predicted return to a signal scale
            # A return of 0 should be 0.5 (neutral)
            # Max/min reasonable returns would be +/- 5%
            scaled_prediction = 0.5 + (predicted_value * 10)  # Scale return to 0-1
            scaled_prediction = max(0, min(1, scaled_prediction))  # Clip to 0-1
            
            return {
                'value': scaled_prediction,
                'confidence': confidence,
                'original_value': float(predicted_value),
                'horizon': '5d',
                'interpretation': 'bullish' if scaled_prediction > 0.6 else ('bearish' if scaled_prediction < 0.4 else 'neutral')
            }
            
        except Exception as e:
            logger.error(f"Error in price prediction for {symbol}: {str(e)}")
            return self._generate_price_prediction_fallback(data)
    
    def _predict_trend(self, symbol, data, period):
        """Make trend prediction using trained models"""
        try:
            # Check if we have a trained model
            if (symbol not in self.model_registry['trend_prediction'] or
                not self.model_registry['trend_prediction'][symbol]):
                logger.warning(f"No trained trend prediction model for {symbol}")
                return self._generate_trend_prediction_fallback(data)
            
            # Prefer ensemble model if available
            model_data = None
            for algorithm in ['ensemble', 'random_forest', 'logistic']:
                if algorithm in self.model_registry['trend_prediction'][symbol]:
                    model_data = self.model_registry['trend_prediction'][symbol][algorithm]
                    break
            
            if not model_data:
                # Use any available model
                model_data = next(iter(self.model_registry['trend_prediction'][symbol].values()))
            
            # Handle ensemble models differently
            if 'models' in model_data:
                # This is an ensemble
                feature_cols = model_data['feature_cols']
                scaler = model_data['scaler']
                weights = model_data['weights']
                models = model_data['models']
                
                # Prepare features
                X = data[feature_cols].iloc[-1:].values
                if scaler:
                    X = scaler.transform(X)
                
                # Make predictions from each model
                predictions = {}
                probabilities = {}
                for name, model in models.items():
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)[0]
                        probabilities[name] = proba[1]  # Probability of class 1 (up)
                        predictions[name] = 1 if proba[1] > 0.5 else 0
                    else:
                        predictions[name] = model.predict(X)[0]
                        probabilities[name] = 0.5 + (predictions[name] - 0.5) * 0.5  # Simulate probability
                
                # Weighted vote
                vote_value = sum(predictions[name] * weights[name] for name in models)
                proba_value = sum(probabilities[name] * weights[name] for name in models)
                
                direction = 'up' if vote_value > 0.5 else 'down'
                
                # Calculate confidence based on agreement and model performance
                confidence = max(0.5, min(0.9, proba_value))
                if 'performance' in model_data['metadata']:
                    # Adjust by performance metrics
                    perf = model_data['metadata']['performance']['combined_accuracy']
                    confidence = confidence * perf
                
            else:
                # Regular single model
                model = model_data['model']
                feature_cols = model_data['feature_cols']
                scaler = model_data['scaler']
                
                # Prepare features
                X = data[feature_cols].iloc[-1:].values
                if scaler:
                    X = scaler.transform(X)
                
                # Make prediction
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0]
                    direction = 'up' if proba[1] > 0.5 else 'down'
                    confidence = max(proba)
                else:
                    pred = model.predict(X)[0]
                    direction = 'up' if pred == 1 else 'down'
                    confidence = 0.7  # Default confidence
                
                # Adjust confidence based on model performance
                if 'performance' in model_data['metadata']:
                    perf = model_data['metadata']['performance']['accuracy']
                    confidence = confidence * perf
            
            return {
                'direction': direction,
                'confidence': confidence,
                'horizon': '5d',
                'probability_up': float(proba_value if 'proba_value' in locals() else (proba[1] if 'proba' in locals() else 0.5))
            }
            
        except Exception as e:
            logger.error(f"Error in trend prediction for {symbol}: {str(e)}")
            return self._generate_trend_prediction_fallback(data)
    
    def _predict_volatility(self, symbol, data, period):
        """Make volatility prediction using trained models"""
        try:
            # Check if we have a trained model
            if (symbol not in self.model_registry['volatility_prediction'] or
                not self.model_registry['volatility_prediction'][symbol]):
                logger.warning(f"No trained volatility prediction model for {symbol}")
                return self._generate_volatility_prediction_fallback(data)
            
            # Get any available volatility model
            model_data = None
            for algorithm in ['gradient_boosting', 'random_forest', 'svr']:
                if algorithm in self.model_registry['volatility_prediction'][symbol]:
                    model_data = self.model_registry['volatility_prediction'][symbol][algorithm]
                    break
            
            if not model_data:
                # Use first available model
                model_data = next(iter(self.model_registry['volatility_prediction'][symbol].values()))
            
            # Extract model components
            model = model_data['model']
            feature_cols = model_data['feature_cols']
            scaler = model_data['scaler']
            
            # Prepare features
            X = data[feature_cols].iloc[-1:].values
            if scaler:
                X = scaler.transform(X)
            
            # Make prediction
            predicted_volatility = model.predict(X)[0]
            
            # Calculate confidence based on model performance
            confidence = 0.7  # Base confidence
            if 'performance' in model_data['metadata']:
                # Lower confidence if error is high
                mse = model_data['metadata']['performance']['mse']
                confidence = max(0.5, min(0.9, 0.9 - mse * 10))
            
            # Get current volatility for comparison
            current_volatility = data['volatility_20d'].iloc[-1]
            
            return {
                'value': float(predicted_volatility),
                'confidence': confidence,
                'horizon': '5d',
                'current_volatility': float(current_volatility),
                'change': float(predicted_volatility - current_volatility),
                'interpretation': 'increasing' if predicted_volatility > current_volatility * 1.1 else 
                                ('decreasing' if predicted_volatility < current_volatility * 0.9 else 'stable')
            }
            
        except Exception as e:
            logger.error(f"Error in volatility prediction for {symbol}: {str(e)}")
            return self._generate_volatility_prediction_fallback(data)
    
    def _detect_anomalies(self, symbol, data, period):
        """Detect market anomalies using trained models"""
        try:
            # Check if we have a trained model
            if (symbol not in self.model_registry['anomaly_detection'] or
                not self.model_registry['anomaly_detection'][symbol]):
                logger.warning(f"No trained anomaly detection model for {symbol}")
                return self._generate_anomaly_detection_fallback(data)
            
            # Get any available anomaly model
            model_data = None
            for algorithm in ['isolation_forest', 'one_class_svm']:
                if algorithm in self.model_registry['anomaly_detection'][symbol]:
                    model_data = self.model_registry['anomaly_detection'][symbol][algorithm]
                    break
            
            if not model_data:
                # Use first available model
                model_data = next(iter(self.model_registry['anomaly_detection'][symbol].values()))
            
            # Extract model components
            model = model_data['model']
            feature_cols = model_data['feature_cols']
            threshold = model_data['threshold']
            
            # Prepare features
            X = data[feature_cols].iloc[-1:].values
            X = StandardScaler().fit_transform(X)
            
            # Get anomaly score
            score = float(model.decision_function(X)[0])
            
            # Determine if anomalous
            is_anomaly = score < threshold
            
            # Calculate confidence based on distance from threshold
            if is_anomaly:
                # Further below threshold = higher confidence
                confidence = min(0.95, 0.7 + (threshold - score) / threshold)
            else:
                # Further above threshold = higher confidence
                confidence = min(0.95, 0.7 + (score - threshold) / abs(threshold))
            
            # Get recent price and volume context
            price_change = data['return_1d'].iloc[-1]
            volume_z = data['volume_z'].iloc[-1] if 'volume_z' in data else 0
            
            # Determine anomaly type
            anomaly_type = 'price_movement'  # Default
            if is_anomaly:
                if abs(price_change) > 0.03:  # 3% move
                    anomaly_type = 'price_shock'
                elif volume_z > 2:
                    anomaly_type = 'volume_spike'
                elif 'volatility_5d' in data and data['volatility_5d'].iloc[-1] > data['volatility_20d'].iloc[-1] * 1.5:
                    anomaly_type = 'volatility_spike'
            
            return {
                'is_anomaly': bool(is_anomaly),
                'score': float(score),
                'threshold': float(threshold),
                'confidence': float(confidence),
                'anomaly_type': anomaly_type if is_anomaly else None,
                'price_change': float(price_change),
                'volume_z': float(volume_z) if 'volume_z' in locals() else None
            }
            
        except Exception as e:
            logger.error(f"Error in anomaly detection for {symbol}: {str(e)}")
            return self._generate_anomaly_detection_fallback(data)
    
    def _predict_regime(self, symbol, data, period):
        """Predict current market regime using trained models"""
        try:
            # Check if we have a trained model
            if (symbol not in self.model_registry['market_regime'] or
                not self.model_registry['market_regime'][symbol]):
                logger.warning(f"No trained market regime model for {symbol}")
                return self._generate_regime_prediction_fallback(data)
            
            # Get any available regime model
            model_data = None
            for algorithm in ['kmeans', 'gmm']:
                if algorithm in self.model_registry['market_regime'][symbol]:
                    model_data = self.model_registry['market_regime'][symbol][algorithm]
                    break
            
            if not model_data:
                # Use first available model
                model_data = next(iter(self.model_registry['market_regime'][symbol].values()))
            
            # Extract model components
            model = model_data['model']
            feature_cols = model_data['feature_cols']
            regime_map = model_data['regime_map']
            
            # Prepare features
            X = data[feature_cols].iloc[-1:].values
            X = StandardScaler().fit_transform(X)
            
            # Get regime cluster
            if hasattr(model, 'predict'):
                cluster = model.predict(X)[0]
            else:
                # Fallback if model doesn't have predict
                cluster = 0
            
            # Map to regime name
            regime = regime_map.get(cluster, 'unknown')
            
            # Calculate confidence based on model metadata
            confidence = 0.7  # Base confidence
            if 'metadata' in model_data and 'avg_returns' in model_data['metadata']:
                # Higher absolute average return = more distinct regime = higher confidence
                avg_returns = model_data['metadata']['avg_returns']
                if str(cluster) in avg_returns:
                    abs_return = abs(avg_returns[str(cluster)])
                    confidence = min(0.9, 0.6 + abs_return * 5)
            
            # Get additional regime context
            context = {
                'adx': float(data['adx_14'].iloc[-1]) if 'adx_14' in data else None,
                'volatility': float(data['volatility_20d'].iloc[-1]) if 'volatility_20d' in data else None,
                'rsi': float(data['rsi_14'].iloc[-1]) if 'rsi_14' in data else None,
                'trend_direction': 'up' if data['triple_ma_bullish'].iloc[-1] else ('down' if data['triple_ma_bearish'].iloc[-1] else 'neutral')
                if 'triple_ma_bullish' in data else 'unknown'
            }
            
            return {
                'regime': regime,
                'confidence': confidence,
                'context': context,
                'raw_cluster': int(cluster)
            }
            
        except Exception as e:
            logger.error(f"Error in regime prediction for {symbol}: {str(e)}")
            return self._generate_regime_prediction_fallback(data)
    
    def _generate_fallback_prediction(self, symbol, market_type):
        """Generate a fallback prediction when models fail"""
        try:
            # Basic fallback that aims to be neutral
            now = datetime.utcnow()
            
            # Simple synthetic prediction based on symbol name for demo consistency
            # Hash the symbol to get a consistent value for the same symbol
            import hashlib
            hash_val = int(hashlib.md5(symbol.encode()).hexdigest(), 16)
            
            # Base prediction on hash (deterministic but looks random)
            rand_val = (hash_val % 100) / 100.0
            
            # Slightly bias toward the middle
            prediction = 0.5 + (rand_val - 0.5) * 0.6
            
            # Middle confidence
            confidence = 0.5 + (hash_val % 20) / 100.0
            
            # Determine signal
            if prediction > 0.6:
                signal = 'BUY'
                strength = (prediction - 0.6) / 0.4
            elif prediction < 0.4:
                signal = 'SELL'
                strength = (0.4 - prediction) / 0.4
            else:
                signal = 'NEUTRAL'
                strength = 1 - abs(prediction - 0.5) * 2
            
            return {
                'symbol': symbol,
                'prediction': prediction,
                'confidence': confidence,
                'timestamp': now,
                'market_type': market_type,
                'interpretation': 'bullish' if prediction > 0.6 else ('bearish' if prediction < 0.4 else 'neutral'),
                'signal': signal,
                'strength': strength,
                'success_probability': min(0.95, 0.5 + abs(prediction - 0.5)),
                'is_fallback': True
            }
            
        except Exception as e:
            logger.error(f"Error generating fallback prediction: {str(e)}")
            # Ultra basic fallback
            return {
                'symbol': symbol,
                'prediction': 0.5,
                'confidence': 0.5,
                'timestamp': datetime.utcnow(),
                'market_type': market_type,
                'interpretation': 'neutral',
                'signal': 'NEUTRAL',
                'strength': 0.5,
                'success_probability': 0.5,
                'is_fallback': True
            }
    
    def _generate_price_prediction_fallback(self, data):
        """Generate fallback price prediction"""
        if data is None or data.empty:
            return {
                'value': 0.5,
                'confidence': 0.5,
                'original_value': 0.0,
                'horizon': '5d',
                'interpretation': 'neutral'
            }
        
        # Use recent momentum to make a simple prediction
        if len(data) > 20:
            # Calculate momentum
            momentum = data['close'].pct_change(5).iloc[-1]
            
            # Scale to prediction value (0-1)
            value = 0.5 + momentum * 5  # Scale momentum to prediction
            value = max(0, min(1, value))  # Clip to 0-1 range
            
            interpretation = 'bullish' if value > 0.6 else ('bearish' if value < 0.4 else 'neutral')
            
            return {
                'value': float(value),
                'confidence': 0.5,  # Low confidence for fallback
                'original_value': float(momentum),
                'horizon': '5d',
                'interpretation': interpretation
            }
        else:
            return {
                'value': 0.5,
                'confidence': 0.5,
                'original_value': 0.0,
                'horizon': '5d',
                'interpretation': 'neutral'
            }
    
    def _generate_trend_prediction_fallback(self, data):
        """Generate fallback trend prediction"""
        if data is None or data.empty:
            return {
                'direction': 'neutral',
                'confidence': 0.5,
                'horizon': '5d',
                'probability_up': 0.5
            }
        
        # Use simple MA crossover
        if len(data) > 50:
            # Calculate SMAs
            sma20 = data['close'].rolling(20).mean()
            sma50 = data['close'].rolling(50).mean()
            
            # Determine direction
            direction = 'up' if sma20.iloc[-1] > sma50.iloc[-1] else 'down'
            
            # Calculate strength of signal by distance between MAs
            spread = abs(sma20.iloc[-1] - sma50.iloc[-1]) / sma50.iloc[-1]
            probability_up = 0.7 if direction == 'up' else 0.3
            
            return {
                'direction': direction,
                'confidence': 0.5 + min(0.3, spread * 10),  # Confidence based on MA spread
                'horizon': '5d',
                'probability_up': float(probability_up)
            }
        else:
            return {
                'direction': 'neutral',
                'confidence': 0.5,
                'horizon': '5d',
                'probability_up': 0.5
            }
    
    def _generate_volatility_prediction_fallback(self, data):
        """Generate fallback volatility prediction"""
        if data is None or data.empty:
            return {
                'value': 0.01,  # 1% daily volatility
                'confidence': 0.5,
                'horizon': '5d',
                'current_volatility': 0.01,
                'change': 0.0,
                'interpretation': 'stable'
            }
        
        # Calculate recent volatility
        if len(data) > 20 and 'log_return_1d' in data.columns:
            current_vol = data['log_return_1d'].rolling(20).std().iloc[-1] * np.sqrt(252)
            
            # Simple prediction using recent change in volatility
            if len(data) > 40:
                prev_vol = data['log_return_1d'].rolling(20).std().iloc[-20] * np.sqrt(252)
                vol_change = current_vol - prev_vol
                pred_vol = current_vol + vol_change * 0.5  # Half the recent change
            else:
                pred_vol = current_vol
            
            interpretation = 'increasing' if pred_vol > current_vol * 1.1 else ('decreasing' if pred_vol < current_vol * 0.9 else 'stable')
            
            return {
                'value': float(pred_vol),
                'confidence': 0.5,
                'horizon': '5d',
                'current_volatility': float(current_vol),
                'change': float(pred_vol - current_vol),
                'interpretation': interpretation
            }
        else:
            return {
                'value': 0.01,  # 1% daily volatility
                'confidence': 0.5,
                'horizon': '5d',
                'current_volatility': 0.01,
                'change': 0.0,
                'interpretation': 'stable'
            }
    
    def _generate_anomaly_detection_fallback(self, data):
        """Generate fallback anomaly detection"""
        if data is None or data.empty:
            return {
                'is_anomaly': False,
                'score': 0.0,
                'threshold': -0.1,
                'confidence': 0.5,
                'anomaly_type': None,
                'price_change': 0.0,
                'volume_z': 0.0
            }
        
        # Simple detection based on price change
        price_change = data['close'].pct_change().iloc[-1] if len(data) > 1 else 0
        is_anomaly = abs(price_change) > 0.03  # 3% move is anomalous
        
        # Check volume if available
        volume_change = 0
        if 'volume' in data.columns and len(data) > 20:
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            volume_change = (data['volume'].iloc[-1] / avg_volume) - 1
        
        volume_anomaly = volume_change > 1.0  # 2x volume is anomalous
        
        is_anomaly = is_anomaly or volume_anomaly
        
        return {
            'is_anomaly': bool(is_anomaly),
            'score': float(-0.2 if is_anomaly else 0.2),
            'threshold': -0.1,
            'confidence': 0.6,
            'anomaly_type': 'price_shock' if abs(price_change) > 0.03 else ('volume_spike' if volume_anomaly else None),
            'price_change': float(price_change),
            'volume_z': float(volume_change)
        }
    
    def _generate_regime_prediction_fallback(self, data):
        """Generate fallback regime prediction"""
        if data is None or data.empty:
            return {
                'regime': 'neutral',
                'confidence': 0.5,
                'context': {
                    'adx': None,
                    'volatility': None,
                    'rsi': None,
                    'trend_direction': 'unknown'
                },
                'raw_cluster': 1
            }
        
        # Simple regime detection based on RSI and trend
        regime = 'neutral'
        if 'rsi_14' in data.columns:
            rsi = data['rsi_14'].iloc[-1]
            if rsi > 70:
                regime = 'bull'
            elif rsi < 30:
                regime = 'bear'
        
        # Use price trend as fallback
        elif len(data) > 20:
            sma20 = data['close'].rolling(20).mean().iloc[-1]
            sma10 = data['close'].rolling(10).mean().iloc[-1]
            
            if sma10 > sma20 * 1.02:
                regime = 'bull'
            elif sma10 < sma20 * 0.98:
                regime = 'bear'
        
        # Get context if available
        context = {
            'adx': float(data['adx_14'].iloc[-1]) if 'adx_14' in data.columns else None,
            'volatility': float(data['volatility_20d'].iloc[-1]) if 'volatility_20d' in data.columns else None,
            'rsi': float(data['rsi_14'].iloc[-1]) if 'rsi_14' in data.columns else None,
            'trend_direction': 'unknown'
        }
        
        return {
            'regime': regime,
            'confidence': 0.6,
            'context': context,
            'raw_cluster': 1
        }
    
    def generate_feature_importance_chart(self, symbol, model_type='price_prediction', top_n=10):
        """Generate feature importance chart as base64 encoded image"""
        try:
            # Check if we have feature importance data
            key = f"{symbol}_{model_type}"
            if key not in self.feature_importance:
                logger.warning(f"No feature importance data for {key}")
                return None
            
            # Get feature importance
            importance = self.feature_importance[key]
            
            # Sort and limit to top N
            sorted_features = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True)[:top_n])
            
            # Create chart
            plt.figure(figsize=(10, 6))
            plt.barh(list(sorted_features.keys()), list(sorted_features.values()))
            plt.xlabel('Importance')
            plt.title(f'Top {top_n} Features for {symbol} {model_type.replace("_", " ").title()}')
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            import base64
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return image_data
            
        except Exception as e:
            logger.error(f"Error generating feature importance chart: {str(e)}")
            return None
    
    def generate_model_performance_chart(self, symbol):
        """Generate model performance comparison chart as base64 encoded image"""
        try:
            # Check if we have performance data
            if symbol not in self.model_performance:
                logger.warning(f"No model performance data for {symbol}")
                return None
            
            # Extract performance metrics
            perf_data = self.model_performance[symbol]
            
            # Organize by model type
            model_types = {}
            for key, value in perf_data.items():
                parts = key.split('_')
                model_type = '_'.join(parts[:-1])  # Everything except the last part (algorithm)
                algorithm = parts[-1]
                
                if model_type not in model_types:
                    model_types[model_type] = {}
                
                # Store relevant metric based on model type
                if model_type == 'price_prediction':
                    metric = value.get('mse', 0)
                    metric_name = 'MSE'
                elif model_type == 'trend_prediction':
                    metric = value.get('accuracy', 0)
                    metric_name = 'Accuracy'
                else:
                    metric = 0
                    metric_name = 'Unknown'
                
                model_types[model_type][algorithm] = metric
            
            # Create chart with subplots
            fig, axs = plt.subplots(len(model_types), 1, figsize=(10, 4 * len(model_types)))
            if len(model_types) == 1:
                axs = [axs]
            
            for i, (model_type, algorithms) in enumerate(model_types.items()):
                # Sort algorithms by performance 
                sorted_algos = dict(sorted(algorithms.items(), key=lambda x: x[1], reverse=True))
                
                # Different color for MSE (lower is better)
                reverse = model_type == 'price_prediction'
                colors = ['green' if not reverse else 'red'] * len(sorted_algos)
                if len(colors) > 0:
                    colors[0] = 'blue' if not reverse else 'orange'
                
                axs[i].bar(list(sorted_algos.keys()), list(sorted_algos.values()), color=colors)
                
                # Determine metric name based on model type
                if model_type == 'price_prediction':
                    metric_name = 'MSE (lower is better)'
                elif model_type == 'trend_prediction':
                    metric_name = 'Accuracy'
                else:
                    metric_name = 'Performance'
                
                axs[i].set_title(f'{model_type.replace("_", " ").title()} - {metric_name}')
                axs[i].set_ylabel(metric_name)
                
                # Add value labels
                for j, v in enumerate(sorted_algos.values()):
                    axs[i].text(j, v, f"{v:.4f}", ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            import base64
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return image_data
            
        except Exception as e:
            logger.error(f"Error generating model performance chart: {str(e)}")
            return None
