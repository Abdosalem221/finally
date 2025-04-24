import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class DeepLearningPredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()

    def prepare_data(self, data):
        """تحضير البيانات للتدريب"""
        try:
            # تحويل البيانات إلى نطاق [0,1]
            scaled_data = self.scaler.fit_transform(data)

            X, y = [], []
            for i in range(len(scaled_data) - self.sequence_length):
                X.append(scaled_data[i:(i + self.sequence_length)])
                y.append(scaled_data[i + self.sequence_length])

            return np.array(X), np.array(y)
        except Exception as e:
            logger.error(f"خطأ في تحضير البيانات: {str(e)}")
            return None, None

    def build_lstm_model(self, input_shape):
        """Build enhanced LSTM model with attention mechanism"""
        model = Sequential([
            LSTM(units=128, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            Bidirectional(LSTM(units=64, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.3),
            Attention(), #Assuming Attention class is defined elsewhere
            Dense(units=32, activation='relu'),
            BatchNormalization(),
            Dense(units=1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    def build_gru_model(self, input_shape):
        """بناء نموذج GRU"""
        model = Sequential([
            GRU(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(units=50, return_sequences=True),
            Dropout(0.2),
            GRU(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        model.compile(optimizer=Adam(), loss='mean_squared_error')
        return model

    def train(self, data, model_type='lstm', epochs=50, batch_size=32):
        """تدريب النموذج"""
        try:
            X, y = self.prepare_data(data)
            if X is None or y is None:
                return False

            input_shape = (X.shape[1], X.shape[2])
            if model_type == 'lstm':
                self.model = self.build_lstm_model(input_shape)
            else:
                self.model = self.build_gru_model(input_shape)

            self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1)
            return True
        except Exception as e:
            logger.error(f"خطأ في تدريب النموذج: {str(e)}")
            return False

    def predict(self, data):
        """التنبؤ باستخدام النموذج"""
        try:
            if self.model is None:
                return None

            scaled_data = self.scaler.transform(data)
            X = np.array([scaled_data[-self.sequence_length:]])
            prediction = self.model.predict(X)

            return self.scaler.inverse_transform(prediction)[0]
        except Exception as e:
            logger.error(f"خطأ في التنبؤ: {str(e)}")
            return None
"""
نماذج التعلم العميق المتقدمة للتداول
"""

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np

class MarketPredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = self._build_model()
        
    def _build_model(self):
        model = tf.keras.Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, 5)),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(16, activation='relu'),
            Dense(1, activation='tanh')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        return model
        
    def prepare_data(self, data):
        """تحضير البيانات للتدريب"""
        features = ['open', 'high', 'low', 'close', 'volume']
        X = []
        y = []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[features].values[i:i+self.sequence_length])
            y.append(data['close'].values[i+self.sequence_length])
            
        return np.array(X), np.array(y)
        
    def train(self, data, epochs=50, batch_size=32):
        """تدريب النموذج"""
        X, y = self.prepare_data(data)
        return self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        
    def predict(self, data):
        """التنبؤ بالأسعار المستقبلية"""
        X, _ = self.prepare_data(data)
        return self.model.predict(X)

class PatternRecognizer:
    def __init__(self):
        self.model = self._build_model()
        
    def _build_model(self):
        model = tf.keras.Sequential([
            LSTM(64, return_sequences=True, input_shape=(30, 5)),
            Dropout(0.2),
            LSTM(32),
            Dense(16, activation='relu'),
            Dense(4, activation='softmax')  # تصنيف 4 أنماط رئيسية
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

class MarketRegimeClassifier:
    def __init__(self):
        self.model = self._build_model()
        
    def _build_model(self):
        model = tf.keras.Sequential([
            Dense(32, activation='relu', input_shape=(10,)),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')  # تصنيف 3 أنظمة سوق
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
