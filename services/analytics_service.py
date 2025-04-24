"""
خدمة التحليلات المتقدمة
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from app.utils.validators import validate_market_data, validate_technical_indicators
from app.utils.helpers import calculate_risk_reward_ratio

class AnalyticsService:
    def __init__(self):
        self.market_data_cache = {}
        self.indicators_cache = {}
        self.performance_metrics = {}
    
    def analyze_market_data(self, market_data: List[Dict], use_high_precision: bool = True) -> Dict:
    """
    تحليل بيانات السوق مع دعم التحليل عالي الدقة
    """
    if use_high_precision:
        from app.high_precision_enhancements import HighPrecisionEnhancements
        enhancer = HighPrecisionEnhancements()
        return enhancer.apply_enhancements(market_data)
        """تحليل بيانات السوق"""
        try:
            # التحقق من صحة البيانات
            for data in market_data:
                error = validate_market_data(data)
                if error:
                    return {'status': 'error', 'message': error}
            
            # تحويل البيانات إلى DataFrame
            df = pd.DataFrame(market_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # حساب المتوسطات المتحركة
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()
            
            # حساب مؤشر RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # حساب Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
            
            # حساب MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['histogram'] = df['macd'] - df['signal']
            
            # تحليل الاتجاه
            df['trend'] = np.where(df['sma_20'] > df['sma_50'], 'UP', 'DOWN')
            
            # حساب التقلب
            df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean() * 100
            
            # تحليل الحجم
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_trend'] = np.where(df['volume'] > df['volume_sma'], 'HIGH', 'LOW')
            
            # تحليل القوة النسبية
            df['relative_strength'] = df['close'] / df['sma_200'] * 100
            
            # تحليل النمط
            df['pattern'] = self._identify_patterns(df)
            
            # تحليل الدعم والمقاومة
            support_resistance = self._find_support_resistance(df)
            
            # تحليل المخاطر
            risk_analysis = self._analyze_risk(df)
            
            return {
                'status': 'success',
                'data': {
                    'technical_indicators': {
                        'moving_averages': {
                            'sma_20': df['sma_20'].iloc[-1],
                            'sma_50': df['sma_50'].iloc[-1],
                            'sma_200': df['sma_200'].iloc[-1]
                        },
                        'rsi': df['rsi'].iloc[-1],
                        'bollinger_bands': {
                            'upper': df['bb_upper'].iloc[-1],
                            'middle': df['bb_middle'].iloc[-1],
                            'lower': df['bb_lower'].iloc[-1]
                        },
                        'macd': {
                            'macd': df['macd'].iloc[-1],
                            'signal': df['signal'].iloc[-1],
                            'histogram': df['histogram'].iloc[-1]
                        }
                    },
                    'trend_analysis': {
                        'current_trend': df['trend'].iloc[-1],
                        'volatility': df['volatility'].iloc[-1],
                        'volume_trend': df['volume_trend'].iloc[-1],
                        'relative_strength': df['relative_strength'].iloc[-1],
                        'pattern': df['pattern'].iloc[-1]
                    },
                    'support_resistance': support_resistance,
                    'risk_analysis': risk_analysis
                }
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _identify_patterns(self, df: pd.DataFrame) -> pd.Series:
        """تحديد الأنماط الفنية"""
        patterns = pd.Series(index=df.index, dtype='object')
        
        # تحديد نمط الرأس والكتفين
        for i in range(2, len(df)-2):
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i+1] and
                df['high'].iloc[i-1] < df['high'].iloc[i-2] and
                df['high'].iloc[i+1] < df['high'].iloc[i+2]):
                patterns.iloc[i] = 'HEAD_AND_SHOULDERS'
        
        # تحديد نمط القمة المزدوجة
        for i in range(1, len(df)-1):
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i+1] and
                abs(df['high'].iloc[i] - df['high'].iloc[i-1]) < 0.01 * df['high'].iloc[i]):
                patterns.iloc[i] = 'DOUBLE_TOP'
        
        # تحديد نمط القاع المزدوج
        for i in range(1, len(df)-1):
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                df['low'].iloc[i] < df['low'].iloc[i+1] and
                abs(df['low'].iloc[i] - df['low'].iloc[i-1]) < 0.01 * df['low'].iloc[i]):
                patterns.iloc[i] = 'DOUBLE_BOTTOM'
        
        return patterns
    
    def _find_support_resistance(self, df: pd.DataFrame) -> Dict:
        """إيجاد مستويات الدعم والمقاومة"""
        # استخدام Bollinger Bands لتحديد المستويات
        support_levels = df[df['close'] < df['bb_lower']]['close'].unique()
        resistance_levels = df[df['close'] > df['bb_upper']]['close'].unique()
        
        # تصفية المستويات القريبة من بعضها
        support_levels = self._filter_nearby_levels(support_levels)
        resistance_levels = self._filter_nearby_levels(resistance_levels)
        
        return {
            'support': sorted(support_levels),
            'resistance': sorted(resistance_levels)
        }
    
    def _filter_nearby_levels(self, levels: np.ndarray) -> np.ndarray:
        """تصفية المستويات القريبة من بعضها"""
        if len(levels) == 0:
            return levels
        
        filtered_levels = [levels[0]]
        for level in levels[1:]:
            if abs(level - filtered_levels[-1]) > 0.01 * filtered_levels[-1]:
                filtered_levels.append(level)
        
        return np.array(filtered_levels)
    
    def _analyze_risk(self, df: pd.DataFrame) -> Dict:
        """تحليل المخاطر"""
        # حساب التقلب
        volatility = df['close'].pct_change().std() * np.sqrt(252) * 100
        
        # حساب الحد الأقصى للانخفاض
        rolling_max = df['close'].expanding().max()
        drawdown = (df['close'] - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # حساب نسبة شارب
        returns = df['close'].pct_change()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        
        # حساب نسبة سورتينو
        downside_returns = returns[returns < 0]
        sortino_ratio = np.sqrt(252) * returns.mean() / downside_returns.std()
        
        return {
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio
        }
    
    def calculate_performance_metrics(self, signals: List[Dict]) -> Dict:
        """حساب مقاييس الأداء"""
        try:
            if not signals:
                return {'status': 'error', 'message': 'No signals provided'}
            
            # تحويل الإشارات إلى DataFrame
            df = pd.DataFrame(signals)
            
            # حساب معدل النجاح
            total_signals = len(df)
            successful_signals = len(df[df['result'] == 'WIN'])
            success_rate = (successful_signals / total_signals) * 100
            
            # حساب متوسط الربح والخسارة
            avg_profit = df[df['result'] == 'WIN']['profit'].mean()
            avg_loss = df[df['result'] == 'LOSS']['loss'].mean()
            
            # حساب نسبة الربح إلى الخسارة
            profit_factor = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
            
            # حساب نسبة المخاطرة إلى العائد
            risk_reward_ratios = []
            for _, signal in df.iterrows():
                ratio = calculate_risk_reward_ratio(
                    signal['entry_price'],
                    signal['stop_loss'],
                    signal['take_profit'],
                    signal['signal_type']
                )
                if ratio is not None:
                    risk_reward_ratios.append(ratio)
            
            avg_risk_reward = sum(risk_reward_ratios) / len(risk_reward_ratios) if risk_reward_ratios else 0
            
            # حساب نسبة الفوز المتتالية
            consecutive_wins = 0
            max_consecutive_wins = 0
            for result in df['result']:
                if result == 'WIN':
                    consecutive_wins += 1
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                else:
                    consecutive_wins = 0
            
            # حساب نسبة الخسارة المتتالية
            consecutive_losses = 0
            max_consecutive_losses = 0
            for result in df['result']:
                if result == 'LOSS':
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                else:
                    consecutive_losses = 0
            
            return {
                'status': 'success',
                'metrics': {
                    'total_signals': total_signals,
                    'success_rate': success_rate,
                    'avg_profit': avg_profit,
                    'avg_loss': avg_loss,
                    'profit_factor': profit_factor,
                    'avg_risk_reward': avg_risk_reward,
                    'max_consecutive_wins': max_consecutive_wins,
                    'max_consecutive_losses': max_consecutive_losses
                }
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def generate_trading_signals(self, market_data: List[Dict], indicators: Dict) -> Dict:
        """إنشاء إشارات التداول"""
        try:
            # التحقق من صحة البيانات
            for data in market_data:
                error = validate_market_data(data)
                if error:
                    return {'status': 'error', 'message': error}
            
            error = validate_technical_indicators(indicators)
            if error:
                return {'status': 'error', 'message': error}
            
            # تحليل البيانات والمؤشرات
            analysis = self.analyze_market_data(market_data)
            if analysis['status'] == 'error':
                return analysis
            
            # استخراج النتائج
            technical_indicators = analysis['data']['technical_indicators']
            trend_analysis = analysis['data']['trend_analysis']
            support_resistance = analysis['data']['support_resistance']
            risk_analysis = analysis['data']['risk_analysis']
            
            # توليد الإشارات بناءً على التحليل
            signals = []
            
            # إشارة شراء
            if (technical_indicators['rsi'] < 30 and 
                trend_analysis['current_trend'] == 'UP' and
                technical_indicators['macd']['histogram'] > 0):
                signals.append({
                    'type': 'BUY',
                    'entry_price': market_data[-1]['close'],
                    'stop_loss': min(support_resistance['support']),
                    'take_profit': max(support_resistance['resistance']),
                    'confidence': self._calculate_confidence(technical_indicators, trend_analysis)
                })
            
            # إشارة بيع
            if (technical_indicators['rsi'] > 70 and 
                trend_analysis['current_trend'] == 'DOWN' and
                technical_indicators['macd']['histogram'] < 0):
                signals.append({
                    'type': 'SELL',
                    'entry_price': market_data[-1]['close'],
                    'stop_loss': max(support_resistance['resistance']),
                    'take_profit': min(support_resistance['support']),
                    'confidence': self._calculate_confidence(technical_indicators, trend_analysis)
                })
            
            return {
                'status': 'success',
                'signals': signals,
                'analysis': {
                    'technical_indicators': technical_indicators,
                    'trend_analysis': trend_analysis,
                    'support_resistance': support_resistance,
                    'risk_analysis': risk_analysis
                }
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _calculate_confidence(self, technical_indicators: Dict, trend_analysis: Dict) -> float:
        """حساب ثقة الإشارة"""
        confidence = 0.0
        
        # إضافة وزن لمؤشر RSI
        rsi = technical_indicators['rsi']
        if rsi < 20 or rsi > 80:
            confidence += 0.3
        elif rsi < 30 or rsi > 70:
            confidence += 0.2
        elif rsi < 40 or rsi > 60:
            confidence += 0.1
        
        # إضافة وزن لـ MACD
        macd_hist = technical_indicators['macd']['histogram']
        if abs(macd_hist) > 0.5:
            confidence += 0.2
        elif abs(macd_hist) > 0.2:
            confidence += 0.1
        
        # إضافة وزن للاتجاه
        if trend_analysis['current_trend'] == 'UP':
            confidence += 0.2
        elif trend_analysis['current_trend'] == 'DOWN':
            confidence += 0.2
        
        # إضافة وزن للتقلب
        if trend_analysis['volatility'] < 20:
            confidence += 0.1
        
        # إضافة وزن للحجم
        if trend_analysis['volume_trend'] == 'HIGH':
            confidence += 0.1
        
        # إضافة وزن للقوة النسبية
        if trend_analysis['relative_strength'] > 100:
            confidence += 0.1
        
        return min(confidence, 1.0) 