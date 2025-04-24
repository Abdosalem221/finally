
"""
نظام إدارة المخاطر المتقدم مع تحسينات للوصول لنسبة دقة 90-99%
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats

class EnhancedRiskManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # حدود المخاطر
        self.max_risk_per_trade = 0.02  # 2% لكل صفقة
        self.max_portfolio_risk = 0.06  # 6% إجمالي
        self.min_risk_reward = 2.5  # نسبة المخاطرة/المكافأة الدنيا
        
        # مستويات التحقق
        self.verification_levels = ['price_action', 'technical', 'volume', 'volatility', 'pattern']
        self.confidence_threshold = 0.90  # عتبة الثقة الدنيا
        
    def verify_signal(self, signal: Dict, market_data: pd.DataFrame) -> Dict:
        """التحقق المتقدم من الإشارة"""
        try:
            verification_scores = []
            
            # 1. التحقق من حركة السعر
            price_score = self._verify_price_action(signal, market_data)
            verification_scores.append(('price_action', price_score))
            
            # 2. التحقق من المؤشرات الفنية
            technical_score = self._verify_technical_indicators(signal, market_data)
            verification_scores.append(('technical', technical_score))
            
            # 3. التحقق من الحجم
            volume_score = self._verify_volume(signal, market_data)
            verification_scores.append(('volume', volume_score))
            
            # 4. التحقق من التقلب
            volatility_score = self._verify_volatility(signal, market_data)
            verification_scores.append(('volatility', volatility_score))
            
            # 5. التحقق من الأنماط
            pattern_score = self._verify_patterns(signal, market_data)
            verification_scores.append(('pattern', pattern_score))
            
            # حساب درجة التحقق النهائية
            final_score = np.mean([score for _, score in verification_scores])
            
            # اكتشاف الإشارات الكاذبة
            is_false_signal = self._detect_false_signal(signal, market_data, verification_scores)
            
            return {
                'verified': final_score >= self.confidence_threshold and not is_false_signal,
                'score': final_score,
                'is_false_signal': is_false_signal,
                'verification_scores': dict(verification_scores),
                'risk_level': self._calculate_risk_level(final_score)
            }
            
        except Exception as e:
            self.logger.error(f"Error in signal verification: {str(e)}")
            return {'verified': False, 'error': str(e)}
    
    def calculate_position_size(self, account_balance: float, signal: Dict) -> float:
        """حساب حجم العقد المناسب"""
        try:
            # التحقق من نسبة المخاطرة/المكافأة
            risk_reward = signal.get('risk_reward_ratio', 0)
            if risk_reward < self.min_risk_reward:
                return 0
            
            # حساب المخاطرة المسموحة
            max_risk_amount = account_balance * self.max_risk_per_trade
            
            # تعديل حجم العقد بناءً على درجة الثقة
            confidence_score = signal.get('confidence_score', 0)
            position_size = max_risk_amount * confidence_score
            
            # التحقق من الحد الأقصى للمخاطر
            if position_size > (account_balance * self.max_portfolio_risk):
                position_size = account_balance * self.max_portfolio_risk
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0
    
    def _verify_price_action(self, signal: Dict, data: pd.DataFrame) -> float:
        """التحقق من حركة السعر"""
        try:
            # تحليل اتجاه السعر
            price_trend = self._analyze_price_trend(data)
            
            # تحليل مستويات الدعم والمقاومة
            levels = self._analyze_support_resistance(data)
            
            # تحليل قوة الحركة
            momentum = self._analyze_momentum(data)
            
            # حساب درجة التحقق
            if signal['signal_type'] in ['BUY', 'CALL']:
                score = (
                    (price_trend['bullish_strength'] * 0.4) +
                    (levels['support_strength'] * 0.3) +
                    (momentum['positive_momentum'] * 0.3)
                )
            else:
                score = (
                    (price_trend['bearish_strength'] * 0.4) +
                    (levels['resistance_strength'] * 0.3) +
                    (momentum['negative_momentum'] * 0.3)
                )
            
            return min(max(score, 0), 1)
            
        except Exception as e:
            self.logger.error(f"Error in price action verification: {str(e)}")
            return 0
    
    def _verify_technical_indicators(self, signal: Dict, data: pd.DataFrame) -> float:
        """التحقق من المؤشرات الفنية"""
        try:
            indicators_agreement = 0
            total_indicators = 0
            
            # RSI تحليل
            rsi = self._calculate_rsi(data)
            if signal['signal_type'] in ['BUY', 'CALL']:
                if rsi < 30:
                    indicators_agreement += 1
            else:
                if rsi > 70:
                    indicators_agreement += 1
            total_indicators += 1
            
            # MACD تحليل
            macd = self._calculate_macd(data)
            if signal['signal_type'] in ['BUY', 'CALL']:
                if macd['histogram'] > 0 and macd['macd'] > macd['signal']:
                    indicators_agreement += 1
            else:
                if macd['histogram'] < 0 and macd['macd'] < macd['signal']:
                    indicators_agreement += 1
            total_indicators += 1
            
            # المتوسطات المتحركة
            ma_analysis = self._analyze_moving_averages(data)
            if signal['signal_type'] in ['BUY', 'CALL']:
                if ma_analysis['trend'] == 'bullish':
                    indicators_agreement += 1
            else:
                if ma_analysis['trend'] == 'bearish':
                    indicators_agreement += 1
            total_indicators += 1
            
            return indicators_agreement / total_indicators if total_indicators > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error in technical indicators verification: {str(e)}")
            return 0
    
    def _verify_volume(self, signal: Dict, data: pd.DataFrame) -> float:
        """التحقق من الحجم"""
        try:
            # تحليل حجم التداول
            volume_analysis = self._analyze_volume(data)
            
            # حساب درجة التحقق
            if signal['signal_type'] in ['BUY', 'CALL']:
                score = (
                    (volume_analysis['buying_volume'] * 0.6) +
                    (volume_analysis['volume_trend'] * 0.4)
                )
            else:
                score = (
                    (volume_analysis['selling_volume'] * 0.6) +
                    (volume_analysis['volume_trend'] * 0.4)
                )
            
            return min(max(score, 0), 1)
            
        except Exception as e:
            self.logger.error(f"Error in volume verification: {str(e)}")
            return 0
    
    def _verify_volatility(self, signal: Dict, data: pd.DataFrame) -> float:
        """التحقق من التقلب"""
        try:
            # حساب مقاييس التقلب
            volatility = self._calculate_volatility_metrics(data)
            
            # تحليل مستوى التقلب
            if volatility['current'] > volatility['historical_high']:
                return 0.3  # تقلب مرتفع جداً - خطر
            elif volatility['current'] < volatility['historical_low']:
                return 0.7  # تقلب منخفض - جيد
            else:
                return 0.5  # تقلب معتدل
                
        except Exception as e:
            self.logger.error(f"Error in volatility verification: {str(e)}")
            return 0
    
    def _verify_patterns(self, signal: Dict, data: pd.DataFrame) -> float:
        """التحقق من الأنماط"""
        try:
            patterns_found = self._detect_patterns(data)
            pattern_score = 0
            
            for pattern in patterns_found:
                if signal['signal_type'] in ['BUY', 'CALL']:
                    if pattern['type'] == 'bullish':
                        pattern_score += pattern['strength']
                else:
                    if pattern['type'] == 'bearish':
                        pattern_score += pattern['strength']
            
            return min(pattern_score / len(patterns_found) if patterns_found else 0, 1)
            
        except Exception as e:
            self.logger.error(f"Error in pattern verification: {str(e)}")
            return 0
    
    def _detect_false_signal(self, signal: Dict, data: pd.DataFrame, verification_scores: List[Tuple[str, float]]) -> bool:
        """اكتشاف الإشارات الكاذبة"""
        try:
            # 1. التحقق من تناقض المؤشرات
            indicators_conflict = self._check_indicators_conflict(data)
            
            # 2. التحقق من ضعف الحجم
            weak_volume = self._check_weak_volume(data)
            
            # 3. التحقق من التقلب غير الطبيعي
            abnormal_volatility = self._check_abnormal_volatility(data)
            
            # 4. التحقق من الأنماط المضللة
            misleading_patterns = self._check_misleading_patterns(data)
            
            # 5. التحقق من تناقض الأطر الزمنية
            timeframe_conflict = self._check_timeframe_conflict(signal)
            
            # حساب عدد العوامل السلبية
            negative_factors = sum([
                indicators_conflict,
                weak_volume,
                abnormal_volatility,
                misleading_patterns,
                timeframe_conflict
            ])
            
            # التحقق من درجات التحقق المنخفضة
            low_scores = sum(1 for _, score in verification_scores if score < 0.5)
            
            return negative_factors >= 2 or low_scores >= 3
            
        except Exception as e:
            self.logger.error(f"Error in false signal detection: {str(e)}")
            return True
    
    def _calculate_risk_level(self, verification_score: float) -> str:
        """حساب مستوى المخاطرة"""
        if verification_score >= 0.9:
            return 'LOW'
        elif verification_score >= 0.7:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _analyze_price_trend(self, data: pd.DataFrame) -> Dict:
        """تحليل اتجاه السعر"""
        try:
            close_prices = data['close'].values
            
            # حساب القوة الصعودية والهبوطية
            upward_movement = np.sum(np.diff(close_prices) > 0)
            downward_movement = np.sum(np.diff(close_prices) < 0)
            
            total_movement = upward_movement + downward_movement
            
            return {
                'bullish_strength': upward_movement / total_movement if total_movement > 0 else 0,
                'bearish_strength': downward_movement / total_movement if total_movement > 0 else 0
            }
        except Exception as e:
            self.logger.error(f"Error in price trend analysis: {str(e)}")
            return {'bullish_strength': 0, 'bearish_strength': 0}
    
    def _analyze_support_resistance(self, data: pd.DataFrame) -> Dict:
        """تحليل مستويات الدعم والمقاومة"""
        try:
            highs = data['high'].values
            lows = data['low'].values
            close = data['close'].values[-1]
            
            # تحديد مستويات الدعم والمقاومة
            resistance_levels = self._find_levels(highs)
            support_levels = self._find_levels(lows)
            
            # حساب قوة المستويات
            resistance_strength = self._calculate_level_strength(close, resistance_levels, 'resistance')
            support_strength = self._calculate_level_strength(close, support_levels, 'support')
            
            return {
                'resistance_strength': resistance_strength,
                'support_strength': support_strength,
                'resistance_levels': resistance_levels,
                'support_levels': support_levels
            }
        except Exception as e:
            self.logger.error(f"Error in support/resistance analysis: {str(e)}")
            return {'resistance_strength': 0, 'support_strength': 0}
    
    def _analyze_momentum(self, data: pd.DataFrame) -> Dict:
        """تحليل قوة الحركة"""
        try:
            close_prices = data['close'].values
            momentum = np.diff(close_prices)
            
            positive_momentum = np.sum(momentum > 0) / len(momentum)
            negative_momentum = np.sum(momentum < 0) / len(momentum)
            
            return {
                'positive_momentum': positive_momentum,
                'negative_momentum': negative_momentum
            }
        except Exception as e:
            self.logger.error(f"Error in momentum analysis: {str(e)}")
            return {'positive_momentum': 0, 'negative_momentum': 0}
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict:
        """تحليل حجم التداول"""
        try:
            volume = data['volume'].values
            close_prices = data['close'].values
            price_changes = np.diff(close_prices)
            
            # حساب حجم البيع والشراء
            buying_volume = np.sum(volume[1:][price_changes > 0])
            selling_volume = np.sum(volume[1:][price_changes < 0])
            total_volume = buying_volume + selling_volume
            
            # حساب اتجاه الحجم
            volume_sma = np.mean(volume[-20:])
            current_volume = volume[-1]
            volume_trend = current_volume / volume_sma if volume_sma > 0 else 0
            
            return {
                'buying_volume': buying_volume / total_volume if total_volume > 0 else 0,
                'selling_volume': selling_volume / total_volume if total_volume > 0 else 0,
                'volume_trend': min(volume_trend, 1)
            }
        except Exception as e:
            self.logger.error(f"Error in volume analysis: {str(e)}")
            return {'buying_volume': 0, 'selling_volume': 0, 'volume_trend': 0}
    
    def _calculate_volatility_metrics(self, data: pd.DataFrame) -> Dict:
        """حساب مقاييس التقلب"""
        try:
            returns = np.diff(np.log(data['close'].values))
            current_volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0
            historical_volatility = np.std(returns) if len(returns) > 0 else 0
            
            return {
                'current': current_volatility,
                'historical': historical_volatility,
                'historical_high': np.percentile(returns, 95) if len(returns) > 0 else 0,
                'historical_low': np.percentile(returns, 5) if len(returns) > 0 else 0
            }
        except Exception as e:
            self.logger.error(f"Error calculating volatility metrics: {str(e)}")
            return {'current': 0, 'historical': 0, 'historical_high': 0, 'historical_low': 0}
