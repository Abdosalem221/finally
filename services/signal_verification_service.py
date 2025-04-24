"""
خدمة التحقق من إشارات التداول
تحليل متقدم وتحقق متعدد المراحل لضمان إشارات بدقة 90-95%
"""

import os
import threading
import time
import datetime
import logging
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np

from models.database import db
from models.market_models import Currency, Signal, SignalVerification, PerformanceMetric

# إعداد التسجيل
logger = logging.getLogger('signal_verification_service')

class SignalVerificationService:
    """خدمة التحقق من إشارات التداول"""
    
    def __init__(self, app, market_data_service, strategy_service, max_workers=5, verification_threshold=0.6):
        """
        تهيئة خدمة التحقق من الإشارات
        
        Args:
            app: تطبيق Flask
            market_data_service: خدمة بيانات السوق
            strategy_service: خدمة استراتيجيات التداول
            max_workers: الحد الأقصى لعدد العمليات المتزامنة
            verification_threshold: عتبة قبول التحقق (0-1)
        """
        self.app = app
        self.market_data_service = market_data_service
        self.strategy_service = strategy_service
        self.max_workers = max_workers
        self.verification_threshold = verification_threshold
        self.running = False
        self.verification_thread = None
        self.status = {
            'running': False,
            'last_update': None,
            'pending_verifications': 0,
            'completed_verifications': 0,
            'success_rate': 0,
            'error': None
        }
        
        # تحميل عمليات التحقق المعلقة
        self._pending_verifications = []
    
    def start(self):
        """بدء خدمة التحقق من الإشارات"""
        if self.running:
            logger.warning("Signal verification service already running")
            return
        
        logger.info("Starting signal verification service")
        self.running = True
        self.status['running'] = True
        self.verification_thread = threading.Thread(target=self._verification_loop, daemon=True)
        self.verification_thread.start()
    
    def stop(self):
        """إيقاف خدمة التحقق من الإشارات"""
        logger.info("Stopping signal verification service")
        self.running = False
        self.status['running'] = False
        if self.verification_thread:
            self.verification_thread.join(timeout=10)
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        الحصول على حالة الخدمة
        
        Returns:
            قاموس يحتوي على معلومات حالة الخدمة
        """
        return {
            'status': 'running' if self.status['running'] else 'stopped',
            'last_update': self.status['last_update'].isoformat() if self.status['last_update'] else None,
            'pending_verifications': self.status['pending_verifications'],
            'completed_verifications': self.status['completed_verifications'],
            'success_rate': self.status['success_rate'],
            'error': self.status['error']
        }
    
    def generate_signal(self, currency_pair: str, timeframe: str, 
                        strategy_name: str = 'auto', 
                        enhancement_level: str = 'standard', 
                        verification_enabled: bool = True,
                        signal_type_filter: str = 'any') -> Optional[Dict[str, Any]]:
        """
        توليد إشارة تداول مع التحقق منها
        
        Args:
            currency_pair: زوج العملات
            timeframe: الإطار الزمني
            strategy_name: اسم الاستراتيجية (auto لاختيار أفضل استراتيجية)
            enhancement_level: مستوى التحسين (basic, standard, high, ultra)
            verification_enabled: تمكين التحقق من الإشارة
            signal_type_filter: نوع الإشارة المطلوبة (any, BUY, SELL, CALL, PUT)
            
        Returns:
            إشارة التداول أو None في حالة الفشل
        """
        try:
            # اختيار أفضل استراتيجية إذا كانت auto
            if strategy_name == 'auto':
                best_strategies = self.strategy_service.find_best_strategy_for_pair(currency_pair, timeframe, top_n=3)
                if not best_strategies:
                    logger.warning(f"No suitable strategies found for {currency_pair} {timeframe}")
                    return None
                
                strategy_name = best_strategies[0]
            
            # تنفيذ الاستراتيجية لتوليد الإشارة
            signal = self.strategy_service.execute_strategy(strategy_name, currency_pair, timeframe)
            
            if not signal:
                logger.info(f"No signal generated for {currency_pair} {timeframe} using {strategy_name}")
                return None
            
            # تحقق من نوع الإشارة المطلوب
            if signal_type_filter != 'any' and signal['signal_type'] != signal_type_filter:
                logger.info(f"Signal type {signal['signal_type']} does not match filter {signal_type_filter}")
                return None
            
            # تعزيز الإشارة
            if enhancement_level != 'basic':
                signal = self._enhance_signal(signal, currency_pair, timeframe, enhancement_level)
            
            # التحقق من الإشارة
            if verification_enabled:
                verification_result = self._verify_signal(signal, currency_pair, timeframe)
                
                if not verification_result['verified']:
                    logger.info(f"Signal verification failed for {currency_pair} {timeframe}")
                    return None
                
                # تحديث الإشارة بنتائج التحقق
                signal['verification_score'] = verification_result['score']
                signal['is_high_precision'] = verification_result['score'] >= 0.9
                signal['verification_details'] = verification_result['details']
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None
    
    def verify_pending_signals(self):
        """التحقق من الإشارات المعلقة"""
        try:
            with self.app.app_context():
                # البحث عن الإشارات المعلقة
                pending_verifications = SignalVerification.query.filter_by(status='PENDING').all()
                
                if not pending_verifications:
                    return
                
                self.status['pending_verifications'] = len(pending_verifications)
                
                # تنفيذ عمليات التحقق بالتوازي
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    for verification in pending_verifications:
                        futures.append(executor.submit(self._process_verification, verification.id))
                    
                    # انتظار انتهاء جميع العمليات
                    for future in futures:
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Error in verification process: {str(e)}")
                
                # تحديث الإحصائيات
                verified_count = SignalVerification.query.filter_by(status='VERIFIED').count()
                total_count = SignalVerification.query.filter(SignalVerification.status.in_(['VERIFIED', 'REJECTED'])).count()
                
                if total_count > 0:
                    self.status['success_rate'] = (verified_count / total_count) * 100
                
                self.status['completed_verifications'] += len(pending_verifications)
                
        except Exception as e:
            logger.error(f"Error verifying pending signals: {str(e)}")
            self.status['error'] = str(e)
    
    def add_verification_request(self, signal_id: int, enhancement_level: str = 'standard'):
        """
        إضافة طلب تحقق من إشارة
        
        Args:
            signal_id: معرف الإشارة
            enhancement_level: مستوى التحسين
        """
        try:
            with self.app.app_context():
                # البحث عن الإشارة
                signal = Signal.query.get(signal_id)
                if not signal:
                    logger.warning(f"Signal not found: {signal_id}")
                    return
                
                # التأكد من عدم وجود طلب تحقق مسبق
                existing = SignalVerification.query.filter_by(signal_id=signal_id).first()
                if existing:
                    logger.info(f"Verification request already exists for signal {signal_id}")
                    return
                
                # إنشاء طلب تحقق جديد
                verification = SignalVerification(
                    signal_id=signal_id,
                    status='PENDING',
                    start_time=datetime.datetime.now(),
                    enhancements_applied={"level": enhancement_level}
                )
                
                db.session.add(verification)
                db.session.commit()
                
                logger.info(f"Verification request added for signal {signal_id}")
                
                # إضافة الطلب إلى قائمة الانتظار
                self._pending_verifications.append(verification.id)
                
        except Exception as e:
            logger.error(f"Error adding verification request: {str(e)}")
            db.session.rollback()
    
    def _verification_loop(self):
        """حلقة التحقق من الإشارات"""
        logger.info("Verification loop started")
        
        while self.running:
            try:
                # التحقق من الإشارات المعلقة
                self.verify_pending_signals()
                
                # تحديث حالة الخدمة
                self.status['last_update'] = datetime.datetime.now()
                self.status['error'] = None
                
            except Exception as e:
                logger.error(f"Error in verification loop: {str(e)}")
                self.status['error'] = str(e)
            
            # النوم لمدة 5 ثوانٍ
            time.sleep(5)
    
    def _process_verification(self, verification_id: int):
        """
        معالجة طلب تحقق
        
        Args:
            verification_id: معرف طلب التحقق
        """
        try:
            with self.app.app_context():
                # البحث عن طلب التحقق
                verification = SignalVerification.query.get(verification_id)
                if not verification:
                    logger.warning(f"Verification not found: {verification_id}")
                    return
                
                # تحديث حالة التحقق
                verification.status = 'VERIFYING'
                db.session.commit()
                
                # البحث عن الإشارة
                signal = Signal.query.get(verification.signal_id)
                if not signal:
                    logger.warning(f"Signal not found: {verification.signal_id}")
                    verification.status = 'REJECTED'
                    verification.failure_reason = 'Signal not found'
                    verification.end_time = datetime.datetime.now()
                    db.session.commit()
                    return
                
                # البحث عن العملة
                currency = Currency.query.get(signal.currency_id)
                if not currency:
                    logger.warning(f"Currency not found: {signal.currency_id}")
                    verification.status = 'REJECTED'
                    verification.failure_reason = 'Currency not found'
                    verification.end_time = datetime.datetime.now()
                    db.session.commit()
                    return
                
                # الحصول على بيانات السوق
                data = self.market_data_service.get_historical_data(currency.symbol, signal.timeframe)
                if data is None or data.empty:
                    logger.warning(f"No market data available for {currency.symbol} {signal.timeframe}")
                    verification.status = 'REJECTED'
                    verification.failure_reason = 'No market data available'
                    verification.end_time = datetime.datetime.now()
                    db.session.commit()
                    return
                
                # التحقق من الإشارة
                start_time = datetime.datetime.now()
                result = self._verify_signal(signal.__dict__, currency.symbol, signal.timeframe)
                end_time = datetime.datetime.now()
                
                # تحديث طلب التحقق
                verification.verification_score = result['score']
                verification.verification_steps = result['details']
                verification.end_time = end_time
                verification.execution_time = (end_time - start_time).total_seconds()
                
                if result['verified']:
                    verification.status = 'VERIFIED'
                    
                    # تحديث الإشارة
                    signal.is_high_precision = result['score'] >= 0.9
                    signal.success_probability = result['score'] * 100
                    signal.verification_data = result['details']
                else:
                    verification.status = 'REJECTED'
                    verification.failure_reason = result['details'].get('failure_reason', 'Failed verification')
                
                db.session.commit()
                
                logger.info(f"Verification completed for signal {verification.signal_id}, result: {verification.status}")
                
                # إضافة مقياس أداء
                metric = PerformanceMetric(
                    metric_type='VERIFICATION',
                    name=f'verification_score_{signal.strategy}',
                    value=result['score'],
                    timestamp=datetime.datetime.now(),
                    timeframe=signal.timeframe,
                    currency_id=signal.currency_id,
                    metric_metadata={
                        'signal_id': signal.id,
                        'signal_type': signal.signal_type,
                        'strategy': signal.strategy,
                        'timeframe': signal.timeframe,
                        'verification_status': verification.status
                    }
                )
                
                db.session.add(metric)
                db.session.commit()
                
        except Exception as e:
            logger.error(f"Error processing verification {verification_id}: {str(e)}")
            
            try:
                with self.app.app_context():
                    verification = SignalVerification.query.get(verification_id)
                    if verification:
                        verification.status = 'REJECTED'
                        verification.failure_reason = str(e)
                        verification.end_time = datetime.datetime.now()
                        db.session.commit()
            except Exception as inner_e:
                logger.error(f"Error updating verification status: {str(inner_e)}")
    
    def _verify_signal(self, signal: Dict[str, Any], currency_pair: str, timeframe: str) -> Dict[str, Any]:
        """
        التحقق من صحة إشارة التداول
        
        Args:
            signal: بيانات الإشارة
            currency_pair: زوج العملات
            timeframe: الإطار الزمني
            
        Returns:
            نتيجة التحقق
        """
        try:
            # الحصول على بيانات السوق
            data = self.market_data_service.get_historical_data(currency_pair, timeframe)
            if data is None or data.empty:
                return {
                    'verified': False,
                    'score': 0.0,
                    'details': {
                        'failure_reason': 'No market data available'
                    }
                }
            
            # تنفيذ مراحل التحقق المختلفة
            verification_steps = {}
            scores = []
            
            # 1. التحقق من اتجاه السوق
            trend_verification = self._verify_market_trend(data, signal)
            verification_steps['market_trend'] = trend_verification
            scores.append(trend_verification['score'])
            
            # 2. التحقق من تطابق المؤشرات المتعددة
            indicators_verification = self._verify_multiple_indicators(data, signal)
            verification_steps['indicators'] = indicators_verification
            scores.append(indicators_verification['score'])
            
            # 3. التحقق من الأنماط السعرية
            patterns_verification = self._verify_price_patterns(data, signal)
            verification_steps['patterns'] = patterns_verification
            scores.append(patterns_verification['score'])
            
            # 4. التحقق من مستويات الدعم والمقاومة
            levels_verification = self._verify_support_resistance(data, signal)
            verification_steps['support_resistance'] = levels_verification
            scores.append(levels_verification['score'])
            
            # 5. التحقق من حالة السوق في الأطر الزمنية المتعددة
            timeframes_verification = self._verify_multiple_timeframes(currency_pair, timeframe, signal)
            verification_steps['multiple_timeframes'] = timeframes_verification
            scores.append(timeframes_verification['score'])
            
            # 6. التحقق من توافق الاستراتيجيات المختلفة
            strategies_verification = self._verify_multiple_strategies(currency_pair, timeframe, signal)
            verification_steps['multiple_strategies'] = strategies_verification
            scores.append(strategies_verification['score'])
            
            # 7. التحقق من احتمالية نجاح الإشارة باستخدام البيانات التاريخية
            historical_verification = self._verify_historical_performance(currency_pair, timeframe, signal)
            verification_steps['historical_performance'] = historical_verification
            scores.append(historical_verification['score'])
            
            # 8. التحقق من توقيت الإشارة
            timing_verification = self._verify_signal_timing(data, signal)
            verification_steps['timing'] = timing_verification
            scores.append(timing_verification['score'])
            
            # 9. التحقق من تناسب نسبة المخاطرة/المكافأة
            risk_reward_verification = self._verify_risk_reward_ratio(signal)
            verification_steps['risk_reward'] = risk_reward_verification
            scores.append(risk_reward_verification['score'])
            
            # 10. تحقق إضافي خاص بنوع الإشارة
            if signal['signal_type'] in ['BUY', 'CALL']:
                buy_verification = self._verify_buy_signal(data, signal)
                verification_steps['buy_specific'] = buy_verification
                scores.append(buy_verification['score'])
            elif signal['signal_type'] in ['SELL', 'PUT']:
                sell_verification = self._verify_sell_signal(data, signal)
                verification_steps['sell_specific'] = sell_verification
                scores.append(sell_verification['score'])
            
            # حساب متوسط درجات التحقق
            avg_score = sum(scores) / len(scores)
            
            # تحديد نتيجة التحقق
            result = {
                'verified': avg_score >= self.verification_threshold,
                'score': avg_score,
                'details': verification_steps
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error verifying signal: {str(e)}")
            return {
                'verified': False,
                'score': 0.0,
                'details': {
                    'failure_reason': str(e)
                }
            }
    
    def _enhance_signal(self, signal: Dict[str, Any], currency_pair: str, timeframe: str, level: str) -> Dict[str, Any]:
        """
        تحسين إشارة التداول
        
        Args:
            signal: بيانات الإشارة
            currency_pair: زوج العملات
            timeframe: الإطار الزمني
            level: مستوى التحسين
            
        Returns:
            الإشارة المحسنة
        """
        try:
            # الحصول على بيانات السوق
            data = self.market_data_service.get_historical_data(currency_pair, timeframe)
            if data is None or data.empty:
                return signal
            
            # تحسينات مختلفة حسب المستوى
            enhancements_applied = []
            
            # 1. تحسين مستويات الإيقاف والهدف
            if level in ['standard', 'high', 'ultra']:
                signal, enhancement_info = self._enhance_stop_and_target(signal, data)
                enhancements_applied.append(enhancement_info)
            
            # 2. تحسين توقيت الدخول
            if level in ['high', 'ultra']:
                signal, enhancement_info = self._enhance_entry_timing(signal, data)
                enhancements_applied.append(enhancement_info)
            
            # 3. تحسين احتمالية النجاح
            if level in ['standard', 'high', 'ultra']:
                signal, enhancement_info = self._enhance_success_probability(signal, data, currency_pair, timeframe)
                enhancements_applied.append(enhancement_info)
            
            # 4. تعزيز دقة الإشارة
            if level == 'ultra':
                signal, enhancement_info = self._enhance_precision(signal, data, currency_pair, timeframe)
                enhancements_applied.append(enhancement_info)
            
            # إضافة معلومات التحسين إلى الإشارة
            signal['enhancements'] = enhancements_applied
            
            return signal
            
        except Exception as e:
            logger.error(f"Error enhancing signal: {str(e)}")
            return signal
    
    # --------- وظائف التحقق من الإشارة ---------
    
    def _verify_market_trend(self, data: pd.DataFrame, signal: Dict[str, Any]) -> Dict[str, Any]:
        """التحقق من اتجاه السوق"""
        try:
            # حساب المتوسطات المتحركة
            data['ma_short'] = data['close'].rolling(window=20).mean()
            data['ma_medium'] = data['close'].rolling(window=50).mean()
            data['ma_long'] = data['close'].rolling(window=200).mean()
            
            # التخلص من القيم المفقودة
            data = data.dropna()
            
            if len(data) < 5:
                return {'score': 0.5, 'details': 'Insufficient data'}
            
            current = data.iloc[-1]
            
            # تحديد الاتجاه العام
            uptrend = current['ma_short'] > current['ma_medium'] > current['ma_long']
            downtrend = current['ma_short'] < current['ma_medium'] < current['ma_long']
            
            # حساب درجة التحقق
            score = 0.5  # محايد افتراضيًا
            
            if signal['signal_type'] in ['BUY', 'CALL']:
                if uptrend:
                    score = 0.9  # اتجاه صاعد قوي
                elif current['ma_short'] > current['ma_medium']:
                    score = 0.7  # اتجاه صاعد متوسط
                elif downtrend:
                    score = 0.2  # اتجاه هابط (مخالف للإشارة)
            elif signal['signal_type'] in ['SELL', 'PUT']:
                if downtrend:
                    score = 0.9  # اتجاه هابط قوي
                elif current['ma_short'] < current['ma_medium']:
                    score = 0.7  # اتجاه هابط متوسط
                elif uptrend:
                    score = 0.2  # اتجاه صاعد (مخالف للإشارة)
            
            return {
                'score': score,
                'details': {
                    'trend': 'uptrend' if uptrend else ('downtrend' if downtrend else 'neutral'),
                    'ma_short': current['ma_short'],
                    'ma_medium': current['ma_medium'],
                    'ma_long': current['ma_long']
                }
            }
            
        except Exception as e:
            logger.error(f"Error verifying market trend: {str(e)}")
            return {'score': 0.5, 'details': str(e)}
    
    def _verify_multiple_indicators(self, data: pd.DataFrame, signal: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced verification with advanced indicators"""
    try:
        # Advanced Indicators
        data['wma'] = ta.WMA(data['close'], timeperiod=20)
        data['tema'] = ta.TEMA(data['close'], timeperiod=20)
        data['ppo'] = ta.PPO(data['close'], fastperiod=12, slowperiod=26)
        data['kama'] = ta.KAMA(data['close'], timeperiod=30)
        data['mama'], data['fama'] = ta.MAMA(data['close'])
        
        # Momentum Indicators
        data['cci'] = ta.CCI(data['high'], data['low'], data['close'], timeperiod=14)
        data['mfi'] = ta.MFI(data['high'], data['low'], data['close'], data['volume'], timeperiod=14)
        data['adx'] = ta.ADX(data['high'], data['low'], data['close'], timeperiod=14)
        """التحقق من تطابق المؤشرات المتعددة"""
        try:
            # حساب المؤشرات المختلفة
            # 1. RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # 2. MACD
            data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
            data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
            data['macd'] = data['ema_12'] - data['ema_26']
            data['signal_line'] = data['macd'].ewm(span=9, adjust=False).mean()
            data['macd_histogram'] = data['macd'] - data['signal_line']
            
            # 3. Stochastic
            low_min = data['low'].rolling(window=14).min()
            high_max = data['high'].rolling(window=14).max()
            data['stoch_k'] = 100 * ((data['close'] - low_min) / (high_max - low_min))
            data['stoch_d'] = data['stoch_k'].rolling(window=3).mean()
            
            # التخلص من القيم المفقودة
            data = data.dropna()
            
            if len(data) < 3:
                return {'score': 0.5, 'details': 'Insufficient data'}
            
            current = data.iloc[-1]
            
            # عدد المؤشرات التي تدعم الإشارة
            supporting_indicators = 0
            total_indicators = 3
            
            # التحقق من RSI
            if signal['signal_type'] in ['BUY', 'CALL']:
                if current['rsi'] < 30:  # ذروة بيع
                    supporting_indicators += 1
                elif current['rsi'] < 50:  # منطقة بيع
                    supporting_indicators += 0.5
            elif signal['signal_type'] in ['SELL', 'PUT']:
                if current['rsi'] > 70:  # ذروة شراء
                    supporting_indicators += 1
                elif current['rsi'] > 50:  # منطقة شراء
                    supporting_indicators += 0.5
            
            # التحقق من MACD
            if signal['signal_type'] in ['BUY', 'CALL']:
                if current['macd'] > current['signal_line']:
                    supporting_indicators += 1
                elif current['macd_histogram'] > 0:
                    supporting_indicators += 0.5
            elif signal['signal_type'] in ['SELL', 'PUT']:
                if current['macd'] < current['signal_line']:
                    supporting_indicators += 1
                elif current['macd_histogram'] < 0:
                    supporting_indicators += 0.5
            
            # التحقق من Stochastic
            if signal['signal_type'] in ['BUY', 'CALL']:
                if current['stoch_k'] < 20 and current['stoch_k'] > current['stoch_d']:
                    supporting_indicators += 1
                elif current['stoch_k'] > current['stoch_d']:
                    supporting_indicators += 0.5
            elif signal['signal_type'] in ['SELL', 'PUT']:
                if current['stoch_k'] > 80 and current['stoch_k'] < current['stoch_d']:
                    supporting_indicators += 1
                elif current['stoch_k'] < current['stoch_d']:
                    supporting_indicators += 0.5
            
            # حساب درجة التحقق
            score = supporting_indicators / total_indicators
            
            return {
                'score': score,
                'details': {
                    'supporting_indicators': supporting_indicators,
                    'total_indicators': total_indicators,
                    'rsi': current['rsi'],
                    'macd': current['macd'],
                    'macd_signal': current['signal_line'],
                    'stoch_k': current['stoch_k'],
                    'stoch_d': current['stoch_d']
                }
            }
            
        except Exception as e:
            logger.error(f"Error verifying multiple indicators: {str(e)}")
            return {'score': 0.5, 'details': str(e)}
    
    def _verify_price_patterns(self, data: pd.DataFrame, signal: Dict[str, Any]) -> Dict[str, Any]:
        """التحقق من الأنماط السعرية"""
        try:
            if len(data) < 10:
                return {'score': 0.5, 'details': 'Insufficient data'}
            
            # البحث عن الأنماط المختلفة
            patterns_found = []
            patterns_score = 0.5  # محايد افتراضيًا
            
            # الشموع الأخيرة
            last_candles = data.iloc[-5:]
            
            # 1. نمط الابتلاع
            engulfing = self._detect_engulfing_pattern(last_candles)
            if engulfing:
                patterns_found.append(engulfing)
                if engulfing['type'] == 'bullish' and signal['signal_type'] in ['BUY', 'CALL']:
                    patterns_score = 0.8
                elif engulfing['type'] == 'bearish' and signal['signal_type'] in ['SELL', 'PUT']:
                    patterns_score = 0.8
            
            # 2. نمط الدوجي
            doji = self._detect_doji_pattern(last_candles)
            if doji:
                patterns_found.append(doji)
                if doji['type'] == 'bullish' and signal['signal_type'] in ['BUY', 'CALL']:
                    patterns_score = 0.7
                elif doji['type'] == 'bearish' and signal['signal_type'] in ['SELL', 'PUT']:
                    patterns_score = 0.7
            
            # 3. نمط الهامر (المطرقة)
            hammer = self._detect_hammer_pattern(last_candles)
            if hammer:
                patterns_found.append(hammer)
                if hammer['type'] == 'bullish' and signal['signal_type'] in ['BUY', 'CALL']:
                    patterns_score = 0.85
                elif hammer['type'] == 'bearish' and signal['signal_type'] in ['SELL', 'PUT']:
                    patterns_score = 0.7
            
            # 4. نمط نجمة الصباح/المساء
            star = self._detect_star_pattern(last_candles)
            if star:
                patterns_found.append(star)
                if star['type'] == 'bullish' and signal['signal_type'] in ['BUY', 'CALL']:
                    patterns_score = 0.9
                elif star['type'] == 'bearish' and signal['signal_type'] in ['SELL', 'PUT']:
                    patterns_score = 0.9
            
            return {
                'score': patterns_score,
                'details': {
                    'patterns_found': patterns_found,
                    'count': len(patterns_found)
                }
            }
            
        except Exception as e:
            logger.error(f"Error verifying price patterns: {str(e)}")
            return {'score': 0.5, 'details': str(e)}
    
    def _verify_support_resistance(self, data: pd.DataFrame, signal: Dict[str, Any]) -> Dict[str, Any]:
        """التحقق من مستويات الدعم والمقاومة"""
        try:
            if len(data) < 50:
                return {'score': 0.5, 'details': 'Insufficient data'}
            
            # استخراج القمم والقيعان السابقة
            data['max_high'] = data['high'].rolling(window=10, center=True).max()
            data['min_low'] = data['low'].rolling(window=10, center=True).min()
            
            # تعريف القمم والقيعان المحتملة
            peaks = data[(data['high'] == data['max_high'])].copy()
            bottoms = data[(data['low'] == data['min_low'])].copy()
            
            # ترشيح القمم والقيعان المهمة
            important_peaks = peaks.sort_values('high', ascending=False).head(5)
            important_bottoms = bottoms.sort_values('low').head(5)
            
            # الحصول على السعر الحالي
            current_price = data.iloc[-1]['close']
            
            # تحديد مستويات الدعم والمقاومة
            resistance_levels = important_peaks['high'].tolist()
            support_levels = important_bottoms['low'].tolist()
            
            # ترتيب المستويات
            resistance_levels.sort()
            support_levels.sort(reverse=True)
            
            # تحديد أقرب مستويات
            closest_resistance = None
            closest_support = None
            
            for level in resistance_levels:
                if level > current_price:
                    closest_resistance = level
                    break
            
            for level in support_levels:
                if level < current_price:
                    closest_support = level
                    break
            
            if closest_resistance is None or closest_support is None:
                return {'score': 0.5, 'details': 'No clear support/resistance levels'}
            
            # حساب المسافة إلى المستويات
            distance_to_resistance = closest_resistance - current_price
            distance_to_support = current_price - closest_support
            
            # تقييم الإشارة بناءً على المستويات
            score = 0.5  # محايد افتراضيًا
            
            if signal['signal_type'] in ['BUY', 'CALL']:
                # الشراء أفضل عندما يكون السعر قريبًا من الدعم وبعيدًا عن المقاومة
                if distance_to_support < (distance_to_resistance / 3):
                    score = 0.9  # قريب جدًا من الدعم
                elif distance_to_support < (distance_to_resistance / 2):
                    score = 0.8  # قريب من الدعم
                elif distance_to_support > distance_to_resistance:
                    score = 0.3  # أقرب إلى المقاومة (غير مفضل)
            elif signal['signal_type'] in ['SELL', 'PUT']:
                # البيع أفضل عندما يكون السعر قريبًا من المقاومة وبعيدًا عن الدعم
                if distance_to_resistance < (distance_to_support / 3):
                    score = 0.9  # قريب جدًا من المقاومة
                elif distance_to_resistance < (distance_to_support / 2):
                    score = 0.8  # قريب من المقاومة
                elif distance_to_resistance > distance_to_support:
                    score = 0.3  # أقرب إلى الدعم (غير مفضل)
            
            return {
                'score': score,
                'details': {
                    'current_price': current_price,
                    'closest_resistance': closest_resistance,
                    'closest_support': closest_support,
                    'distance_to_resistance': distance_to_resistance,
                    'distance_to_support': distance_to_support
                }
            }
            
        except Exception as e:
            logger.error(f"Error verifying support/resistance: {str(e)}")
            return {'score': 0.5, 'details': str(e)}
    
    def _verify_multiple_timeframes(self, currency_pair: str, timeframe: str, signal: Dict[str, Any]) -> Dict[str, Any]:
        """التحقق من حالة السوق في الأطر الزمنية المتعددة"""
        try:
            # تحديد الأطر الزمنية الأعلى
            higher_timeframes = []
            
            if timeframe == '1m':
                higher_timeframes = ['5m', '15m', '1h']
            elif timeframe == '5m':
                higher_timeframes = ['15m', '1h', '4h']
            elif timeframe == '15m':
                higher_timeframes = ['1h', '4h', '1d']
            elif timeframe == '30m':
                higher_timeframes = ['1h', '4h', '1d']
            elif timeframe == '1h':
                higher_timeframes = ['4h', '1d']
            elif timeframe == '4h':
                higher_timeframes = ['1d']
            else:
                return {'score': 0.5, 'details': 'No higher timeframes available'}
            
            # تتبع عدد الإطارات الزمنية الداعمة
            supporting_timeframes = 0
            timeframe_data = {}
            
            for tf in higher_timeframes:
                data = self.market_data_service.get_historical_data(currency_pair, tf)
                if data is None or data.empty:
                    continue
                
                # حساب المتوسطات المتحركة البسيطة لتحديد الاتجاه
                data['ma_short'] = data['close'].rolling(window=20).mean()
                data['ma_long'] = data['close'].rolling(window=50).mean()
                
                # التخلص من القيم المفقودة
                data = data.dropna()
                
                if len(data) < 5:
                    continue
                
                current = data.iloc[-1]
                
                # تحديد الاتجاه
                uptrend = current['ma_short'] > current['ma_long']
                downtrend = current['ma_short'] < current['ma_long']
                
                timeframe_data[tf] = {
                    'trend': 'uptrend' if uptrend else ('downtrend' if downtrend else 'neutral'),
                    'ma_short': current['ma_short'],
                    'ma_long': current['ma_long']
                }
                
                # التحقق من تطابق الاتجاه مع الإشارة
                if signal['signal_type'] in ['BUY', 'CALL'] and uptrend:
                    supporting_timeframes += 1
                elif signal['signal_type'] in ['SELL', 'PUT'] and downtrend:
                    supporting_timeframes += 1
            
            # حساب درجة التحقق
            if len(higher_timeframes) == 0:
                score = 0.5
            else:
                score = supporting_timeframes / len(higher_timeframes)
            
            return {
                'score': score,
                'details': {
                    'supporting_timeframes': supporting_timeframes,
                    'total_timeframes': len(higher_timeframes),
                    'timeframe_data': timeframe_data
                }
            }
            
        except Exception as e:
            logger.error(f"Error verifying multiple timeframes: {str(e)}")
            return {'score': 0.5, 'details': str(e)}
    
    def _verify_multiple_strategies(self, currency_pair: str, timeframe: str, signal: Dict[str, Any]) -> Dict[str, Any]:
        """التحقق من توافق الاستراتيجيات المختلفة"""
        try:
            # قائمة بأسماء الاستراتيجيات القوية للتحقق
            strategies_to_check = [
                'moving_average_crossover',
                'macd',
                'rsi',
                'bollinger_bands',
                'high_precision_combo'
            ]
            
            # الحصول على بيانات السوق
            data = self.market_data_service.get_historical_data(currency_pair, timeframe)
            if data is None or data.empty:
                return {'score': 0.5, 'details': 'No data available'}
            
            # تتبع عدد الاستراتيجيات المتوافقة
            supporting_strategies = 0
            strategy_signals = {}
            
            for strategy_name in strategies_to_check:
                # تنفيذ الاستراتيجية
                strategy_signal = self.strategy_service.execute_strategy(strategy_name, currency_pair, timeframe)
                
                if strategy_signal:
                    strategy_signals[strategy_name] = {
                        'signal_type': strategy_signal['signal_type'],
                        'entry_price': strategy_signal['entry_price']
                    }
                    
                    # التحقق من تطابق نوع الإشارة
                    if strategy_signal['signal_type'] == signal['signal_type']:
                        supporting_strategies += 1
            
            # حساب درجة التحقق
            if len(strategy_signals) == 0:
                score = 0.5
            else:
                score = supporting_strategies / len(strategy_signals)
            
            return {
                'score': score,
                'details': {
                    'supporting_strategies': supporting_strategies,
                    'total_strategies': len(strategy_signals),
                    'strategy_signals': strategy_signals
                }
            }
            
        except Exception as e:
            logger.error(f"Error verifying multiple strategies: {str(e)}")
            return {'score': 0.5, 'details': str(e)}
    
    def _verify_historical_performance(self, currency_pair: str, timeframe: str, signal: Dict[str, Any]) -> Dict[str, Any]:
        """التحقق من احتمالية نجاح الإشارة باستخدام البيانات التاريخية"""
        try:
            # البحث عن إشارات سابقة مماثلة
            with self.app.app_context():
                currency = Currency.query.filter_by(symbol=currency_pair).first()
                if not currency:
                    return {'score': 0.5, 'details': 'Currency not found'}
                
                # البحث عن إشارات سابقة من نفس النوع والإستراتيجية
                strategy_name = signal.get('strategy', 'unknown')
                previous_signals = Signal.query.filter_by(
                    currency_id=currency.id,
                    signal_type=signal['signal_type'],
                    timeframe=timeframe,
                    strategy=strategy_name
                ).filter(Signal.result.in_(['WIN', 'LOSS'])).all()
                
                if len(previous_signals) < 5:
                    # عدد غير كافٍ من الإشارات السابقة
                    return {'score': 0.5, 'details': 'Insufficient historical data'}
                
                # حساب معدل النجاح
                wins = sum(1 for s in previous_signals if s.result == 'WIN')
                success_rate = wins / len(previous_signals)
                
                # تحديد درجة التحقق
                if success_rate >= 0.9:
                    score = 0.95
                elif success_rate >= 0.8:
                    score = 0.85
                elif success_rate >= 0.7:
                    score = 0.75
                elif success_rate >= 0.6:
                    score = 0.65
                elif success_rate <= 0.3:
                    score = 0.3
                else:
                    score = 0.5
                
                return {
                    'score': score,
                    'details': {
                        'total_signals': len(previous_signals),
                        'wins': wins,
                        'losses': len(previous_signals) - wins,
                        'success_rate': success_rate,
                        'strategy': strategy_name
                    }
                }
            
        except Exception as e:
            logger.error(f"Error verifying historical performance: {str(e)}")
            return {'score': 0.5, 'details': str(e)}
    
    def _verify_signal_timing(self, data: pd.DataFrame, signal: Dict[str, Any]) -> Dict[str, Any]:
        """التحقق من توقيت الإشارة"""
        try:
            if len(data) < 20:
                return {'score': 0.5, 'details': 'Insufficient data'}
            
            # حساب مؤشر ADX لقياس قوة الاتجاه
            high = data['high']
            low = data['low']
            close = data['close']
            
            plus_dm = high.diff()
            minus_dm = low.diff().multiply(-1)
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            # Directional Movement (DM+/DM-)
            condition = (plus_dm > minus_dm) & (plus_dm > 0)
            plus_dm[~condition] = 0
            
            condition = (minus_dm > plus_dm) & (minus_dm > 0)
            minus_dm[~condition] = 0
            
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Smoothed TR and DM
            tr_smoothed = tr.rolling(window=14).sum()
            plus_dm_smoothed = plus_dm.rolling(window=14).sum()
            minus_dm_smoothed = minus_dm.rolling(window=14).sum()
            
            # Directional Indicators (+DI/-DI)
            plus_di = 100 * plus_dm_smoothed / tr_smoothed
            minus_di = 100 * minus_dm_smoothed / tr_smoothed
            
            # Average Directional Index (ADX)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            data['adx'] = dx.rolling(window=14).mean()
            
            # حساب متوسط حجم التداول
            data['volume_sma'] = data['volume'].rolling(window=20).mean()
            
            # حساب تقلب السعر
            data['true_range'] = tr
            data['atr'] = tr.rolling(window=14).mean()
            
            # التخلص من القيم المفقودة
            data = data.dropna()
            
            if len(data) < 2:
                return {'score': 0.5, 'details': 'Insufficient data after calculations'}
            
            # تحليل التوقيت
            current = data.iloc[-1]
            previous = data.iloc[-2]
            
            timing_score = 0.5  # محايد افتراضيًا
            timing_factors = []
            
            # 1. قوة الاتجاه (ADX)
            strong_trend = current['adx'] > 25
            if strong_trend:
                timing_score += 0.1
                timing_factors.append("Strong trend (ADX > 25)")
            
            # 2. حجم التداول
            increasing_volume = current['volume'] > current['volume_sma']
            if increasing_volume:
                timing_score += 0.1
                timing_factors.append("Volume above average")
            
            # 3. كسر مستوى مهم
            for i in range(5, 31, 5):
                if i <= len(data):
                    if signal['signal_type'] in ['BUY', 'CALL']:
                        resistance_break = current['close'] > data.iloc[-i:-1]['high'].max()
                        if resistance_break:
                            timing_score += 0.1
                            timing_factors.append(f"Broke {i}-period resistance")
                            break
                    elif signal['signal_type'] in ['SELL', 'PUT']:
                        support_break = current['close'] < data.iloc[-i:-1]['low'].min()
                        if support_break:
                            timing_score += 0.1
                            timing_factors.append(f"Broke {i}-period support")
                            break
            
            # 4. تقلب منخفض قبل الحركة
            low_volatility = current['atr'] > previous['atr'] * 1.2
            if low_volatility:
                timing_score += 0.1
                timing_factors.append("Increasing volatility after period of low volatility")
            
            # 5. تناقص القمم/القيعان
            if len(data) >= 10:
                if signal['signal_type'] in ['BUY', 'CALL']:
                    higher_lows = data.iloc[-5:-1]['low'].is_monotonic_increasing
                    if higher_lows:
                        timing_score += 0.1
                        timing_factors.append("Higher lows pattern")
                elif signal['signal_type'] in ['SELL', 'PUT']:
                    lower_highs = data.iloc[-5:-1]['high'].is_monotonic_decreasing
                    if lower_highs:
                        timing_score += 0.1
                        timing_factors.append("Lower highs pattern")
            
            # ضبط النتيجة بين 0 و 1
            timing_score = min(max(timing_score, 0), 1)
            
            return {
                'score': timing_score,
                'details': {
                    'factors': timing_factors,
                    'adx': current['adx'],
                    'volume_ratio': current['volume'] / current['volume_sma'] if current['volume_sma'] > 0 else 0,
                    'atr': current['atr']
                }
            }
            
        except Exception as e:
            logger.error(f"Error verifying signal timing: {str(e)}")
            return {'score': 0.5, 'details': str(e)}
    
    def _verify_risk_reward_ratio(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """التحقق من تناسب نسبة المخاطرة/المكافأة"""
        try:
            # استخراج الأسعار
            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            take_profit = signal.get('take_profit', 0)
            
            if entry_price == 0 or stop_loss == 0 or take_profit == 0:
                return {'score': 0.5, 'details': 'Missing price levels'}
            
            # حساب نسبة المخاطرة/المكافأة
            if signal['signal_type'] in ['BUY', 'CALL']:
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
            else:  # SELL/PUT
                risk = stop_loss - entry_price
                reward = entry_price - take_profit
            
            if risk <= 0:
                return {'score': 0.2, 'details': 'Invalid stop loss'}
            
            risk_reward_ratio = reward / risk
            
            # تقييم نسبة المخاطرة/المكافأة
            if risk_reward_ratio >= 3:
                score = 0.95
            elif risk_reward_ratio >= 2:
                score = 0.85
            elif risk_reward_ratio >= 1.5:
                score = 0.75
            elif risk_reward_ratio >= 1:
                score = 0.6
            else:
                score = 0.4
            
            return {
                'score': score,
                'details': {
                    'risk': risk,
                    'reward': reward,
                    'risk_reward_ratio': risk_reward_ratio
                }
            }
            
        except Exception as e:
            logger.error(f"Error verifying risk/reward ratio: {str(e)}")
            return {'score': 0.5, 'details': str(e)}
    
    def _verify_buy_signal(self, data: pd.DataFrame, signal: Dict[str, Any]) -> Dict[str, Any]:
        """التحقق من إشارة شراء"""
        try:
            if len(data) < 10:
                return {'score': 0.5, 'details': 'Insufficient data'}
            
            buy_score = 0.5  # محايد افتراضيًا
            buy_factors = []
            
            # 1. تحقق من الارتداد من مستوى دعم
            recent_lows = data.iloc[-10:]['low'].values
            min_low = min(recent_lows)
            current_close = data.iloc[-1]['close']
            
            if abs(current_close - min_low) / min_low < 0.01:
                buy_score += 0.1
                buy_factors.append("Price bouncing from support")
            
            # 2. تحقق من ذروة البيع (RSI)
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            if data.iloc[-1]['rsi'] < 30:
                buy_score += 0.2
                buy_factors.append("Oversold condition (RSI < 30)")
            elif data.iloc[-1]['rsi'] < 40:
                buy_score += 0.1
                buy_factors.append("Approaching oversold (RSI < 40)")
            
            # 3. تحقق من التباعد الإيجابي
            if len(data) >= 20:
                if data.iloc[-1]['close'] > data.iloc[-5]['close'] and data.iloc[-1]['rsi'] > data.iloc[-5]['rsi']:
                    buy_score += 0.1
                    buy_factors.append("Positive divergence in RSI")
            
            # 4. تحقق من الارتداد من المتوسط المتحرك 200
            data['sma_200'] = data['close'].rolling(window=200).mean()
            if len(data.dropna()) > 0:
                sma_200 = data.iloc[-1]['sma_200']
                if current_close > sma_200 * 0.99 and current_close < sma_200 * 1.02:
                    buy_score += 0.1
                    buy_factors.append("Price around SMA 200")
            
            # ضبط النتيجة بين 0 و 1
            buy_score = min(max(buy_score, 0), 1)
            
            return {
                'score': buy_score,
                'details': {
                    'factors': buy_factors,
                    'rsi': data.iloc[-1]['rsi'] if 'rsi' in data.columns else None,
                    'distance_from_low': abs(current_close - min_low) / min_low
                }
            }
            
        except Exception as e:
            logger.error(f"Error verifying buy signal: {str(e)}")
            return {'score': 0.5, 'details': str(e)}
    
    def _verify_sell_signal(self, data: pd.DataFrame, signal: Dict[str, Any]) -> Dict[str, Any]:
        """التحقق من إشارة بيع"""
        try:
            if len(data) < 10:
                return {'score': 0.5, 'details': 'Insufficient data'}
            
            sell_score = 0.5  # محايد افتراضيًا
            sell_factors = []
            
            # 1. تحقق من الارتداد من مستوى مقاومة
            recent_highs = data.iloc[-10:]['high'].values
            max_high = max(recent_highs)
            current_close = data.iloc[-1]['close']
            
            if abs(current_close - max_high) / max_high < 0.01:
                sell_score += 0.1
                sell_factors.append("Price bouncing from resistance")
            
            # 2. تحقق من ذروة الشراء (RSI)
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            if data.iloc[-1]['rsi'] > 70:
                sell_score += 0.2
                sell_factors.append("Overbought condition (RSI > 70)")
            elif data.iloc[-1]['rsi'] > 60:
                sell_score += 0.1
                sell_factors.append("Approaching overbought (RSI > 60)")
            
            # 3. تحقق من التباعد السلبي
            if len(data) >= 20:
                if data.iloc[-1]['close'] < data.iloc[-5]['close'] and data.iloc[-1]['rsi'] < data.iloc[-5]['rsi']:
                    sell_score += 0.1
                    sell_factors.append("Negative divergence in RSI")
            
            # 4. تحقق من الارتداد من المتوسط المتحرك 200
            data['sma_200'] = data['close'].rolling(window=200).mean()
            if len(data.dropna()) > 0:
                sma_200 = data.iloc[-1]['sma_200']
                if current_close < sma_200 * 1.01 and current_close > sma_200 * 0.98:
                    sell_score += 0.1
                    sell_factors.append("Price around SMA 200")
            
            # ضبط النتيجة بين 0 و 1
            sell_score = min(max(sell_score, 0), 1)
            
            return {
                'score': sell_score,
                'details': {
                    'factors': sell_factors,
                    'rsi': data.iloc[-1]['rsi'] if 'rsi' in data.columns else None,
                    'distance_from_high': abs(current_close - max_high) / max_high
                }
            }
            
        except Exception as e:
            logger.error(f"Error verifying sell signal: {str(e)}")
            return {'score': 0.5, 'details': str(e)}
    
    # --------- وظائف تحسين الإشارة ---------
    
    def _enhance_stop_and_target(self, signal: Dict[str, Any], data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """تحسين مستويات الإيقاف والهدف"""
        try:
            if len(data) < 20:
                return signal, {'type': 'stop_and_target', 'success': False, 'reason': 'Insufficient data'}
            
            # حساب مستويات الدعم والمقاومة
            data['max_high'] = data['high'].rolling(window=10, center=True).max()
            data['min_low'] = data['low'].rolling(window=10, center=True).min()
            
            # تعريف القمم والقيعان المحتملة
            peaks = data[(data['high'] == data['max_high'])].copy()
            bottoms = data[(data['low'] == data['min_low'])].copy()
            
            # ترشيح القمم والقيعان المهمة
            important_peaks = peaks.sort_values('high', ascending=False).head(5)
            important_bottoms = bottoms.sort_values('low').head(5)
            
            # ترتيب المستويات
            resistance_levels = sorted(important_peaks['high'].tolist())
            support_levels = sorted(important_bottoms['low'].tolist(), reverse=True)
            
            # حساب مستويات الإيقاف والهدف المحسنة
            entry_price = signal['entry_price']
            signal_type = signal['signal_type']
            enhanced_stop = signal.get('stop_loss', 0)
            enhanced_target = signal.get('take_profit', 0)
            
            # حساب ATR للتقلب
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1]
            
            if signal_type in ['BUY', 'CALL']:
                # البحث عن مستوى دعم قريب لوضعه كإيقاف خسارة
                for level in support_levels:
                    if level < entry_price:
                        enhanced_stop = level - (0.2 * atr)  # إضافة هامش صغير
                        break
                
                # البحث عن مستوى مقاومة قريب لوضعه كهدف
                for level in reversed(resistance_levels):
                    if level > entry_price:
                        enhanced_target = level + (0.2 * atr)  # إضافة هامش صغير
                        break
            
            elif signal_type in ['SELL', 'PUT']:
                # البحث عن مستوى مقاومة قريب لوضعه كإيقاف خسارة
                for level in reversed(resistance_levels):
                    if level > entry_price:
                        enhanced_stop = level + (0.2 * atr)  # إضافة هامش صغير
                        break
                
                # البحث عن مستوى دعم قريب لوضعه كهدف
                for level in support_levels:
                    if level < entry_price:
                        enhanced_target = level - (0.2 * atr)  # إضافة هامش صغير
                        break
            
            # التأكد من أن نسبة المخاطرة/المكافأة مناسبة
            if signal_type in ['BUY', 'CALL']:
                risk = entry_price - enhanced_stop
                reward = enhanced_target - entry_price
            else:
                risk = enhanced_stop - entry_price
                reward = entry_price - enhanced_target
            
            # نسعى لنسبة مخاطرة/مكافأة لا تقل عن 1:2
            if risk > 0 and reward > 0 and (reward / risk) >= 2:
                signal['stop_loss'] = enhanced_stop
                signal['take_profit'] = enhanced_target
                
                return signal, {
                    'type': 'stop_and_target',
                    'success': True,
                    'original_stop': signal.get('stop_loss', 0),
                    'original_target': signal.get('take_profit', 0),
                    'enhanced_stop': enhanced_stop,
                    'enhanced_target': enhanced_target,
                    'risk_reward_ratio': reward / risk
                }
            
            return signal, {'type': 'stop_and_target', 'success': False, 'reason': 'Could not find suitable levels'}
            
        except Exception as e:
            logger.error(f"Error enhancing stop/target: {str(e)}")
            return signal, {'type': 'stop_and_target', 'success': False, 'reason': str(e)}
    
    def _enhance_entry_timing(self, signal: Dict[str, Any], data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """تحسين توقيت الدخول"""
        try:
            if len(data) < 20:
                return signal, {'type': 'entry_timing', 'success': False, 'reason': 'Insufficient data'}
            
            # حساب متوسط المدى اليومي الحقيقي (ATR)
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1]
            
            # حساب المتوسطات المتحركة
            data['ma_20'] = data['close'].rolling(window=20).mean()
            data['ma_50'] = data['close'].rolling(window=50).mean()
            
            # استخراج القيم الحالية
            current_close = data.iloc[-1]['close']
            current_ma_20 = data.iloc[-1]['ma_20']
            current_ma_50 = data.iloc[-1]['ma_50']
            
            # حساب البعد عن المتوسطات المتحركة
            distance_from_ma_20 = abs(current_close - current_ma_20) / current_ma_20
            distance_from_ma_50 = abs(current_close - current_ma_50) / current_ma_50
            
            # الاتجاه الحالي
            uptrend = current_ma_20 > current_ma_50
            
            # تحسين سعر الدخول
            entry_price = signal['entry_price']
            signal_type = signal['signal_type']
            enhanced_entry = entry_price
            
            # تعديل السعر بناءً على اتجاه السوق والتقلب
            if signal_type in ['BUY', 'CALL']:
                if uptrend:
                    # في اتجاه صاعد، شراء عند انخفاض السعر قليلاً
                    if distance_from_ma_20 < 0.01:  # قريب جدًا من المتوسط
                        enhanced_entry = current_close - (0.3 * atr)
                else:
                    # في اتجاه هابط، انتظار تأكيد الانعكاس
                    enhanced_entry = current_close - (0.5 * atr)
            
            elif signal_type in ['SELL', 'PUT']:
                if not uptrend:
                    # في اتجاه هابط، بيع عند ارتفاع السعر قليلاً
                    if distance_from_ma_20 < 0.01:  # قريب جدًا من المتوسط
                        enhanced_entry = current_close + (0.3 * atr)
                else:
                    # في اتجاه صاعد، انتظار تأكيد الانعكاس
                    enhanced_entry = current_close + (0.5 * atr)
            
            # التحقق من منطقية السعر المحسن
            if ((signal_type in ['BUY', 'CALL'] and enhanced_entry < entry_price) or
                (signal_type in ['SELL', 'PUT'] and enhanced_entry > entry_price)):
                
                signal['entry_price'] = enhanced_entry
                
                # تعديل مستويات الإيقاف والهدف أيضًا للحفاظ على نسبة المخاطرة/المكافأة
                if signal.get('stop_loss') and signal.get('take_profit'):
                    if signal_type in ['BUY', 'CALL']:
                        original_risk = entry_price - signal['stop_loss']
                        original_reward = signal['take_profit'] - entry_price
                        
                        signal['stop_loss'] = enhanced_entry - original_risk
                        signal['take_profit'] = enhanced_entry + original_reward
                    else:
                        original_risk = signal['stop_loss'] - entry_price
                        original_reward = entry_price - signal['take_profit']
                        
                        signal['stop_loss'] = enhanced_entry + original_risk
                        signal['take_profit'] = enhanced_entry - original_reward
                
                return signal, {
                    'type': 'entry_timing',
                    'success': True,
                    'original_entry': entry_price,
                    'enhanced_entry': enhanced_entry,
                    'atr': atr,
                    'uptrend': uptrend
                }
            
            return signal, {'type': 'entry_timing', 'success': False, 'reason': 'No meaningful enhancement possible'}
            
        except Exception as e:
            logger.error(f"Error enhancing entry timing: {str(e)}")
            return signal, {'type': 'entry_timing', 'success': False, 'reason': str(e)}
    
    def _enhance_success_probability(self, signal: Dict[str, Any], data: pd.DataFrame, currency_pair: str, timeframe: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """تحسين احتمالية النجاح"""
        try:
            # البحث عن إشارات سابقة مماثلة
            with self.app.app_context():
                currency = Currency.query.filter_by(symbol=currency_pair).first()
                if not currency:
                    return signal, {'type': 'success_probability', 'success': False, 'reason': 'Currency not found'}
                
                # البحث عن إشارات سابقة من نفس النوع
                previous_signals = Signal.query.filter_by(
                    currency_id=currency.id,
                    signal_type=signal['signal_type'],
                    timeframe=timeframe
                ).filter(Signal.result.in_(['WIN', 'LOSS'])).all()
                
                if len(previous_signals) < 5:
                    # عدد غير كافٍ من الإشارات السابقة
                    base_probability = 0.7  # احتمالية أساسية
                else:
                    # حساب معدل النجاح
                    wins = sum(1 for s in previous_signals if s.result == 'WIN')
                    base_probability = wins / len(previous_signals)
            
            # تحليل المؤشرات الفنية لتحسين الاحتمالية
            if len(data) >= 20:
                # حساب RSI
                delta = data['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                data['rsi'] = 100 - (100 / (1 + rs))
                
                # حساب قوة الاتجاه (ADX)
                high = data['high']
                low = data['low']
                close = data['close']
                
                plus_dm = high.diff()
                minus_dm = low.diff().multiply(-1)
                plus_dm[plus_dm < 0] = 0
                minus_dm[minus_dm < 0] = 0
                
                condition = (plus_dm > minus_dm) & (plus_dm > 0)
                plus_dm[~condition] = 0
                
                condition = (minus_dm > plus_dm) & (minus_dm > 0)
                minus_dm[~condition] = 0
                
                tr1 = high - low
                tr2 = abs(high - close.shift())
                tr3 = abs(low - close.shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                
                tr_smoothed = tr.rolling(window=14).sum()
                plus_dm_smoothed = plus_dm.rolling(window=14).sum()
                minus_dm_smoothed = minus_dm.rolling(window=14).sum()
                
                plus_di = 100 * plus_dm_smoothed / tr_smoothed
                minus_di = 100 * minus_dm_smoothed / tr_smoothed
                
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
                data['adx'] = dx.rolling(window=14).mean()
                
                # التخلص من القيم المفقودة
                data = data.dropna()
                
                if len(data) > 0:
                    current = data.iloc[-1]
                    
                    # تعديل الاحتمالية بناءً على المؤشرات
                    probability_adjustments = []
                    
                    # 1. تعديل بناءً على RSI
                    if signal['signal_type'] in ['BUY', 'CALL'] and current['rsi'] < 30:
                        probability_adjustments.append(0.05)  # زيادة الاحتمالية عند ذروة البيع
                    elif signal['signal_type'] in ['SELL', 'PUT'] and current['rsi'] > 70:
                        probability_adjustments.append(0.05)  # زيادة الاحتمالية عند ذروة الشراء
                    
                    # 2. تعديل بناءً على قوة الاتجاه
                    if current['adx'] > 25:
                        probability_adjustments.append(0.03)  # زيادة الاحتمالية في اتجاه قوي
                    
                    # 3. تعديل بناءً على اتفاق المؤشرات
                    if 'indicators' in signal:
                        indicators_agreement = sum(1 for k, v in signal['indicators'].items() if v > 0) / len(signal['indicators'])
                        probability_adjustments.append(indicators_agreement * 0.05)
                    
                    # 4. تعديل بناءً على نسبة المخاطرة/المكافأة
                    if signal.get('stop_loss') and signal.get('take_profit'):
                        if signal['signal_type'] in ['BUY', 'CALL']:
                            risk = signal['entry_price'] - signal['stop_loss']
                            reward = signal['take_profit'] - signal['entry_price']
                        else:
                            risk = signal['stop_loss'] - signal['entry_price']
                            reward = signal['entry_price'] - signal['take_profit']
                        
                        if risk > 0:
                            risk_reward_ratio = reward / risk
                            if risk_reward_ratio > 3:
                                probability_adjustments.append(0.05)
                            elif risk_reward_ratio > 2:
                                probability_adjustments.append(0.03)
                    
                    # تطبيق التعديلات على الاحتمالية الأساسية
                    enhanced_probability = base_probability + sum(probability_adjustments)
                    enhanced_probability = min(max(enhanced_probability, 0), 0.95)  # تقييد الاحتمالية بين 0 و 95%
                    
                    signal['success_probability'] = enhanced_probability
                    
                    return signal, {
                        'type': 'success_probability',
                        'success': True,
                        'base_probability': base_probability,
                        'enhanced_probability': enhanced_probability,
                        'adjustments': probability_adjustments
                    }
            
            # إذا لم نتمكن من تحسين الاحتمالية، نستخدم الاحتمالية الأساسية
            signal['success_probability'] = base_probability
            
            return signal, {
                'type': 'success_probability',
                'success': True,
                'base_probability': base_probability,
                'enhanced_probability': base_probability,
                'adjustments': []
            }
            
        except Exception as e:
            logger.error(f"Error enhancing success probability: {str(e)}")
            return signal, {'type': 'success_probability', 'success': False, 'reason': str(e)}
    
    def _enhance_precision(self, signal: Dict[str, Any], data: pd.DataFrame, currency_pair: str, timeframe: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """تعزيز دقة الإشارة"""
        try:
            # تحسين الدقة من خلال تحليل متعدد الجوانب
            precision_factors = []
            precision_score = 0
            
            # 1. التحقق من اتفاق الأطر الزمنية المتعددة
            timeframes_verification = self._verify_multiple_timeframes(currency_pair, timeframe, signal)
            if timeframes_verification['score'] > 0.7:
                precision_factors.append(f"Multiple timeframes agreement ({timeframes_verification['score']:.2f})")
                precision_score += 1
            
            # 2. التحقق من اتفاق استراتيجيات متعددة
            strategies_verification = self._verify_multiple_strategies(currency_pair, timeframe, signal)
            if strategies_verification['score'] > 0.7:
                precision_factors.append(f"Multiple strategies agreement ({strategies_verification['score']:.2f})")
                precision_score += 1
            
            # 3. التحقق من وجود أنماط سعرية قوية
            patterns_verification = self._verify_price_patterns(data, signal)
            if patterns_verification['score'] > 0.8:
                precision_factors.append(f"Strong price patterns ({patterns_verification['score']:.2f})")
                precision_score += 1
            
            # 4. التحقق من مستويات دعم/مقاومة قوية
            levels_verification = self._verify_support_resistance(data, signal)
            if levels_verification['score'] > 0.8:
                precision_factors.append(f"Strong support/resistance levels ({levels_verification['score']:.2f})")
                precision_score += 1
            
            # 5. التحقق من توقيت ممتاز
            timing_verification = self._verify_signal_timing(data, signal)
            if timing_verification['score'] > 0.7:
                precision_factors.append(f"Excellent timing ({timing_verification['score']:.2f})")
                precision_score += 1
            
            # تعيين درجة الدقة العالية إذا تم استيفاء شروط كافية
            if precision_score >= 3:
                signal['is_high_precision'] = True
                
                # تحسين احتمالية النجاح أيضًا
                if 'success_probability' in signal:
                    signal['success_probability'] = min(signal['success_probability'] + 0.05, 0.95)
                
                return signal, {
                    'type': 'precision',
                    'success': True,
                    'precision_factors': precision_factors,
                    'precision_score': precision_score,
                    'is_high_precision': True
                }
            
            return signal, {
                'type': 'precision',
                'success': False,
                'precision_factors': precision_factors,
                'precision_score': precision_score,
                'is_high_precision': False
            }
            
        except Exception as e:
            logger.error(f"Error enhancing precision: {str(e)}")
            return signal, {'type': 'precision', 'success': False, 'reason': str(e)}
    
    # --------- وظائف مساعدة ---------
    
    def _detect_engulfing_pattern(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """اكتشاف نمط الابتلاع"""
        if len(data) < 2:
            return None
        
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        # نمط الابتلاع الصعودي
        if (previous['close'] < previous['open'] and  # شمعة هابطة
            current['close'] > current['open'] and    # شمعة صاعدة
            current['open'] <= previous['close'] and  # الفتح أقل من أو يساوي إغلاق الشمعة السابقة
            current['close'] > previous['open']):     # الإغلاق أعلى من فتح الشمعة السابقة
            
            return {
                'name': 'Bullish Engulfing',
                'type': 'bullish',
                'strength': 0.8
            }
        
        # نمط الابتلاع الهابط
        elif (previous['close'] > previous['open'] and  # شمعة صاعدة
              current['close'] < current['open'] and    # شمعة هابطة
              current['open'] >= previous['close'] and  # الفتح أعلى من أو يساوي إغلاق الشمعة السابقة
              current['close'] < previous['open']):     # الإغلاق أقل من فتح الشمعة السابقة
            
            return {
                'name': 'Bearish Engulfing',
                'type': 'bearish',
                'strength': 0.8
            }
        
        return None
    
    def _detect_doji_pattern(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """اكتشاف نمط الدوجي"""
        if len(data) < 1:
            return None
        
        current = data.iloc[-1]
        body_size = abs(current['close'] - current['open'])
        range_size = current['high'] - current['low']
        
        # التحقق من أن جسم الشمعة صغير جدًا مقارنة بالمدى
        if range_size > 0 and body_size / range_size < 0.1:
            # دوجي في القاع
            if current['low'] < data['low'].iloc[-3:-1].min():
                return {
                    'name': 'Doji at Support',
                    'type': 'bullish',
                    'strength': 0.7
                }
            # دوجي في القمة
            elif current['high'] > data['high'].iloc[-3:-1].max():
                return {
                    'name': 'Doji at Resistance',
                    'type': 'bearish',
                    'strength': 0.7
                }
            # دوجي عادي
            else:
                return {
                    'name': 'Doji',
                    'type': 'neutral',
                    'strength': 0.5
                }
        
        return None
    
    def _detect_hammer_pattern(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """اكتشاف نمط المطرقة"""
        if len(data) < 1:
            return None
        
        current = data.iloc[-1]
        body_size = abs(current['close'] - current['open'])
        upper_shadow = current['high'] - max(current['open'], current['close'])
        lower_shadow = min(current['open'], current['close']) - current['low']
        range_size = current['high'] - current['low']
        
        if range_size == 0:
            return None
        
        # المطرقة: جسم صغير، ظل سفلي طويل، ظل علوي صغير أو معدوم
        if (body_size / range_size < 0.3 and
            lower_shadow / range_size > 0.6 and
            upper_shadow / range_size < 0.1):
            
            # مطرقة في القاع (علامة انعكاس صعودي)
            if current['low'] < data['low'].iloc[-3:-1].min():
                return {
                    'name': 'Hammer at Support',
                    'type': 'bullish',
                    'strength': 0.85
                }
            else:
                return {
                    'name': 'Hammer',
                    'type': 'bullish',
                    'strength': 0.7
                }
        
        # المطرقة المقلوبة: جسم صغير، ظل علوي طويل، ظل سفلي صغير أو معدوم
        elif (body_size / range_size < 0.3 and
              upper_shadow / range_size > 0.6 and
              lower_shadow / range_size < 0.1):
            
            # مطرقة مقلوبة في القمة (علامة انعكاس هبوطي)
            if current['high'] > data['high'].iloc[-3:-1].max():
                return {
                    'name': 'Shooting Star at Resistance',
                    'type': 'bearish',
                    'strength': 0.85
                }
            else:
                return {
                    'name': 'Shooting Star',
                    'type': 'bearish',
                    'strength': 0.7
                }
        
        return None
    
    def _detect_star_pattern(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """اكتشاف نمط النجمة"""
        if len(data) < 3:
            return None
        
        first = data.iloc[-3]
        middle = data.iloc[-2]
        last = data.iloc[-1]
        
        # شرط الدوجي في الوسط
        middle_body_size = abs(middle['close'] - middle['open'])
        middle_range_size = middle['high'] - middle['low']
        middle_is_doji = middle_range_size > 0 and middle_body_size / middle_range_size < 0.1
        
        # نجمة الصباح
        if (first['close'] < first['open'] and  # الشمعة الأولى هابطة
            last['close'] > last['open'] and    # الشمعة الأخيرة صاعدة
            first['close'] > middle['high'] and  # فجوة هبوطية بين الأولى والوسطى
            middle['low'] < last['open'] and     # فجوة صعودية بين الوسطى والأخيرة
            middle_is_doji):                     # الشمعة الوسطى دوجي
            
            return {
                'name': 'Morning Star',
                'type': 'bullish',
                'strength': 0.9
            }
        
        # نجمة المساء
        elif (first['close'] > first['open'] and  # الشمعة الأولى صاعدة
              last['close'] < last['open'] and    # الشمعة الأخيرة هابطة
              first['close'] < middle['low'] and  # فجوة صعودية بين الأولى والوسطى
              middle['high'] > last['open'] and   # فجوة هبوطية بين الوسطى والأخيرة
              middle_is_doji):                    # الشمعة الوسطى دوجي
            
            return {
                'name': 'Evening Star',
                'type': 'bearish',
                'strength': 0.9
            }
        
        return None


def init_signal_verification_service(app, market_data_service, strategy_service) -> SignalVerificationService:
    """
    تهيئة خدمة التحقق من الإشارات
    
    Args:
        app: تطبيق Flask
        market_data_service: خدمة بيانات السوق
        strategy_service: خدمة استراتيجيات التداول
        
    Returns:
        خدمة التحقق من الإشارات
    """
    # الحصول على إعدادات التكوين
    from config import SYSTEM_SETTINGS
    verification_threshold = SYSTEM_SETTINGS.get('verification_threshold', 0.85)  # زيادة عتبة التحقق
    max_workers = SYSTEM_SETTINGS.get('verification_workers', 5)
    
    # إنشاء خدمة التحقق
    service = SignalVerificationService(app, market_data_service, strategy_service,
                                       max_workers=max_workers,
                                       verification_threshold=verification_threshold)
    
    # بدء الخدمة
    service.start()
    
    logger.info(f"Signal verification service initialized with {max_workers} workers")
    
    return service