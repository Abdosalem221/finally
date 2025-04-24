"""
خدمة الإشارات المتقدمة
"""

from typing import Dict, List, Optional
from datetime import datetime
from app.services.data_service import DataService
from app.services.analysis_service import AnalysisService
from app.utils.validators import validate_symbol, validate_timeframe
from app.models.signal_models import Signal, SignalVerification

class SignalService:
    def __init__(self):
        self.data_service = DataService()
        self.analysis_service = AnalysisService()
    
    def generate_signal(self, symbol: str, timeframe: str) -> Dict:
        """إنشاء إشارة جديدة"""
        try:
            # التحقق من صحة المدخلات
            if not validate_symbol(symbol):
                return {'status': 'error', 'message': 'Invalid symbol'}
            
            if not validate_timeframe(timeframe):
                return {'status': 'error', 'message': 'Invalid timeframe'}
            
            # تحليل السوق
            market_analysis = self.analysis_service.analyze_market(symbol, timeframe)
            if market_analysis['status'] == 'error':
                return market_analysis
            
            # تحديد نوع الإشارة
            signal_type = self._determine_signal_type(market_analysis['analysis'])
            
            # حساب مستويات الدخول والخروج
            levels = self._calculate_levels(symbol, timeframe, signal_type)
            
            # إنشاء الإشارة
            signal = Signal(
                symbol=symbol,
                timeframe=timeframe,
                signal_type=signal_type,
                entry_price=levels['entry'],
                stop_loss=levels['stop_loss'],
                take_profit=levels['take_profit'],
                risk_reward_ratio=levels['risk_reward_ratio'],
                confidence_score=self._calculate_confidence_score(market_analysis['analysis']),
                analysis_data=market_analysis['analysis'],
                created_at=datetime.utcnow()
            )
            
            # حفظ الإشارة في قاعدة البيانات
            signal.save()
            
            return {
                'status': 'success',
                'signal': signal.to_dict()
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def verify_signal(self, signal_id: int, verification_data: Dict) -> Dict:
        """التحقق من صحة الإشارة"""
        try:
            # البحث عن الإشارة
            signal = Signal.get_by_id(signal_id)
            if not signal:
                return {'status': 'error', 'message': 'Signal not found'}
            
            # إنشاء التحقق
            verification = SignalVerification(
                signal_id=signal_id,
                verified_price=verification_data.get('price'),
                verified_time=datetime.utcnow(),
                verification_type=verification_data.get('type', 'manual'),
                notes=verification_data.get('notes', ''),
                created_at=datetime.utcnow()
            )
            
            # حفظ التحقق
            verification.save()
            
            # تحديث الإشارة
            signal.verification_status = 'verified'
            signal.save()
            
            return {
                'status': 'success',
                'verification': verification.to_dict()
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_signals(self, filters: Optional[Dict] = None) -> Dict:
        """الحصول على الإشارات"""
        try:
            # تطبيق الفلاتر
            query = Signal.query
            
            if filters:
                if filters.get('symbol'):
                    query = query.filter(Signal.symbol == filters['symbol'])
                if filters.get('timeframe'):
                    query = query.filter(Signal.timeframe == filters['timeframe'])
                if filters.get('signal_type'):
                    query = query.filter(Signal.signal_type == filters['signal_type'])
                if filters.get('verification_status'):
                    query = query.filter(Signal.verification_status == filters['verification_status'])
                if filters.get('start_date'):
                    query = query.filter(Signal.created_at >= filters['start_date'])
                if filters.get('end_date'):
                    query = query.filter(Signal.created_at <= filters['end_date'])
            
            # ترتيب النتائج
            query = query.order_by(Signal.created_at.desc())
            
            # تطبيق الحد
            if filters and filters.get('limit'):
                query = query.limit(filters['limit'])
            
            # جلب النتائج
            signals = query.all()
            
            return {
                'status': 'success',
                'signals': [signal.to_dict() for signal in signals]
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _determine_signal_type(self, analysis: Dict) -> str:
        """تحديد نوع الإشارة"""
        trend = analysis['trend']['overall_trend']
        strength = analysis['strength']['overall_strength']
        
        if trend == 'bullish' and strength == 'strong':
            return 'BUY'
        elif trend == 'bearish' and strength == 'strong':
            return 'SELL'
        elif trend == 'bullish' and strength == 'weak':
            return 'BUY_WEAK'
        elif trend == 'bearish' and strength == 'weak':
            return 'SELL_WEAK'
        return 'NEUTRAL'
    
    def _calculate_levels(self, symbol: str, timeframe: str, signal_type: str) -> Dict:
        """حساب مستويات الدخول والخروج"""
        try:
            # جلب بيانات السوق
            market_data = self.data_service.fetch_market_data(symbol, timeframe)
            if market_data['status'] == 'error':
                raise Exception(market_data['message'])
            
            # حساب المستويات
            current_price = market_data['data'][-1]['close']
            atr = self._calculate_atr(market_data['data'])
            
            if signal_type.startswith('BUY'):
                entry = current_price
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 3)
            else:
                entry = current_price
                stop_loss = current_price + (atr * 2)
                take_profit = current_price - (atr * 3)
            
            # حساب نسبة المخاطرة إلى العائد
            risk = abs(entry - stop_loss)
            reward = abs(take_profit - entry)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            return {
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': risk_reward_ratio
            }
        except Exception as e:
            raise Exception(f"Error calculating levels: {str(e)}")
    
    def _calculate_atr(self, market_data: List[Dict]) -> float:
        """حساب متوسط المدى الحقيقي (ATR)"""
        try:
            df = pd.DataFrame(market_data)
            
            # حساب المدى الحقيقي
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            
            # حساب المتوسط المتحرك البسيط
            atr = true_range.rolling(window=14).mean().iloc[-1]
            
            return atr
        except Exception as e:
            raise Exception(f"Error calculating ATR: {str(e)}")
    
    def _calculate_confidence_score(self, analysis: Dict) -> float:
        """حساب درجة الثقة"""
        try:
            # تحليل الاتجاه
            trend_score = 1.0 if analysis['trend']['overall_trend'] in ['bullish', 'bearish'] else 0.5
            
            # تحليل القوة
            strength_score = 1.0 if analysis['strength']['overall_strength'] in ['strong', 'weak'] else 0.5
            
            # تحليل التقلب
            volatility_score = 0.8 if analysis['volatility']['overall_volatility'] == 'moderate' else 0.5
            
            # تحليل الحجم
            volume_score = 1.0 if analysis['volume']['overall_volume'] == 'high' else 0.5
            
            # حساب الدرجة النهائية
            confidence_score = (trend_score + strength_score + volatility_score + volume_score) / 4
            
            return round(confidence_score, 2)
        except Exception as e:
            raise Exception(f"Error calculating confidence score: {str(e)}") 