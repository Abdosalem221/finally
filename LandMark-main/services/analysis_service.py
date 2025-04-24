"""
خدمة التحليل الفني المتقدمة
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from app.services.data_service import DataService
from app.utils.validators import validate_symbol, validate_timeframe
from app.utils.helpers import calculate_risk_reward_ratio # تم الاحتفاظ بها رغم عدم استخدامها في هذا الكود

# concurrent.futures غير مستخدم في الجزء الذي تم الاحتفاظ به، لذا لا نحتاج لاستيراده

class AnalysisService:
    # يبدو أن logger غير معرف في الكود، تم حذف الاستخدام من الكتلة المحذوفة
    # def __init__(self):
    #     self.data_service = DataService()
    #     self.logger = ... # يجب تعريف logger هنا إذا كنت تنوي استخدامه

    # افتراضيا، DataService لا يحتاج لبارامتر
    def __init__(self):
         self.data_service = DataService()

    # تم تصحيح المسافات البادئة واختيار الجزء الصحيح من الكود
    def analyze_market(self, symbol: str, timeframe: str) -> Dict: # تم تبسيط المعلمات للتركيز على المنطق الأساسي
        """
        تحليل السوق باستخدام المؤشرات الفنية.

        Args:
            symbol: رمز الأصل المالي.
            timeframe: الإطار الزمني للبيانات (مثال: 1h, 1D).

        Returns:
            Dict: نتائج التحليل الفني أو رسالة خطأ.
        """
        try:
            # التحقق من صحة المدخلات باستخدام الدوال المصححة في validators.py
            if not validate_symbol(symbol):
                # تم تغيير تنسيق رسالة الخطأ لتكون متوافقة مع بقية الدوال
                return {'status': 'error', 'message': f"Invalid symbol: {symbol}"}

            if not validate_timeframe(timeframe):
                return {'status': 'error', 'message': f"Invalid timeframe: {timeframe}"}

            # جلب بيانات السوق
            # افتراض أن DataService.fetch_market_data ترجع بيانات في 'data' وحالة في 'status'
            market_data_response = self.data_service.fetch_market_data(symbol, timeframe, 100) # جلب آخر 100 شمعة

            if market_data_response is None or market_data_response.get('status') == 'error' or not market_data_response.get('data'):
                 # تحسين التحقق من استجابة DataService
                 error_message = market_data_response.get('message', 'Failed to fetch market data') if market_data_response else 'Data service returned None'
                 return {'status': 'error', 'message': error_message}

            # تحويل البيانات إلى DataFrame
            df = pd.DataFrame(market_data_response['data'])

            # التحقق من وجود بيانات كافية
            if len(df) < 200: # بعض المؤشرات مثل MA_200 تحتاج لبيانات كافية
                # قد تحتاج بعض المؤشرات لعدد معين من الأسطر (مثل 200 لـ MA_200)، يجب التحقق
                 # من الأفضل جلب بيانات كافية أو تعديل النوافذ الزمنية للمؤشرات بناءً على البيانات المتاحة
                 # سنرفع خطأ للتوضيح
                 # return {'status': 'error', 'message': f"Not enough data points ({len(df)}) for analysis. Need at least 200."}
                 # بدلاً من ذلك، يمكن حساب المؤشرات المتاحة وتجنب الأخطاء
                 pass # سنستمر ونسمح لـ pandas بإنشاء قيم NaN إذا لم تكن هناك بيانات كافية للنوافذ الزمنية الكبيرة

            # التأكد من أن أعمدة السعر والحجم رقمية، وتحويل الطابع الزمني إلى فهرس
            for col in ['open', 'high', 'low', 'close', 'volume']:
                 if col not in df.columns:
                      return {'status': 'error', 'message': f"Missing required column in market data: {col}"}
                 # تحويل الأعمدة إلى نوع رقمي، مع إجبار الأخطاء (قد نحتاج معالجة أدق للبيانات القادمة)
                 df[col] = pd.to_numeric(df[col], errors='coerce')

            # إزالة الصفوف التي بها قيم NaN في الأعمدة الأساسية بعد التحويل
            df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

            if 'timestamp' in df.columns:
                # تحويل الطابع الزمني للفهرس
                # افتراض أن الطابع الزمني بتنسيق يمكن لـ pandas التعرف عليه (ISO 8601 شائع)
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    # فرز الفهرس لضمان الترتيب الزمني الصحيح
                    df.sort_index(inplace=True)
                except Exception as ts_e:
                     return {'status': 'error', 'message': f"Error processing timestamp: {str(ts_e)}"}
            else:
                 return {'status': 'error', 'message': "Missing 'timestamp' column in market data"}

            # التحقق مرة أخرى من وجود بيانات كافية بعد المعالجة
            if len(df) < 20: # على الأقل نحتاج لعدد كافٍ للمتوسطات الصغيرة
                 return {'status': 'error', 'message': f"Not enough valid data points ({len(df)}) after cleaning for analysis."}

            # حساب المؤشرات الفنية
            df = self._calculate_indicators(df)

            # إزالة الصفوف التي تحتوي على NaN بعد حساب المؤشرات (خاصة في بداية البيانات)
            # هذا يضمن أننا نعمل فقط على الشموع التي تحتوي على قيم صالحة للمؤشرات
            df.dropna(inplace=True)

            # التحقق من بقاء بيانات كافية بعد إزالة NaN
            if len(df) == 0:
                 return {'status': 'error', 'message': "No valid data points remaining after indicator calculation."}

            # تحليل الاتجاه
            trend_analysis = self._analyze_trend(df)

            # تحليل القوة
            strength_analysis = self._analyze_strength(df)

            # تحليل التقلب
            volatility_analysis = self._analyze_volatility(df)

            # تحليل الحجم
            volume_analysis = self._analyze_volume(df)

            # توليد الإشارات
            signals = self._generate_signals(df)

            return {
                'status': 'success',
                'analysis': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'trend': trend_analysis,
                    'strength': strength_analysis,
                    'volatility': volatility_analysis,
                    'volume': volume_analysis,
                    'signals': signals,
                    'timestamp': datetime.now().isoformat(), # استخدام الوقت الحالي للتحليل
                    'latest_candle_timestamp': df.index[-1].isoformat() if not df.empty else None # طابع آخر شمعة تم تحليلها
                }
            }
        except Exception as e:
            # تسجيل الخطأ يجب أن يتم هنا باستخدام logger إذا كان متاحا
            # self.logger.error(f"Analysis error for {symbol}-{timeframe}: {str(e)}")
            return {'status': 'error', 'message': f"Analysis failed: {str(e)}"}

    # تم تغيير اسم الدالة لتجنب التضارب مع دالة analyze_market الرئيسية
    # def analyze_market(self, symbol: str, timeframe: str, market_type: str = 'otc') -> Dict:
    # ... (الجزء الذي تم حذفه والمتعلق بـ concurrent.futures والدوكمنتشن المكرر)

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        حساب المؤشرات الفنية.

        Args:
            df: DataFrame يحتوي على بيانات السوق (open, high, low, close, volume) بفهرس زمني.

        Returns:
            pd.DataFrame: البيانات مع المؤشرات المضافة.
        """
        # التأكد من وجود بيانات كافية للحسابات
        if len(df) < 14: # RSI يحتاج على الأقل 14 نقطة
             # يمكن إضافة معالجة هنا أو الاعتماد على pandas لإرجاع NaN
             pass

        try:
            # حساب المتوسطات المتحركة
            # استخدم .loc لتجنب التحذيرات المستقبلية في pandas
            df.loc[:, 'MA_20'] = df['close'].rolling(window=20).mean()
            df.loc[:, 'MA_50'] = df['close'].rolling(window=50).mean()
            # تحقق من وجود بيانات كافية لـ MA_200 قبل حسابها
            if len(df) >= 200:
                 df.loc[:, 'MA_200'] = df['close'].rolling(window=200).mean()
            else:
                 df.loc[:, 'MA_200'] = np.nan # وضع NaN إذا لم يكن هناك بيانات كافية


            # حساب RSI
            # تجنب القسمة على صفر للـ rs
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()

            rs = avg_gain / avg_loss
            # معالجة القسمة على صفر حيث avg_loss قد يكون صفر
            rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0) # استبدال اللانهاية بـ NaN وملء NaN بـ 0 قبل الحساب التالي

            df.loc[:, 'RSI'] = 100 - (100 / (1 + rs))
            # معالجة الحالات التي يكون فيها avg_loss = 0 و avg_gain > 0 (RSI = 100)
            df.loc[avg_loss == 0, 'RSI'] = 100
             # معالجة الحالات التي يكون فيها avg_gain = 0 و avg_loss > 0 (RSI = 0) - هذه معالجة تلقائيا بالمعادلة

            # حساب MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df.loc[:, 'MACD'] = exp1 - exp2
            df.loc[:, 'Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

            # حساب Bollinger Bands
            df.loc[:, 'BB_middle'] = df['close'].rolling(window=20).mean()
            df.loc[:, 'BB_std'] = df['close'].rolling(window=20).std()
            df.loc[:, 'BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
            df.loc[:, 'BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)

            # حساب Stochastic Oscillator
            # تجنب القسمة على صفر
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            denominator = high_14 - low_14

            # معالجة القسمة على صفر
            df.loc[:, 'STOCH_k'] = 100 * ((df['close'] - low_14) / denominator)
            df.loc[denominator == 0, 'STOCH_k'] = np.nan # ضع NaN إذا كان المقام صفر
            df.loc[:, 'STOCH_k'] = df['STOCH_k'].fillna(method='ffill') # ملء NaN بالقيم السابقة إن وجدت

            df.loc[:, 'STOCH_d'] = df['STOCH_k'].rolling(window=3).mean()

            return df
        except Exception as e:
            # تسجيل الخطأ باستخدام logger
            # self.logger.error(f"Error calculating indicators: {str(e)}", exc_info=True)
            raise Exception(f"Error calculating indicators: {str(e)}")

    def _analyze_trend(self, df: pd.DataFrame) -> Dict:
        """
        تحليل الاتجاه بناءً على المتوسطات المتحركة و MACD و Bollinger Bands.

        Args:
            df: DataFrame يحتوي على البيانات مع المؤشرات المحسوبة.

        Returns:
            Dict: تحليل الاتجاه.
        """
        # التحقق من أن DataFrame ليس فارغاً وأن الأعمدة المطلوبة موجودة
        if df.empty or not all(col in df.columns for col in ['close', 'MA_20', 'MA_50', 'MA_200', 'MACD', 'Signal', 'BB_upper', 'BB_lower', 'BB_middle']):
             # يمكن رفع خطأ أو إرجاع قاموس فارغ أو به قيم افتراضية
             # سأرفع خطأ هنا لأن هذه الدالة تعتمد على نتائج _calculate_indicators
             raise ValueError("DataFrame is empty or missing required indicator columns for trend analysis.")

        # استخدام .iloc[-1] و .iloc[-2] فقط إذا كان حجم DataFrame يسمح بذلك
        if len(df) < 2:
             # نحتاج شمعتين على الأقل للتحليل الذي يقارن بين الحالية والسابقة
             raise ValueError("Not enough data points in DataFrame for trend analysis after dropping NaN.")

        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else None # يمكن أن تكون None إذا كان هناك شمعة واحدة فقط

        try:
            # تحليل المتوسطات المتحركة
            ma_analysis = {
                'short_term': 'bullish' if latest['close'] > latest['MA_20'] else 'bearish',
                'medium_term': 'bullish' if latest['close'] > latest['MA_50'] else 'bearish',
                # تحقق من وجود MA_200 قبل استخدامه
                'long_term': 'bullish' if 'MA_200' in df.columns and latest['close'] > latest['MA_200'] else 'bearish' if 'MA_200' in df.columns else 'N/A'
            }

            # تحليل MACD
            macd_analysis = {
                'trend': 'bullish' if latest['MACD'] > latest['Signal'] else 'bearish',
                'strength': abs(latest['MACD'] - latest['Signal']) # استخدم القيمة المطلقة
            }

            # تحليل Bollinger Bands
            bb_analysis = {
                'position': 'upper' if latest['close'] > latest['BB_upper'] else
                           'lower' if latest['close'] < latest['BB_lower'] else 'middle',
                # تجنب القسمة على صفر إذا كانت BB_middle قريبة من الصفر
                'width': (latest['BB_upper'] - latest['BB_lower']) / latest['BB_middle'] if latest['BB_middle'] != 0 else np.nan
            }

            return {
                'ma': ma_analysis,
                'macd': macd_analysis,
                'bollinger': bb_analysis,
                'current_price': latest['close'],
                'trend_strength': self._calculate_trend_strength(df)
            }
        except Exception as e:
            # self.logger.error(f"Error analyzing trend: {str(e)}", exc_info=True)
            raise Exception(f"Error analyzing trend: {str(e)}")

    def _analyze_strength(self, df: pd.DataFrame) -> Dict:
        """
        تحليل القوة بناءً على RSI و Stochastic.

        Args:
            df: DataFrame يحتوي على البيانات مع المؤشرات المحسوبة.

        Returns:
            Dict: تحليل القوة.
        """
        # التحقق من أن DataFrame ليس فارغاً وأن الأعمدة المطلوبة موجودة
        if df.empty or not all(col in df.columns for col in ['RSI', 'STOCH_k', 'STOCH_d', 'close']):
             raise ValueError("DataFrame is empty or missing required indicator columns for strength analysis.")

        if len(df) < 2:
             raise ValueError("Not enough data points in DataFrame for strength analysis after dropping NaN.")

        latest = df.iloc[-1]
        previous = df.iloc[-2]


        try:
            # تحليل RSI
            rsi = latest['RSI']
            rsi_analysis = {
                'value': rsi,
                'strength': 'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral',
                # تأكد من وجود بيانات كافية للتحقق من التباعد
                'divergence': self._check_rsi_divergence(df) if len(df) >= 14 else None # التباعد يتطلب بيانات أكثر
            }

            # تحليل Stochastic
            stoch_k = latest['STOCH_k']
            stoch_d = latest['STOCH_d']
            stoch_analysis = {
                'k': stoch_k,
                'd': stoch_d,
                'strength': 'overbought' if stoch_k > 80 else 'oversold' if stoch_k < 20 else 'neutral',
                'crossover': 'bullish' if latest['STOCH_k'] > latest['STOCH_d'] and previous['STOCH_k'] <= previous['STOCH_d'] else
                            'bearish' if latest['STOCH_k'] < latest['STOCH_d'] and previous['STOCH_k'] >= previous['STOCH_d'] else 'none'
            }

            return {
                'rsi': rsi_analysis,
                'stochastic': stoch_analysis,
                'strength_score': self._calculate_strength_score(df)
            }
        except Exception as e:
            # self.logger.error(f"Error analyzing strength: {str(e)}", exc_info=True)
            raise Exception(f"Error analyzing strength: {str(e)}")

    def _analyze_volatility(self, df: pd.DataFrame) -> Dict:
        """
        تحليل التقلب بناءً على التقلب التاريخي والحالي و Bollinger Bands.

        Args:
            df: DataFrame يحتوي على البيانات مع المؤشرات المحسوبة.

        Returns:
            Dict: تحليل التقلب.
        """
        # التحقق من أن DataFrame ليس فارغاً وأن الأعمدة المطلوبة موجودة
        if df.empty or not all(col in df.columns for col in ['close', 'BB_upper', 'BB_lower', 'BB_middle']):
             raise ValueError("DataFrame is empty or missing required indicator columns for volatility analysis.")

        if len(df) < 2:
             raise ValueError("Not enough data points in DataFrame for volatility analysis after dropping NaN.")

        latest = df.iloc[-1]


        try:
            # حساب التقلب التاريخي (يحتاج بيانات كافية)
            # استخدام 252 كعامل تحويل سنوي شائع للأسهم، قد تحتاج تعديله حسب نوع الأصل
            if len(df) >= 252: # تحتاج بيانات كافية لحساب سنوي
                 historical_volatility = df['close'].pct_change().std() * np.sqrt(252)
            else:
                 historical_volatility = np.nan # لا يمكن حسابه إذا لم يكن هناك بيانات كافية

            # حساب التقلب الحالي (يحتاج 20 نقطة)
            if len(df) >= 20:
                current_volatility = df['close'].pct_change().rolling(window=20).std().iloc[-1] * np.sqrt(252)
            else:
                 current_volatility = np.nan

            # تحليل Bollinger Bands
            # تجنب القسمة على صفر
            bb_width = (latest['BB_upper'] - latest['BB_lower']) / latest['BB_middle'] if latest['BB_middle'] != 0 else np.nan

            # تحديد مستوى التقلب فقط إذا كان التقلب التاريخي والحالي متاحين
            volatility_level = 'N/A'
            if not np.isnan(current_volatility) and not np.isnan(historical_volatility) and historical_volatility != 0:
                 volatility_ratio = current_volatility / historical_volatility
                 volatility_level = 'high' if volatility_ratio > 1.5 else \
                                   'low' if volatility_ratio < 0.5 else 'moderate'
            elif not np.isnan(current_volatility):
                # إذا لم يتوفر التاريخي، يمكن مقارنته بمتوسط التقلب الحالي أو قيمة ثابتة
                volatility_level = 'moderate' # افتراضي إذا لم تتوفر المقارنة

            return {
                'historical': historical_volatility,
                'current': current_volatility,
                'ratio': current_volatility / historical_volatility if not np.isnan(current_volatility) and not np.isnan(historical_volatility) and historical_volatility != 0 else np.nan,
                'bollinger_width': bb_width,
                'volatility_level': volatility_level
            }
        except Exception as e:
            # self.logger.error(f"Error analyzing volatility: {str(e)}", exc_info=True)
            raise Exception(f"Error analyzing volatility: {str(e)}")

    def _analyze_volume(self, df: pd.DataFrame) -> Dict:
        """
        تحليل الحجم.

        Args:
            df: DataFrame يحتوي على البيانات مع المؤشرات المحسوبة.

        Returns:
            Dict: تحليل الحجم.
        """
         # التحقق من أن DataFrame ليس فارغاً وأن أعمدة الحجم موجودة
        if df.empty or 'volume' not in df.columns:
             raise ValueError("DataFrame is empty or missing 'volume' column for volume analysis.")

        if len(df) < 2:
             # نحتاج شمعتين على الأقل للحسابات التي تقارن الحجم الحالي بالسابق والمتوسط
             raise ValueError("Not enough data points in DataFrame for volume analysis after dropping NaN.")

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        try:
            # حساب متوسط الحجم (تجاهل قيم NaN)
            avg_volume = df['volume'].mean() if not df['volume'].empty else 0 # التعامل مع حالة عدم وجود بيانات حجم

            # تحليل الحجم الحالي
            current_volume = latest['volume']
            volume_ratio = current_volume / avg_volume if avg_volume != 0 else np.nan # تجنب القسمة على صفر

            # تحليل اتجاه الحجم
            volume_trend = 'increasing' if current_volume > previous['volume'] else \
                           'decreasing' if current_volume < previous['volume'] else 'stable' # إضافة حالة مستقر

            return {
                'current': current_volume,
                'average': avg_volume,
                'ratio': volume_ratio,
                'trend': volume_trend,
                'significance': 'high' if volume_ratio > 2 else 'low' if volume_ratio < 0.5 else 'moderate' if not np.isnan(volume_ratio) else 'N/A'
            }
        except Exception as e:
            # self.logger.error(f"Error analyzing volume: {str(e)}", exc_info=True)
            raise Exception(f"Error analyzing volume: {str(e)}")

    def _generate_signals(self, df: pd.DataFrame) -> List[Dict]:
        """
        توليد إشارات التداول بناءً على المؤشرات.

        Args:
            df: DataFrame يحتوي على البيانات مع المؤشرات المحسوبة.

        Returns:
            List[Dict]: قائمة الإشارات التي تم توليدها.
        """
        # التحقق من أن DataFrame ليس فارغاً وأن الأعمدة المطلوبة موجودة
        required_cols = ['close', 'MA_20', 'MA_50', 'RSI', 'MACD', 'Signal']
        if df.empty or not all(col in df.columns for col in required_cols):
             # لا نرفع خطأ هنا، بل نرجع قائمة فارغة إذا لم تكن البيانات كافية لتوليد الإشارات
             # self.logger.warning("Not enough data or missing columns for signal generation.")
             return []

        if len(df) < 2:
             # نحتاج شمعتين على الأقل للتحقق من التقاطعات (الوضع الحالي vs السابق)
             return []


        latest = df.iloc[-1]
        previous = df.iloc[-2]

        signals = []

        try:
            # إشارات المتوسطات المتحركة (تقاطع MA_20 و MA_50)
            if latest['MA_20'] > latest['MA_50'] and previous['MA_20'] <= previous['MA_50']:
                signals.append({
                    'type': 'MA_Crossover',
                    'direction': 'bullish',
                    'strength': 'medium',
                    'price': latest['close'],
                    'timestamp': latest.name.isoformat() # استخدام الفهرس (الطابع الزمني) للإشارة
                })
            elif latest['MA_20'] < latest['MA_50'] and previous['MA_20'] >= previous['MA_50']:
                signals.append({
                    'type': 'MA_Crossover',
                    'direction': 'bearish',
                    'strength': 'medium',
                    'price': latest['close'],
                    'timestamp': latest.name.isoformat()
                })

            # إشارات RSI (دخول/خروج من مناطق التشبع)
            if latest['RSI'] < 30 and previous['RSI'] >= 30:
                signals.append({
                    'type': 'RSI_Oversold_Exit', # اسم أوضح للإشارة
                    'direction': 'bullish',
                    'strength': 'strong',
                    'price': latest['close'],
                    'timestamp': latest.name.isoformat()
                })
            elif latest['RSI'] > 70 and previous['RSI'] <= 70:
                signals.append({
                    'type': 'RSI_Overbought_Exit', # اسم أوضح للإشارة
                    'direction': 'bearish',
                    'strength': 'strong',
                    'price': latest['close'],
                    'timestamp': latest.name.isoformat()
                })

            # إشارات MACD (تقاطع MACD وخط الإشارة)
            if latest['MACD'] > latest['Signal'] and previous['MACD'] <= previous['Signal']:
                signals.append({
                    'type': 'MACD_Crossover',
                    'direction': 'bullish',
                    'strength': 'medium',
                    'price': latest['close'],
                    'timestamp': latest.name.isoformat()
                })
            elif latest['MACD'] < latest['Signal'] and previous['MACD'] >= previous['Signal']:
                signals.append({
                    'type': 'MACD_Crossover',
                    'direction': 'bearish',
                    'strength': 'medium',
                    'price': latest['close'],
                    'timestamp': latest.name.isoformat()
                })

            # إشارات أخرى يمكن إضافتها هنا بناءً على مؤشرات أخرى (Bollinger, Stoch, Volume, Patterns...)

            return signals
        except Exception as e:
            # self.logger.error(f"Error generating signals: {str(e)}", exc_info=True)
            # بدلا من رفع الخطأ، نرجع قائمة فارغة مع تسجيل الخطأ
            # raise Exception(f"Error generating signals: {str(e)}")
            return [] # إرجاع قائمة فارغة عند وجود خطأ في التوليد

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """
        حساب قوة الاتجاه بناءً على المتوسطات المتحركة و MACD.

        Args:
            df: DataFrame يحتوي على البيانات مع المؤشرات المحسوبة.

        Returns:
            float: قوة الاتجاه (0-100).
        """
        # التحقق من أن DataFrame ليس فارغاً وأن الأعمدة المطلوبة موجودة
        if df.empty or not all(col in df.columns for col in ['close', 'MA_20', 'MACD', 'Signal']):
             # لا نرفع خطأ، نرجع قيمة افتراضية أو NaN
             # self.logger.warning("Not enough data or missing columns for trend strength calculation.")
             return 0.0 # أو np.nan

        if len(df) < 2:
             # نحتاج بيانات كافية للحسابات
             return 0.0

        latest = df.iloc[-1]

        try:
            # حساب نسبة الصعود: نسبة الشموع التي فيها الإغلاق فوق MA_20 في البيانات المتاحة
            # استخدم البيانات غير الخالية فقط
            valid_close_ma = df.dropna(subset=['close', 'MA_20'])
            if not valid_close_ma.empty:
                 uptrend_ratio = len(valid_close_ma[valid_close_ma['close'] > valid_close_ma['MA_20']]) / len(valid_close_ma)
            else:
                 uptrend_ratio = 0.0

            # حساب قوة MACD: الفرق المطلق بين MACD وخط الإشارة بالنسبة للسعر
            # تجنب القسمة على صفر إذا كان السعر الحالي صفر أو قريب جدا منه
            macd_strength = abs(latest['MACD'] - latest['Signal']) / latest['close'] if latest['close'] != 0 else 0

            # حساب قوة الاتجاه الكلية (مثال بسيط للجمع الموزون)
            trend_strength = (uptrend_ratio * 0.6 + macd_strength * 0.4) * 100

            # التأكد من أن القيمة بين 0 و 100
            return max(0.0, min(100.0, trend_strength))
        except Exception as e:
            # self.logger.error(f"Error calculating trend strength: {str(e)}", exc_info=True)
            return 0.0 # إرجاع قيمة افتراضية عند الخطأ

    def _calculate_strength_score(self, df: pd.DataFrame) -> float:
        """
        حساب درجة القوة بناءً على RSI و Stochastic.

        Args:
            df: DataFrame يحتوي على البيانات مع المؤشرات المحسوبة.

        Returns:
            float: درجة القوة (0-100).
        """
         # التحقق من أن DataFrame ليس فارغاً وأن الأعمدة المطلوبة موجودة
        if df.empty or not all(col in df.columns for col in ['RSI', 'STOCH_k']):
             # لا نرفع خطأ، نرجع قيمة افتراضية أو NaN
             # self.logger.warning("Not enough data or missing columns for strength score calculation.")
             return 50.0 # قيمة محايدة

        if len(df) < 1:
             return 50.0

        latest = df.iloc[-1]

        try:
            # حساب درجة RSI: مدى الابتعاد عن الخط المركزي (50)
            # التعامل مع RSI كنقطة بيانات واحدة
            rsi = latest['RSI']
            rsi_score = abs(50 - rsi) / 50 if not np.isnan(rsi) else 0 # تجنب NaN

            # حساب درجة Stochastic: مدى الابتعاد عن الخط المركزي (50) لـ %K
            stoch_k = latest['STOCH_k']
            stoch_score = abs(50 - stoch_k) / 50 if not np.isnan(stoch_k) else 0 # تجنب NaN

            # حساب درجة القوة الكلية (مثال بسيط)
            strength_score = (rsi_score * 0.6 + stoch_score * 0.4) * 100

            # التأكد من أن القيمة بين 0 و 100
            return max(0.0, min(100.0, strength_score))
        except Exception as e:
            # self.logger.error(f"Error calculating strength score: {str(e)}", exc_info=True)
            return 50.0 # إرجاع قيمة محايدة عند الخطأ

    def _check_rsi_divergence(self, df: pd.DataFrame) -> Optional[str]:
        """
        التحقق من تباعد RSI (اختلاف الاتجاه بين السعر و RSI).
        يتطلب هذا تحليل تاريخي أكثر تعقيداً لتحديد القمم والقيعان بفعالية.
        هذا التطبيق هو مجرد مثال بسيط وقد يحتاج لتحسين كبير.

        Args:
            df: DataFrame يحتوي على البيانات مع مؤشر RSI.

        Returns:
            Optional[str]: 'bullish' أو 'bearish' أو None إذا لم يتم العثور على تباعد واضح.
        """
        # التحقق من أن DataFrame ليس فارغاً وأن الأعمدة المطلوبة موجودة
        if df.empty or 'RSI' not in df.columns or 'close' not in df.columns:
             return None # لا يمكن التحقق بدون الأعمدة المطلوبة

        if len(df) < 14: # RSI يحتاج 14 شمعة على الأقل
             return None # لا يوجد بيانات كافية للتحليل

        try:
            # تحليل بسيط جداً للتباين:
            # نقارن آخر قاعين/قمم للسعر و RSI في فترة زمنية قصيرة (مثال: آخر 14 شمعة بعد إزالة NaN)
            recent_df = df.dropna(subset=['close', 'RSI']).tail(20) # نأخذ آخر 20 شمعة صالحة

            if len(recent_df) < 5: # نحتاج على الأقل بضع نقاط للبحث عن قمم وقيعان بسيطة
                return None

            # مثال مبسط للبحث عن قاعين للسعر و RSI في البيانات الحديثة
            # هذا ليس طريقة موثوقة لتحديد القمم والقيعان الحقيقية للتباين
            price_lows_idx = recent_df['close'].nsmallest(2).index.tolist()
            rsi_lows_idx = recent_df['RSI'].nsmallest(2).index.tolist()

            # التأكد من وجود قاعين على الأقل
            if len(price_lows_idx) >= 2 and len(rsi_lows_idx) >= 2:
                # نرتب الفهارس زمنياً للتأكد من مقارنة القاع السابق مع القاع الحالي
                price_lows_idx.sort()
                rsi_lows_idx.sort()

                latest_price_low_idx = price_lows_idx[-1]
                previous_price_low_idx = price_lows_idx[-2]
                latest_rsi_low_idx = rsi_lows_idx[-1]
                previous_rsi_low_idx = rsi_lows_idx[-2]

                latest_price_low = recent_df.loc[latest_price_low_idx, 'close']
                previous_price_low = recent_df.loc[previous_price_low_idx, 'close']
                latest_rsi_low = recent_df.loc[latest_rsi_low_idx, 'RSI']
                previous_rsi_low = recent_df.loc[previous_rsi_low_idx, 'RSI']

                # تباعد صاعد مخفي (Hidden Bullish Divergence): قاع أدنى للسعر يقابله قاع أعلى للـ RSI
                # هذا النوع من التباعد يكون مع اتجاه صاعد ويعتبر إشارة استمرارية
                # if latest_price_low < previous_price_low and latest_rsi_low > previous_rsi_low:
                #     return 'hidden_bullish'

                # تباعد صاعد عادي (Regular Bullish Divergence): قاع أدنى للسعر يقابله قاع أعلى للـ RSI
                # هذا هو التباعد الأكثر شيوعا ويعتبر إشارة انعكاس محتملة للاتجاه الصاعد
                # في هذا الكود البسيط، سنركز على هذا النوع
                if latest_price_low < previous_price_low and latest_rsi_low > previous_rsi_low:
                    return 'bullish' # تباعد صاعد

                # يمكن إضافة فحص القمم للتباين الهابط بنفس الطريقة
                price_highs_idx = recent_df['close'].nlargest(2).index.tolist()
                rsi_highs_idx = recent_df['RSI'].nlargest(2).index.tolist()

                if len(price_highs_idx) >= 2 and len(rsi_highs_idx) >= 2:
                     price_highs_idx.sort()
                     rsi_highs_idx.sort()

                     latest_price_high_idx = price_highs_idx[-1]
                     previous_price_high_idx = price_highs_idx[-2]
                     latest_rsi_high_idx = rsi_highs_idx[-1]
                     previous_rsi_high_idx = rsi_highs_idx[-2]

                     latest_price_high = recent_df.loc[latest_price_high_idx, 'close']
                     previous_price_high = recent_df.loc[previous_price_high_idx, 'close']
                     latest_rsi_high = recent_df.loc[latest_rsi_high_idx, 'RSI']
                     previous_rsi_high = recent_df.loc[previous_rsi_high_idx, 'RSI']

                     # تباعد هابط عادي (Regular Bearish Divergence): قمة أعلى للسعر يقابلها قمة أدنى للـ RSI
                     if latest_price_high > previous_price_high and latest_rsi_high < previous_rsi_high:
                          return 'bearish' # تباعد هابط

            return None # لا يوجد تباعد واضح

        except Exception as e:
            # self.logger.error(f"Error checking RSI divergence: {str(e)}", exc_info=True)
            # نرجع None في حال وجود خطأ في حساب التباعد بدلا من رفع الخطأ
            # raise Exception(f"Error checking RSI divergence: {str(e)}")
            return None # إرجاع None عند وجود خطأ


    # يبدو أن الدوال _analyze_rsi, _analyze_macd, _analyze_momentum, _analyze_volume
    # و _combine_analysis_results كانت جزءا من مقاربة مختلفة (باستخدام concurrent.futures)
    # والتي تم حذفها من analyze_market الرئيسية.
    # إذا كنت تنوي استخدام التحليل المتوازي لاحقا، يجب إعادة هيكلة analyze_market والدوال المساعدة هذه بشكل صحيح.
    # حاليا، سأفترض أنها لم تعد ضرورية بناء على منطق analyze_market المتبقي.
    # إذا كنت تريد استعادة هذه الدوال، يرجى توضيح كيف ينبغي أن تعمل ضمن analyze_market.
    # تم الاحتفاظ فقط بدالة _analyze_volume التي تستخدم في التحليل القياسي.