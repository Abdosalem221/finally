"""
خدمة الذكاء الاصطناعي المتقدمة
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
# تأكد من تثبيت مكتبة logging
import logging # <--- إضافة استيراد logging

from app.services.data_service import DataService
from app.services.analysis_service import AnalysisService
from app.utils.validators import validate_symbol, validate_timeframe

# تهيئة بسيطة لـ logger
logging.basicConfig(level=logging.INFO) # يمكنك تغيير المستوى حسب الحاجة (DEBUG, INFO, WARNING, ERROR)
logger = logging.getLogger(__name__) # <--- تعريف logger

# تم تغيير اسم الكلاس ليناسب الاستيراد المتوقع في services/__init__.py
# إذا كنت تريد الاحتفاظ بالاسم AIService، يجب تغيير سطر الاستيراد في __init__.py
# لكن بناء على رسالة الخطأ، يبدو أن AIAnalysis هو الاسم المتوقع
# لا، رسالة الخطأ تقول "cannot import name 'AIAnalysis' from 'app.services.ai_service'".
# هذا يعني أن الملف اسمه ai_service.py ولكن الكلاس بداخله ليس اسمه AIAnalysis.
# اسم الكلاس الموجود في الكود هو AIService.
# الحل الصحيح هو تغيير سطر الاستيراد في services/__init__.py
# من: from .ai_service import AIAnalysis
# إلى: from .ai_service import AIService
# لن أغير اسم الكلاس في هذا الملف. سأفترض أن سطر الاستيراد في __init__.py سيتم تصحيحه.
class AIService: # <-- اسم الكلاس الأصلي في الكود الذي أرسلته
    def __init__(self):
        self.data_service = DataService()
        self.analysis_service = AnalysisService()
        self.model = None
        self.scaler = MinMaxScaler()
        # self.logger = logger # يمكنك تهيئة logger هنا إذا كنت تفضل
        self.load_model()

    def load_model(self):
        """تحميل نموذج التعلم العميق"""
        try:
            # تحديد مسار النموذج
            model_path = os.getenv('AI_MODEL_PATH', 'models/ai_model.h5')

            # التأكد من وجود دليل models قبل محاولة التحميل أو الإنشاء/الحفظ
            model_dir = os.path.dirname(model_path)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir)
                logger.info(f"Created model directory: {model_dir}")

            if os.path.exists(model_path):
                logger.info(f"Loading AI model from {model_path}")
                # تعطيل رسائل التحذير المتعلقة بـ CUDA/cuDNN عند تحميل النموذج
                # يمكن أن تكون مزعجة إذا لم تستخدم GPU
                # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # مثال لتقليل رسائل TensorFlow

                self.model = tf.keras.models.load_model(model_path)
                logger.info("AI model loaded successfully.")
            else:
                logger.warning(f"AI model not found at {model_path}. Creating a new model.")
                self._create_model()
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True) # تسجيل الخطأ كاملاً
            logger.warning("Creating a new model due to loading error.")
            self._create_model()

    def _create_model(self):
        """إنشاء نموذج جديد (نموذج Placeholder)"""
        try:
            logger.info("Creating a new default AI model.")
            # نموذج LSTM بسيط كما كان
            self.model = tf.keras.Sequential([
                # التأكد من حجم المدخلات يطابق _prepare_data
                tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(60, len(['open', 'high', 'low', 'close', 'volume']))), # استخدام len(features) للحفاظ على التوافق
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(50, return_sequences=True),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(50),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1) # طبقة إخراج واحدة لتوقع السعر
            ])

            self.model.compile(optimizer='adam', loss='mse')
            logger.info("Default AI model created and compiled.")
        except Exception as e:
            logger.error(f"Error creating default model: {str(e)}", exc_info=True)
            # يمكنك رفع استثناء هنا إذا فشل الإنشاء حتماً
            # raise

    def train_model(self, symbol: str, timeframe: str) -> Dict:
        """تدريب النموذج"""
        try:
            # التحقق من وجود النموذج
            if self.model is None:
                logger.error("Model is not loaded or created. Cannot train.")
                return {'status': 'error', 'message': 'AI model not available for training.'}

            logger.info(f"Starting training for {symbol}-{timeframe}")
            # جلب البيانات التاريخية
            # جلب بيانات لمدة عام واحد قد لا يكون كافياً لـ MA_200 في AnalysisService إذا تم استخدامه لاحقاً
            # ولكن للتدريب بناءً على 60 نقطة، هذا قد يكون كافياً
            # يجب التأكد من أن get_historical_data تجلب بيانات بترتيب زمني صحيح
            end_time = datetime.now()
            start_time = end_time - timedelta(days=365)
            historical_data_response = self.data_service.get_historical_data(
                symbol, timeframe,
                start_time.isoformat(),
                end_time.isoformat()
            )

            if historical_data_response is None or historical_data_response.get('status') == 'error' or not historical_data_response.get('data'):
                error_message = historical_data_response.get('message', 'Failed to fetch historical data') if historical_data_response else 'Data service returned None for historical data'
                logger.error(error_message)
                return {'status': 'error', 'message': error_message}

            # تحضير البيانات
            try:
                X, y = self._prepare_data(historical_data_response['data'])
            except Exception as prep_e:
                 logger.error(f"Error preparing data for training: {str(prep_e)}", exc_info=True)
                 return {'status': 'error', 'message': f"Failed to prepare data for training: {str(prep_e)}"}

            # التحقق من وجود بيانات كافية بعد التحضير
            if X.shape[0] == 0 or y.shape[0] == 0:
                 logger.warning(f"Not enough data points ({len(historical_data_response['data'])} raw) for training after preparation (need > 60).")
                 return {'status': 'error', 'message': 'Not enough data points available for training (need > 60 candles).'}

            logger.info(f"Prepared data shape for training: X={X.shape}, y={y.shape}")

            # تدريب النموذج
            # يمكن إضافة Callbacks مثل EarlyStopping
            history = self.model.fit(
                X, y,
                epochs=50,
                batch_size=32,
                validation_split=0.2, # استخدام جزء من البيانات للتحقق
                verbose=1 # عرض شريط التقدم أو التفاصيل
            )

            # حفظ النموذج
            model_path = os.getenv('AI_MODEL_PATH', 'models/ai_model.h5')
            self.model.save(model_path)
            logger.info(f"Model trained and saved successfully to {model_path}")

            return {
                'status': 'success',
                'message': 'Model trained successfully',
                'history': history.history # إرجاع تاريخ التدريب
            }
        except Exception as e:
            # log the exception details
            logger.error(f"Error during model training for {symbol}-{timeframe}: {str(e)}", exc_info=True)
            return {'status': 'error', 'message': f"Training failed: {str(e)}"}


    def predict_price(self, symbol: str, timeframe: str) -> Dict:
        """توقع السعر المستقبلي (لشمعة واحدة للأمام)"""
        try:
            # التحقق من وجود النموذج
            if self.model is None:
                logger.error("Model is not loaded or created. Cannot predict.")
                return {'status': 'error', 'message': 'AI model not available for prediction.'}

            logger.info(f"Getting data for prediction for {symbol}-{timeframe}")
            # جلب البيانات الأخيرة (تحتاج 60 شمعة لتغذية نموذج LSTM)
            market_data_response = self.data_service.fetch_market_data(symbol, timeframe, 60) # تأكد أن هذه الدالة تجلب 60 نقطة على الأقل

            if market_data_response is None or market_data_response.get('status') == 'error' or not market_data_response.get('data'):
                 error_message = market_data_response.get('message', 'Failed to fetch market data for prediction') if market_data_response else 'Data service returned None for prediction data'
                 logger.error(error_message)
                 return {'status': 'error', 'message': error_message}

            # تحضير البيانات
            try:
                 X = self._prepare_prediction_data(market_data_response['data'])
            except Exception as prep_e:
                 logger.error(f"Error preparing data for prediction: {str(prep_e)}", exc_info=True)
                 return {'status': 'error', 'message': f"Failed to prepare data for prediction: {str(prep_e)}"}

            # التحقق من وجود بيانات كافية بعد التحضير (يجب أن تكون X تحتوي على شمعة واحدة بـ 60 خطوة زمنية)
            if X.shape[0] != 1 or X.shape[1] != 60 or X.shape[2] != len(['open', 'high', 'low', 'close', 'volume']):
                 logger.warning(f"Prediction data has unexpected shape: {X.shape}. Need (1, 60, 5).")
                 return {'status': 'error', 'message': f'Not enough valid data points available for prediction (need 60 candles).'}

            logger.info(f"Prepared data shape for prediction: {X.shape}")

            # توقع السعر
            prediction = self.model.predict(X, verbose=0) # توقع بصمت

            # التأكد من أن prediction ليست فارغة وأن scaler تم تدريبه
            if prediction.shape[0] == 0:
                 logger.error("Model returned empty prediction.")
                 return {'status': 'error', 'message': 'AI model returned empty prediction.'}

            # يجب أن يكون scaler قد تم تدريبه أثناء التدريب (self.scaler.fit_transform)
            # إذا لم يتم التدريب بعد، scaler لن يعمل بشكل صحيح
            if not hasattr(self.scaler, 'scale_'):
                 logger.error("Scaler has not been fitted. Train the model first.")
                 return {'status': 'error', 'message': 'Scaler not fitted. Train the model before predicting.'}

            predicted_price = self.scaler.inverse_transform(prediction)[0][0]

            logger.info(f"Prediction successful. Predicted price: {predicted_price}")

            return {
                'status': 'success',
                'prediction': {
                    'price': float(predicted_price), # التأكد من إرجاع float قياسي
                    'timestamp': datetime.now().isoformat() # طابع زمني وقت التوقع
                }
            }
        except Exception as e:
            logger.error(f"Error during price prediction for {symbol}-{timeframe}: {str(e)}", exc_info=True)
            return {'status': 'error', 'message': f"Prediction failed: {str(e)}"}

    # تم حذف تعريف analyze_market الأول المتضارب
    # def analyze_market(self, symbol: str, timeframe: str, use_deep_learning: bool = True, use_sentiment: bool = True) -> Dict:
    # ... الكود المحذوف ...

    # تم الاحتفاظ بالتعريف الثاني لدالة analyze_market وتصحيح المسافات البادئة
    def analyze_market(self, symbol: str, timeframe: str) -> Dict: # تم تبسيط المعلمات لتجنب التضارب مع التعريف الأول المحذوف
        """
        تحليل السوق باستخدام الذكاء الاصطناعي و التحليل الفني التقليدي.
        (هذا هو الجزء الذي تم الاحتفاظ به من تعريف analyze_market المزدوج)

        Args:
            symbol: رمز الأصل المالي.
            timeframe: الإطار الزمني.

        Returns:
            Dict: نتائج التحليل المدمج.
        """
        try:
            logger.info(f"Starting AI market analysis for {symbol}-{timeframe}")
            # تحليل السوق التقليدي باستخدام AnalysisService
            # تأكد أن AnalysisService يعمل بشكل صحيح ويعيد القاموس المتوقع
            market_analysis_response = self.analysis_service.analyze_market(symbol, timeframe)
            if market_analysis_response['status'] == 'error':
                logger.error(f"Technical analysis failed: {market_analysis_response['message']}")
                return market_analysis_response # إعادة رسالة الخطأ من التحليل الفني

            # توقع السعر باستخدام نموذج الذكاء الاصطناعي
            price_prediction_response = self.predict_price(symbol, timeframe)
            # Note: Prediction might fail if model isn't trained/loaded or data is insufficient
            # We will proceed with analysis even if prediction fails, but mark it as such
            prediction_success = price_prediction_response['status'] == 'success'
            if not prediction_success:
                 logger.warning(f"Price prediction failed: {price_prediction_response['message']}. Proceeding with technical analysis only.")
                 prediction_data = None # لا يوجد بيانات توقع صالحة
            else:
                 prediction_data = price_prediction_response['prediction']


            # هنا، يمكن إضافة منطق لدمج التحليل الفني والتوقع
            # بدلاً من مجرد إعادة نتائج منفصلة أو محاولة "تحليل" المؤشرات والتوقع بشكل منفصل
            # يمكننا بناء هيكل نتائج جديد يجمع بينهما ويعطي خلاصة
            combined_results = {
                 'symbol': symbol,
                 'timeframe': timeframe,
                 'technical_analysis': market_analysis_response['analysis'],
                 'price_prediction': prediction_data, # قد تكون None إذا فشل التوقع
                 'timestamp': datetime.now().isoformat()
            }

            # يمكنك إضافة منطق هنا لتوليد توصية شاملة أو درجة ثقة بناءً على التحليل الفني والتوقع
            # مثال بسيط: إذا كان التحليل الفني صاعداً والتوقع يظهر زيادة في السعر، فإن التوصية قوية للشراء
            recommendation = "HOLD"
            confidence_score = 0.5 # بين 0 و 1

            # مثال بسيط لمنطق التوصية (يحتاج تحسينات كبيرة)
            if market_analysis_response['analysis'].get('trend', {}).get('ma', {}).get('short_term') == 'bullish' and \
               market_analysis_response['analysis'].get('trend', {}).get('ma', {}).get('medium_term') == 'bullish':
                recommendation = "BUY"
                confidence_score = 0.6

            if prediction_data and prediction_data.get('price', 0) > market_analysis_response['analysis'].get('trend', {}).get('current_price', 0):
                 if recommendation == "BUY": # إذا كان التحليل الفني يدعم الشراء والتوقع أيضاً
                      recommendation = "STRONG_BUY"
                      confidence_score = min(1.0, confidence_score + 0.2) # زيادة الثقة

            # إضافة التوصية ودرجة الثقة إلى النتائج المدمجة
            combined_results['recommendation'] = recommendation
            combined_results['confidence_score'] = confidence_score


            # الدوال _analyze_trend, _analyze_strength, _analyze_volatility أدناه
            # تبدو وكأنها تحاول إعادة تحليل المؤشرات بناءً على التوقع، وهو منطق متضارب بعض الشيء
            # مع نتائج AnalysisService الأصلية.
            # يمكن إعادة استخدام هذه الدوال لدمج النتائج بدلاً من التحليل المنفصل
            # لكن الأفضل هو إعادة هيكلة جزء الدمج هنا ليكون أكثر وضوحاً.
            # حالياً، سأحتفظ بنتائج AnalysisService والتوقع وأضيف لهم الخلاصة البسيطة.

            # يمكنك إرجاع النتائج المدمجة بهذا الشكل
            return {
                'status': 'success',
                'analysis': combined_results # إرجاع النتائج المدمجة
            }

        except Exception as e:
            logger.error(f"Error in AI market analysis for {symbol}-{timeframe}: {str(e)}", exc_info=True)
            return {'status': 'error', 'message': f"AI Analysis failed: {str(e)}"}


    def _prepare_data(self, data: List[Dict]) -> tuple[np.ndarray, np.ndarray]: # إضافة إشارة لنوع الإرجاع
        """تحضير البيانات للتدريب"""
        try:
            # تحويل البيانات إلى DataFrame
            df = pd.DataFrame(data)

            # التحقق من الأعمدة المطلوبة وتحويلها إلى أرقام وتصفية NaN
            features = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in features):
                 raise ValueError(f"Missing required columns in data: {', '.join([c for c in features if c not in df.columns])}")

            for col in features:
                 df[col] = pd.to_numeric(df[col], errors='coerce')

            df.dropna(subset=features, inplace=True) # تصفية الصفوف التي تحتوي على قيم غير رقمية في الأعمدة الأساسية

            if df.empty:
                 raise ValueError("DataFrame is empty after dropping rows with missing feature data.")

            data_values = df[features].values

            # تطبيع البيانات - يجب أن يتم fit_transform هنا لبيانات التدريب
            # self.scaler.fit_transform(data_values) تمت إزالتها من هنا لأنها في train_model
            # يجب أن يتم fit_transform فقط في train_model. هنا نعتمد على scaler الذي تم تدريبه هناك.
            # إذا تم استدعاء _prepare_data بشكل منفصل عن train_model، قد تحدث مشكلة.
            # لنفترض أن _prepare_data يتم استدعاؤها فقط داخل train_model.
            scaled_data = self.scaler.fit_transform(data_values) # <-- التأكد من fit_transform هنا

            # تحضير X و y
            X, y = [], []
            sequence_length = 60 # طول السلسلة الزمنية لـ LSTM
            if len(scaled_data) < sequence_length:
                 # لا يوجد بيانات كافية لإنشاء حتى تسلسل واحد
                 return np.array(X), np.array(y) # إرجاع مصفوفات فارغة

            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i])
                y.append(scaled_data[i, 3])  # سعر الإغلاق (العمود رقم 3 في features)

            return np.array(X), np.array(y)
        except Exception as e:
            logger.error(f"Error preparing data for training: {str(e)}", exc_info=True)
            raise Exception(f"Error preparing data: {str(e)}")

    def _prepare_prediction_data(self, data: List[Dict]) -> np.ndarray:
        """تحضير البيانات للتنبؤ"""
        try:
            # تحويل البيانات إلى DataFrame
            df = pd.DataFrame(data)

            # التحقق من الأعمدة المطلوبة وتحويلها إلى أرقام وتصفية NaN
            features = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in features):
                 raise ValueError(f"Missing required columns in data: {', '.join([c for c in features if c not in df.columns])}")

            for col in features:
                 df[col] = pd.to_numeric(df[col], errors='coerce')

            df.dropna(subset=features, inplace=True) # تصفية الصفوف التي تحتوي على قيم غير رقمية في الأعمدة الأساسية

            if df.empty:
                 raise ValueError("DataFrame is empty after dropping rows with missing feature data.")


            data_values = df[features].values

            # تطبيع البيانات - يجب أن يتم transform فقط هنا باستخدام scaler الذي تم تدريبه
            if not hasattr(self.scaler, 'scale_'):
                 raise RuntimeError("Scaler has not been fitted. Train the model first.")

            scaled_data = self.scaler.transform(data_values)

            # تحضير X
            sequence_length = 60 # طول السلسلة الزمنية لـ LSTM
            # نحتاج آخر 'sequence_length' نقطة لعمل توقع واحد
            if len(scaled_data) < sequence_length:
                 raise ValueError(f"Not enough data points ({len(scaled_data)}) for prediction. Need {sequence_length}.")

            # خذ آخر 'sequence_length' نقطة بيانات
            X = np.array([scaled_data[-sequence_length:]])

            return X
        except Exception as e:
            logger.error(f"Error preparing data for prediction: {str(e)}", exc_info=True)
            raise Exception(f"Error preparing prediction data: {str(e)}")


    # تم حذف الدوال _analyze_trend, _analyze_strength, _analyze_volatility
    # لأن منطق دمج التحليل والتوقع تم نقله وتضمينه بشكل أبسط في analyze_market.
    # إذا كنت ترغب في استعادة منطق هذه الدوال لدمج أكثر تعقيداً، يجب إعادة بنائها
    # لتأخذ نتائج التحليل الفني والتوقع كمدخلات وتخرج نتيجة دمجية.

    # def _analyze_trend(self, market_analysis: Dict, prediction: Dict) -> Dict:
    #     """تحليل الاتجاه باستخدام الذكاء الاصطناعي"""
    #     ... تم حذفه ...

    # def _analyze_strength(self, market_analysis: Dict, prediction: Dict) -> Dict:
    #     """تحليل القوة باستخدام الذكاء الاصطناعي"""
    #     ... تم حذفه ...

    # def _analyze_volatility(self, market_analysis: Dict) -> Dict:
    #     """تحليل التقلب باستخدام الذكاء الاصطناعي"""
    #     ... تم حذفه ...

    # الدوال التالية كانت مذكورة في تعريف analyze_market الأول الذي تم حذفه.
    # إذا كنت تحتاج هذه الوظائف، يجب عليك إضافتها وتنفيذها بشكل منفصل
    # أو دمجها في analyze_market المتبقي.
    # def _analyze_technical(self, symbol, timeframe): ...
    # def _analyze_deep_learning(self, symbol, timeframe): ... (هذه قد تكون هي نفسها predict_price)
    # def _analyze_sentiment(self, symbol): ...
    # def _combine_analysis(self, technical_analysis, deep_learning_analysis, sentiment_analysis): ...