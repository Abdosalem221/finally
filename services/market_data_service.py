"""
خدمة بيانات السوق
توفر واجهة للحصول على بيانات السوق من مصادر متعددة وتخزينها في قاعدة البيانات
تدعم 20 زوج عملة بشكل متزامن مع تحديثات في الوقت الفعلي
"""

import os
import threading
import time
import datetime
import logging
import json
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np

from models.database import db
from models.market_models import Currency, MarketData

# إعداد التسجيل
logger = logging.getLogger('market_data_service')

class MarketDataService:
    """خدمة بيانات السوق"""
    
    def __init__(self, app, update_interval=5):
        """
        تهيئة خدمة بيانات السوق
        
        Args:
            app: تطبيق Flask
            update_interval: الفاصل الزمني للتحديث بالثواني
        """
        self.app = app
        self.update_interval = update_interval
        self.running = False
        self.update_thread = None
        self.data_sources = {}
        self.data_cache = {}
        self.last_update = {}
        self.status = {
            'running': False,
            'last_update': None,
            'active_pairs': 0,
            'error': None
        }
        
        # تحميل مصادر البيانات
        self._load_data_sources()
    
    def start(self):
        """بدء خدمة بيانات السوق"""
        if self.running:
            logger.warning("Market data service already running")
            return
        
        logger.info("Starting market data service")
        self.running = True
        self.status['running'] = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def stop(self):
        """إيقاف خدمة بيانات السوق"""
        logger.info("Stopping market data service")
        self.running = False
        self.status['running'] = False
        if self.update_thread:
            self.update_thread.join(timeout=10)
    
    def get_service_status(self) -> Dict[str, str]:
        """
        الحصول على حالة الخدمة
        
        Returns:
            قاموس يحتوي على معلومات حالة الخدمة
        """
        return {
            'status': 'running' if self.status['running'] else 'stopped',
            'last_update': self.status['last_update'].isoformat() if self.status['last_update'] else None,
            'active_pairs': self.status['active_pairs'],
            'error': self.status['error']
        }
    
    def get_historical_data(self, symbol: str, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """
        الحصول على البيانات التاريخية لعملة وإطار زمني محددين
        
        Args:
            symbol: رمز العملة
            timeframe: الإطار الزمني
            limit: عدد السجلات المطلوبة
            
        Returns:
            إطار بيانات pandas يحتوي على البيانات، أو None في حالة عدم وجود بيانات
        """
        try:
            with self.app.app_context():
                # البحث عن العملة
                currency = Currency.query.filter_by(symbol=symbol).first()
                if not currency:
                    logger.warning(f"Currency not found: {symbol}")
                    return None
                
                # استرجاع البيانات
                data = MarketData.query.filter_by(
                    currency_id=currency.id,
                    timeframe=timeframe
                ).order_by(MarketData.timestamp.desc()).limit(limit).all()
                
                if not data:
                    logger.warning(f"No data found for {symbol} {timeframe}")
                    return None
                
                # إنشاء DataFrame
                df = pd.DataFrame([{
                    'timestamp': d.timestamp,
                    'open': d.open_price,
                    'high': d.high_price,
                    'low': d.low_price,
                    'close': d.close_price,
                    'volume': d.volume
                } for d in data])
                
                # ترتيب البيانات تصاعديًا حسب الوقت
                df = df.sort_values('timestamp')
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return None
    
    def get_last_price(self, symbol: str) -> Optional[float]:
        """
        الحصول على آخر سعر لعملة محددة
        
        Args:
            symbol: رمز العملة
            
        Returns:
            آخر سعر إغلاق، أو None في حالة عدم وجود بيانات
        """
        try:
            with self.app.app_context():
                # البحث عن العملة
                currency = Currency.query.filter_by(symbol=symbol).first()
                if not currency:
                    logger.warning(f"Currency not found: {symbol}")
                    return None
                
                # استرجاع آخر سعر
                last_data = MarketData.query.filter_by(
                    currency_id=currency.id
                ).order_by(MarketData.timestamp.desc()).first()
                
                if not last_data:
                    logger.warning(f"No data found for {symbol}")
                    return None
                
                return last_data.close_price
                
        except Exception as e:
            logger.error(f"Error getting last price: {str(e)}")
            return None
    
    def get_active_currencies(self) -> List[str]:
        """
        الحصول على قائمة العملات النشطة
        
        Returns:
            قائمة برموز العملات النشطة
        """
        try:
            with self.app.app_context():
                currencies = Currency.query.filter_by(is_active=True).all()
                return [c.symbol for c in currencies]
        except Exception as e:
            logger.error(f"Error getting active currencies: {str(e)}")
            return []
    
    def _update_loop(self):
        """حلقة تحديث البيانات"""
        logger.info("Update loop started")
        
        while self.running:
            try:
                with self.app.app_context():
                    self._update_market_data()
                    self.status['last_update'] = datetime.datetime.now()
                    self.status['error'] = None
            except Exception as e:
                logger.error(f"Error in update loop: {str(e)}")
                self.status['error'] = str(e)
            
            # النوم حتى التحديث التالي
            time.sleep(self.update_interval)
    
    def _update_market_data(self):
        """تحديث بيانات السوق"""
        # الحصول على العملات النشطة
        currencies = Currency.query.filter_by(is_active=True).all()
        self.status['active_pairs'] = len(currencies)
        
        if not currencies:
            logger.warning("No active currencies found")
            return
        
        for currency in currencies:
            try:
                # الحصول على البيانات من مصدر البيانات المناسب
                data = self._fetch_data_for_currency(currency.symbol, ['1m', '5m', '15m', '30m', '1h', '4h', '1d'])
                
                if not data:
                    continue
                
                # حفظ البيانات في قاعدة البيانات
                for timeframe, ohlcv_data in data.items():
                    self._save_market_data(currency.id, timeframe, ohlcv_data)
                
                # تحديث ذاكرة التخزين المؤقت
                self.data_cache[currency.symbol] = data
                self.last_update[currency.symbol] = datetime.datetime.now()
                
            except Exception as e:
                logger.error(f"Error updating market data for {currency.symbol}: {str(e)}")
    
    def _fetch_data_for_currency(self, symbol: str, timeframes: List[str]) -> Dict[str, List[Dict]]:
        """
        الحصول على بيانات لعملة معينة من مصادر البيانات
        
        Args:
            symbol: رمز العملة
            timeframes: قائمة الأطر الزمنية المطلوبة
            
        Returns:
            قاموس يحتوي على بيانات لكل إطار زمني
        """
        result = {}
        
        # في البيئة الحقيقية، سيتم الاتصال بواجهات برمجة التطبيقات الخارجية
        # للحصول على بيانات السوق، ولكن هنا نستخدم بيانات عشوائية للعرض
        
        # TODO: استبدال هذا برمز الاتصال بواجهات برمجة التطبيقات الحقيقية
        
        # استخدام البيانات المخزنة مسبقًا إن وجدت
        if symbol in self.data_cache:
            cached_data = self.data_cache[symbol]
            for timeframe in timeframes:
                if timeframe in cached_data:
                    # تحديث آخر شمعة فقط
                    result[timeframe] = cached_data[timeframe]
        
        # إذا لم تكن هناك بيانات في ذاكرة التخزين المؤقت، قم بإنشاء بعض البيانات
        for timeframe in timeframes:
            if timeframe not in result:
                # TODO: الاتصال بمصدر البيانات الحقيقي
                # في هذا المثال، نستخدم وظيفة _get_sample_data
                result[timeframe] = self._get_sample_data(symbol, timeframe)
        
        return result
    
    def _save_market_data(self, currency_id: int, timeframe: str, data: List[Dict]):
        """
        حفظ بيانات السوق في قاعدة البيانات
        
        Args:
            currency_id: معرف العملة
            timeframe: الإطار الزمني
            data: بيانات OHLCV
        """
        for item in data:
            # التحقق من وجود السجل
            existing = MarketData.query.filter_by(
                currency_id=currency_id,
                timeframe=timeframe,
                timestamp=item['timestamp']
            ).first()
            
            if existing:
                # تحديث السجل الموجود
                existing.open_price = item['open']
                existing.high_price = item['high']
                existing.low_price = item['low']
                existing.close_price = item['close']
                existing.volume = item['volume']
            else:
                # إنشاء سجل جديد
                market_data = MarketData(
                    currency_id=currency_id,
                    timeframe=timeframe,
                    timestamp=item['timestamp'],
                    open_price=item['open'],
                    high_price=item['high'],
                    low_price=item['low'],
                    close_price=item['close'],
                    volume=item['volume']
                )
                db.session.add(market_data)
            
        db.session.commit()
    
    def _load_data_sources(self):
        """تحميل مصادر البيانات المتاحة"""
        # في البيئة الحقيقية، سيتم تحميل إعدادات مصادر البيانات من الملفات أو قاعدة البيانات
        
        # TODO: تحميل إعدادات مصادر البيانات الحقيقية
        
        # أمثلة على مصادر البيانات
        self.data_sources = {
            'alphavantage': {
                'api_key': os.environ.get('ALPHAVANTAGE_API_KEY'),
                'base_url': 'https://www.alphavantage.co/query',
                'timeframes': ['1m', '5m', '15m', '30m', '1h', '1d'],
                'symbols': ['EUR/USD', 'GBP/USD', 'USD/JPY']
            },
            'fcsapi': {
                'api_key': os.environ.get('FCSAPI_API_KEY'),
                'base_url': 'https://fcsapi.com/api-v3',
                'timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
                'symbols': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'USD/CAD', 'AUD/USD']
            },
            'twelve_data': {
                'api_key': os.environ.get('TWELVEDATA_API_KEY'),
                'base_url': 'https://api.twelvedata.com',
                'timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
                'symbols': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'USD/CAD', 'AUD/USD', 'NZD/USD']
            }
        }
    
    def _get_sample_data(self, symbol: str, timeframe: str, num_candles: int = 100) -> List[Dict]:
        """
        إنشاء بيانات عينة للعرض
        
        Args:
            symbol: رمز العملة
            timeframe: الإطار الزمني
            num_candles: عدد الشموع المطلوبة
            
        Returns:
            قائمة من بيانات OHLCV
        """
        now = datetime.datetime.now()
        
        # تحديد الفاصل الزمني بالثواني
        if timeframe == '1m':
            interval = 60  # ثانية
        elif timeframe == '5m':
            interval = 5 * 60
        elif timeframe == '15m':
            interval = 15 * 60
        elif timeframe == '30m':
            interval = 30 * 60
        elif timeframe == '1h':
            interval = 60 * 60
        elif timeframe == '4h':
            interval = 4 * 60 * 60
        elif timeframe == '1d':
            interval = 24 * 60 * 60
        else:
            interval = 60 * 60  # افتراضي: ساعة واحدة
        
        # تحديد سعر البداية
        if 'EUR/USD' in symbol:
            base_price = 1.08
        elif 'GBP/USD' in symbol:
            base_price = 1.25
        elif 'USD/JPY' in symbol:
            base_price = 150.0
        elif 'USD/CHF' in symbol:
            base_price = 0.90
        elif 'USD/CAD' in symbol:
            base_price = 1.35
        elif 'AUD/USD' in symbol:
            base_price = 0.65
        elif 'NZD/USD' in symbol:
            base_price = 0.60
        else:
            base_price = 1.0
        
        result = []
        
        for i in range(num_candles):
            # حساب الوقت
            timestamp = now - datetime.timedelta(seconds=interval * (num_candles - i - 1))
            
            # إنشاء بيانات عشوائية مبنية على السعر الأساسي
            price_variation = base_price * 0.005  # 0.5% تغيير
            close = base_price + (np.random.random() - 0.5) * price_variation
            open_price = close + (np.random.random() - 0.5) * price_variation
            high = max(open_price, close) + np.random.random() * price_variation * 0.5
            low = min(open_price, close) - np.random.random() * price_variation * 0.5
            volume = np.random.randint(1000, 10000)
            
            result.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
            
            # تحديث السعر الأساسي للشمعة التالية
            base_price = close
        
        return result


def init_market_data_service(app) -> MarketDataService:
    """
    تهيئة خدمة بيانات السوق
    
    Args:
        app: تطبيق Flask
        
    Returns:
        خدمة بيانات السوق
    """
    # الحصول على الفاصل الزمني من الإعدادات
    from config import SYSTEM_SETTINGS
    update_interval = SYSTEM_SETTINGS.get('update_interval', 5)
    
    # إنشاء خدمة بيانات السوق
    service = MarketDataService(app, update_interval=update_interval)
    
    # بدء الخدمة
    service.start()
    
    logger.info(f"Market data service initialized with update interval {update_interval}s")
    
    return service