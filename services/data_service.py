"""
خدمة جلب وتحليل بيانات السوق
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import requests
import logging
import os

from app.utils.validators import validate_symbol, validate_timeframe
from app.utils.helpers import format_timestamp


class DataService:
            def __init__(self):
                self.logger = logging.getLogger(__name__)
                self.base_url = os.getenv('MARKET_DATA_API_URL', "https://api.binance.com/api/v3")
                self.api_key = os.getenv('MARKET_DATA_API_KEY')
                self.cache = {}
                self.cache_timeout = 300  # 5 minutes

                if not self.api_key:
                    self.logger.warning("API key not configured")

            def fetch_market_data(self, symbol: str, timeframe: str, limit: int = 100) -> Dict:
                """
                جلب بيانات السوق

                Args:
                    symbol: رمز العملة
                    timeframe: الإطار الزمني
                    limit: عدد البيانات المطلوبة

                Returns:
                    Dict: بيانات السوق
                """
                try:
                    # التحقق من صحة المدخلات
                    if not validate_symbol(symbol):
                        return {'status': 'error', 'message': 'رمز العملة غير صالح'}

                    if not validate_timeframe(timeframe):
                        return {'status': 'error', 'message': 'الإطار الزمني غير صالح'}

                    # التحقق من وجود البيانات في الكاش
                    cache_key = f"{symbol}_{timeframe}_{limit}"
                    if cache_key in self.cache:
                        cached_data, timestamp = self.cache[cache_key]
                        if (datetime.now() - timestamp).total_seconds() < self.cache_timeout:
                            return {'status': 'success', 'data': cached_data}

                    # جلب البيانات من API
                    interval = self._convert_timeframe(timeframe)
                    url = f"{self.base_url}/klines"
                    params = {
                        'symbol': symbol,
                        'interval': interval,
                        'limit': limit
                    }

                    response = requests.get(url, params=params)
                    if response.status_code != 200:
                        return {'status': 'error', 'message': 'فشل في جلب البيانات'}

                    # تحويل البيانات إلى DataFrame
                    data = response.json()
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                        'taker_buy_quote', 'ignore'
                    ])

                    # تحويل الأنواع
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)

                    # تخزين في الكاش
                    self.cache[cache_key] = (df.to_dict('records'), datetime.now())

                    return {'status': 'success', 'data': df.to_dict('records')}

                except Exception as e:
                    self.logger.error(f"Error fetching market data: {str(e)}")
                    return {'status': 'error', 'message': str(e)}

            def fetch_order_book(self, symbol: str, limit: int = 100) -> Dict:
                """
                جلب دفتر الطلبات

                Args:
                    symbol: رمز العملة
                    limit: عدد الطلبات المطلوبة

                Returns:
                    Dict: دفتر الطلبات
                """
                try:
                    url = f"{self.base_url}/depth"
                    params = {
                        'symbol': symbol,
                        'limit': limit
                    }

                    response = requests.get(url, params=params)
                    if response.status_code != 200:
                        return {'status': 'error', 'message': 'فشل في جلب دفتر الطلبات'}

                    data = response.json()
                    return {
                        'status': 'success',
                        'data': {
                            'bids': [[float(price), float(qty)] for price, qty in data['bids']],
                            'asks': [[float(price), float(qty)] for price, qty in data['asks']],
                            'last_update_id': data['lastUpdateId']
                        }
                    }

                except Exception as e:
                    self.logger.error(f"Error fetching order book: {str(e)}")
                    return {'status': 'error', 'message': str(e)}

            def fetch_trades(self, symbol: str, limit: int = 100) -> Dict:
                """
                جلب آخر الصفقات

                Args:
                    symbol: رمز العملة
                    limit: عدد الصفقات المطلوبة

                Returns:
                    Dict: آخر الصفقات
                """
                try:
                    url = f"{self.base_url}/trades"
                    params = {
                        'symbol': symbol,
                        'limit': limit
                    }

                    response = requests.get(url, params=params)
                    if response.status_code != 200:
                        return {'status': 'error', 'message': 'فشل في جلب الصفقات'}

                    data = response.json()
                    trades = []
                    for trade in data:
                        trades.append({
                            'id': trade['id'],
                            'price': float(trade['price']),
                            'qty': float(trade['qty']),
                            'time': format_timestamp(trade['time']),
                            'is_buyer_maker': trade['isBuyerMaker']
                        })

                    return {'status': 'success', 'data': trades}

                except Exception as e:
                    self.logger.error(f"Error fetching trades: {str(e)}")
                    return {'status': 'error', 'message': str(e)}

            def fetch_24h_stats(self, symbol: str) -> Dict:
                """
                جلب إحصائيات 24 ساعة

                Args:
                    symbol: رمز العملة

                Returns:
                    Dict: إحصائيات 24 ساعة
                """
                try:
                    url = f"{self.base_url}/ticker/24hr"
                    params = {'symbol': symbol}

                    response = requests.get(url, params=params)
                    if response.status_code != 200:
                        return {'status': 'error', 'message': 'فشل في جلب الإحصائيات'}

                    data = response.json()
                    return {
                        'status': 'success',
                        'data': {
                            'price_change': float(data['priceChange']),
                            'price_change_percent': float(data['priceChangePercent']),
                            'weighted_avg_price': float(data['weightedAvgPrice']),
                            'prev_close_price': float(data['prevClosePrice']),
                            'last_price': float(data['lastPrice']),
                            'bid_price': float(data['bidPrice']),
                            'ask_price': float(data['askPrice']),
                            'open_price': float(data['openPrice']),
                            'high_price': float(data['highPrice']),
                            'low_price': float(data['lowPrice']),
                            'volume': float(data['volume']),
                            'quote_volume': float(data['quoteVolume']),
                            'open_time': format_timestamp(data['openTime']),
                            'close_time': format_timestamp(data['closeTime']),
                            'first_id': data['firstId'],
                            'last_id': data['lastId'],
                            'count': data['count']
                        }
                    }

                except Exception as e:
                    self.logger.error(f"Error fetching 24h stats: {str(e)}")
                    return {'status': 'error', 'message': str(e)}

            def _convert_timeframe(self, timeframe: str) -> str:
                """
                تحويل الإطار الزمني إلى تنسيق Binance

                Args:
                    timeframe: الإطار الزمني

                Returns:
                    str: تنسيق الإطار الزمني
                """
                timeframe_map = {
                    '1m': '1m',
                    '5m': '5m',
                    '15m': '15m',
                    '30m': '30m',
                    '1h': '1h',
                    '4h': '4h',
                    '1d': '1d',
                    '1w': '1w',
                    '1M': '1M'
                }
                return timeframe_map.get(timeframe, '1h')

            def clear_cache(self):
                """مسح الكاش"""
                self.cache.clear()