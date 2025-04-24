"""
دوال التحقق من صحة البيانات
"""

import re
from datetime import datetime
from typing import Dict, Any, Optional


def validate_email(email: str) -> bool:
    """التحقق من صحة البريد الإلكتروني"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_password(password: str) -> bool:
    """التحقق من قوة كلمة المرور"""
    if len(password) < 8:
        return False
    if not re.search(r'[A-Z]', password):
        return False
    if not re.search(r'[a-z]', password):
        return False
    if not re.search(r'\d', password):
        return False
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False
    return True


def validate_username(username: str) -> bool:
    """التحقق من صحة اسم المستخدم"""
    pattern = r'^[a-zA-Z0-9_]{3,20}$'
    return bool(re.match(pattern, username))


# --- الدوال المفقودة التي تسببت في الخطأ ---


def validate_symbol(symbol: str) -> bool:
    """
    التحقق من صحة رمز الأصل المالي (مثال: BTCUSDT, EURUSD, AAPL).
    يمكن تعديل هذه الدالة لتناسب الرموز المدعومة لديك.
    """
    # مثال لنمط بسيط للرموز: حروف كبيرة وأرقام ورموز مثل / . -
    pattern = r'^[A-Z0-9/.-]+$'
    return bool(re.match(pattern,
                         symbol.upper()))  # تحويل الرمز إلى حروف كبيرة للتحقق


def validate_timeframe(timeframe: str) -> bool:
    """
    التحقق من صحة الإطار الزمني (مثال: 1m, 5m, 1h, 1D).
    يمكن تعديل هذه القائمة لتناسب الأطر الزمنية المدعومة لديك.
    """
    allowed_timeframes = {
        '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M'
    }
    return timeframe in allowed_timeframes


# --- بقية الدوال الموجودة في الكود الأصلي ---


def validate_signal_data(data: Dict[str, Any]) -> Optional[str]:
    """التحقق من صحة بيانات الإشارة"""
    required_fields = [
        'symbol', 'timeframe', 'signal_type', 'entry_price', 'stop_loss',
        'take_profit'
    ]

    # التحقق من الحقول المطلوبة
    for field in required_fields:
        if field not in data:
            return f"Missing required field: {field}"

    # التحقق من صحة الرموز
    if not validate_symbol(data['symbol']):
        return "Invalid symbol"

    # التحقق من صحة الإطار الزمني
    if not validate_timeframe(data['timeframe']):
        return "Invalid timeframe"

    # التحقق من صحة نوع الإشارة
    if data['signal_type'] not in ['BUY', 'SELL', 'CALL', 'PUT']:
        return "Invalid signal type"

    # التحقق من صحة الأسعار
    try:
        entry_price = float(data['entry_price'])
        stop_loss = float(data['stop_loss'])
        take_profit = float(data['take_profit'])

        if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
            return "Prices must be positive"

        if data['signal_type'] in ['BUY', 'CALL']:
            if stop_loss >= entry_price or take_profit <= entry_price:
                return "Invalid price levels for BUY/CALL signal"
        else:  # SELL, PUT
            if stop_loss <= entry_price or take_profit >= entry_price:
                return "Invalid price levels for SELL/PUT signal"
    except ValueError:
        return "Invalid price format"
    except TypeError:  # في حال كانت القيمة ليست قابلة للتحويل إلى float
        return "Invalid price format or type"

    return None


def validate_alert_data(data: Dict[str, Any]) -> Optional[str]:
    """التحقق من صحة بيانات التنبيه"""
    required_fields = ['symbol', 'alert_type', 'price_level']

    # التحقق من الحقول المطلوبة
    for field in required_fields:
        if field not in data:
            return f"Missing required field: {field}"

    # التحقق من صحة الرموز
    if not validate_symbol(data['symbol']):
        return "Invalid symbol"

    # التحقق من صحة نوع التنبيه
    if data['alert_type'] not in ['PRICE_ABOVE', 'PRICE_BELOW', 'PRICE_CROSS']:
        return "Invalid alert type"

    # التحقق من صحة مستوى السعر
    try:
        price_level = float(data['price_level'])
        if price_level <= 0:
            return "Price level must be positive"
    except ValueError:
        return "Invalid price level format"
    except TypeError:  # في حال كانت القيمة ليست قابلة للتحويل إلى float
        return "Invalid price level format or type"

    return None


def validate_market_data(data: Dict[str, Any]) -> Optional[str]:
    """التحقق من صحة بيانات السوق"""
    required_fields = [
        'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume'
    ]

    # التحقق من الحقول المطلوبة
    for field in required_fields:
        if field not in data:
            return f"Missing required field: {field}"

    # التحقق من صحة الرموز
    if not validate_symbol(data['symbol']):
        return "Invalid symbol"

    # التحقق من صحة الطابع الزمني
    try:
        # محاولة تحليل الطابع الزمني، يسمح بأشكال مختلفة مثل ISO 8601
        datetime.fromisoformat(data['timestamp'].replace(
            'Z', '+00:00'))  # استبدال Z بتوقيت UTC إذا كانت موجودة
    except (ValueError, AttributeError
            ):  # إضافة AttributeError للتعامل مع إذا لم تكن القيمة سترينج
        return "Invalid timestamp format"

    # التحقق من صحة الأسعار والحجم
    try:
        open_price = float(data['open'])
        high_price = float(data['high'])
        low_price = float(data['low'])
        close_price = float(data['close'])
        volume = float(data['volume'])

        if any(price is None or price <= 0
               for price in [open_price, high_price, low_price, close_price]):
            # Added None check as float(None) throws TypeError, not ValueError
            return "Prices must be positive"

        if volume is None or volume < 0:
            # Added None check
            return "Volume must be non-negative"

        if not (low_price <= open_price <= high_price
                and low_price <= close_price <= high_price):
            return "Invalid price range"
    except (ValueError, TypeError):  # Catch both ValueError and TypeError
        return "Invalid price or volume format or type"

    return None


def validate_technical_indicators(data: Dict[str, Any]) -> Optional[str]:
    """التحقق من صحة المؤشرات الفنية"""
    required_fields = ['symbol', 'timeframe', 'timestamp']

    # التحقق من الحقول المطلوبة
    for field in required_fields:
        if field not in data:
            return f"Missing required field: {field}"

    # التحقق من صحة الرموز
    if not validate_symbol(data['symbol']):
        return "Invalid symbol"

    # التحقق من صحة الإطار الزمني
    if not validate_timeframe(data['timeframe']):
        return "Invalid timeframe"

    # التحقق من صحة الطابع الزمني
    try:
        # محاولة تحليل الطابع الزمني
        datetime.fromisoformat(data['timestamp'].replace(
            'Z', '+00:00'))  # استبدال Z بتوقيت UTC إذا كانت موجودة
    except (ValueError, AttributeError
            ):  # إضافة AttributeError للتعامل مع إذا لم تكن القيمة سترينج
        return "Invalid timestamp format"

    # التحقق من صحة المؤشرات
    indicators = ['rsi', 'macd', 'bollinger_bands', 'moving_averages']
    for indicator in indicators:
        if indicator in data:
            value = data[indicator]
            try:
                if indicator == 'rsi':
                    rsi = float(value)
                    if not (0 <= rsi <= 100):
                        return "RSI must be between 0 and 100"
                elif indicator == 'macd':
                    # Check if it's a dict and contains required keys with numeric values
                    if not (isinstance(value, dict) and all(
                            isinstance(value.get(k), (int, float))
                            for k in ['macd', 'signal', 'histogram'])):
                        return "Invalid MACD values or format"
                elif indicator == 'bollinger_bands':
                    # Check if it's a dict and contains required keys with numeric values
                    if not (isinstance(value, dict) and all(
                            isinstance(value.get(k), (int, float))
                            for k in ['upper', 'middle', 'lower'])):
                        return "Invalid Bollinger Bands values or format"
                elif indicator == 'moving_averages':
                    # Check if it's a dict and all values are numeric
                    if not (isinstance(value, dict) and all(
                            isinstance(v, (int, float))
                            for v in value.values())):
                        return "Invalid Moving Averages values or format"
            except (ValueError, TypeError,
                    AttributeError):  # Catch relevant errors
                return f"Invalid {indicator} format or type"

    return None
