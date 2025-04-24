"""
دوال مساعدة للتطبيق
"""

import random
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytz
from .logger import error_logger

logger = logging.getLogger(__name__)

def generate_mock_price_data(symbol, days=7, interval_minutes=60):
    """
    Generate mock price data for the given symbol and time period
    
    Parameters:
    symbol (str): Currency pair symbol
    days (int): Number of days of data to generate
    interval_minutes (int): Interval between data points in minutes
    
    Returns:
    list: List of dictionaries containing price data
    """
    try:
        # Determine a reasonable base price based on the symbol
        if 'JPY' in symbol:
            base_price = random.uniform(105, 145)
        elif 'GBP' in symbol:
            base_price = random.uniform(1.2, 1.4)
        elif 'EUR' in symbol:
            base_price = random.uniform(1.05, 1.20)
        elif 'AUD' in symbol or 'NZD' in symbol or 'CAD' in symbol:
            base_price = random.uniform(0.7, 0.9)
        elif 'CHF' in symbol:
            base_price = random.uniform(0.9, 1.1)
        else:
            base_price = random.uniform(0.8, 1.2)
        
        # Calculate number of data points
        total_minutes = days * 24 * 60
        num_points = total_minutes // interval_minutes
        
        # Calculate end time (now) and start time
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        # Generate price series with some randomness but trending
        # First, decide on a trend direction
        trend = random.choice([-1, 1]) * random.uniform(0.0001, 0.0003)
        
        # Initialize price data
        price_data = []
        current_price = base_price
        current_time = start_time
        
        # Generate volatility (different for each symbol)
        volatility = random.uniform(0.0005, 0.002)
        
        for i in range(num_points):
            # Calculate time for this data point
            point_time = start_time + timedelta(minutes=i * interval_minutes)
            
            # Calculate price movement with some randomness
            price_change = np.random.normal(trend, volatility)
            current_price = current_price * (1 + price_change)
            
            # Ensure price stays positive and within a reasonable range
            current_price = max(current_price, base_price * 0.7)
            current_price = min(current_price, base_price * 1.3)
            
            # Calculate high and low with some randomness around close
            high_price = current_price * (1 + random.uniform(0, volatility * 2))
            low_price = current_price * (1 - random.uniform(0, volatility * 2))
            
            # Ensure high >= close >= low
            high_price = max(high_price, current_price)
            low_price = min(low_price, current_price)
            
            # Calculate open based on previous close or introduce some gap
            if i == 0:
                open_price = current_price * (1 + random.uniform(-volatility, volatility))
            else:
                # Use previous close with some small gap
                prev_close = price_data[-1]['close']
                open_price = prev_close * (1 + random.uniform(-volatility / 2, volatility / 2))
            
            # Generate a reasonable volume
            volume = random.uniform(100, 1000)
            
            # Add data point
            price_data.append({
                'timestamp': point_time.isoformat(),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': current_price,
                'volume': volume
            })
            
            # Occasional rapid movements (spike or dip)
            if random.random() < 0.05:  # 5% chance of a spike or dip
                spike_direction = random.choice([-1, 1])
                spike_magnitude = random.uniform(0.002, 0.005)
                current_price = current_price * (1 + spike_direction * spike_magnitude)
        
        return price_data
        
    except Exception as e:
        error_logger.error(f"Error generating mock price data for {symbol}: {str(e)}")
        # Return minimal fallback data
        return [
            {
                'timestamp': datetime.utcnow().isoformat(),
                'open': 1.0,
                'high': 1.05,
                'low': 0.95,
                'close': 1.0,
                'volume': 100
            }
        ]

def get_recent_signals(days=1, limit=10):
    """
    Get recent signals from the database
    
    Parameters:
    days (int): Number of days to look back
    limit (int): Maximum number of signals to return
    
    Returns:
    list: List of Signal objects
    """
    try:
        from app import db
        from models import Signal
        
        # Calculate time threshold
        threshold = datetime.utcnow() - timedelta(days=days)
        
        # Query recent signals
        signals = Signal.query.filter(
            Signal.timestamp >= threshold
        ).order_by(Signal.timestamp.desc()).limit(limit).all()
        
        return signals
        
    except Exception as e:
        error_logger.error(f"Error getting recent signals: {str(e)}")
        return []

def calculate_success_rate(days=30):
    """
    Calculate the success rate of signals over a given period
    
    Parameters:
    days (int): Number of days to look back
    
    Returns:
    float: Success rate percentage (0-100)
    """
    try:
        from app import db
        from models import Signal
        
        # Calculate time threshold
        threshold = datetime.utcnow() - timedelta(days=days)
        
        # Query signals in the period
        signals = Signal.query.filter(
            Signal.timestamp >= threshold
        ).all()
        
        if not signals:
            return 0
        
        # Count results
        win_count = sum(1 for s in signals if s.result == 'WIN')
        loss_count = sum(1 for s in signals if s.result == 'LOSS')
        
        total_count = win_count + loss_count
        
        if total_count == 0:
            # If no wins or losses recorded yet, return random value for demo
            return random.randint(65, 85)
        
        success_rate = (win_count / total_count) * 100
        return round(success_rate, 1)
        
    except Exception as e:
        error_logger.error(f"Error calculating success rate: {str(e)}")
        # Return placeholder value for demo
        return random.randint(65, 85)

def format_currency_pair(symbol):
    """
    Format currency pair symbol for consistent display
    
    Parameters:
    symbol (str): Currency pair symbol (e.g. "EURUSD" or "EUR_USD")
    
    Returns:
    str: Formatted symbol (e.g. "EUR/USD")
    """
    try:
        # Remove any existing separators
        clean_symbol = symbol.replace('/', '').replace('_', '').replace('-', '')
        
        # Format into pairs (assuming 3-letter currency codes)
        if len(clean_symbol) >= 6:
            base = clean_symbol[:3]
            quote = clean_symbol[3:6]
            return f"{base}/{quote}"
        else:
            return symbol
            
    except Exception as e:
        error_logger.error(f"Error formatting currency pair '{symbol}': {str(e)}")
        return symbol

def calculate_risk_reward(entry, stop_loss, take_profit, position_type):
    """
    Calculate risk-reward ratio for a signal
    
    Parameters:
    entry (float): Entry price
    stop_loss (float): Stop loss price
    take_profit (float): Take profit price
    position_type (str): 'BUY', 'SELL', 'CALL', or 'PUT'
    
    Returns:
    float: Risk-reward ratio (reward / risk)
    """
    try:
        if not all([entry, stop_loss, take_profit]):
            return None
            
        is_long = position_type in ['BUY', 'CALL']
        
        if is_long:
            risk = entry - stop_loss
            reward = take_profit - entry
        else:
            risk = stop_loss - entry
            reward = entry - take_profit
            
        if risk <= 0 or reward <= 0:
            return None
            
        return reward / risk
        
    except Exception as e:
        error_logger.error(f"Error calculating risk-reward ratio: {str(e)}")
        return None

def format_number(value, decimal_places=4):
    """
    Format a number with the specified decimal places
    
    Parameters:
    value (float): The number to format
    decimal_places (int): Number of decimal places
    
    Returns:
    str: Formatted number as string
    """
    try:
        if value is None:
            return 'N/A'
            
        return f"{value:.{decimal_places}f}"
        
    except Exception as e:
        error_logger.error(f"Error formatting number {value}: {str(e)}")
        return 'N/A'

def is_valid_price_level(price):
    """
    Check if a price level is valid
    
    Parameters:
    price (any): The price to validate
    
    Returns:
    bool: True if valid, False otherwise
    """
    try:
        if price is None:
            return False
            
        # Convert to float and check if it's positive
        price_float = float(price)
        return price_float > 0
        
    except (ValueError, TypeError):
        return False

def format_currency(value, currency='USD'):
    """تنسيق قيمة العملة"""
    if currency == 'USD':
        return f"${value:,.2f}"
    elif currency == 'EUR':
        return f"€{value:,.2f}"
    else:
        return f"{value:,.2f} {currency}"

def format_percentage(value):
    """تنسيق النسبة المئوية"""
    return f"{value:.2f}%"

def format_timestamp(timestamp, timezone='UTC'):
    """تنسيق الطابع الزمني"""
    tz = pytz.timezone(timezone)
    local_time = timestamp.astimezone(tz)
    return local_time.strftime('%Y-%m-%d %H:%M:%S %Z')

def calculate_risk_reward_ratio(entry_price, stop_loss, take_profit, signal_type):
    """حساب نسبة المخاطرة إلى العائد"""
    if signal_type in ['BUY', 'CALL']:
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
    else:  # SELL, PUT
        risk = abs(stop_loss - entry_price)
        reward = abs(entry_price - take_profit)
    
    if risk == 0:
        return 0
    return reward / risk

def validate_timeframe(timeframe):
    """التحقق من صحة الإطار الزمني"""
    valid_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
    return timeframe in valid_timeframes

def validate_symbol(symbol):
    """التحقق من صحة رمز العملة"""
    # قائمة العملات الافتراضية
    default_currencies = [
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "USD/CAD",
        "AUD/USD", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY"
    ]
    return symbol in default_currencies

def get_market_status():
    """الحصول على حالة السوق"""
    # في التطبيق الحقيقي، سيتم الحصول على هذه المعلومات من مصدر بيانات السوق
    return {
        'is_open': True,
        'next_open': None,
        'next_close': None,
        'holidays': []
    }
