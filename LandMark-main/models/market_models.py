"""
نماذج البيانات للسوق المالي
"""

from datetime import datetime
from sqlalchemy.sql import func
from .database import db

class Currency(db.Model):
    """Currency pair model"""
    __tablename__ = 'currencies'
    
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), unique=True, nullable=False)
    name = db.Column(db.String(50))
    type = db.Column(db.String(20))  # forex, crypto, etc.
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    signals = db.relationship('Signal', backref='currency', lazy=True)
    alerts = db.relationship('Alert', backref='currency', lazy=True)
    
    def __repr__(self):
        return f'<Currency {self.symbol}>'


class MarketData(db.Model):
    """نموذج بيانات السوق"""
    __tablename__ = 'market_data'
    
    id = db.Column(db.Integer, primary_key=True)
    currency_id = db.Column(db.Integer, db.ForeignKey('currencies.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    timeframe = db.Column(db.String(10), nullable=False)  # 1m, 5m, 15m, 30m, 1h, 4h, 1d
    open_price = db.Column(db.Float, nullable=False)
    high_price = db.Column(db.Float, nullable=False)
    low_price = db.Column(db.Float, nullable=False)
    close_price = db.Column(db.Float, nullable=False)
    volume = db.Column(db.Float, default=0)
    created_at = db.Column(db.DateTime, default=func.now())
    
    # مؤشر فريد مركب
    __table_args__ = (
        db.UniqueConstraint('currency_id', 'timestamp', 'timeframe', name='uix_market_data_currency_timestamp_timeframe'),
    )
    
    def __repr__(self):
        return f'<MarketData {self.currency.symbol if self.currency else "Unknown"} {self.timestamp} {self.timeframe}>'


class Signal(db.Model):
    """Trading signal model"""
    __tablename__ = 'signals'
    
    id = db.Column(db.Integer, primary_key=True)
    currency_id = db.Column(db.Integer, db.ForeignKey('currencies.id'), nullable=False)
    signal_type = db.Column(db.String(10), nullable=False)  # BUY, SELL
    entry_price = db.Column(db.Float, nullable=False)
    stop_loss = db.Column(db.Float)
    take_profit = db.Column(db.Float)
    timeframe = db.Column(db.String(10), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    strategy = db.Column(db.String(50))
    success_probability = db.Column(db.Float)
    is_high_precision = db.Column(db.Boolean, default=False)
    result = db.Column(db.String(10))  # WIN, LOSS, PENDING
    
    # Relationships
    verifications = db.relationship('SignalVerification', backref='signal', lazy=True)
    
    def __repr__(self):
        return f'<Signal {self.id} - {self.currency.symbol} - {self.signal_type}>'


class Alert(db.Model):
    """User alert model"""
    __tablename__ = 'alerts'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    currency_id = db.Column(db.Integer, db.ForeignKey('currencies.id'), nullable=False)
    alert_type = db.Column(db.String(20), nullable=False)  # PRICE, SIGNAL, etc.
    condition = db.Column(db.String(50), nullable=False)
    value = db.Column(db.Float, nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    is_triggered = db.Column(db.Boolean, default=False)
    triggered_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Alert {self.id} - {self.user.username} - {self.currency.symbol}>'


class TradingStrategy(db.Model):
    """نموذج استراتيجية التداول"""
    __tablename__ = 'trading_strategies'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text)
    category = db.Column(db.String(50))  # TREND, MOMENTUM, BREAKOUT, REVERSAL, VOLATILITY
    parameters = db.Column(db.JSON)
    success_rate = db.Column(db.Float)
    preferred_timeframes = db.Column(db.JSON)  # ['1h', '4h', '1d']
    preferred_market_types = db.Column(db.JSON)  # ['forex', 'crypto']
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=func.now())
    updated_at = db.Column(db.DateTime, default=func.now(), onupdate=func.now())
    performance_data = db.Column(db.JSON)
    
    def __repr__(self):
        return f'<TradingStrategy {self.name}>'


class SignalVerification(db.Model):
    """Signal verification model"""
    __tablename__ = 'signal_verifications'
    
    id = db.Column(db.Integer, primary_key=True)
    signal_id = db.Column(db.Integer, db.ForeignKey('signals.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    status = db.Column(db.String(20), nullable=False)  # VERIFIED, REJECTED, PENDING
    comment = db.Column(db.Text)
    verified_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<SignalVerification {self.id} - {self.signal.id} - {self.status}>'


class PerformanceMetric(db.Model):
    """نموذج مقياس الأداء"""
    __tablename__ = 'performance_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    metric_type = db.Column(db.String(50), nullable=False)  # STRATEGY, SIGNAL, ENHANCEMENT, AI_MODEL
    name = db.Column(db.String(100), nullable=False)
    value = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=func.now())
    timeframe = db.Column(db.String(10))
    currency_id = db.Column(db.Integer, db.ForeignKey('currencies.id'))
    metric_metadata = db.Column(db.JSON)
    
    # العلاقات
    currency = db.relationship('Currency', backref='performance_metrics', lazy=True)
    
    def __repr__(self):
        return f'<PerformanceMetric {self.metric_type} {self.name} {self.value}>'