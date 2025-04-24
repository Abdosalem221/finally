"""
نماذج قاعدة البيانات
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Enum, ForeignKey
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
import enum

Base = declarative_base()

class UserRole(str, enum.Enum):
    """
    أدوار المستخدمين
    """
    ADMIN = "admin"
    TRADER = "trader"
    USER = "user"

class UserStatus(str, enum.Enum):
    """
    حالات المستخدمين
    """
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"

class User(Base):
    """
    نموذج المستخدم
    """
    __tablename__ = "users"

    user_id = Column(String, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String)
    role = Column(Enum(UserRole), default=UserRole.USER)
    status = Column(Enum(UserStatus), default=UserStatus.ACTIVE)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # العلاقات
    portfolios = relationship("Portfolio", back_populates="user")
    strategies = relationship("Strategy", back_populates="user")
    alerts = relationship("Alert", back_populates="user")
    notifications = relationship("Notification", back_populates="user")

class Strategy(Base):
    """
    نموذج الاستراتيجية
    """
    __tablename__ = "strategies"

    strategy_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"))
    name = Column(String, nullable=False)
    description = Column(String)
    parameters = Column(String)  # JSON string
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # العلاقات
    user = relationship("User", back_populates="strategies")
    positions = relationship("Position", back_populates="strategy")

class Portfolio(Base):
    """
    نموذج المحفظة
    """
    __tablename__ = "portfolios"

    portfolio_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"))
    name = Column(String, nullable=False)
    description = Column(String)
    balance = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # العلاقات
    user = relationship("User", back_populates="portfolios")
    positions = relationship("Position", back_populates="portfolio")

class Position(Base):
    """
    نموذج المركز
    """
    __tablename__ = "positions"

    position_id = Column(String, primary_key=True)
    portfolio_id = Column(String, ForeignKey("portfolios.portfolio_id"))
    strategy_id = Column(String, ForeignKey("strategies.strategy_id"))
    symbol = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float)
    is_open = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # العلاقات
    portfolio = relationship("Portfolio", back_populates="positions")
    strategy = relationship("Strategy", back_populates="positions")
    trades = relationship("Trade", back_populates="position")

class Trade(Base):
    """
    نموذج الصفقة
    """
    __tablename__ = "trades"

    trade_id = Column(String, primary_key=True)
    position_id = Column(String, ForeignKey("positions.position_id"))
    type = Column(String, nullable=False)  # buy/sell
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # العلاقات
    position = relationship("Position", back_populates="trades")

class Alert(Base):
    """
    نموذج التنبيه
    """
    __tablename__ = "alerts"

    alert_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"))
    symbol = Column(String, nullable=False)
    condition = Column(String, nullable=False)  # price > X, price < X, etc.
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # العلاقات
    user = relationship("User", back_populates="alerts")

class Notification(Base):
    """
    نموذج الإشعار
    """
    __tablename__ = "notifications"

    notification_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"))
    title = Column(String, nullable=False)
    message = Column(String, nullable=False)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # العلاقات
    user = relationship("User", back_populates="notifications")

class MarketData(Base):
    """
    نموذج بيانات السوق
    """
    __tablename__ = "market_data"

    data_id = Column(String, primary_key=True)
    symbol = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class MarketIndicator(Base):
    """
    نموذج مؤشر السوق
    """
    __tablename__ = "market_indicators"

    indicator_id = Column(String, primary_key=True)
    symbol = Column(String, nullable=False)
    name = Column(String, nullable=False)  # SMA, RSI, etc.
    parameters = Column(String)  # JSON string
    value = Column(Float)
    timestamp = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)