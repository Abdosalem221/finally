"""
إدارة البيانات الأولية لقاعدة البيانات
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
from app.database.db_session import db_manager
from app.database.models import (
    User, UserStatus, UserRole,
    Strategy, StrategyType,
    Portfolio, Position, PositionType, PositionStatus,
    Trade, TradeType,
    Alert, AlertPriority,
    Notification, NotificationType, NotificationStatus,
    MarketData, MarketIndicator
)

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SeedManager:
    def __init__(self):
        """
        تهيئة مدير البيانات الأولية
        """
        self.db = db_manager
    
    def seed_all(self) -> None:
        """
        إضافة جميع البيانات الأولية
        """
        try:
            self.seed_users()
            self.seed_strategies()
            self.seed_portfolios()
            self.seed_positions()
            self.seed_trades()
            self.seed_alerts()
            self.seed_notifications()
            self.seed_market_data()
            self.seed_market_indicators()
            logger.info("تم إضافة جميع البيانات الأولية بنجاح")
        except Exception as e:
            logger.error(f"خطأ في إضافة البيانات الأولية: {str(e)}")
            raise
    
    def seed_users(self) -> None:
        """
        إضافة مستخدمين أوليين
        """
        users = [
            {
                "user_id": str(uuid.uuid4()),
                "username": "admin",
                "email": "admin@algotrader.com",
                "password_hash": "hashed_password",  # يجب تشفير كلمة المرور
                "full_name": "مدير النظام",
                "role": UserRole.ADMIN,
                "status": UserStatus.ACTIVE,
                "created_at": datetime.utcnow()
            },
            {
                "user_id": str(uuid.uuid4()),
                "username": "trader",
                "email": "trader@algotrader.com",
                "password_hash": "hashed_password",
                "full_name": "المتداول",
                "role": UserRole.TRADER,
                "status": UserStatus.ACTIVE,
                "created_at": datetime.utcnow()
            }
        ]
        self.db.bulk_insert(User, users)
    
    def seed_strategies(self) -> None:
        """
        إضافة استراتيجيات أولية
        """
        strategies = [
            {
                "strategy_id": str(uuid.uuid4()),
                "user_id": "admin",  # يجب تحديث هذا بالمعرف الفعلي للمستخدم
                "name": "استراتيجية المتوسط المتحرك",
                "description": "استراتيجية تعتمد على تقاطع المتوسطات المتحركة",
                "parameters": '{"fast_ma": 20, "slow_ma": 50}',
                "is_active": True,
                "created_at": datetime.utcnow()
            }
        ]
        self.db.bulk_insert(Strategy, strategies)
    
    def seed_portfolios(self) -> None:
        """
        إضافة محافظ أولية
        """
        portfolios = [
            {
                "portfolio_id": str(uuid.uuid4()),
                "user_id": "admin",  # يجب تحديث هذا بالمعرف الفعلي للمستخدم
                "name": "المحفظة الرئيسية",
                "description": "محفظة التداول الرئيسية",
                "balance": 10000.0,
                "created_at": datetime.utcnow()
            }
        ]
        self.db.bulk_insert(Portfolio, portfolios)
    
    def seed_positions(self) -> None:
        """
        إضافة مراكز أولية
        """
        positions = [
            {
                "position_id": str(uuid.uuid4()),
                "portfolio_id": "portfolio_id",  # يجب تحديث هذا بالمعرف الفعلي للمحفظة
                "strategy_id": "strategy_id",  # يجب تحديث هذا بالمعرف الفعلي للاستراتيجية
                "symbol": "BTC/USDT",
                "quantity": 0.1,
                "entry_price": 50000.0,
                "current_price": 51000.0,
                "is_open": True,
                "created_at": datetime.utcnow()
            }
        ]
        self.db.bulk_insert(Position, positions)
    
    def seed_trades(self) -> None:
        """
        إضافة صفقات أولية
        """
        trades = [
            {
                "trade_id": str(uuid.uuid4()),
                "position_id": "position_id",  # يجب تحديث هذا بالمعرف الفعلي للمركز
                "type": "buy",
                "quantity": 0.1,
                "price": 50000.0,
                "timestamp": datetime.utcnow()
            }
        ]
        self.db.bulk_insert(Trade, trades)
    
    def seed_alerts(self) -> None:
        """
        إضافة تنبيهات أولية
        """
        alerts = [
            {
                "alert_id": str(uuid.uuid4()),
                "user_id": "admin",  # يجب تحديث هذا بالمعرف الفعلي للمستخدم
                "symbol": "BTC/USDT",
                "condition": "price > 52000",
                "is_active": True,
                "created_at": datetime.utcnow()
            }
        ]
        self.db.bulk_insert(Alert, alerts)
    
    def seed_notifications(self) -> None:
        """
        إضافة إشعارات أولية
        """
        notifications = [
            {
                "notification_id": str(uuid.uuid4()),
                "user_id": "admin",  # يجب تحديث هذا بالمعرف الفعلي للمستخدم
                "title": "مرحباً بك",
                "message": "مرحباً بك في نظام التداول الخوارزمي",
                "is_read": False,
                "created_at": datetime.utcnow()
            }
        ]
        self.db.bulk_insert(Notification, notifications)
    
    def seed_market_data(self) -> None:
        """
        إضافة بيانات سوق أولية
        """
        market_data = [
            {
                "data_id": str(uuid.uuid4()),
                "symbol": "BTC/USDT",
                "timestamp": datetime.utcnow(),
                "open": 50000.0,
                "high": 51000.0,
                "low": 49500.0,
                "close": 50500.0,
                "volume": 1000.0,
                "created_at": datetime.utcnow()
            }
        ]
        self.db.bulk_insert(MarketData, market_data)
    
    def seed_market_indicators(self) -> None:
        """
        إضافة مؤشرات سوق أولية
        """
        indicators = [
            {
                "indicator_id": str(uuid.uuid4()),
                "symbol": "BTC/USDT",
                "name": "SMA",
                "parameters": '{"period": 20}',
                "value": 50500.0,
                "timestamp": datetime.utcnow(),
                "created_at": datetime.utcnow()
            }
        ]
        self.db.bulk_insert(MarketIndicator, indicators)
    
    def clear_all(self) -> None:
        """
        حذف جميع البيانات
        """
        try:
            self.db.drop_tables()
            self.db.create_tables()
            logger.info("تم حذف جميع البيانات بنجاح")
        except Exception as e:
            logger.error(f"خطأ في حذف البيانات: {str(e)}")
            raise

# إنشاء نسخة واحدة من مدير البيانات الأولية
seed_manager = SeedManager() 