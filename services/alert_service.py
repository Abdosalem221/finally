"""
خدمة إدارة وتتبع التنبيهات
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from app.services.data_service import DataService
from app.services.notification_service import NotificationService
from app.utils.helpers import format_currency, format_percentage, format_timestamp

class AlertService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_service = DataService()
        self.notification_service = NotificationService()
        self.alerts = {}
        # Added alert types for flexible alert processing
        self.alert_types = ['price_above', 'price_below', 'price_change_up', 'price_change_down', 'volume_above', 'rsi_above', 'rsi_below', 'macd_cross_up', 'macd_cross_down']

    def create_alert(self, user_id: str,
                    symbol: str,
                    condition: str,
                    value: float,
                    timeframe: str = '1h',
                    priority: str = 'normal',
                    description: Optional[str] = None) -> Dict:
        """
        إنشاء تنبيه جديد

        Args:
            user_id: معرف المستخدم
            symbol: رمز العملة
            condition: شرط التنبيه
            value: قيمة التنبيه
            timeframe: الإطار الزمني
            priority: أولوية التنبيه
            description: وصف التنبيه

        Returns:
            Dict: تفاصيل التنبيه
        """
        try:
            alert_id = f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            alert = {
                'id': alert_id,
                'user_id': user_id,
                'symbol': symbol,
                'condition': condition,
                'value': value,
                'timeframe': timeframe,
                'priority': priority,
                'description': description,
                'status': 'active',
                'created_at': datetime.now(),
                'last_checked': None,
                'triggered_at': None,
                'trigger_count': 0
            }

            self.alerts[alert_id] = alert

            return {
                'status': 'success',
                'alert_id': alert_id,
                'alert': alert
            }

        except Exception as e:
            self.logger.error(f"Error creating alert: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def check_alerts(self) -> Dict:
        """
        فحص التنبيهات

        Returns:
            Dict: نتائج الفحص
        """
        try:
            triggered_alerts = []

            for alert_id, alert in self.alerts.items():
                if alert['status'] != 'active':
                    continue

                # جلب بيانات السوق
                market_data = self.data_service.fetch_market_data(
                    alert['symbol'],
                    alert['timeframe'],
                    100  # عدد الشموع المطلوبة
                )

                if market_data['status'] == 'error':
                    continue

                # تحويل البيانات إلى DataFrame
                df = pd.DataFrame(market_data['data'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

                # فحص شرط التنبيه
                is_triggered = self._check_condition(
                    df,
                    alert['condition'],
                    alert['value']
                )

                if is_triggered:
                    # تحديث التنبيه
                    alert['status'] = 'triggered'
                    alert['triggered_at'] = datetime.now()
                    alert['trigger_count'] += 1

                    # إرسال إشعار
                    self._send_alert_notification(alert)

                    triggered_alerts.append(alert)

                alert['last_checked'] = datetime.now()

            return {
                'status': 'success',
                'triggered_alerts': triggered_alerts
            }

        except Exception as e:
            self.logger.error(f"Error checking alerts: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _check_condition(self, df: pd.DataFrame,
                        condition: str,
                        value: float) -> bool:
        """
        فحص شرط التنبيه

        Args:
            df: بيانات السوق
            condition: شرط التنبيه
            value: قيمة التنبيه

        Returns:
            bool: نتيجة الفحص
        """
        try:
            current_price = float(df['close'].iloc[-1])

            if condition == 'price_above':
                return current_price > value
            elif condition == 'price_below':
                return current_price < value
            elif condition == 'price_change_up':
                prev_price = float(df['close'].iloc[-2])
                return (current_price - prev_price) / prev_price * 100 >= value
            elif condition == 'price_change_down':
                prev_price = float(df['close'].iloc[-2])
                return (prev_price - current_price) / prev_price * 100 >= value
            elif condition == 'volume_above':
                current_volume = float(df['volume'].iloc[-1])
                avg_volume = float(df['volume'].rolling(20).mean().iloc[-1])
                return current_volume > avg_volume * (1 + value/100)
            elif condition == 'rsi_above':
                rsi = self._calculate_rsi(df['close'])
                return rsi.iloc[-1] > value
            elif condition == 'rsi_below':
                rsi = self._calculate_rsi(df['close'])
                return rsi.iloc[-1] < value
            elif condition == 'macd_cross_up':
                macd, signal = self._calculate_macd(df['close'])
                return macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]
            elif condition == 'macd_cross_down':
                macd, signal = self._calculate_macd(df['close'])
                return macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]
            else:
                return False

        except Exception as e:
            self.logger.error(f"Error checking condition: {str(e)}")
            return False

    def _send_alert_notification(self, alert: Dict) -> None:
        """
        إرسال إشعار التنبيه

        Args:
            alert: تفاصيل التنبيه
        """
        try:
            title = f"تنبيه: {alert['symbol']}"

            message = f"""
            تم تفعيل التنبيه:
            العملة: {alert['symbol']}
            الشرط: {self._get_condition_description(alert['condition'])}
            القيمة: {self._format_alert_value(alert['condition'], alert['value'])}
            الإطار الزمني: {alert['timeframe']}
            """

            if alert['description']:
                message += f"\nالوصف: {alert['description']}"

            data = {
                'symbol': alert['symbol'],
                'condition': alert['condition'],
                'value': alert['value'],
                'timeframe': alert['timeframe'],
                'trigger_count': alert['trigger_count']
            }

            self.notification_service.send_notification(
                alert['user_id'],
                title,
                message,
                ['email', 'telegram'], # Default channels
                alert['priority'],
                data
            )

        except Exception as e:
            self.logger.error(f"Error sending alert notification: {str(e)}")

    def _get_condition_description(self, condition: str) -> str:
        """
        الحصول على وصف الشرط

        Args:
            condition: شرط التنبيه

        Returns:
            str: وصف الشرط
        """
        descriptions = {
            'price_above': 'السعر فوق',
            'price_below': 'السعر تحت',
            'price_change_up': 'ارتفاع السعر بنسبة',
            'price_change_down': 'انخفاض السعر بنسبة',
            'volume_above': 'الحجم فوق المتوسط بنسبة',
            'rsi_above': 'مؤشر RSI فوق',
            'rsi_below': 'مؤشر RSI تحت',
            'macd_cross_up': 'تقاطع MACD للأعلى',
            'macd_cross_down': 'تقاطع MACD للأسفل'
        }

        return descriptions.get(condition, condition)

    def _format_alert_value(self, condition: str, value: float) -> str:
        """
        تنسيق قيمة التنبيه

        Args:
            condition: شرط التنبيه
            value: قيمة التنبيه

        Returns:
            str: القيمة المنسقة
        """
        if condition in ['price_above', 'price_below']:
            return format_currency(value)
        elif condition in ['price_change_up', 'price_change_down', 'volume_above']:
            return format_percentage(value)
        else:
            return str(value)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        حساب مؤشر RSI

        Args:
            prices: أسعار الإغلاق
            period: فترة الحساب

        Returns:
            pd.Series: قيم RSI
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series,
                       fast_period: int = 12,
                       slow_period: int = 26,
                       signal_period: int = 9) -> tuple:
        """
        حساب مؤشر MACD

        Args:
            prices: أسعار الإغلاق
            fast_period: الفترة السريعة
            slow_period: الفترة البطيئة
            signal_period: فترة الإشارة

        Returns:
            tuple: (MACD, Signal)
        """
        exp1 = prices.ewm(span=fast_period, adjust=False).mean()
        exp2 = prices.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal

    def update_alert(self, alert_id: str, **kwargs) -> Dict:
        """
        تحديث التنبيه

        Args:
            alert_id: معرف التنبيه
            **kwargs: الحقول المطلوب تحديثها

        Returns:
            Dict: نتيجة التحديث
        """
        try:
            if alert_id not in self.alerts:
                return {'status': 'error', 'message': 'التنبيه غير موجود'}

            alert = self.alerts[alert_id]

            # تحديث الحقول
            for key, value in kwargs.items():
                if key in alert:
                    alert[key] = value

            return {
                'status': 'success',
                'alert': alert
            }

        except Exception as e:
            self.logger.error(f"Error updating alert: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def delete_alert(self, alert_id: str) -> Dict:
        """
        حذف التنبيه

        Args:
            alert_id: معرف التنبيه

        Returns:
            Dict: نتيجة الحذف
        """
        try:
            if alert_id not in self.alerts:
                return {'status': 'error', 'message': 'التنبيه غير موجود'}

            del self.alerts[alert_id]

            return {'status': 'success'}

        except Exception as e:
            self.logger.error(f"Error deleting alert: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def get_user_alerts(self, user_id: str,
                       status: Optional[str] = None) -> Dict:
        """
        الحصول على تنبيهات المستخدم

        Args:
            user_id: معرف المستخدم
            status: حالة التنبيهات

        Returns:
            Dict: قائمة التنبيهات
        """
        try:
            user_alerts = [
                a for a in self.alerts.values()
                if a['user_id'] == user_id
            ]

            if status:
                user_alerts = [
                    a for a in user_alerts
                    if a['status'] == status
                ]

            return {
                'status': 'success',
                'alerts': user_alerts
            }

        except Exception as e:
            self.logger.error(f"Error getting user alerts: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def process_alerts(self, market_data: dict) -> List[Dict]:
        """معالجة التنبيهات بشكل متقدم"""
        try:
            alerts_triggered = []
            for alert_id, alert in self.alerts.items():
                if alert['status'] == 'active':
                  if self.check_conditions(market_data, alert): #check condition against each alert
                    alert['status'] = 'triggered'
                    alert['triggered_at'] = datetime.now()
                    alert['trigger_count'] += 1
                    self._send_alert_notification(alert)
                    alerts_triggered.append(alert)
            return alerts_triggered
        except Exception as e:
            self.logger.error(f"خطأ في معالجة التنبيهات: {str(e)}")
            return []

    def check_conditions(self, market_data: dict, alert: dict) -> bool:
        """Checks if alert conditions are met based on market data."""
        try:
            df = pd.DataFrame(market_data['data'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return self._check_condition(df, alert['condition'], alert['value'])
        except Exception as e:
            self.logger.error(f"Error checking alert conditions: {str(e)}")
            return False


    def notify_users(self, alert: dict):
        """Sends notifications to relevant users."""
        try:
            self._send_alert_notification(alert)
        except Exception as e:
            self.logger.error(f"Error notifying users: {str(e)}")