
class EnhancedAlertSystem:
    def __init__(self):
        self.alert_types = ['price', 'pattern', 'trend', 'volume', 'volatility', 'supertrend', 'support_resistance']
        self.notification_channels = ['email', 'telegram', 'web', 'mobile', 'desktop']
        self.priority_levels = ['low', 'medium', 'high', 'critical']
        
    def create_smart_alert(self, symbol: str, conditions: Dict, channels: List[str] = None) -> Dict:
        """إنشاء تنبيه ذكي مع شروط متعددة"""
        try:
            alert = {
                'symbol': symbol,
                'conditions': conditions,
                'channels': channels or ['telegram'],
                'priority': self._calculate_priority(conditions),
                'created_at': datetime.utcnow()
            }
            return {'status': 'success', 'alert': alert}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
            
    def _calculate_priority(self, conditions: Dict) -> str:
        """حساب أولوية التنبيه بناءً على الشروط"""
        score = 0
        for condition in conditions:
            if condition.get('type') == 'price_breakout':
                score += 3
            elif condition.get('type') == 'volume_spike':
                score += 2
            elif condition.get('type') == 'trend_change':
                score += 2
                
        if score >= 5:
            return 'critical'
        elif score >= 3:
            return 'high'
        elif score >= 2:
            return 'medium'
        return 'low'
        
    def process_alerts(self, market_data, thresholds):
        """معالجة وتصفية التنبيهات"""
        alerts = []
        for alert_type in self.alert_types:
            if self.check_alert_conditions(market_data, alert_type, thresholds):
                alerts.append(self.create_alert(alert_type, market_data))
        return alerts
from typing import Dict, List
import logging
from datetime import datetime

class EnhancedAlertService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_smart_alert(self, 
                         symbol: str,
                         conditions: Dict,
                         notification_channels: List[str] = ["telegram"]) -> Dict:
        """
        إنشاء تنبيه ذكي مع شروط متعددة
        """
        try:
            alert = {
                "symbol": symbol,
                "conditions": conditions,
                "channels": notification_channels,
                "created_at": datetime.utcnow()
            }
            
            self.logger.info(f"تم إنشاء تنبيه جديد: {alert}")
            return {"status": "success", "alert": alert}
            
        except Exception as e:
            self.logger.error(f"خطأ في إنشاء التنبيه: {str(e)}")
            return {"status": "error", "message": str(e)}
