"""
خدمة إدارة وتوزيع الإشعارات
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
from app.utils.helpers import format_currency, format_percentage, format_timestamp

class NotificationService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.notifications = {}
        self.channels = {
            'email': self._send_email,
            'telegram': self._send_telegram,
            'webhook': self._send_webhook
        }
    
    def send_notification(self, user_id: str,
                         title: str,
                         message: str,
                         channels: List[str],
                         priority: str = 'normal',
                         data: Optional[Dict] = None) -> Dict:
        """
        إرسال إشعار
        
        Args:
            user_id: معرف المستخدم
            title: عنوان الإشعار
            message: محتوى الإشعار
            channels: قنوات الإرسال
            priority: أولوية الإشعار
            data: بيانات إضافية
            
        Returns:
            Dict: نتيجة الإرسال
        """
        try:
            notification_id = f"notification_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            notification = {
                'id': notification_id,
                'user_id': user_id,
                'title': title,
                'message': message,
                'channels': channels,
                'priority': priority,
                'data': data or {},
                'status': 'pending',
                'created_at': datetime.now(),
                'delivery_status': {}
            }
            
            # إرسال الإشعار عبر القنوات المحددة
            for channel in channels:
                if channel in self.channels:
                    try:
                        self.channels[channel](notification)
                        notification['delivery_status'][channel] = 'sent'
                    except Exception as e:
                        self.logger.error(f"Error sending notification via {channel}: {str(e)}")
                        notification['delivery_status'][channel] = 'failed'
            
            # تحديث حالة الإشعار
            if all(status == 'sent' for status in notification['delivery_status'].values()):
                notification['status'] = 'sent'
            elif any(status == 'sent' for status in notification['delivery_status'].values()):
                notification['status'] = 'partial'
            else:
                notification['status'] = 'failed'
            
            self.notifications[notification_id] = notification
            
            return {
                'status': 'success',
                'notification_id': notification_id,
                'notification': notification
            }
            
        except Exception as e:
            self.logger.error(f"Error sending notification: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _send_email(self, notification: Dict) -> None:
        """
        إرسال إشعار عبر البريد الإلكتروني
        
        Args:
            notification: تفاصيل الإشعار
        """
        try:
            # TODO: تكوين إعدادات البريد الإلكتروني
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            sender_email = "noreply@algotraderpro.com"
            sender_password = "your_password"
            
            # إنشاء رسالة البريد الإلكتروني
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = notification['user_id']  # TODO: الحصول على بريد المستخدم
            msg['Subject'] = notification['title']
            
            # إضافة محتوى الرسالة
            body = f"""
            {notification['message']}
            
            أولوية: {notification['priority']}
            وقت الإرسال: {format_timestamp(notification['created_at'])}
            """
            
            if notification['data']:
                body += "\nبيانات إضافية:\n"
                for key, value in notification['data'].items():
                    if isinstance(value, (int, float)):
                        if 'price' in key or 'balance' in key:
                            body += f"{key}: {format_currency(value)}\n"
                        elif 'percentage' in key:
                            body += f"{key}: {format_percentage(value)}\n"
                    else:
                        body += f"{key}: {value}\n"
            
            msg.attach(MIMEText(body, 'plain'))
            
            # إرسال البريد الإلكتروني
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
        except Exception as e:
            raise Exception(f"Error sending email: {str(e)}")
    
    def _send_telegram(self, notification: Dict) -> None:
        """
        إرسال إشعار عبر Telegram
        
        Args:
            notification: تفاصيل الإشعار
        """
        try:
            # TODO: تكوين إعدادات Telegram
            bot_token = "your_bot_token"
            chat_id = notification['user_id']  # TODO: الحصول على معرف الدردشة
            
            # إنشاء محتوى الرسالة
            message = f"*{notification['title']}*\n\n"
            message += f"{notification['message']}\n\n"
            message += f"الأولوية: {notification['priority']}\n"
            message += f"وقت الإرسال: {format_timestamp(notification['created_at'])}"
            
            if notification['data']:
                message += "\n\n*بيانات إضافية:*\n"
                for key, value in notification['data'].items():
                    if isinstance(value, (int, float)):
                        if 'price' in key or 'balance' in key:
                            message += f"{key}: {format_currency(value)}\n"
                        elif 'percentage' in key:
                            message += f"{key}: {format_percentage(value)}\n"
                    else:
                        message += f"{key}: {value}\n"
            
            # إرسال الرسالة
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            params = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, params=params)
            response.raise_for_status()
            
        except Exception as e:
            raise Exception(f"Error sending Telegram message: {str(e)}")
    
    def _send_webhook(self, notification: Dict) -> None:
        """
        إرسال إشعار عبر Webhook
        
        Args:
            notification: تفاصيل الإشعار
        """
        try:
            # TODO: تكوين إعدادات Webhook
            webhook_url = "your_webhook_url"
            
            # إنشاء محتوى الإشعار
            payload = {
                'id': notification['id'],
                'user_id': notification['user_id'],
                'title': notification['title'],
                'message': notification['message'],
                'priority': notification['priority'],
                'data': notification['data'],
                'timestamp': format_timestamp(notification['created_at'])
            }
            
            # إرسال الإشعار
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
        except Exception as e:
            raise Exception(f"Error sending webhook: {str(e)}")
    
    def get_notification_status(self, notification_id: str) -> Dict:
        """
        الحصول على حالة الإشعار
        
        Args:
            notification_id: معرف الإشعار
            
        Returns:
            Dict: حالة الإشعار
        """
        try:
            if notification_id not in self.notifications:
                return {'status': 'error', 'message': 'الإشعار غير موجود'}
            
            return {
                'status': 'success',
                'notification': self.notifications[notification_id]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting notification status: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def get_user_notifications(self, user_id: str,
                             limit: int = 10,
                             status: Optional[str] = None) -> Dict:
        """
        الحصول على إشعارات المستخدم
        
        Args:
            user_id: معرف المستخدم
            limit: الحد الأقصى للنتائج
            status: حالة الإشعارات
            
        Returns:
            Dict: قائمة الإشعارات
        """
        try:
            user_notifications = [
                n for n in self.notifications.values()
                if n['user_id'] == user_id
            ]
            
            if status:
                user_notifications = [
                    n for n in user_notifications
                    if n['status'] == status
                ]
            
            user_notifications.sort(
                key=lambda x: x['created_at'],
                reverse=True
            )
            
            return {
                'status': 'success',
                'notifications': user_notifications[:limit]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting user notifications: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def mark_as_read(self, notification_id: str) -> Dict:
        """
        وضع علامة مقروء على الإشعار
        
        Args:
            notification_id: معرف الإشعار
            
        Returns:
            Dict: نتيجة العملية
        """
        try:
            if notification_id not in self.notifications:
                return {'status': 'error', 'message': 'الإشعار غير موجود'}
            
            self.notifications[notification_id]['status'] = 'read'
            
            return {'status': 'success'}
            
        except Exception as e:
            self.logger.error(f"Error marking notification as read: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def delete_notification(self, notification_id: str) -> Dict:
        """
        حذف إشعار
        
        Args:
            notification_id: معرف الإشعار
            
        Returns:
            Dict: نتيجة الحذف
        """
        try:
            if notification_id not in self.notifications:
                return {'status': 'error', 'message': 'الإشعار غير موجود'}
            
            del self.notifications[notification_id]
            
            return {'status': 'success'}
            
        except Exception as e:
            self.logger.error(f"Error deleting notification: {str(e)}")
            return {'status': 'error', 'message': str(e)} 