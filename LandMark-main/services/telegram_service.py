"""
خدمة التليجرام المتقدمة
"""

import os
from typing import Dict, List, Optional
from datetime import datetime
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from app.services.signal_service import SignalService
from app.services.alert_service import AlertService
from app.utils.helpers import format_currency, format_percentage, format_timestamp

class TelegramService:
    def __init__(self):
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.signal_service = SignalService()
        self.alert_service = AlertService()
        self.bot = telegram.Bot(token=self.token)
        self.updater = Updater(token=self.token, use_context=True)
        
        # إضافة معالجات الأوامر
        self.updater.dispatcher.add_handler(CommandHandler('start', self._handle_start))
        self.updater.dispatcher.add_handler(CommandHandler('help', self._handle_help))
        self.updater.dispatcher.add_handler(CommandHandler('signals', self._handle_signals))
        self.updater.dispatcher.add_handler(CommandHandler('alerts', self._handle_alerts))
        self.updater.dispatcher.add_handler(CommandHandler('status', self._handle_status))
        self.updater.dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, self._handle_message))
    
    def start(self):
        """بدء تشغيل البوت"""
        try:
            self.updater.start_polling()
            return {'status': 'success', 'message': 'Bot started successfully'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def stop(self):
        """إيقاف البوت"""
        try:
            self.updater.stop()
            return {'status': 'success', 'message': 'Bot stopped successfully'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def send_signal(self, signal: Dict) -> Dict:
        """إرسال إشارة"""
        try:
            message = self._format_signal_message(signal)
            self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='HTML')
            return {'status': 'success', 'message': 'Signal sent successfully'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def send_alert(self, alert: Dict) -> Dict:
        """إرسال تنبيه"""
        try:
            message = self._format_alert_message(alert)
            self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='HTML')
            return {'status': 'success', 'message': 'Alert sent successfully'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _handle_start(self, update, context):
        """معالجة أمر البدء"""
        update.message.reply_text(
            "مرحباً! أنا بوت AlgoTraderPro.\n"
            "يمكنني إرسال إشارات التداول والتنبيهات.\n"
            "استخدم /help لمعرفة الأوامر المتاحة."
        )
    
    def _handle_help(self, update, context):
        """معالجة أمر المساعدة"""
        update.message.reply_text(
            "الأوامر المتاحة:\n"
            "/start - بدء البوت\n"
            "/help - عرض هذه الرسالة\n"
            "/signals - عرض آخر الإشارات\n"
            "/alerts - عرض التنبيهات النشطة\n"
            "/status - عرض حالة السوق"
        )
    
    def _handle_signals(self, update, context):
        """معالجة أمر الإشارات"""
        try:
            signals = self.signal_service.get_signals({'limit': 5})
            if signals['status'] == 'error':
                update.message.reply_text("حدث خطأ في جلب الإشارات.")
                return
            
            if not signals['signals']:
                update.message.reply_text("لا توجد إشارات حالياً.")
                return
            
            for signal in signals['signals']:
                message = self._format_signal_message(signal)
                update.message.reply_text(message, parse_mode='HTML')
        except Exception as e:
            update.message.reply_text(f"حدث خطأ: {str(e)}")
    
    def _handle_alerts(self, update, context):
        """معالجة أمر التنبيهات"""
        try:
            alerts = self.alert_service.get_alerts({'status': 'PENDING', 'limit': 5})
            if alerts['status'] == 'error':
                update.message.reply_text("حدث خطأ في جلب التنبيهات.")
                return
            
            if not alerts['alerts']:
                update.message.reply_text("لا توجد تنبيهات نشطة حالياً.")
                return
            
            for alert in alerts['alerts']:
                message = self._format_alert_message(alert)
                update.message.reply_text(message, parse_mode='HTML')
        except Exception as e:
            update.message.reply_text(f"حدث خطأ: {str(e)}")
    
    def _handle_status(self, update, context):
        """معالجة أمر الحالة"""
        try:
            # جلب حالة السوق
            market_status = self.signal_service.get_market_status()
            if market_status['status'] == 'error':
                update.message.reply_text("حدث خطأ في جلب حالة السوق.")
                return
            
            message = (
                f"<b>حالة السوق:</b>\n"
                f"السوق: {'مفتوح' if market_status['is_open'] else 'مغلق'}\n"
                f"آخر تحديث: {format_timestamp(market_status['last_update'])}"
            )
            
            update.message.reply_text(message, parse_mode='HTML')
        except Exception as e:
            update.message.reply_text(f"حدث خطأ: {str(e)}")
    
    def _handle_message(self, update, context):
        """معالجة الرسائل النصية"""
        update.message.reply_text(
            "عذراً، لا أفهم هذه الرسالة.\n"
            "استخدم /help لمعرفة الأوامر المتاحة."
        )
    
    def _format_signal_message(self, signal: Dict) -> str:
        """تنسيق رسالة الإشارة"""
        return (
            f"<b>إشارة جديدة:</b>\n"
            f"الرمز: {signal['symbol']}\n"
            f"الإطار الزمني: {signal['timeframe']}\n"
            f"النوع: {signal['signal_type']}\n"
            f"سعر الدخول: {format_currency(signal['entry_price'])}\n"
            f"وقف الخسارة: {format_currency(signal['stop_loss'])}\n"
            f"جني الأرباح: {format_currency(signal['take_profit'])}\n"
            f"نسبة المخاطرة إلى العائد: {format_percentage(signal['risk_reward_ratio'])}\n"
            f"درجة الثقة: {format_percentage(signal['confidence_score'])}\n"
            f"الوقت: {format_timestamp(signal['created_at'])}"
        )
    
    def _format_alert_message(self, alert: Dict) -> str:
        """تنسيق رسالة التنبيه"""
        return (
            f"<b>تنبيه:</b>\n"
            f"الرمز: {alert['symbol']}\n"
            f"الإطار الزمني: {alert['timeframe']}\n"
            f"النوع: {alert['alert_type']}\n"
            f"الشرط: {alert['condition']}\n"
            f"القيمة: {format_currency(alert['value'])}\n"
            f"الحالة: {alert['status']}\n"
            f"الوقت: {format_timestamp(alert['created_at'])}"
        ) 