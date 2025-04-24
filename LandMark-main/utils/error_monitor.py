import logging
from datetime import datetime
from .logger import error_logger

class ErrorMonitor:
    def __init__(self):
        self.errors = []
        self.logger = error_logger

    def log_error(self, error):
        error_info = {
            'timestamp': datetime.now(),
            'message': str(error),
            'type': type(error).__name__
        }
        self.errors.append(error_info)
        self.logger.error(f"Error: {error_info['type']} - {error_info['message']}")
        # حفظ الخطأ في ملف السجلات عبر error_logger

    def get_recent_errors(self, limit=10):
        return self.errors[-limit:]

    def clear_errors(self):
        self.errors = []

    def get_errors(self):
        return self.errors