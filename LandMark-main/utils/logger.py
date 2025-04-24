
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

class LoggerSetup:
    @staticmethod
    def setup_logger(name, log_file, level=logging.INFO):
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        
        handler = RotatingFileHandler(
            log_file, maxBytes=10000000, backupCount=5
        )
        handler.setFormatter(formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        logger.addHandler(console_handler)
        
        return logger

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Setup different loggers
app_logger = LoggerSetup.setup_logger('app', 'logs/app.log')
api_logger = LoggerSetup.setup_logger('api', 'logs/api.log')
error_logger = LoggerSetup.setup_logger('error', 'logs/error.log', logging.ERROR)
