
from flask import Blueprint, render_template
from datetime import datetime, timedelta
import psutil
import json
from ..utils.logger import app_logger

monitoring_bp = Blueprint('monitoring', __name__)

class SystemMonitor:
    @staticmethod
    def get_system_metrics():
        return {
            'cpu': psutil.cpu_percent(),
            'memory': psutil.virtual_memory().percent,
            'disk': psutil.disk_usage('/').percent,
            'python_threads': len(psutil.Process().threads()),
            'timestamp': datetime.now().isoformat()
        }

    @staticmethod
    def get_error_stats():
        with open('logs/error.log', 'r') as f:
            errors = f.readlines()
        return {
            'total_errors': len(errors),
            'recent_errors': len([e for e in errors[-100:] if 'ERROR' in e])
        }

@monitoring_bp.route('/dashboard')
def dashboard():
    metrics = SystemMonitor.get_system_metrics()
    error_stats = SystemMonitor.get_error_stats()
    return render_template('monitoring/dashboard.html',
                         metrics=metrics,
                         error_stats=error_stats)
