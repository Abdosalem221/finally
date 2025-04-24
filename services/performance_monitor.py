
import logging
from datetime import datetime
from typing import Dict, List

class EnhancedPerformanceMonitor:
    def __init__(self):
        self.logger = logging.getLogger("PerformanceMonitor")
        self.metrics = {
            'execution_time': [],
            'signal_accuracy': [],
            'resource_usage': [],
            'indicator_reliability': []
        }
        
    def record_metric(self, category: str, value: float):
        """تسجيل مقياس أداء"""
        if category in self.metrics:
            self.metrics[category].append({
                'value': value,
                'timestamp': datetime.now()
            })
            
    def get_performance_report(self) -> Dict:
        """إنشاء تقرير الأداء"""
        return {
            'summary': {
                cat: {
                    'avg': sum(m['value'] for m in metrics) / len(metrics) if metrics else 0,
                    'count': len(metrics)
                } for cat, metrics in self.metrics.items()
            },
            'detailed': self.metrics
        }
