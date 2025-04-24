
"""
مدير الخدمات المحسن للأداء العالي
"""

import asyncio
from typing import Dict, List
import logging

class EnhancedServiceManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.services = {}
        self.performance_metrics = {}
        
    async def register_service(self, name: str, service: object):
        """تسجيل خدمة جديدة"""
        self.services[name] = service
        self.performance_metrics[name] = {
            'response_times': [],
            'success_rate': 1.0,
            'error_count': 0
        }
        
    async def execute_service(self, name: str, **kwargs) -> Dict:
        """تنفيذ خدمة مع مراقبة الأداء"""
        try:
            service = self.services.get(name)
            if not service:
                raise ValueError(f"Service {name} not found")
                
            # تنفيذ الخدمة مع قياس الوقت
            start_time = asyncio.get_event_loop().time()
            result = await service.execute(**kwargs)
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # تحديث مقاييس الأداء
            self._update_metrics(name, execution_time, True)
            
            return {
                'status': 'success',
                'result': result,
                'execution_time': execution_time
            }
            
        except Exception as e:
            self._update_metrics(name, 0, False)
            self.logger.error(f"Error executing service {name}: {str(e)}")
            return {'status': 'error', 'message': str(e)}
            
    def _update_metrics(self, name: str, execution_time: float, success: bool):
        """تحديث مقاييس الأداء"""
        metrics = self.performance_metrics[name]
        metrics['response_times'].append(execution_time)
        
        if len(metrics['response_times']) > 100:
            metrics['response_times'].pop(0)
            
        if not success:
            metrics['error_count'] += 1
            
        metrics['success_rate'] = 1 - (metrics['error_count'] / len(metrics['response_times']))

