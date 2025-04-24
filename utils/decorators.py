"""
ديكورات التطبيق
"""

from functools import wraps
from flask import request, jsonify
from flask_login import current_user
import time
from datetime import datetime, timedelta

def require_api_key(f):
    """مطلوب مفتاح API"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({
                'status': 'error',
                'message': 'API key is required'
            }), 401
        return f(*args, **kwargs)
    return decorated_function

def require_admin(f):
    """مطلوب صلاحيات مدير"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            return jsonify({
                'status': 'error',
                'message': 'Admin privileges required'
            }), 403
        return f(*args, **kwargs)
    return decorated_function

def require_premium(f):
    """مطلوب اشتراك مميز"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_premium:
            return jsonify({
                'status': 'error',
                'message': 'Premium subscription required'
            }), 403
        return f(*args, **kwargs)
    return decorated_function

def cache_response(seconds=60):
    """تخزين مؤقت للاستجابة"""
    def decorator(f):
        cache = {}
        
        @wraps(f)
        def decorated_function(*args, **kwargs):
            key = f"{f.__name__}:{str(args)}:{str(kwargs)}"
            now = datetime.now()
            
            if key in cache:
                cached_time, cached_result = cache[key]
                if now - cached_time < timedelta(seconds=seconds):
                    return cached_result
            
            result = f(*args, **kwargs)
            cache[key] = (now, result)
            return result
        
        return decorated_function
    return decorator

def rate_limit(requests_per_minute=60):
    """تحديد معدل الطلبات"""
    def decorator(f):
        request_times = []
        
        @wraps(f)
        def decorated_function(*args, **kwargs):
            now = time.time()
            request_times.append(now)
            
            # إزالة الطلبات القديمة
            while request_times and request_times[0] < now - 60:
                request_times.pop(0)
            
            if len(request_times) > requests_per_minute:
                return jsonify({
                    'status': 'error',
                    'message': 'Rate limit exceeded'
                }), 429
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator 