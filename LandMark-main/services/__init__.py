"""
خدمات التطبيق
"""

from .signal_service import SignalService
from .data_service import DataService
from .analysis_service import AnalysisService
# تم تصحيح الاستيراد ليطابق اسم الكلاس الفعلي (AIService) في ملف ai_service.py
from .ai_service import AIService
# تم تصحيح الاستيراد ليطابق اسم الكلاس الفعلي (EnhancedServiceManager) في ملف enhanced_service_manager.py
from .enhanced_service_manager import EnhancedServiceManager

# يمكنك إضافة الكلاسات المستوردة هنا إلى قائمة __all__ إذا أردت التحكم
# في ما يتم استيراده عند استخدام from app.services import *
# __all__ = [
#     "SignalService",
#     "DataService",
#     "AnalysisService",
#     "AIService",
#     "EnhancedServiceManager", # <--- تم تحديثه هنا أيضاً
# ]