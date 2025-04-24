# تأكد من أن هذا الملف (الذي يحتوي على create_app) لديه الاستيرادات الصحيحة
from flask import Flask
from flask_cors import CORS
from database.init_db import init_db
from routes import register_routes
import os  # <-- إضافة استيراد os، ضروري لـ os.environ.get في الجزء السفلي

# يمكنك استيراد كلاسات الإعدادات هنا أو تحديد مساراتها لاحقاً
# مثال:
# from config import DevelopmentConfig, TestingConfig, ProductionConfig
# أو إذا كانت في مكان آخر مثل database.config
# from database.config import DatabaseConfig, TestingConfig # <--- افترض وجود TestingConfig هنا أو في مكان مشابه


def create_app(config_name=None):  # <--- تم إضافة الوسيط هنا وجعله اختيارياً
    app = Flask(__name__)
    CORS(app)

    # تهيئة الإعدادات بناءً على config_name
    # هذا الجزء يحتاج إلى التكيف مع كيفية هيكلة إعداداتك بالضبط
    if config_name == 'testing':
        # حاول تحميل إعدادات الاختبار
        # افترض أن لديك TestingConfig في مكان ما يمكن استيراده
        try:
            # مثال إذا كانت الإعدادات في database.config
            from database.config import TestingConfig
            app.config.from_object(TestingConfig)
            print("Loaded TestingConfig")  # للمساعدة في التتبع
        except ImportError:
            print("Warning: TestingConfig not found. Using default settings.")
            # إذا لم يتم العثور على إعدادات الاختبار، استخدم الإعدادات الافتراضية
            app.config.from_object('database.config.DatabaseConfig')
        except AttributeError:
            print(
                "Warning: TestingConfig found but is not a valid object. Using default settings."
            )
            app.config.from_object('database.config.DatabaseConfig')

    else:
        # استخدم الإعدادات الافتراضية لبيئات التطوير أو الإنتاج إذا لم يتم تحديد config_name
        # أو إذا تم تمرير قيمة أخرى غير 'testing'
        app.config.from_object('database.config.DatabaseConfig')
        print(
            f"Loaded Default Config (database.config.DatabaseConfig) for config_name: {config_name}"
        )

    # Initialize services (تأكد أن هذه الدوال موجودة ويمكن استيرادها)
    # init_db() # <--- قد تحتاج هذه الدالة إلى وسيط app لإعداد الاتصال بالقاعدة بناءً على الإعدادات المحملة
    # مثال: init_db(app) إذا كانت الدالة تحتاج الوصول إلى app.config

    # Register routes (تأكد أن هذه الدالة موجودة ويمكن استيرادها)
    register_routes(app)  # تأكد أن هذه الدالة تقبل وسيط app

    return app


# الجزء الخاص بتشغيل التطبيق مباشرة (عند تشغيل الملف)
if __name__ == '__main__':
    # عند التشغيل المباشر، لا نمرر config_name، سيتم استخدام الإعدادات الافتراضية
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port,
            debug=True)  # يمكنك تشغيل debug=True للتطوير
