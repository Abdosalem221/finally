
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import logging

db = SQLAlchemy()
migrate = Migrate()

def init_db():
    try:
        db.create_all()
        logging.info("تم تهيئة قاعدة البيانات بنجاح")
    except Exception as e:
        logging.error(f"خطأ في تهيئة قاعدة البيانات: {str(e)}")
        raise
