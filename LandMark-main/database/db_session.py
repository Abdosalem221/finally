"""
إدارة جلسات قاعدة البيانات
"""

import os
import logging
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Union
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
from app.database.models import Base

class DatabaseConfig:
    @staticmethod
    def get_test_uri():
        return "sqlite:///:memory:"

class DatabaseSessionManager:
    def __init__(self, testing=False):
        """
        تهيئة مدير جلسات قاعدة البيانات
        """
        self.db_url = DatabaseConfig.get_test_uri() if testing else self._get_db_url()
        self.engine = create_engine(
            self.db_url,
            pool_size=int(os.getenv('DB_POOL_SIZE', 20)),
            max_overflow=int(os.getenv('DB_MAX_OVERFLOW', 10)),
            pool_timeout=int(os.getenv('DB_POOL_TIMEOUT', 30)),
            pool_recycle=int(os.getenv('DB_POOL_RECYCLE', 1800))
        )
        self.Session = sessionmaker(bind=self.engine)

    def _get_db_url(self) -> str:
        """
        إنشاء عنوان قاعدة البيانات من متغيرات البيئة
        """
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '5432')
        db_name = os.getenv('DB_NAME', 'algotrader_db')
        db_user = os.getenv('DB_USER', 'postgres')
        db_password = os.getenv('DB_PASSWORD', '')
        db_ssl_mode = os.getenv('DB_SSL_MODE', 'disable')
        
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode={db_ssl_mode}"
    
    @contextmanager
    def get_session(self) -> Session:
        """
        الحصول على جلسة قاعدة بيانات جديدة
        """
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"خطأ في جلسة قاعدة البيانات: {str(e)}")
            raise
        finally:
            session.close()
    
    def create_record(self, model: Base, data: Dict[str, Any]) -> Base:
        """
        إنشاء سجل جديد
        """
        try:
            with self.get_session() as session:
                record = model(**data)
                session.add(record)
                session.flush()
                return record
        except SQLAlchemyError as e:
            logger.error(f"خطأ في إنشاء السجل: {str(e)}")
            raise
    
    def get_record(self, model: Base, record_id: Union[str, int]) -> Optional[Base]:
        """
        الحصول على سجل محدد
        """
        try:
            with self.get_session() as session:
                return session.query(model).get(record_id)
        except SQLAlchemyError as e:
            logger.error(f"خطأ في الحصول على السجل: {str(e)}")
            raise
    
    def update_record(self, model: Base, record_id: Union[str, int], data: Dict[str, Any]) -> Optional[Base]:
        """
        تحديث سجل محدد
        """
        try:
            with self.get_session() as session:
                record = session.query(model).get(record_id)
                if record:
                    for key, value in data.items():
                        setattr(record, key, value)
                    session.flush()
                return record
        except SQLAlchemyError as e:
            logger.error(f"خطأ في تحديث السجل: {str(e)}")
            raise
    
    def delete_record(self, model: Base, record_id: Union[str, int]) -> bool:
        """
        حذف سجل محدد
        """
        try:
            with self.get_session() as session:
                record = session.query(model).get(record_id)
                if record:
                    session.delete(record)
                    return True
                return False
        except SQLAlchemyError as e:
            logger.error(f"خطأ في حذف السجل: {str(e)}")
            raise
    
    def query_records(self, model: Base, filters: Optional[Dict[str, Any]] = None) -> List[Base]:
        """
        استعلام عن السجلات
        """
        try:
            with self.get_session() as session:
                query = session.query(model)
                if filters:
                    for key, value in filters.items():
                        query = query.filter(getattr(model, key) == value)
                return query.all()
        except SQLAlchemyError as e:
            logger.error(f"خطأ في استعلام السجلات: {str(e)}")
            raise
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        تنفيذ استعلام SQL مباشر
        """
        try:
            with self.get_session() as session:
                result = session.execute(text(query), params or {})
                return [dict(row) for row in result]
        except SQLAlchemyError as e:
            logger.error(f"خطأ في تنفيذ الاستعلام: {str(e)}")
            raise
    
    def bulk_insert(self, model: Base, records: List[Dict[str, Any]]) -> None:
        """
        إدراج مجموعة من السجلات
        """
        try:
            with self.get_session() as session:
                session.bulk_insert_mappings(model, records)
        except SQLAlchemyError as e:
            logger.error(f"خطأ في الإدراج الجماعي: {str(e)}")
            raise
    
    def bulk_update(self, model: Base, records: List[Dict[str, Any]]) -> None:
        """
        تحديث مجموعة من السجلات
        """
        try:
            with self.get_session() as session:
                session.bulk_update_mappings(model, records)
        except SQLAlchemyError as e:
            logger.error(f"خطأ في التحديث الجماعي: {str(e)}")
            raise
    
    def count_records(self, model: Base, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        حساب عدد السجلات
        """
        try:
            with self.get_session() as session:
                query = session.query(model)
                if filters:
                    for key, value in filters.items():
                        query = query.filter(getattr(model, key) == value)
                return query.count()
        except SQLAlchemyError as e:
            logger.error(f"خطأ في حساب السجلات: {str(e)}")
            raise
    
    def exists(self, model: Base, filters: Dict[str, Any]) -> bool:
        """
        التحقق من وجود سجل
        """
        try:
            with self.get_session() as session:
                query = session.query(model)
                for key, value in filters.items():
                    query = query.filter(getattr(model, key) == value)
                return session.query(query.exists()).scalar()
        except SQLAlchemyError as e:
            logger.error(f"خطأ في التحقق من الوجود: {str(e)}")
            raise

# إنشاء نسخة واحدة من مدير الجلسات
db_session = DatabaseSessionManager()

# تحميل متغيرات البيئة
load_dotenv()

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)