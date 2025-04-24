"""
إدارة ترحيل قاعدة البيانات
"""

import os
import logging
import subprocess
from typing import Optional, List
from datetime import datetime
from alembic.config import Config
from alembic import command
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine
from dotenv import load_dotenv

# تحميل متغيرات البيئة
load_dotenv()

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MigrationManager:
    """
    مدير ترحيل قاعدة البيانات
    """
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL", "sqlite:///app/database/trading.db")
        self.engine = create_engine(self.db_url)
        self.alembic_cfg = Config("alembic.ini")
        self.alembic_cfg.set_main_option("sqlalchemy.url", self.db_url)
    
    def init_migrations(self) -> None:
        """
        تهيئة نظام الترحيل
        """
        try:
            if not os.path.exists("alembic"):
                command.init(self.alembic_cfg, "alembic")
                logger.info("تم تهيئة نظام الترحيل بنجاح")
            else:
                logger.info("نظام الترحيل موجود بالفعل")
        except Exception as e:
            logger.error(f"خطأ في تهيئة نظام الترحيل: {str(e)}")
            raise
    
    def create_migration(self, message: str) -> str:
        """
        إنشاء ترحيل جديد
        
        Args:
            message: وصف الترحيل
            
        Returns:
            str: معرف الترحيل
        """
        try:
            revision = command.revision(self.alembic_cfg, message=message, autogenerate=True)
            logger.info(f"تم إنشاء الترحيل بنجاح: {revision.revision}")
            return revision.revision
        except Exception as e:
            logger.error(f"خطأ في إنشاء الترحيل: {str(e)}")
            raise
    
    def upgrade(self, revision: str = "head") -> None:
        """
        تطبيق الترحيلات
        
        Args:
            revision: معرف الترحيل الهدف
        """
        try:
            command.upgrade(self.alembic_cfg, revision)
            logger.info(f"تم تطبيق الترحيلات بنجاح حتى: {revision}")
        except Exception as e:
            logger.error(f"خطأ في تطبيق الترحيلات: {str(e)}")
            raise
    
    def downgrade(self, revision: str) -> None:
        """
        التراجع عن الترحيلات
        
        Args:
            revision: معرف الترحيل الهدف
        """
        try:
            command.downgrade(self.alembic_cfg, revision)
            logger.info(f"تم التراجع عن الترحيلات بنجاح إلى: {revision}")
        except Exception as e:
            logger.error(f"خطأ في التراجع عن الترحيلات: {str(e)}")
            raise
    
    def current(self) -> Optional[str]:
        """
        الحصول على الترحيل الحالي
        
        Returns:
            Optional[str]: معرف الترحيل الحالي
        """
        try:
            with self.engine.connect() as connection:
                context = MigrationContext.configure(connection)
                current_rev = context.get_current_revision()
                logger.info(f"الترحيل الحالي: {current_rev}")
                return current_rev
        except Exception as e:
            logger.error(f"خطأ في الحصول على الترحيل الحالي: {str(e)}")
            raise
    
    def history(self) -> List[str]:
        """
        الحصول على تاريخ الترحيلات
        
        Returns:
            List[str]: قائمة معرفات الترحيلات
        """
        try:
            script = ScriptDirectory.from_config(self.alembic_cfg)
            revisions = [rev.revision for rev in script.walk_revisions()]
            logger.info(f"تاريخ الترحيلات: {revisions}")
            return revisions
        except Exception as e:
            logger.error(f"خطأ في الحصول على تاريخ الترحيلات: {str(e)}")
            raise
    
    def stamp(self, revision: str) -> None:
        """
        تحديث سجل الترحيل
        
        Args:
            revision: معرف الترحيل
        """
        try:
            command.stamp(self.alembic_cfg, revision)
            logger.info(f"تم تحديث سجل الترحيل إلى: {revision}")
        except Exception as e:
            logger.error(f"خطأ في تحديث سجل الترحيل: {str(e)}")
            raise
    
    def check_migrations(self) -> bool:
        """
        التحقق من وجود ترحيلات معلقة
        
        Returns:
            bool: وجود ترحيلات معلقة
        """
        try:
            with self.engine.connect() as connection:
                context = MigrationContext.configure(connection)
                current_rev = context.get_current_revision()
                script = ScriptDirectory.from_config(self.alembic_cfg)
                head_rev = script.get_current_head()
                has_pending = current_rev != head_rev
                logger.info(f"يوجد ترحيلات معلقة: {has_pending}")
                return has_pending
        except Exception as e:
            logger.error(f"خطأ في التحقق من الترحيلات المعلقة: {str(e)}")
            raise
    
    def create_backup(self) -> str:
        """
        إنشاء نسخة احتياطية من قاعدة البيانات
        
        Returns:
            str: مسار النسخة الاحتياطية
        """
        try:
            backup_dir = 'app/database/backups'
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = f"{backup_dir}/backup_{timestamp}.sql"
            
            db_name = os.getenv('DB_NAME', 'algotrader_db')
            db_user = os.getenv('DB_USER', 'postgres')
            db_host = os.getenv('DB_HOST', 'localhost')
            
            cmd = f"pg_dump -U {db_user} -h {db_host} {db_name} > {backup_file}"
            subprocess.run(cmd, shell=True, check=True)
            
            logger.info(f"تم إنشاء النسخة الاحتياطية بنجاح: {backup_file}")
            return backup_file
        except Exception as e:
            logger.error(f"خطأ في إنشاء النسخة الاحتياطية: {str(e)}")
            raise
    
    def restore_backup(self, backup_file: str) -> None:
        """
        استعادة قاعدة البيانات من نسخة احتياطية
        
        Args:
            backup_file: مسار النسخة الاحتياطية
        """
        try:
            if not os.path.exists(backup_file):
                raise FileNotFoundError(f"النسخة الاحتياطية غير موجودة: {backup_file}")
            
            db_name = os.getenv('DB_NAME', 'algotrader_db')
            db_user = os.getenv('DB_USER', 'postgres')
            db_host = os.getenv('DB_HOST', 'localhost')
            
            cmd = f"psql -U {db_user} -h {db_host} {db_name} < {backup_file}"
            subprocess.run(cmd, shell=True, check=True)
            
            logger.info(f"تم استعادة النسخة الاحتياطية بنجاح: {backup_file}")
        except Exception as e:
            logger.error(f"خطأ في استعادة النسخة الاحتياطية: {str(e)}")
            raise
    
    def cleanup_migrations(self) -> None:
        """
        تنظيف سجلات الترحيل
        """
        try:
            with self.engine.connect() as conn:
                conn.execute("DROP TABLE IF EXISTS alembic_version")
            logger.info("تم تنظيف سجلات الترحيل بنجاح")
        except Exception as e:
            logger.error(f"خطأ في تنظيف سجلات الترحيل: {str(e)}")
            raise

# إنشاء نسخة واحدة من مدير الترحيل
migration_manager = MigrationManager() 