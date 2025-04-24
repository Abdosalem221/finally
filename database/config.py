"""
Database configuration settings and utilities.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class DatabaseConfig:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = 'dbece72d6763bc23cc16bb055fbae1051b49366756661151fdc15646fc915679'  # تغيير هذا في الإنتاج
    
    @staticmethod
    def get_test_uri():
        return 'sqlite:///test.db'

# Create global database configuration instance
db_config = DatabaseConfig()