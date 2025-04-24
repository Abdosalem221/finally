"""
نماذج بيانات المستخدم
"""

from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
from app.database.models import UserRole, UserStatus

class UserBase(BaseModel):
    """
    النموذج الأساسي للمستخدم
    """
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    role: UserRole = UserRole.USER
    status: UserStatus = UserStatus.ACTIVE

class UserCreate(UserBase):
    """
    نموذج إنشاء مستخدم جديد
    """
    password: str

class UserUpdate(BaseModel):
    """
    نموذج تحديث بيانات المستخدم
    """
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    role: Optional[UserRole] = None
    status: Optional[UserStatus] = None

class UserResponse(UserBase):
    """
    نموذج استجابة بيانات المستخدم
    """
    user_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True 