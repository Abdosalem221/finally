"""
مسارات المستخدمين
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.database.models import User # Using absolute import
from app.database.db_session import db_session # Assuming this function exists
from app.schemas.user import UserCreate, UserUpdate, UserResponse

router = APIRouter()

# Mock user data for testing
mock_users = [
    {"id": "1", "name": "Test User 1", "email": "test1@example.com"},
    {"id": "2", "name": "Test User 2", "email": "test2@example.com"},
]

@router.post("/", response_model=UserResponse)
async def create_user(user: UserCreate):
    """
    إنشاء مستخدم جديد (مؤقتاً بدون حفظ في قاعدة البيانات)
    """
    mock_user = {"id": str(len(mock_users) + 1), "name": user.name, "email": user.email}
    mock_users.append(mock_user)
    return mock_user


@router.get("/", response_model=List[UserResponse])
async def get_users():
    """
    الحصول على جميع المستخدمين (بيانات وهمية)
    """
    return mock_users

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    """
    الحصول على مستخدم محدد (بيانات وهمية)
    """
    user = next((user for user in mock_users if user["id"] == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    return user

@router.put("/{user_id}", response_model=UserResponse)
async def update_user(user_id: str, user: UserUpdate):
    """
    تحديث بيانات مستخدم (مؤقتاً بدون حفظ في قاعدة البيانات)
    """
    user_index = next((i for i, user in enumerate(mock_users) if user["id"] == user_id), None)
    if user_index is None:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    updated_data = user.dict(exclude_unset=True)
    mock_users[user_index].update(updated_data)  #Update in place
    return mock_users[user_index]


@router.delete("/{user_id}")
async def delete_user(user_id: str):
    """
    حذف مستخدم (مؤقتاً بدون حذف من قاعدة البيانات)
    """
    user_index = next((i for i, user in enumerate(mock_users) if user["id"] == user_id), None)
    if user_index is None:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    del mock_users[user_index]
    return {"message": "تم حذف المستخدم بنجاح"}