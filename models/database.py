"""
إعداد قاعدة البيانات الرئيسية للمشروع
"""

import os
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from flask_migrate import Migrate
from sqlalchemy import MetaData
from flask_login import LoginManager, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

# إنشاء فئة أساسية للنماذج
class Base(DeclarativeBase):
    pass

# Use a custom naming convention for constraints
convention = {
    "ix": 'ix_%(column_0_label)s',
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

from app import db, migrate

# Flask-Login setup
login_manager = LoginManager()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return '<User %r>' % self.username




# Placeholder routes (replace with your actual routes)
from flask import Flask, render_template, request, redirect, url_for
from flask_login import login_user, logout_user, login_required, current_user