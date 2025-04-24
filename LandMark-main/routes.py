"""
تسجيل مسارات التطبيق
"""

from flask import render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import login_required, login_user, logout_user, current_user
from datetime import datetime
from app.models.market_models import Currency, Signal, Alert
from app.models.database import User, db
from flask import Blueprint, jsonify, render_template
from app.utils.error_monitor import ErrorMonitor # Assuming this file and class exist

error_monitor = ErrorMonitor() # Instantiate the error monitor


def register_routes(app):
    @app.route('/')
    def index():
        try:
            return render_template('index.html')
        except Exception as e:
            error_monitor.log_error(e)
            return render_template('errors/500.html'), 500

    @app.route('/dashboard')
    def dashboard():
        try:
            stats = {
                'currencies_count': Currency.query.filter_by(is_active=True).count(),
                'signals_today': Signal.query.filter(Signal.timestamp >= datetime.now().date()).count(),
                'active_alerts': Alert.query.filter_by(is_active=True, is_triggered=False).count()
            }
            return render_template('dashboard.html', stats=stats)
        except Exception as e:
            error_monitor.log_error(e)
            return render_template('errors/500.html'), 500

    @app.route('/system/health')
    def system_health():
        return jsonify({
            'status': 'healthy',
            'errors': error_monitor.get_recent_errors()
        })

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        """صفحة التسجيل"""
        if request.method == 'POST':
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')

            existing_user = User.query.filter((User.email == email) | (User.username == username)).first()
            if existing_user:
                flash('البريد الإلكتروني أو اسم المستخدم موجود بالفعل', 'danger')
                return render_template('register.html')

            new_user = User(username=username, email=email)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()

            flash('تم إنشاء الحساب بنجاح', 'success')
            return redirect(url_for('dashboard'))

        return render_template('register.html')

    @app.route('/errors') # Added route for error display
    def display_errors():
        errors = error_monitor.get_errors() # Assumes get_errors() method exists in ErrorMonitor
        return render_template('errors.html', errors=errors) #Assumes errors.html template exists

    @app.errorhandler(404)
    def page_not_found(e):
        """معالجة خطأ 404"""
        error_monitor.log_error(e) # Log the error
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def server_error(e):
        """معالجة خطأ 500"""
        error_monitor.log_error(e) # Log the error
        return render_template('errors/500.html'), 500