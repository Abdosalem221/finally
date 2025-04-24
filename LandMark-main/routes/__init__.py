
from flask import Blueprint

# Create blueprint
bp = Blueprint('main', __name__)

# Import routes
from .user_routes import register_user_routes

def register_routes(app):
    """Register all blueprints/routes with the app"""
    register_user_routes(bp)
    app.register_blueprint(bp)
