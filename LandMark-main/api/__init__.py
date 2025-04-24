"""
واجهة برمجة التطبيق (API)
"""

from flask import Blueprint

api = Blueprint('api', __name__)

# Import routes after creating the blueprint to avoid circular imports
from .analytics_api import analytics_bp
from .high_precision_api import high_precision_bp

# Register API blueprints
api.register_blueprint(analytics_bp, url_prefix='/analytics')
api.register_blueprint(high_precision_bp, url_prefix='/high_precision')

def init_api(app):
    """Initialize API and register blueprints with the app"""
    app.register_blueprint(api, url_prefix='/api')