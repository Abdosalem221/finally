
from functools import wraps
from flask import jsonify
from .logger import error_logger

class AppError(Exception):
    def __init__(self, message, status_code=500, payload=None):
        super().__init__()
        self.message = message
        self.status_code = status_code
        self.payload = payload

def handle_error(error):
    error_logger.error(f"Error: {error.message}")
    response = {
        "error": True,
        "message": error.message,
        "status": error.status_code
    }
    if error.payload:
        response["data"] = error.payload
    return jsonify(response), error.status_code

def error_handler(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except AppError as e:
            return handle_error(e)
        except Exception as e:
            error_logger.exception("Unexpected error")
            return handle_error(AppError(str(e)))
    return decorated_function
