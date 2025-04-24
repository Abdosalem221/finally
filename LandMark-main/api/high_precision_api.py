"""
واجهة برمجة التطبيق للإشارات عالية الدقة
"""

import os
import sys
import json
import logging
import datetime
import random
from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('high_precision_api.log')
    ]
)

logger = logging.getLogger("high_precision_api")

# Create API blueprint
high_precision_bp = Blueprint('high_precision', __name__)

@high_precision_bp.route('/signals')
@login_required
def get_signals():
    """الحصول على الإشارات عالية الدقة"""
    try:
        from services.high_precision_signals import HighPrecisionSignals
        from models.market_models import Signal
        
        # الحصول على معلمات البحث
        symbol = request.args.get('symbol')
        timeframe = request.args.get('timeframe')
        limit = request.args.get('limit', 10, type=int)
        
        # الحصول على الإشارات
        signals = Signal.query.filter_by(
            is_high_precision=True,
            symbol=symbol if symbol else Signal.symbol,
            timeframe=timeframe if timeframe else Signal.timeframe
        ).order_by(Signal.timestamp.desc()).limit(limit).all()
        
        return jsonify({
            'status': 'success',
            'data': [signal.to_dict() for signal in signals]
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@high_precision_bp.route('/generate_signal', methods=['POST'])
@login_required
def generate_signal():
    """إنشاء إشارة جديدة"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        
        if not symbol or not timeframe:
            return jsonify({
                'status': 'error',
                'message': 'Symbol and timeframe are required'
            }), 400
        
        from services.high_precision_signals import HighPrecisionSignals
        generator = HighPrecisionSignals()
        signal = generator.generate_signal(symbol, timeframe)
        
        return jsonify({
            'status': 'success',
            'data': signal.to_dict()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@high_precision_bp.route('/verify_signal/<int:signal_id>', methods=['POST'])
@login_required
def verify_signal(signal_id):
    """التحقق من إشارة"""
    try:
        from models.market_models import Signal, SignalVerification
        from models.database import db
        
        signal = Signal.query.get_or_404(signal_id)
        verification = SignalVerification(signal_id=signal_id)
        
        db.session.add(verification)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'data': verification.to_dict()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@high_precision_bp.route('/active_verifications', methods=['GET'])
def get_active_verifications():
    """Get all active signal verifications"""
    try:
        # Get verifier
        verifier = current_app.signal_verifier if hasattr(current_app, 'signal_verifier') else None
        
        if not verifier:
            return jsonify({
                'success': False,
                'message': 'Signal verifier not available'
            })
        
        # Get active verifications
        verifications = verifier.get_all_active_verifications()
        
        # Format verification data
        formatted_verifications = []
        
        for v_id, verification in verifications.items():
            signal = verification.get('signal', {})
            
            # Calculate elapsed time
            start_time = verification.get('start_time')
            if start_time:
                if isinstance(start_time, str):
                    start_time = datetime.datetime.fromisoformat(start_time)
                
                now = datetime.datetime.now()
                elapsed_seconds = (now - start_time).total_seconds()
                
                if elapsed_seconds < 60:
                    elapsed_str = f"{int(elapsed_seconds)} seconds"
                elif elapsed_seconds < 3600:
                    elapsed_str = f"{int(elapsed_seconds / 60)} minutes"
                else:
                    elapsed_str = f"{int(elapsed_seconds / 3600)} hours"
            else:
                elapsed_str = "unknown"
            
            # Get verification details
            verification_progress = verification.get('verification_progress', {})
            
            formatted_verification = {
                'id': v_id,
                'signal': {
                    'currency_pair': signal.get('currency_pair', 'unknown'),
                    'signal_type': signal.get('signal_type', 'unknown'),
                    'entry_price': signal.get('entry_price'),
                    'take_profit': signal.get('take_profit'),
                    'stop_loss': signal.get('stop_loss'),
                    'timeframe': signal.get('timeframe', 'unknown'),
                    'strategy': signal.get('strategy', 'unknown')
                },
                'verification': {
                    'status': verification.get('status', 'pending'),
                    'progress': verification_progress.get('completed', 0) / verification_progress.get('total', 1) * 100,
                    'start_time': start_time.isoformat() if isinstance(start_time, datetime.datetime) else start_time,
                    'elapsed': elapsed_str,
                    'current_score': verification.get('current_score', 0) * 100
                }
            }
            
            formatted_verifications.append(formatted_verification)
        
        return jsonify({
            'success': True,
            'verifications': formatted_verifications
        })
    except Exception as e:
        logger.error(f"Error getting active verifications: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error getting active verifications: {str(e)}'
        })

@high_precision_bp.route('/verification_history', methods=['GET'])
def get_verification_history():
    """Get verification history"""
    try:
        # Get parameters
        limit = request.args.get('limit', 10, type=int)
        
        # Get verifier
        verifier = current_app.signal_verifier if hasattr(current_app, 'signal_verifier') else None
        
        if not verifier:
            return jsonify({
                'success': False,
                'message': 'Signal verifier not available'
            })
        
        # Get verification history
        history = verifier.get_verification_history(limit=limit)
        
        # Format history
        formatted_history = []
        
        for v_id, verification in history.items():
            signal = verification.get('signal', {})
            final_result = verification.get('final_result', {})
            
            # Format timestamps
            start_time = verification.get('start_time')
            if start_time and isinstance(start_time, str):
                start_time = datetime.datetime.fromisoformat(start_time)
            
            end_time = final_result.get('end_time')
            if end_time and isinstance(end_time, str):
                end_time = datetime.datetime.fromisoformat(end_time)
            
            # Calculate elapsed time
            elapsed_str = "unknown"
            if start_time and end_time:
                elapsed_seconds = (end_time - start_time).total_seconds()
                
                if elapsed_seconds < 60:
                    elapsed_str = f"{int(elapsed_seconds)} seconds"
                elif elapsed_seconds < 3600:
                    elapsed_str = f"{int(elapsed_seconds / 60)} minutes"
                else:
                    elapsed_str = f"{int(elapsed_seconds / 3600)} hours"
            
            formatted_verification = {
                'id': v_id,
                'signal': {
                    'currency_pair': signal.get('currency_pair', 'unknown'),
                    'signal_type': signal.get('signal_type', 'unknown'),
                    'entry_price': signal.get('entry_price'),
                    'take_profit': signal.get('take_profit'),
                    'stop_loss': signal.get('stop_loss'),
                    'timeframe': signal.get('timeframe', 'unknown'),
                    'strategy': signal.get('strategy', 'unknown')
                },
                'verification': {
                    'status': final_result.get('final_status', 'unknown'),
                    'final_score': final_result.get('final_score', 0) * 100,
                    'start_time': start_time.isoformat() if isinstance(start_time, datetime.datetime) else start_time,
                    'end_time': end_time.isoformat() if isinstance(end_time, datetime.datetime) else end_time,
                    'elapsed': elapsed_str
                }
            }
            
            formatted_history.append(formatted_verification)
        
        return jsonify({
            'success': True,
            'history': formatted_history
        })
    except Exception as e:
        logger.error(f"Error getting verification history: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error getting verification history: {str(e)}'
        })

@high_precision_bp.route('/enhancement_details', methods=['GET'])
def get_enhancement_details():
    """Get details of high precision signal enhancements"""
    try:
        # Get enhancer
        enhancer = current_app.enhancer if hasattr(current_app, 'enhancer') else None
        
        if not enhancer:
            return jsonify({
                'success': False,
                'message': 'Enhancer not available'
            })
        
        # Get enhancements
        enhancements = enhancer.enhancements
        
        # Group enhancements by category
        grouped_enhancements = {}
        
        for name, enhancement in enhancements.items():
            category = enhancement.get('category', 'other')
            
            if category not in grouped_enhancements:
                grouped_enhancements[category] = []
            
            grouped_enhancements[category].append({
                'name': name.replace('_', ' ').title(),
                'description': enhancement.get('description', ''),
                'weight': enhancement.get('weight', 0)
            })
        
        # Sort enhancements in each category by weight
        for category in grouped_enhancements:
            grouped_enhancements[category].sort(key=lambda x: x['weight'], reverse=True)
        
        return jsonify({
            'success': True,
            'enhancements': grouped_enhancements
        })
    except Exception as e:
        logger.error(f"Error getting enhancement details: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error getting enhancement details: {str(e)}'
        })

@high_precision_bp.route('/strategy_details', methods=['GET'])
def get_strategy_details():
    """Get details of trading strategies"""
    try:
        # Get strategies
        strategies = current_app.strategies if hasattr(current_app, 'strategies') else None
        
        if not strategies:
            return jsonify({
                'success': False,
                'message': 'Strategies not available'
            })
        
        # Get strategy details
        strategy_details = {}
        
        for name, strategy in strategies.strategies.items():
            category = strategy.get('category', 'other')
            
            if category not in strategy_details:
                strategy_details[category] = []
            
            strategy_details[category].append({
                'name': name.replace('_', ' ').title(),
                'description': strategy.get('description', ''),
                'success_rate': strategy.get('success_rate', 0) * 100,
                'preferred_timeframes': strategy.get('preferred_timeframes', []),
                'preferred_market_types': strategy.get('preferred_market_types', [])
            })
        
        # Sort strategies in each category by success rate
        for category in strategy_details:
            strategy_details[category].sort(key=lambda x: x['success_rate'], reverse=True)
        
        return jsonify({
            'success': True,
            'strategies': strategy_details
        })
    except Exception as e:
        logger.error(f"Error getting strategy details: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error getting strategy details: {str(e)}'
        })

@high_precision_bp.route('/performance_metrics', methods=['GET'])
def get_performance_metrics():
    """Get performance metrics for high precision signals"""
    try:
        # Get verifier
        verifier = current_app.signal_verifier if hasattr(current_app, 'signal_verifier') else None
        
        if not verifier:
            return jsonify({
                'success': False,
                'message': 'Signal verifier not available'
            })
        
        # Get verification stats
        stats = verifier.verification_stats
        
        # Calculate metrics
        total_signals = stats.get('total_signals', 0)
        verified_signals = stats.get('verified_signals', 0)
        rejected_signals = stats.get('rejected_signals', 0)
        successful_signals = stats.get('successful_signals', 0)
        unsuccessful_signals = stats.get('unsuccessful_signals', 0)
        
        verification_rate = 0
        if total_signals > 0:
            verification_rate = verified_signals / total_signals * 100
        
        success_rate = 0
        if verified_signals > 0:
            success_rate = successful_signals / verified_signals * 100
        
        # Return metrics
        return jsonify({
            'success': True,
            'metrics': {
                'total_signals': total_signals,
                'verified_signals': verified_signals,
                'rejected_signals': rejected_signals,
                'successful_signals': successful_signals,
                'unsuccessful_signals': unsuccessful_signals,
                'verification_rate': round(verification_rate, 1),
                'success_rate': round(success_rate, 1),
                'false_positives_prevented': stats.get('false_positives_prevented', 0),
                'average_verification_time': stats.get('average_verification_time', 0)
            }
        })
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error getting performance metrics: {str(e)}'
        })

# Function to initialize API
def init_high_precision_api(app):
    """Initialize high precision API and register blueprint"""
    app.register_blueprint(high_precision_bp, url_prefix='/api/high_precision')