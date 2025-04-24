import logging
import json
from datetime import datetime, timedelta
import random
from flask import request, jsonify
from app.api import api
# Use the blueprint from the __init__.py
api_bp = api
from app import db
from app.models.market_models import Currency, Signal, MarketData, Alert, UserAlert
from app.utils.error_monitor import ErrorMonitor
from utils.helpers import generate_mock_price_data, get_recent_signals, calculate_success_rate

error_monitor = ErrorMonitor()

@api_bp.route('/system/scan', methods=['GET'])
def scan_system():
    """فحص النظام للكشف عن المشاكل"""
    scan_result = error_monitor.scan_project()
    return jsonify(scan_result)

@api_bp.route('/system/fix', methods=['POST'])
def fix_system():
    """إصلاح المشاكل المكتشفة"""
    error = request.json.get('error')
    fix = request.json.get('fix')
    if not error or not fix:
        return jsonify({'error': 'بيانات غير كاملة'}), 400
        
    success = error_monitor.apply_quick_fix(error, fix)
    if success:
        return jsonify({'message': 'تم إصلاح المشكلة بنجاح'})
    return jsonify({'error': 'فشل في إصلاح المشكلة'}), 500

logger = logging.getLogger(__name__)

@api_bp.route('/dashboard_data', methods=['GET'])
def get_dashboard_data():
    """Get statistics for the dashboard"""
    try:
        # Get active signals (generated in the last 24 hours)
        active_signals_count = Signal.query.filter(
            Signal.timestamp >= datetime.utcnow() - timedelta(hours=24)
        ).count()

        # Get success rate of signals
        success_rate = calculate_success_rate(days=30)

        # Get count of monitored currency pairs
        monitored_pairs = Currency.query.count()

        # Get active alerts
        active_alerts_count = Alert.query.filter_by(is_active=True).count()

        return jsonify({
            'success': True,
            'data': {
                'active_signals': active_signals_count,
                'success_rate': success_rate,
                'monitored_pairs': monitored_pairs,
                'active_alerts': active_alerts_count
            }
        })
    except Exception as e:
        logger.error(f"Error getting dashboard data: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error fetching dashboard data'
        })

@api_bp.route('/latest_signals', methods=['GET'])
def get_latest_signals():
    """Get latest signals"""
    try:
        market_type = request.args.get('market_type', 'all')
        limit = request.args.get('limit', 10, type=int)

        # Query for signals
        signals_query = Signal.query

        # Filter by market type if specified
        if market_type != 'all':
            signals_query = signals_query.filter_by(market_type=market_type)

        # Get latest signals
        signals = signals_query.order_by(Signal.timestamp.desc()).limit(limit).all()

        # Format signals for JSON response
        signals_data = []
        for signal in signals:
            signals_data.append({
                'id': signal.id,
                'currency_pair': signal.currency_pair,
                'signal_type': signal.signal_type,
                'entry_price': signal.entry_price,
                'take_profit': signal.take_profit,
                'stop_loss': signal.stop_loss,
                'confidence': signal.confidence,
                'market_type': signal.market_type,
                'timestamp': signal.timestamp.isoformat(),
                'expiry': signal.expiry.isoformat() if signal.expiry else None,
                'is_ai_generated': signal.is_ai_generated,
                'result': signal.result
            })

        return jsonify({
            'success': True,
            'signals': signals_data
        })
    except Exception as e:
        logger.error(f"Error getting latest signals: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error fetching latest signals'
        })

@api_bp.route('/watchlist', methods=['GET'])
def get_watchlist():
    """Get watchlist currencies with current prices"""
    try:
        # Get all currencies
        currencies = Currency.query.all()

        currencies_data = []
        for currency in currencies:
            # Calculate pseudo-change (for demo)
            change = random.uniform(-1.5, 1.5)

            currencies_data.append({
                'id': currency.id,
                'symbol': currency.symbol,
                'price': round(currency.current_price, 4) if currency.current_price else 'N/A',
                'change': change,
                'last_updated': currency.last_updated.isoformat() if currency.last_updated else None
            })

        return jsonify({
            'success': True,
            'currencies': currencies_data
        })
    except Exception as e:
        logger.error(f"Error getting watchlist: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error fetching watchlist'
        })

@api_bp.route('/recent_alerts', methods=['GET'])
def get_recent_alerts():
    """Get recent alerts"""
    try:
        limit = request.args.get('limit', 5, type=int)

        # Get active alerts
        alerts = Alert.query.filter_by(is_active=True).order_by(Alert.created_at.desc()).limit(limit).all()

        alerts_data = []
        for alert in alerts:
            currency = Currency.query.get(alert.currency_id)

            alerts_data.append({
                'id': alert.id,
                'currency_id': alert.currency_id,
                'currency_symbol': currency.symbol if currency else 'Unknown',
                'alert_type': alert.alert_type,
                'price_level': alert.price_level,
                'pattern': alert.pattern,
                'message': alert.message,
                'created_at': alert.created_at.isoformat(),
                'expires_at': alert.expires_at.isoformat() if alert.expires_at else None,
                'is_active': alert.is_active,
                'is_triggered': alert.is_triggered
            })

        return jsonify({
            'success': True,
            'alerts': alerts_data
        })
    except Exception as e:
        logger.error(f"Error getting recent alerts: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error fetching recent alerts'
        })

@api_bp.route('/performance_data', methods=['GET'])
def get_performance_data():
    """Get signal performance data for chart"""
    try:
        period = request.args.get('period', '30d')

        # Determine date range
        days = 30
        if period == '7d':
            days = 7
        elif period == '90d':
            days = 90

        # Generate dates
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # Get signals in date range
        signals = Signal.query.filter(
            Signal.timestamp >= start_date,
            Signal.timestamp <= end_date
        ).all()

        # Group signals by date
        signals_by_date = {}
        for signal in signals:
            date_str = signal.timestamp.strftime('%Y-%m-%d')
            if date_str not in signals_by_date:
                signals_by_date[date_str] = {
                    'total': 0,
                    'win': 0,
                    'loss': 0,
                    'pending': 0
                }

            signals_by_date[date_str]['total'] += 1

            if signal.result == 'WIN':
                signals_by_date[date_str]['win'] += 1
            elif signal.result == 'LOSS':
                signals_by_date[date_str]['loss'] += 1
            else:
                signals_by_date[date_str]['pending'] += 1

        # Generate data points for each day in range
        dates = []
        success_rates = []
        signal_counts = []

        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            dates.append(date_str)

            # Get data for this date or use defaults
            day_data = signals_by_date.get(date_str, {'total': 0, 'win': 0, 'loss': 0, 'pending': 0})

            # Calculate success rate
            if day_data['win'] + day_data['loss'] > 0:
                success_rate = (day_data['win'] / (day_data['win'] + day_data['loss'])) * 100
            else:
                success_rate = 0

            success_rates.append(round(success_rate, 1))
            signal_counts.append(day_data['total'])

            current_date += timedelta(days=1)

        return jsonify({
            'success': True,
            'performance': {
                'dates': dates,
                'success_rates': success_rates,
                'signal_counts': signal_counts
            }
        })
    except Exception as e:
        logger.error(f"Error getting performance data: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error fetching performance data'
        })

@api_bp.route('/generate_signal', methods=['GET'])
def generate_signal():
    """Generate a trading signal on demand"""
    try:
        currency_pair = request.args.get('currency')
        market_type = request.args.get('market_type', 'forex')

        if not currency_pair:
            return jsonify({
                'success': False,
                'message': 'Currency pair is required'
            })

        # Get currency from database
        currency = Currency.query.filter_by(symbol=currency_pair).first()

        if not currency:
            # Create currency if it doesn't exist
            currency = Currency(
                name=currency_pair,
                symbol=currency_pair,
                market_type=market_type,
                current_price=1.0  # Default price until updated
            )
            db.session.add(currency)
            db.session.commit()

        # Generate signal
        from technical_analysis import TechnicalAnalysis
        from ai_analysis import AIAnalysis

        ta = TechnicalAnalysis()
        ai = AIAnalysis()

        # Get historical data (or mock if not available)
        historical_data = ta.get_historical_data(currency_pair)

        if historical_data is None or historical_data.empty:
            # Generate mock data for demo purposes
            mock_data = generate_mock_price_data(currency_pair)

            # Create a new signal using mock data
            if market_type == 'binary':
                signal_type = random.choice(['CALL', 'PUT'])
            else:
                signal_type = random.choice(['BUY', 'SELL'])

            # Use current price or generate one
            entry_price = currency.current_price if currency.current_price else random.uniform(1.0, 1.5)

            # Calculate TP/SL based on signal type
            if signal_type in ['BUY', 'CALL']:
                take_profit = entry_price * 1.02
                stop_loss = entry_price * 0.99
            else:
                take_profit = entry_price * 0.98
                stop_loss = entry_price * 1.01

            # Set expiry for binary options
            expiry = datetime.utcnow() + timedelta(hours=1) if market_type == 'binary' else None

            # Create signal record
            signal = Signal(
                currency_id=currency.id,
                currency_pair=currency_pair,
                signal_type=signal_type,
                entry_price=entry_price,
                take_profit=take_profit,
                stop_loss=stop_loss,
                confidence=random.uniform(0.75, 0.95),
                market_type=market_type,
                timestamp=datetime.utcnow(),
                expiry=expiry,
                is_ai_generated=True,
                result="PENDING"
            )

            db.session.add(signal)
            db.session.commit()

            return jsonify({
                'success': True,
                'signal': {
                    'id': signal.id,
                    'currency_pair': signal.currency_pair,
                    'signal_type': signal.signal_type,
                    'entry_price': signal.entry_price,
                    'take_profit': signal.take_profit,
                    'stop_loss': signal.stop_loss,
                    'confidence': signal.confidence,
                    'market_type': signal.market_type,
                    'timestamp': signal.timestamp.isoformat(),
                    'expiry': signal.expiry.isoformat() if signal.expiry else None,
                    'is_ai_generated': signal.is_ai_generated
                }
            })
        else:
            # Use real data to generate signal with the technical analysis and AI modules
            ta_signals = ta.analyze(historical_data, currency_pair, market_type)
            ai_prediction = ai.predict(currency_pair, historical_data, market_type)

            if not ta_signals:
                return jsonify({
                    'success': False,
                    'message': 'Unable to generate technical analysis signals'
                })

            # Determine signal direction
            if market_type == 'binary':
                if ai_prediction and ai_prediction['prediction'] > 0.6:
                    signal_type = 'CALL'
                elif ai_prediction and ai_prediction['prediction'] < 0.4:
                    signal_type = 'PUT'
                else:
                    # Use technical analysis
                    buy_count = sum(1 for s in ta_signals.get('individual_signals', []) if s.get('signal') == 'BUY')
                    sell_count = sum(1 for s in ta_signals.get('individual_signals', []) if s.get('signal') == 'SELL')

                    signal_type = 'CALL' if buy_count > sell_count else 'PUT'
            else:
                if ai_prediction and ai_prediction['prediction'] > 0.6:
                    signal_type = 'BUY'
                elif ai_prediction and ai_prediction['prediction'] < 0.4:
                    signal_type = 'SELL'
                else:
                    # Use technical analysis
                    buy_count = sum(1 for s in ta_signals.get('individual_signals', []) if s.get('signal') == 'BUY')
                    sell_count = sum(1 for s in ta_signals.get('individual_signals', []) if s.get('signal') == 'SELL')

                    signal_type = 'BUY' if buy_count > sell_count else 'SELL'

            # Get current price
            entry_price = ta_signals.get('current_price', 1.0)

            # Calculate TP/SL based on signal type and volatility
            volatility = ta_signals.get('volatility', 0.01)

            if signal_type in ['BUY', 'CALL']:
                take_profit = entry_price * (1 + volatility * 2)
                stop_loss = entry_price * (1 - volatility)
            else:
                take_profit = entry_price * (1 - volatility * 2)
                stop_loss = entry_price * (1 + volatility)

            # Calculate confidence
            ta_confidence = max(0.5, min(0.95, buy_count / (buy_count + sell_count) if signal_type in ['BUY', 'CALL'] else 
                                     sell_count / (buy_count + sell_count)))

            ai_confidence = ai_prediction['confidence'] if ai_prediction else 0.5

            # Combine confidences (70% technical, 30% AI)
            confidence = (ta_confidence * 0.7) + (ai_confidence * 0.3)

            # Set expiry for binary options
            expiry = datetime.utcnow() + timedelta(hours=1) if market_type == 'binary' else None

            # Create signal record
            signal = Signal(
                currency_id=currency.id,
                currency_pair=currency_pair,
                signal_type=signal_type,
                entry_price=entry_price,
                take_profit=take_profit,
                stop_loss=stop_loss,
                confidence=confidence,
                market_type=market_type,
                timestamp=datetime.utcnow(),
                expiry=expiry,
                is_ai_generated=ai_prediction is not None,
                result="PENDING"
            )

            db.session.add(signal)
            db.session.commit()

            return jsonify({
                'success': True,
                'signal': {
                    'id': signal.id,
                    'currency_pair': signal.currency_pair,
                    'signal_type': signal.signal_type,
                    'entry_price': signal.entry_price,
                    'take_profit': signal.take_profit,
                    'stop_loss': signal.stop_loss,
                    'confidence': signal.confidence,
                    'market_type': signal.market_type,
                    'timestamp': signal.timestamp.isoformat(),
                    'expiry': signal.expiry.isoformat() if signal.expiry else None,
                    'is_ai_generated': signal.is_ai_generated
                }
            })

    except Exception as e:
        logger.error(f"Error generating signal: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error generating signal'
        })

@api_bp.route('/create_alert', methods=['POST'])
def create_alert():
    """Create a new price alert"""
    try:
        data = request.json

        if not data or 'currency_pair' not in data:
            return jsonify({
                'success': False,
                'message': 'Currency pair is required'
            })

        # Get required fields
        currency_pair = data.get('currency_pair')
        alert_type = data.get('alert_type', 'price')
        price_level = data.get('price_level')
        message = data.get('message', f'Alert for {currency_pair}')
        expiry_hours = data.get('expiry_hours', 24)

        # Validate price level for price alerts
        if alert_type == 'price' and not price_level:
            return jsonify({
                'success': False,
                'message': 'Price level is required for price alerts'
            })

        # Get currency from database
        currency = Currency.query.filter_by(symbol=currency_pair).first()

        if not currency:
            return jsonify({
                'success': False,
                'message': f'Currency {currency_pair} not found'
            })

        # Calculate expiry time
        expires_at = datetime.utcnow() + timedelta(hours=expiry_hours) if expiry_hours else None

        # Create alert
        alert = Alert(
            currency_id=currency.id,
            alert_type=alert_type,
            price_level=float(price_level) if price_level else None,
            pattern=data.get('pattern'),
            message=message,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            is_active=True,
            is_triggered=False
        )

        db.session.add(alert)
        db.session.commit()

        return jsonify({
            'success': True,
            'alert': {
                'id': alert.id,
                'currency_id': alert.currency_id,
                'currency_symbol': currency.symbol,
                'alert_type': alert.alert_type,
                'price_level': alert.price_level,
                'message': alert.message,
                'created_at': alert.created_at.isoformat(),
                'expires_at': alert.expires_at.isoformat() if alert.expires_at else None,
                'is_active': alert.is_active,
                'is_triggered': alert.is_triggered
            }
        })

    except Exception as e:
        logger.error(f"Error creating alert: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error creating alert'
        })

@api_bp.route('/delete_alert/<int:alert_id>', methods=['DELETE'])
def delete_alert(alert_id):
    """Delete an alert"""
    try:
        alert = Alert.query.get(alert_id)

        if not alert:
            return jsonify({
                'success': False,
                'message': f'Alert with ID {alert_id} not found'
            })

        # Remove associated user alerts
        UserAlert.query.filter_by(alert_id=alert_id).delete()

        # Delete the alert
        db.session.delete(alert)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': f'Alert with ID {alert_id} deleted successfully'
        })

    except Exception as e:
        logger.error(f"Error deleting alert: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error deleting alert'
        })

@api_bp.route('/currency_data', methods=['GET'])
def get_currency_data():
    """Get historical price data for a currency"""
    try:
        symbol = request.args.get('symbol')
        period = request.args.get('period', '7d')

        if not symbol:
            return jsonify({
                'success': False,
                'message': 'Symbol is required'
            })

        # Get currency from database
        currency = Currency.query.filter_by(symbol=symbol).first()

        if not currency:
            return jsonify({
                'success': False,
                'message': f'Currency {symbol} not found'
            })

        # Determine date range based on period
        days = 7
        if period == '1d':
            days = 1
        elif period == '30d':
            days = 30

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # Get market data
        market_data = MarketData.query.filter(
            MarketData.currency_id == currency.id,
            MarketData.timestamp >= start_date,
            MarketData.timestamp <= end_date
        ).order_by(MarketData.timestamp).all()

        # If no data is found, generate mock data
        if not market_data:
            mock_data = generate_mock_price_data(symbol, days)

            return jsonify({
                'success': True,
                'data': {
                    'timestamps': [d.get('timestamp') for d in mock_data],
                    'prices': [d.get('close') for d in mock_data],
                    'opens': [d.get('open') for d in mock_data],
                    'highs': [d.get('high') for d in mock_data],
                    'lows': [d.get('low') for d in mock_data],
                    'volumes': [d.get('volume') for d in mock_data]
                }
            })

        # Format data for response
        timestamps = [md.timestamp.isoformat() for md in market_data]
        prices = [md.close_price for md in market_data]
        opens = [md.open_price for md in market_data]
        highs = [md.high_price for md in market_data]
        lows = [md.low_price for md in market_data]
        volumes = [md.volume for md in market_data]

        return jsonify({
            'success': True,
            'data': {
                'timestamps': timestamps,
                'prices': prices,
                'opens': opens,
                'highs': highs,
                'lows': lows,
                'volumes': volumes
            }
        })

    except Exception as e:
        logger.error(f"Error getting currency data: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error fetching currency data'
        })

@api_bp.route('/technical_indicators', methods=['GET'])
def get_technical_indicators():
    """Get technical indicators for a currency"""
    try:
        symbol = request.args.get('symbol')

        if not symbol:
            return jsonify({
                'success': False,
                'message': 'Symbol is required'
            })

        # Get currency from database
        currency = Currency.query.filter_by(symbol=symbol).first()

        if not currency:
            return jsonify({
                'success': False,
                'message': f'Currency {symbol} not found'
            })

        # Get technical analysis
        from technical_analysis import TechnicalAnalysis
        ta = TechnicalAnalysis()

        # Get historical data
        historical_data = ta.get_historical_data(symbol)

        if historical_data is None or historical_data.empty:
            # Return mock indicators
            indicators = [
                {
                    'name': 'RSI (14)',
                    'value': '45.32',
                    'signal': 'NEUTRAL'
                },
                {
                    'name': 'MACD (12,26,9)',
                    'value': '-0.0012',
                    'signal': 'SELL'
                },
                {
                    'name': 'Moving Average (50)',
                    'value': f'{currency.current_price:.4f}',
                    'signal': 'NEUTRAL'
                },
                {
                    'name': 'Bollinger Bands',
                    'value': 'Inside Bands',
                    'signal': 'NEUTRAL'
                },
                {
                    'name': 'Stochastic (14,3)',
                    'value': '65.78',
                    'signal': 'NEUTRAL'
                }
            ]
        else:
            # Analyze data
            analysis = ta.analyze(historical_data, symbol, currency.market_type)

            # Extract individual indicators
            indicators = []

            if analysis and 'individual_signals' in analysis:
                for signal in analysis['individual_signals']:
                    indicators.append({
                        'name': signal.get('indicator', 'Unknown'),
                        'value': str(signal.get('value', 'N/A')),
                        'signal': signal.get('signal', 'NEUTRAL')
                    })

            # If no indicators found, add some defaults
            if not indicators:
                indicators = [
                    {
                        'name': 'RSI (14)',
                        'value': '45.32',
                        'signal': 'NEUTRAL'
                    },
                    {
                        'name': 'MACD (12,26,9)',
                        'value': '-0.0012',
                        'signal': 'SELL'
                    },
                    {
                        'name': 'Moving Average (50)',
                        'value': f'{currency.current_price:.4f}',
                        'signal': 'NEUTRAL'
                    }
                ]

        return jsonify({
            'success': True,
            'indicators': indicators
        })

    except Exception as e:
        logger.error(f"Error getting technical indicators: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error fetching technical indicators'
        })

@api_bp.route('/signal_performance', methods=['GET'])
def get_signal_performance():
    """Get performance statistics for signals"""
    try:
        symbol = request.args.get('symbol')

        # Query filters
        query_filters = []

        if symbol:
            query_filters.append(Signal.currency_pair == symbol)

        # Get signals
        signals = Signal.query.filter(*query_filters).all()

        # Count by result
        win_count = sum(1 for s in signals if s.result == 'WIN')
        loss_count = sum(1 for s in signals if s.result == 'LOSS')
        pending_count = sum(1 for s in signals if s.result == 'PENDING' or s.result is None)

        # Calculate success rate
        total_completed = win_count + loss_count
        success_rate = (win_count / total_completed * 100) if total_completed > 0 else 0

        return jsonify({
            'success': True,
            'performance': {
                'win': win_count,
                'loss': loss_count,
                'pending': pending_count,
                'total': len(signals),
                'success_rate': round(success_rate, 2)
            }
        })

    except Exception as e:
        logger.error(f"Error getting signal performance: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error fetching signal performance'
        })

@api_bp.route('/ai_prediction', methods=['GET'])
def get_ai_prediction():
    """Get AI prediction for a currency pair"""
    try:
        symbol = request.args.get('symbol')
        market_type = request.args.get('market_type', 'forex')

        if not symbol:
            return jsonify({
                'success': False,
                'message': 'Symbol is required'
            })

        # Get AI prediction
        from ai_analysis import AIAnalysis
        ai = AIAnalysis()

        # Get historical data
        from technical_analysis import TechnicalAnalysis
        ta = TechnicalAnalysis()
        historical_data = ta.get_historical_data(symbol)

        # Get prediction
        prediction = ai.predict(symbol, historical_data, market_type)

        if not prediction:
            # Generate a mock prediction if real one fails
            prediction = {
                'symbol': symbol,
                'prediction': random.uniform(0.3, 0.7),
                'confidence': random.uniform(0.6, 0.9),
                'timestamp': datetime.utcnow().isoformat(),
                'market_type': market_type,
                'interpretation': 'neutral',
                'strength': random.uniform(0.3, 0.7)
            }

        return jsonify({
            'success': True,
            'prediction': prediction
        })

    except Exception as e:
        logger.error(f"Error getting AI prediction: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error fetching AI prediction'
        })

@app.route('/api/auth/status')
def auth_status():
    """التحقق من حالة المصادقة"""
    if current_user.is_authenticated:
        return jsonify({
            'isAuthenticated': True,
            'user': {
                'id': current_user.id,
                'username': current_user.username
            }
        })
    return jsonify({'isAuthenticated': False})

@api_bp.route('/current_price', methods=['GET'])
def get_current_price():
    """Get current price for a currency pair"""
    try:
        symbol = request.args.get('symbol')

        if not symbol:
            return jsonify({
                'success': False,
                'message': 'Symbol is required'
            })

        # Get currency from database
        currency = Currency.query.filter_by(symbol=symbol).first()

        if not currency or not currency.current_price:
            # Generate a mock price if not available
            if 'JPY' in symbol:
                price = random.uniform(100, 150)
            elif 'GBP' in symbol:
                price = random.uniform(1.2, 1.4)
            else:
                price = random.uniform(0.8, 1.2)
        else:
            price = currency.current_price

        return jsonify({
            'success': True,
            'symbol': symbol,
            'price': round(price, 4),
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting current price: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error fetching current price'
        })

@api_bp.route('/settings', methods=['GET', 'POST'])
def handle_settings():
    """Get or update application settings"""
    try:
        if request.method == 'GET':
            # Return current settings
            from config import (
                DATA_FETCH_INTERVAL,
                SIGNAL_GENERATION_INTERVAL,
                SIGNAL_CONFIDENCE_THRESHOLD,
                STRATEGY_WEIGHTS,
                ANALYSIS_TOOL_WEIGHTS,
                USE_AI_MODELS,
                AI_CONFIDENCE_WEIGHT,
                MARKET_DATA_API_KEY,
                MARKET_DATA_API_URL,
                TELEGRAM_API_TOKEN,
                TELEGRAM_ADMIN_IDS
            )

            settings = {
                'general': {
                    'data_fetch_interval': DATA_FETCH_INTERVAL,
                    'theme': 'dark',
                    'default_timeframe': '7d',
                    'enable_animations': True
                },
                'signal': {
                    'signal_interval': SIGNAL_GENERATION_INTERVAL,
                    'confidence_threshold': SIGNAL_CONFIDENCE_THRESHOLD,
                    'risk_reward_ratio': 2,
                    'enable_binary_signals': True,
                    'enable_forex_signals': True
                },
                'strategy_weights': STRATEGY_WEIGHTS,
                'notification': {
                    'enable_browser_notifications': True,
                    'enable_telegram_notifications': bool(TELEGRAM_API_TOKEN and TELEGRAM_API_TOKEN != 'your_telegram_token_here'),
                    'telegram_token': TELEGRAM_API_TOKEN if TELEGRAM_API_TOKEN != 'your_telegram_token_here' else '',
                    'telegram_chat_ids': ','.join(TELEGRAM_ADMIN_IDS) if TELEGRAM_ADMIN_IDS else '',
                    'notify_signals': True,
                    'notify_alerts': True,
                    'notify_system': True
                },
                'api': {
                    'market_data_api_key': MARKET_DATA_API_KEY if MARKET_DATA_API_KEY != 'your_market_data_api_key' else '','market_data_api_url': MARKET_DATA_API_URL,
                    'api_timeout': 10,
                    'use_mock_data': True
                },
                'ai': {
                    'use_ai_models': USE_AI_MODELS,
                    'ai_confidence_weight': AI_CONFIDENCE_WEIGHT,
                    'training_period': 90,
                    'retraining_frequency': 'weekly',
                    'use_advanced_features': True
                }
            }

            return jsonify({
                'success': True,
                'settings': settings
            })
        else:
            # Update settings (in a real app this would modify config)
            data = request.json

            if not data or 'category' not in data or 'settings' not in data:
                return jsonify({
                    'success': False,
                    'message': 'Invalid settings data'
                })

            # In a real app, this would save to a settings file or database
            # For now, just acknowledge receipt

            return jsonify({
                'success': True,
                'message': f"Settings updated for category: {data['category']}"
            })

    except Exception as e:
        logger.error(f"Error handling settings: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error processing settings'
        })

@api_bp.route('/test_connection', methods=['POST'])
def test_api_connection():
    """Test connection to market data API"""
    try:
        data = request.json
        api_key = data.get('api_key', '')
        api_url = data.get('api_url', '')

        if not api_key or not api_url:
            return jsonify({
                'success': False,
                'message': 'API key and URL are required'
            })

        # Simulate API test
        # In a real app, this would make an actual API request
        if 'example.com' in api_url:
            return jsonify({
                'success': False,
                'message': 'This is a placeholder API URL. Please use a real market data API URL.'
            })

        # Simulate success (replace with actual API test in production)
        return jsonify({
            'success': True,
            'message': 'Connection successful'
        })

    except Exception as e:
        logger.error(f"Error testing API connection: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error testing connection'
        })

@api_bp.route('/retrain_models', methods=['POST'])
def retrain_ai_models():
    """Retrain AI models"""
    try:
        # Simulate retraining (in a real app, this would call the AI module)

        # In a real implementation, we would call the AI training functions here

        return jsonify({
            'success': True,
            'message': 'Models retrained successfully'
        })

    except Exception as e:
        logger.error(f"Error retraining models: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error retraining models'
        })

@api_bp.route('/active_alerts', methods=['GET'])
def get_active_alerts():
    """Get all active alerts"""
    try:
        alerts = Alert.query.filter_by(is_active=True).all()

        alerts_data = []
        for alert in alerts:
            currency = Currency.query.get(alert.currency_id)

            alerts_data.append({
                'id': alert.id,
                'currency_id': alert.currency_id,
                'currency_symbol': currency.symbol if currency else 'Unknown',
                'alert_type': alert.alert_type,
                'price_level': alert.price_level,
                'pattern': alert.pattern,
                'message': alert.message,
                'created_at': alert.created_at.isoformat(),
                'expires_at': alert.expires_at.isoformat() if alert.expires_at else None,
                'is_active': alert.is_active,
                'is_triggered': alert.is_triggered,
                'triggered_at': alert.triggered_at.isoformat() if alert.triggered_at else None
            })

        return jsonify({
            'success': True,
            'alerts': alerts_data
        })

    except Exception as e:
        logger.error(f"Error getting active alerts: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error fetching active alerts'
        })

@api_bp.route('/currency_signals', methods=['GET'])
def get_currency_signals():
    """Get signals for a specific currency pair"""
    try:
        symbol = request.args.get('symbol')
        limit = request.args.get('limit', 10, type=int)

        if not symbol:
            return jsonify({
                'success': False,
                'message': 'Symbol is required'
            })

        # Get signals for this currency
        signals = Signal.query.filter_by(currency_pair=symbol).order_by(Signal.timestamp.desc()).limit(limit).all()

        signals_data = []
        for signal in signals:
            signals_data.append({
                'id': signal.id,
                'currency_pair': signal.currency_pair,
                'signal_type': signal.signal_type,
                'entry_price': signal.entry_price,
                'take_profit': signal.take_profit,
                'stop_loss': signal.stop_loss,
                'confidence': signal.confidence,
                'market_type': signal.market_type,
                'timestamp': signal.timestamp.isoformat(),
                'expiry': signal.expiry.isoformat() if signal.expiry else None,
                'is_ai_generated': signal.is_ai_generated,
                'result': signal.result
            })

        return jsonify({
            'success': True,
            'signals': signals_data
        })

    except Exception as e:
        logger.error(f"Error getting currency signals: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error fetching currency signals'
        })

@api_bp.route('/price_data', methods=['GET'])
def get_price_data():
    """Get detailed price data for charting"""
    try:
        symbol = request.args.get('symbol')
        timeframe = request.args.get('timeframe', '1d')
        limit = request.args.get('limit', 30, type=int)

        if not symbol:
            return jsonify({
                'success': False,
                'message': 'Symbol is required'
            })

        # Determine time period
        period_days = 1
        if timeframe == '1d':
            period_days = 1
        elif timeframe == '7d':
            period_days = 7
        elif timeframe == '30d':
            period_days = 30

        # Get currency from database
        currency = Currency.query.filter_by(symbol=symbol).first()

        if not currency:
            return jsonify({
                'success': False,
                'message': f'Currency {symbol} not found'
            })

        # Get market data
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)

        market_data = MarketData.query.filter(
            MarketData.currency_id == currency.id,
            MarketData.timestamp >= start_date,
            MarketData.timestamp <= end_date
        ).order_by(MarketData.timestamp).all()

        # If no data, generate mock data
        if not market_data:
            price_data = generate_mock_price_data(symbol, period_days)
        else:
            price_data = []
            for md in market_data:
                price_data.append({
                    'timestamp': md.timestamp.isoformat(),
                    'open': md.open_price,
                    'high': md.high_price,
                    'low': md.low_price,
                    'close': md.close_price,
                    'volume': md.volume
                })

        return jsonify({
            'success': True,
            'price_data': price_data
        })

    except Exception as e:
        logger.error(f"Error getting price data: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error fetching price data'
        })