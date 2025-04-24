"""
خدمة إدارة وتتبع المحافظ الاستثمارية
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
from app.services.data_service import DataService
from app.services.risk_service import RiskService
from app.utils.helpers import format_currency, format_percentage, format_timestamp

class PortfolioService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_service = DataService()
        self.risk_service = RiskService()
        self.portfolios = {}
    
    def create_portfolio(self, user_id: str,
                        name: str,
                        initial_balance: float,
                        risk_level: str = 'medium',
                        description: Optional[str] = None) -> Dict:
        """
        إنشاء محفظة جديدة
        
        Args:
            user_id: معرف المستخدم
            name: اسم المحفظة
            initial_balance: الرصيد الأولي
            risk_level: مستوى المخاطرة
            description: وصف المحفظة
            
        Returns:
            Dict: تفاصيل المحفظة
        """
        try:
            portfolio_id = f"portfolio_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            portfolio = {
                'id': portfolio_id,
                'user_id': user_id,
                'name': name,
                'initial_balance': initial_balance,
                'current_balance': initial_balance,
                'risk_level': risk_level,
                'description': description,
                'positions': [],
                'transactions': [],
                'created_at': datetime.now(),
                'last_updated': datetime.now()
            }
            
            self.portfolios[portfolio_id] = portfolio
            
            return {
                'status': 'success',
                'portfolio_id': portfolio_id,
                'portfolio': portfolio
            }
            
        except Exception as e:
            self.logger.error(f"Error creating portfolio: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def add_position(self, portfolio_id: str,
                    symbol: str,
                    quantity: float,
                    entry_price: float,
                    stop_loss: Optional[float] = None,
                    take_profit: Optional[float] = None) -> Dict:
        """
        إضافة مركز جديد للمحفظة
        
        Args:
            portfolio_id: معرف المحفظة
            symbol: رمز العملة
            quantity: الكمية
            entry_price: سعر الدخول
            stop_loss: سعر وقف الخسارة
            take_profit: سعر جني الأرباح
            
        Returns:
            Dict: تفاصيل المركز
        """
        try:
            if portfolio_id not in self.portfolios:
                return {'status': 'error', 'message': 'المحفظة غير موجودة'}
            
            portfolio = self.portfolios[portfolio_id]
            
            # حساب قيمة المركز
            position_value = quantity * entry_price
            
            # التحقق من توفر الرصيد
            if position_value > portfolio['current_balance']:
                return {'status': 'error', 'message': 'رصيد غير كاف'}
            
            # حساب مستويات وقف الخسارة وجني الأرباح
            if stop_loss is None or take_profit is None:
                risk_analysis = self.risk_service.analyze_risk(symbol)
                if risk_analysis['status'] == 'success':
                    if stop_loss is None:
                        stop_loss = self.risk_service.get_stop_loss(
                            symbol,
                            entry_price,
                            'long'
                        )
                    if take_profit is None:
                        take_profit = self.risk_service.get_take_profit(
                            symbol,
                            entry_price,
                            stop_loss,
                            'long'
                        )
            
            # إنشاء المركز
            position_id = f"position_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            position = {
                'id': position_id,
                'symbol': symbol,
                'quantity': quantity,
                'entry_price': entry_price,
                'current_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now(),
                'status': 'open',
                'pnl': 0,
                'pnl_percentage': 0
            }
            
            # تحديث المحفظة
            portfolio['positions'].append(position)
            portfolio['current_balance'] -= position_value
            
            # تسجيل المعاملة
            transaction = {
                'id': f"transaction_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'type': 'open',
                'position_id': position_id,
                'symbol': symbol,
                'quantity': quantity,
                'price': entry_price,
                'value': position_value,
                'time': datetime.now()
            }
            portfolio['transactions'].append(transaction)
            
            portfolio['last_updated'] = datetime.now()
            
            return {
                'status': 'success',
                'position_id': position_id,
                'position': position
            }
            
        except Exception as e:
            self.logger.error(f"Error adding position: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def close_position(self, portfolio_id: str,
                      position_id: str,
                      exit_price: float) -> Dict:
        """
        إغلاق مركز
        
        Args:
            portfolio_id: معرف المحفظة
            position_id: معرف المركز
            exit_price: سعر الخروج
            
        Returns:
            Dict: تفاصيل الإغلاق
        """
        try:
            if portfolio_id not in self.portfolios:
                return {'status': 'error', 'message': 'المحفظة غير موجودة'}
            
            portfolio = self.portfolios[portfolio_id]
            
            # البحث عن المركز
            position = None
            for p in portfolio['positions']:
                if p['id'] == position_id:
                    position = p
                    break
            
            if position is None:
                return {'status': 'error', 'message': 'المركز غير موجود'}
            
            if position['status'] != 'open':
                return {'status': 'error', 'message': 'المركز مغلق بالفعل'}
            
            # حساب الربح/الخسارة
            pnl = (exit_price - position['entry_price']) * position['quantity']
            pnl_percentage = (exit_price - position['entry_price']) / position['entry_price'] * 100
            
            # تحديث المركز
            position['exit_price'] = exit_price
            position['exit_time'] = datetime.now()
            position['status'] = 'closed'
            position['pnl'] = pnl
            position['pnl_percentage'] = pnl_percentage
            
            # تحديث المحفظة
            portfolio['current_balance'] += position['quantity'] * exit_price
            
            # تسجيل المعاملة
            transaction = {
                'id': f"transaction_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'type': 'close',
                'position_id': position_id,
                'symbol': position['symbol'],
                'quantity': position['quantity'],
                'price': exit_price,
                'value': position['quantity'] * exit_price,
                'pnl': pnl,
                'pnl_percentage': pnl_percentage,
                'time': datetime.now()
            }
            portfolio['transactions'].append(transaction)
            
            portfolio['last_updated'] = datetime.now()
            
            return {
                'status': 'success',
                'position': position,
                'transaction': transaction
            }
            
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def update_positions(self, portfolio_id: str) -> Dict:
        """
        تحديث أسعار المراكز
        
        Args:
            portfolio_id: معرف المحفظة
            
        Returns:
            Dict: نتائج التحديث
        """
        try:
            if portfolio_id not in self.portfolios:
                return {'status': 'error', 'message': 'المحفظة غير موجودة'}
            
            portfolio = self.portfolios[portfolio_id]
            updated_positions = []
            
            for position in portfolio['positions']:
                if position['status'] == 'open':
                    # جلب السعر الحالي
                    market_data = self.data_service.fetch_market_data(
                        position['symbol'],
                        '1m',
                        1
                    )
                    
                    if market_data['status'] == 'error':
                        continue
                    
                    current_price = float(market_data['data'][-1]['close'])
                    
                    # تحديث المركز
                    position['current_price'] = current_price
                    position['pnl'] = (current_price - position['entry_price']) * position['quantity']
                    position['pnl_percentage'] = (current_price - position['entry_price']) / position['entry_price'] * 100
                    
                    # التحقق من مستويات وقف الخسارة وجني الأرباح
                    if position['stop_loss'] is not None and current_price <= position['stop_loss']:
                        self.close_position(portfolio_id, position['id'], current_price)
                    elif position['take_profit'] is not None and current_price >= position['take_profit']:
                        self.close_position(portfolio_id, position['id'], current_price)
                    else:
                        updated_positions.append(position)
            
            portfolio['last_updated'] = datetime.now()
            
            return {
                'status': 'success',
                'updated_positions': updated_positions
            }
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def get_portfolio_performance(self, portfolio_id: str) -> Dict:
        """
        حساب أداء المحفظة
        
        Args:
            portfolio_id: معرف المحفظة
            
        Returns:
            Dict: أداء المحفظة
        """
        try:
            if portfolio_id not in self.portfolios:
                return {'status': 'error', 'message': 'المحفظة غير موجودة'}
            
            portfolio = self.portfolios[portfolio_id]
            
            # حساب المقاييس
            total_pnl = sum(p['pnl'] for p in portfolio['positions'] if p['status'] == 'closed')
            total_return = (portfolio['current_balance'] - portfolio['initial_balance']) / portfolio['initial_balance'] * 100
            
            # حساب نسبة الربح/الخسارة
            winning_positions = [p for p in portfolio['positions'] if p['status'] == 'closed' and p['pnl'] > 0]
            win_rate = len(winning_positions) / len([p for p in portfolio['positions'] if p['status'] == 'closed']) * 100 if portfolio['positions'] else 0
            
            # حساب عامل الربح
            total_profit = sum(p['pnl'] for p in winning_positions)
            total_loss = abs(sum(p['pnl'] for p in portfolio['positions'] if p['status'] == 'closed' and p['pnl'] < 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # حساب الحد الأقصى للانخفاض
            balance_history = [portfolio['initial_balance']]
            for transaction in portfolio['transactions']:
                if transaction['type'] == 'open':
                    balance_history.append(balance_history[-1] - transaction['value'])
                else:
                    balance_history.append(balance_history[-1] + transaction['value'])
            
            max_balance = max(balance_history)
            min_balance = min(balance_history)
            max_drawdown = (max_balance - min_balance) / max_balance * 100 if max_balance > 0 else 0
            
            return {
                'status': 'success',
                'performance': {
                    'total_pnl': total_pnl,
                    'total_return': total_return,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'max_drawdown': max_drawdown,
                    'current_balance': portfolio['current_balance'],
                    'initial_balance': portfolio['initial_balance']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio performance: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def rebalance_portfolio(self, portfolio_id: str,
                          target_distribution: Dict[str, float]) -> Dict:
        """
        إعادة توازن المحفظة
        
        Args:
            portfolio_id: معرف المحفظة
            target_distribution: التوزيع المستهدف
            
        Returns:
            Dict: نتائج إعادة التوازن
        """
        try:
            if portfolio_id not in self.portfolios:
                return {'status': 'error', 'message': 'المحفظة غير موجودة'}
            
            portfolio = self.portfolios[portfolio_id]
            
            # تحديث أسعار المراكز
            self.update_positions(portfolio_id)
            
            # حساب القيم الحالية
            total_value = portfolio['current_balance']
            current_distribution = {}
            
            for position in portfolio['positions']:
                if position['status'] == 'open':
                    position_value = position['quantity'] * position['current_price']
                    total_value += position_value
                    current_distribution[position['symbol']] = position_value
            
            # حساب التغييرات المطلوبة
            changes = []
            
            for symbol, target_weight in target_distribution.items():
                target_value = total_value * target_weight
                current_value = current_distribution.get(symbol, 0)
                
                if current_value < target_value:
                    # شراء
                    quantity = (target_value - current_value) / position['current_price']
                    changes.append({
                        'symbol': symbol,
                        'action': 'buy',
                        'quantity': quantity
                    })
                elif current_value > target_value:
                    # بيع
                    position = next((p for p in portfolio['positions'] if p['symbol'] == symbol and p['status'] == 'open'), None)
                    if position:
                        quantity = (current_value - target_value) / position['current_price']
                        changes.append({
                            'symbol': symbol,
                            'action': 'sell',
                            'quantity': quantity
                        })
            
            # تنفيذ التغييرات
            for change in changes:
                if change['action'] == 'buy':
                    self.add_position(
                        portfolio_id,
                        change['symbol'],
                        change['quantity'],
                        position['current_price']
                    )
                else:
                    position = next((p for p in portfolio['positions'] if p['symbol'] == change['symbol'] and p['status'] == 'open'), None)
                    if position:
                        self.close_position(
                            portfolio_id,
                            position['id'],
                            position['current_price']
                        )
            
            return {
                'status': 'success',
                'changes': changes
            }
            
        except Exception as e:
            self.logger.error(f"Error rebalancing portfolio: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def get_portfolio_details(self, portfolio_id: str) -> Dict:
        """
        الحصول على تفاصيل المحفظة
        
        Args:
            portfolio_id: معرف المحفظة
            
        Returns:
            Dict: تفاصيل المحفظة
        """
        try:
            if portfolio_id not in self.portfolios:
                return {'status': 'error', 'message': 'المحفظة غير موجودة'}
            
            portfolio = self.portfolios[portfolio_id]
            
            # تحديث أسعار المراكز
            self.update_positions(portfolio_id)
            
            # حساب الأداء
            performance = self.get_portfolio_performance(portfolio_id)
            
            return {
                'status': 'success',
                'portfolio': portfolio,
                'performance': performance['performance'] if performance['status'] == 'success' else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio details: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def get_user_portfolios(self, user_id: str) -> Dict:
        """
        الحصول على محافظ المستخدم
        
        Args:
            user_id: معرف المستخدم
            
        Returns:
            Dict: قائمة المحافظ
        """
        try:
            user_portfolios = [
                p for p in self.portfolios.values()
                if p['user_id'] == user_id
            ]
            
            return {
                'status': 'success',
                'portfolios': user_portfolios
            }
            
        except Exception as e:
            self.logger.error(f"Error getting user portfolios: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def update_portfolio(self, portfolio_id: str, **kwargs) -> Dict:
        """
        تحديث المحفظة
        
        Args:
            portfolio_id: معرف المحفظة
            **kwargs: الحقول المطلوب تحديثها
            
        Returns:
            Dict: نتيجة التحديث
        """
        try:
            if portfolio_id not in self.portfolios:
                return {'status': 'error', 'message': 'المحفظة غير موجودة'}
            
            portfolio = self.portfolios[portfolio_id]
            
            # تحديث الحقول
            for key, value in kwargs.items():
                if key in portfolio:
                    portfolio[key] = value
            
            portfolio['last_updated'] = datetime.now()
            
            return {
                'status': 'success',
                'portfolio': portfolio
            }
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def delete_portfolio(self, portfolio_id: str) -> Dict:
        """
        حذف المحفظة
        
        Args:
            portfolio_id: معرف المحفظة
            
        Returns:
            Dict: نتيجة الحذف
        """
        try:
            if portfolio_id not in self.portfolios:
                return {'status': 'error', 'message': 'المحفظة غير موجودة'}
            
            del self.portfolios[portfolio_id]
            
            return {'status': 'success'}
            
        except Exception as e:
            self.logger.error(f"Error deleting portfolio: {str(e)}")
            return {'status': 'error', 'message': str(e)} 