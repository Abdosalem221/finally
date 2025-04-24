"""
خدمة اختبار وتقييم استراتيجيات التداول
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from app.services.data_service import DataService
from app.utils.validators import validate_symbol, validate_timeframe
from app.utils.helpers import calculate_risk_reward_ratio

class BacktestService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_service = DataService()
        self.positions = []
        self.trades = []
        self.initial_balance = 10000  # رصيد ابتدائي افتراضي
        self.current_balance = self.initial_balance
        self.commission_rate = 0.001  # 0.1% عمولة
    
    def run_backtest(self, strategy: Dict, symbol: str, timeframe: str, 
                    start_date: str, end_date: str) -> Dict:
        """
        تشغيل اختبار استراتيجية
        
        Args:
            strategy: استراتيجية التداول
            symbol: رمز العملة
            timeframe: الإطار الزمني
            start_date: تاريخ البداية
            end_date: تاريخ النهاية
            
        Returns:
            Dict: نتائج الاختبار
        """
        try:
            # التحقق من صحة المدخلات
            if not validate_symbol(symbol):
                return {'status': 'error', 'message': 'رمز العملة غير صالح'}
            
            if not validate_timeframe(timeframe):
                return {'status': 'error', 'message': 'الإطار الزمني غير صالح'}
            
            # جلب البيانات التاريخية
            market_data = self._fetch_historical_data(symbol, timeframe, start_date, end_date)
            if market_data['status'] == 'error':
                return market_data
            
            # تحويل البيانات إلى DataFrame
            df = pd.DataFrame(market_data['data'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # تطبيق الاستراتيجية
            signals = self._apply_strategy(df, strategy)
            
            # محاكاة التداول
            trades = self._simulate_trading(df, signals, strategy)
            
            # حساب المقاييس
            metrics = self._calculate_metrics(trades)
            
            return {
                'status': 'success',
                'results': {
                    'trades': trades,
                    'metrics': metrics,
                    'equity_curve': self._calculate_equity_curve(trades),
                    'drawdown': self._calculate_drawdown(trades)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _apply_strategy(self, df: pd.DataFrame, strategy: Dict) -> pd.DataFrame:
        """
        تطبيق استراتيجية التداول
        
        Args:
            df: DataFrame يحتوي على البيانات
            strategy: استراتيجية التداول
            
        Returns:
            pd.DataFrame: إشارات التداول
        """
        try:
            # حساب المؤشرات الفنية
            df = self._calculate_indicators(df, strategy['indicators'])
            
            # توليد الإشارات
            signals = pd.DataFrame(index=df.index)
            signals['signal'] = 0  # 0: لا شيء، 1: شراء، -1: بيع
            
            # تطبيق قواعد الاستراتيجية
            for rule in strategy['rules']:
                if rule['type'] == 'crossover':
                    signals.loc[df[rule['fast']] > df[rule['slow']], 'signal'] = 1
                    signals.loc[df[rule['fast']] < df[rule['slow']], 'signal'] = -1
                elif rule['type'] == 'threshold':
                    signals.loc[df[rule['indicator']] > rule['upper'], 'signal'] = -1
                    signals.loc[df[rule['indicator']] < rule['lower'], 'signal'] = 1
            
            return signals
            
        except Exception as e:
            raise Exception(f"Error applying strategy: {str(e)}")
    
    def _simulate_trading(self, df: pd.DataFrame, signals: pd.DataFrame, 
                         strategy: Dict) -> List[Dict]:
        """
        محاكاة التداول
        
        Args:
            df: DataFrame يحتوي على البيانات
            signals: DataFrame يحتوي على الإشارات
            strategy: استراتيجية التداول
            
        Returns:
            List[Dict]: قائمة الصفقات
        """
        try:
            trades = []
            position = None
            
            for i in range(1, len(df)):
                current_signal = signals['signal'].iloc[i]
                current_price = df['close'].iloc[i]
                
                # إغلاق الصفقة الحالية إذا كانت موجودة
                if position and self._should_close_position(position, current_price, strategy):
                    trade = self._close_position(position, current_price)
                    trades.append(trade)
                    position = None
                
                # فتح صفقة جديدة
                if not position and current_signal != 0:
                    position = self._open_position(current_signal, current_price, strategy)
            
            # إغلاق الصفقة المفتوحة في نهاية الفترة
            if position:
                trade = self._close_position(position, df['close'].iloc[-1])
                trades.append(trade)
            
            return trades
            
        except Exception as e:
            raise Exception(f"Error simulating trading: {str(e)}")
    
    def _calculate_metrics(self, trades: List[Dict]) -> Dict:
        """
        حساب مقاييس الأداء
        
        Args:
            trades: قائمة الصفقات
            
        Returns:
            Dict: مقاييس الأداء
        """
        try:
            if not trades:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'total_return': 0
                }
            
            # حساب المقاييس الأساسية
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['profit'] > 0])
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # حساب الربح والخسارة
            total_profit = sum([t['profit'] for t in trades if t['profit'] > 0])
            total_loss = abs(sum([t['profit'] for t in trades if t['profit'] < 0]))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # حساب العائد
            returns = [t['profit'] / t['entry_price'] for t in trades]
            total_return = sum(returns)
            
            # حساب نسبة شارب
            risk_free_rate = 0.02  # 2% معدل خالي من المخاطر
            excess_returns = [r - risk_free_rate/252 for r in returns]
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': self._calculate_max_drawdown(trades),
                'total_return': total_return
            }
            
        except Exception as e:
            raise Exception(f"Error calculating metrics: {str(e)}")
    
    def _calculate_indicators(self, df: pd.DataFrame, indicators: List[Dict]) -> pd.DataFrame:
        """
        حساب المؤشرات الفنية
        
        Args:
            df: DataFrame يحتوي على البيانات
            indicators: قائمة المؤشرات
            
        Returns:
            pd.DataFrame: البيانات مع المؤشرات
        """
        try:
            for indicator in indicators:
                if indicator['type'] == 'sma':
                    df[f"SMA_{indicator['period']}"] = df['close'].rolling(window=indicator['period']).mean()
                elif indicator['type'] == 'ema':
                    df[f"EMA_{indicator['period']}"] = df['close'].ewm(span=indicator['period']).mean()
                elif indicator['type'] == 'rsi':
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=indicator['period']).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=indicator['period']).mean()
                    rs = gain / loss
                    df['RSI'] = 100 - (100 / (1 + rs))
                elif indicator['type'] == 'macd':
                    exp1 = df['close'].ewm(span=12, adjust=False).mean()
                    exp2 = df['close'].ewm(span=26, adjust=False).mean()
                    df['MACD'] = exp1 - exp2
                    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            return df
            
        except Exception as e:
            raise Exception(f"Error calculating indicators: {str(e)}")
    
    def _should_close_position(self, position: Dict, current_price: float, 
                             strategy: Dict) -> bool:
        """
        التحقق من إغلاق الصفقة
        
        Args:
            position: الصفقة الحالية
            current_price: السعر الحالي
            strategy: استراتيجية التداول
            
        Returns:
            bool: True إذا كان يجب إغلاق الصفقة
        """
        try:
            # حساب الربح/الخسارة
            if position['type'] == 'long':
                profit = (current_price - position['entry_price']) / position['entry_price']
                if profit >= strategy['take_profit'] or profit <= -strategy['stop_loss']:
                    return True
            else:
                profit = (position['entry_price'] - current_price) / position['entry_price']
                if profit >= strategy['take_profit'] or profit <= -strategy['stop_loss']:
                    return True
            
            return False
            
        except Exception as e:
            raise Exception(f"Error checking position close: {str(e)}")
    
    def _open_position(self, signal: int, price: float, strategy: Dict) -> Dict:
        """
        فتح صفقة جديدة
        
        Args:
            signal: إشارة التداول
            price: سعر الدخول
            strategy: استراتيجية التداول
            
        Returns:
            Dict: الصفقة الجديدة
        """
        try:
            position_size = self.current_balance * strategy['position_size']
            commission = position_size * self.commission_rate
            
            return {
                'type': 'long' if signal == 1 else 'short',
                'entry_price': price,
                'entry_time': datetime.now(),
                'position_size': position_size,
                'commission': commission
            }
            
        except Exception as e:
            raise Exception(f"Error opening position: {str(e)}")
    
    def _close_position(self, position: Dict, price: float) -> Dict:
        """
        إغلاق الصفقة
        
        Args:
            position: الصفقة الحالية
            price: سعر الخروج
            
        Returns:
            Dict: الصفقة المغلقة
        """
        try:
            # حساب الربح/الخسارة
            if position['type'] == 'long':
                profit = (price - position['entry_price']) * position['position_size']
            else:
                profit = (position['entry_price'] - price) * position['position_size']
            
            # خصم العمولة
            profit -= position['commission'] * 2  # عمولة الدخول والخروج
            
            # تحديث الرصيد
            self.current_balance += profit
            
            return {
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': price,
                'entry_time': position['entry_time'],
                'exit_time': datetime.now(),
                'position_size': position['position_size'],
                'commission': position['commission'] * 2,
                'profit': profit
            }
            
        except Exception as e:
            raise Exception(f"Error closing position: {str(e)}")
    
    def _calculate_equity_curve(self, trades: List[Dict]) -> List[Dict]:
        """
        حساب منحنى الأسهم
        
        Args:
            trades: قائمة الصفقات
            
        Returns:
            List[Dict]: منحنى الأسهم
        """
        try:
            equity_curve = []
            current_balance = self.initial_balance
            
            for trade in trades:
                current_balance += trade['profit']
                equity_curve.append({
                    'timestamp': trade['exit_time'],
                    'balance': current_balance
                })
            
            return equity_curve
            
        except Exception as e:
            raise Exception(f"Error calculating equity curve: {str(e)}")
    
    def _calculate_drawdown(self, trades: List[Dict]) -> Dict:
        """
        حساب الانخفاض
        
        Args:
            trades: قائمة الصفقات
            
        Returns:
            Dict: معلومات الانخفاض
        """
        try:
            if not trades:
                return {
                    'max_drawdown': 0,
                    'max_drawdown_period': None,
                    'recovery_period': None
                }
            
            equity_curve = self._calculate_equity_curve(trades)
            peak = self.initial_balance
            max_drawdown = 0
            drawdown_start = None
            drawdown_end = None
            
            for point in equity_curve:
                if point['balance'] > peak:
                    peak = point['balance']
                drawdown = (peak - point['balance']) / peak
                
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    if not drawdown_start:
                        drawdown_start = point['timestamp']
                    drawdown_end = point['timestamp']
                elif drawdown == 0 and drawdown_start:
                    drawdown_end = point['timestamp']
            
            return {
                'max_drawdown': max_drawdown,
                'max_drawdown_period': {
                    'start': drawdown_start,
                    'end': drawdown_end
                },
                'recovery_period': self._calculate_recovery_period(equity_curve)
            }
            
        except Exception as e:
            raise Exception(f"Error calculating drawdown: {str(e)}")
    
    def _calculate_recovery_period(self, equity_curve: List[Dict]) -> Optional[Dict]:
        """
        حساب فترة الاسترداد
        
        Args:
            equity_curve: منحنى الأسهم
            
        Returns:
            Optional[Dict]: فترة الاسترداد
        """
        try:
            if not equity_curve:
                return None
            
            initial_balance = self.initial_balance
            recovery_start = None
            recovery_end = None
            
            for point in equity_curve:
                if point['balance'] < initial_balance and not recovery_start:
                    recovery_start = point['timestamp']
                elif point['balance'] >= initial_balance and recovery_start:
                    recovery_end = point['timestamp']
                    break
            
            if recovery_start and recovery_end:
                return {
                    'start': recovery_start,
                    'end': recovery_end,
                    'duration': (recovery_end - recovery_start).total_seconds() / 3600  # بالساعات
                }
            
            return None
            
        except Exception as e:
            raise Exception(f"Error calculating recovery period: {str(e)}")
    
    def _fetch_historical_data(self, symbol: str, timeframe: str, 
                             start_date: str, end_date: str) -> Dict:
        """
        جلب البيانات التاريخية
        
        Args:
            symbol: رمز العملة
            timeframe: الإطار الزمني
            start_date: تاريخ البداية
            end_date: تاريخ النهاية
            
        Returns:
            Dict: البيانات التاريخية
        """
        try:
            # تحويل التواريخ
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
            
            # حساب عدد البيانات المطلوبة
            timeframe_map = {
                '1m': 1,
                '5m': 5,
                '15m': 15,
                '30m': 30,
                '1h': 60,
                '4h': 240,
                '1d': 1440,
                '1w': 10080,
                '1M': 43200
            }
            
            minutes = (end - start).total_seconds() / 60
            limit = int(minutes / timeframe_map[timeframe]) + 1
            
            # جلب البيانات
            return self.data_service.fetch_market_data(symbol, timeframe, limit)
            
        except Exception as e:
            raise Exception(f"Error fetching historical data: {str(e)}") 