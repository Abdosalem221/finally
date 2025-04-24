"""
Database utility functions for calculations and updates that would normally be
handled by PostgreSQL functions and triggers in a more advanced RDBMS.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database.models import Position, Trade, Portfolio, DailyPerformance

logger = logging.getLogger(__name__)

def calculate_profit_loss(session: Session, position_id: int) -> float:
    """Calculate profit/loss for a position based on its trades."""
    try:
        position = session.query(Position).filter_by(position_id=position_id).first()
        if not position:
            return 0.0
            
        trades = session.query(Trade).filter_by(position_id=position_id).all()
        total_pl = sum(trade.price * trade.quantity * (-1 if trade.type == 'SELL' else 1) 
                      for trade in trades)
        return total_pl
    except Exception as e:
        logger.error(f"Error calculating profit/loss for position {position_id}: {e}")
        return 0.0

def calculate_max_drawdown(session: Session, portfolio_id: int, 
                         start_date: datetime, end_date: datetime) -> float:
    """Calculate maximum drawdown for a portfolio over a given period."""
    try:
        daily_performances = session.query(DailyPerformance)\
            .filter(DailyPerformance.portfolio_id == portfolio_id,
                   DailyPerformance.date >= start_date,
                   DailyPerformance.date <= end_date)\
            .order_by(DailyPerformance.date).all()
            
        if not daily_performances:
            return 0.0
            
        equity_values = [perf.equity for perf in daily_performances]
        peak = equity_values[0]
        max_dd = 0.0
        
        for equity in equity_values[1:]:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
            
        return max_dd
    except Exception as e:
        logger.error(f"Error calculating max drawdown for portfolio {portfolio_id}: {e}")
        return 0.0

def calculate_sharpe_ratio(session: Session, portfolio_id: int,
                         start_date: datetime, end_date: datetime,
                         risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio for a portfolio over a given period."""
    try:
        daily_performances = session.query(DailyPerformance)\
            .filter(DailyPerformance.portfolio_id == portfolio_id,
                   DailyPerformance.date >= start_date,
                   DailyPerformance.date <= end_date)\
            .order_by(DailyPerformance.date).all()
            
        if not daily_performances or len(daily_performances) < 2:
            return 0.0
            
        daily_returns = []
        prev_equity = daily_performances[0].equity
        
        for perf in daily_performances[1:]:
            if prev_equity > 0:
                daily_return = (perf.equity - prev_equity) / prev_equity
                daily_returns.append(daily_return)
            prev_equity = perf.equity
            
        if not daily_returns:
            return 0.0
            
        returns_array = np.array(daily_returns)
        avg_return = np.mean(returns_array)
        std_dev = np.std(returns_array)
        
        if std_dev == 0:
            return 0.0
            
        daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
        sharpe_ratio = (avg_return - daily_rf_rate) / std_dev
        annualized_sharpe = sharpe_ratio * np.sqrt(252)
        
        return annualized_sharpe
    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio for portfolio {portfolio_id}: {e}")
        return 0.0

def update_timestamps(session: Session, table_name: str, row_id: int) -> None:
    """Update the timestamps table when a record is modified."""
    try:
        session.execute(
            text("INSERT INTO update_timestamps (table_name, row_id, updated_at) "
                 "VALUES (:table, :id, :now)"),
            {"table": table_name, "id": row_id, "now": datetime.utcnow()}
        )
    except Exception as e:
        logger.error(f"Error updating timestamps for {table_name} id {row_id}: {e}")

def close_position(session: Session, position_id: int) -> None:
    """Record position closure in the position_closures table."""
    try:
        session.execute(
            text("INSERT INTO position_closures (position_id, closed_at) "
                 "VALUES (:id, :now)"),
            {"id": position_id, "now": datetime.utcnow()}
        )
    except Exception as e:
        logger.error(f"Error recording position closure for id {position_id}: {e}")

def update_daily_performance(session: Session, portfolio_id: int) -> None:
    """Update daily performance records for a portfolio."""
    try:
        portfolio = session.query(Portfolio).filter_by(portfolio_id=portfolio_id).first()
        if not portfolio:
            return
            
        today = datetime.utcnow().date()
        
        # Calculate current equity and profit/loss
        positions = session.query(Position).filter_by(portfolio_id=portfolio_id).all()
        total_pl = sum(calculate_profit_loss(session, pos.position_id) for pos in positions)
        current_equity = portfolio.balance + total_pl
        
        # Calculate drawdown
        max_dd = calculate_max_drawdown(session, portfolio_id, 
                                      today - timedelta(days=30), today)
        
        session.execute(
            text("INSERT INTO daily_performance "
                 "(portfolio_id, date, balance, equity, profit_loss, drawdown) "
                 "VALUES (:pid, :date, :balance, :equity, :pl, :dd)"),
            {
                "pid": portfolio_id,
                "date": today,
                "balance": portfolio.balance,
                "equity": current_equity,
                "pl": total_pl,
                "dd": max_dd
            }
        )
    except Exception as e:
        logger.error(f"Error updating daily performance for portfolio {portfolio_id}: {e}") 