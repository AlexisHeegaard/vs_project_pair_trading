# src/analysis.py
import numpy as np
import pandas as pd

def calculate_drawdown(equity_curve):
    """
    Calculates the worst peak-to-valley loss.
    """
    if len(equity_curve) == 0:
        return 0
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return drawdown.min()

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Annualized Sharpe Ratio (assuming daily data).
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * (excess_returns.mean() / returns.std())

def generate_performance_report(equity_curve, trade_log):
    """
    Returns a dictionary of metrics for your thesis table.
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return {
            'Total Return': 0,
            'Annualized Volatility': 0,
            'Sharpe Ratio': 0,
            'Max Drawdown': 0,
            'Total Trades': 0,
            'Win Rate': 0
        }
    
    returns = equity_curve.pct_change().dropna()
    
    # Filter only closed trades (exits have realized_pnl)
    closed_trades = [t for t in trade_log if t['type'] == 'exit']
    winning_trades = len([t for t in closed_trades if t['realized_pnl'] > 0])
    
    report = {
        'Total Return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1,
        'Annualized Volatility': returns.std() * np.sqrt(252),
        'Sharpe Ratio': calculate_sharpe_ratio(returns),
        'Max Drawdown': calculate_drawdown(equity_curve),
        'Total Trades': len(closed_trades),
        'Win Rate': (winning_trades / len(closed_trades) * 100) if closed_trades else 0
    }
    
    return report