# src/analysis.py
import numpy as np
import pandas as pd

def calculate_drawdown(equity_curve):
    """
    Calculates the worst peak-to-valley loss.
    """
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return drawdown.min()

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Annualized Sharpe Ratio (assuming daily data).
    """
    excess_returns = returns - risk_free_rate/252
    if returns.std() == 0:
        return 0
    return np.sqrt(252) * (excess_returns.mean() / returns.std())

def generate_performance_report(equity_curve, trade_log):
    """
    Returns a dictionary of metrics for your thesis table.
    """
    returns = equity_curve.pct_change().dropna()
    
    report = {
        'Total Return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1,
        'Annualized Volatility': returns.std() * np.sqrt(252),
        'Sharpe Ratio': calculate_sharpe_ratio(returns),
        'Max Drawdown': calculate_drawdown(equity_curve),
        'Total Trades': len(trade_log),
        'Win Rate': len([t for t in trade_log if t['pnl'] > 0]) / len(trade_log) if trade_log else 0
    }
    
    return report