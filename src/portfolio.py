# src/portfolio.py
import numpy as np
from src.statistics import kelly_criterion

class PortfolioManager:
    def __init__(self, initial_capital, max_position_size=0.2):
        self.capital = initial_capital
        self.current_equity = initial_capital
        self.max_size = max_position_size  # Risk constraint (e.g., max 20% per trade)

    def calculate_position_size(self, signal_strength, win_rate, win_loss_ratio, current_volatility):
        """
        Determines how many units to buy based on risk math.
        """
        # 1. Get raw percentage from Kelly Criterion
        raw_kelly = kelly_criterion(win_rate, win_loss_ratio)
        
        # 2. Volatility Scaling (optional but recommended)
        # If volatility is high, reduce size.
        vol_adjustment = 1 / (current_volatility * 100) if current_volatility > 0 else 1
        
        # 3. Apply Constraints (The "Brakes")
        # Don't bet more than Kelly, and don't bet more than your max_size cap
        safe_size = min(raw_kelly * vol_adjustment, self.max_size)
        
        # 4. Convert % to Cash
        allocation_cash = self.current_equity * safe_size
        
        return allocation_cash

    def update_equity(self, realized_pnl):
        self.current_equity += realized_pnl