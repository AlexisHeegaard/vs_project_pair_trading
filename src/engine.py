# src/engine.py
#MECHANICS OF TRAINING AND EVALUATION FOR PAIR TRADING MODELS



import pandas as pd
import numpy as np
from src.portfolio import PortfolioManager

class BacktestEngine:
    def __init__(self, initial_capital=10000, transaction_cost_pct=0.001, slippage_pct=0.0005):
        """
        :param initial_capital: Starting cash.
        :param transaction_cost_pct: Exchange fees (e.g., 0.1% = 0.001).
        :param slippage_pct: Est. market impact (e.g., 0.05% = 0.0005).
        """
        self.portfolio = PortfolioManager(initial_capital)
        self.cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        
        # State tracking
        self.current_position = 0  # 0 = Flat, 1 = Long, -1 = Short (if simple), or specific quantity
        self.entry_price = 0.0
        self.equity_curve = []
        self.trade_log = []

    def _get_execution_price(self, raw_price, side):
        """
        Simulates Market Microstructure.
        Buys are more expensive, Sells are cheaper (you pay the spread/impact).
        """
        friction = self.cost_pct + self.slippage_pct
        if side == 'buy':
            return raw_price * (1 + friction)
        elif side == 'sell':
            return raw_price * (1 - friction)
        return raw_price

    def run(self, data, signals):
        """
        Iterates through the dataframe to simulate the strategy.
        :param data: DataFrame containing 'Close' prices.
        :param signals: Series/Array of signals (1, -1, 0) aligned with data.
        """
        prices = data['Close'].values
        dates = data.index
        
        # Loop through time
        for t in range(1, len(prices)):
            price = prices[t]
            date = dates[t]
            signal = signals[t]
            
            # 1. MARK TO MARKET (Update Equity based on open position)
            if self.current_position != 0:
                unrealized_pnl = (price - self.entry_price) * self.current_position
                # Note: precise MTM logic depends on if you hold quantity or raw units. 
                # For simplicity here, we assume position size is tracked in PortfolioManager implicitly.
            
            # 2. CHECK FOR ENTRY (Flat -> Open)
            if self.current_position == 0 and signal != 0:
                # ASK PORTFOLIO FOR SIZING
                # We assume a fixed win_rate/ratio for now, or you can calculate dynamic metrics here
                target_allocation = self.portfolio.calculate_position_size(
                    signal_strength=1, 
                    win_rate=0.55,       # You can make this dynamic later
                    win_loss_ratio=1.2,  # You can make this dynamic later
                    current_volatility=0.01 # Placeholder for vol calculation
                )
                
                # Calculate Quantity (Cash / Price)
                exec_price = self._get_execution_price(price, 'buy' if signal > 0 else 'sell')
                quantity = target_allocation / exec_price
                
                self.current_position = quantity * signal # Positive for Long, Negative for Short
                self.entry_price = exec_price
                
                self.trade_log.append({
                    'date': date, 'type': 'entry', 'price': exec_price, 'size': self.current_position
                })

            # 3. CHECK FOR EXIT (Open -> Flat or Flip)
            elif self.current_position != 0 and (signal == 0 or np.sign(signal) != np.sign(self.current_position)):
                
                # EXECUTE EXIT
                exec_price = self._get_execution_price(price, 'sell' if self.current_position > 0 else 'buy')
                
                # Calculate PnL
                pnl = (exec_price - self.entry_price) * self.current_position
                
                # Update Portfolio (Bank Account)
                self.portfolio.update_equity(pnl)
                
                self.trade_log.append({
                    'date': date, 'type': 'exit', 'price': exec_price, 'pnl': pnl
                })
                
                # Reset Position
                self.current_position = 0
                self.entry_price = 0.0

            # Record Daily Equity
            self.equity_curve.append(self.portfolio.current_equity)

        return pd.DataFrame(self.equity_curve, index=dates[1:], columns=['Equity'])