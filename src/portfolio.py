# src/portfolio.py
class PortfolioManager:
    
    #Simplified Portfolio perforamnce tracker for pair trading backtest.
    #Tracks equity gains and losses and applies realized P&L from trades. 

    def __init__(self, initial_capital):
        self.current_equity = initial_capital

    def update_equity(self, realized_pnl):
        """
        Update portfolio equity with realized profit/loss from a closed trade.
        
        Args:
            realized_pnl (float): Profit or loss from closed position
        """
        self.current_equity += realized_pnl