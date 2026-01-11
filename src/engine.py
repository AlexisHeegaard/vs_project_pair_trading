# Multi-pair backtest engine 
import sys
import os
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


class BacktestEngine:
    """
    Simple multi-pair backtesting engine.
    
    Key principle: ALL positions are sized in DOLLARS using position_risk_pct.
    This ensures equity curve stays in dollars throughout.
    """

    def __init__(self, initial_capital=10000, transaction_cost_pct=0.003, 
                 slippage_pct=0.001, position_risk_pct=0.02, max_leverage=0.10,
                 allow_same_day_reentry=False):
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.position_risk_pct = position_risk_pct
        self.max_leverage = max_leverage  # Max % of equity that can be allocated across all positions
        self.allow_same_day_reentry = allow_same_day_reentry  # Prevent re-entry on same day as exit

        # Strategy parameters
        self.ENTRY_Z = 1.5
        self.EXIT_Z = 0.5
        self.MODEL_CONFIDENCE = 0.5

        # State
        self.realized_equity = initial_capital  # Track realized gains/losses directly
        self.positions = {}  # pair_id -> position dict
        self.trade_log = []
        self.equity_curve = []
        self.closed_today = {}  # pair_id -> date, to prevent same-day re-entry

    def _apply_costs(self, price, side):
        """Apply transaction costs and slippage."""
        total_friction = self.transaction_cost_pct + self.slippage_pct
        if side == 'buy':
            return price * (1 + total_friction)
        elif side == 'sell':
            return price * (1 - total_friction)
        return price

    def _get_total_allocated_capital(self):
        """Calculate total capital allocated across all open positions."""
        return sum(pos['capital_allocated'] for pos in self.positions.values())

    def run(self, data, model_col_name='Ridge_Pred'):
        """
        Run the backtest on multi-pair data.
        
        Args:
            data: DataFrame with columns ['Spread', 'Z_Score', 'Pair_ID', model_col_name]
                  indexed by datetime
            model_col_name: prediction column name
            
        Returns:
            pd.Series: equity curve indexed by datetime
        """
        if data.empty:
            return pd.Series(dtype=float)

        df = data.copy().sort_index()

        # Reset state
        self.realized_equity = self.initial_capital
        self.positions = {}
        self.trade_log = []
        self.equity_curve = []
        self.closed_today = {}

        dates = sorted(df.index.unique())
        print(f"Running backtest on {len(dates)} dates, {len(df)} total observations")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        print(f"Max Leverage: {self.max_leverage*100:.1f}%")
        print(f"Allow Same-Day Re-entry: {self.allow_same_day_reentry}\n")

        for date in dates:
            # FIX #1: Improved datetime filtering
            day_data = df[df.index == date].copy()
            
            if day_data.empty:
                continue

            # Clear closed_today at start of new day (allow re-entry next day if desired)
            if not self.allow_same_day_reentry:
                self.closed_today = {}

            # Step 1: Process exits (check z-score reversal for open positions)
            for pair_id, pos in list(self.positions.items()):
                pair_row = day_data[day_data['Pair_ID'] == pair_id]
                if pair_row.empty:
                    continue  # No data for this pair today
                
                z_score = float(pair_row['Z_Score'].iloc[0])
                spread_price = float(pair_row['Spread'].iloc[0])
                
                # Exit if mean reversion detected (z-score crosses EXIT_Z threshold)
                if abs(z_score) < self.EXIT_Z:
                    self._close_position(pair_id, spread_price, date, reason='Mean Reversion')
                    
                    # Track that we closed this pair today (for same-day re-entry prevention)
                    if not self.allow_same_day_reentry:
                        self.closed_today[pair_id] = date

            # Step 2: Process entries (for pairs without open positions)
            for _, row in day_data.iterrows():
                pair_id = row['Pair_ID']
                
                # Skip if already have position
                if pair_id in self.positions:
                    continue
                
                # FIX #4: Prevent same-day re-entry
                if not self.allow_same_day_reentry and pair_id in self.closed_today:
                    continue
                
                z_score = float(row['Z_Score'])
                spread_price = float(row['Spread'])
                pred = float(row.get(model_col_name, 0.0))
                
                # Generate signal from model prediction
                model_signal = 1 if pred > self.MODEL_CONFIDENCE else -1
                
                # Entry logic: z-score + model agreement
                long_signal = (z_score < -self.ENTRY_Z) and (model_signal == 1)
                short_signal = (z_score > self.ENTRY_Z) and (model_signal == -1)
                
                if long_signal or short_signal:
                    # FIX #3: Check leverage before opening position
                    capital_to_allocate = self.realized_equity * self.position_risk_pct
                    total_allocated = self._get_total_allocated_capital()
                    
                    if (total_allocated + capital_to_allocate) > (self.realized_equity * self.max_leverage):
                        continue  # Skip entry due to leverage limit
                    
                    # Determine direction
                    direction = 1 if long_signal else -1
                    self._open_position(pair_id, spread_price, date, direction=direction)

            # Step 3: Mark-to-market all open positions (compute unrealized PnL in DOLLARS)
            # FIX #2: Account for costs in unrealized P&L
            unrealized_total = 0.0
            
            for pair_id, pos in self.positions.items():
                pair_row = day_data[day_data['Pair_ID'] == pair_id]
                if pair_row.empty:
                    continue  # Use last known price if not updated today
                
                current_spread = float(pair_row['Spread'].iloc[0])
                
                # Apply exit costs to current price (what we'd actually get)
                exit_side = 'sell' if pos['direction'] > 0 else 'buy'
                current_price_with_exit_costs = self._apply_costs(current_spread, exit_side)
                
                # Unrealized PnL in dollars = (price change after costs) * (position size in units)
                spread_change = current_price_with_exit_costs - pos['entry_spread']
                unrealized_dollars = spread_change * pos['position_size_units'] * pos['direction']
                unrealized_total += unrealized_dollars
            
            # Total portfolio equity = realized + unrealized
            total_equity = self.realized_equity + unrealized_total
            self.equity_curve.append((date, total_equity))

        # Step 4: Close all remaining open positions (end of backtest)
        if self.positions:
            last_date = dates[-1]
            last_day_data = df[df.index == last_date].copy()
            
            for pair_id in list(self.positions.keys()):
                pair_row = last_day_data[last_day_data['Pair_ID'] == pair_id]
                if not pair_row.empty:
                    final_spread = float(pair_row['Spread'].iloc[0])
                    self._close_position(pair_id, final_spread, last_date, reason='End of Backtest')

        # Convert to Series
        if self.equity_curve:
            idx, vals = zip(*self.equity_curve)
            return pd.Series(list(vals), index=pd.to_datetime(list(idx)))
        else:
            return pd.Series(dtype=float)

    def _open_position(self, pair_id, spread_price, date, direction):
        """
        Open a new position.
        
        Position size is calculated as: (% of equity) / (spread price)
        This gives us the number of units we can buy with our allocated capital.
        """
        # Amount of capital to risk on this trade (based on current realized equity)
        capital_allocated = self.realized_equity * self.position_risk_pct
        
        # How many "units" of spread can we buy with this capital?
        position_size_units = capital_allocated / spread_price
        
        # Apply entry costs
        exec_price = self._apply_costs(spread_price, 'buy' if direction > 0 else 'sell')
        
        # Store position
        self.positions[pair_id] = {
            'entry_spread': exec_price,
            'entry_spread_raw': spread_price,  # Store raw for reference
            'direction': direction,
            'position_size_units': position_size_units,
            'capital_allocated': capital_allocated,
        }
        
        self.trade_log.append({
            'date': date,
            'pair': pair_id,
            'type': 'entry',
            'spread_price': spread_price,
            'spread_price_with_costs': exec_price,
            'direction': 'LONG' if direction > 0 else 'SHORT',
            'position_size_units': position_size_units,
            'capital_allocated': capital_allocated,
        })

    def _close_position(self, pair_id, spread_price, date, reason='exit'):
        """Close a position and realize PnL."""
        if pair_id not in self.positions:
            return
        
        pos = self.positions.pop(pair_id)
        
        # Apply exit costs
        exec_price = self._apply_costs(spread_price, 'sell' if pos['direction'] > 0 else 'buy')
        
        # Realized PnL in dollars = (spread change) * (units) * (direction)
        spread_change = exec_price - pos['entry_spread']
        realized_pnl = spread_change * pos['position_size_units'] * pos['direction']
        
        # Update realized equity with gains/losses
        self.realized_equity += realized_pnl
        
        # Log trade
        self.trade_log.append({
            'date': date,
            'pair': pair_id,
            'type': 'exit',
            'spread_exit': spread_price,
            'spread_exit_with_costs': exec_price,
            'spread_entry': pos['entry_spread_raw'],
            'spread_entry_with_costs': pos['entry_spread'],
            'realized_pnl': realized_pnl,
            'pnl_pct': (realized_pnl / pos['capital_allocated'] * 100) if pos['capital_allocated'] != 0 else 0,
            'reason': reason,
            'portfolio_equity_after': self.realized_equity,
        })

    def get_trade_summary(self):
        """Return summary statistics of closed trades."""
        exit_trades = [t for t in self.trade_log if t['type'] == 'exit']
        
        if not exit_trades:
            return None
        
        pnl_values = [t['realized_pnl'] for t in exit_trades]
        winning_trades = sum(1 for p in pnl_values if p > 0)
        losing_trades = len(exit_trades) - winning_trades
        
        return {
            'total_trades': len(exit_trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': (winning_trades / len(exit_trades) * 100) if exit_trades else 0,
            'total_pnl': sum(pnl_values),
            'avg_pnl': np.mean(pnl_values) if pnl_values else 0,
            'avg_win': np.mean([p for p in pnl_values if p > 0]) if winning_trades > 0 else 0,
            'avg_loss': np.mean([p for p in pnl_values if p < 0]) if losing_trades > 0 else 0,
        }


if __name__ == "__main__":
    print("✅ BacktestEngine (FIXED - Direct Equity Tracking) loaded successfully.")