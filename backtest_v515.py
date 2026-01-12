#!/usr/bin/env python3
"""
LUXOR V7 PRANA v5.1.5 - HYBRID REGIME-AWARE TRAILING STOP
Dynamic trailing stop that adapts to market regime + profit curve
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from luxor_v7_prana import LuxorV7PranaSystem

logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# HYBRID TRAILING STOP CALCULATOR
# ============================================================

class HybridTrailingStopCalculator:
    """
    Regime-aware dynamic trailing stop calculator.
    Adapts stop distance based on:
    1. Market regime (TRENDING/RANGING/VOLATILE)
    2. Current profit R-multiple
    3. Current ATR volatility
    """
    
    def __init__(self, params: Dict = None):
        """
        Initialize with optimization parameters.
        
        Default parameters (conservative):
            TRENDING: [1.5, 1.2, 1.0] for profit stages [<1R, <2R, >=2R]
            RANGING:  [1.8, 0.6] for profit stages [<0.5R, >=0.5R]
            VOLATILE: [2.0, 1.3] for profit stages [<1R, >=1R]
            UNDEFINED: 1.5 (fallback)
        """
        self.params = params or {
            "TRENDING": {
                "stage_0": 1.5,  # profit_r < 1.0
                "stage_1": 1.2,  # 1.0 <= profit_r < 2.0
                "stage_2": 1.0   # profit_r >= 2.0
            },
            "RANGING": {
                "stage_0": 1.8,  # profit_r < 0.5
                "stage_1": 0.6   # profit_r >= 0.5
            },
            "VOLATILE": {
                "stage_0": 2.0,  # profit_r < 1.0
                "stage_1": 1.3   # profit_r >= 1.0
            },
            "UNDEFINED": 1.5
        }
        
        self.trailing_events = []  # Log all trailing calculations
    
    def calculate(self, regime: str, atr: float, profit_r: float, 
                  current_price: float, entry_price: float) -> float:
        """
        Calculate dynamic trailing stop distance.
        
        Args:
            regime: Market regime ("TRENDING", "RANGING", "VOLATILE", "UNDEFINED")
            atr: Current ATR value
            profit_r: Current profit in R-multiples
            current_price: Current market price
            entry_price: Original entry price
        
        Returns:
            trailing_distance: Distance in price points from current price
        """
        if regime == "TRENDING":
            if profit_r < 1.0:
                multiplier = self.params["TRENDING"]["stage_0"]
                stage = "entry_phase"
            elif profit_r < 2.0:
                multiplier = self.params["TRENDING"]["stage_1"]
                stage = "accumulation"
            else:
                multiplier = self.params["TRENDING"]["stage_2"]
                stage = "breakout"
        
        elif regime == "RANGING":
            if profit_r < 0.5:
                multiplier = self.params["RANGING"]["stage_0"]
                stage = "avoid_whipsaw"
            else:
                multiplier = self.params["RANGING"]["stage_1"]
                stage = "lock_profit"
        
        elif regime == "VOLATILE":
            if profit_r < 1.0:
                multiplier = self.params["VOLATILE"]["stage_0"]
                stage = "absorb_noise"
            else:
                multiplier = self.params["VOLATILE"]["stage_1"]
                stage = "gradual_tightening"
        
        else:  # UNDEFINED or unknown
            multiplier = self.params["UNDEFINED"]
            stage = "default_conservative"
        
        trailing_distance = atr * multiplier
        
        # Log event
        self.trailing_events.append({
            "regime": regime,
            "stage": stage,
            "profit_r": round(profit_r, 2),
            "atr": round(atr, 2),
            "multiplier": multiplier,
            "trailing_distance": round(trailing_distance, 2),
            "current_price": round(current_price, 2),
            "entry_price": round(entry_price, 2)
        })
        
        return trailing_distance
    
    def get_statistics(self) -> Dict:
        """Get statistics about trailing events."""
        if not self.trailing_events:
            return {}
        
        df = pd.DataFrame(self.trailing_events)
        
        return {
            "total_events": len(df),
            "regime_distribution": df['regime'].value_counts().to_dict(),
            "stage_distribution": df['stage'].value_counts().to_dict(),
            "avg_multiplier": float(df['multiplier'].mean()),
            "avg_profit_r_at_trail": float(df['profit_r'].mean()),
            "max_profit_r_trailed": float(df['profit_r'].max())
        }


# ============================================================
# SIGNAL ADAPTER
# ============================================================

def adapt_signal_for_backtest(mtf_signal: dict) -> dict:
    """Convert generate_mtf_signal() output to backtest-compatible format."""
    if mtf_signal.get("status") != "success":
        return {
            "action": "HOLD",
            "reason": mtf_signal.get("detail", "signal_error")
        }
    
    trade_setups = mtf_signal.get("trade_setups", [])
    
    if not trade_setups:
        return {
            "action": "HOLD",
            "reason": "no_valid_setup"
        }
    
    setup = trade_setups[0]
    
    if not setup.get("valid", False):
        return {
            "action": "HOLD",
            "reason": f"invalid_{setup.get('status', 'unknown')}"
        }
    
    direction = setup.get("direction", "HOLD")
    
    return {
        "action": direction,
        "entry": setup.get("entry"),
        "stop_loss": setup.get("stop_loss"),
        "tp1": setup.get("tp1"),
        "tp2": setup.get("tp2"),
        "tp3": setup.get("tp3"),
        "risk": setup.get("risk"),
        "risk_pct": setup.get("risk_pct"),
        "rr_ratio": setup.get("rr_ratio"),
        "confidence": setup.get("confidence"),
        "position_size": setup.get("position_size"),
        "rationale": setup.get("rationale", ""),
        "bias": mtf_signal.get("primary_bias", {}).get("primary_bias", "NEUTRAL"),
        "regime": mtf_signal.get("regime", {}).get("current", "UNKNOWN")
    }


# ============================================================
# BACKTEST ENGINE WITH HYBRID TRAILING
# ============================================================

class BacktestEngineV515:
    """
    Enhanced backtest engine with hybrid regime-aware trailing stop.
    """
    
    def __init__(self, initial_capital: float = 10000.0,
                 risk_per_trade: float = 0.01,
                 min_rr_ratio: float = 1.30,
                 enable_shorts: bool = True,
                 min_hold_bars: int = 3,
                 trailing_params: Dict = None):
        
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.min_rr_ratio = min_rr_ratio
        self.enable_shorts = enable_shorts
        self.min_hold_bars = min_hold_bars
        
        # Hybrid Trailing Calculator
        self.trailing_calculator = HybridTrailingStopCalculator(trailing_params)
        
        self.trades = []
        self.position = None
        self.equity_curve = []
        
    def calculate_position_size(self, entry: float, stop_loss: float) -> float:
        """Calculate position size based on risk."""
        risk_amount = self.capital * self.risk_per_trade
        risk_per_unit = abs(entry - stop_loss)
        
        if risk_per_unit == 0:
            return 0
        
        return risk_amount / risk_per_unit
    
    def open_position(self, bar_idx: int, signal: dict, price: float, 
                      atr: float, date: datetime):
        """Open a new position."""
        action = signal['action']
        
        if action == "HOLD":
            return
        
        if action == "SHORT" and not self.enable_shorts:
            logger.info(f"[BAR {bar_idx}] SHORT signal ignored (shorts disabled)")
            return
        
        entry = signal.get('entry', price)
        stop_loss = signal['stop_loss']
        rr_ratio = signal.get('rr_ratio', 0)
        
        if rr_ratio < self.min_rr_ratio:
            logger.info(f"[BAR {bar_idx}] Signal rejected: RR {rr_ratio:.2f} < {self.min_rr_ratio}")
            return
        
        position_size = self.calculate_position_size(entry, stop_loss)
        
        if position_size <= 0:
            logger.info(f"[BAR {bar_idx}] Invalid position size: {position_size}")
            return
        
        # Calculate initial R (risk per unit)
        initial_r = abs(entry - stop_loss)
        
        self.position = {
            'action': action,
            'entry_bar': bar_idx,
            'entry_date': date,
            'entry_price': entry,
            'stop_loss': stop_loss,
            'initial_stop': stop_loss,  # Save original stop
            'position_size': position_size,
            'tp1': signal.get('tp1'),
            'tp2': signal.get('tp2'),
            'tp3': signal.get('tp3'),
            'confidence': signal.get('confidence'),
            'regime': signal.get('regime', 'UNKNOWN'),
            'bias': signal.get('bias', 'NEUTRAL'),
            'initial_r': initial_r,
            'highest_profit_r': 0.0,
            'trailing_active': False,
            'bars_held': 0,
            'atr_at_entry': atr
        }
        
        logger.info(f"[BAR {bar_idx}] OPEN {action}: entry={entry:.2f}, stop={stop_loss:.2f}, "
                   f"size={position_size:.4f}, RR={rr_ratio:.2f}, regime={self.position['regime']}")
    
    def update_trailing_stop(self, current_price: float, atr: float):
        """Update trailing stop using hybrid calculator."""
        if not self.position or not self.position.get('trailing_active'):
            return
        
        action = self.position['action']
        entry = self.position['entry_price']
        current_stop = self.position['stop_loss']
        initial_r = self.position['initial_r']
        regime = self.position['regime']
        
        # Calculate current profit in R-multiples
        if action == "LONG":
            profit_points = current_price - entry
        else:  # SHORT
            profit_points = entry - current_price
        
        profit_r = profit_points / initial_r if initial_r > 0 else 0
        
        # Update highest profit R
        if profit_r > self.position['highest_profit_r']:
            self.position['highest_profit_r'] = profit_r
        
        # Calculate new trailing distance
        trailing_distance = self.trailing_calculator.calculate(
            regime=regime,
            atr=atr,
            profit_r=profit_r,
            current_price=current_price,
            entry_price=entry
        )
        
        # Update stop loss
        if action == "LONG":
            new_stop = current_price - trailing_distance
            if new_stop > current_stop:
                self.position['stop_loss'] = new_stop
                logger.debug(f"  TRAIL UPDATE: LONG stop {current_stop:.2f} -> {new_stop:.2f} "
                           f"(profit_r={profit_r:.2f}, regime={regime})")
        else:  # SHORT
            new_stop = current_price + trailing_distance
            if new_stop < current_stop:
                self.position['stop_loss'] = new_stop
                logger.debug(f"  TRAIL UPDATE: SHORT stop {current_stop:.2f} -> {new_stop:.2f} "
                           f"(profit_r={profit_r:.2f}, regime={regime})")
    
    def check_exit(self, bar_idx: int, high: float, low: float, close: float, 
                   atr: float, date: datetime) -> bool:
        """Check if position should be exited.
        Note: bars_held is already incremented in force_min_hold().
        """
        if not self.position:
            return False
        
        # bars_held already incremented in force_min_hold()
        # self.position['bars_held'] += 1
        action = self.position['action']
        entry = self.position['entry_price']
        stop = self.position['stop_loss']
        initial_r = self.position['initial_r']
        
        # Calculate current profit in R
        if action == "LONG":
            profit_points = close - entry
        else:
            profit_points = entry - close
        
        profit_r = profit_points / initial_r if initial_r > 0 else 0
        
        # Activate trailing at 1R profit
        if profit_r >= 1.0 and not self.position['trailing_active']:
            self.position['trailing_active'] = True
            logger.info(f"[BAR {bar_idx}] TRAILING ACTIVATED at {profit_r:.2f}R")
        
        # Update trailing stop if active
        if self.position['trailing_active']:
            self.update_trailing_stop(close, atr)
        
        # Check stop loss
        exit_triggered = False
        exit_reason = None
        exit_price = None
        
        # Check profit targets first (TP3 -> TP2 -> TP1)
        tp3 = self.position.get('tp3')
        tp2 = self.position.get('tp2')
        tp1 = self.position.get('tp1')
        
        if action == "LONG":
            # Check profit targets
            if tp3 and high >= tp3:
                exit_triggered = True
                exit_reason = "tp3"
                exit_price = tp3
            elif tp2 and high >= tp2:
                exit_triggered = True
                exit_reason = "tp2"
                exit_price = tp2
            elif tp1 and high >= tp1:
                exit_triggered = True
                exit_reason = "tp1"
                exit_price = tp1
            # Check stop loss
            elif low <= stop:
                exit_triggered = True
                exit_reason = "stop_loss"
                exit_price = stop
        else:  # SHORT
            # Check profit targets
            if tp3 and low <= tp3:
                exit_triggered = True
                exit_reason = "tp3"
                exit_price = tp3
            elif tp2 and low <= tp2:
                exit_triggered = True
                exit_reason = "tp2"
                exit_price = tp2
            elif tp1 and low <= tp1:
                exit_triggered = True
                exit_reason = "tp1"
                exit_price = tp1
            # Check stop loss
            elif high >= stop:
                exit_triggered = True
                exit_reason = "stop_loss"
                exit_price = stop
        
        if exit_triggered:
            self.close_position(bar_idx, exit_price, exit_reason, date)
            return True
        
        return False
    
    def close_position(self, bar_idx: int, exit_price: float, 
                       exit_reason: str, date: datetime):
        """Close current position and record trade."""
        if not self.position:
            return
        
        action = self.position['action']
        entry = self.position['entry_price']
        size = self.position['position_size']
        initial_r = self.position['initial_r']
        
        # Calculate P&L
        if action == "LONG":
            pnl = (exit_price - entry) * size
            pnl_pct = ((exit_price - entry) / entry) * 100
        else:
            pnl = (entry - exit_price) * size
            pnl_pct = ((entry - exit_price) / entry) * 100
        
        # Calculate realized R
        realized_points = exit_price - entry if action == "LONG" else entry - exit_price
        realized_r = realized_points / initial_r if initial_r > 0 else 0
        
        # Update capital
        self.capital += pnl
        
        # Record trade
        trade = {
            'trade_id': len(self.trades) + 1,
            'action': action,
            'entry_bar': self.position['entry_bar'],
            'entry_date': self.position['entry_date'],
            'entry_price': entry,
            'exit_bar': bar_idx,
            'exit_date': date,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'position_size': size,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'realized_r': realized_r,
            'highest_profit_r': self.position['highest_profit_r'],
            'bars_held': self.position['bars_held'],
            'initial_stop': self.position['initial_stop'],
            'final_stop': self.position['stop_loss'],
            'stop_moved': abs(self.position['stop_loss'] - self.position['initial_stop']) > 0.01,
            'trailing_was_active': self.position.get('trailing_active', False),
            'confidence': self.position['confidence'],
            'regime': self.position['regime'],
            'bias': self.position['bias'],
            'capital_after': self.capital
        }
        
        self.trades.append(trade)
        
        result = "WIN" if pnl > 0 else "LOSS"
        logger.info(f"[BAR {bar_idx}] CLOSE {action} ({result}): exit={exit_price:.2f}, "
                   f"pnl=${pnl:.2f} ({pnl_pct:+.2f}%), R={realized_r:+.2f}, "
                   f"bars={self.position['bars_held']}, reason={exit_reason}")
        
        self.position = None
    
    def force_min_hold(self, bar_idx: int) -> bool:
        """Check if position must be held due to MIN_HOLD rule.
        INCREMENT bars_held HERE to fix circular dependency.
        """
        if not self.position:
            return False
        
        # INCREMENT bars_held BEFORE checking (FIX circular dependency!)
        self.position['bars_held'] = self.position.get('bars_held', 0) + 1
        bars_held = self.position['bars_held']
        
        if bars_held < self.min_hold_bars:
            # logger.debug(f"[BAR {bar_idx}] MIN_HOLD: {bars_held}/{self.min_hold_bars} bars")
            return True
        
        return False
    
    def run(self, df: pd.DataFrame, system: LuxorV7PranaSystem) -> Dict:
        """Run backtest with hybrid trailing stop."""
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING BACKTEST v5.1.5 HYBRID")
        logger.info(f"{'='*60}")
        logger.info(f"Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Risk per trade: {self.risk_per_trade*100:.1f}%")
        logger.info(f"Min RR: {self.min_rr_ratio:.2f}")
        logger.info(f"Shorts enabled: {self.enable_shorts}")
        logger.info(f"Min hold bars: {self.min_hold_bars}")
        logger.info(f"Total bars: {len(df)}")
        logger.info(f"{'='*60}\n")
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            date = row.name
            close = row['close']
            high = row['high']
            low = row['low']
            
            # Calculate ATR (14-period default)
            if idx >= 14:
                atr = df['atr'].iloc[idx] if 'atr' in df.columns else (high - low)
            else:
                atr = high - low
            
            # Record equity
            self.equity_curve.append({
                'bar': idx,
                'date': date,
                'equity': self.capital
            })
            
            # DEBUG: Track position state
            if idx < 5 or (self.position and idx <= self.position['entry_bar'] + 5):
                logger.debug(f"[BAR {idx}] Position state: {self.position is not None}")
            
            # Check existing position
            if self.position:
                # Force min hold
                if self.force_min_hold(idx):
                    continue
                
                # Check exit
                if self.check_exit(idx, high, low, close, atr, date):
                    continue  # Position closed
            
            # Generate new signal if no position
            if not self.position:
                # Use historical data up to current bar - PRESERVE DatetimeIndex!
                df_hist = df.iloc[:idx+1].copy()
                # Re-set the DatetimeIndex explicitly if it was lost
                if 'date' in df_hist.columns and not isinstance(df_hist.index, pd.DatetimeIndex):
                    df_hist = df_hist.set_index('date')
                
                try:
                    mtf_signal = system.generate_mtf_signal(
                        symbol="BTCUSDT",
                        df_historical=df_hist,
                        reference_date=date
                    )
                    
                    signal = adapt_signal_for_backtest(mtf_signal)
                    
                    if signal['action'] != "HOLD":
                        self.open_position(idx, signal, close, atr, date)
                
                except Exception as e:
                    logger.error(f"[BAR {idx}] Signal generation error: {str(e)}")
                    continue
        
        # Close any remaining position at end
        if self.position:
            self.close_position(
                len(df) - 1,
                df.iloc[-1]['close'],
                "end_of_data",
                df.iloc[-1].name
            )
        
        return self.generate_report()
    
    def generate_report(self) -> Dict:
        """Generate comprehensive backtest report."""
        if not self.trades:
            return {
                "error": "No trades executed",
                "total_trades": 0
            }
        
        df_trades = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = df_trades['pnl'].sum()
        total_return_pct = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        
        # Win/Loss averages
        winners = df_trades[df_trades['pnl'] > 0]
        losers = df_trades[df_trades['pnl'] <= 0]
        
        avg_win = winners['pnl'].mean() if len(winners) > 0 else 0
        avg_loss = losers['pnl'].mean() if len(losers) > 0 else 0
        avg_win_r = winners['realized_r'].mean() if len(winners) > 0 else 0
        avg_loss_r = losers['realized_r'].mean() if len(losers) > 0 else 0
        
        # Profit factor
        gross_profit = winners['pnl'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['pnl'].sum()) if len(losers) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Drawdown
        equity = [e['equity'] for e in self.equity_curve]
        peak = equity[0]
        max_dd = 0
        
        for eq in equity:
            if eq > peak:
                peak = eq
            dd = ((eq - peak) / peak) * 100
            if dd < max_dd:
                max_dd = dd
        
        # R-multiples
        avg_r = df_trades['realized_r'].mean()
        
        # Sharpe (simplified)
        returns = df_trades['pnl_pct'].values
        sharpe = (returns.mean() / returns.std()) if returns.std() > 0 else 0
        
        # Bars held
        avg_bars_held = df_trades['bars_held'].mean()
        
        # Trailing statistics
        trailing_stats = self.trailing_calculator.get_statistics()
        trailing_exits = len(df_trades[df_trades['trailing_was_active'] == True])
        
        # Direction breakdown
        long_trades = len(df_trades[df_trades['action'] == 'LONG'])
        short_trades = len(df_trades[df_trades['action'] == 'SHORT'])
        
        long_wins = len(df_trades[(df_trades['action'] == 'LONG') & (df_trades['pnl'] > 0)])
        short_wins = len(df_trades[(df_trades['action'] == 'SHORT') & (df_trades['pnl'] > 0)])
        
        report = {
            "version": "v5.1.5_HYBRID",
            "backtest_completed": datetime.now().isoformat(),
            "summary": {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": round(win_rate, 2),
                "total_pnl": round(total_pnl, 2),
                "total_return_pct": round(total_return_pct, 2),
                "initial_capital": self.initial_capital,
                "final_capital": round(self.capital, 2),
                "max_drawdown_pct": round(max_dd, 2),
                "profit_factor": round(profit_factor, 2),
                "sharpe_ratio": round(sharpe, 2)
            },
            "trade_quality": {
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "avg_win_r": round(avg_win_r, 2),
                "avg_loss_r": round(avg_loss_r, 2),
                "avg_realized_r": round(avg_r, 2),
                "avg_bars_held": round(avg_bars_held, 1)
            },
            "direction_breakdown": {
                "long_trades": long_trades,
                "long_wins": long_wins,
                "long_win_rate": round((long_wins/long_trades)*100, 2) if long_trades > 0 else 0,
                "short_trades": short_trades,
                "short_wins": short_wins,
                "short_win_rate": round((short_wins/short_trades)*100, 2) if short_trades > 0 else 0
            },
            "trailing_stop_stats": {
                "trailing_exits": trailing_exits,
                "trailing_exit_rate": round((trailing_exits/total_trades)*100, 2) if total_trades > 0 else 0,
                **trailing_stats
            },
            "trades": self.trades,
            "equity_curve": self.equity_curve
        }
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"BACKTEST v5.1.5 HYBRID RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Total Return: {total_return_pct:+.2f}%")
        logger.info(f"Final Capital: ${self.capital:,.2f}")
        logger.info(f"Max Drawdown: {max_dd:.2f}%")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"")
        logger.info(f"Avg Win: ${avg_win:.2f} ({avg_win_r:+.2f}R)")
        logger.info(f"Avg Loss: ${avg_loss:.2f} ({avg_loss_r:.2f}R)")
        logger.info(f"Avg R: {avg_r:+.2f}R")
        logger.info(f"Avg Bars Held: {avg_bars_held:.1f}")
        logger.info(f"")
        logger.info(f"Trailing Exits: {trailing_exits} ({(trailing_exits/total_trades)*100:.1f}%)")
        logger.info(f"Trailing Events: {trailing_stats.get('total_events', 0)}")
        logger.info(f"{'='*60}\n")
        
        return report


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Run v5.1.5 hybrid backtest."""
    
    # Load historical data
    logger.info("Loading historical data...")
    df = pd.read_csv('/tmp/luxor-v7-python/data/btcusdt_daily_1000.csv')
    
    # Fix: Ensure proper DatetimeIndex
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    else:
        raise ValueError("No timestamp or date column found")
    
    logger.info(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Initialize system
    logger.info("Initializing Luxor V7 Prana system...")
    system = LuxorV7PranaSystem()
    
    # Initialize backtest engine with default hybrid parameters
    logger.info("Initializing backtest engine v5.1.5...")
    engine = BacktestEngineV515(
        initial_capital=10000.0,
        risk_per_trade=0.01,
        min_rr_ratio=1.30,
        enable_shorts=True,  # Enable for testing
        min_hold_bars=3,
        trailing_params=None  # Use defaults
    )
    
    # Run backtest
    results = engine.run(df, system)
    
    # Save results
    output_path = '/mnt/user-data/outputs/backtest_v515_hybrid_baseline.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
