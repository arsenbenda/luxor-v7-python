# ============================================================
# LUXOR V7 PRANA - BACKTEST MODULE v5.1.1
# Walk-Forward Analysis with Performance Metrics
# Compatible with increased min_bars requirements
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Single trade record."""
    id: int
    entry_date: str
    exit_date: Optional[str]
    direction: str  # LONG or SHORT
    entry_price: float
    exit_price: Optional[float]
    stop_loss: float
    take_profit: float
    position_size: float
    confidence: str
    status: str  # OPEN, WIN, LOSS, STOPPED
    pnl: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0
    exit_reason: str = ""
    signal_data: Dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Complete backtest results."""
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    max_drawdown_duration: int
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    avg_trade_pnl: float
    avg_winner: float
    avg_loser: float
    largest_winner: float
    largest_loser: float
    avg_bars_held: float
    long_trades: int
    short_trades: int
    long_win_rate: float
    short_win_rate: float
    equity_curve: List[Dict]
    trades: List[Dict]
    monthly_returns: Dict[str, float]
    signals_generated: int
    signals_traded: int
    signals_skipped: int
    data_quality: Dict


class LuxorBacktester:
    """
    Backtest engine for LUXOR V7 PRANA system.
    
    Features:
    - Walk-forward analysis with rolling windows
    - Position tracking with realistic execution
    - Comprehensive performance metrics
    - Equity curve generation
    - Trade logging with signal data
    - Compatible with v5.1.1 increased min_bars
    """
    
    def __init__(self, system, initial_capital: float = 10000):
        """
        Initialize backtester.
        
        Args:
            system: LuxorV7PranaSystem instance
            initial_capital: Starting capital in USD
        """
        self.system = system
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.open_position: Optional[Trade] = None
        self.trade_counter = 0
        self.signals_generated = 0
        self.signals_traded = 0
        self.signals_skipped = 0
        self.skip_reasons: Dict[str, int] = {}
    
    def reset(self):
        """Reset backtester state."""
        self.capital = self.initial_capital
        self.trades = []
        self.equity_curve = []
        self.open_position = None
        self.trade_counter = 0
        self.signals_generated = 0
        self.signals_traded = 0
        self.signals_skipped = 0
        self.skip_reasons = {}
    
    def run(self, df_historical: pd.DataFrame,
            lookback_days: int = 400,
            min_confidence: str = "LOW",
            allow_shorts: bool = True,
            max_position_pct: float = 0.25,
            slippage_pct: float = 0.001,
            commission_pct: float = 0.001,
            require_valid_rr: bool = True) -> BacktestResult:
        """
        Run walk-forward backtest.
        
        Args:
            df_historical: Complete historical OHLCV DataFrame
            lookback_days: Rolling window size for signal generation (default 400 for v5.1.1)
            min_confidence: Minimum confidence to take trade (HIGH, MEDIUM, LOW)
            allow_shorts: Whether to take short trades
            max_position_pct: Maximum position size as % of capital
            slippage_pct: Slippage percentage (default 0.1%)
            commission_pct: Commission percentage per trade (default 0.1%)
            require_valid_rr: Only trade setups with valid R:R >= 1.5
            
        Returns:
            BacktestResult with all metrics
        """
        self.reset()
        logger.info(f"[BACKTEST] Starting with {len(df_historical)} candles, lookback={lookback_days}")
        
        # Validate DataFrame
        df = self.system.validate_dataframe(df_historical)
        
        # v5.1.1: Need more data for increased min_bars
        min_required = lookback_days + 50
        if len(df) < min_required:
            raise ValueError(f"Insufficient data: need {min_required}, have {len(df)}. Consider using lookback_days=300 or fetching more historical data.")
        
        confidence_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}
        min_conf_value = confidence_order.get(min_confidence, 1)
        
        logger.info(f"[BACKTEST] Config: min_confidence={min_confidence}, allow_shorts={allow_shorts}, max_position={max_position_pct*100}%, require_valid_rr={require_valid_rr}")
        
        # Walk-forward loop
        for i in range(lookback_days, len(df)):
            current_date = df.iloc[i]['date'] if 'date' in df.columns else df.iloc[i]['timestamp']
            current_price = float(df.iloc[i]['close'])
            current_high = float(df.iloc[i]['high'])
            current_low = float(df.iloc[i]['low'])
            
            # Get reference date
            try:
                ref_date = pd.to_datetime(current_date)
                if hasattr(ref_date, 'tzinfo') and ref_date.tzinfo:
                    ref_date = ref_date.replace(tzinfo=None)
            except:
                ref_date = datetime.now()
            
            # Check open position for exit
            if self.open_position:
                self._check_exit(current_high, current_low, current_price,
                               str(current_date)[:10], slippage_pct, commission_pct)
            
            # Generate signal using rolling window
            df_window = df.iloc[i - lookback_days:i + 1].copy()
            
            try:
                signal = self.system.generate_mtf_signal(
                    df_historical=df_window,
                    reference_date=ref_date
                )
                
                if signal.get("status") == "success":
                    self.signals_generated += 1
                    
                    # Check for new entry
                    if self.open_position is None:
                        self._evaluate_entry(
                            signal=signal,
                            current_price=current_price,
                            current_date=str(current_date)[:10],
                            min_conf_value=min_conf_value,
                            confidence_order=confidence_order,
                            allow_shorts=allow_shorts,
                            max_position_pct=max_position_pct,
                            slippage_pct=slippage_pct,
                            commission_pct=commission_pct,
                            require_valid_rr=require_valid_rr
                        )
                        
            except Exception as e:
                logger.debug(f"[BACKTEST] Signal error at index {i}: {e}")
                continue
            
            # Update equity curve
            equity = self._calculate_equity(current_price)
            self.equity_curve.append({
                "date": str(current_date)[:10],
                "equity": round(equity, 2),
                "price": round(current_price, 2),
                "position": self.open_position.direction if self.open_position else "FLAT"
            })
            
            # Progress logging
            if i % 100 == 0:
                pct = 100 * (i - lookback_days) / (len(df) - lookback_days)
                logger.info(f"[BACKTEST] Progress: {pct:.1f}% | Trades: {len(self.trades)} | Equity: ${equity:,.2f}")
        
        # Close any open position at end
        if self.open_position:
            final_price = float(df.iloc[-1]['close'])
            final_date = str(df.iloc[-1]['date'] if 'date' in df.columns else df.iloc[-1]['timestamp'])[:10]
            self._close_position(final_price, final_date, "END_OF_TEST", slippage_pct, commission_pct)
        
        # Calculate metrics
        result = self._calculate_metrics(df)
        
        logger.info(f"[BACKTEST] Complete: {result.total_trades} trades, {result.win_rate:.1f}% WR, {result.total_return_pct:+.2f}% return, {result.max_drawdown_pct:.2f}% max DD")
        
        return result
    
    def _evaluate_entry(self, signal: Dict, current_price: float, current_date: str,
                       min_conf_value: int, confidence_order: Dict,
                       allow_shorts: bool, max_position_pct: float,
                       slippage_pct: float, commission_pct: float,
                       require_valid_rr: bool):
        """Evaluate signal for trade entry."""
        
        trade_setups = signal.get("trade_setups", [])
        if not trade_setups:
            self._record_skip("no_setup")
            return
        
        setup = trade_setups[0]
        direction = setup.get("direction", "WAIT")
        confidence = setup.get("confidence", "NONE")
        conf_value = confidence_order.get(confidence, 0)
        
        # Check direction
        if direction in ["WAIT", "FLAT", None]:
            self._record_skip("wait_signal")
            return
        
        # Check shorts allowed
        if direction == "SHORT" and not allow_shorts:
            self._record_skip("shorts_disabled")
            return
        
        # Check confidence
        if conf_value < min_conf_value:
            self._record_skip(f"low_confidence_{confidence}")
            return
        
        # Check R:R validity
        if require_valid_rr and not setup.get("valid", False):
            self._record_skip(f"invalid_rr_{setup.get('rr_ratio', 0):.2f}")
            return
        
        # Check regime restrictions
        regime = signal.get("regime", {})
        if direction == "LONG" and not regime.get("allows_long", True):
            self._record_skip("regime_blocks_long")
            return
        if direction == "SHORT" and not regime.get("allows_short", True):
            self._record_skip("regime_blocks_short")
            return
        
        # Enter trade
        position_size = min(setup.get("position_size", 0.25), max_position_pct)
        
        self._enter_trade(
            direction=direction,
            entry_price=current_price,
            stop_loss=setup.get("stop_loss", current_price * (0.95 if direction == "LONG" else 1.05)),
            take_profit=setup.get("tp1", current_price * (1.05 if direction == "LONG" else 0.95)),
            position_size=position_size,
            confidence=confidence,
            entry_date=current_date,
            slippage_pct=slippage_pct,
            commission_pct=commission_pct,
            signal_data={
                "primary_bias": signal.get("primary_bias", {}).get("primary_bias"),
                "regime": regime.get("current"),
                "consensus": signal.get("consensus", {}).get("verdict"),
                "rr_ratio": setup.get("rr_ratio"),
                "gann_context": signal.get("gann_context", {}).get("position")
            }
        )
        self.signals_traded += 1
    
    def _record_skip(self, reason: str):
        """Record skipped signal reason."""
        self.signals_skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1
    
    def _enter_trade(self, direction: str, entry_price: float, stop_loss: float,
                    take_profit: float, position_size: float, confidence: str,
                    entry_date: str, slippage_pct: float, commission_pct: float,
                    signal_data: Dict = None):
        """Enter a new trade."""
        self.trade_counter += 1
        
        # Apply slippage
        if direction == "LONG":
            actual_entry = entry_price * (1 + slippage_pct)
        else:
            actual_entry = entry_price * (1 - slippage_pct)
        
        # Deduct commission
        position_value = self.capital * position_size
        commission = position_value * commission_pct
        self.capital -= commission
        
        self.open_position = Trade(
            id=self.trade_counter,
            entry_date=entry_date,
            exit_date=None,
            direction=direction,
            entry_price=round(actual_entry, 2),
            exit_price=None,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            position_size=round(position_size, 4),
            confidence=confidence,
            status="OPEN",
            signal_data=signal_data or {}
        )
        
        logger.debug(f"[TRADE #{self.trade_counter}] ENTER {direction} @ ${actual_entry:.2f}, SL=${stop_loss:.2f}, TP=${take_profit:.2f}")
    
    def _check_exit(self, high: float, low: float, close: float,
                   current_date: str, slippage_pct: float, commission_pct: float):
        """Check if open position should be exited."""
        if not self.open_position:
            return
        
        pos = self.open_position
        
        if pos.direction == "LONG":
            if low <= pos.stop_loss:
                self._close_position(pos.stop_loss, current_date, "STOP_LOSS", slippage_pct, commission_pct)
            elif high >= pos.take_profit:
                self._close_position(pos.take_profit, current_date, "TAKE_PROFIT", slippage_pct, commission_pct)
        else:  # SHORT
            if high >= pos.stop_loss:
                self._close_position(pos.stop_loss, current_date, "STOP_LOSS", slippage_pct, commission_pct)
            elif low <= pos.take_profit:
                self._close_position(pos.take_profit, current_date, "TAKE_PROFIT", slippage_pct, commission_pct)
        
        # Update bars held
        if self.open_position:
            self.open_position.bars_held += 1
    
    def _close_position(self, exit_price: float, exit_date: str, reason: str,
                       slippage_pct: float, commission_pct: float):
        """Close the open position."""
        if not self.open_position:
            return
        
        pos = self.open_position
        
        # Apply slippage
        if pos.direction == "LONG":
            actual_exit = exit_price * (1 - slippage_pct)
        else:
            actual_exit = exit_price * (1 + slippage_pct)
        
        # Calculate PnL
        position_value = self.capital * pos.position_size
        
        if pos.direction == "LONG":
            pnl_pct = (actual_exit - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - actual_exit) / pos.entry_price
        
        pnl = position_value * pnl_pct
        
        # Apply commission
        commission = position_value * commission_pct
        pnl -= commission
        
        # Update capital
        self.capital += pnl
        
        # Update trade record
        pos.exit_date = exit_date
        pos.exit_price = round(actual_exit, 2)
        pos.pnl = round(pnl, 2)
        pos.pnl_pct = round(pnl_pct * 100, 2)
        pos.exit_reason = reason
        pos.status = "WIN" if pnl > 0 else "LOSS"
        
        self.trades.append(pos)
        self.open_position = None
        
        logger.debug(f"[TRADE #{pos.id}] EXIT {pos.direction} @ ${actual_exit:.2f}, PnL=${pnl:+.2f} ({pnl_pct*100:+.2f}%), Reason={reason}")
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity including unrealized PnL."""
        equity = self.capital
        
        if self.open_position:
            pos = self.open_position
            position_value = self.capital * pos.position_size
            
            if pos.direction == "LONG":
                unrealized_pnl = position_value * ((current_price - pos.entry_price) / pos.entry_price)
            else:
                unrealized_pnl = position_value * ((pos.entry_price - current_price) / pos.entry_price)
            
            equity += unrealized_pnl
        
        return equity
    
    def _calculate_metrics(self, df: pd.DataFrame) -> BacktestResult:
        """Calculate all performance metrics."""
        
        # Basic stats
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.status == "WIN"])
        losing_trades = len([t for t in self.trades if t.status == "LOSS"])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # PnL stats
        wins = [t.pnl for t in self.trades if t.status == "WIN"]
        losses = [abs(t.pnl) for t in self.trades if t.status == "LOSS"]
        
        total_profit = sum(wins)
        total_loss = sum(losses)
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
        
        avg_trade_pnl = sum(t.pnl for t in self.trades) / total_trades if total_trades > 0 else 0
        avg_winner = sum(wins) / len(wins) if wins else 0
        avg_loser = sum(losses) / len(losses) if losses else 0
        
        largest_winner = max(wins) if wins else 0
        largest_loser = max(losses) if losses else 0
        
        # Bars held
        avg_bars_held = sum(t.bars_held for t in self.trades) / total_trades if total_trades > 0 else 0
        
        # Direction breakdown
        long_trades = [t for t in self.trades if t.direction == "LONG"]
        short_trades = [t for t in self.trades if t.direction == "SHORT"]
        long_wins = len([t for t in long_trades if t.status == "WIN"])
        short_wins = len([t for t in short_trades if t.status == "WIN"])
        long_win_rate = (long_wins / len(long_trades) * 100) if long_trades else 0
        short_win_rate = (short_wins / len(short_trades) * 100) if short_trades else 0
        
        # Drawdown calculation
        equity_values = [e["equity"] for e in self.equity_curve]
        max_drawdown_pct, max_dd_duration = self._calculate_drawdown(equity_values)
        
        # Risk-adjusted returns
        sharpe_ratio, sortino_ratio, calmar_ratio = self._calculate_risk_metrics(equity_values, max_drawdown_pct)
        
        # Monthly returns
        monthly_returns = self._calculate_monthly_returns()
        
        # Get dates
        start_date = str(df.iloc[0]['date'] if 'date' in df.columns else df.iloc[0]['timestamp'])[:10]
        end_date = str(df.iloc[-1]['date'] if 'date' in df.columns else df.iloc[-1]['timestamp'])[:10]
        
        # Total return
        total_return_pct = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=round(self.capital, 2),
            total_return_pct=round(total_return_pct, 2),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=round(win_rate, 2),
            profit_factor=round(profit_factor, 2) if profit_factor != float('inf') else 999.99,
            max_drawdown_pct=round(max_drawdown_pct, 2),
            max_drawdown_duration=max_dd_duration,
            sharpe_ratio=round(sharpe_ratio, 2),
            sortino_ratio=round(sortino_ratio, 2),
            calmar_ratio=round(calmar_ratio, 2),
            avg_trade_pnl=round(avg_trade_pnl, 2),
            avg_winner=round(avg_winner, 2),
            avg_loser=round(avg_loser, 2),
            largest_winner=round(largest_winner, 2),
            largest_loser=round(largest_loser, 2),
            avg_bars_held=round(avg_bars_held, 1),
            long_trades=len(long_trades),
            short_trades=len(short_trades),
            long_win_rate=round(long_win_rate, 2),
            short_win_rate=round(short_win_rate, 2),
            equity_curve=self.equity_curve,
            trades=[self._trade_to_dict(t) for t in self.trades],
            monthly_returns=monthly_returns,
            signals_generated=self.signals_generated,
            signals_traded=self.signals_traded,
            signals_skipped=self.signals_skipped,
            data_quality={
                "total_candles": len(df),
                "skip_reasons": self.skip_reasons
            }
        )
    
    def _calculate_drawdown(self, equity_values: List[float]) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        if not equity_values:
            return 0, 0
        
        peak = equity_values[0]
        max_dd = 0
        max_dd_duration = 0
        current_dd_duration = 0
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
                current_dd_duration = 0
            else:
                dd = (peak - equity) / peak * 100
                current_dd_duration += 1
                if dd > max_dd:
                    max_dd = dd
                if current_dd_duration > max_dd_duration:
                    max_dd_duration = current_dd_duration
        
        return max_dd, max_dd_duration
    
    def _calculate_risk_metrics(self, equity_values: List[float], max_dd: float) -> Tuple[float, float, float]:
        """Calculate Sharpe, Sortino, and Calmar ratios."""
        if len(equity_values) < 2:
            return 0, 0, 0
        
        returns = pd.Series(equity_values).pct_change().dropna()
        
        if len(returns) == 0 or returns.std() == 0:
            return 0, 0, 0
        
        # Annualized metrics (assuming daily data)
        annual_return = returns.mean() * 252
        annual_std = returns.std() * np.sqrt(252)
        
        sharpe = annual_return / annual_std if annual_std > 0 else 0
        
        # Sortino (downside deviation)
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else annual_std
        sortino = annual_return / downside_std if downside_std > 0 else 0
        
        # Calmar (return / max drawdown)
        total_return = (equity_values[-1] - equity_values[0]) / equity_values[0] * 100
        calmar = total_return / max_dd if max_dd > 0 else 0
        
        return sharpe, sortino, calmar
    
    def _calculate_monthly_returns(self) -> Dict[str, float]:
        """Calculate monthly returns."""
        monthly = {}
        
        for entry in self.equity_curve:
            month = entry["date"][:7]  # YYYY-MM
            if month not in monthly:
                monthly[month] = {"start": entry["equity"], "end": entry["equity"]}
            monthly[month]["end"] = entry["equity"]
        
        returns = {}
        for month, data in monthly.items():
            ret = ((data["end"] - data["start"]) / data["start"]) * 100 if data["start"] > 0 else 0
            returns[month] = round(ret, 2)
        
        return returns
    
    def _trade_to_dict(self, trade: Trade) -> Dict:
        """Convert Trade to dictionary."""
        return {
            "id": trade.id,
            "entry_date": trade.entry_date,
            "exit_date": trade.exit_date,
            "direction": trade.direction,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "stop_loss": trade.stop_loss,
            "take_profit": trade.take_profit,
            "position_size": trade.position_size,
            "confidence": trade.confidence,
            "status": trade.status,
            "pnl": trade.pnl,
            "pnl_pct": trade.pnl_pct,
            "bars_held": trade.bars_held,
            "exit_reason": trade.exit_reason,
            "signal_data": trade.signal_data
        }
    
    def save_results(self, result: BacktestResult, filepath: str):
        """Save backtest results to JSON file."""
        data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "version": self.system.VERSION,
                "initial_capital": result.initial_capital
            },
            "summary": {
                "start_date": result.start_date,
                "end_date": result.end_date,
                "final_capital": result.final_capital,
                "total_return_pct": result.total_return_pct,
                "total_trades": result.total_trades,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "max_drawdown_pct": result.max_drawdown_pct,
                "sharpe_ratio": result.sharpe_ratio,
                "sortino_ratio": result.sortino_ratio,
                "calmar_ratio": result.calmar_ratio
            },
            "detailed_metrics": {
                "winning_trades": result.winning_trades,
                "losing_trades": result.losing_trades,
                "avg_trade_pnl": result.avg_trade_pnl,
                "avg_winner": result.avg_winner,
                "avg_loser": result.avg_loser,
                "largest_winner": result.largest_winner,
                "largest_loser": result.largest_loser,
                "avg_bars_held": result.avg_bars_held,
                "long_trades": result.long_trades,
                "short_trades": result.short_trades,
                "long_win_rate": result.long_win_rate,
                "short_win_rate": result.short_win_rate,
                "signals_generated": result.signals_generated,
                "signals_traded": result.signals_traded,
                "signals_skipped": result.signals_skipped
            },
            "data_quality": result.data_quality,
            "monthly_returns": result.monthly_returns,
            "trades": result.trades,
            "equity_curve": result.equity_curve
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"[BACKTEST] Results saved to {filepath}")
    
    def print_summary(self, result: BacktestResult):
        """Print backtest summary to console."""
        print("\n" + "=" * 70)
        print("LUXOR V7 PRANA v5.1.1 - BACKTEST RESULTS")
        print("=" * 70)
        print(f"Period: {result.start_date} to {result.end_date}")
        print(f"Initial Capital: ${result.initial_capital:,.2f}")
        print(f"Final Capital:   ${result.final_capital:,.2f}")
        print(f"Total Return:    {result.total_return_pct:+.2f}%")
        print("-" * 70)
        print(f"Total Trades:    {result.total_trades}")
        print(f"Win Rate:        {result.win_rate:.1f}%")
        print(f"Profit Factor:   {result.profit_factor:.2f}")
        print(f"Max Drawdown:    {result.max_drawdown_pct:.2f}% ({result.max_drawdown_duration} days)")
        print("-" * 70)
        print(f"Sharpe Ratio:    {result.sharpe_ratio:.2f}")
        print(f"Sortino Ratio:   {result.sortino_ratio:.2f}")
        print(f"Calmar Ratio:    {result.calmar_ratio:.2f}")
        print("-" * 70)
        print(f"Avg Trade PnL:   ${result.avg_trade_pnl:+.2f}")
        print(f"Avg Winner:      ${result.avg_winner:+.2f}")
        print(f"Avg Loser:       ${result.avg_loser:.2f}")
        print(f"Largest Winner:  ${result.largest_winner:+.2f}")
        print(f"Largest Loser:   ${result.largest_loser:.2f}")
        print("-" * 70)
        print(f"Long Trades:     {result.long_trades} (WR: {result.long_win_rate:.1f}%)")
        print(f"Short Trades:    {result.short_trades} (WR: {result.short_win_rate:.1f}%)")
        print(f"Avg Bars Held:   {result.avg_bars_held:.1f}")
        print("-" * 70)
        print(f"Signals Generated: {result.signals_generated}")
        print(f"Signals Traded:    {result.signals_traded}")
        print(f"Signals Skipped:   {result.signals_skipped}")
        if result.data_quality.get("skip_reasons"):
            print("Skip Reasons:")
            for reason, count in sorted(result.data_quality["skip_reasons"].items(), key=lambda x: -x[1])[:5]:
                print(f"  - {reason}: {count}")
        print("=" * 70 + "\n")


# ============================================================
# EXAMPLE USAGE
# ============================================================

def run_backtest_example():
    """Example backtest run."""
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("\n" + "=" * 70)
    print("LUXOR V7 PRANA v5.1.1 - BACKTEST EXAMPLE")
    print("=" * 70)
    
    # Initialize system
    try:
        from luxor_v7_prana import LuxorV7PranaSystem
        system = LuxorV7PranaSystem(initial_capital=10000)
        print(f"System initialized: v{system.VERSION}")
    except ImportError as e:
        print(f"Error: Could not import LuxorV7PranaSystem: {e}")
        print("Make sure luxor_v7_prana.py is in the same directory")
        return
    
    # Load historical data
    print("\nFetching historical data...")
    try:
        df = system.fetch_real_binance_data(use_cache=False, symbol="BTCUSDT")
        print(f"Loaded {len(df)} daily candles")
        print(f"Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        print("\nYou can also load data from CSV:")
        print("  df = pd.read_csv('btc_historical.csv')")
        return
    
    # Check data sufficiency for v5.1.1 min_bars
    print("\nv5.1.1 min_bars requirements:")
    print("  1M: 24 monthly bars (need ~750 daily candles)")
    print("  1W: 52 weekly bars (need ~365 daily candles)")
    print("  3D: 90 3-day bars (need ~270 daily candles)")
    print("  1D: 200 daily bars")
    
    if len(df) < 400:
        print(f"\nWarning: Only {len(df)} candles available. Recommend 500+ for reliable backtest.")
    
    # Initialize backtester
    backtester = LuxorBacktester(system, initial_capital=10000)
    
    # Run backtest
    print("\nRunning backtest...")
    print("Config: lookback=350, min_confidence=MEDIUM, allow_shorts=True")
    
    try:
        result = backtester.run(
            df_historical=df,
            lookback_days=350,  # Increased for v5.1.1 min_bars
            min_confidence="MEDIUM",
            allow_shorts=True,
            max_position_pct=0.25,
            slippage_pct=0.001,
            commission_pct=0.001,
            require_valid_rr=True
        )
        
        # Print summary
        backtester.print_summary(result)
        
        # Save results
        output_file = "backtest_results_v511.json"
        backtester.save_results(result, output_file)
        print(f"Results saved to: {output_file}")
        
        return result
        
    except Exception as e:
        print(f"Backtest error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_backtest_example()
