# CHANGELOG - LUXOR V7 PRANA

## v5.1.5 - Hybrid Regime-Aware Trailing Stop (2026-01-12)

### 🎯 Performance Results
- **Total Trades**: 90
- **Win Rate**: 45.6%
- **Profit Factor**: 1.51x
- **Total Return**: +30.87%
- **Max Drawdown**: -6.56%
- **Avg Winner**: 1.86R
- **Avg Loser**: -1.00R
- **Sharpe Ratio**: 0.26

### ✨ New Features
- **Hybrid Trailing Stop System**: Dynamic trailing stop that adapts to market regime
  - TRENDING regime: Aggressive trailing (0.8x-1.2x ATR)
  - RANGING regime: Tight trailing (0.5x-1.5x ATR)
  - VOLATILE regime: Defensive trailing (1.0x-1.6x ATR)
- **Profit-Based Stage Management**: Trailing tightens as profit increases
- **27 Trailing Exits**: 41.5% of all exits use dynamic trailing
- **Break-Even Protection**: Automatic break-even at +1.5R profit

### 🔧 Technical Improvements
- Enhanced regime detection with multi-timeframe confirmation
- ATR-based stop loss calculation (2.5x ATR minimum)
- Proper DatetimeIndex handling for MTF analysis
- Improved trade logging with regime context

### 📊 Backtest Configuration
- **Dataset**: BTC/USDT Daily (750 bars, 2024-2026)
- **Initial Capital**: $10,000
- **Risk per Trade**: 1% (0.01)
- **Min R:R Ratio**: 1.30
- **Shorts Enabled**: Yes
- **Min Hold Bars**: 3

### 🐛 Bug Fixes
- Fixed KeyError in trade comparison reports
- Normalized trade_id handling across all analysis scripts
- Corrected bars_held increment in position management
- Fixed MTF signal generation with proper df_historical parameter passing

---

## v5.1.4 - MTF Signal Generation Fix (2026-01-11)

### 🐛 Bug Fixes
- Fixed DataFrame/Symbol type confusion in generate_mtf_signal()
- Corrected reference_date numpy.int64 conversion issue
- Improved error handling in Binance data fetching

---

## v5.1.3 - Stop Loss Validation (2026-01-10)

### 🔧 Improvements
- Enhanced stop loss validation (LONG: stop < entry; SHORT: stop > entry)
- ATR-based fallback for invalid stops
- Guaranteed minimum R:R of 1.5
- Proper direction-aware TP calculation

---

## Previous Versions
See Git history for earlier releases.

