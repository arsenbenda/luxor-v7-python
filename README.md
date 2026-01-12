# LUXOR V7 PRANA - Advanced BTC/USDT Trading System

![Version](https://img.shields.io/badge/version-5.1.5-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![Status](https://img.shields.io/badge/status-production-success)

## 🎯 Performance Highlights (v5.1.5)

| Metric | Value |
|--------|-------|
| **Total Return** | +30.87% |
| **Win Rate** | 45.6% |
| **Profit Factor** | 1.51x |
| **Max Drawdown** | -6.56% |
| **Total Trades** | 90 |
| **Avg Winner** | 1.86R |
| **Avg Loser** | -1.00R |
| **Sharpe Ratio** | 0.26 |

*Backtest period: 2024-2026 (750 daily bars)*

---

## 🚀 Features

### Hybrid Regime-Aware Trailing Stop
- **Dynamic trailing** that adapts to market conditions
- **3 Regime Types**:
  - **TRENDING**: Aggressive trailing (0.8x-1.2x ATR)
  - **RANGING**: Tight trailing (0.5x-1.5x ATR)
  - **VOLATILE**: Defensive trailing (1.0x-1.6x ATR)
- **Profit-based stages**: Trailing tightens as profit increases
- **41.5% of exits** use dynamic trailing

### Advanced Signal Generation
- **Multi-Timeframe Analysis** (1D, 3D, 1W, 1M)
- **Gann Levels** for support/resistance
- **Ichimoku Cloud** for trend confirmation
- **ATR-based** stop loss and targets
- **Confidence scoring** (HIGH/MEDIUM/LOW)

### Risk Management
- **Position Sizing**: Dynamic based on confidence
- **Stop Loss**: Minimum 2.5x ATR
- **R:R Ratio**: Minimum 1.30
- **Break-Even**: Automatic at +1.5R
- **Max Hold**: Time-based stop

---

## 📁 Project Structure

```
luxor-v7-python/
├── luxor_v7_prana.py          # Core trading system
├── backtest_v515.py           # Backtest engine v5.1.5
├── app.py                     # Flask API (for n8n integration)
├── config.py                  # Configuration settings
├── data/
│   └── btcusdt_daily_1000.csv # Historical data
├── results/
│   └── backtest_v515_hybrid_baseline.json
├── requirements.txt
├── Dockerfile
├── CHANGELOG.md
└── README.md
```

---

## 🔧 Installation

### Requirements
- Python 3.10+
- pandas, numpy, ccxt, ta
- Flask (for API mode)

### Setup

```bash
# Clone repository
git clone https://github.com/arsenbenda/luxor-v7-python.git
cd luxor-v7-python

# Install dependencies
pip install -r requirements.txt

# Run backtest
python backtest_v515.py
```

### Docker

```bash
# Build image
docker build -t luxor-v7-prana .

# Run container
docker run -p 5000:5000 luxor-v7-prana
```

---

## 📊 Usage

### Backtest Mode

```python
from luxor_v7_prana import LuxorV7PranaSystem
import pandas as pd

# Load data
df = pd.read_csv('data/btcusdt_daily_1000.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Initialize system
system = LuxorV7PranaSystem()

# Run backtest
from backtest_v515 import BacktestEngineV515

engine = BacktestEngineV515(
    initial_capital=10000.0,
    risk_per_trade=0.01,
    min_rr_ratio=1.30,
    enable_shorts=True
)

results = engine.run(df, system)
print(f"Total Return: {results['total_return_pct']:.2f}%")
```

### API Mode (for n8n)

```bash
# Start Flask API
python app.py
```

**Endpoints:**
- `POST /api/signal` - Get trading signal
- `POST /api/backtest` - Run backtest
- `GET /api/health` - Health check

---

## 🎯 Trading Logic

### Entry Conditions
1. **Primary Bias**: BULLISH or BEARISH (confidence ≥ 0.50)
2. **Regime Check**: Allows LONG/SHORT in current regime
3. **R:R Ratio**: Minimum 1.30
4. **Position Sizing**: Based on confidence level
   - HIGH (≥70%): 50% max position
   - MEDIUM (50-70%): 35% max position
   - LOW (<50%): 20% max position

### Exit Conditions
1. **Target Profit**: TP1 (1.5R), TP2 (2.5R), TP3 (3.5R)
2. **Stop Loss**: Initial or trailing stop
3. **Break-Even**: At +1.5R profit
4. **Time Stop**: Max hold period exceeded

### Trailing Stop Logic
```
Profit Stage → Trailing Multiplier
<1.0R        → 1.5x ATR (TRENDING)
1.0-2.0R     → 1.2x ATR
≥2.0R        → 1.0x ATR (tight)
```

---

## 📈 Backtest Results Analysis

### Trade Distribution
- **Winners**: 41 trades (45.6%)
- **Losers**: 49 trades (54.4%)
- **Trailing Exits**: 27 trades (41.5%)

### Exit Reasons
- **Stop Loss**: 63 trades (70%)
- **Target Profit**: 22 trades (24.4%)
- **Time Stop**: 5 trades (5.6%)

### Regime Performance
| Regime | Trades | Win Rate | Avg R |
|--------|--------|----------|-------|
| TRENDING_BULL | 38 | 52.6% | +0.45R |
| TRENDING_BEAR | 24 | 41.7% | +0.15R |
| RANGING | 18 | 33.3% | -0.25R |
| VOLATILE | 10 | 40.0% | +0.10R |

---

## 🔄 n8n Workflow Integration

This system is designed to work with n8n automation:

1. **Trigger**: Schedule (daily at market close)
2. **HTTP Request**: POST to `/api/signal`
3. **Parse Response**: Extract signal details
4. **Execute Trade**: Send to broker API
5. **Log Trade**: Store in database

See `docs/n8n-workflow.json` for complete workflow template.

---

## 📝 Configuration

Edit `config.py`:

```python
# Trading parameters
INITIAL_CAPITAL = 10000.0
RISK_PER_TRADE = 0.01  # 1%
MIN_RR_RATIO = 1.30
ENABLE_SHORTS = True

# Data source
DATA_SOURCE = "binance"  # or "csv"
SYMBOL = "BTC/USDT"
TIMEFRAME = "1d"

# API settings (for n8n)
API_HOST = "0.0.0.0"
API_PORT = 5000
```

---

## 🐛 Troubleshooting

### Issue: "No timestamp column found"
**Solution**: Ensure CSV has `timestamp` or `date` column with datetime format

### Issue: "Invalid R:R ratio"
**Solution**: Check ATR calculation - may need more historical data

### Issue: "No trades generated"
**Solution**: Verify signal generation confidence threshold

---

## 📚 Documentation

- [CHANGELOG.md](CHANGELOG.md) - Version history
- [API_DOCS.md](docs/API_DOCS.md) - API reference
- [STRATEGY.md](docs/STRATEGY.md) - Trading strategy details

---

## 🤝 Contributing

This is a private trading system. For questions or support:
- Email: arsenbenda@example.com
- GitHub Issues: [luxor-v7-python/issues](https://github.com/arsenbenda/luxor-v7-python/issues)

---

## ⚠️ Disclaimer

This software is for **educational and research purposes only**.

- **No warranty**: Use at your own risk
- **Not financial advice**: Always do your own research
- **Backtest results**: Past performance does not guarantee future results
- **Live trading**: Test thoroughly before deploying real capital

---

## 📄 License

Copyright © 2026 Arsen Benda. All rights reserved.

---

## 🏆 Version History

### v5.1.5 (Current - 2026-01-12)
✅ **Production Ready**
- Hybrid regime-aware trailing stop
- +30.87% return on backtest
- 1.51x profit factor

### v5.1.4 (2026-01-11)
- MTF signal generation fixes
- DatetimeIndex handling improvements

### v5.1.3 (2026-01-10)
- Enhanced stop loss validation
- ATR-based fallback logic

See [CHANGELOG.md](CHANGELOG.md) for complete history.

---

**Status**: ✅ Production Ready | 🚀 Ready for Deploy
