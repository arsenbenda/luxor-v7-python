# ============================================================
# LUXOR V7 PRANA - GANN EGYPT-INDIA UNIFIED SYSTEM v5.1.1
# COMPLETE VERSION - Increased min_bars for Reliable Backtests
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings
import logging
import math

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ============================================================
# HELPERS
# ============================================================

def convert_numpy_types(obj):
    """Convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj

def safe_float(value, default=0.0):
    """Safe float conversion."""
    if value is None:
        return default
    try:
        result = float(value)
        return default if np.isnan(result) else result
    except:
        return default

def safe_round(value, decimals=2):
    """Safe rounding."""
    val = safe_float(value)
    return round(val, decimals)

def filter_valid_levels(levels: List, current_price: float, direction: str) -> List:
    """Filter out None, 0, and invalid levels for TP/Stop calculation."""
    valid = []
    for level in levels:
        if level is None or level == 0:
            continue
        if direction == "above" and level > current_price:
            valid.append(level)
        elif direction == "below" and level < current_price:
            valid.append(level)
        elif direction == "any":
            valid.append(level)
    return sorted(valid) if direction == "above" else sorted(valid, reverse=True)

# ============================================================
# CONFIGURATION - v5.1.1 INCREASED min_bars
# ============================================================

class Regime(Enum):
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    RANGING = "RANGING"
    CAPITULATION = "CAPITULATION"
    EUPHORIA = "EUPHORIA"

@dataclass
class TimeframeConfig:
    name: str
    base_weight: float
    gann_lookback: int
    min_bars: int
    trend_weight: float
    range_weight: float

# v5.1.1: INCREASED min_bars for reliable backtests
# Logic: More bars = better statistical reliability for indicators
TIMEFRAME_CONFIGS = {
    "1M": TimeframeConfig(
        name="1M",
        base_weight=0.35,
        gann_lookback=24,    # 2 years lookback for Gann
        min_bars=24,         # 2 years minimum (was 8)
        trend_weight=0.40,
        range_weight=0.25
    ),
    "1W": TimeframeConfig(
        name="1W",
        base_weight=0.30,
        gann_lookback=52,    # 1 year lookback for Gann
        min_bars=52,         # 1 year minimum (was 20)
        trend_weight=0.35,
        range_weight=0.30
    ),
    "3D": TimeframeConfig(
        name="3D",
        base_weight=0.20,
        gann_lookback=120,   # ~1 year of 3D bars
        min_bars=90,         # ~9 months minimum (was 45)
        trend_weight=0.15,
        range_weight=0.25
    ),
    "1D": TimeframeConfig(
        name="1D",
        base_weight=0.15,
        gann_lookback=252,   # 1 trading year
        min_bars=200,        # ~10 months minimum (was 100)
        trend_weight=0.10,
        range_weight=0.20
    ),
}

# Indicator thresholds
RSI_OVERSOLD = 30
RSI_EXTREME_OVERSOLD = 25
RSI_OVERBOUGHT = 70
RSI_EXTREME_OVERBOUGHT = 75
ADX_STRONG_TREND = 25
ADX_VERY_STRONG = 50
VOLUME_SPIKE_THRESHOLD = 2.0
MIN_RR_RATIO = 1.5

# Gann cycles (days)
GANN_CYCLES = [30, 45, 60, 90, 120, 144, 180, 270, 360]

# ============================================================
# MAIN CLASS
# ============================================================

class LuxorV7PranaSystem:
    """LUXOR V7 PRANA - Complete v5.1.1 with Increased min_bars"""
    
    CACHE = {'df': None, 'last_fetch': None, 'cache_duration': 3600}
    VERSION = "5.1.1"
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self._ccxt_available = self._check_ccxt()
        logger.info(f"[INIT] LuxorV7PranaSystem v{self.VERSION} (ccxt: {self._ccxt_available})")
    
    def _check_ccxt(self) -> bool:
        """Check if ccxt is available."""
        try:
            import ccxt
            return True
        except ImportError:
            return False
    
    # ========================================================
    # DATA FETCHING
    # ========================================================
    
    def fetch_ohlcv_ccxt(self, symbol: str = "BTC/USDT", interval: str = "1d", limit: int = 750) -> pd.DataFrame:
        """Fetch OHLCV with multi-exchange fallback. Increased default limit for better backtests."""
        if not self._ccxt_available:
            raise ImportError("ccxt not available - use df_historical parameter instead")
        
        import ccxt
        base = symbol.split('/')[0].upper() if '/' in symbol else symbol[:3].upper()
        
        # Multi-exchange fallback (Binance excluded for geo-restrictions)
        exchanges = [
            ('kucoin', f'{base}/USDT'),
            ('bybit', f'{base}/USDT'),
            ('okx', f'{base}/USDT'),
            ('kraken', f'{base}/USD'),
            ('gate', f'{base}/USDT'),
        ]
        
        last_error = None
        for exchange_id, sym in exchanges:
            try:
                logger.info(f"[DATA] Trying {exchange_id} for {sym}")
                exc_class = getattr(ccxt, exchange_id, None)
                if not exc_class:
                    continue
                
                exc = exc_class({'enableRateLimit': True, 'timeout': 30000})
                ohlcv = exc.fetch_ohlcv(sym, interval, limit=limit)
                
                if ohlcv and len(ohlcv) > 0:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['date'] = df['timestamp']
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                    df = df.dropna(subset=['close']).reset_index(drop=True)
                    logger.info(f"[DATA] Got {len(df)} candles from {exchange_id}")
                    return df
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[DATA] {exchange_id} failed: {str(e)[:80]}")
                continue
        
        raise Exception(f"All exchanges failed. Last error: {last_error}")
    
    def fetch_real_binance_data(self, use_cache: bool = True, symbol: str = "BTCUSDT") -> pd.DataFrame:
        """Fetch with caching - uses multi-exchange fallback."""
        if use_cache and self.CACHE['df'] is not None and self.CACHE['last_fetch']:
            age = (datetime.now() - self.CACHE['last_fetch']).total_seconds()
            if age < self.CACHE['cache_duration']:
                logger.info(f"[DATA] Using cached data ({len(self.CACHE['df'])} candles)")
                return self.CACHE['df'].copy()
        
        ccxt_sym = symbol[:-4] + '/USDT' if symbol.endswith('USDT') and '/' not in symbol else symbol
        df = self.fetch_ohlcv_ccxt(ccxt_sym, "1d", 750)  # Increased to 750 for better backtest coverage
        self.CACHE['df'] = df.copy()
        self.CACHE['last_fetch'] = datetime.now()
        return df
    
    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize DataFrame for processing."""
        if df is None or len(df) == 0:
            raise ValueError("DataFrame is empty or None")
        
        df = df.copy()
        
        # Ensure required columns
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Add volume if missing
        if 'volume' not in df.columns:
            df['volume'] = 0.0
        
        # Ensure date column
        if 'date' not in df.columns:
            if 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp'])
            elif df.index.dtype == 'datetime64[ns]':
                df['date'] = df.index
                df = df.reset_index(drop=True)
            else:
                df['date'] = pd.date_range(end=datetime.now(), periods=len(df), freq='D')
        
        # Ensure timestamp column
        if 'timestamp' not in df.columns:
            df['timestamp'] = df['date']
        
        # Convert to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        
        # Drop NaN in close
        df = df.dropna(subset=['close']).reset_index(drop=True)
        
        return df
    
    def resample_ohlcv(self, df_1d: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """Resample daily data to higher timeframe."""
        df = df_1d.copy()
        idx = 'date' if 'date' in df.columns else 'timestamp'
        df.set_index(idx, inplace=True)
        
        rule_map = {'3D': '3D', '1W': 'W', '1M': 'ME'}  # ME for month-end (pandas 2.x)
        rule = rule_map.get(target_tf, '1D')
        
        try:
            resampled = df.resample(rule).agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna()
        except Exception:
            # Fallback for older pandas
            if rule == 'ME':
                rule = 'M'
            resampled = df.resample(rule).agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna()
        
        resampled.reset_index(inplace=True)
        resampled.rename(columns={resampled.columns[0]: 'timestamp'}, inplace=True)
        resampled['date'] = resampled['timestamp']
        return resampled
    
    # ========================================================
    # INDICATORS
    # ========================================================
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI with Wilder smoothing."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.inf)
        return (100 - (100 / (1 + rs))).fillna(50)
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal, Histogram."""
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal, macd - signal
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX with upper cap at 100."""
        high, low, close = df['high'], df['low'], df['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff().abs() * -1
        plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
        minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0)
        
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, min_periods=period).mean().replace(0, 1)
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1)
        
        adx = dx.ewm(alpha=1/period, min_periods=period).mean().fillna(0)
        return adx.clip(upper=100)  # Cap at 100
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.fillna(tr.mean() if len(tr) > 0 else 1.0)
    
    def calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate SMA."""
        return prices.rolling(window=period).mean()
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate EMA."""
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_bollinger(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return {
            'upper': upper,
            'middle': sma,
            'lower': lower,
            'bandwidth': ((upper - lower) / sma * 100).fillna(0)
        }
    
    def calculate_ichimoku(self, df: pd.DataFrame) -> Dict:
        """Calculate Ichimoku Cloud components."""
        high, low, close = df['high'], df['low'], df['close']
        
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        chikou = close.shift(-26)
        
        # Future cloud (leading indicator)
        future_a = (tenkan + kijun) / 2
        future_b = (high.rolling(52).max() + low.rolling(52).min()) / 2
        
        return {
            'tenkan': tenkan.fillna(close),
            'kijun': kijun.fillna(close),
            'senkou_a': senkou_a.fillna(close),
            'senkou_b': senkou_b.fillna(close),
            'chikou': chikou.fillna(close),
            'future_a': future_a.fillna(close),
            'future_b': future_b.fillna(close)
        }
    
    # ========================================================
    # PIVOT & DIVERGENCE DETECTION
    # ========================================================
    
    def find_pivots(self, prices: pd.Series, left: int = 5, right: int = 5) -> Dict:
        """Find pivot highs and lows using left/right bar confirmation."""
        highs, lows = [], []
        prices_list = prices.values
        
        for i in range(left, len(prices_list) - right):
            curr = prices_list[i]
            
            # Check pivot high
            is_high = True
            for j in range(1, left + 1):
                if prices_list[i - j] >= curr:
                    is_high = False
                    break
            if is_high:
                for j in range(1, right + 1):
                    if prices_list[i + j] >= curr:
                        is_high = False
                        break
            
            # Check pivot low
            is_low = True
            for j in range(1, left + 1):
                if prices_list[i - j] <= curr:
                    is_low = False
                    break
            if is_low:
                for j in range(1, right + 1):
                    if prices_list[i + j] <= curr:
                        is_low = False
                        break
            
            if is_high:
                highs.append({'idx': i, 'price': float(curr)})
            if is_low:
                lows.append({'idx': i, 'price': float(curr)})
        
        return {'highs': highs, 'lows': lows}
    
    def detect_divergence(self, df: pd.DataFrame, rsi: pd.Series, lookback: int = 50) -> Dict:
        """Detect RSI divergence with pivot-based method + simple fallback."""
        if len(df) < lookback:
            return {"bullish": False, "bearish": False, "method": "insufficient_data", "confidence": 0}
        
        recent_df = df.tail(lookback).reset_index(drop=True)
        recent_rsi = rsi.tail(lookback).reset_index(drop=True)
        
        price_pivots = self.find_pivots(recent_df['close'], 3, 3)
        rsi_pivots = self.find_pivots(recent_rsi, 3, 3)
        
        bullish, bearish = False, False
        confidence = 0.0
        method = "pivot"
        desc = "No divergence detected"
        
        # Bullish divergence: Price Lower Low + RSI Higher Low
        if len(price_pivots['lows']) >= 2 and len(rsi_pivots['lows']) >= 2:
            p_lows = price_pivots['lows'][-2:]
            r_lows = rsi_pivots['lows'][-2:]
            
            if p_lows[1]['price'] < p_lows[0]['price'] and r_lows[1]['price'] > r_lows[0]['price']:
                bullish = True
                price_diff = (p_lows[0]['price'] - p_lows[1]['price']) / p_lows[0]['price']
                rsi_diff = r_lows[1]['price'] - r_lows[0]['price']
                confidence = min(0.85, 0.5 + price_diff * 8 + rsi_diff / 80)
                desc = f"Bullish: Price LL ({p_lows[1]['price']:.0f}<{p_lows[0]['price']:.0f}), RSI HL ({r_lows[1]['price']:.1f}>{r_lows[0]['price']:.1f})"
        
        # Bearish divergence: Price Higher High + RSI Lower High
        if len(price_pivots['highs']) >= 2 and len(rsi_pivots['highs']) >= 2:
            p_highs = price_pivots['highs'][-2:]
            r_highs = rsi_pivots['highs'][-2:]
            
            if p_highs[1]['price'] > p_highs[0]['price'] and r_highs[1]['price'] < r_highs[0]['price']:
                bearish = True
                price_diff = (p_highs[1]['price'] - p_highs[0]['price']) / p_highs[0]['price']
                rsi_diff = r_highs[0]['price'] - r_highs[1]['price']
                confidence = min(0.85, 0.5 + price_diff * 8 + rsi_diff / 80)
                desc = f"Bearish: Price HH ({p_highs[1]['price']:.0f}>{p_highs[0]['price']:.0f}), RSI LH ({r_highs[1]['price']:.1f}<{r_highs[0]['price']:.1f})"
        
        # Fallback to simple method if no pivots found
        if not bullish and not bearish:
            if len(price_pivots['lows']) < 2 or len(rsi_pivots['lows']) < 2:
                method = "simple"
                half = lookback // 2
                
                # Simple bullish check
                p1_low = recent_df['close'].iloc[:half].min()
                p2_low = recent_df['close'].iloc[half:].min()
                r1_low = recent_rsi.iloc[:half].min()
                r2_low = recent_rsi.iloc[half:].min()
                
                if p2_low < p1_low and r2_low > r1_low:
                    bullish = True
                    confidence = 0.45
                    desc = "Bullish divergence (simple method)"
                
                # Simple bearish check
                p1_high = recent_df['close'].iloc[:half].max()
                p2_high = recent_df['close'].iloc[half:].max()
                r1_high = recent_rsi.iloc[:half].max()
                r2_high = recent_rsi.iloc[half:].max()
                
                if p2_high > p1_high and r2_high < r1_high:
                    bearish = True
                    confidence = 0.45
                    desc = "Bearish divergence (simple method)"
        
        return {
            "bullish": bullish,
            "bearish": bearish,
            "method": method,
            "confidence": safe_round(confidence, 2),
            "description": desc
        }
    
    # ========================================================
    # GANN LEVELS
    # ========================================================
    
    def calculate_gann_levels(self, df: pd.DataFrame, lookback: int) -> Dict:
        """Calculate Gann 8-level grid from swing high/low."""
        lookback = min(lookback, len(df))
        recent = df.tail(lookback)
        
        high = safe_float(recent['high'].max())
        low = safe_float(recent['low'].min())
        range_val = high - low
        current = safe_float(df['close'].iloc[-1])
        
        # Get dates of high/low
        ts_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        try:
            high_idx = recent['high'].idxmax()
            low_idx = recent['low'].idxmin()
            high_date = str(df.loc[high_idx, ts_col])[:10]
            low_date = str(df.loc[low_idx, ts_col])[:10]
        except:
            high_date, low_date = "N/A", "N/A"
        
        # Calculate all 8 levels
        levels = {}
        for i in range(9):
            level_price = low + (range_val * i / 8)
            levels[f"{i}_8"] = safe_round(level_price, 2)
        
        gann_50 = safe_round(low + range_val * 0.5, 2)
        
        # Determine current position in grid
        if current >= levels["7_8"]:
            zone = "EXTREME_HIGH"
        elif current >= levels["5_8"]:
            zone = "HIGH"
        elif current >= levels["4_8"]:
            zone = "MID_HIGH"
        elif current >= levels["3_8"]:
            zone = "MID_LOW"
        elif current >= levels["1_8"]:
            zone = "LOW"
        else:
            zone = "EXTREME_LOW"
        
        return {
            "high": safe_round(high, 2),
            "low": safe_round(low, 2),
            "high_date": high_date,
            "low_date": low_date,
            "range": safe_round(range_val, 2),
            "range_pct": safe_round((range_val / current) * 100 if current > 0 else 0, 2),
            "lookback": lookback,
            "levels": levels,
            "gann_50": gann_50,
            "current_zone": zone
        }
    
    def analyze_gann_context(self, price: float, gann: Dict, df: pd.DataFrame, atr: float) -> Dict:
        """Analyze price position relative to Gann 50% with cross detection."""
        g50 = gann["gann_50"]
        
        distance = price - g50
        dist_pct = (distance / g50) * 100 if g50 > 0 else 0
        dist_atr = distance / atr if atr > 0 else 0
        
        # Extended cross detection (last 10 bars)
        closes = df['close'].tail(10).tolist()
        cross_above = any(closes[i-1] < g50 <= closes[i] for i in range(1, len(closes)))
        cross_below = any(closes[i-1] > g50 >= closes[i] for i in range(1, len(closes)))
        
        # Determine position and bias
        if price > g50 + atr:
            position, bias = "STRONG_ABOVE", "BULLISH"
        elif price > g50:
            position = "ABOVE"
            bias = "BULLISH_BREAKOUT" if cross_above else "BULLISH"
        elif price < g50 - atr:
            position, bias = "STRONG_BELOW", "BEARISH"
        elif price < g50:
            position = "BELOW"
            bias = "BEARISH_BREAKDOWN" if cross_below else "BEARISH"
        else:
            position, bias = "AT_50", "NEUTRAL"
        
        return {
            "position": position,
            "bias": bias,
            "gann_50": g50,
            "distance": safe_round(distance, 2),
            "distance_pct": safe_round(dist_pct, 2),
            "distance_atr": safe_round(dist_atr, 2),
            "cross_above": cross_above,
            "cross_below": cross_below,
            "description": f"Price ${price:,.0f} is {position} Gann 50% (${g50:,.0f}), distance {dist_pct:+.1f}%"
        }
    
    # ========================================================
    # CAPITULATION DETECTION
    # ========================================================
    
    def detect_capitulation(self, df_w: pd.DataFrame, df_d: pd.DataFrame,
                           w_rsi: float, gann: Dict, price: float) -> Dict:
        """Weighted scoring system for capitulation detection."""
        score = 0.0
        criteria = []
        
        # 1. RSI Score (0-3 points)
        if w_rsi < 20:
            score += 3
            criteria.append({"name": "RSI_EXTREME", "score": 3, "value": f"{w_rsi:.1f}"})
        elif w_rsi < 25:
            score += 2.5
            criteria.append({"name": "RSI_VERY_LOW", "score": 2.5, "value": f"{w_rsi:.1f}"})
        elif w_rsi < 30:
            score += 1.5
            criteria.append({"name": "RSI_LOW", "score": 1.5, "value": f"{w_rsi:.1f}"})
        elif w_rsi < 35:
            score += 0.5
            criteria.append({"name": "RSI_OVERSOLD", "score": 0.5, "value": f"{w_rsi:.1f}"})
        
        # 2. Volume Score (0-3 points)
        vol_ratio = 1.0
        if len(df_d) >= 20 and 'volume' in df_d.columns:
            avg_vol = safe_float(df_d['volume'].tail(20).mean(), 1)
            current_vol = safe_float(df_d['volume'].iloc[-1], 0)
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        
        if vol_ratio >= 3.0:
            score += 3
            criteria.append({"name": "VOL_EXTREME", "score": 3, "value": f"{vol_ratio:.1f}x"})
        elif vol_ratio >= 2.0:
            score += 2
            criteria.append({"name": "VOL_SPIKE", "score": 2, "value": f"{vol_ratio:.1f}x"})
        elif vol_ratio >= 1.5:
            score += 1
            criteria.append({"name": "VOL_HIGH", "score": 1, "value": f"{vol_ratio:.1f}x"})
        
        # 3. Gann Support Score (0-3 points)
        g28 = gann["levels"].get("2_8", price * 0.88)
        g38 = gann["levels"].get("3_8", price * 0.94)
        
        d28 = abs(price - g28) / price if price > 0 else 1
        d38 = abs(price - g38) / price if price > 0 else 1
        
        if d28 < 0.02:
            score += 3
            criteria.append({"name": "GANN_2_8", "score": 3, "value": f"${g28:,.0f}"})
        elif d38 < 0.03:
            score += 2
            criteria.append({"name": "GANN_3_8", "score": 2, "value": f"${g38:,.0f}"})
        elif d38 < 0.05:
            score += 1
            criteria.append({"name": "GANN_NEAR", "score": 1, "value": "Near support"})
        
        # 4. Divergence Score (0-3 points)
        if len(df_w) >= 20:
            w_rsi_series = self.calculate_rsi(df_w['close'])
            div = self.detect_divergence(df_w, w_rsi_series, min(30, len(df_w)))
            
            if div["bullish"]:
                div_score = 1 + (div["confidence"] * 2)
                score += div_score
                criteria.append({"name": "BULL_DIV", "score": round(div_score, 1), "value": div["description"]})
        
        # Calculate status
        max_score = 12
        score_pct = score / max_score
        
        if score_pct >= 0.70:
            status, confidence = "CONFIRMED", 0.85
        elif score_pct >= 0.50:
            status, confidence = "POTENTIAL", 0.60
        elif score_pct >= 0.30:
            status, confidence = "DEVELOPING", 0.40
        else:
            status, confidence = "NONE", 0.15
        
        return {
            "is_capitulation": score_pct >= 0.50,
            "status": status,
            "score": safe_round(score, 1),
            "max_score": max_score,
            "score_pct": safe_round(score_pct * 100, 1),
            "confidence": safe_round(confidence, 2),
            "criteria": criteria,
            "details": {
                "weekly_rsi": safe_round(w_rsi, 2),
                "volume_ratio": safe_round(vol_ratio, 2),
                "gann_2_8": gann["levels"].get("2_8"),
                "gann_3_8": gann["levels"].get("3_8")
            }
        }
    
    # ========================================================
    # REGIME DETECTION
    # ========================================================
    
    def determine_regime(self, df_d: pd.DataFrame, w_rsi: float, w_adx: float,
                        cap: Dict, price: float) -> Dict:
        """Determine market regime with override logic."""
        # Calculate SMAs
        sma200 = safe_float(df_d['close'].rolling(200).mean().iloc[-1]) if len(df_d) >= 200 else price
        sma50 = safe_float(df_d['close'].rolling(50).mean().iloc[-1]) if len(df_d) >= 50 else price
        
        vs200 = "ABOVE" if price > sma200 else "BELOW"
        dist_pct = ((price - sma200) / sma200) * 100 if sma200 > 0 else 0
        
        # ADX strength
        if w_adx > ADX_VERY_STRONG:
            strength = "VERY_STRONG"
        elif w_adx > ADX_STRONG_TREND:
            strength = "STRONG"
        else:
            strength = "WEAK"
        
        warnings = []
        override = False
        override_reason = None
        
        # Regime determination with override
        if cap.get("is_capitulation"):
            regime = Regime.CAPITULATION
            override = True
            override_reason = f"Capitulation {cap['status']} (score {cap['score']}/{cap['max_score']})"
            allow_short, allow_long = False, True
            size_cap = 0.50 if cap["status"] == "CONFIRMED" else 0.25
            warnings.append("CAPITULATION: Shorts blocked, consider accumulation")
        elif w_rsi > RSI_EXTREME_OVERBOUGHT:
            regime = Regime.EUPHORIA
            override = True
            override_reason = f"Euphoria - Weekly RSI {w_rsi:.0f}"
            allow_short, allow_long = True, False
            size_cap = 0.25
            warnings.append("EUPHORIA: New longs blocked, watch for reversal")
        elif vs200 == "ABOVE" and strength in ["STRONG", "VERY_STRONG"]:
            regime = Regime.TRENDING_BULL
            allow_short, allow_long = True, True
            size_cap = 0.75 if w_rsi > 65 else 1.0
            if w_rsi > 65:
                warnings.append("RSI elevated in uptrend - reduce size")
        elif vs200 == "BELOW" and strength in ["STRONG", "VERY_STRONG"]:
            regime = Regime.TRENDING_BEAR
            allow_short, allow_long = True, True
            size_cap = 1.0
            if w_rsi < 35:
                warnings.append("RSI oversold in downtrend - watch for bounce")
        else:
            regime = Regime.RANGING
            allow_short, allow_long = True, True
            size_cap = 0.75
        
        return {
            "current": regime.value,
            "strength": safe_round(min(w_adx / 100, 1.0), 2),
            "strength_label": f"{strength} (ADX {w_adx:.0f})",
            "override_active": override,
            "override_reason": override_reason,
            "allows_short": allow_short,
            "allows_long": allow_long,
            "position_size_cap": safe_round(size_cap, 2),
            "warnings": warnings,
            "sma_200": safe_round(sma200, 2),
            "sma_50": safe_round(sma50, 2),
            "price_vs_sma_200": vs200,
            "sma_distance_pct": safe_round(dist_pct, 2)
        }
    
    # ========================================================
    # TIMEFRAME ANALYSIS
    # ========================================================
    
    def analyze_timeframe(self, df: pd.DataFrame, tf: str) -> Dict:
        """Complete analysis for a single timeframe."""
        cfg = TIMEFRAME_CONFIGS[tf]
        
        # Check data sufficiency
        if len(df) < cfg.min_bars:
            logger.warning(f"[{tf}] Insufficient data: {len(df)} < {cfg.min_bars} required")
            return self._empty_tf_result(tf, data_sufficient=False, bars_available=len(df), bars_required=cfg.min_bars)
        
        price = safe_float(df['close'].iloc[-1])
        
        # Calculate indicators
        rsi_series = self.calculate_rsi(df['close'])
        rsi = safe_float(rsi_series.iloc[-1], 50)
        
        macd_line, macd_signal, macd_hist = self.calculate_macd(df['close'])
        macd_h = safe_float(macd_hist.iloc[-1])
        macd_cross = "BULLISH" if macd_line.iloc[-1] > macd_signal.iloc[-1] and macd_line.iloc[-2] <= macd_signal.iloc[-2] else \
                     "BEARISH" if macd_line.iloc[-1] < macd_signal.iloc[-1] and macd_line.iloc[-2] >= macd_signal.iloc[-2] else "NONE"
        
        adx = safe_float(self.calculate_adx(df).iloc[-1])
        atr = safe_float(self.calculate_atr(df).iloc[-1])
        
        sma50 = safe_float(self.calculate_sma(df['close'], 50).iloc[-1]) if len(df) >= 50 else price
        sma200 = safe_float(self.calculate_sma(df['close'], 200).iloc[-1]) if len(df) >= 200 else price
        ema21 = safe_float(self.calculate_ema(df['close'], 21).iloc[-1]) if len(df) >= 21 else price
        
        # Bollinger Bands
        bb = self.calculate_bollinger(df['close'])
        bb_upper = safe_float(bb['upper'].iloc[-1])
        bb_lower = safe_float(bb['lower'].iloc[-1])
        bb_mid = safe_float(bb['middle'].iloc[-1])
        bb_width = safe_float(bb['bandwidth'].iloc[-1])
        
        # Ichimoku
        ichi = self.calculate_ichimoku(df)
        tenkan = safe_float(ichi['tenkan'].iloc[-1])
        kijun = safe_float(ichi['kijun'].iloc[-1])
        senkou_a = safe_float(ichi['senkou_a'].iloc[-1])
        senkou_b = safe_float(ichi['senkou_b'].iloc[-1])
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        future_a = safe_float(ichi['future_a'].iloc[-1])
        future_b = safe_float(ichi['future_b'].iloc[-1])
        future_bullish = future_a > future_b
        
        # TK Cross detection
        tk_cross = "BULLISH" if tenkan > kijun and ichi['tenkan'].iloc[-2] <= ichi['kijun'].iloc[-2] else \
                   "BEARISH" if tenkan < kijun and ichi['tenkan'].iloc[-2] >= ichi['kijun'].iloc[-2] else "NONE"
        
        # Gann levels
        gann = self.calculate_gann_levels(df, cfg.gann_lookback)
        g50 = gann["gann_50"]
        
        # Divergence
        div = self.detect_divergence(df, rsi_series, min(50, len(df)))
        
        # Volume analysis
        vol_ratio = 1.0
        vol_trend = "NORMAL"
        if 'volume' in df.columns and len(df) >= 20:
            avg_vol = safe_float(df['volume'].tail(20).mean(), 1)
            current_vol = safe_float(df['volume'].iloc[-1], 0)
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
            vol_trend = "HIGH" if vol_ratio >= 1.5 else "LOW" if vol_ratio < 0.5 else "NORMAL"
        
        # Weighted scoring for direction
        bull_score, bear_score = 0.0, 0.0
        signals = {}
        
        # RSI (weight: 1.0)
        if rsi < RSI_OVERSOLD:
            bull_score += 1.0
            signals["RSI"] = f"OVERSOLD ({rsi:.1f})"
        elif rsi > RSI_OVERBOUGHT:
            bear_score += 1.0
            signals["RSI"] = f"OVERBOUGHT ({rsi:.1f})"
        else:
            signals["RSI"] = f"NEUTRAL ({rsi:.1f})"
        
        # MACD (weight: 1.0)
        if macd_h > 0:
            bull_score += 1.0
            signals["MACD"] = "BULLISH"
        else:
            bear_score += 1.0
            signals["MACD"] = "BEARISH"
        
        # TK Cross (weight: 0.8)
        if tenkan > kijun:
            bull_score += 0.8
            signals["TK"] = "BULLISH"
        else:
            bear_score += 0.8
            signals["TK"] = "BEARISH"
        
        # Cloud position (weight: 1.2)
        if price > cloud_top:
            bull_score += 1.2
            signals["CLOUD"] = "ABOVE"
            vs_cloud = "ABOVE"
        elif price < cloud_bottom:
            bear_score += 1.2
            signals["CLOUD"] = "BELOW"
            vs_cloud = "BELOW"
        else:
            signals["CLOUD"] = "INSIDE"
            vs_cloud = "INSIDE"
        
        # Gann 50% (weight: 1.5 - highest)
        if price > g50:
            bull_score += 1.5
            signals["GANN_50"] = "ABOVE"
            vs_g50 = "ABOVE"
        else:
            bear_score += 1.5
            signals["GANN_50"] = "BELOW"
            vs_g50 = "BELOW"
        
        # SMA 200 (weight: 1.0)
        if price > sma200:
            bull_score += 1.0
            signals["SMA200"] = "ABOVE"
        else:
            bear_score += 1.0
            signals["SMA200"] = "BELOW"
        
        # Future cloud (weight: 0.5)
        if future_bullish:
            bull_score += 0.5
            signals["FUTURE_CLOUD"] = "BULLISH"
        else:
            bear_score += 0.5
            signals["FUTURE_CLOUD"] = "BEARISH"
        
        # Divergence bonus
        if div["bullish"]:
            bull_score += div["confidence"]
            signals["DIVERGENCE"] = "BULLISH"
        elif div["bearish"]:
            bear_score += div["confidence"]
            signals["DIVERGENCE"] = "BEARISH"
        
        # Normalize scores to 0-100
        total_score = bull_score + bear_score
        bull_pct = (bull_score / total_score) * 100 if total_score > 0 else 50
        
        # Direction determination (stricter for 1M)
        if tf == "1M":
            if bull_pct > 65:
                direction = "BULLISH"
            elif bull_pct < 35:
                direction = "BEARISH"
            else:
                direction = "NEUTRAL"
        else:
            if bull_pct > 55:
                direction = "BULLISH"
            elif bull_pct < 45:
                direction = "BEARISH"
            else:
                direction = "NEUTRAL"
        
        # State name
        state = self._determine_state_name(rsi, adx, direction)
        
        return {
            "timeframe": tf,
            "direction": direction,
            "state_name": state,
            "data_sufficient": True,
            "bars_available": len(df),
            "bars_required": cfg.min_bars,
            "bullish_score": safe_round(bull_score, 2),
            "bearish_score": safe_round(bear_score, 2),
            "bullish_pct": safe_round(bull_pct, 1),
            "current_price": safe_round(price, 2),
            "rsi": safe_round(rsi, 2),
            "macd_histogram": safe_round(macd_h, 4),
            "macd_cross": macd_cross,
            "adx": safe_round(adx, 2),
            "adx_label": "VERY_STRONG" if adx > ADX_VERY_STRONG else "STRONG" if adx > ADX_STRONG_TREND else "WEAK",
            "atr": safe_round(atr, 2),
            "atr_pct": safe_round((atr / price) * 100 if price > 0 else 0, 2),
            "sma_50": safe_round(sma50, 2),
            "sma_200": safe_round(sma200, 2),
            "ema_21": safe_round(ema21, 2),
            "price_vs_sma_200": "ABOVE" if price > sma200 else "BELOW",
            "bollinger": {
                "upper": safe_round(bb_upper, 2),
                "middle": safe_round(bb_mid, 2),
                "lower": safe_round(bb_lower, 2),
                "bandwidth": safe_round(bb_width, 2)
            },
            "volume_ratio": safe_round(vol_ratio, 2),
            "volume_trend": vol_trend,
            "gann_high": gann["high"],
            "gann_low": gann["low"],
            "gann_50_pct": g50,
            "gann_zone": gann["current_zone"],
            "price_vs_gann_50": vs_g50,
            "divergence": div,
            "signal_details": signals,
            "ichimoku": {
                "tenkan": safe_round(tenkan, 2),
                "kijun": safe_round(kijun, 2),
                "cloud_top": safe_round(cloud_top, 2),
                "cloud_bottom": safe_round(cloud_bottom, 2),
                "tk_cross": tk_cross,
                "price_vs_cloud": vs_cloud,
                "future_cloud": "BULLISH" if future_bullish else "BEARISH"
            },
            "gann": gann
        }
    
    def _empty_tf_result(self, tf: str, data_sufficient: bool = False, bars_available: int = 0, bars_required: int = 0) -> Dict:
        """Return empty timeframe result with proper structure."""
        return {
            "timeframe": tf,
            "direction": "NEUTRAL",
            "state_name": "Insufficient Data",
            "data_sufficient": data_sufficient,
            "bars_available": bars_available,
            "bars_required": bars_required,
            "bullish_score": 0,
            "bearish_score": 0,
            "bullish_pct": 50,
            "current_price": 0,
            "rsi": 50,
            "macd_histogram": 0,
            "macd_cross": "NONE",
            "adx": 0,
            "adx_label": "N/A",
            "atr": 0,
            "atr_pct": 0,
            "sma_50": 0,
            "sma_200": 0,
            "ema_21": 0,
            "price_vs_sma_200": "N/A",
            "bollinger": {"upper": 0, "middle": 0, "lower": 0, "bandwidth": 0},
            "volume_ratio": 1,
            "volume_trend": "N/A",
            "gann_high": 0,
            "gann_low": 0,
            "gann_50_pct": 0,
            "gann_zone": "N/A",
            "price_vs_gann_50": "N/A",
            "divergence": {"bullish": False, "bearish": False, "method": "N/A", "confidence": 0},
            "signal_details": {},
            "ichimoku": {
                "tenkan": 0, "kijun": 0, "cloud_top": 0, "cloud_bottom": 0,
                "tk_cross": "N/A", "price_vs_cloud": "N/A", "future_cloud": "N/A"
            },
            "gann": {"high": 0, "low": 0, "levels": {}, "gann_50": 0, "current_zone": "N/A"}
        }
    
    def _determine_state_name(self, rsi: float, adx: float, direction: str) -> str:
        """Determine market state name based on indicators."""
        if rsi < 25:
            return "Capitulation"
        if rsi < 35:
            return "Fear"
        if rsi > 75:
            return "Euphoria"
        if rsi > 65:
            return "Greed"
        if adx > 50:
            return "Expansion" if direction == "BULLISH" else "Contraction"
        if adx < 20:
            return "Consolidation"
        return "Transition"
    
    # ========================================================
    # CONSENSUS CALCULATION
    # ========================================================
    
    def calculate_consensus(self, timeframes: Dict, regime: Dict) -> Dict:
        """Calculate MTF consensus with adaptive weights."""
        trending = regime["current"] in ["TRENDING_BULL", "TRENDING_BEAR"]
        
        weighted_bull, weighted_bear = 0.0, 0.0
        alignment_count = 0
        conflicts = []
        valid_tfs = 0
        tf_directions = {}
        
        for tf_name, analysis in timeframes.items():
            if not analysis.get("data_sufficient", True):
                tf_directions[tf_name] = "INSUFFICIENT_DATA"
                continue
            
            valid_tfs += 1
            cfg = TIMEFRAME_CONFIGS[tf_name]
            weight = cfg.trend_weight if trending else cfg.range_weight
            
            # Normalize scores to 0-100 range
            bull_pct = analysis.get("bullish_pct", 50)
            weighted_bull += weight * bull_pct
            weighted_bear += weight * (100 - bull_pct)
            
            tf_directions[tf_name] = analysis["direction"]
        
        if valid_tfs == 0:
            return self._empty_consensus()
        
        # Calculate weighted score (-100 to +100)
        total_weight = weighted_bull + weighted_bear
        weighted_score = int(((weighted_bull - weighted_bear) / total_weight) * 100) if total_weight > 0 else 0
        
        # Determine primary direction
        if weighted_score > 15:
            primary_direction = "BULLISH"
        elif weighted_score < -15:
            primary_direction = "BEARISH"
        else:
            primary_direction = "NEUTRAL"
        
        # Count alignments and conflicts
        for tf_name, tf_dir in tf_directions.items():
            if tf_dir == "INSUFFICIENT_DATA":
                continue
            if tf_dir == primary_direction:
                alignment_count += 1
            elif tf_dir != "NEUTRAL" and primary_direction != "NEUTRAL":
                conflicts.append({"timeframe": tf_name, "direction": tf_dir})
        
        # Confidence level
        if alignment_count >= 3 and abs(weighted_score) > 40:
            confidence = "HIGH"
        elif alignment_count >= 2 and abs(weighted_score) > 25:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        # Override reduces confidence
        if regime.get("override_active") and confidence == "HIGH":
            confidence = "MEDIUM"
        
        # Verdict
        strength_label = "STRONG" if confidence == "HIGH" else "MODERATE" if confidence == "MEDIUM" else "WEAK"
        if primary_direction == "NEUTRAL":
            verdict = "MIXED/NEUTRAL"
        else:
            verdict = f"{strength_label} {primary_direction}"
        
        return {
            "primary_direction": primary_direction,
            "weighted_score": weighted_score,
            "alignment": f"{alignment_count}/{valid_tfs}",
            "alignment_count": alignment_count,
            "valid_timeframes": valid_tfs,
            "confidence_level": confidence,
            "verdict": verdict,
            "conflicts": conflicts,
            "has_conflicts": len(conflicts) > 0,
            "weight_mode": "TREND" if trending else "RANGE",
            "tf_1m": tf_directions.get("1M", "N/A"),
            "tf_1w": tf_directions.get("1W", "N/A"),
            "tf_3d": tf_directions.get("3D", "N/A"),
            "tf_1d": tf_directions.get("1D", "N/A")
        }
    
    def _empty_consensus(self) -> Dict:
        """Return empty consensus result."""
        return {
            "primary_direction": "NEUTRAL",
            "weighted_score": 0,
            "alignment": "0/0",
            "alignment_count": 0,
            "valid_timeframes": 0,
            "confidence_level": "NONE",
            "verdict": "NO DATA",
            "conflicts": [],
            "has_conflicts": False,
            "weight_mode": "N/A",
            "tf_1m": "N/A",
            "tf_1w": "N/A",
            "tf_3d": "N/A",
            "tf_1d": "N/A"
        }
    
    # ========================================================
    # PRIMARY BIAS DETERMINATION
    # ========================================================
    
    def determine_bias(self, price: float, gann_ctx: Dict, consensus: Dict, regime: Dict) -> Dict:
        """Determine primary bias with Gann 50% override."""
        gann_bias = gann_ctx["bias"]
        consensus_dir = consensus["primary_direction"]
        
        # Extract direction from Gann bias
        if "BULLISH" in gann_bias:
            gann_dir = "BULLISH"
        elif "BEARISH" in gann_bias:
            gann_dir = "BEARISH"
        else:
            gann_dir = "NEUTRAL"
        
        # Check for conflict between Gann and consensus
        conflict = False
        if gann_dir != "NEUTRAL" and gann_dir != consensus_dir and consensus_dir != "NEUTRAL":
            primary = gann_dir
            source = "GANN_OVERRIDE"
            conflict = True
            note = f"Gann 50% overrides consensus ({consensus_dir} → {gann_dir})"
        else:
            primary = consensus_dir if consensus_dir != "NEUTRAL" else gann_dir
            source = "CONSENSUS" if consensus_dir == primary else "GANN"
            note = f"Aligned: {source} = {primary}"
        
        # Calculate confidence
        base_conf = 0.5
        if consensus["confidence_level"] == "HIGH":
            base_conf += 0.25
        elif consensus["confidence_level"] == "MEDIUM":
            base_conf += 0.15
        
        if not conflict:
            base_conf += 0.15
        
        if regime.get("override_active"):
            base_conf -= 0.10
        
        confidence = min(0.95, max(0.20, base_conf))
        
        # Invalidation levels
        g50 = gann_ctx["gann_50"]
        atr_buffer = abs(gann_ctx.get("distance", 0)) * 0.1  # 10% of distance as buffer
        
        if primary == "BULLISH":
            invalidation_price = g50 - atr_buffer
            invalidation_desc = f"Bullish bias invalidated below ${invalidation_price:,.0f}"
        elif primary == "BEARISH":
            invalidation_price = g50 + atr_buffer
            invalidation_desc = f"Bearish bias invalidated above ${invalidation_price:,.0f}"
        else:
            invalidation_price = g50
            invalidation_desc = "Watch for Gann 50% breakout"
        
        return {
            "primary_bias": primary,
            "source": source,
            "gann_direction": gann_dir,
            "consensus_direction": consensus_dir,
            "conflict": conflict,
            "confidence": safe_round(confidence, 2),
            "note": note,
            "invalidation": {
                "price": safe_round(invalidation_price, 2),
                "description": invalidation_desc
            }
        }
    
    # ========================================================
    # TIME FORECAST (GANN CYCLES)
    # ========================================================
    
    def calculate_time_forecast(self, df: pd.DataFrame, gann: Dict, reference_date: datetime = None) -> Dict:
        """Calculate time-based forecast using Gann cycles."""
        if reference_date is None:
            reference_date = datetime.now()
        
        # Find last significant pivot
        ts_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        
        try:
            # Get high/low dates from Gann
            high_date_str = gann.get("high_date", "N/A")
            low_date_str = gann.get("low_date", "N/A")
            
            # Parse dates
            high_date = pd.to_datetime(high_date_str) if high_date_str != "N/A" else None
            low_date = pd.to_datetime(low_date_str) if low_date_str != "N/A" else None
            
            # Determine which pivot is more recent
            if high_date and low_date:
                if high_date > low_date:
                    ref_type = "HIGH"
                    ref_pivot_date = high_date
                    ref_price = gann["high"]
                else:
                    ref_type = "LOW"
                    ref_pivot_date = low_date
                    ref_price = gann["low"]
            elif high_date:
                ref_type = "HIGH"
                ref_pivot_date = high_date
                ref_price = gann["high"]
            elif low_date:
                ref_type = "LOW"
                ref_pivot_date = low_date
                ref_price = gann["low"]
            else:
                return self._empty_time_forecast()
            
            # Ensure timezone-naive
            if ref_pivot_date.tzinfo:
                ref_pivot_date = ref_pivot_date.replace(tzinfo=None)
            if reference_date.tzinfo:
                reference_date = reference_date.replace(tzinfo=None)
            
            days_since = (reference_date - ref_pivot_date).days
            
        except Exception as e:
            logger.warning(f"[TIME] Error parsing pivot dates: {e}")
            return self._empty_time_forecast()
        
        # Calculate next pivot candidates for each cycle
        next_pivots = []
        for cycle in GANN_CYCLES:
            # How many complete cycles have passed?
            cycles_passed = days_since // cycle
            next_cycle_day = (cycles_passed + 1) * cycle
            days_to_next = next_cycle_day - days_since
            
            if days_to_next > 0 and days_to_next <= 180:  # Only show pivots within 6 months
                next_date = reference_date + timedelta(days=days_to_next)
                expected_type = "LOW" if ref_type == "HIGH" else "HIGH"  # Alternate
                
                # Confidence based on cycle strength
                if cycle in [90, 180, 360]:
                    conf = 0.75
                elif cycle in [60, 120, 144]:
                    conf = 0.60
                else:
                    conf = 0.45
                
                next_pivots.append({
                    "cycle": cycle,
                    "days_away": days_to_next,
                    "expected_date": next_date.strftime("%Y-%m-%d"),
                    "expected_type": expected_type,
                    "confidence": conf
                })
        
        # Sort by days away
        next_pivots.sort(key=lambda x: x["days_away"])
        
        # Primary forecast (nearest high-confidence pivot)
        primary = None
        for p in next_pivots:
            if p["confidence"] >= 0.60:
                primary = p
                break
        
        if not primary and next_pivots:
            primary = next_pivots[0]
        
        return {
            "cycle_origin": {
                "reference_type": ref_type,
                "reference_date": ref_pivot_date.strftime("%Y-%m-%d"),
                "reference_price": safe_round(ref_price, 2),
                "days_since": days_since
            },
            "primary_forecast": primary,
            "all_forecasts": next_pivots[:5],  # Top 5
            "gann_cycles_used": GANN_CYCLES
        }
    
    def _empty_time_forecast(self) -> Dict:
        """Return empty time forecast."""
        return {
            "cycle_origin": {
                "reference_type": "N/A",
                "reference_date": "N/A",
                "reference_price": 0,
                "days_since": 0
            },
            "primary_forecast": None,
            "all_forecasts": [],
            "gann_cycles_used": GANN_CYCLES
        }
    
    # ========================================================
    # TRADE SETUP GENERATION
    # ========================================================
    
    def generate_trade_setups(self, price: float, bias: Dict, regime: Dict,
                             gann: Dict, tf_1d: Dict, atr: float) -> List[Dict]:
        """Generate trade setups with R:R validation."""
        setups = []
        
        direction = bias["primary_bias"]
        confidence = bias["confidence"]
        
        # Check regime restrictions
        if direction == "LONG" and not regime.get("allows_long", True):
            return [self._wait_setup("Longs blocked by regime override")]
        if direction == "SHORT" and not regime.get("allows_short", True):
            return [self._wait_setup("Shorts blocked by regime override")]
        if direction == "NEUTRAL":
            return [self._wait_setup("No clear directional bias")]
        
        # Get Gann levels for TP/Stop
        levels = gann.get("levels", {})
        g50 = gann.get("gann_50", price)
        
        # Get cloud levels from 1D analysis
        cloud_top = tf_1d.get("ichimoku", {}).get("cloud_top", price * 1.05)
        cloud_bottom = tf_1d.get("ichimoku", {}).get("cloud_bottom", price * 0.95)
        
        # Position size based on regime
        base_size = 0.25  # 25% default
        size_cap = regime.get("position_size_cap", 1.0)
        
        if confidence >= 0.70:
            position_size = min(0.50, base_size * 2 * size_cap)
            conf_label = "HIGH"
        elif confidence >= 0.50:
            position_size = min(0.35, base_size * 1.4 * size_cap)
            conf_label = "MEDIUM"
        else:
            position_size = min(0.20, base_size * 0.8 * size_cap)
            conf_label = "LOW"
        
        if direction == "BULLISH":
            # LONG setup
            entry = price
            
            # Stop loss: below Gann support or ATR-based
            support_levels = filter_valid_levels(
                [levels.get("3_8"), levels.get("2_8"), cloud_bottom, g50 - atr],
                price, "below"
            )
            stop_loss = support_levels[0] if support_levels else price - (2 * atr)
            
            # Take profits: above resistance levels
            resistance_levels = filter_valid_levels(
                [levels.get("5_8"), levels.get("6_8"), levels.get("7_8"), cloud_top, g50 + atr * 2],
                price, "above"
            )
            
            # Calculate risk
            risk = entry - stop_loss
            if risk <= 0:
                risk = atr
            
            # TP levels with R:R validation
            tp1 = entry + (risk * 1.5)  # 1.5R
            tp2 = entry + (risk * 2.5)  # 2.5R
            tp3 = entry + (risk * 3.5)  # 3.5R
            
            # Adjust to nearest resistance if available
            if resistance_levels:
                if resistance_levels[0] > entry and resistance_levels[0] < tp1:
                    tp1 = resistance_levels[0]
                if len(resistance_levels) > 1 and resistance_levels[1] > tp1:
                    tp2 = min(tp2, resistance_levels[1])
            
            rr_ratio = (tp1 - entry) / risk if risk > 0 else 0
            
            setup = {
                "id": 1,
                "direction": "LONG",
                "entry": safe_round(entry, 2),
                "stop_loss": safe_round(stop_loss, 2),
                "tp1": safe_round(tp1, 2),
                "tp2": safe_round(tp2, 2),
                "tp3": safe_round(tp3, 2),
                "risk": safe_round(risk, 2),
                "risk_pct": safe_round((risk / entry) * 100, 2),
                "rr_ratio": safe_round(rr_ratio, 2),
                "position_size": safe_round(position_size, 2),
                "confidence": conf_label,
                "confidence_score": safe_round(confidence, 2),
                "rationale": f"Bullish bias ({bias['source']}), price above Gann 50%"
            }
            
            # Validate R:R
            if rr_ratio >= MIN_RR_RATIO:
                setup["valid"] = True
                setup["status"] = "VALID"
            else:
                setup["valid"] = False
                setup["status"] = f"R:R too low ({rr_ratio:.2f} < {MIN_RR_RATIO})"
            
            setups.append(setup)
            
        elif direction == "BEARISH":
            # SHORT setup
            entry = price
            
            # Stop loss: above Gann resistance
            resistance_levels = filter_valid_levels(
                [levels.get("5_8"), levels.get("6_8"), cloud_top, g50 + atr],
                price, "above"
            )
            stop_loss = resistance_levels[0] if resistance_levels else price + (2 * atr)
            
            # Take profits: below support levels
            support_levels = filter_valid_levels(
                [levels.get("3_8"), levels.get("2_8"), levels.get("1_8"), cloud_bottom],
                price, "below"
            )
            
            # Calculate risk
            risk = stop_loss - entry
            if risk <= 0:
                risk = atr
            
            # TP levels
            tp1 = entry - (risk * 1.5)
            tp2 = entry - (risk * 2.5)
            tp3 = entry - (risk * 3.5)
            
            # Adjust to nearest support
            if support_levels:
                if support_levels[0] < entry and support_levels[0] > tp1:
                    tp1 = support_levels[0]
                if len(support_levels) > 1 and support_levels[1] < tp1:
                    tp2 = max(tp2, support_levels[1])
            
            rr_ratio = (entry - tp1) / risk if risk > 0 else 0
            
            setup = {
                "id": 1,
                "direction": "SHORT",
                "entry": safe_round(entry, 2),
                "stop_loss": safe_round(stop_loss, 2),
                "tp1": safe_round(tp1, 2),
                "tp2": safe_round(tp2, 2),
                "tp3": safe_round(tp3, 2),
                "risk": safe_round(risk, 2),
                "risk_pct": safe_round((risk / entry) * 100, 2),
                "rr_ratio": safe_round(rr_ratio, 2),
                "position_size": safe_round(position_size, 2),
                "confidence": conf_label,
                "confidence_score": safe_round(confidence, 2),
                "rationale": f"Bearish bias ({bias['source']}), price below Gann 50%"
            }
            
            if rr_ratio >= MIN_RR_RATIO:
                setup["valid"] = True
                setup["status"] = "VALID"
            else:
                setup["valid"] = False
                setup["status"] = f"R:R too low ({rr_ratio:.2f} < {MIN_RR_RATIO})"
            
            setups.append(setup)
        
        return setups if setups else [self._wait_setup("No valid setup")]
    
    def _wait_setup(self, reason: str) -> Dict:
        """Return wait/no-trade setup."""
        return {
            "id": 0,
            "direction": "WAIT",
            "entry": 0,
            "stop_loss": 0,
            "tp1": 0,
            "tp2": 0,
            "tp3": 0,
            "risk": 0,
            "risk_pct": 0,
            "rr_ratio": 0,
            "position_size": 0,
            "confidence": "NONE",
            "confidence_score": 0,
            "valid": False,
            "status": "NO_TRADE",
            "rationale": reason
        }
    
    # ========================================================
    # MAIN SIGNAL GENERATION
    # ========================================================
    
    def generate_mtf_signal(self, symbol: str = "BTCUSDT",
                           df_historical: pd.DataFrame = None,
                           reference_date: datetime = None) -> Dict:
        """
        Generate complete MTF signal.
        
        Args:
            symbol: Trading symbol (default BTCUSDT)
            df_historical: Optional historical DataFrame for backtest
            reference_date: Optional reference date for time calculations
        
        Returns:
            Complete signal dictionary
        """
        try:
            # Get data
            if df_historical is not None:
                df_1d = self.validate_dataframe(df_historical)
                logger.info(f"[SIGNAL] Using provided historical data: {len(df_1d)} candles")
            else:
                df_1d = self.fetch_real_binance_data(symbol=symbol)
                logger.info(f"[SIGNAL] Fetched live data: {len(df_1d)} candles")
            
            if reference_date is None:
                reference_date = datetime.now()
            
            # Current price
            price = safe_float(df_1d['close'].iloc[-1])
            
            # Resample to higher timeframes
            df_3d = self.resample_ohlcv(df_1d, '3D')
            df_1w = self.resample_ohlcv(df_1d, '1W')
            df_1m = self.resample_ohlcv(df_1d, '1M')
            
            logger.info(f"[SIGNAL] Resampled: 1D={len(df_1d)}, 3D={len(df_3d)}, 1W={len(df_1w)}, 1M={len(df_1m)}")
            
            # Analyze each timeframe
            tf_1d_analysis = self.analyze_timeframe(df_1d, "1D")
            tf_3d_analysis = self.analyze_timeframe(df_3d, "3D")
            tf_1w_analysis = self.analyze_timeframe(df_1w, "1W")
            tf_1m_analysis = self.analyze_timeframe(df_1m, "1M")
            
            timeframes = {
                "1M": tf_1m_analysis,
                "1W": tf_1w_analysis,
                "3D": tf_3d_analysis,
                "1D": tf_1d_analysis
            }
            
            # Weekly RSI and ADX for regime/capitulation
            w_rsi = safe_float(tf_1w_analysis.get("rsi", 50), 50)
            w_adx = safe_float(tf_1w_analysis.get("adx", 25), 25)
            
            # Weekly Gann for capitulation
            w_gann = tf_1w_analysis.get("gann", self.calculate_gann_levels(df_1w, 52))
            
            # Capitulation detection
            capitulation = self.detect_capitulation(df_1w, df_1d, w_rsi, w_gann, price)
            
            # Regime determination
            regime = self.determine_regime(df_1d, w_rsi, w_adx, capitulation, price)
            
            # MTF Consensus
            consensus = self.calculate_consensus(timeframes, regime)
            
            # Daily Gann context for bias
            d_gann = tf_1d_analysis.get("gann", self.calculate_gann_levels(df_1d, 252))
            d_atr = safe_float(tf_1d_analysis.get("atr", price * 0.02))
            gann_context = self.analyze_gann_context(price, d_gann, df_1d, d_atr)
            
            # Primary bias
            bias = self.determine_bias(price, gann_context, consensus, regime)
            
            # Time forecast
            time_forecast = self.calculate_time_forecast(df_1d, d_gann, reference_date)
            
            # Trade setups
            trade_setups = self.generate_trade_setups(price, bias, regime, d_gann, tf_1d_analysis, d_atr)
            
            # Price levels summary
            price_levels = {
                "current": safe_round(price, 2),
                "daily_gann_50": d_gann.get("gann_50"),
                "weekly_gann_50": w_gann.get("gann_50"),
                "daily_gann_high": d_gann.get("high"),
                "daily_gann_low": d_gann.get("low"),
                "sma_200": tf_1d_analysis.get("sma_200"),
                "sma_50": tf_1d_analysis.get("sma_50"),
                "cloud_top": tf_1d_analysis.get("ichimoku", {}).get("cloud_top"),
                "cloud_bottom": tf_1d_analysis.get("ichimoku", {}).get("cloud_bottom")
            }
            
            # Build final signal
            signal = {
                "status": "success",
                "version": self.VERSION,
                "symbol": symbol,
                "timestamp": reference_date.isoformat(),
                "current_price": safe_round(price, 2),
                
                # Core analysis
                "primary_bias": bias,
                "regime": regime,
                "consensus": consensus,
                "gann_context": gann_context,
                "capitulation": capitulation,
                
                # Timeframe details
                "timeframes": timeframes,
                
                # Price levels
                "price_levels": price_levels,
                
                # Time analysis
                "time_forecast": time_forecast,
                
                # Trade setups
                "trade_setups": trade_setups,
                
                # Data quality
                "data_quality": {
                    "daily_bars": len(df_1d),
                    "weekly_bars": len(df_1w),
                    "monthly_bars": len(df_1m),
                    "all_tf_sufficient": all(tf.get("data_sufficient", False) for tf in timeframes.values())
                }
            }
            
            return convert_numpy_types(signal)
            
        except Exception as e:
            logger.error(f"[SIGNAL] Error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "version": self.VERSION,
                "symbol": symbol,
                "detail": str(e),
                "timestamp": datetime.now().isoformat()
            }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    system = LuxorV7PranaSystem()
    print(f"\nLUXOR V7 PRANA v{system.VERSION}")
    print("="*50)
    
    # Test signal generation
    signal = system.generate_mtf_signal("BTCUSDT")
    
    if signal["status"] == "success":
        print(f"Symbol: {signal['symbol']}")
        print(f"Price: ${signal['current_price']:,.2f}")
        print(f"Primary Bias: {signal['primary_bias']['primary_bias']}")
        print(f"Confidence: {signal['primary_bias']['confidence']}")
        print(f"Regime: {signal['regime']['current']}")
        print(f"Consensus: {signal['consensus']['verdict']}")
        print(f"MTF Alignment: {signal['consensus']['alignment']}")
        
        if signal['trade_setups']:
            setup = signal['trade_setups'][0]
            print(f"\nTrade Setup:")
            print(f"  Direction: {setup['direction']}")
            print(f"  Entry: ${setup['entry']:,.2f}")
            print(f"  Stop: ${setup['stop_loss']:,.2f}")
            print(f"  TP1: ${setup['tp1']:,.2f}")
            print(f"  R:R: {setup['rr_ratio']}")
    else:
        print(f"Error: {signal.get('detail')}")
