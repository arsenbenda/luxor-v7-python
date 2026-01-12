# ============================================================
# LUXOR V7 PRANA - GANN EGYPT-INDIA UNIFIED SYSTEM v5.0.10
# FINAL VERIFIED VERSION - All inconsistencies fixed
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
import logging
import ccxt
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
    return valid

# ============================================================
# CONFIGURATION
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

TIMEFRAME_CONFIGS = {
    "1M": TimeframeConfig("1M", 0.35, 24, 12, 0.40, 0.25),
    "1W": TimeframeConfig("1W", 0.30, 52, 26, 0.35, 0.30),
    "3D": TimeframeConfig("3D", 0.20, 120, 60, 0.15, 0.25),
    "1D": TimeframeConfig("1D", 0.15, 252, 100, 0.10, 0.20),
}

RSI_OVERSOLD = 30
RSI_EXTREME_OVERSOLD = 25
RSI_OVERBOUGHT = 70
RSI_EXTREME_OVERBOUGHT = 75
ADX_STRONG_TREND = 25
ADX_VERY_STRONG = 50
VOLUME_SPIKE_THRESHOLD = 2.0
MIN_RR_RATIO = 1.5

GANN_CYCLES = [30, 45, 60, 90, 120, 144, 180, 270, 360]

# ============================================================
# MAIN CLASS
# ============================================================

class LuxorV7PranaSystem:
    """LUXOR V7 PRANA - Final Verified v5.0.10"""
    
    CACHE = {'df': None, 'last_fetch': None, 'cache_duration': 3600}
    VERSION = "5.0.10"
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        logger.info(f"[INIT] LuxorV7PranaSystem v{self.VERSION}")
    
    # ========================================================
    # DATA FETCHING
    # ========================================================
    
    def fetch_ohlcv_ccxt(self, symbol: str = "BTC/USDT", interval: str = "1d", limit: int = 500) -> pd.DataFrame:
        """Fetch OHLCV with multi-exchange fallback."""
        base = symbol.split('/')[0].upper() if '/' in symbol else symbol[:3].upper()
        
        exchanges = [
            ('kucoin', f'{base}/USDT'),
            ('bybit', f'{base}/USDT'),
            ('okx', f'{base}/USDT'),
            ('kraken', f'{base}/USD'),
            ('gate', f'{base}/USDT'),
        ]
        
        for exchange_id, sym in exchanges:
            try:
                logger.info(f"[DATA] Trying {exchange_id}")
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
                logger.warning(f"[DATA] {exchange_id}: {str(e)[:60]}")
                continue
        
        raise Exception("All exchanges failed")
    
    def fetch_real_binance_data(self, use_cache=True, symbol: str = "BTCUSDT") -> pd.DataFrame:
        """Fetch with caching."""
        if use_cache and self.CACHE['df'] is not None and self.CACHE['last_fetch']:
            age = (datetime.now() - self.CACHE['last_fetch']).total_seconds()
            if age < self.CACHE['cache_duration']:
                return self.CACHE['df'].copy()
        
        ccxt_sym = symbol[:-4] + '/USDT' if symbol.endswith('USDT') and '/' not in symbol else symbol
        df = self.fetch_ohlcv_ccxt(ccxt_sym, "1d", 500)
        self.CACHE['df'] = df.copy()
        self.CACHE['last_fetch'] = datetime.now()
        return df
    
    def resample_ohlcv(self, df_1d: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """Resample to higher TF."""
        df = df_1d.copy()
        idx = 'date' if 'date' in df.columns else 'timestamp'
        df.set_index(idx, inplace=True)
        
        rule = {'3D': '3D', '1W': 'W', '1M': 'M'}.get(target_tf, '1D')
        
        resampled = df.resample(rule).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        
        resampled.reset_index(inplace=True)
        resampled.rename(columns={resampled.columns[0]: 'timestamp'}, inplace=True)
        return resampled
    
    # ========================================================
    # INDICATORS
    # ========================================================
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.inf)
        return (100 - (100 / (1 + rs))).fillna(50)
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal, macd - signal
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
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
        # FIX #2: Cap ADX at 100
        return adx.clip(upper=100)
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.fillna(tr.mean() if len(tr) > 0 else 1.0)
    
    def calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        return prices.rolling(window=period).mean()
    
    def calculate_ichimoku(self, df: pd.DataFrame) -> Dict:
        high, low, close = df['high'], df['low'], df['close']
        
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        
        future_a = (tenkan + kijun) / 2
        future_b = (high.rolling(52).max() + low.rolling(52).min()) / 2
        
        return {
            'tenkan': tenkan.fillna(close),
            'kijun': kijun.fillna(close),
            'senkou_a': senkou_a.fillna(close),
            'senkou_b': senkou_b.fillna(close),
            'future_a': future_a.fillna(close),
            'future_b': future_b.fillna(close)
        }
    
    # ========================================================
    # PIVOT & DIVERGENCE
    # ========================================================
    
    def find_pivots(self, prices: pd.Series, left: int = 5, right: int = 5) -> Dict:
        """Find pivot highs/lows."""
        highs, lows = [], []
        
        for i in range(left, len(prices) - right):
            curr = prices.iloc[i]
            
            is_high = all(prices.iloc[i-j] < curr for j in range(1, left+1)) and \
                      all(prices.iloc[i+j] < curr for j in range(1, right+1))
            
            is_low = all(prices.iloc[i-j] > curr for j in range(1, left+1)) and \
                     all(prices.iloc[i+j] > curr for j in range(1, right+1))
            
            if is_high:
                highs.append({'idx': i, 'price': float(curr)})
            if is_low:
                lows.append({'idx': i, 'price': float(curr)})
        
        return {'highs': highs, 'lows': lows}
    
    def detect_divergence(self, df: pd.DataFrame, rsi: pd.Series, lookback: int = 50) -> Dict:
        """Divergence detection with pivot + fallback."""
        if len(df) < lookback:
            return {"bullish": False, "bearish": False, "method": "insufficient_data", "confidence": 0}
        
        recent_df = df.tail(lookback)
        recent_rsi = rsi.tail(lookback)
        
        # Try pivot method first
        price_pivots = self.find_pivots(recent_df['close'].reset_index(drop=True), 3, 3)
        rsi_pivots = self.find_pivots(recent_rsi.reset_index(drop=True), 3, 3)
        
        bullish, bearish = False, False
        confidence = 0
        method = "pivot"
        desc = "No divergence"
        
        # Bullish: price LL, RSI HL
        if len(price_pivots['lows']) >= 2 and len(rsi_pivots['lows']) >= 2:
            p_lows = price_pivots['lows'][-2:]
            r_lows = rsi_pivots['lows'][-2:]
            
            if p_lows[1]['price'] < p_lows[0]['price'] and r_lows[1]['price'] > r_lows[0]['price']:
                bullish = True
                price_diff = (p_lows[0]['price'] - p_lows[1]['price']) / p_lows[0]['price']
                rsi_diff = r_lows[1]['price'] - r_lows[0]['price']
                confidence = min(0.85, 0.5 + price_diff * 8 + rsi_diff / 80)
                desc = f"Bullish: Price LL, RSI HL"
        
        # Bearish: price HH, RSI LH
        if len(price_pivots['highs']) >= 2 and len(rsi_pivots['highs']) >= 2:
            p_highs = price_pivots['highs'][-2:]
            r_highs = rsi_pivots['highs'][-2:]
            
            if p_highs[1]['price'] > p_highs[0]['price'] and r_highs[1]['price'] < r_highs[0]['price']:
                bearish = True
                price_diff = (p_highs[1]['price'] - p_highs[0]['price']) / p_highs[0]['price']
                rsi_diff = r_highs[0]['price'] - r_highs[1]['price']
                confidence = min(0.85, 0.5 + price_diff * 8 + rsi_diff / 80)
                desc = f"Bearish: Price HH, RSI LH"
        
        # FIX #3: Fallback to simple method if no pivots found
        if not bullish and not bearish and (len(price_pivots['lows']) < 2 or len(rsi_pivots['lows']) < 2):
            method = "simple"
            half = lookback // 2
            
            p1_low = recent_df['close'].iloc[:half].min()
            p2_low = recent_df['close'].iloc[half:].min()
            r1_low = recent_rsi.iloc[:half].min()
            r2_low = recent_rsi.iloc[half:].min()
            
            if p2_low < p1_low and r2_low > r1_low:
                bullish = True
                confidence = 0.45
                desc = "Bullish (simple): Price lower, RSI higher"
            
            p1_high = recent_df['close'].iloc[:half].max()
            p2_high = recent_df['close'].iloc[half:].max()
            r1_high = recent_rsi.iloc[:half].max()
            r2_high = recent_rsi.iloc[half:].max()
            
            if p2_high > p1_high and r2_high < r1_high:
                bearish = True
                confidence = 0.45
                desc = "Bearish (simple): Price higher, RSI lower"
        
        return {
            "bullish": bullish,
            "bearish": bearish,
            "method": method,
            "confidence": safe_round(confidence, 2),
            "description": desc
        }
    
    # ========================================================
    # GANN
    # ========================================================
    
    def calculate_gann_levels(self, df: pd.DataFrame, lookback: int) -> Dict:
        """Calculate Gann levels."""
        lookback = min(lookback, len(df))
        recent = df.tail(lookback)
        
        high = safe_float(recent['high'].max())
        low = safe_float(recent['low'].min())
        range_val = high - low
        current = safe_float(df['close'].iloc[-1])
        
        ts_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        try:
            high_idx = recent['high'].idxmax()
            low_idx = recent['low'].idxmin()
            high_date = str(df.loc[high_idx, ts_col])[:10]
            low_date = str(df.loc[low_idx, ts_col])[:10]
        except:
            high_date, low_date = "N/A", "N/A"
        
        levels = {f"{i}_8": safe_round(low + (range_val * i / 8), 2) for i in range(9)}
        gann_50 = safe_round(low + range_val * 0.5, 2)
        
        return {
            "high": safe_round(high, 2),
            "low": safe_round(low, 2),
            "high_date": high_date,
            "low_date": low_date,
            "range": safe_round(range_val, 2),
            "range_pct": safe_round((range_val / current) * 100 if current > 0 else 0, 2),
            "lookback": lookback,
            "levels": levels,
            "gann_50": gann_50
        }
    
    def analyze_gann_context(self, price: float, gann: Dict, df: pd.DataFrame, atr: float) -> Dict:
        """Gann 50% context with extended cross detection."""
        g50 = gann["gann_50"]
        
        distance = price - g50
        dist_pct = (distance / g50) * 100 if g50 > 0 else 0
        dist_atr = distance / atr if atr > 0 else 0
        
        # FIX #4: Extended lookback for cross detection (10 bars)
        closes = df['close'].tail(10).tolist()
        cross_above = any(closes[i-1] < g50 <= closes[i] for i in range(1, len(closes)))
        cross_below = any(closes[i-1] > g50 >= closes[i] for i in range(1, len(closes)))
        
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
            "description": f"${price:,.0f} is {position} Gann 50% (${g50:,.0f})"
        }
    
    # ========================================================
    # CAPITULATION
    # ========================================================
    
    def detect_capitulation(self, df_w: pd.DataFrame, df_d: pd.DataFrame,
                           w_rsi: float, gann: Dict, price: float) -> Dict:
        """Capitulation with weighted scoring."""
        score = 0
        criteria = []
        
        # RSI (0-3)
        if w_rsi < 20:
            score += 3
            criteria.append({"name": "RSI_EXTREME", "score": 3, "val": f"{w_rsi:.1f}"})
        elif w_rsi < 25:
            score += 2.5
            criteria.append({"name": "RSI_VERY_LOW", "score": 2.5, "val": f"{w_rsi:.1f}"})
        elif w_rsi < 30:
            score += 1.5
            criteria.append({"name": "RSI_LOW", "score": 1.5, "val": f"{w_rsi:.1f}"})
        elif w_rsi < 35:
            score += 0.5
            criteria.append({"name": "RSI_OVERSOLD", "score": 0.5, "val": f"{w_rsi:.1f}"})
        
        # Volume (0-3)
        vol_ratio = 1.0
        if len(df_d) >= 20 and 'volume' in df_d.columns:
            avg = safe_float(df_d['volume'].tail(20).mean(), 1)
            vol_ratio = safe_float(df_d['volume'].iloc[-1], 0) / avg
        
        if vol_ratio >= 3.0:
            score += 3
            criteria.append({"name": "VOL_EXTREME", "score": 3, "val": f"{vol_ratio:.1f}x"})
        elif vol_ratio >= 2.0:
            score += 2
            criteria.append({"name": "VOL_SPIKE", "score": 2, "val": f"{vol_ratio:.1f}x"})
        elif vol_ratio >= 1.5:
            score += 1
            criteria.append({"name": "VOL_HIGH", "score": 1, "val": f"{vol_ratio:.1f}x"})
        
        # Gann support (0-3)
        g28 = gann["levels"].get("2_8", price * 0.88)
        g38 = gann["levels"].get("3_8", price * 0.94)
        
        d28 = abs(price - g28) / price
        d38 = abs(price - g38) / price
        
        if d28 < 0.02:
            score += 3
            criteria.append({"name": "GANN_2_8", "score": 3, "val": f"${g28:,.0f}"})
        elif d38 < 0.03:
            score += 2
            criteria.append({"name": "GANN_3_8", "score": 2, "val": f"${g38:,.0f}"})
        elif d38 < 0.05:
            score += 1
            criteria.append({"name": "GANN_NEAR", "score": 1, "val": "Near support"})
        
        # Divergence (0-3)
        w_rsi_s = self.calculate_rsi(df_w['close'])
        div = self.detect_divergence(df_w, w_rsi_s, 30)
        
        if div["bullish"]:
            div_score = 1 + (div["confidence"] * 2)
            score += div_score
            criteria.append({"name": "BULL_DIV", "score": round(div_score, 1), "val": div["description"]})
        
        max_score = 12
        pct = score / max_score
        
        if pct >= 0.70:
            status, conf = "CONFIRMED", 0.85
        elif pct >= 0.50:
            status, conf = "POTENTIAL", 0.60
        elif pct >= 0.30:
            status, conf = "DEVELOPING", 0.40
        else:
            status, conf = "NONE", 0.15
        
        return {
            "is_capitulation": pct >= 0.50,
            "status": status,
            "score": safe_round(score, 1),
            "max_score": max_score,
            "score_pct": safe_round(pct * 100, 1),
            "confidence": safe_round(conf, 2),
            "criteria": criteria,
            "details": {
                "weekly_rsi": safe_round(w_rsi, 2),
                "volume_ratio": safe_round(vol_ratio, 2),
                "divergence": div
            }
        }
    
    # ========================================================
    # REGIME
    # ========================================================
    
    def determine_regime(self, df_d: pd.DataFrame, w_rsi: float, w_adx: float,
                        cap: Dict, price: float) -> Dict:
        """Determine regime."""
        sma200 = safe_float(df_d['close'].rolling(200).mean().iloc[-1]) if len(df_d) >= 200 else price
        sma50 = safe_float(df_d['close'].rolling(50).mean().iloc[-1]) if len(df_d) >= 50 else price
        
        vs200 = "ABOVE" if price > sma200 else "BELOW"
        dist_pct = ((price - sma200) / sma200) * 100
        
        strength = "VERY_STRONG" if w_adx > ADX_VERY_STRONG else "STRONG" if w_adx > ADX_STRONG_TREND else "WEAK"
        
        warnings = []
        override = False
        override_reason = None
        
        if cap.get("is_capitulation"):
            regime = Regime.CAPITULATION
            override = True
            override_reason = f"Capitulation {cap['status']}"
            allow_short, allow_long = False, True
            size_cap = 0.50 if cap["status"] == "CONFIRMED" else 0.25
            warnings.append(f"⚠️ CAPITULATION: Shorts blocked")
        elif w_rsi > RSI_EXTREME_OVERBOUGHT:
            regime = Regime.EUPHORIA
            override = True
            override_reason = f"Euphoria RSI {w_rsi:.0f}"
            allow_short, allow_long = True, False
            size_cap = 0.25
            warnings.append("⚠️ EUPHORIA: Longs blocked")
        elif vs200 == "ABOVE" and strength in ["STRONG", "VERY_STRONG"]:
            regime = Regime.TRENDING_BULL
            allow_short, allow_long = True, True
            size_cap = 0.75 if w_rsi > 65 else 1.0
            if w_rsi > 65:
                warnings.append("⚠️ RSI elevated")
        elif vs200 == "BELOW" and strength in ["STRONG", "VERY_STRONG"]:
            regime = Regime.TRENDING_BEAR
            allow_short, allow_long = True, True
            size_cap = 1.0
            if w_rsi < 35:
                warnings.append("⚠️ RSI oversold in downtrend")
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
        """Analyze single timeframe."""
        cfg = TIMEFRAME_CONFIGS[tf]
        
        # FIX #1: Track data quality
        if len(df) < cfg.min_bars:
            return self._empty_tf(tf, data_sufficient=False)
        
        price = safe_float(df['close'].iloc[-1])
        
        rsi_s = self.calculate_rsi(df['close'])
        rsi = safe_float(rsi_s.iloc[-1], 50)
        
        _, _, macd_hist = self.calculate_macd(df['close'])
        macd_h = safe_float(macd_hist.iloc[-1])
        
        adx = safe_float(self.calculate_adx(df).iloc[-1])
        atr = safe_float(self.calculate_atr(df).iloc[-1])
        
        sma50 = safe_float(self.calculate_sma(df['close'], 50).iloc[-1]) if len(df) >= 50 else price
        sma200 = safe_float(self.calculate_sma(df['close'], 200).iloc[-1]) if len(df) >= 200 else price
        
        ichi = self.calculate_ichimoku(df)
        tenkan = safe_float(ichi['tenkan'].iloc[-1])
        kijun = safe_float(ichi['kijun'].iloc[-1])
        sa = safe_float(ichi['senkou_a'].iloc[-1])
        sb = safe_float(ichi['senkou_b'].iloc[-1])
        cloud_top = max(sa, sb)
        cloud_bottom = min(sa, sb)
        
        fa = safe_float(ichi['future_a'].iloc[-1])
        fb = safe_float(ichi['future_b'].iloc[-1])
        future_bull = fa > fb
        
        gann = self.calculate_gann_levels(df, cfg.gann_lookback)
        g50 = gann["gann_50"]
        
        # Weighted scoring
        bull, bear = 0.0, 0.0
        signals = {}
        
        # RSI (w=1.0)
        if rsi < RSI_OVERSOLD:
            bull += 1.0
            signals["RSI"] = f"OVERSOLD ({rsi:.1f})"
        elif rsi > RSI_OVERBOUGHT:
            bear += 1.0
            signals["RSI"] = f"OVERBOUGHT ({rsi:.1f})"
        else:
            signals["RSI"] = f"NEUTRAL ({rsi:.1f})"
        
        # MACD (w=1.0)
        if macd_h > 0:
            bull += 1.0
            signals["MACD"] = "BULLISH"
        else:
            bear += 1.0
            signals["MACD"] = "BEARISH"
        
        # TK (w=0.8)
        if tenkan > kijun:
            bull += 0.8
            signals["TK"] = "BULLISH"
        else:
            bear += 0.8
            signals["TK"] = "BEARISH"
        
        # Cloud (w=1.2)
        if price > cloud_top:
            bull += 1.2
            signals["CLOUD"] = "ABOVE"
            vs_cloud = "ABOVE"
        elif price < cloud_bottom:
            bear += 1.2
            signals["CLOUD"] = "BELOW"
            vs_cloud = "BELOW"
        else:
            signals["CLOUD"] = "INSIDE"
            vs_cloud = "INSIDE"
        
        # Gann 50 (w=1.5)
        if price > g50:
            bull += 1.5
            signals["GANN"] = f"ABOVE"
            vs_g50 = "ABOVE"
        else:
            bear += 1.5
            signals["GANN"] = f"BELOW"
            vs_g50 = "BELOW"
        
        # SMA200 (w=1.0)
        if price > sma200:
            bull += 1.0
            signals["SMA200"] = "ABOVE"
        else:
            bear += 1.0
            signals["SMA200"] = "BELOW"
        
        # Future cloud (w=0.5)
        if future_bull:
            bull += 0.5
            signals["FUTURE"] = "BULLISH"
        else:
            bear += 0.5
            signals["FUTURE"] = "BEARISH"
        
        # FIX #5: Normalize scores to 0-100
        total = bull + bear
        bull_pct = (bull / total) * 100 if total > 0 else 50
        bear_pct = 100 - bull_pct
        
        # Direction
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
        
        state = self._state_name(rsi, adx, direction)
        
        vol_r = 1.0
        if 'volume' in df.columns and len(df) >= 20:
            avg = safe_float(df['volume'].tail(20).mean(), 1)
            vol_r = safe_float(df['volume'].iloc[-1], 0) / avg
        
        return {
            "timeframe": tf,
            "direction": direction,
            "state_name": state,
            "data_sufficient": True,
            "bullish_score": safe_round(bull, 2),
            "bearish_score": safe_round(bear, 2),
            "bullish_pct": safe_round(bull_pct, 1),
            "rsi": safe_round(rsi, 2),
            "macd_histogram": safe_round(macd_h, 2),
            "adx": safe_round(adx, 2),
            "adx_label": "STRONG" if adx > ADX_STRONG_TREND else "WEAK",
            "atr": safe_round(atr, 2),
            "atr_pct": safe_round((atr / price) * 100, 2),
            "sma_50": safe_round(sma50, 2),
            "sma_200": safe_round(sma200, 2),
            "price_vs_sma_200": "ABOVE" if price > sma200 else "BELOW",
            "volume_ratio": safe_round(vol_r, 2),
            "gann_high": gann["high"],
            "gann_low": gann["low"],
            "gann_50_pct": g50,
            "price_vs_gann_50": vs_g50,
            "signal_details": signals,
            "ichimoku": {
                "tenkan": safe_round(tenkan, 2),
                "kijun": safe_round(kijun, 2),
                "cloud_top": safe_round(cloud_top, 2),
                "cloud_bottom": safe_round(cloud_bottom, 2),
                "tk_cross": signals.get("TK"),
                "price_vs_cloud": vs_cloud,
                "future_cloud": signals.get("FUTURE")
            },
            "gann": gann
        }
    
    def _empty_tf(self, tf: str, data_sufficient: bool = False) -> Dict:
        return {
            "timeframe": tf, "direction": "NEUTRAL", "state_name": "Insufficient Data",
            "data_sufficient": data_sufficient,
            "bullish_score": 0, "bearish_score": 0, "bullish_pct": 50,
            "rsi": 50, "macd_histogram": 0, "adx": 0, "adx_label": "N/A",
            "atr": 0, "atr_pct": 0, "sma_50": 0, "sma_200": 0,
            "price_vs_sma_200": "N/A", "volume_ratio": 1,
            "gann_high": 0, "gann_low": 0, "gann_50_pct": 0,
            "price_vs_gann_50": "N/A", "signal_details": {},
            "ichimoku": {"tenkan": 0, "kijun": 0, "cloud_top": 0, "cloud_bottom": 0,
                        "tk_cross": "N/A", "price_vs_cloud": "N/A", "future_cloud": "N/A"},
            "gann": {"high": 0, "low": 0, "levels": {}, "gann_50": 0}
        }
    
    def _state_name(self, rsi, adx, direction):
        if rsi < 25: return "Capitulation"
        if rsi < 35: return "Fear"
        if rsi > 75: return "Euphoria"
        if rsi > 65: return "Greed"
        if adx > 50: return "Expansion" if direction == "BULLISH" else "Contraction"
        if adx < 20: return "Consolidation"
        return "Transition"
    
    # ========================================================
    # CONSENSUS
    # ========================================================
    
    def calculate_consensus(self, tfs: Dict, regime: Dict) -> Dict:
        """MTF consensus with adaptive weights."""
        trending = regime["current"] in ["TRENDING_BULL", "TRENDING_BEAR"]
        
        w_bull, w_bear = 0.0, 0.0
        align = 0
        conflicts = []
        valid_tfs = 0
        
        for name, analysis in tfs.items():
            # FIX #1: Skip TFs with insufficient data
            if not analysis.get("data_sufficient", True):
                continue
            
            valid_tfs += 1
            cfg = TIMEFRAME_CONFIGS[name]
            weight = cfg.trend_weight if trending else cfg.range_weight
            
            # Use normalized scores
            w_bull += weight * analysis["bullish_pct"]
            w_bear += weight * (100 - analysis["bullish_pct"])
        
        if valid_tfs == 0:
            return self._empty_consensus()
        
        total = w_bull + w_bear
        score = int(((w_bull - w_bear) / total) * 100) if total > 0 else 0
        
        if score > 15:
            direction = "BULLISH"
        elif score < -15:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"
        
        for name, analysis in tfs.items():
            if not analysis.get("data_sufficient", True):
                continue
            if analysis["direction"] == direction:
                align += 1
            elif analysis["direction"] != "NEUTRAL" and direction != "NEUTRAL":
                conflicts.append({"tf": name, "dir": analysis["direction"]})
        
        if align >= 3 and abs(score) > 40:
            conf = "HIGH"
        elif align >= 2 and abs(score) > 25:
            conf = "MEDIUM"
        else:
            conf = "LOW"
        
        if regime.get("override_active") and conf == "HIGH":
            conf = "MEDIUM"
        
        verdict = f"{'STRONG' if conf == 'HIGH' else 'MODERATE' if conf == 'MEDIUM' else 'WEAK'} {direction}" if direction != "NEUTRAL" else "MIXED"
        
        return {
            "primary_direction": direction,
            "weighted_score": score,
            "alignment": f"{align}/{valid_tfs}",
            "alignment_count": align,
            "valid_timeframes": valid_tfs,
            "confidence_level": conf,
            "verdict": verdict,
            "conflicts": conflicts,
            "has_conflicts": len(conflicts) > 0,
            "weight_mode": "TREND" if trending else "RANGE",
            "tf_1m": tfs.get("1M", {}).get("direction", "N/A"),
            "tf_1w": tfs.get("1W", {}).get("direction", "N/A"),
            "tf_3d": tfs.get("3D", {}).get("direction", "N/A"),
            "tf_1d": tfs.get("1D", {}).get("direction", "N/A")
        }
    
    def _empty_consensus(self):
        return {
            "primary_direction": "NEUTRAL", "weighted_score": 0,
            "alignment": "0/0", "alignment_count": 0, "valid_timeframes": 0,
            "confidence_level": "NONE", "verdict": "NO DATA",
            "conflicts": [], "has_conflicts": False, "weight_mode": "N/A",
            "tf_1m": "N/A", "tf_1w": "N/A", "tf_3d": "N/A", "tf_1d": "N/A"
        }
    
    # ========================================================
    # PRIMARY BIAS
    # ========================================================
    
    def determine_bias(self, price: float, gann_ctx: Dict, consensus: Dict) -> Dict:
        """Primary bias from Gann 50%."""
        g_bias = gann_ctx["bias"]
        c_dir = consensus["primary_direction"]
        
        g_dir = "BULLISH" if "BULLISH" in g_bias else "BEARISH" if "BEARISH" in g_bias else "NEUTRAL"
        
        conflict = False
        if g_dir != "NEUTRAL" and g_dir != c_dir and c_dir != "NEUTRAL":
            primary = g_dir
            source = "GANN_OVERRIDE"
            conflict = True
            note = f"⚠️ Gann overrides consensus ({c_dir}→{g_dir})"
        else:
            primary = c_dir if c_dir != "NEUTRAL" else g_dir
            source = "CONSENSUS" if c_dir == primary else "GANN"
            note = f"Aligned: {primary}"
        
        return {
            "primary_bias": primary,
            "bias_source": source,
            "gann_50_level": gann_ctx["gann_50"],
            "gann_position": gann_ctx["position"],
            "gann_bias_detail": g_bias,
            "consensus_direction": c_dir,
            "conflict": conflict,
            "note": note,
            "description": gann_ctx["description"]
        }
    
    # ========================================================
    # TRADE SETUPS
    # ========================================================
    
    def generate_setups(self, price: float, tfs: Dict, consensus: Dict,
                       regime: Dict, cap: Dict, gann: Dict, gann_ctx: Dict) -> List[Dict]:
        """Generate trade setups with realistic R:R."""
        setups = []
        
        atr = safe_float(tfs["1D"]["atr"])
        if atr == 0:
            atr = price * 0.02
        
        g28 = gann["levels"].get("2_8", price * 0.88)
        g38 = gann["levels"].get("3_8", price * 0.94)
        g58 = gann["levels"].get("5_8", price * 1.06)
        g68 = gann["levels"].get("6_8", price * 1.12)
        
        # FIX #6: Get cloud levels, filter invalids
        cloud_top = safe_float(tfs["1W"]["ichimoku"]["cloud_top"])
        cloud_bot = safe_float(tfs["1W"]["ichimoku"]["cloud_bottom"])
        
        direction = consensus["primary_direction"]
        allow_long = regime.get("allows_long", True)
        allow_short = regime.get("allows_short", True)
        size_cap = regime.get("position_size_cap", 1.0)
        
        # LONG
        if direction == "BULLISH" and allow_long:
            entry = price
            
            # Stop: below support levels
            stop_cands = filter_valid_levels([g38 - atr * 0.5, cloud_bot - atr * 0.3, entry - atr * 1.5], entry, "below")
            stop = max(stop_cands) if stop_cands else entry - atr * 1.5
            risk = entry - stop
            
            # TP1: nearest resistance
            tp1_cands = filter_valid_levels([g58, cloud_top, entry + risk * 1.5], entry, "above")
            tp1 = min(tp1_cands) if tp1_cands else entry + risk * 1.5
            
            if risk > 0 and (tp1 - entry) / risk < MIN_RR_RATIO:
                tp1 = entry + risk * MIN_RR_RATIO
            
            tp2 = max(filter_valid_levels([g68, entry + risk * 2.5], entry, "above") or [entry + risk * 2.5])
            tp3 = entry + risk * 3.5
            
            rr = (tp1 - entry) / risk if risk > 0 else 0
            conf = "HIGH" if consensus["alignment_count"] >= 3 else "MEDIUM" if consensus["alignment_count"] >= 2 else "LOW"
            size = min(1.0 if conf == "HIGH" else 0.5 if conf == "MEDIUM" else 0.25, size_cap)
            
            setups.append({
                "id": 1, "type": "PRIMARY", "direction": "LONG", "confidence": conf,
                "entry": safe_round(entry, 2), "stop_loss": safe_round(stop, 2),
                "tp1": safe_round(tp1, 2), "tp2": safe_round(tp2, 2), "tp3": safe_round(tp3, 2),
                "rr_ratio": safe_round(max(rr, MIN_RR_RATIO), 2),
                "risk_pct": safe_round((risk / entry) * 100, 2),
                "position_size": safe_round(size, 2),
                "rationale": f"MTF {consensus['verdict']}, {gann_ctx['bias']}"
            })
        
        # SHORT
        elif direction == "BEARISH" and allow_short:
            entry = price
            
            stop_cands = filter_valid_levels([g58 + atr * 0.5, cloud_top + atr * 0.3, entry + atr * 1.5], entry, "above")
            stop = min(stop_cands) if stop_cands else entry + atr * 1.5
            risk = stop - entry
            
            tp1_cands = filter_valid_levels([g38, cloud_bot, entry - risk * 1.5], entry, "below")
            tp1 = max(tp1_cands) if tp1_cands else entry - risk * 1.5
            
            if risk > 0 and (entry - tp1) / risk < MIN_RR_RATIO:
                tp1 = entry - risk * MIN_RR_RATIO
            
            tp2 = min(filter_valid_levels([g28, entry - risk * 2.5], entry, "below") or [entry - risk * 2.5])
            tp3 = entry - risk * 3.5
            
            rr = (entry - tp1) / risk if risk > 0 else 0
            conf = "HIGH" if consensus["alignment_count"] >= 3 else "MEDIUM" if consensus["alignment_count"] >= 2 else "LOW"
            size = min(1.0 if conf == "HIGH" else 0.5 if conf == "MEDIUM" else 0.25, size_cap)
            
            setups.append({
                "id": 1, "type": "PRIMARY", "direction": "SHORT", "confidence": conf,
                "entry": safe_round(entry, 2), "stop_loss": safe_round(stop, 2),
                "tp1": safe_round(tp1, 2), "tp2": safe_round(tp2, 2), "tp3": safe_round(tp3, 2),
                "rr_ratio": safe_round(max(rr, MIN_RR_RATIO), 2),
                "risk_pct": safe_round((risk / entry) * 100, 2),
                "position_size": safe_round(size, 2),
                "rationale": f"MTF {consensus['verdict']}, {gann_ctx['bias']}"
            })
        
        # Counter-trend at capitulation
        if cap.get("is_capitulation") and allow_long:
            entry = g38
            stop = g28 - atr
            risk = entry - stop
            tp1 = price
            tp2 = g58
            rr = (tp1 - entry) / risk if risk > 0 else 0
            
            setups.append({
                "id": 2, "type": "COUNTER_TREND", "direction": "LONG",
                "confidence": "MEDIUM" if cap["status"] == "CONFIRMED" else "LOW",
                "entry": safe_round(entry, 2), "stop_loss": safe_round(stop, 2),
                "tp1": safe_round(tp1, 2), "tp2": safe_round(tp2, 2), "tp3": None,
                "rr_ratio": safe_round(max(rr, MIN_RR_RATIO), 2),
                "risk_pct": safe_round((risk / entry) * 100 if entry > 0 else 0, 2),
                "position_size": 0.25,
                "rationale": f"Capitulation {cap['status']}"
            })
        
        # Wait
        if not setups or consensus["alignment_count"] < 2:
            setups.append({
                "id": len(setups) + 1, "type": "WAIT", "direction": "FLAT",
                "confidence": "NONE", "entry": safe_round(price, 2),
                "stop_loss": None, "tp1": None, "tp2": None, "tp3": None,
                "rr_ratio": 0, "risk_pct": 0, "position_size": 0,
                "rationale": "Low alignment"
            })
        
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}
        setups.sort(key=lambda x: order.get(x["confidence"], 4))
        
        return setups
    
    # ========================================================
    # TIME FORECAST
    # ========================================================
    
    def time_forecast(self, df: pd.DataFrame, price: float, atr: float, gann: Dict) -> Dict:
        """Time-based pivot forecast."""
        lookback = min(365, len(df))
        recent = df.tail(lookback)
        
        high = safe_float(recent['high'].max())
        low = safe_float(recent['low'].min())
        
        ts = 'date' if 'date' in df.columns else 'timestamp'
        
        try:
            hi = recent['high'].idxmax()
            lo = recent['low'].idxmin()
            
            if hi > lo:
                ref_type, pivot_type = "HIGH", "LOW"
                ref_idx, ref_price = hi, high
            else:
                ref_type, pivot_type = "LOW", "HIGH"
                ref_idx, ref_price = lo, low
            
            ref_date = pd.to_datetime(df.loc[ref_idx, ts])
            if ref_date.tzinfo:
                ref_date = ref_date.replace(tzinfo=None)
            ref_str = ref_date.strftime("%Y-%m-%d")
            days_since = int((datetime.now() - ref_date).days)
        except:
            ref_type, pivot_type = "HIGH", "LOW"
            days_since, ref_str, ref_price = 90, "N/A", price
        
        next_date, days_to, conf, cycle = None, 90, 0.40, 90
        
        for c in GANN_CYCLES:
            if c > days_since:
                days_to = c - days_since
                next_date = (datetime.now() + timedelta(days=days_to)).strftime("%Y-%m-%d")
                conf = 0.60 if c in [180, 360] else 0.50 if c in [90, 144] else 0.45
                cycle = c
                break
        
        if not next_date:
            next_date = (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d")
        
        g38 = gann["levels"].get("3_8", price * 0.9)
        g58 = gann["levels"].get("5_8", price * 1.1)
        atr_proj = atr * np.sqrt(days_to) if atr > 0 else price * 0.10
        
        low_proj = max(price - atr_proj, g38 * 0.95)
        high_proj = min(price + atr_proj, g58 * 1.05)
        
        return {
            "next_pivot_date": next_date,
            "days_to_pivot": days_to,
            "pivot_type": pivot_type,
            "confidence": safe_round(conf, 2),
            "confidence_pct": f"{int(conf * 100)}%",
            "confidence_level": "HIGH" if conf >= 0.60 else "MEDIUM" if conf >= 0.50 else "LOW",
            "probable_price_low": safe_round(low_proj, 2),
            "probable_price_high": safe_round(high_proj, 2),
            "price_range": f"${low_proj:,.0f} - ${high_proj:,.0f}",
            "cycle_origin": {
                "reference_type": ref_type,
                "reference_date": ref_str,
                "reference_price": safe_round(ref_price, 2),
                "days_since": days_since,
                "cycle_length": cycle
            }
        }
    
    # ========================================================
    # MAIN SIGNAL
    # ========================================================
    
    def generate_mtf_signal(self, symbol: str = "BTCUSDT") -> Dict:
        """Main signal generator - Final v5.0.10."""
        try:
            logger.info(f"[MTF] {symbol}")
            
            df_1d = self.fetch_real_binance_data(symbol=symbol)
            if df_1d is None or len(df_1d) < 100:
                raise ValueError(f"Insufficient data")
            
            df_3d = self.resample_ohlcv(df_1d, "3D")
            df_1w = self.resample_ohlcv(df_1d, "1W")
            df_1m = self.resample_ohlcv(df_1d, "1M")
            
            price = safe_float(df_1d['close'].iloc[-1])
            ts = 'date' if 'date' in df_1d.columns else 'timestamp'
            sig_date = str(df_1d[ts].iloc[-1])[:10]
            
            tfs = {
                "1D": self.analyze_timeframe(df_1d, "1D"),
                "3D": self.analyze_timeframe(df_3d, "3D"),
                "1W": self.analyze_timeframe(df_1w, "1W"),
                "1M": self.analyze_timeframe(df_1m, "1M"),
            }
            
            w_rsi = safe_float(tfs["1W"]["rsi"], 50)
            w_adx = safe_float(tfs["1W"]["adx"], 20)
            w_gann = tfs["1W"]["gann"]
            d_atr = safe_float(tfs["1D"]["atr"])
            
            cap = self.detect_capitulation(df_1w, df_1d, w_rsi, w_gann, price)
            regime = self.determine_regime(df_1d, w_rsi, w_adx, cap, price)
            consensus = self.calculate_consensus(tfs, regime)
            gann_ctx = self.analyze_gann_context(price, w_gann, df_1d, d_atr)
            bias = self.determine_bias(price, gann_ctx, consensus)
            forecast = self.time_forecast(df_1d, price, d_atr, w_gann)
            setups = self.generate_setups(price, tfs, consensus, regime, cap, w_gann, gann_ctx)
            
            if bias["primary_bias"] == "BULLISH":
                inv = w_gann["levels"].get("3_8", price * 0.92)
                inv_desc = f"Bullish invalid below ${inv:,.0f}"
            elif bias["primary_bias"] == "BEARISH":
                inv = w_gann["levels"].get("5_8", price * 1.08)
                inv_desc = f"Bearish invalid above ${inv:,.0f}"
            else:
                inv, inv_desc = None, "No clear bias"
            
            levels = {
                "monthly": {"high": tfs["1M"]["gann_high"], "low": tfs["1M"]["gann_low"], "gann_50": tfs["1M"]["gann_50_pct"]},
                "weekly": {"high": w_gann["high"], "low": w_gann["low"], "gann_50": w_gann["gann_50"],
                          "gann_3_8": w_gann["levels"].get("3_8"), "gann_5_8": w_gann["levels"].get("5_8"),
                          "cloud_top": tfs["1W"]["ichimoku"]["cloud_top"], "cloud_bottom": tfs["1W"]["ichimoku"]["cloud_bottom"]},
                "daily": {"high": tfs["1D"]["gann_high"], "low": tfs["1D"]["gann_low"], "gann_50": tfs["1D"]["gann_50_pct"], "sma_200": tfs["1D"]["sma_200"]}
            }
            
            enneagram = {
                "1M": tfs["1M"]["state_name"], "1W": tfs["1W"]["state_name"],
                "3D": tfs["3D"]["state_name"], "1D": tfs["1D"]["state_name"],
                "dominant": tfs["1W"]["state_name"],
                "phase": "Accumulation" if bias["primary_bias"] == "BULLISH" else "Distribution" if bias["primary_bias"] == "BEARISH" else "Transition",
                "arrow": "↑" if bias["primary_bias"] == "BULLISH" else "↓" if bias["primary_bias"] == "BEARISH" else "→"
            }
            
            logger.info(f"[MTF] Done: {consensus['verdict']}")
            
            result = {
                "status": "success",
                "symbol": symbol,
                "current_price": safe_round(price, 2),
                "signal_date": sig_date,
                "timestamp": datetime.now().isoformat(),
                "version": self.VERSION,
                "regime": regime,
                "timeframes": tfs,
                "consensus": consensus,
                "primary_bias": bias,
                "gann_context": gann_ctx,
                "price_levels": levels,
                "capitulation": cap,
                "time_forecast": forecast,
                "trade_setups": setups,
                "invalidation": {"price": safe_round(inv, 2), "description": inv_desc},
                "enneagram": enneagram,
                "gann_interpretation": {
                    "weekly_50_pct": w_gann["gann_50"],
                    "current_position": gann_ctx["position"],
                    "primary_bias": bias["primary_bias"],
                    "description": gann_ctx["description"],
                    "rule": "Above 50% = Bulls | Below 50% = Bears"
                },
                "signal": {
                    "type": setups[0]["direction"] if setups else "WAIT",
                    "confidence": setups[0]["confidence"] if setups else "NONE",
                    "entry": setups[0].get("entry"),
                    "stop_loss": setups[0].get("stop_loss"),
                    "take_profit": setups[0].get("tp1"),
                    "tp2": setups[0].get("tp2"),
                    "tp3": setups[0].get("tp3"),
                    "rr_ratio": setups[0].get("rr_ratio"),
                    "position_size": setups[0].get("position_size", 0),
                    "rationale": setups[0].get("rationale", "")
                }
            }
            
            return convert_numpy_types(result)
        
        except Exception as e:
            logger.error(f"[MTF] Error: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "detail": str(e), "version": self.VERSION}
    
    def get_daily_signal(self, df=None):
        return self.generate_mtf_signal()
