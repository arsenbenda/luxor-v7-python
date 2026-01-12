# ============================================================
# LUXOR V7 PRANA - GANN EGYPT-INDIA UNIFIED SYSTEM v5.0.9
# COHERENT SYSTEM - Synergistic indicator logic
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
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

def safe_float(value, default=0.0):
    """Safely convert to float."""
    if value is None:
        return default
    try:
        result = float(value)
        return default if np.isnan(result) else result
    except (TypeError, ValueError):
        return default

def safe_round(value, decimals=2):
    """Safely round a value."""
    val = safe_float(value)
    return round(val, decimals) if val is not None else None

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
    trend_weight: float      # Weight in trending market
    range_weight: float      # Weight in ranging market

# Adaptive weights based on market regime
TIMEFRAME_CONFIGS = {
    "1M": TimeframeConfig("1M", 0.35, 24, 12, trend_weight=0.40, range_weight=0.25),
    "1W": TimeframeConfig("1W", 0.30, 52, 26, trend_weight=0.35, range_weight=0.30),
    "3D": TimeframeConfig("3D", 0.20, 120, 60, trend_weight=0.15, range_weight=0.25),
    "1D": TimeframeConfig("1D", 0.15, 252, 100, trend_weight=0.10, range_weight=0.20),
}

# Thresholds
RSI_OVERSOLD = 30
RSI_EXTREME_OVERSOLD = 25
RSI_OVERBOUGHT = 70
RSI_EXTREME_OVERBOUGHT = 75
ADX_STRONG_TREND = 25
ADX_VERY_STRONG = 50
VOLUME_SPIKE_THRESHOLD = 2.0
MIN_RR_RATIO = 1.5

# Gann cycles
GANN_CYCLES = [30, 45, 60, 90, 120, 144, 180, 270, 360]

# ============================================================
# MAIN CLASS
# ============================================================

class LuxorV7PranaSystem:
    """LUXOR V7 PRANA - Coherent Multi-Timeframe Gann System v5.0.9"""
    
    CACHE = {'df': None, 'last_fetch': None, 'cache_duration': 3600}
    VERSION = "5.0.9"
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        logger.info(f"[INIT] LuxorV7PranaSystem v{self.VERSION} initialized")
    
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
                logger.info(f"[DATA] Trying {exchange_id} for {sym}")
                exchange_class = getattr(ccxt, exchange_id, None)
                if exchange_class is None:
                    continue
                
                exchange = exchange_class({'enableRateLimit': True, 'timeout': 30000})
                ohlcv = exchange.fetch_ohlcv(sym, interval, limit=limit)
                
                if ohlcv and len(ohlcv) > 0:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['date'] = df['timestamp']
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                    df = df.dropna(subset=['close']).reset_index(drop=True)
                    logger.info(f"[DATA] Fetched {len(df)} candles from {exchange_id}")
                    return df
            except Exception as e:
                logger.warning(f"[DATA] {exchange_id} failed: {str(e)[:80]}")
                continue
        
        raise Exception("All exchanges failed")
    
    def fetch_real_binance_data(self, use_cache=True, symbol: str = "BTCUSDT") -> pd.DataFrame:
        """Fetch with caching."""
        if use_cache and self.CACHE['df'] is not None and self.CACHE['last_fetch']:
            age = (datetime.now() - self.CACHE['last_fetch']).total_seconds()
            if age < self.CACHE['cache_duration']:
                logger.info(f"[CACHE] Using cached data (age: {age:.0f}s)")
                return self.CACHE['df'].copy()
        
        ccxt_symbol = symbol[:-4] + '/USDT' if symbol.endswith('USDT') and '/' not in symbol else symbol
        df = self.fetch_ohlcv_ccxt(ccxt_symbol, "1d", 500)
        self.CACHE['df'] = df.copy()
        self.CACHE['last_fetch'] = datetime.now()
        return df
    
    def resample_ohlcv(self, df_1d: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """Resample to higher timeframes."""
        df = df_1d.copy()
        idx_col = 'date' if 'date' in df.columns else 'timestamp'
        df.set_index(idx_col, inplace=True)
        
        resample_map = {'3D': '3D', '1W': 'W', '1M': 'M'}
        rule = resample_map.get(target_tf, '1D')
        
        resampled = df.resample(rule).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        
        resampled.reset_index(inplace=True)
        resampled.rename(columns={resampled.columns[0]: 'timestamp'}, inplace=True)
        return resampled
    
    # ========================================================
    # TECHNICAL INDICATORS
    # ========================================================
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.inf)
        return (100 - (100 / (1 + rs))).fillna(50)
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal, macd - signal
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX."""
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
        
        return dx.ewm(alpha=1/period, min_periods=period).mean().fillna(0)
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(window=period).mean().fillna(tr.mean() if len(tr) > 0 else 1.0)
    
    def calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate SMA."""
        return prices.rolling(window=period).mean()
    
    def calculate_ichimoku(self, df: pd.DataFrame) -> Dict:
        """Calculate Ichimoku Cloud."""
        high, low, close = df['high'], df['low'], df['close']
        
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        
        # Future cloud (for leading indication)
        future_senkou_a = (tenkan + kijun) / 2  # Not shifted - current value
        future_senkou_b = (high.rolling(52).max() + low.rolling(52).min()) / 2
        
        return {
            'tenkan': tenkan.fillna(close),
            'kijun': kijun.fillna(close),
            'senkou_a': senkou_a.fillna(close),
            'senkou_b': senkou_b.fillna(close),
            'future_senkou_a': future_senkou_a.fillna(close),
            'future_senkou_b': future_senkou_b.fillna(close)
        }
    
    # ========================================================
    # PIVOT POINT DETECTION (for accurate divergence)
    # ========================================================
    
    def find_pivot_points(self, prices: pd.Series, left_bars: int = 5, right_bars: int = 5) -> Dict:
        """
        Find actual pivot highs and lows using left/right bar confirmation.
        A pivot high is a bar with 'left_bars' lower highs on left and 'right_bars' lower highs on right.
        """
        pivot_highs = []
        pivot_lows = []
        
        for i in range(left_bars, len(prices) - right_bars):
            # Check for pivot high
            is_pivot_high = True
            current = prices.iloc[i]
            
            for j in range(1, left_bars + 1):
                if prices.iloc[i - j] >= current:
                    is_pivot_high = False
                    break
            
            if is_pivot_high:
                for j in range(1, right_bars + 1):
                    if prices.iloc[i + j] >= current:
                        is_pivot_high = False
                        break
            
            if is_pivot_high:
                pivot_highs.append({'index': i, 'price': float(current)})
            
            # Check for pivot low
            is_pivot_low = True
            for j in range(1, left_bars + 1):
                if prices.iloc[i - j] <= current:
                    is_pivot_low = False
                    break
            
            if is_pivot_low:
                for j in range(1, right_bars + 1):
                    if prices.iloc[i + j] <= current:
                        is_pivot_low = False
                        break
            
            if is_pivot_low:
                pivot_lows.append({'index': i, 'price': float(current)})
        
        return {'highs': pivot_highs, 'lows': pivot_lows}
    
    def detect_divergence_accurate(self, df: pd.DataFrame, rsi_series: pd.Series, lookback: int = 50) -> Dict:
        """
        Accurate divergence detection using pivot points.
        Bullish: Price makes lower low, RSI makes higher low
        Bearish: Price makes higher high, RSI makes lower high
        """
        if len(df) < lookback:
            return {
                "bullish_divergence": False,
                "bearish_divergence": False,
                "description": "Insufficient data",
                "confidence": 0
            }
        
        recent_df = df.tail(lookback)
        recent_rsi = rsi_series.tail(lookback)
        
        # Find pivot points
        price_pivots = self.find_pivot_points(recent_df['close'].reset_index(drop=True), 3, 3)
        rsi_pivots = self.find_pivot_points(recent_rsi.reset_index(drop=True), 3, 3)
        
        bullish_div = False
        bearish_div = False
        confidence = 0
        description = "No divergence"
        
        # Check for bullish divergence (need at least 2 pivot lows)
        if len(price_pivots['lows']) >= 2 and len(rsi_pivots['lows']) >= 2:
            price_lows = price_pivots['lows'][-2:]
            rsi_lows = rsi_pivots['lows'][-2:]
            
            # Price making lower low, RSI making higher low
            if price_lows[1]['price'] < price_lows[0]['price'] and rsi_lows[1]['price'] > rsi_lows[0]['price']:
                bullish_div = True
                # Confidence based on strength of divergence
                price_diff = (price_lows[0]['price'] - price_lows[1]['price']) / price_lows[0]['price']
                rsi_diff = rsi_lows[1]['price'] - rsi_lows[0]['price']
                confidence = min(0.9, 0.5 + price_diff * 10 + rsi_diff / 100)
                description = f"Bullish divergence: Price LL ({price_lows[1]['price']:.0f} < {price_lows[0]['price']:.0f}), RSI HL ({rsi_lows[1]['price']:.1f} > {rsi_lows[0]['price']:.1f})"
        
        # Check for bearish divergence (need at least 2 pivot highs)
        if len(price_pivots['highs']) >= 2 and len(rsi_pivots['highs']) >= 2:
            price_highs = price_pivots['highs'][-2:]
            rsi_highs = rsi_pivots['highs'][-2:]
            
            # Price making higher high, RSI making lower high
            if price_highs[1]['price'] > price_highs[0]['price'] and rsi_highs[1]['price'] < rsi_highs[0]['price']:
                bearish_div = True
                price_diff = (price_highs[1]['price'] - price_highs[0]['price']) / price_highs[0]['price']
                rsi_diff = rsi_highs[0]['price'] - rsi_highs[1]['price']
                confidence = min(0.9, 0.5 + price_diff * 10 + rsi_diff / 100)
                description = f"Bearish divergence: Price HH ({price_highs[1]['price']:.0f} > {price_highs[0]['price']:.0f}), RSI LH ({rsi_highs[1]['price']:.1f} < {rsi_highs[0]['price']:.1f})"
        
        return {
            "bullish_divergence": bullish_div,
            "bearish_divergence": bearish_div,
            "description": description,
            "confidence": safe_round(confidence, 2),
            "pivot_lows_found": len(price_pivots['lows']),
            "pivot_highs_found": len(price_pivots['highs'])
        }
    
    # ========================================================
    # GANN CALCULATIONS
    # ========================================================
    
    def calculate_gann_levels(self, df: pd.DataFrame, lookback: int) -> Dict:
        """Calculate Gann levels."""
        lookback = min(lookback, len(df))
        recent = df.tail(lookback)
        
        high = safe_float(recent['high'].max())
        low = safe_float(recent['low'].min())
        range_val = high - low
        current = safe_float(df['close'].iloc[-1])
        
        # Get dates
        ts_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        try:
            high_idx = recent['high'].idxmax()
            low_idx = recent['low'].idxmin()
            high_date = str(df.loc[high_idx, ts_col])[:10]
            low_date = str(df.loc[low_idx, ts_col])[:10]
        except:
            high_date = "N/A"
            low_date = "N/A"
        
        # Calculate all levels
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
    
    def analyze_gann_50_context(self, current_price: float, gann: Dict, df: pd.DataFrame, atr: float) -> Dict:
        """
        Analyze price position relative to Gann 50% with context.
        Not just above/below, but direction, momentum, and confirmation.
        """
        gann_50 = gann["gann_50"]
        gann_3_8 = gann["levels"].get("3_8", gann_50 * 0.9)
        gann_5_8 = gann["levels"].get("5_8", gann_50 * 1.1)
        
        # Distance from 50%
        distance = current_price - gann_50
        distance_pct = (distance / gann_50) * 100 if gann_50 > 0 else 0
        distance_atr = distance / atr if atr > 0 else 0
        
        # Check recent price action relative to 50%
        recent_closes = df['close'].tail(5).tolist()
        crosses_above = False
        crosses_below = False
        
        for i in range(1, len(recent_closes)):
            if recent_closes[i-1] < gann_50 <= recent_closes[i]:
                crosses_above = True
            elif recent_closes[i-1] > gann_50 >= recent_closes[i]:
                crosses_below = True
        
        # Determine position and bias
        if current_price > gann_50 + atr:
            position = "STRONG_ABOVE"
            gann_bias = "BULLISH"
        elif current_price > gann_50:
            position = "ABOVE"
            gann_bias = "BULLISH" if not crosses_below else "WEAK_BULLISH"
        elif current_price < gann_50 - atr:
            position = "STRONG_BELOW"
            gann_bias = "BEARISH"
        elif current_price < gann_50:
            position = "BELOW"
            gann_bias = "BEARISH" if not crosses_above else "WEAK_BEARISH"
        else:
            position = "AT_50"
            gann_bias = "NEUTRAL"
        
        # Recent cross provides stronger signal
        if crosses_above and current_price > gann_50:
            gann_bias = "BULLISH_BREAKOUT"
        elif crosses_below and current_price < gann_50:
            gann_bias = "BEARISH_BREAKDOWN"
        
        return {
            "position": position,
            "bias": gann_bias,
            "gann_50": gann_50,
            "distance": safe_round(distance, 2),
            "distance_pct": safe_round(distance_pct, 2),
            "distance_atr": safe_round(distance_atr, 2),
            "recent_cross_above": crosses_above,
            "recent_cross_below": crosses_below,
            "support_3_8": gann_3_8,
            "resistance_5_8": gann_5_8,
            "description": f"Price ${current_price:,.0f} is {position} Gann 50% (${gann_50:,.0f}), bias: {gann_bias}"
        }
    
    # ========================================================
    # CAPITULATION DETECTION (Weighted Scoring)
    # ========================================================
    
    def detect_capitulation(self, df_weekly: pd.DataFrame, df_daily: pd.DataFrame,
                           weekly_rsi: float, weekly_gann: Dict, current_price: float) -> Dict:
        """
        Capitulation detection with weighted scoring system.
        More nuanced than binary criteria.
        """
        score = 0
        max_score = 12
        criteria = []
        
        # Criterion 1: RSI Level (0-3 points)
        if weekly_rsi < 20:
            score += 3
            criteria.append({"name": "RSI_EXTREME", "score": 3, "detail": f"RSI {weekly_rsi:.1f} < 20"})
        elif weekly_rsi < 25:
            score += 2.5
            criteria.append({"name": "RSI_VERY_LOW", "score": 2.5, "detail": f"RSI {weekly_rsi:.1f} < 25"})
        elif weekly_rsi < 30:
            score += 1.5
            criteria.append({"name": "RSI_LOW", "score": 1.5, "detail": f"RSI {weekly_rsi:.1f} < 30"})
        elif weekly_rsi < 35:
            score += 0.5
            criteria.append({"name": "RSI_OVERSOLD_ZONE", "score": 0.5, "detail": f"RSI {weekly_rsi:.1f} < 35"})
        
        # Criterion 2: Volume Spike (0-3 points)
        volume_ratio = 1.0
        if len(df_daily) >= 20 and 'volume' in df_daily.columns:
            avg_vol = safe_float(df_daily['volume'].tail(20).mean(), 1)
            volume_ratio = safe_float(df_daily['volume'].iloc[-1], 0) / avg_vol
        
        if volume_ratio >= 3.0:
            score += 3
            criteria.append({"name": "VOLUME_EXTREME", "score": 3, "detail": f"Volume {volume_ratio:.1f}x avg"})
        elif volume_ratio >= 2.0:
            score += 2
            criteria.append({"name": "VOLUME_SPIKE", "score": 2, "detail": f"Volume {volume_ratio:.1f}x avg"})
        elif volume_ratio >= 1.5:
            score += 1
            criteria.append({"name": "VOLUME_HIGH", "score": 1, "detail": f"Volume {volume_ratio:.1f}x avg"})
        
        # Criterion 3: Price near Gann Support (0-3 points)
        gann_2_8 = weekly_gann["levels"].get("2_8", current_price * 0.88)
        gann_3_8 = weekly_gann["levels"].get("3_8", current_price * 0.94)
        
        dist_to_2_8 = abs(current_price - gann_2_8) / current_price
        dist_to_3_8 = abs(current_price - gann_3_8) / current_price
        
        if dist_to_2_8 < 0.02:
            score += 3
            criteria.append({"name": "GANN_2_8_TEST", "score": 3, "detail": f"At Gann 2/8 ${gann_2_8:,.0f}"})
        elif dist_to_3_8 < 0.03:
            score += 2
            criteria.append({"name": "GANN_3_8_TEST", "score": 2, "detail": f"Near Gann 3/8 ${gann_3_8:,.0f}"})
        elif dist_to_3_8 < 0.05:
            score += 1
            criteria.append({"name": "GANN_SUPPORT_ZONE", "score": 1, "detail": f"Approaching Gann 3/8"})
        
        # Criterion 4: Bullish Divergence (0-3 points)
        weekly_rsi_series = self.calculate_rsi(df_weekly['close'])
        divergence = self.detect_divergence_accurate(df_weekly, weekly_rsi_series, 30)
        
        if divergence["bullish_divergence"]:
            div_confidence = divergence.get("confidence", 0.5)
            div_score = 1 + (div_confidence * 2)  # 1-3 based on confidence
            score += div_score
            criteria.append({"name": "BULLISH_DIVERGENCE", "score": round(div_score, 1), "detail": divergence["description"]})
        
        # Determine status
        score_pct = score / max_score
        
        if score_pct >= 0.70:
            status = "CONFIRMED"
            confidence = 0.85 + (score_pct - 0.70) * 0.5
        elif score_pct >= 0.50:
            status = "POTENTIAL"
            confidence = 0.60 + (score_pct - 0.50) * 1.0
        elif score_pct >= 0.30:
            status = "DEVELOPING"
            confidence = 0.35 + (score_pct - 0.30) * 1.25
        else:
            status = "NONE"
            confidence = score_pct * 1.17
        
        return {
            "is_capitulation": score_pct >= 0.50,
            "status": status,
            "score": safe_round(score, 1),
            "max_score": max_score,
            "score_pct": safe_round(score_pct * 100, 1),
            "confidence": safe_round(min(confidence, 0.95), 2),
            "criteria": criteria,
            "details": {
                "weekly_rsi": safe_round(weekly_rsi, 2),
                "volume_ratio": safe_round(volume_ratio, 2),
                "divergence": divergence
            }
        }
    
    # ========================================================
    # REGIME DETECTION
    # ========================================================
    
    def determine_regime(self, df_daily: pd.DataFrame, weekly_rsi: float, weekly_adx: float,
                        capitulation: Dict, current_price: float) -> Dict:
        """Determine market regime with nuanced logic."""
        
        # SMA 200
        sma_200 = safe_float(df_daily['close'].rolling(200).mean().iloc[-1]) if len(df_daily) >= 200 else current_price
        sma_50 = safe_float(df_daily['close'].rolling(50).mean().iloc[-1]) if len(df_daily) >= 50 else current_price
        
        price_vs_sma200 = "ABOVE" if current_price > sma_200 else "BELOW"
        price_vs_sma50 = "ABOVE" if current_price > sma_50 else "BELOW"
        sma_distance_pct = ((current_price - sma_200) / sma_200) * 100
        
        # Trend strength
        if weekly_adx > ADX_VERY_STRONG:
            trend_strength = "VERY_STRONG"
        elif weekly_adx > ADX_STRONG_TREND:
            trend_strength = "STRONG"
        else:
            trend_strength = "WEAK"
        
        # Base regime determination
        warnings = []
        override_active = False
        override_reason = None
        
        if capitulation.get("is_capitulation", False):
            base_regime = Regime.CAPITULATION
            override_active = True
            override_reason = f"Capitulation {capitulation.get('status')}"
            allows_short = False
            allows_long = True
            size_cap = 0.50 if capitulation["status"] == "CONFIRMED" else 0.25
            warnings.append(f"⚠️ CAPITULATION ({capitulation['status']}): Shorts blocked")
        elif weekly_rsi > RSI_EXTREME_OVERBOUGHT:
            base_regime = Regime.EUPHORIA
            override_active = True
            override_reason = f"Euphoria RSI {weekly_rsi:.0f}"
            allows_short = True
            allows_long = False
            size_cap = 0.25
            warnings.append(f"⚠️ EUPHORIA: Longs blocked")
        elif price_vs_sma200 == "ABOVE" and trend_strength in ["STRONG", "VERY_STRONG"]:
            base_regime = Regime.TRENDING_BULL
            allows_short = True  # But with caution
            allows_long = True
            size_cap = 1.0
            if weekly_rsi > 65:
                warnings.append("⚠️ RSI elevated in uptrend - size cautiously")
                size_cap = 0.75
        elif price_vs_sma200 == "BELOW" and trend_strength in ["STRONG", "VERY_STRONG"]:
            base_regime = Regime.TRENDING_BEAR
            allows_short = True
            allows_long = True  # But counter-trend
            size_cap = 1.0
            if weekly_rsi < 35:
                warnings.append("⚠️ RSI oversold in downtrend - bounce possible")
        else:
            base_regime = Regime.RANGING
            allows_short = True
            allows_long = True
            size_cap = 0.75  # Reduced in ranging
        
        regime_strength = min(weekly_adx / 100, 1.0) if base_regime in [Regime.TRENDING_BULL, Regime.TRENDING_BEAR] else 0.4
        
        return {
            "current": base_regime.value,
            "strength": safe_round(regime_strength, 2),
            "strength_label": f"{trend_strength} (ADX {weekly_adx:.0f})",
            "override_active": override_active,
            "override_reason": override_reason,
            "allows_short": allows_short,
            "allows_long": allows_long,
            "position_size_cap": safe_round(size_cap, 2),
            "warnings": warnings,
            "sma_200": safe_round(sma_200, 2),
            "sma_50": safe_round(sma_50, 2),
            "price_vs_sma_200": price_vs_sma200,
            "price_vs_sma_50": price_vs_sma50,
            "sma_distance_pct": safe_round(sma_distance_pct, 2)
        }
    
    # ========================================================
    # TIMEFRAME ANALYSIS
    # ========================================================
    
    def analyze_timeframe(self, df: pd.DataFrame, tf_name: str) -> Dict:
        """Comprehensive timeframe analysis."""
        config = TIMEFRAME_CONFIGS[tf_name]
        
        if len(df) < config.min_bars:
            return self._empty_tf_analysis(tf_name)
        
        current = safe_float(df['close'].iloc[-1])
        
        # Indicators
        rsi_series = self.calculate_rsi(df['close'])
        rsi = safe_float(rsi_series.iloc[-1], 50)
        
        macd_line, signal_line, macd_hist = self.calculate_macd(df['close'])
        macd = safe_float(macd_line.iloc[-1])
        macd_histogram = safe_float(macd_hist.iloc[-1])
        
        adx = safe_float(self.calculate_adx(df).iloc[-1])
        atr = safe_float(self.calculate_atr(df).iloc[-1])
        
        sma_50 = safe_float(self.calculate_sma(df['close'], 50).iloc[-1]) if len(df) >= 50 else current
        sma_200 = safe_float(self.calculate_sma(df['close'], 200).iloc[-1]) if len(df) >= 200 else current
        
        ichimoku = self.calculate_ichimoku(df)
        tenkan = safe_float(ichimoku['tenkan'].iloc[-1])
        kijun = safe_float(ichimoku['kijun'].iloc[-1])
        senkou_a = safe_float(ichimoku['senkou_a'].iloc[-1])
        senkou_b = safe_float(ichimoku['senkou_b'].iloc[-1])
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        # Future cloud for leading indication
        future_a = safe_float(ichimoku['future_senkou_a'].iloc[-1])
        future_b = safe_float(ichimoku['future_senkou_b'].iloc[-1])
        future_cloud_bullish = future_a > future_b
        
        gann = self.calculate_gann_levels(df, config.gann_lookback)
        
        # Signal scoring with weights
        bullish = 0
        bearish = 0
        signals = {}
        
        # RSI (weight: 1.0)
        if rsi < RSI_OVERSOLD:
            bullish += 1.0
            signals["RSI"] = f"OVERSOLD ({rsi:.1f})"
        elif rsi > RSI_OVERBOUGHT:
            bearish += 1.0
            signals["RSI"] = f"OVERBOUGHT ({rsi:.1f})"
        else:
            signals["RSI"] = f"NEUTRAL ({rsi:.1f})"
        
        # MACD (weight: 1.0)
        if macd_histogram > 0:
            bullish += 1.0
            signals["MACD"] = "BULLISH"
        else:
            bearish += 1.0
            signals["MACD"] = "BEARISH"
        
        # TK Cross (weight: 0.8)
        if tenkan > kijun:
            bullish += 0.8
            signals["TK_CROSS"] = "BULLISH"
        else:
            bearish += 0.8
            signals["TK_CROSS"] = "BEARISH"
        
        # Price vs Cloud (weight: 1.2)
        if current > cloud_top:
            bullish += 1.2
            signals["CLOUD"] = "ABOVE"
            price_vs_cloud = "ABOVE"
        elif current < cloud_bottom:
            bearish += 1.2
            signals["CLOUD"] = "BELOW"
            price_vs_cloud = "BELOW"
        else:
            signals["CLOUD"] = "INSIDE"
            price_vs_cloud = "INSIDE"
        
        # Gann 50% (weight: 1.5 - most important)
        gann_50 = gann["gann_50"]
        if current > gann_50:
            bullish += 1.5
            signals["GANN_50"] = f"ABOVE (${gann_50:,.0f})"
            price_vs_gann = "ABOVE"
        else:
            bearish += 1.5
            signals["GANN_50"] = f"BELOW (${gann_50:,.0f})"
            price_vs_gann = "BELOW"
        
        # SMA 200 (weight: 1.0)
        if current > sma_200:
            bullish += 1.0
            signals["SMA_200"] = f"ABOVE"
        else:
            bearish += 1.0
            signals["SMA_200"] = f"BELOW"
        
        # Future cloud (weight: 0.5 - leading)
        if future_cloud_bullish:
            bullish += 0.5
            signals["FUTURE_CLOUD"] = "BULLISH"
        else:
            bearish += 0.5
            signals["FUTURE_CLOUD"] = "BEARISH"
        
        # Direction determination with threshold
        total_signals = bullish + bearish
        bullish_pct = (bullish / total_signals) * 100 if total_signals > 0 else 50
        
        if tf_name == "1M":
            # Monthly needs stronger conviction
            if bullish_pct > 65:
                direction = "BULLISH"
            elif bullish_pct < 35:
                direction = "BEARISH"
            else:
                direction = "NEUTRAL"
        else:
            if bullish_pct > 55:
                direction = "BULLISH"
            elif bullish_pct < 45:
                direction = "BEARISH"
            else:
                direction = "NEUTRAL"
        
        # State name
        state = self._get_state_name(rsi, adx, direction)
        
        # Volume
        vol_ratio = 1.0
        if 'volume' in df.columns and len(df) >= 20:
            avg_vol = safe_float(df['volume'].tail(20).mean(), 1)
            vol_ratio = safe_float(df['volume'].iloc[-1], 0) / avg_vol
        
        return {
            "timeframe": tf_name,
            "direction": direction,
            "state_name": state,
            "bullish_score": safe_round(bullish, 2),
            "bearish_score": safe_round(bearish, 2),
            "bullish_pct": safe_round(bullish_pct, 1),
            "rsi": safe_round(rsi, 2),
            "macd": safe_round(macd, 2),
            "macd_histogram": safe_round(macd_histogram, 2),
            "adx": safe_round(adx, 2),
            "adx_label": "STRONG" if adx > ADX_STRONG_TREND else "WEAK",
            "atr": safe_round(atr, 2),
            "atr_pct": safe_round((atr / current) * 100, 2),
            "sma_50": safe_round(sma_50, 2),
            "sma_200": safe_round(sma_200, 2),
            "price_vs_sma_200": "ABOVE" if current > sma_200 else "BELOW",
            "volume_ratio": safe_round(vol_ratio, 2),
            "gann_high": gann["high"],
            "gann_low": gann["low"],
            "gann_50_pct": gann["gann_50"],
            "price_vs_gann_50": price_vs_gann,
            "signal_details": signals,
            "ichimoku": {
                "tenkan": safe_round(tenkan, 2),
                "kijun": safe_round(kijun, 2),
                "cloud_top": safe_round(cloud_top, 2),
                "cloud_bottom": safe_round(cloud_bottom, 2),
                "tk_cross": signals.get("TK_CROSS"),
                "price_vs_cloud": price_vs_cloud,
                "future_cloud": signals.get("FUTURE_CLOUD")
            },
            "gann": gann
        }
    
    def _empty_tf_analysis(self, tf_name: str) -> Dict:
        return {
            "timeframe": tf_name, "direction": "NEUTRAL", "state_name": "Insufficient Data",
            "bullish_score": 0, "bearish_score": 0, "bullish_pct": 50,
            "rsi": 50, "macd": 0, "macd_histogram": 0, "adx": 0, "adx_label": "N/A",
            "atr": 0, "atr_pct": 0, "sma_50": 0, "sma_200": 0, "price_vs_sma_200": "N/A",
            "volume_ratio": 1, "gann_high": 0, "gann_low": 0, "gann_50_pct": 0,
            "price_vs_gann_50": "N/A", "signal_details": {},
            "ichimoku": {"tenkan": 0, "kijun": 0, "cloud_top": 0, "cloud_bottom": 0, "tk_cross": "N/A", "price_vs_cloud": "N/A", "future_cloud": "N/A"},
            "gann": {"high": 0, "low": 0, "levels": {}, "gann_50": 0}
        }
    
    def _get_state_name(self, rsi: float, adx: float, direction: str) -> str:
        if rsi < 25:
            return "Capitulation"
        elif rsi < 35:
            return "Fear"
        elif rsi > 75:
            return "Euphoria"
        elif rsi > 65:
            return "Greed"
        elif adx > 50:
            return "Expansion" if direction == "BULLISH" else "Contraction"
        elif adx < 20:
            return "Consolidation"
        return "Transition"
    
    # ========================================================
    # MTF CONSENSUS (Adaptive Weights)
    # ========================================================
    
    def calculate_mtf_consensus(self, timeframes: Dict, regime: Dict) -> Dict:
        """Calculate consensus with adaptive weights based on regime."""
        
        # Determine weight mode
        is_trending = regime["current"] in ["TRENDING_BULL", "TRENDING_BEAR"]
        
        weighted_bullish = 0
        weighted_bearish = 0
        alignment_count = 0
        conflicts = []
        
        for tf_name, analysis in timeframes.items():
            config = TIMEFRAME_CONFIGS[tf_name]
            
            # Use adaptive weight based on regime
            if is_trending:
                weight = config.trend_weight
            else:
                weight = config.range_weight
            
            # Add weighted score
            weighted_bullish += weight * analysis["bullish_score"]
            weighted_bearish += weight * analysis["bearish_score"]
        
        # Determine primary direction
        total = weighted_bullish + weighted_bearish
        if total == 0:
            total = 1
        
        score_diff = weighted_bullish - weighted_bearish
        weighted_score = int((score_diff / total) * 100)
        
        if weighted_score > 15:
            primary_direction = "BULLISH"
        elif weighted_score < -15:
            primary_direction = "BEARISH"
        else:
            primary_direction = "NEUTRAL"
        
        # Count alignment
        for tf_name, analysis in timeframes.items():
            if analysis["direction"] == primary_direction:
                alignment_count += 1
            elif analysis["direction"] != "NEUTRAL" and primary_direction != "NEUTRAL":
                conflicts.append({
                    "timeframe": tf_name,
                    "direction": analysis["direction"],
                    "expected": primary_direction
                })
        
        # Confidence level
        if alignment_count >= 4 and abs(weighted_score) > 40:
            confidence = "HIGH"
        elif alignment_count >= 3 and abs(weighted_score) > 25:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        # Downgrade if regime override active
        if regime.get("override_active") and confidence == "HIGH":
            confidence = "MEDIUM"
        
        # Verdict
        if confidence == "HIGH":
            verdict = f"STRONG {primary_direction}"
        elif confidence == "MEDIUM":
            verdict = f"MODERATE {primary_direction}"
        else:
            verdict = f"WEAK {primary_direction}" if primary_direction != "NEUTRAL" else "MIXED"
        
        return {
            "primary_direction": primary_direction,
            "weighted_score": weighted_score,
            "alignment": f"{alignment_count}/4",
            "alignment_count": alignment_count,
            "confidence_level": confidence,
            "verdict": verdict,
            "conflicts": conflicts,
            "has_conflicts": len(conflicts) > 0,
            "weight_mode": "TREND" if is_trending else "RANGE",
            "tf_1m": timeframes.get("1M", {}).get("direction", "N/A"),
            "tf_1w": timeframes.get("1W", {}).get("direction", "N/A"),
            "tf_3d": timeframes.get("3D", {}).get("direction", "N/A"),
            "tf_1d": timeframes.get("1D", {}).get("direction", "N/A")
        }
    
    # ========================================================
    # PRIMARY BIAS (Gann 50% Rule with Context)
    # ========================================================
    
    def determine_primary_bias(self, current_price: float, gann_context: Dict, consensus: Dict) -> Dict:
        """Determine primary bias using Gann 50% with full context."""
        
        gann_bias = gann_context["bias"]
        consensus_dir = consensus["primary_direction"]
        
        # Map gann bias variations to simple direction
        if gann_bias in ["BULLISH", "BULLISH_BREAKOUT", "WEAK_BULLISH"]:
            gann_direction = "BULLISH"
        elif gann_bias in ["BEARISH", "BEARISH_BREAKDOWN", "WEAK_BEARISH"]:
            gann_direction = "BEARISH"
        else:
            gann_direction = "NEUTRAL"
        
        # Determine final bias
        conflict = False
        
        if gann_direction != "NEUTRAL" and gann_direction != consensus_dir and consensus_dir != "NEUTRAL":
            # Gann 50% rule takes precedence
            primary_bias = gann_direction
            bias_source = "GANN_50_OVERRIDE"
            conflict = True
            note = f"⚠️ Gann 50% ({gann_bias}) overrides MTF consensus ({consensus_dir})"
        else:
            primary_bias = consensus_dir if consensus_dir != "NEUTRAL" else gann_direction
            bias_source = "CONSENSUS" if consensus_dir == primary_bias else "GANN_50"
            note = f"Bias aligned: {primary_bias}"
        
        return {
            "primary_bias": primary_bias,
            "bias_source": bias_source,
            "gann_50_level": gann_context["gann_50"],
            "gann_position": gann_context["position"],
            "gann_bias_detail": gann_bias,
            "consensus_direction": consensus_dir,
            "conflict": conflict,
            "note": note,
            "description": gann_context["description"]
        }
    
    # ========================================================
    # TRADE SETUPS (Realistic R:R)
    # ========================================================
    
    def generate_trade_setups(self, current_price: float, timeframes: Dict, consensus: Dict,
                             regime: Dict, capitulation: Dict, weekly_gann: Dict,
                             gann_context: Dict) -> List[Dict]:
        """Generate trade setups with realistic R:R."""
        setups = []
        
        atr = safe_float(timeframes["1D"]["atr"])
        if atr == 0:
            atr = current_price * 0.02
        
        # Key levels
        gann_2_8 = weekly_gann["levels"].get("2_8", current_price * 0.88)
        gann_3_8 = weekly_gann["levels"].get("3_8", current_price * 0.94)
        gann_5_8 = weekly_gann["levels"].get("5_8", current_price * 1.06)
        gann_6_8 = weekly_gann["levels"].get("6_8", current_price * 1.12)
        
        # Ichimoku levels for additional S/R
        weekly_cloud_top = safe_float(timeframes["1W"]["ichimoku"]["cloud_top"])
        weekly_cloud_bottom = safe_float(timeframes["1W"]["ichimoku"]["cloud_bottom"])
        
        primary_dir = consensus["primary_direction"]
        allows_long = regime.get("allows_long", True)
        allows_short = regime.get("allows_short", True)
        size_cap = regime.get("position_size_cap", 1.0)
        
        # PRIMARY LONG
        if primary_dir == "BULLISH" and allows_long:
            entry = current_price
            
            # Stop: below nearest support or 1.5 ATR
            stop_candidates = [gann_3_8 - atr * 0.5, weekly_cloud_bottom - atr * 0.3, entry - atr * 1.5]
            stop = max([s for s in stop_candidates if s < entry])
            risk = entry - stop
            
            # TP1: nearest resistance or 1.5x risk
            tp1_candidates = [gann_5_8, weekly_cloud_top if weekly_cloud_top > entry else None, entry + risk * 1.5]
            tp1_candidates = [t for t in tp1_candidates if t and t > entry]
            tp1 = min(tp1_candidates) if tp1_candidates else entry + risk * 1.5
            
            # Ensure minimum R:R
            if (tp1 - entry) / risk < MIN_RR_RATIO:
                tp1 = entry + risk * MIN_RR_RATIO
            
            tp2 = max(gann_6_8, entry + risk * 2.5)
            tp3 = entry + risk * 3.5
            
            rr = (tp1 - entry) / risk if risk > 0 else 0
            
            # Confidence based on alignment
            if consensus["alignment_count"] >= 3:
                confidence = "HIGH"
                base_size = 1.0
            elif consensus["alignment_count"] >= 2:
                confidence = "MEDIUM"
                base_size = 0.5
            else:
                confidence = "LOW"
                base_size = 0.25
            
            setups.append({
                "id": 1, "type": "PRIMARY", "direction": "LONG",
                "confidence": confidence,
                "entry": safe_round(entry, 2),
                "stop_loss": safe_round(stop, 2),
                "tp1": safe_round(tp1, 2),
                "tp2": safe_round(tp2, 2),
                "tp3": safe_round(tp3, 2),
                "rr_ratio": safe_round(max(rr, MIN_RR_RATIO), 2),
                "risk_pct": safe_round((risk / entry) * 100, 2),
                "position_size": safe_round(min(base_size, size_cap), 2),
                "rationale": f"MTF {consensus['verdict']}, {gann_context['bias']}"
            })
        
        # PRIMARY SHORT
        elif primary_dir == "BEARISH" and allows_short:
            entry = current_price
            
            # Stop: above nearest resistance or 1.5 ATR
            stop_candidates = [gann_5_8 + atr * 0.5, weekly_cloud_top + atr * 0.3, entry + atr * 1.5]
            stop = min([s for s in stop_candidates if s > entry])
            risk = stop - entry
            
            # TP1: nearest support or 1.5x risk
            tp1_candidates = [gann_3_8, weekly_cloud_bottom if weekly_cloud_bottom < entry else None, entry - risk * 1.5]
            tp1_candidates = [t for t in tp1_candidates if t and t < entry]
            tp1 = max(tp1_candidates) if tp1_candidates else entry - risk * 1.5
            
            # Ensure minimum R:R
            if (entry - tp1) / risk < MIN_RR_RATIO:
                tp1 = entry - risk * MIN_RR_RATIO
            
            tp2 = min(gann_2_8, entry - risk * 2.5)
            tp3 = entry - risk * 3.5
            
            rr = (entry - tp1) / risk if risk > 0 else 0
            
            if consensus["alignment_count"] >= 3:
                confidence = "HIGH"
                base_size = 1.0
            elif consensus["alignment_count"] >= 2:
                confidence = "MEDIUM"
                base_size = 0.5
            else:
                confidence = "LOW"
                base_size = 0.25
            
            setups.append({
                "id": 1, "type": "PRIMARY", "direction": "SHORT",
                "confidence": confidence,
                "entry": safe_round(entry, 2),
                "stop_loss": safe_round(stop, 2),
                "tp1": safe_round(tp1, 2),
                "tp2": safe_round(tp2, 2),
                "tp3": safe_round(tp3, 2),
                "rr_ratio": safe_round(max(rr, MIN_RR_RATIO), 2),
                "risk_pct": safe_round((risk / entry) * 100, 2),
                "position_size": safe_round(min(base_size, size_cap), 2),
                "rationale": f"MTF {consensus['verdict']}, {gann_context['bias']}"
            })
        
        # COUNTER-TREND at capitulation
        if capitulation.get("is_capitulation") and allows_long:
            entry = gann_3_8
            stop = gann_2_8 - atr
            risk = entry - stop
            tp1 = current_price
            tp2 = gann_5_8
            rr = (tp1 - entry) / risk if risk > 0 else 0
            
            setups.append({
                "id": 2, "type": "COUNTER_TREND", "direction": "LONG",
                "confidence": "MEDIUM" if capitulation["status"] == "CONFIRMED" else "LOW",
                "entry": safe_round(entry, 2),
                "stop_loss": safe_round(stop, 2),
                "tp1": safe_round(tp1, 2),
                "tp2": safe_round(tp2, 2),
                "tp3": None,
                "rr_ratio": safe_round(max(rr, MIN_RR_RATIO), 2),
                "risk_pct": safe_round((risk / entry) * 100, 2),
                "position_size": 0.25,
                "rationale": f"Capitulation bounce ({capitulation['status']})"
            })
        
        # WAIT if unclear
        if not setups or consensus["alignment_count"] < 2:
            setups.append({
                "id": len(setups) + 1, "type": "WAIT", "direction": "FLAT",
                "confidence": "NONE",
                "entry": safe_round(current_price, 2),
                "stop_loss": None, "tp1": None, "tp2": None, "tp3": None,
                "rr_ratio": 0, "risk_pct": 0, "position_size": 0,
                "rationale": "Low alignment or conflicting signals"
            })
        
        # Sort by confidence
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}
        setups.sort(key=lambda x: order.get(x["confidence"], 4))
        
        return setups
    
    # ========================================================
    # TIME FORECAST
    # ========================================================
    
    def calculate_time_forecast(self, df_daily: pd.DataFrame, current_price: float,
                                atr: float, weekly_gann: Dict) -> Dict:
        """Calculate time-based pivot forecast."""
        lookback = min(365, len(df_daily))
        recent = df_daily.tail(lookback)
        
        major_high = safe_float(recent['high'].max())
        major_low = safe_float(recent['low'].min())
        
        ts_col = 'date' if 'date' in df_daily.columns else 'timestamp'
        
        try:
            high_idx = recent['high'].idxmax()
            low_idx = recent['low'].idxmin()
            
            if high_idx > low_idx:
                ref_type = "HIGH"
                pivot_type = "LOW"
                ref_idx = high_idx
                ref_price = major_high
            else:
                ref_type = "LOW"
                pivot_type = "HIGH"
                ref_idx = low_idx
                ref_price = major_low
            
            ref_date = pd.to_datetime(df_daily.loc[ref_idx, ts_col])
            if ref_date.tzinfo:
                ref_date = ref_date.replace(tzinfo=None)
            ref_date_str = ref_date.strftime("%Y-%m-%d")
            days_since = int((datetime.now() - ref_date).days)
        except:
            ref_type = "HIGH"
            pivot_type = "LOW"
            days_since = 90
            ref_date_str = "N/A"
            ref_price = current_price
        
        # Find next cycle
        next_date = None
        days_to = 90
        confidence = 0.40
        cycle_used = 90
        
        for cycle in GANN_CYCLES:
            if cycle > days_since:
                days_to = cycle - days_since
                next_date = (datetime.now() + timedelta(days=days_to)).strftime("%Y-%m-%d")
                confidence = 0.55 if cycle in [180, 360] else 0.50 if cycle in [90, 144] else 0.45
                cycle_used = cycle
                break
        
        if not next_date:
            days_to = 90
            next_date = (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d")
        
        # Price projection
        gann_3_8 = weekly_gann["levels"].get("3_8", current_price * 0.9)
        gann_5_8 = weekly_gann["levels"].get("5_8", current_price * 1.1)
        atr_proj = atr * np.sqrt(days_to) if atr > 0 else current_price * 0.10
        
        low_proj = max(current_price - atr_proj, gann_3_8 * 0.95)
        high_proj = min(current_price + atr_proj, gann_5_8 * 1.05)
        
        return {
            "next_pivot_date": next_date,
            "days_to_pivot": days_to,
            "pivot_type": pivot_type,
            "confidence": safe_round(confidence, 2),
            "confidence_pct": f"{int(confidence * 100)}%",
            "confidence_level": "HIGH" if confidence >= 0.65 else "MEDIUM" if confidence >= 0.50 else "LOW",
            "probable_price_low": safe_round(low_proj, 2),
            "probable_price_high": safe_round(high_proj, 2),
            "price_range": f"${low_proj:,.0f} - ${high_proj:,.0f}",
            "cycle_origin": {
                "reference_type": ref_type,
                "reference_date": ref_date_str,
                "reference_price": safe_round(ref_price, 2),
                "days_since": days_since,
                "cycle_length": cycle_used
            }
        }
    
    # ========================================================
    # MAIN SIGNAL GENERATOR
    # ========================================================
    
    def generate_mtf_signal(self, symbol: str = "BTCUSDT") -> Dict:
        """Main MTF signal generator - Coherent v5.0.9."""
        try:
            logger.info(f"[MTF] Generating signal for {symbol}")
            
            # Fetch data
            df_1d = self.fetch_real_binance_data(use_cache=True, symbol=symbol)
            if df_1d is None or len(df_1d) < 100:
                raise ValueError(f"Insufficient data: {len(df_1d) if df_1d else 0}")
            
            logger.info(f"[MTF] {len(df_1d)} candles")
            
            # Resample
            df_3d = self.resample_ohlcv(df_1d, "3D")
            df_1w = self.resample_ohlcv(df_1d, "1W")
            df_1m = self.resample_ohlcv(df_1d, "1M")
            
            current_price = safe_float(df_1d['close'].iloc[-1])
            ts_col = 'date' if 'date' in df_1d.columns else 'timestamp'
            signal_date = str(df_1d[ts_col].iloc[-1])[:10]
            
            # Analyze timeframes
            timeframes = {
                "1D": self.analyze_timeframe(df_1d, "1D"),
                "3D": self.analyze_timeframe(df_3d, "3D"),
                "1W": self.analyze_timeframe(df_1w, "1W"),
                "1M": self.analyze_timeframe(df_1m, "1M"),
            }
            
            # Weekly values for regime/capitulation
            weekly_rsi = safe_float(timeframes["1W"]["rsi"], 50)
            weekly_adx = safe_float(timeframes["1W"]["adx"], 20)
            weekly_gann = timeframes["1W"]["gann"]
            daily_atr = safe_float(timeframes["1D"]["atr"])
            
            # Capitulation detection
            capitulation = self.detect_capitulation(df_1w, df_1d, weekly_rsi, weekly_gann, current_price)
            
            # Regime determination
            regime = self.determine_regime(df_1d, weekly_rsi, weekly_adx, capitulation, current_price)
            
            # MTF consensus
            consensus = self.calculate_mtf_consensus(timeframes, regime)
            
            # Gann 50% context
            gann_context = self.analyze_gann_50_context(current_price, weekly_gann, df_1d, daily_atr)
            
            # Primary bias
            primary_bias = self.determine_primary_bias(current_price, gann_context, consensus)
            
            # Time forecast
            time_forecast = self.calculate_time_forecast(df_1d, current_price, daily_atr, weekly_gann)
            
            # Trade setups
            trade_setups = self.generate_trade_setups(
                current_price, timeframes, consensus, regime, capitulation, weekly_gann, gann_context
            )
            
            # Invalidation
            if primary_bias["primary_bias"] == "BULLISH":
                inv_price = weekly_gann["levels"].get("3_8", current_price * 0.92)
                inv_desc = f"Bullish invalidated below ${inv_price:,.0f}"
            elif primary_bias["primary_bias"] == "BEARISH":
                inv_price = weekly_gann["levels"].get("5_8", current_price * 1.08)
                inv_desc = f"Bearish invalidated above ${inv_price:,.0f}"
            else:
                inv_price = None
                inv_desc = "No clear bias"
            
            # Price levels
            price_levels = {
                "monthly": {
                    "high": timeframes["1M"]["gann_high"],
                    "low": timeframes["1M"]["gann_low"],
                    "gann_50": timeframes["1M"]["gann_50_pct"]
                },
                "weekly": {
                    "high": weekly_gann["high"],
                    "low": weekly_gann["low"],
                    "gann_50": weekly_gann["gann_50"],
                    "gann_3_8": weekly_gann["levels"].get("3_8"),
                    "gann_5_8": weekly_gann["levels"].get("5_8"),
                    "cloud_top": timeframes["1W"]["ichimoku"]["cloud_top"],
                    "cloud_bottom": timeframes["1W"]["ichimoku"]["cloud_bottom"]
                },
                "daily": {
                    "high": timeframes["1D"]["gann_high"],
                    "low": timeframes["1D"]["gann_low"],
                    "gann_50": timeframes["1D"]["gann_50_pct"],
                    "sma_200": timeframes["1D"]["sma_200"]
                }
            }
            
            # Enneagram
            enneagram = {
                "1M": timeframes["1M"]["state_name"],
                "1W": timeframes["1W"]["state_name"],
                "3D": timeframes["3D"]["state_name"],
                "1D": timeframes["1D"]["state_name"],
                "dominant": timeframes["1W"]["state_name"],
                "phase": "Accumulation" if primary_bias["primary_bias"] == "BULLISH" else "Distribution" if primary_bias["primary_bias"] == "BEARISH" else "Transition",
                "arrow": "↑" if primary_bias["primary_bias"] == "BULLISH" else "↓" if primary_bias["primary_bias"] == "BEARISH" else "→"
            }
            
            logger.info(f"[MTF] Done: {consensus['primary_direction']} ({consensus['confidence_level']})")
            
            result = {
                "status": "success",
                "symbol": symbol,
                "current_price": safe_round(current_price, 2),
                "signal_date": signal_date,
                "timestamp": datetime.now().isoformat(),
                "version": self.VERSION,
                
                "regime": regime,
                "timeframes": timeframes,
                "consensus": consensus,
                "primary_bias": primary_bias,
                "gann_context": gann_context,
                "price_levels": price_levels,
                "capitulation": capitulation,
                "time_forecast": time_forecast,
                "trade_setups": trade_setups,
                
                "invalidation": {
                    "price": safe_round(inv_price, 2),
                    "description": inv_desc
                },
                
                "enneagram": enneagram,
                
                "gann_interpretation": {
                    "weekly_50_pct": weekly_gann["gann_50"],
                    "current_position": gann_context["position"],
                    "primary_bias": primary_bias["primary_bias"],
                    "description": gann_context["description"],
                    "rule": "Above 50% = Bulls | Below 50% = Bears"
                },
                
                "signal": {
                    "type": trade_setups[0]["direction"] if trade_setups else "WAIT",
                    "confidence": trade_setups[0]["confidence"] if trade_setups else "NONE",
                    "entry": trade_setups[0].get("entry"),
                    "stop_loss": trade_setups[0].get("stop_loss"),
                    "take_profit": trade_setups[0].get("tp1"),
                    "tp2": trade_setups[0].get("tp2"),
                    "tp3": trade_setups[0].get("tp3"),
                    "rr_ratio": trade_setups[0].get("rr_ratio"),
                    "position_size": trade_setups[0].get("position_size", 0),
                    "rationale": trade_setups[0].get("rationale", "")
                }
            }
            
            return convert_numpy_types(result)
        
        except Exception as e:
            logger.error(f"[MTF] Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "detail": str(e),
                "version": self.VERSION,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_daily_signal(self, df=None):
        return self.generate_mtf_signal()
