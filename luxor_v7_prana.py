# ============================================================
# LUXOR V7 PRANA - GANN EGYPT-INDIA UNIFIED SYSTEM v5.0.8
# COMPLETE VERSION - All fields mapped for Telegram
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
# HELPER: Convert numpy types to native Python
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

def safe_round(value, decimals=2):
    """Safely round a value, handling None and NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), decimals)

def safe_format_price(value):
    """Format price with comma separator."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"${value:,.0f}"

# ============================================================
# ENUMS & CONFIGURATION
# ============================================================

class Direction(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

class Regime(Enum):
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    RANGING = "RANGING"
    CAPITULATION = "CAPITULATION"
    EUPHORIA = "EUPHORIA"

class Confidence(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"

@dataclass
class TimeframeConfig:
    name: str
    weight: float
    gann_lookback: int
    min_bars: int

# Timeframe weights: Monthly most important, Daily least
TIMEFRAME_CONFIGS = {
    "1M": TimeframeConfig("1M", 0.35, 24, 12),
    "1W": TimeframeConfig("1W", 0.30, 52, 26),
    "3D": TimeframeConfig("3D", 0.20, 120, 60),
    "1D": TimeframeConfig("1D", 0.15, 252, 100),
}

# Thresholds
RSI_OVERSOLD = 30
RSI_EXTREME_OVERSOLD = 25
RSI_OVERBOUGHT = 70
RSI_EXTREME_OVERBOUGHT = 75
ADX_STRONG_TREND = 25
ADX_VERY_STRONG = 50
VOLUME_SPIKE_THRESHOLD = 2.0
CAPITULATION_RSI_THRESHOLD = 25
EUPHORIA_RSI_THRESHOLD = 75
MIN_RR_RATIO = 1.5

# Gann cycles (days)
GANN_CYCLES = [30, 45, 60, 90, 120, 144, 180, 270, 360]

# SQ9 angles for level calculation
SQ9_ANGLES = [45, 90, 120, 135, 180, 225, 270, 315, 360]

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class GannLevels:
    high: float
    low: float
    high_date: str
    low_date: str
    range_value: float
    range_pct: float
    lookback_bars: int
    levels: Dict[str, float]
    gann_50_pct: float

# ============================================================
# MAIN CLASS
# ============================================================

class LuxorV7PranaSystem:
    """LUXOR V7 PRANA - GANN EGYPT-INDIA UNIFIED SYSTEM v5.0.8"""
    
    CACHE = {
        'df': None,
        'last_fetch': None,
        'cache_duration': 3600
    }
    
    VERSION = "5.0.8"
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.sidereal_epoch = pd.Timestamp('1900-01-01')
        logger.info(f"[INIT] LuxorV7PranaSystem v{self.VERSION} initialized")
    
    # ========================================================
    # DATA FETCHING (Multi-Exchange)
    # ========================================================
    
    def fetch_ohlcv_ccxt(self, symbol: str = "BTC/USDT", interval: str = "1d", limit: int = 500) -> pd.DataFrame:
        """Fetch OHLCV using CCXT with multi-exchange fallback."""
        base_symbol = symbol.split('/')[0].upper() if '/' in symbol else symbol[:3].upper()
        
        if base_symbol == 'BTC' or base_symbol == 'XBT':
            exchanges_to_try = [
                ('kucoin', 'BTC/USDT'),
                ('bybit', 'BTC/USDT'),
                ('okx', 'BTC/USDT'),
                ('kraken', 'BTC/USD'),
                ('gate', 'BTC/USDT'),
            ]
        elif base_symbol == 'ETH':
            exchanges_to_try = [
                ('kucoin', 'ETH/USDT'),
                ('bybit', 'ETH/USDT'),
                ('okx', 'ETH/USDT'),
                ('kraken', 'ETH/USD'),
            ]
        else:
            exchanges_to_try = [
                ('kucoin', f'{base_symbol}/USDT'),
                ('bybit', f'{base_symbol}/USDT'),
                ('okx', f'{base_symbol}/USDT'),
            ]
        
        last_error = None
        
        for exchange_id, sym in exchanges_to_try:
            try:
                logger.info(f"[DATA] Trying {exchange_id} for {sym}")
                exchange_class = getattr(ccxt, exchange_id, None)
                
                if exchange_class is None:
                    continue
                
                exchange = exchange_class({
                    'enableRateLimit': True,
                    'timeout': 30000,
                })
                
                ohlcv = exchange.fetch_ohlcv(sym, interval, limit=limit)
                
                if ohlcv and len(ohlcv) > 0:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['date'] = df['timestamp']
                    
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                    
                    df = df.dropna(subset=['close'])
                    df = df.reset_index(drop=True)
                    
                    logger.info(f"[DATA] Fetched {len(df)} candles from {exchange_id}")
                    return df
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[DATA] {exchange_id} failed: {str(e)[:100]}")
                continue
        
        raise Exception(f"All exchanges failed. Last error: {last_error}")
    
    def fetch_real_binance_data(self, use_cache=True, symbol: str = "BTCUSDT") -> pd.DataFrame:
        """Fetch data with caching."""
        try:
            if use_cache and self.CACHE['df'] is not None and self.CACHE['last_fetch'] is not None:
                cache_age = (datetime.now() - self.CACHE['last_fetch']).total_seconds()
                if cache_age < self.CACHE['cache_duration']:
                    logger.info(f"[CACHE] Using cached data (age: {cache_age:.0f}s)")
                    return self.CACHE['df'].copy()
            
            # Convert symbol format
            if '/' not in symbol:
                if symbol.endswith('USDT'):
                    ccxt_symbol = symbol[:-4] + '/USDT'
                elif symbol.endswith('USD'):
                    ccxt_symbol = symbol[:-3] + '/USD'
                else:
                    ccxt_symbol = symbol + '/USDT'
            else:
                ccxt_symbol = symbol
            
            df = self.fetch_ohlcv_ccxt(ccxt_symbol, "1d", 500)
            
            self.CACHE['df'] = df.copy()
            self.CACHE['last_fetch'] = datetime.now()
            
            return df
            
        except Exception as e:
            logger.error(f"[ERROR] fetch_real_binance_data: {e}")
            raise
    
    def resample_ohlcv(self, df_1d: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """Resample daily data to higher timeframes."""
        df = df_1d.copy()
        
        # Set index
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
        elif 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        
        # Use 'M' not 'ME' for pandas <2.2 compatibility
        resample_map = {'3D': '3D', '1W': 'W', '1M': 'M'}
        rule = resample_map.get(target_tf, '1D')
        
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
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
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX."""
        high, low, close = df['high'], df['low'], df['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff().abs() * -1
        
        plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
        minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0)
        
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        atr = atr.replace(0, 1)
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / atr)
        
        di_sum = (plus_di + minus_di).replace(0, 1)
        dx = 100 * (plus_di - minus_di).abs() / di_sum
        
        adx = dx.ewm(alpha=1/period, min_periods=period).mean()
        return adx.fillna(0)
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.fillna(tr.mean() if len(tr) > 0 and tr.mean() > 0 else 1.0)
    
    def calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate SMA."""
        return prices.rolling(window=period).mean()
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate EMA."""
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict:
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
        
        # Tenkan-sen (Conversion Line): 9-period
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        
        # Kijun-sen (Base Line): 26-period
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        
        # Senkou Span A (Leading Span A): shifted 26 periods ahead
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): 52-period, shifted 26 ahead
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span): close shifted 26 periods back
        chikou = close.shift(-26)
        
        return {
            'tenkan': tenkan.fillna(close),
            'kijun': kijun.fillna(close),
            'senkou_a': senkou_a.fillna(close),
            'senkou_b': senkou_b.fillna(close),
            'chikou': chikou.fillna(close)
        }
    
    def calculate_volume_analysis(self, df: pd.DataFrame) -> Dict:
        """Comprehensive volume analysis."""
        volume = df['volume']
        close = df['close']
        
        # Volume MA
        vol_sma_20 = volume.rolling(20).mean()
        vol_sma_50 = volume.rolling(50).mean()
        
        # Current volume ratio
        current_vol = float(volume.iloc[-1]) if len(volume) > 0 else 0
        avg_vol_20 = float(vol_sma_20.iloc[-1]) if len(vol_sma_20) > 0 and not pd.isna(vol_sma_20.iloc[-1]) else 1
        volume_ratio = current_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0
        
        # Volume trend
        vol_trend = "INCREASING" if current_vol > avg_vol_20 * 1.2 else "DECREASING" if current_vol < avg_vol_20 * 0.8 else "NORMAL"
        
        # Volume spike detection
        is_spike = volume_ratio >= VOLUME_SPIKE_THRESHOLD
        
        # OBV (On Balance Volume) simplified
        obv_direction = "UP" if len(close) > 1 and close.iloc[-1] > close.iloc[-2] else "DOWN"
        
        return {
            'current_volume': safe_round(current_vol, 0),
            'avg_volume_20': safe_round(avg_vol_20, 0),
            'volume_ratio': safe_round(volume_ratio, 2),
            'volume_trend': vol_trend,
            'is_spike': is_spike,
            'obv_direction': obv_direction,
            'interpretation': "High volume confirms move" if is_spike else "Normal volume"
        }
    
    # ========================================================
    # GANN CALCULATIONS
    # ========================================================
    
    def calculate_gann_levels(self, df: pd.DataFrame, tf_config: TimeframeConfig) -> GannLevels:
        """Calculate Gann levels with all 8 divisions."""
        lookback = min(tf_config.gann_lookback, len(df))
        recent_df = df.tail(lookback)
        
        high = float(recent_df['high'].max())
        low = float(recent_df['low'].min())
        high_idx = recent_df['high'].idxmax()
        low_idx = recent_df['low'].idxmin()
        
        ts_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        
        try:
            high_date = str(df.loc[high_idx, ts_col])[:10]
            low_date = str(df.loc[low_idx, ts_col])[:10]
        except:
            high_date = "N/A"
            low_date = "N/A"
        
        range_value = high - low
        current_price = float(df['close'].iloc[-1])
        range_pct = (range_value / current_price) * 100 if current_price > 0 else 0
        
        # Calculate all Gann levels (0/8 through 8/8)
        levels = {}
        for i in range(9):
            level_value = low + (range_value * i / 8)
            levels[f"{i}_8"] = safe_round(level_value, 2)
        
        # Explicit 50% level (4/8)
        gann_50_pct = safe_round(low + (range_value * 0.5), 2)
        
        return GannLevels(
            high=safe_round(high, 2),
            low=safe_round(low, 2),
            high_date=high_date,
            low_date=low_date,
            range_value=safe_round(range_value, 2),
            range_pct=safe_round(range_pct, 2),
            lookback_bars=int(lookback),
            levels=levels,
            gann_50_pct=gann_50_pct
        )
    
    def calculate_sq9_levels(self, current_price: float, filter_pct_min: float = 2.0, filter_pct_max: float = 5.0) -> Dict:
        """Calculate Square of 9 levels with 2-5% filter for actionable moves."""
        sqrt_price = math.sqrt(current_price)
        levels = {'support': [], 'resistance': []}
        
        for angle in SQ9_ANGLES:
            # Calculate levels at each angle
            for direction in [-1, 1]:  # -1 for support, +1 for resistance
                for rotation in range(1, 4):  # 1-3 rotations
                    adjustment = (angle / 360) * direction * rotation
                    new_sqrt = sqrt_price + adjustment
                    new_price = new_sqrt ** 2
                    
                    # Calculate distance percentage
                    distance_pct = abs((new_price - current_price) / current_price) * 100
                    
                    # Filter: only include levels 2-5% away
                    if filter_pct_min <= distance_pct <= filter_pct_max:
                        level_data = {
                            'price': safe_round(new_price, 2),
                            'angle': angle,
                            'rotation': rotation,
                            'distance_pct': safe_round(distance_pct, 2)
                        }
                        
                        if new_price < current_price:
                            levels['support'].append(level_data)
                        else:
                            levels['resistance'].append(level_data)
        
        # Sort by distance
        levels['support'] = sorted(levels['support'], key=lambda x: x['distance_pct'])[:5]
        levels['resistance'] = sorted(levels['resistance'], key=lambda x: x['distance_pct'])[:5]
        
        return {
            'levels': levels,
            'nearest_support': levels['support'][0] if levels['support'] else None,
            'nearest_resistance': levels['resistance'][0] if levels['resistance'] else None
        }
    
    # ========================================================
    # DIVERGENCE & CAPITULATION
    # ========================================================
    
    def detect_divergence(self, df: pd.DataFrame, rsi: pd.Series, lookback: int = 14) -> Dict:
        """Detect bullish and bearish divergences."""
        if len(df) < lookback * 2:
            return {"bullish_divergence": False, "bearish_divergence": False, "description": "Insufficient data"}
        
        recent_prices = df['close'].tail(lookback * 2)
        recent_rsi = rsi.tail(lookback * 2)
        
        # Bullish divergence: lower price low, higher RSI low
        price_min_1 = float(recent_prices.iloc[:lookback].min())
        price_min_2 = float(recent_prices.iloc[lookback:].min())
        rsi_min_1 = float(recent_rsi.iloc[:lookback].min())
        rsi_min_2 = float(recent_rsi.iloc[lookback:].min())
        
        bullish_div = bool((price_min_2 < price_min_1) and (rsi_min_2 > rsi_min_1))
        
        # Bearish divergence: higher price high, lower RSI high
        price_max_1 = float(recent_prices.iloc[:lookback].max())
        price_max_2 = float(recent_prices.iloc[lookback:].max())
        rsi_max_1 = float(recent_rsi.iloc[:lookback].max())
        rsi_max_2 = float(recent_rsi.iloc[lookback:].max())
        
        bearish_div = bool((price_max_2 > price_max_1) and (rsi_max_2 < rsi_max_1))
        
        description = ""
        if bullish_div:
            description = "Bullish divergence: Price made lower low but RSI made higher low"
        elif bearish_div:
            description = "Bearish divergence: Price made higher high but RSI made lower high"
        else:
            description = "No divergence detected"
        
        return {
            "bullish_divergence": bullish_div,
            "bearish_divergence": bearish_div,
            "description": description
        }
    
    def detect_capitulation(self, df_weekly: pd.DataFrame, df_daily: pd.DataFrame,
                           weekly_rsi: float, weekly_gann: GannLevels, current_price: float) -> Dict:
        """
        Detect capitulation with 4 criteria:
        1. Weekly RSI extreme (<25)
        2. Volume spike (>2x average)
        3. Near Gann 2/8 or 3/8 support
        4. Bullish RSI divergence
        
        Status: CONFIRMED (≥3), POTENTIAL (2), DEVELOPING (1), NONE (0)
        """
        criteria_met = []
        criteria_missing = []
        
        # Criterion 1: RSI extreme
        rsi_extreme = bool(weekly_rsi < CAPITULATION_RSI_THRESHOLD)
        if rsi_extreme:
            criteria_met.append(f"RSI_EXTREME ({weekly_rsi:.1f} < 25)")
        else:
            criteria_missing.append(f"RSI not extreme ({weekly_rsi:.1f})")
        
        # Criterion 2: Volume spike
        volume_ratio = 1.0
        if len(df_daily) >= 20 and 'volume' in df_daily.columns:
            avg_vol = float(df_daily['volume'].tail(20).mean())
            if avg_vol > 0:
                volume_ratio = float(df_daily['volume'].iloc[-1] / avg_vol)
        
        volume_spike = bool(volume_ratio >= VOLUME_SPIKE_THRESHOLD)
        if volume_spike:
            criteria_met.append(f"VOLUME_SPIKE ({volume_ratio:.1f}x avg)")
        else:
            criteria_missing.append(f"No volume spike ({volume_ratio:.1f}x)")
        
        # Criterion 3: Near Gann support (2/8 or 3/8)
        gann_2_8 = float(weekly_gann.levels.get("2_8", weekly_gann.low))
        gann_3_8 = float(weekly_gann.levels.get("3_8", weekly_gann.low))
        
        near_2_8 = abs(current_price - gann_2_8) / current_price < 0.03
        near_3_8 = abs(current_price - gann_3_8) / current_price < 0.05
        gann_support_test = bool(near_2_8 or near_3_8)
        
        if gann_support_test:
            level = gann_2_8 if near_2_8 else gann_3_8
            criteria_met.append(f"GANN_SUPPORT (near ${level:,.0f})")
        else:
            criteria_missing.append(f"Not near Gann support")
        
        # Criterion 4: Bullish divergence
        weekly_rsi_series = self.calculate_rsi(df_weekly['close'])
        divergence = self.detect_divergence(df_weekly, weekly_rsi_series, 8)
        bullish_divergence = bool(divergence.get("bullish_divergence", False))
        
        if bullish_divergence:
            criteria_met.append("BULLISH_DIVERGENCE")
        else:
            criteria_missing.append("No bullish divergence")
        
        # Determine status
        num_criteria = len(criteria_met)
        if num_criteria >= 3:
            status = "CONFIRMED"
            confidence = 0.85
        elif num_criteria >= 2:
            status = "POTENTIAL"
            confidence = 0.60
        elif num_criteria >= 1:
            status = "DEVELOPING"
            confidence = 0.40
        else:
            status = "NONE"
            confidence = 0.10
        
        return {
            "is_capitulation": bool(num_criteria >= 3),
            "status": status,
            "confidence": safe_round(confidence, 2),
            "criteria_count": f"{num_criteria}/4",
            "criteria_met": criteria_met,
            "criteria_missing": criteria_missing,
            "details": {
                "rsi_extreme": rsi_extreme,
                "volume_spike": volume_spike,
                "gann_support_test": gann_support_test,
                "bullish_divergence": bullish_divergence,
                "weekly_rsi": safe_round(weekly_rsi, 2),
                "volume_ratio": safe_round(volume_ratio, 2)
            }
        }
    
    # ========================================================
    # REGIME DETECTION
    # ========================================================
    
    def determine_regime(self, df_daily: pd.DataFrame, weekly_rsi: float, weekly_adx: float,
                        capitulation: Dict, current_price: float) -> Dict:
        """Determine market regime with override logic."""
        warnings_list = []
        override_active = False
        override_reason = None
        
        # Calculate SMA 200
        sma_200 = float(df_daily['close'].rolling(200).mean().iloc[-1]) if len(df_daily) >= 200 else current_price
        price_vs_sma = "ABOVE" if current_price > sma_200 else "BELOW"
        sma_distance_pct = ((current_price - sma_200) / sma_200) * 100
        
        # Trend strength
        if weekly_adx > ADX_VERY_STRONG:
            trend_strength = "VERY_STRONG"
            trend_label = f"STRONG ({weekly_adx:.0f})"
        elif weekly_adx > ADX_STRONG_TREND:
            trend_strength = "STRONG"
            trend_label = f"MODERATE ({weekly_adx:.0f})"
        else:
            trend_strength = "WEAK"
            trend_label = f"WEAK ({weekly_adx:.0f})"
        
        # Base regime
        if price_vs_sma == "ABOVE" and trend_strength in ["STRONG", "VERY_STRONG"]:
            base_regime = "TRENDING_BULL"
        elif price_vs_sma == "BELOW" and trend_strength in ["STRONG", "VERY_STRONG"]:
            base_regime = "TRENDING_BEAR"
        else:
            base_regime = "RANGING"
        
        # Override checks
        if capitulation.get("is_capitulation", False):
            override_active = True
            override_reason = f"CAPITULATION {capitulation.get('status', 'DETECTED')}"
            final_regime = "CAPITULATION"
            allows_short = False
            allows_long = True
            position_size_cap = 0.50 if capitulation.get("status") == "CONFIRMED" else 0.25
            warnings_list.append("⚠️ CAPITULATION: Shorts blocked, reduced size")
        elif weekly_rsi > EUPHORIA_RSI_THRESHOLD:
            override_active = True
            override_reason = f"EUPHORIA (RSI {weekly_rsi:.0f})"
            final_regime = "EUPHORIA"
            allows_short = True
            allows_long = False
            position_size_cap = 0.25
            warnings_list.append("⚠️ EUPHORIA: Longs blocked, reduced size")
        else:
            final_regime = base_regime
            allows_short = True
            allows_long = True
            position_size_cap = 1.0
        
        # Regime strength
        if final_regime in ["TRENDING_BULL", "TRENDING_BEAR"]:
            regime_strength = min(weekly_adx / 100, 1.0)
        else:
            regime_strength = 0.5
        
        return {
            "current": final_regime,
            "base_regime": base_regime,
            "strength": safe_round(regime_strength, 2),
            "strength_label": trend_label,
            "override_active": override_active,
            "override_reason": override_reason,
            "allows_short": allows_short,
            "allows_long": allows_long,
            "position_size_cap": safe_round(position_size_cap, 2),
            "warnings": warnings_list,
            "sma_200": safe_round(sma_200, 2),
            "price_vs_sma_200": price_vs_sma,
            "sma_distance_pct": safe_round(sma_distance_pct, 2)
        }
    
    # ========================================================
    # TIMEFRAME ANALYSIS
    # ========================================================
    
    def analyze_timeframe(self, df: pd.DataFrame, tf_name: str) -> Dict:
        """Complete timeframe analysis with all indicators."""
        tf_config = TIMEFRAME_CONFIGS[tf_name]
        
        # Check minimum bars
        if len(df) < tf_config.min_bars:
            return self._empty_timeframe_analysis(tf_name)
        
        current_price = float(df['close'].iloc[-1])
        
        # Calculate all indicators
        rsi_series = self.calculate_rsi(df['close'])
        rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
        
        macd, macd_signal, macd_hist = self.calculate_macd(df['close'])
        macd_value = float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0.0
        macd_hist_value = float(macd_hist.iloc[-1]) if not pd.isna(macd_hist.iloc[-1]) else 0.0
        
        adx_series = self.calculate_adx(df)
        adx = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0.0
        
        atr_series = self.calculate_atr(df)
        atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0
        
        # SMA
        sma_50 = float(self.calculate_sma(df['close'], 50).iloc[-1]) if len(df) >= 50 else current_price
        sma_200 = float(self.calculate_sma(df['close'], 200).iloc[-1]) if len(df) >= 200 else current_price
        
        # Ichimoku
        ichimoku = self.calculate_ichimoku(df)
        tenkan_val = float(ichimoku['tenkan'].iloc[-1])
        kijun_val = float(ichimoku['kijun'].iloc[-1])
        senkou_a = float(ichimoku['senkou_a'].iloc[-1])
        senkou_b = float(ichimoku['senkou_b'].iloc[-1])
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        # Bollinger Bands
        bb = self.calculate_bollinger_bands(df['close'])
        bb_upper = float(bb['upper'].iloc[-1]) if not pd.isna(bb['upper'].iloc[-1]) else current_price * 1.02
        bb_lower = float(bb['lower'].iloc[-1]) if not pd.isna(bb['lower'].iloc[-1]) else current_price * 0.98
        bb_bandwidth = float(bb['bandwidth'].iloc[-1]) if not pd.isna(bb['bandwidth'].iloc[-1]) else 0
        
        # Gann levels
        gann = self.calculate_gann_levels(df, tf_config)
        
        # Volume analysis
        volume_analysis = self.calculate_volume_analysis(df)
        
        # Signal scoring
        bullish_signals = 0
        bearish_signals = 0
        signal_details = {}
        
        # RSI Signal
        if rsi < RSI_OVERSOLD:
            bullish_signals += 1
            signal_details["RSI"] = f"OVERSOLD ({rsi:.1f})"
        elif rsi > RSI_OVERBOUGHT:
            bearish_signals += 1
            signal_details["RSI"] = f"OVERBOUGHT ({rsi:.1f})"
        else:
            signal_details["RSI"] = f"NEUTRAL ({rsi:.1f})"
        
        # MACD Signal
        if macd_hist_value > 0:
            bullish_signals += 1
            signal_details["MACD"] = "BULLISH"
        else:
            bearish_signals += 1
            signal_details["MACD"] = "BEARISH"
        
        # TK Cross
        if tenkan_val > kijun_val:
            bullish_signals += 1
            signal_details["TK_CROSS"] = "BULLISH"
        else:
            bearish_signals += 1
            signal_details["TK_CROSS"] = "BEARISH"
        
        # Price vs Cloud
        if current_price > cloud_top:
            bullish_signals += 1
            signal_details["CLOUD"] = "ABOVE"
            price_vs_cloud = "ABOVE"
        elif current_price < cloud_bottom:
            bearish_signals += 1
            signal_details["CLOUD"] = "BELOW"
            price_vs_cloud = "BELOW"
        else:
            signal_details["CLOUD"] = "INSIDE"
            price_vs_cloud = "INSIDE"
        
        # Gann 50% Signal
        gann_50 = gann.gann_50_pct
        if current_price > gann_50:
            bullish_signals += 1
            signal_details["GANN_50"] = f"ABOVE (${gann_50:,.0f})"
            price_vs_gann_50 = "ABOVE"
        else:
            bearish_signals += 1
            signal_details["GANN_50"] = f"BELOW (${gann_50:,.0f})"
            price_vs_gann_50 = "BELOW"
        
        # SMA 200 Signal
        if current_price > sma_200:
            bullish_signals += 1
            signal_details["SMA_200"] = f"ABOVE (${sma_200:,.0f})"
        else:
            bearish_signals += 1
            signal_details["SMA_200"] = f"BELOW (${sma_200:,.0f})"
        
        # Determine direction
        if tf_name == "1M":
            # Monthly needs stronger signal
            if bullish_signals > bearish_signals + 2:
                direction = "BULLISH"
            elif bearish_signals > bullish_signals + 2:
                direction = "BEARISH"
            else:
                direction = "NEUTRAL"
        else:
            if bullish_signals > bearish_signals:
                direction = "BULLISH"
            elif bearish_signals > bullish_signals:
                direction = "BEARISH"
            else:
                direction = "NEUTRAL"
        
        # State name (Enneagram)
        state_name = self._determine_state_name(rsi, adx, direction)
        
        return {
            "timeframe": tf_name,
            "direction": direction,
            "state_name": state_name,
            "rsi": safe_round(rsi, 2),
            "macd": safe_round(macd_value, 2),
            "macd_histogram": safe_round(macd_hist_value, 2),
            "adx": safe_round(adx, 2),
            "adx_label": "STRONG" if adx > ADX_STRONG_TREND else "WEAK",
            "atr": safe_round(atr, 2),
            "atr_pct": safe_round((atr / current_price) * 100, 2),
            "sma_50": safe_round(sma_50, 2),
            "sma_200": safe_round(sma_200, 2),
            "price_vs_sma_200": "ABOVE" if current_price > sma_200 else "BELOW",
            "bullish_signals": int(bullish_signals),
            "bearish_signals": int(bearish_signals),
            "signal_score": f"{bullish_signals}B/{bearish_signals}S",
            "signal_details": signal_details,
            "volume_ratio": volume_analysis['volume_ratio'],
            "volume_trend": volume_analysis['volume_trend'],
            "gann_high": gann.high,
            "gann_low": gann.low,
            "gann_50_pct": gann.gann_50_pct,
            "price_vs_gann_50": price_vs_gann_50,
            "gann_range_pct": gann.range_pct,
            "ichimoku": {
                "tenkan": safe_round(tenkan_val, 2),
                "kijun": safe_round(kijun_val, 2),
                "cloud_top": safe_round(cloud_top, 2),
                "cloud_bottom": safe_round(cloud_bottom, 2),
                "tk_cross": signal_details.get("TK_CROSS", "NEUTRAL"),
                "price_vs_cloud": price_vs_cloud
            },
            "bollinger": {
                "upper": safe_round(bb_upper, 2),
                "lower": safe_round(bb_lower, 2),
                "bandwidth": safe_round(bb_bandwidth, 2)
            },
            "gann": {
                "high": gann.high,
                "low": gann.low,
                "high_date": gann.high_date,
                "low_date": gann.low_date,
                "lookback_bars": gann.lookback_bars,
                "levels": gann.levels,
                "gann_50_pct": gann.gann_50_pct
            }
        }
    
    def _empty_timeframe_analysis(self, tf_name: str) -> Dict:
        """Return empty analysis for insufficient data."""
        return {
            "timeframe": tf_name,
            "direction": "NEUTRAL",
            "state_name": "Insufficient Data",
            "rsi": 50.0, "macd": 0.0, "macd_histogram": 0.0,
            "adx": 0.0, "adx_label": "N/A", "atr": 0.0, "atr_pct": 0.0,
            "sma_50": 0.0, "sma_200": 0.0, "price_vs_sma_200": "N/A",
            "bullish_signals": 0, "bearish_signals": 0, "signal_score": "0B/0S",
            "signal_details": {}, "volume_ratio": 1.0, "volume_trend": "N/A",
            "gann_high": 0.0, "gann_low": 0.0, "gann_50_pct": 0.0,
            "price_vs_gann_50": "N/A", "gann_range_pct": 0.0,
            "ichimoku": {"tenkan": 0, "kijun": 0, "cloud_top": 0, "cloud_bottom": 0, "tk_cross": "N/A", "price_vs_cloud": "N/A"},
            "bollinger": {"upper": 0, "lower": 0, "bandwidth": 0},
            "gann": {"high": 0, "low": 0, "high_date": "N/A", "low_date": "N/A", "lookback_bars": 0, "levels": {}, "gann_50_pct": 0}
        }
    
    def _determine_state_name(self, rsi: float, adx: float, direction: str) -> str:
        """Determine Enneagram state name."""
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
        else:
            return "Transition"
    
    # ========================================================
    # MTF CONSENSUS
    # ========================================================
    
    def calculate_mtf_consensus(self, timeframes: Dict, regime: Dict) -> Dict:
        """Calculate multi-timeframe consensus."""
        weighted_bullish = 0.0
        weighted_bearish = 0.0
        alignment_count = 0
        conflicts = []
        
        for tf_name, analysis in timeframes.items():
            weight = TIMEFRAME_CONFIGS[tf_name].weight
            if analysis["direction"] == "BULLISH":
                weighted_bullish += weight * (analysis["bullish_signals"] + 1)
            elif analysis["direction"] == "BEARISH":
                weighted_bearish += weight * (analysis["bearish_signals"] + 1)
        
        # Determine primary direction
        if weighted_bullish > weighted_bearish * 1.2:
            primary_direction = "BULLISH"
        elif weighted_bearish > weighted_bullish * 1.2:
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
        
        # Calculate weighted score
        total = weighted_bullish + weighted_bearish
        weighted_score = int(((weighted_bullish - weighted_bearish) / total) * 100) if total > 0 else 0
        
        # Determine confidence
        if alignment_count >= 4 and abs(weighted_score) > 60:
            confidence_level = "HIGH"
        elif alignment_count >= 3 and abs(weighted_score) > 40:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        # Downgrade if regime override
        if regime.get("override_active", False) and confidence_level == "HIGH":
            confidence_level = "MEDIUM"
        
        # Verdict description
        if confidence_level == "HIGH":
            verdict = f"STRONG {primary_direction}"
        elif confidence_level == "MEDIUM":
            verdict = f"MODERATE {primary_direction}"
        else:
            verdict = f"WEAK {primary_direction}" if primary_direction != "NEUTRAL" else "MIXED/NEUTRAL"
        
        return {
            "primary_direction": primary_direction,
            "weighted_score": int(weighted_score),
            "alignment": f"{alignment_count}/4",
            "alignment_count": int(alignment_count),
            "confidence_level": confidence_level,
            "verdict": verdict,
            "conflicts": conflicts,
            "has_conflicts": len(conflicts) > 0,
            "tf_1m": timeframes.get("1M", {}).get("direction", "N/A"),
            "tf_1w": timeframes.get("1W", {}).get("direction", "N/A"),
            "tf_3d": timeframes.get("3D", {}).get("direction", "N/A"),
            "tf_1d": timeframes.get("1D", {}).get("direction", "N/A")
        }
    
    # ========================================================
    # PRIMARY BIAS (Gann 50% Rule)
    # ========================================================
    
    def determine_primary_bias(self, current_price: float, weekly_gann: GannLevels,
                              daily_atr: float, consensus: Dict) -> Dict:
        """Determine primary bias using Gann 50% rule."""
        gann_50 = weekly_gann.gann_50_pct
        atr_buffer = daily_atr * 0.5
        
        if current_price > gann_50 + atr_buffer:
            gann_bias = "BULLISH"
            gann_position = "ABOVE"
        elif current_price < gann_50 - atr_buffer:
            gann_bias = "BEARISH"
            gann_position = "BELOW"
        else:
            gann_bias = "NEUTRAL"
            gann_position = "AT_50"
        
        consensus_bias = consensus["primary_direction"]
        
        # Gann 50% rule takes precedence when there's a conflict
        if gann_bias != "NEUTRAL" and gann_bias != consensus_bias and consensus_bias != "NEUTRAL":
            primary_bias = gann_bias
            bias_source = "GANN_50_OVERRIDE"
            conflict = True
            note = f"⚠️ Gann 50% rule overrides consensus ({consensus_bias} → {gann_bias})"
        else:
            primary_bias = consensus_bias if consensus_bias != "NEUTRAL" else gann_bias
            bias_source = "CONSENSUS" if consensus_bias == primary_bias else "GANN_50"
            conflict = False
            note = f"Bias aligned: {primary_bias}"
        
        return {
            "primary_bias": primary_bias,
            "bias_source": bias_source,
            "gann_50_level": gann_50,
            "gann_position": gann_position,
            "consensus_direction": consensus_bias,
            "conflict": conflict,
            "note": note,
            "description": f"Price ${current_price:,.0f} is {gann_position} Gann 50% (${gann_50:,.0f})"
        }
    
    # ========================================================
    # TRADE SETUPS
    # ========================================================
    
    def generate_trade_setups(self, current_price: float, timeframes: Dict, consensus: Dict,
                             regime: Dict, capitulation: Dict, weekly_gann: GannLevels) -> List[Dict]:
        """Generate trade setups with R:R >= 1.5 enforcement."""
        setups = []
        
        daily_atr = float(timeframes["1D"]["atr"])
        if daily_atr == 0 or daily_atr is None:
            daily_atr = current_price * 0.02
        
        # Gann levels
        gann_2_8 = float(weekly_gann.levels.get("2_8", current_price * 0.88))
        gann_3_8 = float(weekly_gann.levels.get("3_8", current_price * 0.94))
        gann_5_8 = float(weekly_gann.levels.get("5_8", current_price * 1.06))
        gann_6_8 = float(weekly_gann.levels.get("6_8", current_price * 1.12))
        
        primary_direction = consensus["primary_direction"]
        allows_long = regime.get("allows_long", True)
        allows_short = regime.get("allows_short", True)
        size_cap = regime.get("position_size_cap", 1.0)
        
        # PRIMARY SETUP - LONG
        if primary_direction == "BULLISH" and allows_long:
            entry = current_price
            stop = max(gann_3_8 - daily_atr, current_price * 0.95)
            risk = entry - stop
            
            # Enforce min R:R of 1.5
            min_tp = entry + (risk * MIN_RR_RATIO)
            tp1 = max(gann_5_8, min_tp)
            tp2 = max(gann_6_8, entry + (risk * 2.5))
            tp3 = entry + (risk * 3.5)
            
            rr = (tp1 - entry) / risk if risk > 0 else 0
            
            confidence = "HIGH" if consensus["alignment_count"] >= 3 else "MEDIUM" if consensus["alignment_count"] >= 2 else "LOW"
            base_size = 1.0 if confidence == "HIGH" else 0.5 if confidence == "MEDIUM" else 0.25
            
            setups.append({
                "id": 1,
                "type": "PRIMARY",
                "direction": "LONG",
                "confidence": confidence,
                "entry": safe_round(entry, 2),
                "stop_loss": safe_round(stop, 2),
                "tp1": safe_round(tp1, 2),
                "tp2": safe_round(tp2, 2),
                "tp3": safe_round(tp3, 2),
                "rr_ratio": safe_round(max(rr, MIN_RR_RATIO), 2),
                "risk_pct": safe_round((risk / entry) * 100, 2),
                "position_size": safe_round(min(base_size, size_cap), 2),
                "rationale": f"MTF {consensus['alignment']}, {consensus['verdict']}"
            })
        
        # PRIMARY SETUP - SHORT
        elif primary_direction == "BEARISH" and allows_short:
            entry = current_price
            stop = min(gann_5_8 + daily_atr, current_price * 1.05)
            risk = stop - entry
            
            # Enforce min R:R of 1.5
            min_tp = entry - (risk * MIN_RR_RATIO)
            tp1 = min(gann_3_8, min_tp)
            tp2 = min(gann_2_8, entry - (risk * 2.5))
            tp3 = entry - (risk * 3.5)
            
            rr = (entry - tp1) / risk if risk > 0 else 0
            
            confidence = "HIGH" if consensus["alignment_count"] >= 3 else "MEDIUM" if consensus["alignment_count"] >= 2 else "LOW"
            base_size = 1.0 if confidence == "HIGH" else 0.5 if confidence == "MEDIUM" else 0.25
            
            setups.append({
                "id": 1,
                "type": "PRIMARY",
                "direction": "SHORT",
                "confidence": confidence,
                "entry": safe_round(entry, 2),
                "stop_loss": safe_round(stop, 2),
                "tp1": safe_round(tp1, 2),
                "tp2": safe_round(tp2, 2),
                "tp3": safe_round(tp3, 2),
                "rr_ratio": safe_round(max(rr, MIN_RR_RATIO), 2),
                "risk_pct": safe_round((risk / entry) * 100, 2),
                "position_size": safe_round(min(base_size, size_cap), 2),
                "rationale": f"MTF {consensus['alignment']}, {consensus['verdict']}"
            })
        
        # COUNTER-TREND SETUP (at capitulation)
        if capitulation.get("is_capitulation", False) and allows_long:
            entry = gann_3_8
            stop = gann_2_8 - daily_atr
            risk = entry - stop
            tp1 = current_price
            tp2 = gann_5_8
            rr = (tp1 - entry) / risk if risk > 0 else 0
            
            setups.append({
                "id": 2,
                "type": "COUNTER_TREND",
                "direction": "LONG",
                "confidence": "LOW",
                "entry": safe_round(entry, 2),
                "stop_loss": safe_round(stop, 2),
                "tp1": safe_round(tp1, 2),
                "tp2": safe_round(tp2, 2),
                "tp3": None,
                "rr_ratio": safe_round(max(rr, MIN_RR_RATIO), 2),
                "risk_pct": safe_round((risk / entry) * 100, 2),
                "position_size": 0.25,
                "rationale": f"Capitulation {capitulation.get('status', 'POTENTIAL')}"
            })
        
        # WAIT SETUP (if no clear direction)
        if not setups or consensus["alignment_count"] < 2:
            setups.append({
                "id": len(setups) + 1,
                "type": "WAIT",
                "direction": "FLAT",
                "confidence": "NONE",
                "entry": safe_round(current_price, 2),
                "stop_loss": None,
                "tp1": None,
                "tp2": None,
                "tp3": None,
                "rr_ratio": 0,
                "risk_pct": 0,
                "position_size": 0,
                "rationale": "Insufficient alignment or conflicting signals"
            })
        
        # Sort by confidence
        confidence_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}
        setups.sort(key=lambda x: confidence_order.get(x["confidence"], 4))
        
        return setups
    
    # ========================================================
    # TIME FORECAST
    # ========================================================
    
    def calculate_time_forecast(self, df_daily: pd.DataFrame, current_price: float,
                                atr: float, weekly_gann: GannLevels) -> Dict:
        """Calculate time-based pivot forecast."""
        lookback = min(365, len(df_daily))
        recent_df = df_daily.tail(lookback)
        
        major_high = float(recent_df['high'].max())
        major_low = float(recent_df['low'].min())
        major_high_idx = recent_df['high'].idxmax()
        major_low_idx = recent_df['low'].idxmin()
        
        ts_col = 'date' if 'date' in df_daily.columns else 'timestamp'
        
        try:
            if major_high_idx > major_low_idx:
                reference_type = "HIGH"
                expected_pivot = "LOW"
                ref_idx = major_high_idx
                ref_price = major_high
            else:
                reference_type = "LOW"
                expected_pivot = "HIGH"
                ref_idx = major_low_idx
                ref_price = major_low
            
            ref_date = pd.to_datetime(df_daily.loc[ref_idx, ts_col])
            if ref_date.tzinfo is not None:
                ref_date = ref_date.replace(tzinfo=None)
            
            ref_date_str = ref_date.strftime("%Y-%m-%d")
            days_since = int((datetime.now() - ref_date).days)
        except:
            reference_type = "HIGH"
            expected_pivot = "LOW"
            days_since = 90
            ref_date_str = "N/A"
            ref_price = current_price
        
        # Find next cycle
        next_pivot_date = None
        days_to_cycle = 90
        confidence = 0.40
        cycle_used = 90
        
        for cycle in GANN_CYCLES:
            if cycle > days_since:
                days_to_cycle = cycle - days_since
                next_pivot_date = (datetime.now() + timedelta(days=days_to_cycle)).strftime("%Y-%m-%d")
                
                # Higher confidence for major cycles
                if cycle in [180, 360]:
                    confidence = 0.65
                elif cycle in [90, 144]:
                    confidence = 0.55
                else:
                    confidence = 0.45
                
                cycle_used = cycle
                break
        
        if next_pivot_date is None:
            days_to_cycle = 90
            next_pivot_date = (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d")
            cycle_used = 90
        
        # Price projections
        gann_3_8 = float(weekly_gann.levels.get("3_8", current_price * 0.9))
        gann_5_8 = float(weekly_gann.levels.get("5_8", current_price * 1.1))
        atr_proj = float(atr * np.sqrt(days_to_cycle)) if atr > 0 else current_price * 0.10
        
        probable_low = max(current_price - atr_proj, gann_3_8 * 0.95)
        probable_high = min(current_price + atr_proj, gann_5_8 * 1.05)
        
        return {
            "next_pivot_date": next_pivot_date,
            "days_to_pivot": days_to_cycle,
            "pivot_type": expected_pivot,
            "confidence": safe_round(confidence, 2),
            "confidence_pct": f"{int(confidence * 100)}%",
            "confidence_level": "HIGH" if confidence >= 0.65 else "MEDIUM" if confidence >= 0.50 else "LOW",
            "probable_price_low": safe_round(probable_low, 2),
            "probable_price_high": safe_round(probable_high, 2),
            "price_range": f"${probable_low:,.0f} - ${probable_high:,.0f}",
            "cycle_origin": {
                "reference_type": reference_type,
                "reference_date": ref_date_str,
                "reference_price": safe_round(ref_price, 2),
                "days_since": days_since,
                "cycle_length": cycle_used,
                "cycle_name": f"{cycle_used}D Gann Cycle"
            }
        }
    
    # ========================================================
    # SUPPORT / RESISTANCE
    # ========================================================
    
    def calculate_support_resistance(self, current_price: float, timeframes: Dict, 
                                     weekly_gann: GannLevels, sq9: Dict) -> Dict:
        """Calculate key support and resistance levels."""
        
        # Gann-based levels
        gann_supports = [
            {"level": weekly_gann.levels.get("3_8"), "type": "Gann 3/8", "strength": "STRONG"},
            {"level": weekly_gann.levels.get("2_8"), "type": "Gann 2/8", "strength": "STRONG"},
            {"level": weekly_gann.low, "type": "Gann Low", "strength": "MAJOR"},
        ]
        
        gann_resistances = [
            {"level": weekly_gann.levels.get("5_8"), "type": "Gann 5/8", "strength": "STRONG"},
            {"level": weekly_gann.levels.get("6_8"), "type": "Gann 6/8", "strength": "STRONG"},
            {"level": weekly_gann.high, "type": "Gann High", "strength": "MAJOR"},
        ]
        
        # Add Ichimoku cloud levels
        weekly_ichi = timeframes.get("1W", {}).get("ichimoku", {})
        if weekly_ichi.get("cloud_bottom"):
            gann_supports.append({
                "level": weekly_ichi["cloud_bottom"],
                "type": "Weekly Cloud Bottom",
                "strength": "MEDIUM"
            })
        if weekly_ichi.get("cloud_top"):
            gann_resistances.append({
                "level": weekly_ichi["cloud_top"],
                "type": "Weekly Cloud Top",
                "strength": "MEDIUM"
            })
        
        # Add SQ9 levels
        if sq9.get("nearest_support"):
            gann_supports.append({
                "level": sq9["nearest_support"]["price"],
                "type": f"SQ9 {sq9['nearest_support']['angle']}°",
                "strength": "MEDIUM"
            })
        if sq9.get("nearest_resistance"):
            gann_resistances.append({
                "level": sq9["nearest_resistance"]["price"],
                "type": f"SQ9 {sq9['nearest_resistance']['angle']}°",
                "strength": "MEDIUM"
            })
        
        # Filter and sort
        supports = [s for s in gann_supports if s["level"] and s["level"] < current_price]
        resistances = [r for r in gann_resistances if r["level"] and r["level"] > current_price]
        
        supports = sorted(supports, key=lambda x: x["level"], reverse=True)[:5]
        resistances = sorted(resistances, key=lambda x: x["level"])[:5]
        
        return {
            "supports": supports,
            "resistances": resistances,
            "nearest_support": supports[0] if supports else None,
            "nearest_resistance": resistances[0] if resistances else None
        }
    
    # ========================================================
    # MAIN MTF SIGNAL GENERATOR
    # ========================================================
    
    def generate_mtf_signal(self, symbol: str = "BTCUSDT") -> Dict:
        """Main MTF signal generator - COMPLETE v5.0.8."""
        try:
            logger.info(f"[MTF] Generating signal for {symbol}")
            
            # Fetch data
            df_1d = self.fetch_real_binance_data(use_cache=True, symbol=symbol)
            if df_1d is None or len(df_1d) < 100:
                raise ValueError(f"Insufficient data: {len(df_1d) if df_1d is not None else 0} candles")
            
            logger.info(f"[MTF] {len(df_1d)} candles fetched")
            
            # Resample to higher timeframes
            df_3d = self.resample_ohlcv(df_1d, "3D")
            df_1w = self.resample_ohlcv(df_1d, "1W")
            df_1m = self.resample_ohlcv(df_1d, "1M")
            
            current_price = float(df_1d['close'].iloc[-1])
            ts_col = 'date' if 'date' in df_1d.columns else 'timestamp'
            signal_date = str(df_1d[ts_col].iloc[-1])[:10]
            
            # Analyze all timeframes
            timeframes = {
                "1D": self.analyze_timeframe(df_1d, "1D"),
                "3D": self.analyze_timeframe(df_3d, "3D"),
                "1W": self.analyze_timeframe(df_1w, "1W"),
                "1M": self.analyze_timeframe(df_1m, "1M"),
            }
            
            # Weekly analysis for capitulation/regime
            weekly_rsi = float(timeframes["1W"]["rsi"])
            weekly_adx = float(timeframes["1W"]["adx"])
            weekly_gann = self.calculate_gann_levels(df_1w, TIMEFRAME_CONFIGS["1W"])
            
            # Detect capitulation
            capitulation = self.detect_capitulation(df_1w, df_1d, weekly_rsi, weekly_gann, current_price)
            
            # Determine regime
            regime = self.determine_regime(df_1d, weekly_rsi, weekly_adx, capitulation, current_price)
            
            # Calculate consensus
            consensus = self.calculate_mtf_consensus(timeframes, regime)
            
            # Primary bias (Gann 50% rule)
            daily_atr = float(timeframes["1D"]["atr"])
            primary_bias = self.determine_primary_bias(current_price, weekly_gann, daily_atr, consensus)
            
            # Time forecast
            time_forecast = self.calculate_time_forecast(df_1d, current_price, daily_atr, weekly_gann)
            
            # SQ9 levels (filtered 2-5%)
            sq9 = self.calculate_sq9_levels(current_price, 2.0, 5.0)
            
            # Support/Resistance
            sr_levels = self.calculate_support_resistance(current_price, timeframes, weekly_gann, sq9)
            
            # Generate trade setups
            trade_setups = self.generate_trade_setups(
                current_price, timeframes, consensus, regime, capitulation, weekly_gann
            )
            
            # Invalidation price
            if primary_bias["primary_bias"] == "BULLISH":
                invalidation_price = weekly_gann.levels.get("3_8", current_price * 0.92)
                invalidation_desc = f"Bullish bias invalidated below ${invalidation_price:,.0f}"
            elif primary_bias["primary_bias"] == "BEARISH":
                invalidation_price = weekly_gann.levels.get("5_8", current_price * 1.08)
                invalidation_desc = f"Bearish bias invalidated above ${invalidation_price:,.0f}"
            else:
                invalidation_price = None
                invalidation_desc = "No clear bias to invalidate"
            
            # Price levels for each timeframe
            price_levels = {
                "monthly": {
                    "high": timeframes["1M"]["gann_high"],
                    "low": timeframes["1M"]["gann_low"],
                    "gann_50": timeframes["1M"]["gann_50_pct"],
                    "range_pct": timeframes["1M"]["gann_range_pct"]
                },
                "weekly": {
                    "high": weekly_gann.high,
                    "low": weekly_gann.low,
                    "gann_50": weekly_gann.gann_50_pct,
                    "gann_3_8": weekly_gann.levels.get("3_8"),
                    "gann_5_8": weekly_gann.levels.get("5_8"),
                    "cloud_top": timeframes["1W"]["ichimoku"]["cloud_top"],
                    "cloud_bottom": timeframes["1W"]["ichimoku"]["cloud_bottom"]
                },
                "daily": {
                    "high": timeframes["1D"]["gann_high"],
                    "low": timeframes["1D"]["gann_low"],
                    "gann_50": timeframes["1D"]["gann_50_pct"],
                    "sma_200": timeframes["1D"]["sma_200"],
                    "bb_upper": timeframes["1D"]["bollinger"]["upper"],
                    "bb_lower": timeframes["1D"]["bollinger"]["lower"]
                }
            }
            
            # Enneagram states
            enneagram = {
                "1M": timeframes["1M"]["state_name"],
                "1W": timeframes["1W"]["state_name"],
                "3D": timeframes["3D"]["state_name"],
                "1D": timeframes["1D"]["state_name"],
                "dominant": timeframes["1W"]["state_name"],
                "phase": "Accumulation" if primary_bias["primary_bias"] == "BULLISH" else "Distribution" if primary_bias["primary_bias"] == "BEARISH" else "Transition",
                "arrow": "↑" if primary_bias["primary_bias"] == "BULLISH" else "↓" if primary_bias["primary_bias"] == "BEARISH" else "→"
            }
            
            # Gann interpretation
            gann_interpretation = {
                "weekly_50_pct": weekly_gann.gann_50_pct,
                "current_position": primary_bias["gann_position"],
                "description": primary_bias["description"],
                "primary_bias": primary_bias["primary_bias"],
                "rule": "Above 50% = Bulls control | Below 50% = Bears control"
            }
            
            logger.info(f"[MTF] Signal generated: {consensus['primary_direction']} ({consensus['confidence_level']})")
            
            # Build complete result
            result = {
                "status": "success",
                "symbol": symbol,
                "current_price": safe_round(current_price, 2),
                "signal_date": signal_date,
                "timestamp": datetime.now().isoformat(),
                "version": self.VERSION,
                
                # Core sections
                "regime": regime,
                "timeframes": timeframes,
                "consensus": consensus,
                "primary_bias": primary_bias,
                "price_levels": price_levels,
                "capitulation": capitulation,
                "time_forecast": time_forecast,
                "trade_setups": trade_setups,
                "sq9_levels": sq9,
                "support_resistance": sr_levels,
                
                # Invalidation
                "invalidation": {
                    "price": safe_round(invalidation_price, 2) if invalidation_price else None,
                    "description": invalidation_desc
                },
                
                # Enneagram
                "enneagram": enneagram,
                
                # Gann interpretation
                "gann_interpretation": gann_interpretation,
                
                # Signal summary (for quick access)
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
        """Legacy method - calls MTF signal."""
        return self.generate_mtf_signal()
