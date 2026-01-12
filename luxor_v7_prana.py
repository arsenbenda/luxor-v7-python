# ============================================================
# LUXOR V7 PRANA - GANN EGYPT-INDIA UNIFIED SYSTEM v5.0.7
# Fixed: All N/A fields, Gann exposure, complete Telegram mapping
# ============================================================

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
import logging
import ccxt

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

TIMEFRAME_CONFIGS = {
    "1M": TimeframeConfig("1M", 0.35, 24, 12),
    "1W": TimeframeConfig("1W", 0.30, 52, 26),
    "3D": TimeframeConfig("3D", 0.20, 120, 60),
    "1D": TimeframeConfig("1D", 0.15, 252, 100),
}

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

GANN_CYCLES = [30, 45, 60, 90, 120, 144, 180, 270, 360]

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
    gann_50_pct: float  # Explicit 50% level

@dataclass
class IchimokuSignals:
    tk_cross: str
    price_vs_cloud: str
    cloud_color: str
    kijun_value: float
    tenkan_value: float
    cloud_top: float
    cloud_bottom: float
    kijun_flat: bool

@dataclass
class CapitulationAnalysis:
    is_capitulation: bool
    status: str
    confidence: float
    criteria_met: List[str]
    criteria_missing: List[str]
    rsi_extreme: bool
    volume_spike: bool
    gann_support_test: bool
    bullish_divergence: bool
    weekly_rsi: float
    volume_ratio: float

@dataclass
class RegimeAnalysis:
    regime: Regime
    regime_strength: float
    override_active: bool
    override_reason: Optional[str]
    allows_short: bool
    allows_long: bool
    position_size_cap: float
    warnings: List[str]

# ============================================================
# MAIN CLASS
# ============================================================

class LuxorV7PranaSystem:
    """LUXOR V7 PRANA - GANN EGYPT-INDIA UNIFIED SYSTEM v5.0.7"""
    
    CACHE = {
        'df': None,
        'last_fetch': None,
        'cache_duration': 3600
    }
    
    VERSION = "5.0.7"
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.sidereal_epoch = pd.Timestamp('1900-01-01')
        logger.info(f"[INIT] LuxorV7PranaSystem v{self.VERSION} initialized")
    
    # ========================================================
    # DATA FETCHING
    # ========================================================
    
    def fetch_ohlcv_ccxt(self, symbol: str = "BTC/USDT", interval: str = "1d", limit: int = 500) -> pd.DataFrame:
        """Fetch OHLCV using CCXT with multi-exchange fallback."""
        exchanges_to_try = [
            ('kucoin', 'BTC/USDT'),
            ('bybit', 'BTC/USDT'),
            ('okx', 'BTC/USDT'),
            ('kraken', 'BTC/USD'),
            ('bitfinex', 'tBTCUSD'),
            ('huobi', 'BTC/USDT'),
            ('gate', 'BTC/USDT'),
        ]
        
        if 'ETH' in symbol.upper():
            exchanges_to_try = [
                ('kucoin', 'ETH/USDT'),
                ('bybit', 'ETH/USDT'),
                ('okx', 'ETH/USDT'),
                ('kraken', 'ETH/USD'),
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
                    logger.info(f"[DATA] Fetched {len(df)} candles from {exchange_id}")
                    return df
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[DATA] {exchange_id} failed: {str(e)[:80]}")
                continue
        
        raise Exception(f"All exchanges failed. Last error: {last_error}")
    
    def fetch_real_binance_data(self, use_cache=True, symbol: str = "BTCUSDT"):
        """Fetch data with caching."""
        try:
            if use_cache and self.CACHE['df'] is not None and self.CACHE['last_fetch'] is not None:
                cache_age = (datetime.now() - self.CACHE['last_fetch']).total_seconds()
                if cache_age < self.CACHE['cache_duration']:
                    logger.info(f"[CACHE] Using cached data (age: {cache_age:.0f}s)")
                    return self.CACHE['df'].copy()
            
            if '/' not in symbol:
                ccxt_symbol = symbol[:-4] + '/USDT' if symbol.endswith('USDT') else symbol
            else:
                ccxt_symbol = symbol
            
            df = self.fetch_ohlcv_ccxt(ccxt_symbol, "1d", 500)
            
            self.CACHE['df'] = df.copy()
            self.CACHE['last_fetch'] = datetime.now()
            
            return df
            
        except Exception as e:
            logger.error(f"[ERROR] fetch_real_binance_data: {e}")
            return None
    
    def resample_ohlcv(self, df_1d: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """Resample daily to higher timeframes."""
        df = df_1d.copy()
        
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
        elif 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        
        # Use 'M' not 'ME' for pandas compatibility
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
    # INDICATORS
    # ========================================================
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
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
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        atr = atr.replace(0, 1)
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / atr)
        di_sum = (plus_di + minus_di).replace(0, 1)
        dx = 100 * (plus_di - minus_di).abs() / di_sum
        return dx.ewm(alpha=1/period, min_periods=period).mean().fillna(0)
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.fillna(tr.mean() if tr.mean() > 0 else 1.0)
    
    def calculate_ichimoku(self, df: pd.DataFrame) -> Dict:
        high, low, close = df['high'], df['low'], df['close']
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        return {
            'tenkan': tenkan.fillna(close),
            'kijun': kijun.fillna(close),
            'senkou_a': senkou_a.fillna(close),
            'senkou_b': senkou_b.fillna(close)
        }
    
    def calculate_gann_levels(self, df: pd.DataFrame, tf_config: TimeframeConfig) -> GannLevels:
        """Calculate Gann levels with explicit 50% exposure."""
        lookback = min(tf_config.gann_lookback, len(df))
        recent_df = df.tail(lookback)
        
        high = float(recent_df['high'].max())
        low = float(recent_df['low'].min())
        high_idx = recent_df['high'].idxmax()
        low_idx = recent_df['low'].idxmin()
        
        ts_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        high_date = str(df.loc[high_idx, ts_col])[:10] if ts_col in df.columns else str(high_idx)[:10]
        low_date = str(df.loc[low_idx, ts_col])[:10] if ts_col in df.columns else str(low_idx)[:10]
        
        range_value = high - low
        current_price = float(df['close'].iloc[-1])
        range_pct = (range_value / current_price) * 100 if current_price > 0 else 0
        
        # Calculate all Gann levels (0/8 through 8/8)
        levels = {}
        for i in range(9):
            level_value = low + (range_value * i / 8)
            levels[f"{i}_8"] = round(float(level_value), 2)
        
        # Explicit 50% level (4/8)
        gann_50_pct = round(float(low + (range_value * 0.5)), 2)
        
        return GannLevels(
            high=round(high, 2),
            low=round(low, 2),
            high_date=high_date,
            low_date=low_date,
            range_value=round(float(range_value), 2),
            range_pct=round(float(range_pct), 2),
            lookback_bars=int(lookback),
            levels=levels,
            gann_50_pct=gann_50_pct
        )
    
    # ========================================================
    # DIVERGENCE & CAPITULATION
    # ========================================================
    
    def detect_divergence(self, df: pd.DataFrame, rsi: pd.Series, lookback: int = 14) -> Dict:
        if len(df) < lookback * 2:
            return {"bullish_divergence": False, "bearish_divergence": False}
        
        recent_prices = df['close'].tail(lookback * 2)
        recent_rsi = rsi.tail(lookback * 2)
        
        price_min_1 = float(recent_prices.iloc[:lookback].min())
        price_min_2 = float(recent_prices.iloc[lookback:].min())
        rsi_min_1 = float(recent_rsi.iloc[:lookback].min())
        rsi_min_2 = float(recent_rsi.iloc[lookback:].min())
        
        bullish_div = bool((price_min_2 < price_min_1) and (rsi_min_2 > rsi_min_1))
        
        price_max_1 = float(recent_prices.iloc[:lookback].max())
        price_max_2 = float(recent_prices.iloc[lookback:].max())
        rsi_max_1 = float(recent_rsi.iloc[:lookback].max())
        rsi_max_2 = float(recent_rsi.iloc[lookback:].max())
        
        bearish_div = bool((price_max_2 > price_max_1) and (rsi_max_2 < rsi_max_1))
        
        return {"bullish_divergence": bullish_div, "bearish_divergence": bearish_div}
    
    def detect_capitulation(self, df_weekly: pd.DataFrame, df_daily: pd.DataFrame,
                           weekly_rsi: float, weekly_gann: GannLevels, current_price: float) -> CapitulationAnalysis:
        criteria_met, criteria_missing = [], []
        
        rsi_extreme = bool(weekly_rsi < CAPITULATION_RSI_THRESHOLD)
        if rsi_extreme:
            criteria_met.append(f"RSI_EXTREME: {weekly_rsi:.1f}")
        else:
            criteria_missing.append(f"RSI not extreme: {weekly_rsi:.1f}")
        
        volume_ratio = 1.0
        if len(df_daily) >= 20 and 'volume' in df_daily.columns:
            avg_vol = float(df_daily['volume'].tail(20).mean())
            if avg_vol > 0:
                volume_ratio = float(df_daily['volume'].iloc[-1] / avg_vol)
        
        volume_spike = bool(volume_ratio >= VOLUME_SPIKE_THRESHOLD)
        if volume_spike:
            criteria_met.append(f"VOLUME_SPIKE: {volume_ratio:.1f}x")
        else:
            criteria_missing.append(f"No volume spike: {volume_ratio:.1f}x")
        
        gann_3_8 = float(weekly_gann.levels.get("3_8", weekly_gann.low))
        gann_support_test = bool(abs(current_price - gann_3_8) / current_price < 0.05)
        if gann_support_test:
            criteria_met.append(f"GANN_SUPPORT: Near {gann_3_8:,.0f}")
        else:
            criteria_missing.append(f"Not near Gann support: {gann_3_8:,.0f}")
        
        weekly_rsi_series = self.calculate_rsi(df_weekly['close'])
        divergence = self.detect_divergence(df_weekly, weekly_rsi_series, 8)
        bullish_divergence = bool(divergence.get("bullish_divergence", False))
        if bullish_divergence:
            criteria_met.append("BULLISH_DIVERGENCE")
        else:
            criteria_missing.append("No divergence")
        
        num_criteria = len(criteria_met)
        if num_criteria >= 4:
            status, confidence = "CONFIRMED", 0.90
        elif num_criteria >= 3:
            status, confidence = "POTENTIAL", 0.70
        elif num_criteria >= 2:
            status, confidence = "DEVELOPING", 0.50
        else:
            status, confidence = "NONE", 0.20
        
        return CapitulationAnalysis(
            is_capitulation=bool(num_criteria >= 3),
            status=status,
            confidence=float(confidence),
            criteria_met=criteria_met,
            criteria_missing=criteria_missing,
            rsi_extreme=rsi_extreme,
            volume_spike=volume_spike,
            gann_support_test=gann_support_test,
            bullish_divergence=bullish_divergence,
            weekly_rsi=float(weekly_rsi),
            volume_ratio=float(volume_ratio)
        )
    
    # ========================================================
    # REGIME
    # ========================================================
    
    def determine_regime(self, df_daily: pd.DataFrame, weekly_rsi: float, weekly_adx: float,
                        capitulation: CapitulationAnalysis, current_price: float) -> RegimeAnalysis:
        warnings_list = []
        override_active = False
        override_reason = None
        
        sma_200 = float(df_daily['close'].rolling(200).mean().iloc[-1]) if len(df_daily) >= 200 else current_price
        price_vs_sma = "above" if current_price > sma_200 else "below"
        trend_strength = "strong" if weekly_adx > ADX_STRONG_TREND else "weak"
        
        if price_vs_sma == "above" and trend_strength == "strong":
            base_regime = Regime.TRENDING_BULL
        elif price_vs_sma == "below" and trend_strength == "strong":
            base_regime = Regime.TRENDING_BEAR
        else:
            base_regime = Regime.RANGING
        
        if capitulation.is_capitulation:
            override_active = True
            override_reason = f"Capitulation {capitulation.status}"
            final_regime = Regime.CAPITULATION
            allows_short = False
            allows_long = True
            position_size_cap = 0.25 if capitulation.status == "POTENTIAL" else 0.50
            warnings_list.append("CAPITULATION OVERRIDE: Shorts blocked")
        elif weekly_rsi > EUPHORIA_RSI_THRESHOLD:
            override_active = True
            override_reason = f"Euphoria: RSI {weekly_rsi:.1f}"
            final_regime = Regime.EUPHORIA
            allows_short = True
            allows_long = False
            position_size_cap = 0.25
            warnings_list.append("EUPHORIA OVERRIDE: Longs blocked")
        else:
            final_regime = base_regime
            allows_short = True
            allows_long = True
            position_size_cap = 1.0
        
        regime_strength = float(min(weekly_adx / 100, 1.0)) if final_regime in [Regime.TRENDING_BULL, Regime.TRENDING_BEAR] else 0.5
        
        return RegimeAnalysis(
            regime=final_regime,
            regime_strength=regime_strength,
            override_active=bool(override_active),
            override_reason=override_reason,
            allows_short=bool(allows_short),
            allows_long=bool(allows_long),
            position_size_cap=float(position_size_cap),
            warnings=warnings_list
        )
    
    # ========================================================
    # TIMEFRAME ANALYSIS (FIXED - Exposes Gann)
    # ========================================================
    
    def analyze_timeframe(self, df: pd.DataFrame, tf_name: str) -> Dict:
        tf_config = TIMEFRAME_CONFIGS[tf_name]
        
        if len(df) < tf_config.min_bars:
            return {
                "timeframe": tf_name,
                "direction": "NEUTRAL",
                "state_name": "Insufficient Data",
                "rsi": 50.0,
                "adx": 0.0,
                "atr": 0.0,
                "bullish_signals": 0,
                "bearish_signals": 0,
                "signal_details": {},
                "volume_ratio": 1.0,
                "gann_high": 0.0,
                "gann_low": 0.0,
                "gann_50_pct": 0.0,
                "price_vs_gann_50": "N/A",
                "ichimoku": {
                    "tk_cross": "NEUTRAL",
                    "price_vs_cloud": "INSIDE",
                    "kijun": 0.0,
                    "cloud_top": 0.0,
                    "cloud_bottom": 0.0
                },
                "gann": {"high": 0.0, "low": 0.0, "levels": {}, "gann_50_pct": 0.0}
            }
        
        current_price = float(df['close'].iloc[-1])
        
        # Calculate indicators
        rsi_series = self.calculate_rsi(df['close'])
        rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
        
        macd, signal, hist = self.calculate_macd(df['close'])
        macd_hist = float(hist.iloc[-1]) if not pd.isna(hist.iloc[-1]) else 0.0
        
        adx_series = self.calculate_adx(df)
        adx = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0.0
        
        atr_series = self.calculate_atr(df)
        atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0
        
        ichimoku = self.calculate_ichimoku(df)
        gann = self.calculate_gann_levels(df, tf_config)
        
        # Signal scoring
        bullish, bearish = 0, 0
        signal_details = {}
        
        # RSI Signal
        if rsi < RSI_OVERSOLD:
            bullish += 1
            signal_details["RSI"] = f"OVERSOLD ({rsi:.1f})"
        elif rsi > RSI_OVERBOUGHT:
            bearish += 1
            signal_details["RSI"] = f"OVERBOUGHT ({rsi:.1f})"
        else:
            signal_details["RSI"] = f"NEUTRAL ({rsi:.1f})"
        
        # MACD Signal
        if macd_hist > 0:
            bullish += 1
            signal_details["MACD"] = "BULLISH"
        else:
            bearish += 1
            signal_details["MACD"] = "BEARISH"
        
        # Ichimoku TK Cross
        tenkan_val = float(ichimoku['tenkan'].iloc[-1])
        kijun_val = float(ichimoku['kijun'].iloc[-1])
        if tenkan_val > kijun_val:
            bullish += 1
            signal_details["TK_CROSS"] = "BULLISH"
        else:
            bearish += 1
            signal_details["TK_CROSS"] = "BEARISH"
        
        # Ichimoku Cloud
        senkou_a = float(ichimoku['senkou_a'].iloc[-1])
        senkou_b = float(ichimoku['senkou_b'].iloc[-1])
        cloud_top = float(max(senkou_a, senkou_b))
        cloud_bottom = float(min(senkou_a, senkou_b))
        
        if current_price > cloud_top:
            bullish += 1
            signal_details["CLOUD"] = "ABOVE"
        elif current_price < cloud_bottom:
            bearish += 1
            signal_details["CLOUD"] = "BELOW"
        else:
            signal_details["CLOUD"] = "INSIDE"
        
        # Gann 50% Signal
        gann_50 = gann.gann_50_pct
        if current_price > gann_50:
            bullish += 1
            signal_details["GANN_50"] = f"ABOVE (${gann_50:,.0f})"
            price_vs_gann_50 = "ABOVE"
        else:
            bearish += 1
            signal_details["GANN_50"] = f"BELOW (${gann_50:,.0f})"
            price_vs_gann_50 = "BELOW"
        
        # Determine direction
        if tf_name == "1M":
            # Monthly needs stronger signal
            if bullish > bearish + 1:
                direction = "BULLISH"
            elif bearish > bullish + 1:
                direction = "BEARISH"
            else:
                direction = "NEUTRAL"
        else:
            if bullish > bearish:
                direction = "BULLISH"
            elif bearish > bullish:
                direction = "BEARISH"
            else:
                direction = "NEUTRAL"
        
        # State name
        if rsi < 25:
            state_name = "Capitulation"
        elif rsi < 35:
            state_name = "Fear"
        elif rsi > 75:
            state_name = "Euphoria"
        elif rsi > 65:
            state_name = "Greed"
        elif adx > 50:
            state_name = "Expansion" if direction == "BULLISH" else "Contraction"
        elif adx < 20:
            state_name = "Consolidation"
        else:
            state_name = "Transition"
        
        # Volume ratio
        volume_ratio = 1.0
        if 'volume' in df.columns and len(df) >= 20:
            avg_vol = float(df['volume'].tail(20).mean())
            if avg_vol > 0:
                volume_ratio = float(df['volume'].iloc[-1] / avg_vol)
        
        return {
            "timeframe": tf_name,
            "direction": direction,
            "state_name": state_name,
            "rsi": round(rsi, 2),
            "adx": round(adx, 2),
            "atr": round(atr, 2),
            "bullish_signals": int(bullish),
            "bearish_signals": int(bearish),
            "signal_details": signal_details,
            "volume_ratio": round(volume_ratio, 2),
            # EXPLICIT GANN EXPOSURE
            "gann_high": gann.high,
            "gann_low": gann.low,
            "gann_50_pct": gann.gann_50_pct,
            "price_vs_gann_50": price_vs_gann_50,
            "ichimoku": {
                "tk_cross": signal_details.get("TK_CROSS", "NEUTRAL"),
                "price_vs_cloud": signal_details.get("CLOUD", "INSIDE"),
                "kijun": round(kijun_val, 2),
                "cloud_top": round(cloud_top, 2),
                "cloud_bottom": round(cloud_bottom, 2),
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
    
    # ========================================================
    # CONSENSUS
    # ========================================================
    
    def calculate_mtf_consensus(self, timeframes: Dict, regime: RegimeAnalysis) -> Dict:
        weighted_bullish, weighted_bearish = 0.0, 0.0
        alignment_count = 0
        conflicts = []
        
        for tf_name, analysis in timeframes.items():
            weight = TIMEFRAME_CONFIGS[tf_name].weight
            if analysis["direction"] == "BULLISH":
                weighted_bullish += weight * analysis["bullish_signals"]
            elif analysis["direction"] == "BEARISH":
                weighted_bearish += weight * analysis["bearish_signals"]
        
        if weighted_bullish > weighted_bearish * 1.2:
            primary_direction = "BULLISH"
        elif weighted_bearish > weighted_bullish * 1.2:
            primary_direction = "BEARISH"
        else:
            primary_direction = "NEUTRAL"
        
        for tf_name, analysis in timeframes.items():
            if analysis["direction"] == primary_direction:
                alignment_count += 1
            elif analysis["direction"] != "NEUTRAL" and primary_direction != "NEUTRAL":
                conflicts.append({"timeframe": tf_name, "actual": analysis["direction"], "expected": primary_direction})
        
        total = weighted_bullish + weighted_bearish
        weighted_score = int(((weighted_bullish - weighted_bearish) / total) * 100) if total > 0 else 0
        
        if alignment_count >= 4 and abs(weighted_score) > 60:
            confidence_level = "HIGH"
        elif alignment_count >= 3 and abs(weighted_score) > 40:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        if regime.override_active and confidence_level == "HIGH":
            confidence_level = "MEDIUM"
        
        return {
            "primary_direction": primary_direction,
            "weighted_score": int(weighted_score),
            "alignment": f"{alignment_count}/4",
            "alignment_count": int(alignment_count),
            "confidence_level": confidence_level,
            "conflicts": conflicts,
            "tf_1m": timeframes.get("1M", {}).get("direction", "N/A"),
            "tf_1w": timeframes.get("1W", {}).get("direction", "N/A"),
            "tf_3d": timeframes.get("3D", {}).get("direction", "N/A"),
            "tf_1d": timeframes.get("1D", {}).get("direction", "N/A"),
        }
    
    # ========================================================
    # TRADE SETUPS (FIXED R:R Enforcement)
    # ========================================================
    
    def generate_trade_setups(self, current_price: float, timeframes: Dict, consensus: Dict,
                             regime: RegimeAnalysis, capitulation: CapitulationAnalysis,
                             weekly_gann: GannLevels) -> List[Dict]:
        setups = []
        
        daily_atr = float(timeframes["1D"]["atr"])
        if daily_atr == 0:
            daily_atr = current_price * 0.02  # Fallback 2%
        
        gann_3_8 = float(weekly_gann.levels.get("3_8", current_price * 0.95))
        gann_5_8 = float(weekly_gann.levels.get("5_8", current_price * 1.05))
        gann_2_8 = float(weekly_gann.levels.get("2_8", current_price * 0.90))
        gann_6_8 = float(weekly_gann.levels.get("6_8", current_price * 1.10))
        
        primary_direction = consensus["primary_direction"]
        
        if primary_direction == "BULLISH" and regime.allows_long:
            entry = current_price
            stop = max(gann_3_8 - daily_atr, current_price * 0.95)
            risk = entry - stop
            
            # Enforce min R:R of 1.5
            min_tp = entry + (risk * MIN_RR_RATIO)
            tp1 = max(gann_5_8, min_tp)
            tp2 = gann_6_8
            
            rr = (tp1 - entry) / risk if risk > 0 else 0
            
            confidence = "HIGH" if consensus["alignment_count"] >= 3 else "MEDIUM" if consensus["alignment_count"] >= 2 else "LOW"
            size = 1.0 if confidence == "HIGH" else 0.5 if confidence == "MEDIUM" else 0.25
            
            setups.append({
                "id": 1,
                "type": "PRIMARY",
                "direction": "LONG",
                "confidence": confidence,
                "entry": round(entry, 2),
                "stop_loss": round(stop, 2),
                "tp1": round(tp1, 2),
                "tp2": round(tp2, 2),
                "rr_ratio": round(max(rr, MIN_RR_RATIO), 2),
                "position_size": round(min(size, regime.position_size_cap), 2),
                "risk_amount": round(risk, 2)
            })
        
        elif primary_direction == "BEARISH" and regime.allows_short:
            entry = current_price
            stop = min(gann_5_8 + daily_atr, current_price * 1.05)
            risk = stop - entry
            
            # Enforce min R:R of 1.5
            min_tp = entry - (risk * MIN_RR_RATIO)
            tp1 = min(gann_3_8, min_tp)
            tp2 = gann_2_8
            
            rr = (entry - tp1) / risk if risk > 0 else 0
            
            confidence = "HIGH" if consensus["alignment_count"] >= 3 else "MEDIUM" if consensus["alignment_count"] >= 2 else "LOW"
            size = 1.0 if confidence == "HIGH" else 0.5 if confidence == "MEDIUM" else 0.25
            
            setups.append({
                "id": 1,
                "type": "PRIMARY",
                "direction": "SHORT",
                "confidence": confidence,
                "entry": round(entry, 2),
                "stop_loss": round(stop, 2),
                "tp1": round(tp1, 2),
                "tp2": round(tp2, 2),
                "rr_ratio": round(max(rr, MIN_RR_RATIO), 2),
                "position_size": round(min(size, regime.position_size_cap), 2),
                "risk_amount": round(risk, 2)
            })
        
        # Counter-trend setup at capitulation
        if capitulation.is_capitulation and regime.allows_long:
            entry = gann_3_8
            stop = gann_2_8 - daily_atr
            risk = entry - stop
            tp1 = current_price * 1.05
            rr = (tp1 - entry) / risk if risk > 0 else 0
            
            setups.append({
                "id": 2,
                "type": "COUNTER_TREND",
                "direction": "LONG",
                "confidence": "MEDIUM" if capitulation.status == "CONFIRMED" else "LOW",
                "entry": round(entry, 2),
                "stop_loss": round(stop, 2),
                "tp1": round(tp1, 2),
                "tp2": round(current_price * 1.10, 2),
                "rr_ratio": round(max(rr, MIN_RR_RATIO), 2),
                "position_size": 0.25,
                "risk_amount": round(risk, 2)
            })
        
        # Default WAIT setup if no valid setups
        if not setups or consensus["alignment_count"] < 2:
            setups.append({
                "id": len(setups) + 1,
                "type": "WAIT",
                "direction": "FLAT",
                "confidence": "NONE",
                "entry": round(current_price, 2),
                "stop_loss": None,
                "tp1": None,
                "tp2": None,
                "rr_ratio": 0,
                "position_size": 0,
                "risk_amount": 0,
                "reason": "Low alignment or conflicting signals"
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
                confidence = 0.45 + (0.15 if cycle in [180, 360] else 0.05)
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
        
        return {
            "next_pivot_date": next_pivot_date,
            "days_to_pivot": days_to_cycle,
            "pivot_type": expected_pivot,
            "confidence": round(confidence, 2),
            "confidence_level": "HIGH" if confidence >= 0.7 else "MEDIUM" if confidence >= 0.5 else "LOW",
            "probable_price_low": round(max(current_price - atr_proj, gann_3_8 * 0.95), 2),
            "probable_price_high": round(min(current_price + atr_proj, gann_5_8 * 1.05), 2),
            "cycle_origin": {
                "reference_type": reference_type,
                "reference_date": ref_date_str,
                "reference_price": round(ref_price, 2),
                "days_since": days_since,
                "cycle_length": cycle_used
            }
        }
    
    # ========================================================
    # PRIMARY BIAS (Gann 50% Rule)
    # ========================================================
    
    def determine_primary_bias(self, current_price: float, weekly_gann: GannLevels, 
                               daily_atr: float, consensus: Dict) -> Dict:
        """Determine primary bias using Gann 50% rule with ATR buffer."""
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
        
        # Gann 50% rule takes precedence
        if gann_bias != "NEUTRAL" and gann_bias != consensus_bias:
            primary_bias = gann_bias
            bias_source = "GANN_50_OVERRIDE"
            conflict = True
        else:
            primary_bias = consensus_bias if consensus_bias != "NEUTRAL" else gann_bias
            bias_source = "CONSENSUS" if consensus_bias == primary_bias else "GANN_50"
            conflict = False
        
        return {
            "primary_bias": primary_bias,
            "bias_source": bias_source,
            "gann_50_level": gann_50,
            "gann_position": gann_position,
            "consensus_direction": consensus_bias,
            "conflict": conflict,
            "description": f"Price ${current_price:,.0f} is {gann_position} Gann 50% (${gann_50:,.0f})"
        }
    
    # ========================================================
    # MAIN MTF SIGNAL GENERATOR
    # ========================================================
    
    def generate_mtf_signal(self, symbol: str = "BTCUSDT") -> Dict:
        """Main MTF signal generator - v5.0.7 with all N/A fixes."""
        try:
            logger.info(f"[MTF] Generating signal for {symbol}")
            
            # Fetch data
            df_1d = self.fetch_real_binance_data(use_cache=True, symbol=symbol)
            if df_1d is None or len(df_1d) < 100:
                raise ValueError(f"Insufficient data: {len(df_1d) if df_1d is not None else 0} candles")
            
            logger.info(f"[MTF] {len(df_1d)} candles fetched")
            
            # Resample
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
            
            # Detect capitulation and determine regime
            capitulation = self.detect_capitulation(df_1w, df_1d, weekly_rsi, weekly_gann, current_price)
            regime = self.determine_regime(df_1d, weekly_rsi, weekly_adx, capitulation, current_price)
            
            # Calculate consensus
            consensus = self.calculate_mtf_consensus(timeframes, regime)
            
            # Time forecast
            daily_atr = float(timeframes["1D"]["atr"])
            time_forecast = self.calculate_time_forecast(df_1d, current_price, daily_atr, weekly_gann)
            
            # Primary bias (Gann 50% rule)
            primary_bias = self.determine_primary_bias(current_price, weekly_gann, daily_atr, consensus)
            
            # Generate trade setups
            trade_setups = self.generate_trade_setups(
                current_price, timeframes, consensus, regime, capitulation, weekly_gann
            )
            
            # Calculate invalidation price
            if primary_bias["primary_bias"] == "BULLISH":
                invalidation_price = round(weekly_gann.levels.get("3_8", current_price * 0.92), 2)
            elif primary_bias["primary_bias"] == "BEARISH":
                invalidation_price = round(weekly_gann.levels.get("5_8", current_price * 1.08), 2)
            else:
                invalidation_price = None
            
            # Build price levels
            price_levels = {
                "monthly": {
                    "high": timeframes["1M"]["gann_high"],
                    "low": timeframes["1M"]["gann_low"],
                    "gann_50": timeframes["1M"]["gann_50_pct"]
                },
                "weekly": {
                    "high": weekly_gann.high,
                    "low": weekly_gann.low,
                    "gann_50": weekly_gann.gann_50_pct,
                    "gann_3_8": weekly_gann.levels.get("3_8"),
                    "gann_5_8": weekly_gann.levels.get("5_8")
                },
                "daily": {
                    "high": timeframes["1D"]["gann_high"],
                    "low": timeframes["1D"]["gann_low"],
                    "gann_50": timeframes["1D"]["gann_50_pct"]
                }
            }
            
            logger.info(f"[MTF] Signal generated: {consensus['primary_direction']} ({consensus['confidence_level']})")
            
            # Build complete result
            result = {
                "status": "success",
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "signal_date": signal_date,
                "timestamp": datetime.now().isoformat(),
                "version": self.VERSION,
                
                # Regime
                "regime": {
                    "current": regime.regime.value,
                    "strength": round(regime.regime_strength, 2),
                    "override_active": bool(regime.override_active),
                    "override_reason": regime.override_reason,
                    "allows_short": bool(regime.allows_short),
                    "allows_long": bool(regime.allows_long),
                    "position_size_cap": round(regime.position_size_cap, 2),
                    "warnings": regime.warnings
                },
                
                # Timeframes
                "timeframes": timeframes,
                
                # Consensus
                "consensus": consensus,
                
                # Primary Bias (Gann 50% Rule)
                "primary_bias": primary_bias,
                
                # Price Levels
                "price_levels": price_levels,
                
                # Capitulation
                "capitulation": {
                    "is_capitulation": bool(capitulation.is_capitulation),
                    "status": capitulation.status,
                    "confidence": round(capitulation.confidence, 2),
                    "criteria_met": capitulation.criteria_met,
                    "criteria_missing": capitulation.criteria_missing,
                    "details": {
                        "rsi_extreme": bool(capitulation.rsi_extreme),
                        "volume_spike": bool(capitulation.volume_spike),
                        "gann_support_test": bool(capitulation.gann_support_test),
                        "bullish_divergence": bool(capitulation.bullish_divergence),
                        "weekly_rsi": round(capitulation.weekly_rsi, 2),
                        "volume_ratio": round(capitulation.volume_ratio, 2)
                    }
                },
                
                # Time Forecast
                "time_forecast": time_forecast,
                
                # Trade Setups
                "trade_setups": trade_setups,
                
                # Invalidation
                "invalidation": {
                    "price": invalidation_price,
                    "description": f"Bias invalidated if price crosses ${invalidation_price:,.0f}" if invalidation_price else "N/A"
                },
                
                # Enneagram States
                "enneagram": {
                    "1M": timeframes["1M"]["state_name"],
                    "1W": timeframes["1W"]["state_name"],
                    "3D": timeframes["3D"]["state_name"],
                    "1D": timeframes["1D"]["state_name"],
                    "dominant": timeframes["1W"]["state_name"],
                    "phase": "Accumulation" if primary_bias["primary_bias"] == "BULLISH" else "Distribution" if primary_bias["primary_bias"] == "BEARISH" else "Transition",
                    "arrow": "UP" if primary_bias["primary_bias"] == "BULLISH" else "DOWN" if primary_bias["primary_bias"] == "BEARISH" else "FLAT"
                },
                
                # Signal summary
                "signal": {
                    "type": trade_setups[0]["direction"] if trade_setups else "WAIT",
                    "confidence": trade_setups[0]["confidence"] if trade_setups else "NONE",
                    "entry": trade_setups[0].get("entry"),
                    "stop_loss": trade_setups[0].get("stop_loss"),
                    "take_profit": trade_setups[0].get("tp1"),
                    "tp2": trade_setups[0].get("tp2"),
                    "rr_ratio": trade_setups[0].get("rr_ratio"),
                    "position_size": trade_setups[0].get("position_size", 0)
                },
                
                # Gann interpretation
                "gann_interpretation": {
                    "weekly_50_pct": weekly_gann.gann_50_pct,
                    "current_position": primary_bias["gann_position"],
                    "description": primary_bias["description"],
                    "primary_bias": primary_bias["primary_bias"]
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
