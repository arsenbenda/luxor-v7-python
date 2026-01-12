# ============================================================
# LUXOR V7 PRANA - GANN EGYPT-INDIA UNIFIED SYSTEM v5.0.6
# Enhanced with Multi-Timeframe Analysis
# ============================================================

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
import logging
import ccxt

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

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
    "1D": TimeframeConfig("1D", 0.15, 252, 126),
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
    """LUXOR V7 PRANA - GANN EGYPT-INDIA UNIFIED SYSTEM v5.0.6"""
    
    CACHE = {
        'df': None,
        'last_fetch': None,
        'cache_duration': 3600
    }
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.sidereal_epoch = pd.Timestamp('1900-01-01')
        logger.info("[INIT] LuxorV7PranaSystem v5.0.6 initialized")
    
    # ========================================================
    # DATA FETCHING (Multi-Exchange via CCXT)
    # ========================================================
    
    def fetch_ohlcv_ccxt(self, symbol: str = "BTC/USDT", interval: str = "1d", limit: int = 500) -> pd.DataFrame:
        """Fetch OHLCV using CCXT with multi-exchange fallback."""
        exchanges_to_try = [
            ('kraken', 'XBT/USD' if 'BTC' in symbol else symbol.replace('USDT', 'USD')),
            ('coinbasepro', symbol.replace('/', '-').replace('USDT', 'USD')),
            ('kucoin', symbol),
            ('bybit', symbol),
            ('okx', symbol),
            ('bitfinex', symbol),
        ]
        
        for exchange_id, sym in exchanges_to_try:
            try:
                logger.info(f"Trying {exchange_id} for {sym}")
                exchange_class = getattr(ccxt, exchange_id)
                exchange = exchange_class({'enableRateLimit': True, 'timeout': 30000})
                
                ohlcv = exchange.fetch_ohlcv(sym, interval, limit=limit)
                
                if ohlcv and len(ohlcv) > 0:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['date'] = df['timestamp']
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    
                    logger.info(f"Fetched {len(df)} candles from {exchange_id}")
                    return df
            except Exception as e:
                logger.warning(f"{exchange_id} failed: {str(e)[:80]}")
                continue
        
        raise Exception("All exchanges failed")
    
    def fetch_real_binance_data(self, use_cache=True, symbol: str = "BTCUSDT"):
        """Fetch data with caching - now uses CCXT."""
        try:
            if use_cache and self.CACHE['df'] is not None:
                cache_age = (datetime.now() - self.CACHE['last_fetch']).total_seconds()
                if cache_age < self.CACHE['cache_duration']:
                    logger.info(f"[CACHE] Using cached data (age: {cache_age:.0f}s)")
                    return self.CACHE['df'].copy()
            
            # Convert symbol format
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
        
        resample_map = {'3D': '3D', '1W': 'W', '1M': 'ME'}
        rule = resample_map.get(target_tf, '1D')
        
        resampled = df.resample(rule).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        
        resampled.reset_index(inplace=True)
        resampled.rename(columns={'index': 'timestamp', 'date': 'timestamp'}, inplace=True)
        return resampled
    
    # ========================================================
    # INDICATOR CALCULATIONS
    # ========================================================
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))
    
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
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1)
        return dx.ewm(alpha=1/period, min_periods=period).mean()
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def calculate_ichimoku(self, df: pd.DataFrame) -> Dict:
        high, low, close = df['high'], df['low'], df['close']
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        return {'tenkan': tenkan, 'kijun': kijun, 'senkou_a': senkou_a, 'senkou_b': senkou_b}
    
    def calculate_gann_levels(self, df: pd.DataFrame, tf_config: TimeframeConfig) -> GannLevels:
        lookback = min(tf_config.gann_lookback, len(df))
        recent_df = df.tail(lookback)
        
        high = recent_df['high'].max()
        low = recent_df['low'].min()
        high_idx = recent_df['high'].idxmax()
        low_idx = recent_df['low'].idxmin()
        
        ts_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        high_date = str(df.loc[high_idx, ts_col]) if ts_col in df.columns else str(high_idx)
        low_date = str(df.loc[low_idx, ts_col]) if ts_col in df.columns else str(low_idx)
        
        range_value = high - low
        current_price = df['close'].iloc[-1]
        range_pct = (range_value / current_price) * 100 if current_price > 0 else 0
        
        levels = {f"{i}_8": low + (range_value * i / 8) for i in range(9)}
        
        return GannLevels(
            high=float(high), low=float(low), high_date=high_date, low_date=low_date,
            range_value=float(range_value), range_pct=float(range_pct),
            lookback_bars=lookback, levels=levels
        )
    
    # ========================================================
    # DIVERGENCE & CAPITULATION DETECTION
    # ========================================================
    
    def detect_divergence(self, df: pd.DataFrame, rsi: pd.Series, lookback: int = 14) -> Dict:
        if len(df) < lookback * 2:
            return {"bullish_divergence": False, "bearish_divergence": False}
        
        recent_prices = df['close'].tail(lookback * 2)
        recent_rsi = rsi.tail(lookback * 2)
        
        price_min_1, price_min_2 = recent_prices.iloc[:lookback].min(), recent_prices.iloc[lookback:].min()
        rsi_min_1, rsi_min_2 = recent_rsi.iloc[:lookback].min(), recent_rsi.iloc[lookback:].min()
        
        bullish_div = (price_min_2 < price_min_1) and (rsi_min_2 > rsi_min_1)
        
        return {"bullish_divergence": bullish_div, "bearish_divergence": False}
    
    def detect_capitulation(self, df_weekly: pd.DataFrame, df_daily: pd.DataFrame, 
                           weekly_rsi: float, weekly_gann: GannLevels, current_price: float) -> CapitulationAnalysis:
        criteria_met, criteria_missing = [], []
        
        # RSI extreme
        rsi_extreme = weekly_rsi < CAPITULATION_RSI_THRESHOLD
        if rsi_extreme:
            criteria_met.append(f"RSI_EXTREME: {weekly_rsi:.1f}")
        else:
            criteria_missing.append(f"RSI not extreme: {weekly_rsi:.1f}")
        
        # Volume spike
        volume_ratio = 1.0
        if len(df_daily) >= 20 and 'volume' in df_daily.columns:
            volume_ratio = df_daily['volume'].iloc[-1] / df_daily['volume'].tail(20).mean()
        volume_spike = volume_ratio >= VOLUME_SPIKE_THRESHOLD
        if volume_spike:
            criteria_met.append(f"VOLUME_SPIKE: {volume_ratio:.1f}x")
        else:
            criteria_missing.append(f"No volume spike: {volume_ratio:.1f}x")
        
        # Gann support
        gann_3_8 = weekly_gann.levels.get("3_8", weekly_gann.low)
        gann_support_test = abs(current_price - gann_3_8) / current_price < 0.05
        if gann_support_test:
            criteria_met.append("GANN_SUPPORT")
        else:
            criteria_missing.append("Not near Gann support")
        
        # Divergence
        weekly_rsi_series = self.calculate_rsi(df_weekly['close'])
        divergence = self.detect_divergence(df_weekly, weekly_rsi_series, 8)
        bullish_divergence = divergence.get("bullish_divergence", False)
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
            is_capitulation=num_criteria >= 3, status=status, confidence=confidence,
            criteria_met=criteria_met, criteria_missing=criteria_missing,
            rsi_extreme=rsi_extreme, volume_spike=volume_spike,
            gann_support_test=gann_support_test, bullish_divergence=bullish_divergence,
            weekly_rsi=weekly_rsi, volume_ratio=volume_ratio
        )
    
    # ========================================================
    # REGIME DETECTION WITH OVERRIDE
    # ========================================================
    
    def determine_regime(self, df_daily: pd.DataFrame, weekly_rsi: float, weekly_adx: float,
                        capitulation: CapitulationAnalysis, current_price: float) -> RegimeAnalysis:
        warnings = []
        override_active = False
        override_reason = None
        
        sma_200 = df_daily['close'].rolling(200).mean().iloc[-1] if len(df_daily) >= 200 else current_price
        price_vs_sma = "above" if current_price > sma_200 else "below"
        trend_strength = "strong" if weekly_adx > ADX_STRONG_TREND else "weak"
        
        if price_vs_sma == "above" and trend_strength == "strong":
            base_regime = Regime.TRENDING_BULL
        elif price_vs_sma == "below" and trend_strength == "strong":
            base_regime = Regime.TRENDING_BEAR
        else:
            base_regime = Regime.RANGING
        
        # CAPITULATION OVERRIDE
        if capitulation.is_capitulation:
            override_active = True
            override_reason = f"Capitulation {capitulation.status}"
            final_regime = Regime.CAPITULATION
            allows_short = False
            allows_long = True
            position_size_cap = 0.25 if capitulation.status == "POTENTIAL" else 0.50
            warnings.append("CAPITULATION OVERRIDE: Shorts blocked")
        
        # EUPHORIA OVERRIDE
        elif weekly_rsi > EUPHORIA_RSI_THRESHOLD:
            override_active = True
            override_reason = f"Euphoria: RSI {weekly_rsi:.1f}"
            final_regime = Regime.EUPHORIA
            allows_short = True
            allows_long = False
            position_size_cap = 0.25
            warnings.append("EUPHORIA OVERRIDE: Longs blocked")
        
        else:
            final_regime = base_regime
            allows_short = True
            allows_long = True
            position_size_cap = 1.0
        
        regime_strength = min(weekly_adx / 100, 1.0) if final_regime in [Regime.TRENDING_BULL, Regime.TRENDING_BEAR] else 0.5
        
        return RegimeAnalysis(
            regime=final_regime, regime_strength=regime_strength,
            override_active=override_active, override_reason=override_reason,
            allows_short=allows_short, allows_long=allows_long,
            position_size_cap=position_size_cap, warnings=warnings
        )
    
    # ========================================================
    # TIMEFRAME ANALYSIS
    # ========================================================
    
    def analyze_timeframe(self, df: pd.DataFrame, tf_name: str) -> Dict:
        """Analyze a single timeframe."""
        tf_config = TIMEFRAME_CONFIGS[tf_name]
        
        if len(df) < tf_config.min_bars:
            raise ValueError(f"Insufficient data for {tf_name}")
        
        current_price = float(df['close'].iloc[-1])
        
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
        
        # Signal counting
        bullish, bearish = 0, 0
        signal_details = {}
        
        if rsi < RSI_OVERSOLD:
            bullish += 1
            signal_details["RSI"] = f"OVERSOLD ({rsi:.1f})"
        elif rsi > RSI_OVERBOUGHT:
            bearish += 1
            signal_details["RSI"] = f"OVERBOUGHT ({rsi:.1f})"
        else:
            signal_details["RSI"] = f"NEUTRAL ({rsi:.1f})"
        
        if macd_hist > 0:
            bullish += 1
            signal_details["MACD"] = "BULLISH"
        else:
            bearish += 1
            signal_details["MACD"] = "BEARISH"
        
        tenkan_val = ichimoku['tenkan'].iloc[-1]
        kijun_val = ichimoku['kijun'].iloc[-1]
        if tenkan_val > kijun_val:
            bullish += 1
            signal_details["TK_CROSS"] = "BULLISH"
        else:
            bearish += 1
            signal_details["TK_CROSS"] = "BEARISH"
        
        senkou_a = ichimoku['senkou_a'].iloc[-1]
        senkou_b = ichimoku['senkou_b'].iloc[-1]
        cloud_top = max(senkou_a, senkou_b) if not pd.isna(senkou_a) else current_price
        cloud_bottom = min(senkou_a, senkou_b) if not pd.isna(senkou_b) else current_price
        
        if current_price > cloud_top:
            bullish += 1
            signal_details["CLOUD"] = "ABOVE"
        elif current_price < cloud_bottom:
            bearish += 1
            signal_details["CLOUD"] = "BELOW"
        else:
            signal_details["CLOUD"] = "INSIDE"
        
        gann_50 = gann.levels.get("4_8", current_price)
        if current_price > gann_50:
            bullish += 1
            signal_details["GANN_50"] = f"ABOVE (${gann_50:,.0f})"
        else:
            bearish += 1
            signal_details["GANN_50"] = f"BELOW (${gann_50:,.0f})"
        
        # Direction
        if tf_name == "1M":
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
        
        # Volume
        volume_ratio = 1.0
        if 'volume' in df.columns and len(df) >= 20:
            volume_ratio = df['volume'].iloc[-1] / df['volume'].tail(20).mean()
        
        return {
            "direction": direction,
            "state_name": state_name,
            "rsi": round(rsi, 2),
            "adx": round(adx, 2),
            "atr": round(atr, 2),
            "bullish_signals": bullish,
            "bearish_signals": bearish,
            "signal_details": signal_details,
            "volume_ratio": round(volume_ratio, 2),
            "ichimoku": {
                "tk_cross": signal_details.get("TK_CROSS", "NEUTRAL"),
                "price_vs_cloud": signal_details.get("CLOUD", "INSIDE"),
                "kijun": float(kijun_val) if not pd.isna(kijun_val) else current_price,
                "cloud_top": float(cloud_top),
                "cloud_bottom": float(cloud_bottom),
            },
            "gann": {
                "high": gann.high,
                "low": gann.low,
                "levels": gann.levels
            }
        }
    
    # ========================================================
    # MTF CONSENSUS
    # ========================================================
    
    def calculate_mtf_consensus(self, timeframes: Dict, regime: RegimeAnalysis) -> Dict:
        """Calculate weighted consensus across timeframes."""
        weighted_bullish, weighted_bearish = 0, 0
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
                conflicts.append({"timeframe": tf_name, "actual": analysis["direction"]})
        
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
            "weighted_score": weighted_score,
            "alignment": f"{alignment_count}/4",
            "alignment_count": alignment_count,
            "confidence_level": confidence_level,
            "conflicts": conflicts
        }
    
    # ========================================================
    # TRADE SETUPS
    # ========================================================
    
    def generate_trade_setups(self, current_price: float, timeframes: Dict, consensus: Dict,
                             regime: RegimeAnalysis, capitulation: CapitulationAnalysis,
                             weekly_gann: GannLevels) -> List[Dict]:
        """Generate prioritized trade setups."""
        setups = []
        
        daily_atr = timeframes["1D"]["atr"]
        gann_3_8 = weekly_gann.levels.get("3_8", current_price * 0.95)
        gann_5_8 = weekly_gann.levels.get("5_8", current_price * 1.05)
        gann_2_8 = weekly_gann.levels.get("2_8", current_price * 0.90)
        
        primary_direction = consensus["primary_direction"]
        
        if primary_direction == "BULLISH" and regime.allows_long:
            entry = current_price
            stop = gann_3_8 - daily_atr
            tp1 = gann_5_8
            rr = (tp1 - entry) / (entry - stop) if entry > stop else 0
            
            confidence = "HIGH" if consensus["alignment_count"] >= 3 else "MEDIUM" if consensus["alignment_count"] >= 2 else "LOW"
            size = 1.0 if confidence == "HIGH" else 0.5 if confidence == "MEDIUM" else 0.25
            
            setups.append({
                "id": 1, "type": "PRIMARY", "direction": "LONG", "confidence": confidence,
                "entry": round(entry, 2), "stop_loss": round(stop, 2), "tp1": round(tp1, 2),
                "rr_ratio": round(rr, 2), "position_size": min(size, regime.position_size_cap)
            })
        
        elif primary_direction == "BEARISH" and regime.allows_short:
            entry = current_price
            stop = gann_5_8 + daily_atr
            tp1 = gann_3_8
            rr = (entry - tp1) / (stop - entry) if stop > entry else 0
            
            confidence = "HIGH" if consensus["alignment_count"] >= 3 else "MEDIUM" if consensus["alignment_count"] >= 2 else "LOW"
            size = 1.0 if confidence == "HIGH" else 0.5 if confidence == "MEDIUM" else 0.25
            
            setups.append({
                "id": 1, "type": "PRIMARY", "direction": "SHORT", "confidence": confidence,
                "entry": round(entry, 2), "stop_loss": round(stop, 2), "tp1": round(tp1, 2),
                "rr_ratio": round(rr, 2), "position_size": min(size, regime.position_size_cap)
            })
        
        if capitulation.is_capitulation and regime.allows_long:
            entry = gann_3_8
            stop = gann_2_8 - daily_atr
            tp1 = current_price * 1.05
            rr = (tp1 - entry) / (entry - stop) if entry > stop else 0
            
            setups.append({
                "id": 2, "type": "COUNTER_TREND", "direction": "LONG",
                "confidence": "MEDIUM" if capitulation.status == "CONFIRMED" else "LOW",
                "entry": round(entry, 2), "stop_loss": round(stop, 2), "tp1": round(tp1, 2),
                "rr_ratio": round(rr, 2), "position_size": 0.25
            })
        
        if not setups or consensus["alignment_count"] < 2:
            setups.append({
                "id": 3, "type": "WAIT", "direction": "FLAT", "confidence": "NONE",
                "entry": None, "stop_loss": None, "tp1": None, "rr_ratio": None, "position_size": 0
            })
        
        confidence_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}
        setups.sort(key=lambda x: confidence_order.get(x["confidence"], 4))
        
        return setups
    
    # ========================================================
    # TIME FORECAST
    # ========================================================
    
    def calculate_time_forecast(self, df_daily: pd.DataFrame, current_price: float, 
                                atr: float, weekly_gann: GannLevels) -> Dict:
        """Calculate time forecast with Gann cycles."""
        lookback = min(365, len(df_daily))
        recent_df = df_daily.tail(lookback)
        
        major_high = recent_df['high'].max()
        major_low = recent_df['low'].min()
        major_high_idx = recent_df['high'].idxmax()
        major_low_idx = recent_df['low'].idxmin()
        
        if major_high_idx > major_low_idx:
            reference_type, expected_pivot = "HIGH", "LOW"
            days_since = (datetime.now() - pd.to_datetime(df_daily.loc[major_high_idx, 'date' if 'date' in df_daily.columns else 'timestamp'])).days
        else:
            reference_type, expected_pivot = "LOW", "HIGH"
            days_since = (datetime.now() - pd.to_datetime(df_daily.loc[major_low_idx, 'date' if 'date' in df_daily.columns else 'timestamp'])).days
        
        # Find next cycle
        for cycle in [90, 180, 360]:
            days_to_cycle = cycle - (days_since % cycle)
            if days_to_cycle > 0:
                next_pivot_date = (datetime.now() + timedelta(days=days_to_cycle)).strftime("%Y-%m-%d")
                confidence = 0.45 + (0.15 if cycle in [180, 360] else 0)
                break
        else:
            next_pivot_date = (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d")
            days_to_cycle = 90
            confidence = 0.40
        
        gann_3_8 = weekly_gann.levels.get("3_8", current_price * 0.9)
        gann_5_8 = weekly_gann.levels.get("5_8", current_price * 1.1)
        atr_proj = atr * np.sqrt(days_to_cycle)
        
        return {
            "next_pivot_date": next_pivot_date,
            "days_to_pivot": days_to_cycle,
            "pivot_type": expected_pivot,
            "confidence": round(confidence, 2),
            "confidence_level": "HIGH" if confidence >= 0.7 else "MEDIUM" if confidence >= 0.5 else "LOW",
            "probable_price_low": round(max(current_price - atr_proj, gann_3_8 * 0.95), 2),
            "probable_price_high": round(min(current_price + atr_proj, gann_5_8 * 1.05), 2),
            "cycle_origin": {"reference_type": reference_type, "days_since": days_since}
        }
    
    # ========================================================
    # MAIN MTF SIGNAL GENERATOR (NEW - Replaces evaluate_signals_optimized)
    # ========================================================
    
    def generate_mtf_signal(self, symbol: str = "BTCUSDT") -> Dict:
        """
        Main MTF signal generator.
        This replaces the old evaluate_signals_optimized() method.
        """
        try:
            logger.info(f"[MTF] Generating signal for {symbol}")
            
            # Fetch daily data
            df_1d = self.fetch_real_binance_data(use_cache=True, symbol=symbol)
            if df_1d is None or len(df_1d) < 200:
                raise ValueError("Insufficient data")
            
            # Resample to higher timeframes
            df_3d = self.resample_ohlcv(df_1d, "3D")
            df_1w = self.resample_ohlcv(df_1d, "1W")
            df_1m = self.resample_ohlcv(df_1d, "1M")
            
            current_price = float(df_1d['close'].iloc[-1])
            signal_date = str(df_1d['date'].iloc[-1] if 'date' in df_1d.columns else df_1d['timestamp'].iloc[-1])
            
            # Analyze each timeframe
            timeframes = {
                "1D": self.analyze_timeframe(df_1d, "1D"),
                "3D": self.analyze_timeframe(df_3d, "3D"),
                "1W": self.analyze_timeframe(df_1w, "1W"),
                "1M": self.analyze_timeframe(df_1m, "1M"),
            }
            
            # Weekly data for regime/capitulation
            weekly_rsi = timeframes["1W"]["rsi"]
            weekly_adx = timeframes["1W"]["adx"]
            weekly_gann = self.calculate_gann_levels(df_1w, TIMEFRAME_CONFIGS["1W"])
            
            # Capitulation detection
            capitulation = self.detect_capitulation(df_1w, df_1d, weekly_rsi, weekly_gann, current_price)
            
            # Regime with override
            regime = self.determine_regime(df_1d, weekly_rsi, weekly_adx, capitulation, current_price)
            
            # Consensus
            consensus = self.calculate_mtf_consensus(timeframes, regime)
            
            # Time forecast
            daily_atr = timeframes["1D"]["atr"]
            time_forecast = self.calculate_time_forecast(df_1d, current_price, daily_atr, weekly_gann)
            
            # Trade setups
            trade_setups = self.generate_trade_setups(
                current_price, timeframes, consensus, regime, capitulation, weekly_gann
            )
            
            logger.info(f"[MTF] Signal generated: {consensus['primary_direction']} ({consensus['confidence_level']})")
            
            return {
                "status": "success",
                "symbol": symbol,
                "current_price": current_price,
                "signal_date": signal_date,
                "timestamp": datetime.now().isoformat(),
                "version": "5.0.6",
                
                "timeframes": timeframes,
                "consensus": consensus,
                
                "regime": {
                    "current": regime.regime.value,
                    "strength": regime.regime_strength,
                    "override_active": regime.override_active,
                    "override_reason": regime.override_reason,
                    "allows_short": regime.allows_short,
                    "allows_long": regime.allows_long,
                    "position_size_cap": regime.position_size_cap,
                    "warnings": regime.warnings
                },
                
                "capitulation": {
                    "is_capitulation": capitulation.is_capitulation,
                    "status": capitulation.status,
                    "confidence": capitulation.confidence,
                    "criteria_met": capitulation.criteria_met,
                    "criteria_missing": capitulation.criteria_missing,
                    "details": {
                        "rsi_extreme": capitulation.rsi_extreme,
                        "volume_spike": capitulation.volume_spike,
                        "gann_support_test": capitulation.gann_support_test,
                        "bullish_divergence": capitulation.bullish_divergence,
                        "weekly_rsi": capitulation.weekly_rsi,
                        "volume_ratio": capitulation.volume_ratio
                    }
                },
                
                "time_forecast": time_forecast,
                "trade_setups": trade_setups,
                
                "signal": {
                    "type": trade_setups[0]["direction"] if trade_setups else "WAIT",
                    "confidence": trade_setups[0]["confidence"] if trade_setups else "NONE",
                    "entry": trade_setups[0].get("entry"),
                    "stop_loss": trade_setups[0].get("stop_loss"),
                    "take_profit": trade_setups[0].get("tp1"),
                    "rr_ratio": trade_setups[0].get("rr_ratio"),
                    "position_size": trade_setups[0].get("position_size", 0)
                }
            }
        
        except Exception as e:
            logger.error(f"[MTF] Error: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "detail": str(e), "version": "5.0.6"}
    
    # ========================================================
    # LEGACY METHOD (for backward compatibility)
    # ========================================================
    
    def get_daily_signal(self, df=None):
        """Legacy method - now calls generate_mtf_signal."""
        return self.generate_mtf_signal()
