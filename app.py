# ============================================================
# LUXOR V7 PRANA RUNTIME - MTF EDITION v5.0.6
# Complete Multi-Timeframe Analysis with Capitulation Detection
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION & CONSTANTS
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
MIN_RR_RATIO = 1.5
CAPITULATION_RSI_THRESHOLD = 25
EUPHORIA_RSI_THRESHOLD = 75

# Gann time cycles (days)
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
    levels: Dict[str, float]  # 0_8 through 8_8

@dataclass
class IchimokuSignals:
    tk_cross: str  # BULLISH, BEARISH, NEUTRAL
    price_vs_cloud: str  # ABOVE, BELOW, INSIDE
    cloud_color: str  # BULLISH, BEARISH
    kijun_value: float
    tenkan_value: float
    cloud_top: float
    cloud_bottom: float
    kijun_flat: bool

@dataclass
class TimeframeAnalysis:
    name: str
    direction: Direction
    confidence: Confidence
    state_name: str
    rsi: float
    adx: float
    trend_strength: str
    ichimoku: IchimokuSignals
    gann: GannLevels
    bullish_signals: int
    bearish_signals: int
    signal_details: Dict[str, str]
    volume_ratio: float
    atr: float
    atr_pct: float

@dataclass
class CapitulationAnalysis:
    is_capitulation: bool
    status: str  # CONFIRMED, POTENTIAL, NONE
    confidence: float
    criteria_met: List[str]
    criteria_missing: List[str]
    rsi_extreme: bool
    volume_spike: bool
    gann_support_test: bool
    bullish_divergence: bool
    weekly_rsi: float
    volume_ratio: float
    divergence_details: Optional[Dict]

@dataclass
class TimeForecast:
    next_pivot_date: str
    days_to_pivot: int
    pivot_type: str  # HIGH, LOW
    confidence: float
    confidence_level: str
    probable_price_low: float
    probable_price_high: float
    cycle_origin: Dict  # NEW: transparency
    active_cycles: List[Dict]
    suppressed: bool
    suppression_reason: Optional[str]

@dataclass
class RegimeAnalysis:
    regime: Regime
    regime_strength: float
    override_active: bool
    override_reason: Optional[str]
    original_regime: Optional[Regime]
    allows_short: bool
    allows_long: bool
    position_size_cap: float
    warnings: List[str]

# ============================================================
# CORE CALCULATION FUNCTIONS
# ============================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI with proper handling of edge cases."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, Signal, and Histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ADX for trend strength."""
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
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / atr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1)
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()
    
    return adx

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> Dict:
    """Calculate Ichimoku Cloud components."""
    high, low, close = df['high'], df['low'], df['close']
    
    # Tenkan-sen (Conversion Line)
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    
    # Kijun-sen (Base Line)
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    
    # Senkou Span A (Leading Span A)
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    
    # Senkou Span B (Leading Span B)
    senkou_b_line = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    
    # Chikou Span (Lagging Span)
    chikou = close.shift(-kijun)
    
    return {
        'tenkan': tenkan_sen,
        'kijun': kijun_sen,
        'senkou_a': senkou_a,
        'senkou_b': senkou_b_line,
        'chikou': chikou
    }

def analyze_ichimoku(df: pd.DataFrame, ichimoku_data: Dict) -> IchimokuSignals:
    """Analyze Ichimoku signals for current bar."""
    current_price = df['close'].iloc[-1]
    tenkan = ichimoku_data['tenkan'].iloc[-1]
    kijun = ichimoku_data['kijun'].iloc[-1]
    senkou_a = ichimoku_data['senkou_a'].iloc[-1]
    senkou_b = ichimoku_data['senkou_b'].iloc[-1]
    
    # Handle NaN values
    if pd.isna(senkou_a) or pd.isna(senkou_b):
        cloud_top = cloud_bottom = current_price
    else:
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
    
    # TK Cross
    prev_tenkan = ichimoku_data['tenkan'].iloc[-2] if len(df) > 1 else tenkan
    prev_kijun = ichimoku_data['kijun'].iloc[-2] if len(df) > 1 else kijun
    
    if tenkan > kijun and prev_tenkan <= prev_kijun:
        tk_cross = "BULLISH"
    elif tenkan < kijun and prev_tenkan >= prev_kijun:
        tk_cross = "BEARISH"
    elif tenkan > kijun:
        tk_cross = "BULLISH"
    elif tenkan < kijun:
        tk_cross = "BEARISH"
    else:
        tk_cross = "NEUTRAL"
    
    # Price vs Cloud
    if current_price > cloud_top:
        price_vs_cloud = "ABOVE"
    elif current_price < cloud_bottom:
        price_vs_cloud = "BELOW"
    else:
        price_vs_cloud = "INSIDE"
    
    # Cloud Color (future)
    cloud_color = "BULLISH" if senkou_a > senkou_b else "BEARISH"
    
    # Kijun flat detection
    kijun_values = ichimoku_data['kijun'].tail(5)
    kijun_flat = kijun_values.std() < (current_price * 0.001)  # <0.1% variation
    
    return IchimokuSignals(
        tk_cross=tk_cross,
        price_vs_cloud=price_vs_cloud,
        cloud_color=cloud_color,
        kijun_value=float(kijun) if not pd.isna(kijun) else current_price,
        tenkan_value=float(tenkan) if not pd.isna(tenkan) else current_price,
        cloud_top=float(cloud_top),
        cloud_bottom=float(cloud_bottom),
        kijun_flat=kijun_flat
    )

# ============================================================
# GANN CALCULATIONS (Per-Timeframe)
# ============================================================

def calculate_gann_levels(df: pd.DataFrame, tf_config: TimeframeConfig) -> GannLevels:
    """Calculate Gann levels using timeframe-specific lookback."""
    lookback = min(tf_config.gann_lookback, len(df))
    recent_df = df.tail(lookback)
    
    high = recent_df['high'].max()
    low = recent_df['low'].min()
    high_idx = recent_df['high'].idxmax()
    low_idx = recent_df['low'].idxmin()
    
    high_date = df.loc[high_idx, 'timestamp'] if 'timestamp' in df.columns else str(high_idx)
    low_date = df.loc[low_idx, 'timestamp'] if 'timestamp' in df.columns else str(low_idx)
    
    range_value = high - low
    current_price = df['close'].iloc[-1]
    range_pct = (range_value / current_price) * 100 if current_price > 0 else 0
    
    # Calculate 8ths
    levels = {}
    for i in range(9):
        level_name = f"{i}_8"
        levels[level_name] = low + (range_value * i / 8)
    
    return GannLevels(
        high=float(high),
        low=float(low),
        high_date=str(high_date),
        low_date=str(low_date),
        range_value=float(range_value),
        range_pct=float(range_pct),
        lookback_bars=lookback,
        levels=levels
    )

# ============================================================
# DIVERGENCE DETECTION
# ============================================================

def detect_divergence(df: pd.DataFrame, rsi: pd.Series, lookback: int = 14) -> Dict:
    """
    Detect bullish and bearish divergence between price and RSI.
    
    Bullish divergence: Price makes lower low, RSI makes higher low
    Bearish divergence: Price makes higher high, RSI makes lower high
    """
    if len(df) < lookback * 2:
        return {
            "bullish_divergence": False,
            "bearish_divergence": False,
            "divergence_type": None,
            "details": None
        }
    
    recent_prices = df['close'].tail(lookback * 2)
    recent_rsi = rsi.tail(lookback * 2)
    
    # Find local minima and maxima
    price_min_1 = recent_prices.iloc[:lookback].min()
    price_min_2 = recent_prices.iloc[lookback:].min()
    price_max_1 = recent_prices.iloc[:lookback].max()
    price_max_2 = recent_prices.iloc[lookback:].max()
    
    rsi_at_price_min_1 = recent_rsi.iloc[recent_prices.iloc[:lookback].idxmin() - recent_prices.index[0]]
    rsi_at_price_min_2 = recent_rsi.iloc[-1]  # Current RSI at recent low
    rsi_at_price_max_1 = recent_rsi.iloc[recent_prices.iloc[:lookback].idxmax() - recent_prices.index[0]]
    rsi_at_price_max_2 = recent_rsi.iloc[-1]
    
    # Bullish divergence: lower price low + higher RSI low
    bullish_div = (price_min_2 < price_min_1) and (rsi_at_price_min_2 > rsi_at_price_min_1)
    
    # Bearish divergence: higher price high + lower RSI high
    bearish_div = (price_max_2 > price_max_1) and (rsi_at_price_max_2 < rsi_at_price_max_1)
    
    divergence_type = None
    if bullish_div:
        divergence_type = "BULLISH"
    elif bearish_div:
        divergence_type = "BEARISH"
    
    return {
        "bullish_divergence": bullish_div,
        "bearish_divergence": bearish_div,
        "divergence_type": divergence_type,
        "details": {
            "price_low_1": float(price_min_1),
            "price_low_2": float(price_min_2),
            "rsi_at_low_1": float(rsi_at_price_min_1) if not pd.isna(rsi_at_price_min_1) else None,
            "rsi_at_low_2": float(rsi_at_price_min_2) if not pd.isna(rsi_at_price_min_2) else None,
            "price_high_1": float(price_max_1),
            "price_high_2": float(price_max_2),
        }
    }

# ============================================================
# CAPITULATION DETECTION (NEW - Complete Implementation)
# ============================================================

def detect_capitulation(
    df_weekly: pd.DataFrame,
    df_daily: pd.DataFrame,
    weekly_rsi: float,
    weekly_gann: GannLevels,
    current_price: float
) -> CapitulationAnalysis:
    """
    Comprehensive capitulation detection with multiple confirmation criteria.
    
    Criteria for CONFIRMED capitulation:
    1. Weekly RSI < 25 (extreme oversold)
    2. Volume spike > 2x average
    3. Price testing Gann 2/8 - 3/8 support zone
    4. Bullish RSI divergence (price lower low, RSI higher low)
    
    Status:
    - CONFIRMED: 4/4 criteria met
    - POTENTIAL: 3/4 criteria met
    - DEVELOPING: 2/4 criteria met
    - NONE: <2 criteria met
    """
    criteria_met = []
    criteria_missing = []
    
    # Criterion 1: Extreme RSI
    rsi_extreme = weekly_rsi < CAPITULATION_RSI_THRESHOLD
    if rsi_extreme:
        criteria_met.append(f"RSI_EXTREME: Weekly RSI {weekly_rsi:.1f} < {CAPITULATION_RSI_THRESHOLD}")
    else:
        criteria_missing.append(f"RSI_EXTREME: Weekly RSI {weekly_rsi:.1f} >= {CAPITULATION_RSI_THRESHOLD}")
    
    # Criterion 2: Volume spike
    if len(df_daily) >= 20:
        recent_volume = df_daily['volume'].iloc[-1]
        avg_volume = df_daily['volume'].tail(20).mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        volume_spike = volume_ratio >= VOLUME_SPIKE_THRESHOLD
    else:
        volume_ratio = 1.0
        volume_spike = False
    
    if volume_spike:
        criteria_met.append(f"VOLUME_SPIKE: {volume_ratio:.1f}x average (>= {VOLUME_SPIKE_THRESHOLD}x)")
    else:
        criteria_missing.append(f"VOLUME_SPIKE: {volume_ratio:.1f}x average (< {VOLUME_SPIKE_THRESHOLD}x required)")
    
    # Criterion 3: Gann support test (price near 2/8 - 3/8 zone)
    gann_2_8 = weekly_gann.levels.get("2_8", weekly_gann.low)
    gann_3_8 = weekly_gann.levels.get("3_8", weekly_gann.low)
    support_zone_high = gann_3_8
    support_zone_low = gann_2_8
    
    # Price within 5% of support zone
    distance_to_support = abs(current_price - support_zone_high) / current_price
    gann_support_test = distance_to_support < 0.05 or (support_zone_low <= current_price <= support_zone_high * 1.05)
    
    if gann_support_test:
        criteria_met.append(f"GANN_SUPPORT: Price ${current_price:,.0f} near 2/8-3/8 zone (${gann_2_8:,.0f}-${gann_3_8:,.0f})")
    else:
        criteria_missing.append(f"GANN_SUPPORT: Price ${current_price:,.0f} not near support zone (${gann_2_8:,.0f}-${gann_3_8:,.0f})")
    
    # Criterion 4: Bullish divergence
    weekly_rsi_series = calculate_rsi(df_weekly['close'])
    divergence = detect_divergence(df_weekly, weekly_rsi_series, lookback=8)
    bullish_divergence = divergence.get("bullish_divergence", False)
    
    if bullish_divergence:
        criteria_met.append("BULLISH_DIVERGENCE: Price lower low + RSI higher low detected")
    else:
        criteria_missing.append("BULLISH_DIVERGENCE: No bullish divergence detected")
    
    # Determine status
    num_criteria = len(criteria_met)
    if num_criteria >= 4:
        status = "CONFIRMED"
        confidence = 0.90
    elif num_criteria >= 3:
        status = "POTENTIAL"
        confidence = 0.70
    elif num_criteria >= 2:
        status = "DEVELOPING"
        confidence = 0.50
    else:
        status = "NONE"
        confidence = 0.20
    
    is_capitulation = num_criteria >= 3
    
    return CapitulationAnalysis(
        is_capitulation=is_capitulation,
        status=status,
        confidence=confidence,
        criteria_met=criteria_met,
        criteria_missing=criteria_missing,
        rsi_extreme=rsi_extreme,
        volume_spike=volume_spike,
        gann_support_test=gann_support_test,
        bullish_divergence=bullish_divergence,
        weekly_rsi=weekly_rsi,
        volume_ratio=volume_ratio,
        divergence_details=divergence.get("details")
    )

# ============================================================
# REGIME DETECTION WITH OVERRIDE LOGIC
# ============================================================

def determine_regime(
    df_daily: pd.DataFrame,
    weekly_analysis: TimeframeAnalysis,
    capitulation: CapitulationAnalysis,
    current_price: float
) -> RegimeAnalysis:
    """
    Determine market regime with capitulation/euphoria override logic.
    
    Override rules:
    1. If capitulation CONFIRMED/POTENTIAL → Block shorts, allow only counter-trend longs
    2. If euphoria detected → Block longs, allow only counter-trend shorts
    3. Position sizing capped based on regime confidence
    """
    warnings = []
    override_active = False
    override_reason = None
    original_regime = None
    
    # Calculate base regime from trend indicators
    sma_200 = df_daily['close'].rolling(200).mean().iloc[-1] if len(df_daily) >= 200 else current_price
    adx = weekly_analysis.adx
    weekly_direction = weekly_analysis.direction
    
    price_vs_sma = "above" if current_price > sma_200 else "below"
    trend_strength = "strong" if adx > ADX_STRONG_TREND else "weak"
    
    # Base regime determination
    if price_vs_sma == "above" and trend_strength == "strong" and weekly_direction == Direction.BULLISH:
        base_regime = Regime.TRENDING_BULL
    elif price_vs_sma == "below" and trend_strength == "strong" and weekly_direction == Direction.BEARISH:
        base_regime = Regime.TRENDING_BEAR
    elif price_vs_sma == "below" and trend_strength == "strong":
        base_regime = Regime.TRENDING_BEAR
    elif price_vs_sma == "above" and trend_strength == "strong":
        base_regime = Regime.TRENDING_BULL
    else:
        base_regime = Regime.RANGING
    
    # CAPITULATION OVERRIDE
    if capitulation.is_capitulation or capitulation.status in ["CONFIRMED", "POTENTIAL"]:
        original_regime = base_regime
        override_active = True
        override_reason = f"Capitulation {capitulation.status}: {len(capitulation.criteria_met)}/4 criteria met"
        
        # Override to CAPITULATION regime
        final_regime = Regime.CAPITULATION
        allows_short = False  # Block shorts at capitulation
        allows_long = True    # Allow counter-trend longs
        position_size_cap = 0.25 if capitulation.status == "POTENTIAL" else 0.50
        
        warnings.append(f"⚠️ CAPITULATION OVERRIDE: Shorts blocked until weekly RSI > {RSI_OVERSOLD}")
        warnings.append(f"⚠️ Position size capped at {position_size_cap*100:.0f}%")
        
        if capitulation.bullish_divergence:
            warnings.append("✅ Bullish divergence detected - watch for reversal")
    
    # EUPHORIA OVERRIDE (opposite case)
    elif weekly_analysis.rsi > EUPHORIA_RSI_THRESHOLD:
        original_regime = base_regime
        override_active = True
        override_reason = f"Euphoria detected: Weekly RSI {weekly_analysis.rsi:.1f} > {EUPHORIA_RSI_THRESHOLD}"
        
        final_regime = Regime.EUPHORIA
        allows_short = True   # Allow counter-trend shorts
        allows_long = False   # Block longs at euphoria
        position_size_cap = 0.25
        
        warnings.append(f"⚠️ EUPHORIA OVERRIDE: Longs blocked until weekly RSI < {RSI_OVERBOUGHT}")
    
    else:
        # No override - use base regime
        final_regime = base_regime
        allows_short = True
        allows_long = True
        position_size_cap = 1.0
    
    # Calculate regime strength
    if final_regime in [Regime.TRENDING_BULL, Regime.TRENDING_BEAR]:
        regime_strength = min(adx / 100, 1.0)
    elif final_regime == Regime.CAPITULATION:
        regime_strength = capitulation.confidence
    elif final_regime == Regime.EUPHORIA:
        regime_strength = min((weekly_analysis.rsi - 70) / 30, 1.0)
    else:
        regime_strength = 0.5
    
    return RegimeAnalysis(
        regime=final_regime,
        regime_strength=regime_strength,
        override_active=override_active,
        override_reason=override_reason,
        original_regime=original_regime,
        allows_short=allows_short,
        allows_long=allows_long,
        position_size_cap=position_size_cap,
        warnings=warnings
    )

# ============================================================
# TIME FORECAST WITH CYCLE TRANSPARENCY
# ============================================================

def calculate_time_forecast(
    df_daily: pd.DataFrame,
    current_price: float,
    atr: float,
    weekly_gann: GannLevels,
    confidence_threshold: float = 0.50
) -> TimeForecast:
    """
    Calculate time forecast with full cycle transparency.
    
    Exposes:
    - cycle_origin: What date/event the cycle is measured from
    - cycle_length: Which Gann cycle is active
    - confidence_factors: What contributes to confidence score
    """
    # Find last major high and low
    lookback_days = 365
    recent_df = df_daily.tail(lookback_days) if len(df_daily) >= lookback_days else df_daily
    
    major_high = recent_df['high'].max()
    major_low = recent_df['low'].min()
    major_high_idx = recent_df['high'].idxmax()
    major_low_idx = recent_df['low'].idxmin()
    
    major_high_date = df_daily.loc[major_high_idx, 'timestamp'] if 'timestamp' in df_daily.columns else None
    major_low_date = df_daily.loc[major_low_idx, 'timestamp'] if 'timestamp' in df_daily.columns else None
    
    # Determine reference point (most recent extreme)
    if major_high_idx > major_low_idx:
        reference_date = major_high_date
        reference_price = major_high
        reference_type = "HIGH"
        expected_pivot_type = "LOW"  # After high, expect low
    else:
        reference_date = major_low_date
        reference_price = major_low
        reference_type = "LOW"
        expected_pivot_type = "HIGH"  # After low, expect high
    
    # Calculate days since reference
    today = datetime.now()
    if reference_date:
        if isinstance(reference_date, str):
            ref_dt = datetime.fromisoformat(reference_date.replace('Z', '+00:00'))
        else:
            ref_dt = pd.to_datetime(reference_date)
        days_since_reference = (today - ref_dt.replace(tzinfo=None)).days
    else:
        days_since_reference = 90  # Default
    
    # Find next applicable Gann cycle
    active_cycles = []
    primary_cycle = None
    
    for cycle in GANN_CYCLES:
        days_to_cycle = cycle - (days_since_reference % cycle)
        if days_to_cycle <= 0:
            days_to_cycle += cycle
        
        cycle_date = today + timedelta(days=days_to_cycle)
        
        # Confidence based on proximity to clean cycle
        cycle_alignment = 1 - (abs(days_since_reference % cycle) / cycle)
        cycle_confidence = 0.3 + (0.4 * cycle_alignment)
        
        # Higher confidence for major cycles (90, 180, 360)
        if cycle in [90, 180, 360]:
            cycle_confidence += 0.15
        
        active_cycles.append({
            "cycle_days": cycle,
            "days_to_pivot": days_to_cycle,
            "pivot_date": cycle_date.strftime("%Y-%m-%d"),
            "confidence": round(cycle_confidence, 2),
            "is_major": cycle in [90, 180, 360]
        })
        
        # Select primary cycle (next major cycle with highest confidence)
        if primary_cycle is None or (cycle in [90, 180, 360] and days_to_cycle < primary_cycle["days_to_pivot"]):
            primary_cycle = active_cycles[-1]
    
    # Sort by days to pivot
    active_cycles.sort(key=lambda x: x["days_to_pivot"])
    
    # Use primary cycle for forecast
    if primary_cycle:
        next_pivot_date = primary_cycle["pivot_date"]
        days_to_pivot = primary_cycle["days_to_pivot"]
        base_confidence = primary_cycle["confidence"]
    else:
        next_pivot_date = (today + timedelta(days=90)).strftime("%Y-%m-%d")
        days_to_pivot = 90
        base_confidence = 0.40
    
    # Calculate price range using ATR and Gann bounds
    atr_projection = atr * np.sqrt(days_to_pivot)  # Volatility scales with sqrt of time
    
    # Constrain with Gann levels
    gann_3_8 = weekly_gann.levels.get("3_8", current_price * 0.9)
    gann_5_8 = weekly_gann.levels.get("5_8", current_price * 1.1)
    
    raw_low = current_price - atr_projection
    raw_high = current_price + atr_projection
    
    # Cap with Gann bounds
    probable_low = max(raw_low, gann_3_8 * 0.95)
    probable_high = min(raw_high, gann_5_8 * 1.05)
    
    # Build cycle origin transparency
    cycle_origin = {
        "reference_type": reference_type,
        "reference_date": str(reference_date) if reference_date else "Unknown",
        "reference_price": float(reference_price),
        "days_since_reference": days_since_reference,
        "primary_cycle_days": primary_cycle["cycle_days"] if primary_cycle else 90,
        "calculation_method": "Gann time cycles from last major extreme",
        "confidence_factors": [
            f"Cycle alignment: {primary_cycle['confidence']:.0%}" if primary_cycle else "Default cycle",
            f"Major cycle bonus: +15%" if primary_cycle and primary_cycle.get("is_major") else "Minor cycle",
            f"Days since {reference_type}: {days_since_reference}"
        ]
    }
    
    # Suppress if confidence below threshold
    suppressed = base_confidence < confidence_threshold
    suppression_reason = f"Confidence {base_confidence:.0%} below {confidence_threshold:.0%} threshold" if suppressed else None
    
    # Confidence level label
    if base_confidence >= 0.70:
        confidence_level = "HIGH"
    elif base_confidence >= 0.50:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"
    
    return TimeForecast(
        next_pivot_date=next_pivot_date,
        days_to_pivot=days_to_pivot,
        pivot_type=expected_pivot_type,
        confidence=round(base_confidence, 2),
        confidence_level=confidence_level,
        probable_price_low=round(probable_low, 2),
        probable_price_high=round(probable_high, 2),
        cycle_origin=cycle_origin,
        active_cycles=active_cycles[:5],  # Top 5 cycles
        suppressed=suppressed,
        suppression_reason=suppression_reason
    )

# ============================================================
# TIMEFRAME ANALYSIS (Complete)
# ============================================================

def analyze_timeframe(df: pd.DataFrame, tf_name: str, tf_config: TimeframeConfig) -> TimeframeAnalysis:
    """
    Complete analysis for a single timeframe.
    Returns all indicators, signals, and classifications.
    """
    if len(df) < tf_config.min_bars:
        raise ValueError(f"Insufficient data for {tf_name}: {len(df)} bars < {tf_config.min_bars} required")
    
    current_price = df['close'].iloc[-1]
    
    # Calculate indicators
    rsi_series = calculate_rsi(df['close'])
    rsi = rsi_series.iloc[-1]
    
    macd_line, signal_line, histogram = calculate_macd(df['close'])
    macd_hist = histogram.iloc[-1]
    
    adx = calculate_adx(df).iloc[-1]
    atr_series = calculate_atr(df)
    atr = atr_series.iloc[-1]
    atr_pct = (atr / current_price) * 100
    
    # Ichimoku
    ichimoku_data = calculate_ichimoku(df)
    ichimoku = analyze_ichimoku(df, ichimoku_data)
    
    # Gann levels (timeframe-specific)
    gann = calculate_gann_levels(df, tf_config)
    
    # Volume
    if 'volume' in df.columns and len(df) >= 20:
        recent_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].tail(20).mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
    else:
        volume_ratio = 1.0
    
    # Signal counting
    bullish_signals = 0
    bearish_signals = 0
    signal_details = {}
    
    # RSI signals
    if rsi < RSI_OVERSOLD:
        bullish_signals += 1
        signal_details["RSI"] = f"OVERSOLD ({rsi:.1f})"
    elif rsi > RSI_OVERBOUGHT:
        bearish_signals += 1
        signal_details["RSI"] = f"OVERBOUGHT ({rsi:.1f})"
    else:
        signal_details["RSI"] = f"NEUTRAL ({rsi:.1f})"
    
    # MACD signals
    if macd_hist > 0:
        bullish_signals += 1
        signal_details["MACD"] = "BULLISH"
    else:
        bearish_signals += 1
        signal_details["MACD"] = "BEARISH"
    
    # Ichimoku signals
    if ichimoku.tk_cross == "BULLISH":
        bullish_signals += 1
    elif ichimoku.tk_cross == "BEARISH":
        bearish_signals += 1
    signal_details["TK_CROSS"] = ichimoku.tk_cross
    
    if ichimoku.price_vs_cloud == "ABOVE":
        bullish_signals += 1
    elif ichimoku.price_vs_cloud == "BELOW":
        bearish_signals += 1
    signal_details["CLOUD"] = ichimoku.price_vs_cloud
    
    # Gann 50% signal
    gann_50 = gann.levels.get("4_8", current_price)
    if current_price > gann_50:
        bullish_signals += 1
        signal_details["GANN_50"] = f"ABOVE (${gann_50:,.0f})"
    else:
        bearish_signals += 1
        signal_details["GANN_50"] = f"BELOW (${gann_50:,.0f})"
    
    # Trend strength classification
    if adx > ADX_VERY_STRONG:
        trend_strength = "VERY_STRONG"
    elif adx > ADX_STRONG_TREND:
        trend_strength = "STRONG"
    else:
        trend_strength = "WEAK"
    
    # Direction determination (skip cloud for monthly due to displacement)
    if tf_name == "1M":
        # Monthly: weight TK cross more, ignore cloud
        if bullish_signals > bearish_signals + 1:
            direction = Direction.BULLISH
        elif bearish_signals > bullish_signals + 1:
            direction = Direction.BEARISH
        else:
            direction = Direction.NEUTRAL
    else:
        if bullish_signals > bearish_signals:
            direction = Direction.BULLISH
        elif bearish_signals > bullish_signals:
            direction = Direction.BEARISH
        else:
            direction = Direction.NEUTRAL
    
    # Confidence based on signal alignment
    total_signals = bullish_signals + bearish_signals
    if total_signals > 0:
        alignment_ratio = abs(bullish_signals - bearish_signals) / total_signals
        if alignment_ratio > 0.6:
            confidence = Confidence.HIGH
        elif alignment_ratio > 0.3:
            confidence = Confidence.MEDIUM
        else:
            confidence = Confidence.LOW
    else:
        confidence = Confidence.NONE
    
    # State name (Enneagram-style)
    state_name = determine_state_name(rsi, adx, direction, trend_strength)
    
    return TimeframeAnalysis(
        name=tf_name,
        direction=direction,
        confidence=confidence,
        state_name=state_name,
        rsi=round(rsi, 2),
        adx=round(adx, 2),
        trend_strength=trend_strength,
        ichimoku=ichimoku,
        gann=gann,
        bullish_signals=bullish_signals,
        bearish_signals=bearish_signals,
        signal_details=signal_details,
        volume_ratio=round(volume_ratio, 2),
        atr=round(atr, 2),
        atr_pct=round(atr_pct, 2)
    )

def determine_state_name(rsi: float, adx: float, direction: Direction, trend_strength: str) -> str:
    """Determine Enneagram-style state name based on indicators."""
    if rsi < 25:
        return "Capitulation"
    elif rsi < 35:
        return "Fear"
    elif rsi > 75:
        return "Euphoria"
    elif rsi > 65:
        return "Greed"
    elif adx > 50 and direction == Direction.BULLISH:
        return "Expansion"
    elif adx > 50 and direction == Direction.BEARISH:
        return "Contraction"
    elif adx < 20:
        return "Consolidation"
    elif direction == Direction.BULLISH:
        return "Accumulation"
    elif direction == Direction.BEARISH:
        return "Distribution"
    else:
        return "Transition"

# ============================================================
# MULTI-TIMEFRAME CONSENSUS
# ============================================================

def calculate_mtf_consensus(
    timeframes: Dict[str, TimeframeAnalysis],
    regime: RegimeAnalysis
) -> Dict:
    """
    Calculate weighted consensus across all timeframes.
    
    Returns:
    - primary_direction: Weighted direction
    - weighted_score: -100 to +100
    - alignment: X/4 timeframes agreeing
    - confidence_level: HIGH/MEDIUM/LOW
    - conflicts: List of timeframe conflicts
    """
    weighted_bullish = 0
    weighted_bearish = 0
    alignment_count = 0
    conflicts = []
    
    # First pass: calculate weighted scores
    for tf_name, analysis in timeframes.items():
        weight = TIMEFRAME_CONFIGS[tf_name].weight
        
        if analysis.direction == Direction.BULLISH:
            weighted_bullish += weight * analysis.bullish_signals
        elif analysis.direction == Direction.BEARISH:
            weighted_bearish += weight * analysis.bearish_signals
    
    # Determine primary direction
    if weighted_bullish > weighted_bearish * 1.2:
        primary_direction = Direction.BULLISH
    elif weighted_bearish > weighted_bullish * 1.2:
        primary_direction = Direction.BEARISH
    else:
        primary_direction = Direction.NEUTRAL
    
    # Count alignment
    for tf_name, analysis in timeframes.items():
        if analysis.direction == primary_direction:
            alignment_count += 1
        elif analysis.direction != Direction.NEUTRAL and primary_direction != Direction.NEUTRAL:
            conflicts.append({
                "timeframe": tf_name,
                "expected": primary_direction.value,
                "actual": analysis.direction.value,
                "weight": TIMEFRAME_CONFIGS[tf_name].weight
            })
    
    # Calculate weighted score (-100 to +100)
    total_weight = weighted_bullish + weighted_bearish
    if total_weight > 0:
        weighted_score = int(((weighted_bullish - weighted_bearish) / total_weight) * 100)
    else:
        weighted_score = 0
    
    # Confidence level
    if alignment_count >= 4 and abs(weighted_score) > 60:
        confidence_level = Confidence.HIGH
    elif alignment_count >= 3 and abs(weighted_score) > 40:
        confidence_level = Confidence.MEDIUM
    else:
        confidence_level = Confidence.LOW
    
    # Apply regime override to confidence
    if regime.override_active:
        if confidence_level == Confidence.HIGH:
            confidence_level = Confidence.MEDIUM
        confidence_note = f"Downgraded due to {regime.regime.value} regime"
    else:
        confidence_note = None
    
    return {
        "primary_direction": primary_direction.value,
        "weighted_score": weighted_score,
        "alignment": f"{alignment_count}/4",
        "alignment_count": alignment_count,
        "confidence_level": confidence_level.value,
        "confidence_note": confidence_note,
        "conflicts": conflicts,
        "weighted_bullish": round(weighted_bullish, 2),
        "weighted_bearish": round(weighted_bearish, 2)
    }

# ============================================================
# TRADE SETUP GENERATION
# ============================================================

def generate_trade_setups(
    current_price: float,
    timeframes: Dict[str, TimeframeAnalysis],
    consensus: Dict,
    regime: RegimeAnalysis,
    capitulation: CapitulationAnalysis,
    weekly_gann: GannLevels
) -> List[Dict]:
    """
    Generate prioritized trade setups based on analysis.
    Ordered by confidence (HIGH → MEDIUM → LOW).
    """
    setups = []
    
    primary_direction = consensus["primary_direction"]
    weekly_kijun = timeframes["1W"].ichimoku.kijun_value
    weekly_cloud_top = timeframes["1W"].ichimoku.cloud_top
    weekly_cloud_bottom = timeframes["1W"].ichimoku.cloud_bottom
    daily_atr = timeframes["1D"].atr
    
    gann_3_8 = weekly_gann.levels.get("3_8", current_price * 0.95)
    gann_5_8 = weekly_gann.levels.get("5_8", current_price * 1.05)
    gann_6_8 = weekly_gann.levels.get("6_8", current_price * 1.10)
    gann_2_8 = weekly_gann.levels.get("2_8", current_price * 0.90)
    
    # Setup 1: Primary direction trade
    if primary_direction == "BULLISH" and regime.allows_long:
        entry = current_price
        stop = min(gann_3_8, weekly_cloud_bottom) - daily_atr
        tp1 = gann_5_8
        tp2 = weekly_kijun if weekly_kijun > current_price else gann_6_8
        tp3 = weekly_cloud_top if weekly_cloud_top > tp2 else gann_6_8 * 1.05
        
        rr = (tp1 - entry) / (entry - stop) if entry > stop else 0
        
        # Confidence based on alignment and regime
        if consensus["alignment_count"] >= 3 and not regime.override_active:
            confidence = "HIGH"
            size = 1.0
        elif consensus["alignment_count"] >= 2:
            confidence = "MEDIUM"
            size = 0.5
        else:
            confidence = "LOW"
            size = 0.25
        
        setups.append({
            "id": 1,
            "type": "PRIMARY",
            "direction": "LONG",
            "confidence": confidence,
            "trigger": f"Price holds above ${gann_3_8:,.0f} (Gann 3/8)",
            "entry": round(entry, 2),
            "stop_loss": round(stop, 2),
            "tp1": round(tp1, 2),
            "tp2": round(tp2, 2),
            "tp3": round(tp3, 2),
            "rr_ratio": round(rr, 2),
            "position_size": min(size, regime.position_size_cap),
            "notes": f"Primary bullish setup. Alignment: {consensus['alignment']}",
            "invalidation": f"Daily close below ${stop:,.0f}"
        })
    
    elif primary_direction == "BEARISH" and regime.allows_short:
        entry = current_price
        stop = max(gann_5_8, weekly_cloud_top) + daily_atr
        tp1 = gann_3_8
        tp2 = gann_2_8
        tp3 = weekly_gann.low * 1.02
        
        rr = (entry - tp1) / (stop - entry) if stop > entry else 0
        
        if consensus["alignment_count"] >= 3 and not regime.override_active:
            confidence = "HIGH"
            size = 1.0
        elif consensus["alignment_count"] >= 2:
            confidence = "MEDIUM"
            size = 0.5
        else:
            confidence = "LOW"
            size = 0.25
        
        setups.append({
            "id": 1,
            "type": "PRIMARY",
            "direction": "SHORT",
            "confidence": confidence,
            "trigger": f"Price rejects at ${gann_5_8:,.0f} (Gann 5/8)",
            "entry": round(entry, 2),
            "stop_loss": round(stop, 2),
            "tp1": round(tp1, 2),
            "tp2": round(tp2, 2),
            "tp3": round(tp3, 2),
            "rr_ratio": round(rr, 2),
            "position_size": min(size, regime.position_size_cap),
            "notes": f"Primary bearish setup. Alignment: {consensus['alignment']}",
            "invalidation": f"Daily close above ${stop:,.0f}"
        })
    
    # Setup 2: Counter-trend at capitulation
    if capitulation.is_capitulation and regime.allows_long:
        entry = min(gann_3_8, weekly_cloud_bottom)
        stop = gann_2_8 - daily_atr
        tp1 = current_price * 1.05
        tp2 = gann_5_8
        tp3 = weekly_kijun
        
        rr = (tp1 - entry) / (entry - stop) if entry > stop else 0
        
        confidence = "MEDIUM" if capitulation.status == "CONFIRMED" else "LOW"
        
        setups.append({
            "id": 2,
            "type": "COUNTER_TREND",
            "direction": "LONG",
            "confidence": confidence,
            "trigger": f"Capitulation bounce from ${entry:,.0f} with volume confirmation",
            "entry": round(entry, 2),
            "stop_loss": round(stop, 2),
            "tp1": round(tp1, 2),
            "tp2": round(tp2, 2),
            "tp3": round(tp3, 2),
            "rr_ratio": round(rr, 2),
            "position_size": 0.25,  # Always conservative for counter-trend
            "notes": f"Counter-trend long at capitulation. Status: {capitulation.status}",
            "invalidation": f"Weekly close below ${stop:,.0f}",
            "requires": ["Volume spike >2x", "RSI divergence preferred"]
        })
    
    # Setup 3: Wait setup (when signals conflict)
    if consensus["alignment_count"] < 2 or len(consensus["conflicts"]) >= 2:
        setups.append({
            "id": 3,
            "type": "WAIT",
            "direction": "FLAT",
            "confidence": "NONE",
            "trigger": "Timeframe conflict - wait for alignment",
            "entry": None,
            "stop_loss": None,
            "tp1": None,
            "tp2": None,
            "tp3": None,
            "rr_ratio": None,
            "position_size": 0,
            "notes": f"Conflicting signals across timeframes. Wait for: {[c['timeframe'] for c in consensus['conflicts']]} to align",
            "invalidation": None,
            "wait_for": [
                f"Weekly direction matches daily",
                f"Price breaks above ${gann_5_8:,.0f} (bullish) or below ${gann_3_8:,.0f} (bearish)"
            ]
        })
    
    # Sort by confidence priority
    confidence_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}
    setups.sort(key=lambda x: confidence_order.get(x["confidence"], 4))
    
    return setups

# ============================================================
# MAIN SIGNAL GENERATOR
# ============================================================

def generate_mtf_signal(
    df_1d: pd.DataFrame,
    df_3d: pd.DataFrame,
    df_1w: pd.DataFrame,
    df_1m: pd.DataFrame,
    symbol: str = "BTCUSDT"
) -> Dict:
    """
    Main entry point for MTF signal generation.
    
    Returns complete signal payload with:
    - Multi-timeframe analysis
    - Capitulation detection
    - Regime with override logic
    - Time forecast with transparency
    - Prioritized trade setups
    """
    current_price = df_1d['close'].iloc[-1]
    signal_date = df_1d['timestamp'].iloc[-1] if 'timestamp' in df_1d.columns else datetime.now().isoformat()
    
    # Analyze each timeframe
    timeframes = {}
    try:
        timeframes["1D"] = analyze_timeframe(df_1d, "1D", TIMEFRAME_CONFIGS["1D"])
        timeframes["3D"] = analyze_timeframe(df_3d, "3D", TIMEFRAME_CONFIGS["3D"])
        timeframes["1W"] = analyze_timeframe(df_1w, "1W", TIMEFRAME_CONFIGS["1W"])
        timeframes["1M"] = analyze_timeframe(df_1m, "1M", TIMEFRAME_CONFIGS["1M"])
    except Exception as e:
        logger.error(f"Timeframe analysis error: {e}")
        raise
    
    # Capitulation detection (uses weekly data)
    weekly_rsi = timeframes["1W"].rsi
    weekly_gann = timeframes["1W"].gann
    
    capitulation = detect_capitulation(
        df_weekly=df_1w,
        df_daily=df_1d,
        weekly_rsi=weekly_rsi,
        weekly_gann=weekly_gann,
        current_price=current_price
    )
    
    # Regime determination with override logic
    regime = determine_regime(
        df_daily=df_1d,
        weekly_analysis=timeframes["1W"],
        capitulation=capitulation,
        current_price=current_price
    )
    
    # MTF consensus
    consensus = calculate_mtf_consensus(timeframes, regime)
    
    # Time forecast with transparency
    daily_atr = timeframes["1D"].atr
    time_forecast = calculate_time_forecast(
        df_daily=df_1d,
        current_price=current_price,
        atr=daily_atr,
        weekly_gann=weekly_gann
    )
    
    # Generate trade setups
    trade_setups = generate_trade_setups(
        current_price=current_price,
        timeframes=timeframes,
        consensus=consensus,
        regime=regime,
        capitulation=capitulation,
        weekly_gann=weekly_gann
    )
    
    # Build response
    response = {
        "status": "success",
        "symbol": symbol,
        "current_price": round(current_price, 2),
        "signal_date": str(signal_date),
        "timestamp": datetime.now().isoformat(),
        "version": "5.0.6",
        
        # Timeframe analysis
        "timeframes": {
            tf_name: {
                "direction": analysis.direction.value,
                "confidence": analysis.confidence.value,
                "state_name": analysis.state_name,
                "rsi": analysis.rsi,
                "adx": analysis.adx,
                "trend_strength": analysis.trend_strength,
                "volume_ratio": analysis.volume_ratio,
                "atr": analysis.atr,
                "atr_pct": analysis.atr_pct,
                "bullish_signals": analysis.bullish_signals,
                "bearish_signals": analysis.bearish_signals,
                "signal_details": analysis.signal_details,
                "ichimoku": {
                    "tk_cross": analysis.ichimoku.tk_cross,
                    "price_vs_cloud": analysis.ichimoku.price_vs_cloud,
                    "cloud_color": analysis.ichimoku.cloud_color,
                    "kijun": analysis.ichimoku.kijun_value,
                    "tenkan": analysis.ichimoku.tenkan_value,
                    "cloud_top": analysis.ichimoku.cloud_top,
                    "cloud_bottom": analysis.ichimoku.cloud_bottom,
                    "kijun_flat": analysis.ichimoku.kijun_flat
                },
                "gann": {
                    "high": analysis.gann.high,
                    "low": analysis.gann.low,
                    "high_date": analysis.gann.high_date,
                    "low_date": analysis.gann.low_date,
                    "range": analysis.gann.range_value,
                    "range_pct": analysis.gann.range_pct,
                    "lookback_bars": analysis.gann.lookback_bars,
                    "levels": analysis.gann.levels
                }
            }
            for tf_name, analysis in timeframes.items()
        },
        
        # Consensus
        "consensus": consensus,
        
        # Regime with override info
        "regime": {
            "current": regime.regime.value,
            "strength": regime.regime_strength,
            "override_active": regime.override_active,
            "override_reason": regime.override_reason,
            "original_regime": regime.original_regime.value if regime.original_regime else None,
            "allows_short": regime.allows_short,
            "allows_long": regime.allows_long,
            "position_size_cap": regime.position_size_cap,
            "warnings": regime.warnings
        },
        
        # Capitulation analysis
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
            },
            "divergence": capitulation.divergence_details
        },
        
        # Time forecast with transparency
        "time_forecast": {
            "next_pivot_date": time_forecast.next_pivot_date,
            "days_to_pivot": time_forecast.days_to_pivot,
            "pivot_type": time_forecast.pivot_type,
            "confidence": time_forecast.confidence,
            "confidence_level": time_forecast.confidence_level,
            "probable_price_range": {
                "low": time_forecast.probable_price_low,
                "high": time_forecast.probable_price_high
            },
            "cycle_origin": time_forecast.cycle_origin,
            "active_cycles": time_forecast.active_cycles,
            "suppressed": time_forecast.suppressed,
            "suppression_reason": time_forecast.suppression_reason
        },
        
        # Trade setups (ordered by confidence)
        "trade_setups": trade_setups,
        
        # Primary setup summary
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
    
    return response


# ============================================================
# FLASK ENDPOINT (Integration)
# ============================================================

from flask import Flask, jsonify
import ccxt

app = Flask(__name__)

def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """Fetch OHLCV data from exchange."""
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def resample_ohlcv(df_1d: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample daily OHLCV to higher timeframes."""
    df = df_1d.copy()
    df.set_index('timestamp', inplace=True)
    
    resample_map = {
        '3D': '3D',
        '1W': 'W',
        '1M': 'ME'
    }
    
    rule = resample_map.get(target_tf, '1D')
    
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    resampled.reset_index(inplace=True)
    return resampled

@app.route('/signal/daily', methods=['GET'])
def get_daily_signal():
    """Main endpoint for MTF signal generation."""
    try:
        symbol = "BTC/USDT"
        
        # Fetch daily data
        df_1d = fetch_ohlcv(symbol, '1d', limit=500)
        
        # Resample to higher timeframes
        df_3d = resample_ohlcv(df_1d, '3D')
        df_1w = resample_ohlcv(df_1d, '1W')
        df_1m = resample_ohlcv(df_1d, '1M')
        
        # Generate MTF signal
        signal = generate_mtf_signal(
            df_1d=df_1d,
            df_3d=df_3d,
            df_1w=df_1w,
            df_1m=df_1m,
            symbol="BTCUSDT"
        )
        
        return jsonify(signal)
    
    except Exception as e:
        logger.error(f"Signal generation error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "version": "5.0.6"
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
