"""
LUXOR V7 PRANA RUNTIME - MULTI-TIMEFRAME EDITION
Version: 5.0.2
Fixed: Gann calculations, weighted scoring, direction classification, alignment count, R:R ratio
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import the core system
from luxor_v7_prana import LuxorV7PranaSystem

# Configuration
INITIAL_CAPITAL = 100000
API_HOST = "0.0.0.0"
API_PORT = 8000

# Initialize FastAPI
app = FastAPI(
    title="LUXOR V7 PRANA RUNTIME - MTF EDITION",
    version="5.0.2",
    description="Multi-Timeframe Gann + Ichimoku + Enneagram Analysis"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Luxor System
luxor = LuxorV7PranaSystem(initial_capital=INITIAL_CAPITAL)

# ============================================================
# CONFIGURATION
# ============================================================

TIMEFRAME_WEIGHTS = {
    '1M': 40,
    '1W': 35,
    '3D': 15,
    '1D': 10
}

ICHIMOKU_PARAMS = {
    '1D': (9, 26, 52),
    '3D': (9, 26, 52),
    '1W': (9, 26, 26),
    '1M': (9, 26, 52)
}

ENNEAGRAM_STATES = {
    1: {'name': 'Initiation', 'bias': 'NEUTRAL', 'phase': 'New cycle beginning'},
    2: {'name': 'Accumulation', 'bias': 'BULLISH', 'phase': 'Smart money buying'},
    3: {'name': 'Markup', 'bias': 'BULLISH', 'phase': 'Trend acceleration'},
    4: {'name': 'Retracement', 'bias': 'NEUTRAL', 'phase': 'Healthy pullback'},
    5: {'name': 'Capitulation', 'bias': 'BEARISH', 'phase': 'Panic selling / washout'},
    6: {'name': 'Recovery', 'bias': 'BULLISH', 'phase': 'Bottom formation'},
    7: {'name': 'Expansion', 'bias': 'BULLISH', 'phase': 'Strong trend continuation'},
    8: {'name': 'Distribution', 'bias': 'BEARISH', 'phase': 'Smart money selling'},
    9: {'name': 'Completion', 'bias': 'NEUTRAL', 'phase': 'Cycle ending'}
}

ARROW_MEANINGS = {
    (1, 'STRESS'): "Initiation failing - Loss of momentum expected",
    (1, 'GROWTH'): "Initiation strengthening - Trend developing",
    (2, 'STRESS'): "Accumulation under pressure - Possible failed bottom",
    (2, 'GROWTH'): "Accumulation confirmed - Markup phase coming",
    (3, 'STRESS'): "Markup losing steam - Distribution starting",
    (3, 'GROWTH'): "Markup accelerating - Strong trend",
    (4, 'STRESS'): "Retracement deepening - Watch for breakdown",
    (4, 'GROWTH'): "Retracement ending - Trend resuming",
    (5, 'STRESS'): "Capitulation intensifying - Crash mode",
    (5, 'GROWTH'): "Capitulation ending - Sharp reversal rally expected",
    (6, 'STRESS'): "Recovery failing - Lower low coming",
    (6, 'GROWTH'): "Recovery confirmed - New uptrend starting",
    (7, 'STRESS'): "Expansion peaking - Pullback expected",
    (7, 'GROWTH'): "Expansion continuing - Ride the trend",
    (8, 'STRESS'): "Distribution accelerating - Breakdown imminent",
    (8, 'GROWTH'): "Distribution pausing - One more high possible",
    (9, 'STRESS'): "Completion turning bearish - New downtrend",
    (9, 'GROWTH'): "Completion turning bullish - New uptrend"
}

# ============================================================
# HELPER FUNCTIONS - TYPE CONVERSION
# ============================================================

def to_native(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization"""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    if isinstance(obj, (np.bool_, )):
        return bool(obj)
    if isinstance(obj, (np.integer, )):
        return int(obj)
    if isinstance(obj, (np.floating, )):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [to_native(v) for v in obj.tolist()]
    if pd.isna(obj):
        return None
    return obj

def safe_float(val, default=0.0):
    """Safely convert to float"""
    try:
        if val is None or pd.isna(val):
            return default
        return float(val)
    except:
        return default

def safe_int(val, default=0):
    """Safely convert to int"""
    try:
        if val is None or pd.isna(val):
            return default
        return int(val)
    except:
        return default

def safe_round(val, decimals=2):
    """Safely round a number"""
    try:
        if val is None or pd.isna(val):
            return 0.0
        return round(float(val), decimals)
    except:
        return 0.0

# ============================================================
# DATA RESAMPLING
# ============================================================

def resample_ohlcv(df, timeframe):
    """Resample daily OHLCV data to higher timeframes"""
    if df is None or df.empty:
        return None
    
    freq_map = {
        '1D': '1D',
        '3D': '3D',
        '1W': 'W',
        '1M': 'M'  # Changed from 'ME' for pandas 1.x compatibility
    }
    
    freq = freq_map.get(timeframe, '1D')
    
    try:
        df_copy = df.copy()
        
        # Ensure datetime index
        if 'date' in df_copy.columns:
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            df_copy.set_index('date', inplace=True)
        
        # Resample
        resampled = df_copy.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        resampled = resampled.reset_index()
        resampled = resampled.rename(columns={'index': 'date'})
        
        return resampled
    except Exception as e:
        print(f"Resample error for {timeframe}: {e}")
        return None

# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_rsi(df, period=14):
    """Calculate RSI"""
    if df is None or len(df) < period + 1:
        return 50.0
    
    close = df['close']
    delta = close.diff()
    
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return safe_float(rsi.iloc[-1], 50.0)

def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD and histogram"""
    if df is None or len(df) < slow + signal:
        return 0.0, 0.0, 0.0
    
    close = df['close']
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return (
        safe_float(macd_line.iloc[-1]),
        safe_float(signal_line.iloc[-1]),
        safe_float(histogram.iloc[-1])
    )

def calculate_atr(df, period=14):
    """Calculate ATR"""
    if df is None or len(df) < period + 1:
        return 0.0
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return safe_float(atr.iloc[-1])

def calculate_adx(df, period=14):
    """Calculate ADX for trend strength"""
    if df is None or len(df) < period * 2:
        return 25.0, 'MODERATE'
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate +DM and -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Calculate TR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smooth
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
    adx = dx.rolling(window=period).mean()
    
    adx_value = safe_float(adx.iloc[-1], 25.0)
    
    if adx_value >= 50:
        strength = 'STRONG'
    elif adx_value >= 25:
        strength = 'MODERATE'
    else:
        strength = 'WEAK'
    
    return adx_value, strength

def calculate_sma(df, period=200):
    """Calculate Simple Moving Average"""
    if df is None or len(df) < period:
        return None
    return safe_float(df['close'].rolling(window=period).mean().iloc[-1])

# ============================================================
# ICHIMOKU CALCULATION
# ============================================================

def calculate_ichimoku(df, params=(9, 26, 52)):
    """Calculate Ichimoku Cloud components"""
    if df is None or len(df) < params[2] + 26:
        return {
            'tenkan': 0, 'kijun': 0, 'senkou_a': 0, 'senkou_b': 0,
            'cloud_top': 0, 'cloud_bottom': 0, 'chikou': 0,
            'cloud_signal': 'NEUTRAL', 'tk_cross': 'NEUTRAL', 'kijun_flat': False
        }
    
    tenkan_period, kijun_period, senkou_period = params
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Tenkan-sen (Conversion Line)
    tenkan = (high.rolling(window=tenkan_period).max() + 
              low.rolling(window=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line)
    kijun = (high.rolling(window=kijun_period).max() + 
             low.rolling(window=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A)
    senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)
    
    # Senkou Span B (Leading Span B)
    senkou_b = ((high.rolling(window=senkou_period).max() + 
                 low.rolling(window=senkou_period).min()) / 2).shift(kijun_period)
    
    # Current values
    current_close = safe_float(close.iloc[-1])
    current_tenkan = safe_float(tenkan.iloc[-1])
    current_kijun = safe_float(kijun.iloc[-1])
    current_senkou_a = safe_float(senkou_a.iloc[-1])
    current_senkou_b = safe_float(senkou_b.iloc[-1])
    
    cloud_top = max(current_senkou_a, current_senkou_b)
    cloud_bottom = min(current_senkou_a, current_senkou_b)
    
    # Cloud signal
    if current_close > cloud_top:
        cloud_signal = 'BULLISH'
    elif current_close < cloud_bottom:
        cloud_signal = 'BEARISH'
    else:
        cloud_signal = 'NEUTRAL'
    
    # TK Cross
    prev_tenkan = safe_float(tenkan.iloc[-2]) if len(tenkan) > 1 else current_tenkan
    prev_kijun = safe_float(kijun.iloc[-2]) if len(kijun) > 1 else current_kijun
    
    if current_tenkan > current_kijun and prev_tenkan <= prev_kijun:
        tk_cross = 'BULLISH'
    elif current_tenkan < current_kijun and prev_tenkan >= prev_kijun:
        tk_cross = 'BEARISH'
    elif current_tenkan > current_kijun:
        tk_cross = 'BULLISH'
    elif current_tenkan < current_kijun:
        tk_cross = 'BEARISH'
    else:
        tk_cross = 'NEUTRAL'
    
    # Kijun flat detection (potential support/resistance)
    kijun_flat = False
    if len(kijun) >= 5:
        recent_kijun = kijun.iloc[-5:]
        kijun_flat = recent_kijun.std() < (current_kijun * 0.001)
    
    return {
        'tenkan': current_tenkan,
        'kijun': current_kijun,
        'senkou_a': current_senkou_a,
        'senkou_b': current_senkou_b,
        'cloud_top': cloud_top,
        'cloud_bottom': cloud_bottom,
        'chikou': current_close,
        'cloud_signal': cloud_signal,
        'tk_cross': tk_cross,
        'kijun_flat': kijun_flat
    }

# ============================================================
# GANN CALCULATIONS - FIXED v5.0.2
# ============================================================

def calculate_gann_levels_for_timeframe(df):
    """
    Calculate Gann Rule of Eighths for a specific timeframe
    Uses the HIGH and LOW of the entire timeframe dataset
    """
    if df is None or df.empty:
        return {
            'high': 0, 'low': 0, 'range': 0,
            '0_8': 0, '1_8': 0, '2_8': 0, '3_8': 0,
            '4_8': 0, '5_8': 0, '6_8': 0, '7_8': 0, '8_8': 0
        }
    
    high = safe_float(df['high'].max())
    low = safe_float(df['low'].min())
    range_val = high - low
    
    return {
        'high': high,
        'low': low,
        'range': range_val,
        '0_8': low,                           # 0% - Major Low
        '1_8': low + range_val * 0.125,       # 12.5%
        '2_8': low + range_val * 0.25,        # 25%
        '3_8': low + range_val * 0.375,       # 37.5%
        '4_8': low + range_val * 0.5,         # 50% - KEY LEVEL
        '5_8': low + range_val * 0.625,       # 62.5%
        '6_8': low + range_val * 0.75,        # 75%
        '7_8': low + range_val * 0.875,       # 87.5%
        '8_8': high                           # 100% - Major High
    }

def calculate_square_of_9(base_price):
    """Calculate Gann Square of 9 levels"""
    if base_price <= 0:
        return []
    
    sqrt_price = np.sqrt(base_price)
    levels = []
    
    for angle in [45, 90, 135, 180, 225, 270, 315, 360]:
        increment = angle / 180
        
        price_up = (sqrt_price + increment) ** 2
        price_down = (sqrt_price - increment) ** 2
        
        levels.append({
            'angle': angle,
            'price_up': safe_round(price_up),
            'price_down': safe_round(price_down),
            'distance_up_pct': safe_round((price_up - base_price) / base_price * 100),
            'distance_down_pct': safe_round((price_down - base_price) / base_price * 100)
        })
    
    return levels

# ============================================================
# ENNEAGRAM STATE IDENTIFICATION
# ============================================================

def identify_enneagram_state(df, rsi, macd_hist, volume_ratio=1.0):
    """Identify market state using Enneagram model"""
    if df is None or len(df) < 20:
        return 1, ENNEAGRAM_STATES[1]
    
    close = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-20]
    price_change_pct = ((close - prev_close) / prev_close) * 100
    
    # State determination logic
    if rsi < 30 and macd_hist < 0:
        state = 5  # Capitulation
    elif rsi < 35 and macd_hist > 0:
        state = 6  # Recovery
    elif rsi > 70 and macd_hist > 0:
        state = 3  # Markup
    elif rsi > 65 and macd_hist < 0:
        state = 8  # Distribution
    elif 40 <= rsi <= 60 and abs(macd_hist) < abs(df['close'].mean() * 0.01):
        if price_change_pct > 2:
            state = 2  # Accumulation
        elif price_change_pct < -2:
            state = 4  # Retracement
        else:
            state = 1  # Initiation
    elif rsi > 55 and macd_hist > 0:
        state = 7  # Expansion
    elif rsi < 45 and macd_hist < 0:
        state = 9  # Completion
    else:
        state = 1  # Default to Initiation
    
    return state, ENNEAGRAM_STATES[state]

def determine_arrow(state, rsi, macd_hist, prev_rsi=None):
    """Determine STRESS or GROWTH arrow"""
    if prev_rsi is None:
        prev_rsi = rsi
    
    # GROWTH: improving conditions
    # STRESS: deteriorating conditions
    
    if state in [2, 3, 6, 7]:  # Bullish states
        if rsi > prev_rsi and macd_hist > 0:
            arrow = 'GROWTH'
        else:
            arrow = 'STRESS'
    elif state in [5, 8, 9]:  # Bearish states
        if rsi < prev_rsi and macd_hist < 0:
            arrow = 'STRESS'
        else:
            arrow = 'GROWTH'
    else:  # Neutral states (1, 4)
        if macd_hist > 0:
            arrow = 'GROWTH'
        else:
            arrow = 'STRESS'
    
    meaning = ARROW_MEANINGS.get((state, arrow), "Transition in progress")
    
    return arrow, meaning

# ============================================================
# DIRECTION CLASSIFICATION - FIXED v5.0.2
# ============================================================

def classify_timeframe_direction(rsi, adx, cloud_signal, tk_cross, macd_hist, kijun_flat=False):
    """
    Classify direction for a timeframe based on multiple indicators
    Returns: direction, bullish_count, bearish_count
    
    FIXED: Now properly weights signals and handles oversold/overbought in trends
    """
    bullish = 0
    bearish = 0
    
    # 1. Cloud Signal (HEAVY WEIGHT - 2 points)
    if cloud_signal == 'BULLISH':
        bullish += 2
    elif cloud_signal == 'BEARISH':
        bearish += 2
    
    # 2. TK Cross (1 point)
    if tk_cross == 'BULLISH':
        bullish += 1
    elif tk_cross == 'BEARISH':
        bearish += 1
    
    # 3. MACD Histogram (1 point)
    if macd_hist > 0:
        bullish += 1
    elif macd_hist < 0:
        bearish += 1
    
    # 4. RSI with TREND CONTEXT (1 point)
    # In a downtrend (cloud bearish), oversold RSI confirms bearish, not bullish
    # In an uptrend (cloud bullish), overbought RSI confirms bullish, not bearish
    if cloud_signal == 'BEARISH':
        if rsi < 40:  # Oversold in downtrend = bearish momentum
            bearish += 1
        elif rsi > 60:  # Strength in downtrend = potential reversal
            bullish += 1
    elif cloud_signal == 'BULLISH':
        if rsi > 60:  # Overbought in uptrend = bullish momentum
            bullish += 1
        elif rsi < 40:  # Weakness in uptrend = potential reversal
            bearish += 1
    else:  # Neutral cloud
        if rsi > 60:
            bullish += 1
        elif rsi < 40:
            bearish += 1
    
    # 5. ADX shows trend STRENGTH, not direction
    # Strong trend (ADX > 25) amplifies the dominant signal
    if adx > 40:  # Very strong trend
        if bullish > bearish:
            bullish += 1
        elif bearish > bullish:
            bearish += 1
    
    # 6. Kijun Flat (potential support/resistance)
    if kijun_flat:
        # Flat Kijun often acts as magnet - slight neutral bias
        pass  # No change, but note it
    
    # Determine direction - need CLEAR majority
    total_signals = bullish + bearish
    
    if total_signals == 0:
        return 'NEUTRAL', 0, 0
    
    bullish_pct = bullish / total_signals * 100
    bearish_pct = bearish / total_signals * 100
    
    # Need > 60% for clear direction
    if bullish_pct >= 60:
        return 'BULLISH', bullish, bearish
    elif bearish_pct >= 60:
        return 'BEARISH', bullish, bearish
    else:
        return 'NEUTRAL', bullish, bearish

# ============================================================
# CONSENSUS CALCULATION - FIXED v5.0.2
# ============================================================

def calculate_consensus(timeframes):
    """
    Calculate weighted consensus across timeframes
    
    FIXED: 
    - Proper weighted contribution calculation
    - Correct alignment count
    - Primary direction from weighted scores
    """
    total_bullish_weight = 0
    total_bearish_weight = 0
    total_weight = 0
    
    directions = {}
    
    for tf_name, tf_data in timeframes.items():
        weight = tf_data['weight']
        direction = tf_data['direction']
        total_weight += weight
        
        directions[tf_name] = direction
        
        if direction == 'BULLISH':
            total_bullish_weight += weight
            tf_data['weighted_contribution'] = weight
        elif direction == 'BEARISH':
            total_bearish_weight += weight
            tf_data['weighted_contribution'] = weight
        else:  # NEUTRAL
            tf_data['weighted_contribution'] = 0
    
    # Calculate percentages
    bullish_pct = (total_bullish_weight / total_weight) * 100 if total_weight > 0 else 0
    bearish_pct = (total_bearish_weight / total_weight) * 100 if total_weight > 0 else 0
    neutral_pct = 100 - bullish_pct - bearish_pct
    
    # Determine primary direction
    if bullish_pct > bearish_pct and bullish_pct >= 40:
        primary_direction = 'BULLISH'
        weighted_score = bullish_pct
    elif bearish_pct > bullish_pct and bearish_pct >= 40:
        primary_direction = 'BEARISH'
        weighted_score = bearish_pct
    else:
        primary_direction = 'NEUTRAL'
        weighted_score = max(bullish_pct, bearish_pct)
    
    # Count alignment - how many TFs match primary direction
    if primary_direction == 'NEUTRAL':
        aligned_count = sum(1 for d in directions.values() if d == 'NEUTRAL')
    else:
        aligned_count = sum(1 for d in directions.values() if d == primary_direction)
    
    total_tfs = len(timeframes)
    alignment_str = f"{aligned_count}/{total_tfs}"
    
    # Alignment detail
    alignment_detail = ", ".join([f"{tf}:{dir}" for tf, dir in directions.items()])
    
    # Confidence level
    if weighted_score >= 70 and aligned_count >= 3:
        confidence = 'HIGH'
    elif weighted_score >= 50 and aligned_count >= 2:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'
    
    # Signal type
    if primary_direction == 'BULLISH' and confidence in ['HIGH', 'MEDIUM']:
        signal_type = 'BUY'
    elif primary_direction == 'BEARISH' and confidence in ['HIGH', 'MEDIUM']:
        signal_type = 'SELL'
    else:
        signal_type = 'WAIT'
    
    # Interpretation
    monthly_dir = directions.get('1M', 'NEUTRAL')
    weekly_dir = directions.get('1W', 'NEUTRAL')
    daily_dir = directions.get('1D', 'NEUTRAL')
    
    if monthly_dir == weekly_dir == daily_dir and monthly_dir != 'NEUTRAL':
        interpretation = f"Strong {monthly_dir.lower()} alignment across all timeframes. High conviction setup."
    elif monthly_dir != 'NEUTRAL' and weekly_dir == monthly_dir:
        interpretation = f"Monthly and weekly aligned {monthly_dir.lower()}. Strong trend."
    elif daily_dir != weekly_dir and weekly_dir != 'NEUTRAL':
        interpretation = f"Daily diverging from weekly trend. Potential counter-trend move or reversal starting."
    elif weekly_dir == 'BEARISH' and daily_dir == 'BULLISH':
        interpretation = "Daily bounce within weekly downtrend. Likely a bear market rally - sell opportunity."
    elif weekly_dir == 'BULLISH' and daily_dir == 'BEARISH':
        interpretation = "Daily pullback within weekly uptrend. Likely a buying opportunity."
    else:
        interpretation = "Mixed signals. Wait for clearer alignment before taking positions."
    
    return {
        'direction': primary_direction,
        'signal_type': signal_type,
        'weighted_bullish_pct': round(bullish_pct),
        'weighted_bearish_pct': round(bearish_pct),
        'weighted_neutral_pct': round(neutral_pct),
        'weighted_score': round(weighted_score),
        'confidence_level': confidence,
        'alignment_count': alignment_str,
        'alignment_detail': alignment_detail,
        'monthly_direction': monthly_dir,
        'weekly_direction': weekly_dir,
        'daily_direction': daily_dir,
        'interpretation': interpretation
    }

# ============================================================
# RISK:REWARD CALCULATION - NEW v5.0.2
# ============================================================

def calculate_risk_reward(entry, target, stop):
    """Calculate Risk:Reward ratio"""
    if entry == 0 or stop == entry:
        return 0.0, 0.0, 0.0
    
    reward = abs(target - entry)
    risk = abs(entry - stop)
    
    reward_pct = (reward / entry) * 100
    risk_pct = (risk / entry) * 100
    
    rr_ratio = reward / risk if risk > 0 else 0
    
    return safe_round(rr_ratio, 2), safe_round(reward_pct, 2), safe_round(risk_pct, 2)

# ============================================================
# PRICE TARGETS & STOP LOSS - FIXED v5.0.2
# ============================================================

def calculate_targets_and_stops(current_price, direction, gann_levels_by_tf, ichimoku_by_tf, atr):
    """
    Calculate TP and SL based on timeframe-specific levels
    
    FIXED: Proper source tracking and level selection
    """
    targets = {
        'tp1': None, 'tp1_sources': [],
        'tp2': None, 'tp2_sources': [],
        'tp3': None, 'tp3_sources': [],
        'stop_loss': None, 'stop_sources': []
    }
    
    resistance_levels = []
    support_levels = []
    
    # Collect levels from each timeframe
    for tf_name in ['1M', '1W', '3D', '1D']:
        gann = gann_levels_by_tf.get(tf_name, {})
        ichi = ichimoku_by_tf.get(tf_name, {})
        
        # Gann levels
        for level_name in ['3_8', '4_8', '5_8', '6_8', '7_8']:
            level_val = gann.get(level_name, 0)
            if level_val > 0:
                source = f"GANN_{level_name}_{tf_name}"
                if level_val > current_price:
                    resistance_levels.append({'price': level_val, 'source': source, 'tf': tf_name})
                else:
                    support_levels.append({'price': level_val, 'source': source, 'tf': tf_name})
        
        # Ichimoku levels
        cloud_top = ichi.get('cloud_top', 0)
        cloud_bottom = ichi.get('cloud_bottom', 0)
        kijun = ichi.get('kijun', 0)
        
        if cloud_top > current_price:
            resistance_levels.append({'price': cloud_top, 'source': f'CLOUD_TOP_{tf_name}', 'tf': tf_name})
        if cloud_bottom < current_price:
            support_levels.append({'price': cloud_bottom, 'source': f'CLOUD_BOTTOM_{tf_name}', 'tf': tf_name})
        if kijun > 0:
            if kijun > current_price:
                resistance_levels.append({'price': kijun, 'source': f'KIJUN_{tf_name}', 'tf': tf_name})
            else:
                support_levels.append({'price': kijun, 'source': f'KIJUN_{tf_name}', 'tf': tf_name})
    
    # Sort levels
    resistance_levels.sort(key=lambda x: x['price'])
    support_levels.sort(key=lambda x: x['price'], reverse=True)
    
    if direction == 'BULLISH' or direction == 'NEUTRAL':
        # Targets are resistance levels above price
        if len(resistance_levels) >= 1:
            targets['tp1'] = resistance_levels[0]['price']
            targets['tp1_sources'] = [resistance_levels[0]['source']]
        if len(resistance_levels) >= 2:
            targets['tp2'] = resistance_levels[1]['price']
            targets['tp2_sources'] = [resistance_levels[1]['source']]
        if len(resistance_levels) >= 3:
            targets['tp3'] = resistance_levels[2]['price']
            targets['tp3_sources'] = [resistance_levels[2]['source']]
        
        # Stop is support below price
        if len(support_levels) >= 1:
            targets['stop_loss'] = support_levels[0]['price']
            targets['stop_sources'] = [support_levels[0]['source']]
    
    else:  # BEARISH
        # Targets are support levels below price
        if len(support_levels) >= 1:
            targets['tp1'] = support_levels[0]['price']
            targets['tp1_sources'] = [support_levels[0]['source']]
        if len(support_levels) >= 2:
            targets['tp2'] = support_levels[1]['price']
            targets['tp2_sources'] = [support_levels[1]['source']]
        if len(support_levels) >= 3:
            targets['tp3'] = support_levels[2]['price']
            targets['tp3_sources'] = [support_levels[2]['source']]
        
        # Stop is resistance above price
        if len(resistance_levels) >= 1:
            targets['stop_loss'] = resistance_levels[0]['price']
            targets['stop_sources'] = [resistance_levels[0]['source']]
    
    # Fallback to ATR-based if no levels found
    if targets['tp1'] is None:
        targets['tp1'] = current_price + (atr * 2 if direction != 'BEARISH' else -atr * 2)
        targets['tp1_sources'] = ['ATR_2X']
    if targets['tp2'] is None:
        targets['tp2'] = current_price + (atr * 3 if direction != 'BEARISH' else -atr * 3)
        targets['tp2_sources'] = ['ATR_3X']
    if targets['tp3'] is None:
        targets['tp3'] = current_price + (atr * 4 if direction != 'BEARISH' else -atr * 4)
        targets['tp3_sources'] = ['ATR_4X']
    if targets['stop_loss'] is None:
        targets['stop_loss'] = current_price - (atr * 1.5 if direction != 'BEARISH' else -atr * 1.5)
        targets['stop_sources'] = ['ATR_1.5X']
    
    return targets

# ============================================================
# TIME FORECAST
# ============================================================

def calculate_time_forecast(df, current_price, atr):
    """Calculate time-based pivot forecasts using Gann cycles"""
    if df is None or df.empty:
        return {}
    
    last_date = pd.to_datetime(df['date'].iloc[-1])
    
    # Gann time cycles (in days)
    cycles = [30, 45, 60, 90, 120, 180, 360]
    
    # Find recent pivots
    highs = df.nlargest(5, 'high')
    lows = df.nsmallest(5, 'low')
    
    pivot_forecasts = []
    
    for cycle in cycles:
        # Project from recent high
        if not highs.empty:
            high_date = pd.to_datetime(highs['date'].iloc[0])
            projected_date = high_date + timedelta(days=cycle)
            if projected_date > last_date:
                days_from_now = (projected_date - last_date).days
                if 5 <= days_from_now <= 180:
                    pivot_forecasts.append({
                        'date': projected_date.strftime('%Y-%m-%d'),
                        'date_display': projected_date.strftime('%d/%m/%Y'),
                        'days_from_now': days_from_now,
                        'cycle': f'{cycle}D_CYCLE',
                        'from': 'HIGH',
                        'confidence': 50 if cycle in [45, 90, 180] else 30
                    })
    
    # Sort by days from now
    pivot_forecasts.sort(key=lambda x: x['days_from_now'])
    
    # Get primary pivot
    primary_pivot = pivot_forecasts[0] if pivot_forecasts else {
        'date': (last_date + timedelta(days=30)).strftime('%Y-%m-%d'),
        'date_display': (last_date + timedelta(days=30)).strftime('%d/%m/%Y'),
        'days_from_now': 30,
        'cycle': 'DEFAULT_30D',
        'from': 'ESTIMATE',
        'confidence': 20
    }
    
    # Calculate price range by pivot
    days_to_pivot = primary_pivot['days_from_now']
    max_move = atr * days_to_pivot * 0.7  # 70% efficiency factor
    
    return {
        'next_pivot_date': primary_pivot['date'],
        'next_pivot_display': primary_pivot['date_display'],
        'days_to_pivot': days_to_pivot,
        'pivot_type': 'HIGH' if primary_pivot['from'] == 'HIGH' else 'LOW',
        'pivot_confidence': primary_pivot['confidence'],
        'cycle_sources': [f"{primary_pivot['cycle']} from {primary_pivot['from']}"],
        'probable_price_low': safe_round(current_price - max_move),
        'probable_price_high': safe_round(current_price + max_move),
        'probable_range_text': f"${safe_round(current_price - max_move):,.0f} - ${safe_round(current_price + max_move):,.0f}",
        'atr_daily': safe_round(atr),
        'max_expected_move': safe_round(max_move)
    }

# ============================================================
# INVALIDATION RULES - FIXED v5.0.2
# ============================================================

def build_invalidation_rules(direction, current_price, gann_levels, ichimoku, rsi):
    """
    Build invalidation rules based on current state
    
    FIXED: Check if conditions are already triggered
    """
    rules = []
    
    kijun = ichimoku.get('kijun', 0)
    cloud_bottom = ichimoku.get('cloud_bottom', 0)
    cloud_top = ichimoku.get('cloud_top', 0)
    gann_50 = gann_levels.get('4_8', 0)
    
    if direction == 'BULLISH':
        # Bullish invalidation: price breaks below key supports
        invalidation_price = cloud_bottom if cloud_bottom > 0 else gann_levels.get('3_8', current_price * 0.95)
        
        rules.append({
            'condition': 'Daily close below cloud bottom',
            'price': safe_round(cloud_bottom),
            'triggered': current_price < cloud_bottom
        })
        
        rules.append({
            'condition': 'Weekly close below Gann 50%',
            'price': safe_round(gann_50),
            'triggered': current_price < gann_50
        })
        
        rules.append({
            'condition': 'RSI breaks below 40',
            'current': safe_round(rsi),
            'triggered': rsi < 40
        })
        
    else:  # BEARISH or NEUTRAL
        # Bearish invalidation: price breaks above key resistances
        invalidation_price = cloud_top if cloud_top > 0 else gann_levels.get('5_8', current_price * 1.05)
        
        rules.append({
            'condition': 'Daily close above cloud top',
            'price': safe_round(cloud_top),
            'triggered': current_price > cloud_top
        })
        
        rules.append({
            'condition': 'Weekly close above Gann 50%',
            'price': safe_round(gann_50),
            'triggered': current_price > gann_50
        })
        
        rules.append({
            'condition': 'RSI breaks above 60',
            'current': safe_round(rsi),
            'triggered': rsi > 60
        })
    
    # Check if any rules already triggered
    triggered_count = sum(1 for r in rules if r.get('triggered', False))
    
    return {
        'invalidation_price': safe_round(invalidation_price),
        'invalidation_reason': 'Key level breach',
        'rules': rules,
        'rules_triggered': triggered_count,
        'warning': f"{triggered_count} invalidation rules already triggered!" if triggered_count > 0 else None
    }

# ============================================================
# MARKET REGIME DETECTION
# ============================================================

def detect_market_regime(current_price, sma_200, adx, adx_strength):
    """Detect overall market regime"""
    if sma_200 is None or sma_200 == 0:
        return {
            'regime': 'UNKNOWN',
            'strength': 'UNKNOWN',
            'adx': adx,
            'price_vs_sma200': 'UNKNOWN',
            'sma_200': 0,
            'description': 'Insufficient data for regime detection'
        }
    
    price_vs_sma = 'ABOVE' if current_price > sma_200 else 'BELOW'
    
    if current_price > sma_200:
        if adx >= 25:
            regime = 'TRENDING_BULL'
            description = f'Strong uptrend above 200 SMA'
        else:
            regime = 'RANGING_BULL'
            description = f'Weak uptrend / consolidation above 200 SMA'
    else:
        if adx >= 25:
            regime = 'TRENDING_BEAR'
            description = f'Strong downtrend below 200 SMA'
        else:
            regime = 'RANGING_BEAR'
            description = f'Weak downtrend / consolidation below 200 SMA'
    
    return {
        'regime': regime,
        'strength': adx_strength,
        'adx': safe_round(adx),
        'price_vs_sma200': price_vs_sma,
        'sma_200': safe_round(sma_200),
        'description': description
    }

# ============================================================
# STRATEGY GENERATION
# ============================================================

def generate_strategy(consensus, invalidation, timeframes, current_price):
    """Generate trading strategy based on analysis"""
    direction = consensus['direction']
    confidence = consensus['confidence_level']
    weekly_dir = consensus.get('weekly_direction', 'NEUTRAL')
    daily_dir = consensus.get('daily_direction', 'NEUTRAL')
    
    # Primary bias
    primary_bias = direction
    
    # Action based on direction and confidence
    if direction == 'BULLISH' and confidence == 'HIGH':
        action = 'Buy dips toward support levels'
        entry_method = 'Wait for pullback to Kijun or cloud top, then bullish TK cross'
        size = 'Full size (high confidence)'
    elif direction == 'BULLISH' and confidence == 'MEDIUM':
        action = 'Buy dips cautiously'
        entry_method = 'Wait for daily RSI < 50 then bullish reversal candle'
        size = '50% size (medium confidence)'
    elif direction == 'BEARISH' and confidence == 'HIGH':
        action = 'Sell rallies toward resistance levels'
        entry_method = 'Wait for rally to Kijun or cloud bottom, then bearish TK cross'
        size = 'Full size (high confidence)'
    elif direction == 'BEARISH' and confidence == 'MEDIUM':
        action = 'Sell rallies cautiously'
        entry_method = 'Wait for daily RSI > 50 then bearish reversal candle'
        size = '50% size (medium confidence)'
    else:
        action = 'Wait for clearer setup'
        entry_method = 'No entry until alignment improves'
        size = 'No position (low confidence)'
    
    # Handle divergences
    if weekly_dir == 'BEARISH' and daily_dir == 'BULLISH':
        action = 'CAUTION: Daily bounce in weekly downtrend - sell rallies or stay flat'
        entry_method = 'If trading: wait for rally exhaustion near weekly resistance, then short'
        primary_bias = 'BEARISH'
    elif weekly_dir == 'BULLISH' and daily_dir == 'BEARISH':
        action = 'Daily pullback in weekly uptrend - look for buying opportunity'
        entry_method = 'Wait for daily oversold + bullish reversal near weekly support'
        primary_bias = 'BULLISH'
    
    # Invalidation action
    inv_price = invalidation.get('invalidation_price', 0)
    if direction == 'BULLISH':
        inv_action = f"Close long position if price closes below ${inv_price:,.0f}"
    elif direction == 'BEARISH':
        inv_action = f"Close short position if price closes above ${inv_price:,.0f}"
    else:
        inv_action = "No position to invalidate"
    
    return {
        'primary_bias': primary_bias,
        'action': action,
        'entry_method': entry_method,
        'position_size_recommendation': size,
        'time_in_trade': f"Hold until TP1 or next pivot date",
        'interpretation': consensus['interpretation'],
        'invalidation_action': inv_action
    }

# ============================================================
# ANALYZE SINGLE TIMEFRAME
# ============================================================

def analyze_timeframe(df, tf_name):
    """Complete analysis for a single timeframe"""
    if df is None or df.empty or len(df) < 50:
        return None
    
    # Get parameters
    weight = TIMEFRAME_WEIGHTS.get(tf_name, 10)
    ichi_params = ICHIMOKU_PARAMS.get(tf_name, (9, 26, 52))
    
    # Calculate indicators
    rsi = calculate_rsi(df)
    macd_line, macd_signal, macd_hist = calculate_macd(df)
    adx, adx_strength = calculate_adx(df)
    ichimoku = calculate_ichimoku(df, ichi_params)
    gann = calculate_gann_levels_for_timeframe(df)
    
    # Identify state
    state, state_info = identify_enneagram_state(df, rsi, macd_hist)
    arrow, arrow_meaning = determine_arrow(state, rsi, macd_hist)
    
    # Classify direction
    direction, bullish_count, bearish_count = classify_timeframe_direction(
        rsi, adx, ichimoku['cloud_signal'], ichimoku['tk_cross'], macd_hist, ichimoku['kijun_flat']
    )
    
    return {
        'direction': direction,
        'signal_type': 'BUY' if direction == 'BULLISH' else 'SELL' if direction == 'BEARISH' else 'WAIT',
        'weight': weight,
        'weighted_contribution': 0,  # Will be set by consensus
        
        # Enneagram
        'enneagram_state': state,
        'state_name': state_info['name'],
        'state_bias': state_info['bias'],
        'phase': state_info['phase'],
        'arrow': arrow,
        'arrow_meaning': arrow_meaning,
        
        # Indicators
        'rsi': safe_round(rsi),
        'macd_histogram': safe_round(macd_hist),
        'adx': safe_round(adx),
        'trend_strength': adx_strength,
        
        # Ichimoku
        'cloud_signal': ichimoku['cloud_signal'],
        'tk_cross': ichimoku['tk_cross'],
        'kijun_flat': ichimoku['kijun_flat'],
        'tenkan': safe_round(ichimoku['tenkan']),
        'kijun': safe_round(ichimoku['kijun']),
        'cloud_top': safe_round(ichimoku['cloud_top']),
        'cloud_bottom': safe_round(ichimoku['cloud_bottom']),
        
        # Gann
        'gann_high': safe_round(gann['high']),
        'gann_low': safe_round(gann['low']),
        'gann_50_pct': safe_round(gann['4_8']),
        'gann_38_pct': safe_round(gann['3_8']),
        'gann_62_pct': safe_round(gann['5_8']),
        
        # Signal counts
        'bullish_signals': bullish_count,
        'bearish_signals': bearish_count,
        
        # Full Gann levels
        'gann_levels': gann,
        'ichimoku': ichimoku
    }

# ============================================================
# MAIN API ENDPOINT
# ============================================================

@app.get("/signal/daily")
async def get_daily_signal():
    """Generate comprehensive multi-timeframe signal"""
    try:
        # Fetch daily data
        df_daily = luxor.fetch_real_binance_data(use_cache=True)
        
        if df_daily is None or df_daily.empty:
            raise HTTPException(status_code=500, detail="Failed to fetch market data")
        
        if len(df_daily) < 100:
            raise HTTPException(status_code=500, detail=f"Insufficient data: {len(df_daily)} candles")
        
        # Current price
        current_price = safe_float(df_daily['close'].iloc[-1])
        current_date = df_daily['date'].iloc[-1]
        
        # Resample to higher timeframes
        df_3d = resample_ohlcv(df_daily, '3D')
        df_1w = resample_ohlcv(df_daily, '1W')
        df_1m = resample_ohlcv(df_daily, '1M')
        
        # Analyze each timeframe
        timeframes = {}
        gann_levels_by_tf = {}
        ichimoku_by_tf = {}
        
        for tf_name, df_tf in [('1D', df_daily), ('3D', df_3d), ('1W', df_1w), ('1M', df_1m)]:
            analysis = analyze_timeframe(df_tf, tf_name)
            if analysis:
                timeframes[tf_name] = analysis
                gann_levels_by_tf[tf_name] = analysis['gann_levels']
                ichimoku_by_tf[tf_name] = analysis['ichimoku']
        
        if not timeframes:
            raise HTTPException(status_code=500, detail="Failed to analyze timeframes")
        
        # Calculate consensus
        consensus = calculate_consensus(timeframes)
        
        # Calculate ATR from daily
        atr = calculate_atr(df_daily)
        
        # Calculate targets and stops
        targets = calculate_targets_and_stops(
            current_price, 
            consensus['direction'],
            gann_levels_by_tf,
            ichimoku_by_tf,
            atr
        )
        
        # Calculate R:R
        rr_ratio, reward_pct, risk_pct = calculate_risk_reward(
            current_price,
            targets['tp1'],
            targets['stop_loss']
        )
        
        # Time forecast
        time_forecast = calculate_time_forecast(df_daily, current_price, atr)
        
        # Market regime
        sma_200 = calculate_sma(df_daily, 200)
        daily_adx = timeframes['1D']['adx']
        daily_strength = timeframes['1D']['trend_strength']
        regime = detect_market_regime(current_price, sma_200, daily_adx, daily_strength)
        
        # Invalidation rules
        weekly_gann = gann_levels_by_tf.get('1W', {})
        weekly_ichi = ichimoku_by_tf.get('1W', {})
        weekly_rsi = timeframes.get('1W', {}).get('rsi', 50)
        invalidation = build_invalidation_rules(
            consensus['direction'],
            current_price,
            weekly_gann,
            weekly_ichi,
            weekly_rsi
        )
        
        # Strategy
        strategy = generate_strategy(consensus, invalidation, timeframes, current_price)
        
        # Get dominant timeframe state (weekly for primary)
        weekly_data = timeframes.get('1W', timeframes.get('1D', {}))
        
        # Build response
        response_data = {
            # Meta
            'status': 'success',
            'version': '5.0.2',
            'timestamp': datetime.now().isoformat(),
            'symbol': 'BTCUSDT',
            
            # Legacy fields (for DB compatibility)
            'signal_type': consensus['signal_type'],
            'signal_date': str(current_date),
            'entry_price': safe_round(current_price),
            'take_profit': safe_round(targets['tp1']),
            'stop_loss': safe_round(targets['stop_loss']),
            'confidence': consensus['weighted_score'],
            'confluence_score': consensus['weighted_score'],
            'active_pivot_id': safe_int(len(df_daily)),
            'enneagram_state': weekly_data.get('enneagram_state', 1),
            'price_confluences': len(targets['tp1_sources']),
            'time_confluences': len(time_forecast.get('cycle_sources', [])),
            'gann_cycle_target': time_forecast.get('days_to_pivot', 30),
            'enneagram_arrow': weekly_data.get('arrow', 'NEUTRAL'),
            'macd_signal': str(safe_round(weekly_data.get('macd_histogram', 0), 4)),
            'ichimoku_signal': weekly_data.get('cloud_signal', 'NEUTRAL'),
            'rsi_value': safe_round(timeframes['1D']['rsi']),
            
            # New MTF fields
            'primary_direction': consensus['direction'],
            'weighted_score': consensus['weighted_score'],
            'mtf_alignment': consensus['alignment_count'],
            'market_regime': regime['regime'],
            
            # Timeframes
            'timeframes': timeframes,
            
            # Consensus
            'consensus': consensus,
            
            # Price targets with R:R
            'price_targets': {
                'source_timeframe': '1W',
                'calculation_method': 'Gann + Ichimoku Confluence',
                'tp1': safe_round(targets['tp1']),
                'tp1_sources': targets['tp1_sources'],
                'tp2': safe_round(targets['tp2']),
                'tp2_sources': targets['tp2_sources'],
                'tp3': safe_round(targets['tp3']),
                'tp3_sources': targets['tp3_sources'],
                'stop_loss': safe_round(targets['stop_loss']),
                'stop_sources': targets['stop_sources'],
                'rr_ratio': rr_ratio,
                'reward_pct': reward_pct,
                'risk_pct': risk_pct
            },
            
            # Legacy target fields
            'target_1': safe_round(targets['tp1']),
            'target_2': safe_round(targets['tp2']),
            'target_3': safe_round(targets['tp3']),
            
            # Time forecast
            'time_forecast': time_forecast,
            
            # Pivot forecast (legacy format)
            'pivot_forecast_primary': {
                'date': time_forecast['next_pivot_date'],
                'date_display': time_forecast['next_pivot_display'],
                'days_from_now': time_forecast['days_to_pivot'],
                'expected_pivot': time_forecast['pivot_type'],
                'confidence': time_forecast['pivot_confidence'],
                'cycle_type': time_forecast['cycle_sources'][0] if time_forecast['cycle_sources'] else 'DEFAULT'
            },
            
            # Square of 9 analysis
            'sq9_analysis': {
                'from_current': calculate_square_of_9(current_price),
                'from_high': calculate_square_of_9(gann_levels_by_tf.get('1D', {}).get('high', current_price)),
                'from_low': calculate_square_of_9(gann_levels_by_tf.get('1D', {}).get('low', current_price)),
                'anchors': {
                    'current': safe_round(current_price),
                    'high_52': safe_round(gann_levels_by_tf.get('1D', {}).get('high', 0)),
                    'low_52': safe_round(gann_levels_by_tf.get('1D', {}).get('low', 0)),
                    'midpoint': safe_round(gann_levels_by_tf.get('1D', {}).get('4_8', 0))
                }
            },
            
            # Gann levels (legacy)
            'gann_sq9_levels': str(calculate_square_of_9(current_price)[:4]),
            'gann_angles_active': str([45, 90, 135, 180, 225, 270, 315, 360]),
            
            # Major levels
            'major_high': safe_round(gann_levels_by_tf.get('1M', {}).get('high', 0)),
            'major_low': safe_round(gann_levels_by_tf.get('1M', {}).get('low', 0)),
            'gann_range': safe_round(gann_levels_by_tf.get('1M', {}).get('range', 0)),
            'gann_3_8': safe_round(gann_levels_by_tf.get('1W', {}).get('3_8', 0)),
            'gann_4_8': safe_round(gann_levels_by_tf.get('1W', {}).get('4_8', 0)),
            'gann_5_8': safe_round(gann_levels_by_tf.get('1W', {}).get('5_8', 0)),
            
            # Ichimoku (weekly)
            'tenkan': safe_round(weekly_ichi.get('tenkan', 0)),
            'kijun': safe_round(weekly_ichi.get('kijun', 0)),
            'cloud_top': safe_round(weekly_ichi.get('cloud_top', 0)),
            'cloud_bottom': safe_round(weekly_ichi.get('cloud_bottom', 0)),
            'cloud_signal': weekly_data.get('cloud_signal', 'NEUTRAL'),
            'tk_cross': weekly_data.get('tk_cross', 'NEUTRAL'),
            'kijun_flat': weekly_data.get('kijun_flat', False),
            
            # Regime
            'regime': regime,
            'adx': safe_round(daily_adx),
            'trend_strength': daily_strength,
            
            # State (weekly)
            'state': weekly_data.get('enneagram_state', 1),
            'state_name': weekly_data.get('state_name', 'Unknown'),
            'phase': weekly_data.get('phase', 'Unknown'),
            'arrow': weekly_data.get('arrow', 'NEUTRAL'),
            'arrow_meaning': weekly_data.get('arrow_meaning', ''),
            
            # Daily indicators
            'rsi': safe_round(timeframes['1D']['rsi']),
            'macd': safe_round(timeframes['1D']['macd_histogram']),
            'macd_hist': safe_round(weekly_data.get('macd_histogram', 0)),
            'atr': safe_round(atr),
            
            # Invalidation
            'invalidation': invalidation,
            
            # Strategy
            'strategy': strategy,
            
            # Analysis meta
            'candles_analyzed': len(df_daily),
            'last_candle_date': str(current_date),
            'signal_strength': consensus['confidence_level'],
            'confirmation_score': consensus['weighted_score'],
            'confirmation_score_display': f"{consensus['weighted_score']}%"
        }
        
        return to_native(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_daily_signal: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "5.0.2",
        "timestamp": datetime.now().isoformat()
    }

# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print("LUXOR V7 PRANA RUNTIME - MTF EDITION v5.0.2")
    print("=" * 50)
    print(f"Timeframe Weights: {TIMEFRAME_WEIGHTS}")
    print(f"API Host: {API_HOST}:{API_PORT}")
    print("=" * 50)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
