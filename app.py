"""
LUXOR V7 PRANA RUNTIME - MULTI-TIMEFRAME EDITION
Version: 5.0.3
Fixed: 
- Gann levels use recent range (not all-time)
- Invalidation blocks signal when triggered
- Monthly cloud ignored (too displaced)
- Time forecast capped to realistic range
- tf_name passed to all relevant functions
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
    version="5.0.3",
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

# Gann lookback periods per timeframe
GANN_LOOKBACK = {
    '1D': 252,   # ~1 year of daily candles
    '3D': 120,   # ~1 year of 3-day candles
    '1W': 52,    # 1 year of weekly candles
    '1M': 24     # 2 years of monthly candles
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
        '1M': 'M'
    }
    
    freq = freq_map.get(timeframe, '1D')
    
    try:
        df_copy = df.copy()
        
        if 'date' in df_copy.columns:
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            df_copy.set_index('date', inplace=True)
        
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
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
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
    
    tenkan = (high.rolling(window=tenkan_period).max() + 
              low.rolling(window=tenkan_period).min()) / 2
    
    kijun = (high.rolling(window=kijun_period).max() + 
             low.rolling(window=kijun_period).min()) / 2
    
    senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)
    
    senkou_b = ((high.rolling(window=senkou_period).max() + 
                 low.rolling(window=senkou_period).min()) / 2).shift(kijun_period)
    
    current_close = safe_float(close.iloc[-1])
    current_tenkan = safe_float(tenkan.iloc[-1])
    current_kijun = safe_float(kijun.iloc[-1])
    current_senkou_a = safe_float(senkou_a.iloc[-1])
    current_senkou_b = safe_float(senkou_b.iloc[-1])
    
    cloud_top = max(current_senkou_a, current_senkou_b)
    cloud_bottom = min(current_senkou_a, current_senkou_b)
    
    if current_close > cloud_top:
        cloud_signal = 'BULLISH'
    elif current_close < cloud_bottom:
        cloud_signal = 'BEARISH'
    else:
        cloud_signal = 'NEUTRAL'
    
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
# GANN CALCULATIONS - FIXED v5.0.3
# ============================================================

def calculate_gann_levels_for_timeframe(df, tf_name='1D'):
    """
    Calculate Gann Rule of Eighths for a specific timeframe
    Uses RECENT range based on timeframe lookback, not all-time data
    
    Lookback periods:
    - 1D: 252 trading days (~1 year)
    - 3D: 120 candles (~1 year)
    - 1W: 52 weeks (~1 year)
    - 1M: 24 months (~2 years)
    """
    if df is None or df.empty:
        return {
            'high': 0, 'low': 0, 'range': 0,
            '0_8': 0, '1_8': 0, '2_8': 0, '3_8': 0,
            '4_8': 0, '5_8': 0, '6_8': 0, '7_8': 0, '8_8': 0
        }
    
    # Get lookback period for this timeframe
    lookback = GANN_LOOKBACK.get(tf_name, 252)
    
    # Use only recent data (not all-time)
    recent_df = df.tail(lookback)
    
    high = safe_float(recent_df['high'].max())
    low = safe_float(recent_df['low'].min())
    range_val = high - low
    
    # Prevent division issues
    if range_val <= 0:
        range_val = high * 0.1
    
    return {
        'high': high,
        'low': low,
        'range': range_val,
        '0_8': low,
        '1_8': low + range_val * 0.125,
        '2_8': low + range_val * 0.25,
        '3_8': low + range_val * 0.375,
        '4_8': low + range_val * 0.5,
        '5_8': low + range_val * 0.625,
        '6_8': low + range_val * 0.75,
        '7_8': low + range_val * 0.875,
        '8_8': high
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
    
    if rsi < 30 and macd_hist < 0:
        state = 5
    elif rsi < 35 and macd_hist > 0:
        state = 6
    elif rsi > 70 and macd_hist > 0:
        state = 3
    elif rsi > 65 and macd_hist < 0:
        state = 8
    elif 40 <= rsi <= 60 and abs(macd_hist) < abs(df['close'].mean() * 0.01):
        if price_change_pct > 2:
            state = 2
        elif price_change_pct < -2:
            state = 4
        else:
            state = 1
    elif rsi > 55 and macd_hist > 0:
        state = 7
    elif rsi < 45 and macd_hist < 0:
        state = 9
    else:
        state = 1
    
    return state, ENNEAGRAM_STATES[state]

def determine_arrow(state, rsi, macd_hist, prev_rsi=None):
    """Determine STRESS or GROWTH arrow"""
    if prev_rsi is None:
        prev_rsi = rsi
    
    if state in [2, 3, 6, 7]:
        if rsi > prev_rsi and macd_hist > 0:
            arrow = 'GROWTH'
        else:
            arrow = 'STRESS'
    elif state in [5, 8, 9]:
        if rsi < prev_rsi and macd_hist < 0:
            arrow = 'STRESS'
        else:
            arrow = 'GROWTH'
    else:
        if macd_hist > 0:
            arrow = 'GROWTH'
        else:
            arrow = 'STRESS'
    
    meaning = ARROW_MEANINGS.get((state, arrow), "Transition in progress")
    
    return arrow, meaning

# ============================================================
# DIRECTION CLASSIFICATION - FIXED v5.0.3
# ============================================================

def classify_timeframe_direction(rsi, adx, cloud_signal, tk_cross, macd_hist, kijun_flat=False, tf_name='1D'):
    """
    Classify direction for a timeframe based on multiple indicators
    
    FIXED v5.0.3: 
    - Monthly uses TK cross only (cloud too displaced - 26 months!)
    - RSI properly weighted in trend context
    """
    bullish = 0
    bearish = 0
    
    # 1. Cloud Signal (HEAVY WEIGHT - 2 points)
    # SKIP cloud for monthly - 26-month displacement is too old to be relevant
    if tf_name != '1M':
        if cloud_signal == 'BULLISH':
            bullish += 2
        elif cloud_signal == 'BEARISH':
            bearish += 2
    
    # 2. TK Cross (1 point normally, 2 for monthly since cloud skipped)
    tk_weight = 2 if tf_name == '1M' else 1
    if tk_cross == 'BULLISH':
        bullish += tk_weight
    elif tk_cross == 'BEARISH':
        bearish += tk_weight
    
    # 3. MACD Histogram (1 point)
    if macd_hist > 0:
        bullish += 1
    elif macd_hist < 0:
        bearish += 1
    
    # 4. RSI with TREND CONTEXT (1 point)
    if cloud_signal == 'BEARISH' or (tf_name == '1M' and tk_cross == 'BEARISH'):
        if rsi < 40:
            bearish += 1
        elif rsi > 60:
            bullish += 1
    elif cloud_signal == 'BULLISH' or (tf_name == '1M' and tk_cross == 'BULLISH'):
        if rsi > 60:
            bullish += 1
        elif rsi < 40:
            bearish += 1
    else:
        if rsi > 60:
            bullish += 1
        elif rsi < 40:
            bearish += 1
    
    # 5. ADX amplifies dominant signal
    if adx > 40:
        if bullish > bearish:
            bullish += 1
        elif bearish > bullish:
            bearish += 1
    
    # Determine direction - need CLEAR majority (60%+)
    total_signals = bullish + bearish
    
    if total_signals == 0:
        return 'NEUTRAL', 0, 0
    
    bullish_pct = bullish / total_signals * 100
    bearish_pct = bearish / total_signals * 100
    
    if bullish_pct >= 60:
        return 'BULLISH', bullish, bearish
    elif bearish_pct >= 60:
        return 'BEARISH', bullish, bearish
    else:
        return 'NEUTRAL', bullish, bearish

# ============================================================
# CONSENSUS CALCULATION - FIXED v5.0.3
# ============================================================

def calculate_consensus(timeframes):
    """
    Calculate weighted consensus across timeframes
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
        else:
            tf_data['weighted_contribution'] = 0
    
    bullish_pct = (total_bullish_weight / total_weight) * 100 if total_weight > 0 else 0
    bearish_pct = (total_bearish_weight / total_weight) * 100 if total_weight > 0 else 0
    neutral_pct = 100 - bullish_pct - bearish_pct
    
    if bullish_pct > bearish_pct and bullish_pct >= 40:
        primary_direction = 'BULLISH'
        weighted_score = bullish_pct
    elif bearish_pct > bullish_pct and bearish_pct >= 40:
        primary_direction = 'BEARISH'
        weighted_score = bearish_pct
    else:
        primary_direction = 'NEUTRAL'
        weighted_score = max(bullish_pct, bearish_pct)
    
    if primary_direction == 'NEUTRAL':
        aligned_count = sum(1 for d in directions.values() if d == 'NEUTRAL')
    else:
        aligned_count = sum(1 for d in directions.values() if d == primary_direction)
    
    total_tfs = len(timeframes)
    alignment_str = f"{aligned_count}/{total_tfs}"
    
    alignment_detail = ", ".join([f"{tf}:{dir}" for tf, dir in directions.items()])
    
    if weighted_score >= 70 and aligned_count >= 3:
        confidence = 'HIGH'
    elif weighted_score >= 50 and aligned_count >= 2:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'
    
    if primary_direction == 'BULLISH' and confidence in ['HIGH', 'MEDIUM']:
        signal_type = 'BUY'
    elif primary_direction == 'BEARISH' and confidence in ['HIGH', 'MEDIUM']:
        signal_type = 'SELL'
    else:
        signal_type = 'WAIT'
    
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
        'interpretation': interpretation,
        'is_valid': True  # Will be updated by validate_setup
    }

# ============================================================
# SETUP VALIDATION - NEW v5.0.3
# ============================================================

def validate_setup(invalidation, consensus):
    """
    Check if setup is still valid (no invalidation rules triggered)
    Returns modified consensus if invalid
    """
    rules_triggered = invalidation.get('rules_triggered', 0)
    
    if rules_triggered > 0:
        triggered_rules = [r['condition'] for r in invalidation.get('rules', []) if r.get('triggered', False)]
        return {
            **consensus,
            'direction': 'INVALID',
            'signal_type': 'NO_TRADE',
            'confidence_level': 'NONE',
            'weighted_score': 0,
            'interpretation': f"Setup invalidated: {', '.join(triggered_rules)}. Stay flat until new setup forms.",
            'is_valid': False,
            'invalidation_reason': triggered_rules
        }
    
    return {
        **consensus,
        'is_valid': True
    }

# ============================================================
# RISK:REWARD CALCULATION
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
# PRICE TARGETS & STOP LOSS
# ============================================================

def calculate_targets_and_stops(current_price, direction, gann_levels_by_tf, ichimoku_by_tf, atr):
    """
    Calculate TP and SL based on timeframe-specific levels
    """
    targets = {
        'tp1': None, 'tp1_sources': [],
        'tp2': None, 'tp2_sources': [],
        'tp3': None, 'tp3_sources': [],
        'stop_loss': None, 'stop_sources': []
    }
    
    resistance_levels = []
    support_levels = []
    
    for tf_name in ['1M', '1W', '3D', '1D']:
        gann = gann_levels_by_tf.get(tf_name, {})
        ichi = ichimoku_by_tf.get(tf_name, {})
        
        for level_name in ['3_8', '4_8', '5_8', '6_8', '7_8']:
            level_val = gann.get(level_name, 0)
            if level_val > 0:
                source = f"GANN_{level_name}_{tf_name}"
                if level_val > current_price * 1.005:  # 0.5% buffer
                    resistance_levels.append({'price': level_val, 'source': source, 'tf': tf_name})
                elif level_val < current_price * 0.995:
                    support_levels.append({'price': level_val, 'source': source, 'tf': tf_name})
        
        cloud_top = ichi.get('cloud_top', 0)
        cloud_bottom = ichi.get('cloud_bottom', 0)
        kijun = ichi.get('kijun', 0)
        
        if cloud_top > current_price * 1.005:
            resistance_levels.append({'price': cloud_top, 'source': f'CLOUD_TOP_{tf_name}', 'tf': tf_name})
        if cloud_bottom > 0 and cloud_bottom < current_price * 0.995:
            support_levels.append({'price': cloud_bottom, 'source': f'CLOUD_BOTTOM_{tf_name}', 'tf': tf_name})
        if kijun > 0:
            if kijun > current_price * 1.005:
                resistance_levels.append({'price': kijun, 'source': f'KIJUN_{tf_name}', 'tf': tf_name})
            elif kijun < current_price * 0.995:
                support_levels.append({'price': kijun, 'source': f'KIJUN_{tf_name}', 'tf': tf_name})
    
    resistance_levels.sort(key=lambda x: x['price'])
    support_levels.sort(key=lambda x: x['price'], reverse=True)
    
    # Remove duplicates (within 0.5% of each other)
    def dedupe_levels(levels):
        if not levels:
            return levels
        result = [levels[0]]
        for level in levels[1:]:
            if abs(level['price'] - result[-1]['price']) / result[-1]['price'] > 0.005:
                result.append(level)
            else:
                # Merge sources
                result[-1]['source'] += f" + {level['source']}"
        return result
    
    resistance_levels = dedupe_levels(resistance_levels)
    support_levels = dedupe_levels(support_levels)
    
    if direction == 'BULLISH' or direction == 'NEUTRAL':
        if len(resistance_levels) >= 1:
            targets['tp1'] = resistance_levels[0]['price']
            targets['tp1_sources'] = [resistance_levels[0]['source']]
        if len(resistance_levels) >= 2:
            targets['tp2'] = resistance_levels[1]['price']
            targets['tp2_sources'] = [resistance_levels[1]['source']]
        if len(resistance_levels) >= 3:
            targets['tp3'] = resistance_levels[2]['price']
            targets['tp3_sources'] = [resistance_levels[2]['source']]
        
        if len(support_levels) >= 1:
            targets['stop_loss'] = support_levels[0]['price']
            targets['stop_sources'] = [support_levels[0]['source']]
    
    else:  # BEARISH
        if len(support_levels) >= 1:
            targets['tp1'] = support_levels[0]['price']
            targets['tp1_sources'] = [support_levels[0]['source']]
        if len(support_levels) >= 2:
            targets['tp2'] = support_levels[1]['price']
            targets['tp2_sources'] = [support_levels[1]['source']]
        if len(support_levels) >= 3:
            targets['tp3'] = support_levels[2]['price']
            targets['tp3_sources'] = [support_levels[2]['source']]
        
        if len(resistance_levels) >= 1:
            targets['stop_loss'] = resistance_levels[0]['price']
            targets['stop_sources'] = [resistance_levels[0]['source']]
    
    # Fallback to ATR-based
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
# TIME FORECAST - FIXED v5.0.3
# ============================================================

def calculate_time_forecast(df, current_price, atr, gann_levels_1w):
    """
    Calculate time-based pivot forecasts using Gann cycles
    
    FIXED v5.0.3: Cap range to realistic bounds using Gann levels
    """
    if df is None or df.empty:
        return {}
    
    last_date = pd.to_datetime(df['date'].iloc[-1])
    
    cycles = [30, 45, 60, 90, 120, 180, 360]
    
    highs = df.nlargest(5, 'high')
    lows = df.nsmallest(5, 'low')
    
    pivot_forecasts = []
    
    for cycle in cycles:
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
    
    pivot_forecasts.sort(key=lambda x: x['days_from_now'])
    
    primary_pivot = pivot_forecasts[0] if pivot_forecasts else {
        'date': (last_date + timedelta(days=30)).strftime('%Y-%m-%d'),
        'date_display': (last_date + timedelta(days=30)).strftime('%d/%m/%Y'),
        'days_from_now': 30,
        'cycle': 'DEFAULT_30D',
        'from': 'ESTIMATE',
        'confidence': 20
    }
    
    days_to_pivot = primary_pivot['days_from_now']
    
    # ATR-based projection
    max_move = atr * days_to_pivot * 0.7
    
    # Calculate raw bounds
    raw_high = current_price + max_move
    raw_low = current_price - max_move
    
    # CAP to Gann levels for realistic bounds
    gann_high = gann_levels_1w.get('7_8', raw_high)  # 87.5% level
    gann_low = gann_levels_1w.get('2_8', raw_low)    # 25% level
    
    # Don't exceed Gann bounds by more than 5%
    probable_high = min(raw_high, gann_high * 1.05)
    probable_low = max(raw_low, gann_low * 0.95)
    
    # Ensure we have at least some range
    if probable_high <= current_price:
        probable_high = current_price * 1.10
    if probable_low >= current_price:
        probable_low = current_price * 0.90
    
    return {
        'next_pivot_date': primary_pivot['date'],
        'next_pivot_display': primary_pivot['date_display'],
        'days_to_pivot': days_to_pivot,
        'pivot_type': 'HIGH' if primary_pivot['from'] == 'HIGH' else 'LOW',
        'pivot_confidence': primary_pivot['confidence'],
        'cycle_sources': [f"{primary_pivot['cycle']} from {primary_pivot['from']}"],
        'probable_price_low': safe_round(probable_low),
        'probable_price_high': safe_round(probable_high),
        'probable_range_text': f"${safe_round(probable_low):,.0f} - ${safe_round(probable_high):,.0f}",
        'atr_daily': safe_round(atr),
        'max_expected_move': safe_round(max_move)
    }

# ============================================================
# INVALIDATION RULES - FIXED v5.0.3
# ============================================================

def build_invalidation_rules(direction, current_price, gann_levels, ichimoku, rsi):
    """
    Build invalidation rules based on current state
    Checks if conditions are already triggered
    """
    rules = []
    
    kijun = ichimoku.get('kijun', 0)
    cloud_bottom = ichimoku.get('cloud_bottom', 0)
    cloud_top = ichimoku.get('cloud_top', 0)
    gann_50 = gann_levels.get('4_8', 0)
    gann_38 = gann_levels.get('3_8', 0)
    gann_62 = gann_levels.get('5_8', 0)
    
    if direction == 'BULLISH':
        # Use the nearest SUPPORT level below price
        invalidation_price = max(
            cloud_bottom if cloud_bottom > 0 and cloud_bottom < current_price else 0,
            gann_38 if gann_38 < current_price else 0
        )
        if invalidation_price == 0:
            invalidation_price = current_price * 0.95
        
        rules.append({
            'condition': 'Daily close below Gann 3/8',
            'price': safe_round(gann_38),
            'triggered': current_price < gann_38 if gann_38 > 0 else False
        })
        
        rules.append({
            'condition': 'Daily close below cloud bottom',
            'price': safe_round(cloud_bottom),
            'triggered': current_price < cloud_bottom if cloud_bottom > 0 else False
        })
        
        rules.append({
            'condition': 'RSI breaks below 35',
            'current': safe_round(rsi),
            'triggered': rsi < 35
        })
        
    else:  # BEARISH or NEUTRAL
        # Use the nearest RESISTANCE level above price
        invalidation_price = min(
            cloud_top if cloud_top > current_price else float('inf'),
            gann_62 if gann_62 > current_price else float('inf')
        )
        if invalidation_price == float('inf'):
            invalidation_price = current_price * 1.05
        
        rules.append({
            'condition': 'Daily close above Gann 5/8',
            'price': safe_round(gann_62),
            'triggered': current_price > gann_62 if gann_62 > 0 else False
        })
        
        rules.append({
            'condition': 'Daily close above cloud top',
            'price': safe_round(cloud_top),
            'triggered': current_price > cloud_top if cloud_top > 0 else False
        })
        
        rules.append({
            'condition': 'RSI breaks above 65',
            'current': safe_round(rsi),
            'triggered': rsi > 65
        })
    
    triggered_count = sum(1 for r in rules if r.get('triggered', False))
    
    return {
        'invalidation_price': safe_round(invalidation_price),
        'invalidation_reason': 'Key level breach',
        'rules': rules,
        'rules_triggered': triggered_count,
        'warning': f"⚠️ {triggered_count} invalidation rule(s) already triggered!" if triggered_count > 0 else None
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
# STRATEGY GENERATION - FIXED v5.0.3
# ============================================================

def generate_strategy(consensus, invalidation, timeframes, current_price):
    """Generate trading strategy based on analysis"""
    
    # Check if setup is invalid
    if not consensus.get('is_valid', True):
        return {
            'primary_bias': 'FLAT',
            'action': 'No trade - setup already invalidated',
            'entry_method': 'Wait for new setup to form',
            'position_size_recommendation': 'Zero (stay flat)',
            'time_in_trade': 'N/A',
            'interpretation': consensus.get('interpretation', 'Setup invalid'),
            'invalidation_action': 'Already invalidated - no position to manage'
        }
    
    direction = consensus['direction']
    confidence = consensus['confidence_level']
    weekly_dir = consensus.get('weekly_direction', 'NEUTRAL')
    daily_dir = consensus.get('daily_direction', 'NEUTRAL')
    
    primary_bias = direction
    
    if direction == 'BULLISH' and confidence == 'HIGH':
        action = 'Buy dips toward support levels'
        entry_method = 'Wait for pullback to Kijun or Gann 50%, then bullish TK cross'
        size = 'Full size (high confidence)'
    elif direction == 'BULLISH' and confidence == 'MEDIUM':
        action = 'Buy dips cautiously'
        entry_method = 'Wait for daily RSI < 50 then bullish reversal candle'
        size = '50% size (medium confidence)'
    elif direction == 'BEARISH' and confidence == 'HIGH':
        action = 'Sell rallies toward resistance levels'
        entry_method = 'Wait for rally to Kijun or Gann 50%, then bearish TK cross'
        size = 'Full size (high confidence)'
    elif direction == 'BEARISH' and confidence == 'MEDIUM':
        action = 'Sell rallies cautiously'
        entry_method = 'Wait for daily RSI > 50 then bearish reversal candle'
        size = '50% size (medium confidence)'
    else:
        action = 'Wait for clearer setup'
        entry_method = 'No entry until alignment improves'
        size = 'No position (low confidence)'
    
    if weekly_dir == 'BEARISH' and daily_dir == 'BULLISH':
        action = 'CAUTION: Daily bounce in weekly downtrend - sell rallies or stay flat'
        entry_method = 'If trading: wait for rally exhaustion near weekly resistance, then short'
        primary_bias = 'BEARISH'
    elif weekly_dir == 'BULLISH' and daily_dir == 'BEARISH':
        action = 'Daily pullback in weekly uptrend - look for buying opportunity'
        entry_method = 'Wait for daily oversold + bullish reversal near weekly support'
        primary_bias = 'BULLISH'
    
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
# ANALYZE SINGLE TIMEFRAME - FIXED v5.0.3
# ============================================================

def analyze_timeframe(df, tf_name):
    """Complete analysis for a single timeframe"""
    if df is None or df.empty or len(df) < 50:
        return None
    
    weight = TIMEFRAME_WEIGHTS.get(tf_name, 10)
    ichi_params = ICHIMOKU_PARAMS.get(tf_name, (9, 26, 52))
    
    rsi = calculate_rsi(df)
    macd_line, macd_signal, macd_hist = calculate_macd(df)
    adx, adx_strength = calculate_adx(df)
    ichimoku = calculate_ichimoku(df, ichi_params)
    
    # FIXED: Pass tf_name to Gann calculation
    gann = calculate_gann_levels_for_timeframe(df, tf_name)
    
    state, state_info = identify_enneagram_state(df, rsi, macd_hist)
    arrow, arrow_meaning = determine_arrow(state, rsi, macd_hist)
    
    # FIXED: Pass tf_name to direction classification
    direction, bullish_count, bearish_count = classify_timeframe_direction(
        rsi, adx, ichimoku['cloud_signal'], ichimoku['tk_cross'], macd_hist, ichimoku['kijun_flat'], tf_name
    )
    
    return {
        'direction': direction,
        'signal_type': 'BUY' if direction == 'BULLISH' else 'SELL' if direction == 'BEARISH' else 'WAIT',
        'weight': weight,
        'weighted_contribution': 0,
        
        'enneagram_state': state,
        'state_name': state_info['name'],
        'state_bias': state_info['bias'],
        'phase': state_info['phase'],
        'arrow': arrow,
        'arrow_meaning': arrow_meaning,
        
        'rsi': safe_round(rsi),
        'macd_histogram': safe_round(macd_hist),
        'adx': safe_round(adx),
        'trend_strength': adx_strength,
        
        'cloud_signal': ichimoku['cloud_signal'],
        'tk_cross': ichimoku['tk_cross'],
        'kijun_flat': ichimoku['kijun_flat'],
        'tenkan': safe_round(ichimoku['tenkan']),
        'kijun': safe_round(ichimoku['kijun']),
        'cloud_top': safe_round(ichimoku['cloud_top']),
        'cloud_bottom': safe_round(ichimoku['cloud_bottom']),
        
        'gann_high': safe_round(gann['high']),
        'gann_low': safe_round(gann['low']),
        'gann_50_pct': safe_round(gann['4_8']),
        'gann_38_pct': safe_round(gann['3_8']),
        'gann_62_pct': safe_round(gann['5_8']),
        
        'bullish_signals': bullish_count,
        'bearish_signals': bearish_count,
        
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
        df_daily = luxor.fetch_real_binance_data(use_cache=True)
        
        if df_daily is None or df_daily.empty:
            raise HTTPException(status_code=500, detail="Failed to fetch market data")
        
        if len(df_daily) < 100:
            raise HTTPException(status_code=500, detail=f"Insufficient data: {len(df_daily)} candles")
        
        current_price = safe_float(df_daily['close'].iloc[-1])
        current_date = df_daily['date'].iloc[-1]
        
        df_3d = resample_ohlcv(df_daily, '3D')
        df_1w = resample_ohlcv(df_daily, '1W')
        df_1m = resample_ohlcv(df_daily, '1M')
        
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
        
        # Calculate ATR
        atr = calculate_atr(df_daily)
        
        # Calculate targets
        targets = calculate_targets_and_stops(
            current_price, 
            consensus['direction'],
            gann_levels_by_tf,
            ichimoku_by_tf,
            atr
        )
        
        # R:R
        rr_ratio, reward_pct, risk_pct = calculate_risk_reward(
            current_price,
            targets['tp1'],
            targets['stop_loss']
        )
        
        # Time forecast - FIXED: pass weekly Gann levels
        time_forecast = calculate_time_forecast(
            df_daily, 
            current_price, 
            atr, 
            gann_levels_by_tf.get('1W', {})
        )
        
        # Regime
        sma_200 = calculate_sma(df_daily, 200)
        daily_adx = timeframes['1D']['adx']
        daily_strength = timeframes['1D']['trend_strength']
        regime = detect_market_regime(current_price, sma_200, daily_adx, daily_strength)
        
        # Invalidation
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
        
        # FIXED: Validate setup - check if invalidation already triggered
        consensus = validate_setup(invalidation, consensus)
        
        # Strategy
        strategy = generate_strategy(consensus, invalidation, timeframes, current_price)
        
        # Weekly data for dominant state
        weekly_data = timeframes.get('1W', timeframes.get('1D', {}))
        
        # Build response
        response_data = {
            'status': 'success',
            'version': '5.0.3',
            'timestamp': datetime.now().isoformat(),
            'symbol': 'BTCUSDT',
            
            # Legacy fields
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
            
            # MTF fields
            'primary_direction': consensus['direction'],
            'weighted_score': consensus['weighted_score'],
            'mtf_alignment': consensus['alignment_count'],
            'market_regime': regime['regime'],
            'setup_valid': consensus.get('is_valid', True),
            
            # Timeframes
            'timeframes': timeframes,
            
            # Consensus
            'consensus': consensus,
            
            # Price targets
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
            
            'target_1': safe_round(targets['tp1']),
            'target_2': safe_round(targets['tp2']),
            'target_3': safe_round(targets['tp3']),
            
            # Time forecast
            'time_forecast': time_forecast,
            
            'pivot_forecast_primary': {
                'date': time_forecast['next_pivot_date'],
                'date_display': time_forecast['next_pivot_display'],
                'days_from_now': time_forecast['days_to_pivot'],
                'expected_pivot': time_forecast['pivot_type'],
                'confidence': time_forecast['pivot_confidence'],
                'cycle_type': time_forecast['cycle_sources'][0] if time_forecast['cycle_sources'] else 'DEFAULT'
            },
            
            # SQ9
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
            
            'gann_sq9_levels': str(calculate_square_of_9(current_price)[:4]),
            'gann_angles_active': str([45, 90, 135, 180, 225, 270, 315, 360]),
            
            # Major levels (from monthly for overall, weekly for trading)
            'major_high': safe_round(gann_levels_by_tf.get('1M', {}).get('high', 0)),
            'major_low': safe_round(gann_levels_by_tf.get('1M', {}).get('low', 0)),
            'gann_range': safe_round(gann_levels_by_tf.get('1W', {}).get('range', 0)),
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
            
            # State (weekly dominant)
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
            
            # Meta
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
        "version": "5.0.3",
        "timestamp": datetime.now().isoformat()
    }

@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print("LUXOR V7 PRANA RUNTIME - MTF EDITION v5.0.3")
    print("=" * 50)
    print(f"Timeframe Weights: {TIMEFRAME_WEIGHTS}")
    print(f"Gann Lookback: {GANN_LOOKBACK}")
    print(f"API Host: {API_HOST}:{API_PORT}")
    print("=" * 50)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
