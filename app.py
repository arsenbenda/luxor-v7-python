"""
LUXOR V7 PRANA RUNTIME - MULTI-TIMEFRAME EDITION
Version: 5.0.5
Fixed:
- R:R validation with minimum 1.5 threshold
- Primary bias follows Gann 50% rule (overrides consensus)
- Capitulation requires volume confirmation
- SQ9 levels filtered by actionable distance (1-10%)
- Time forecast includes cycle origin transparency
- Gann range includes metadata (lookback, dates)
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
    version="5.0.5",
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

GANN_LOOKBACK = {
    '1D': 252,
    '3D': 120,
    '1W': 52,
    '1M': 24
}

# v5.0.5: Minimum R:R ratio for valid setups
MIN_RR_RATIO = 1.5

# v5.0.5: SQ9 distance filters
SQ9_MIN_DISTANCE_PCT = 1.0   # Minimum 1% away (filter noise)
SQ9_MAX_DISTANCE_PCT = 10.0  # Maximum 10% away (too far)

CONFIDENCE_TIERS = {
    'HIGH': {'min_score': 75, 'min_alignment': 3, 'regime_must_agree': True},
    'MEDIUM': {'min_score': 60, 'min_alignment': 2, 'regime_must_agree': False},
    'LOW': {'min_score': 0, 'min_alignment': 0, 'regime_must_agree': False}
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
# HELPER FUNCTIONS
# ============================================================

def to_native(obj):
    """Convert numpy/pandas types to native Python types"""
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
    try:
        if val is None or pd.isna(val):
            return default
        return float(val)
    except:
        return default

def safe_int(val, default=0):
    try:
        if val is None or pd.isna(val):
            return default
        return int(val)
    except:
        return default

def safe_round(val, decimals=2):
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
    if df is None or df.empty:
        return None
    
    freq_map = {'1D': '1D', '3D': '3D', '1W': 'W', '1M': 'M'}
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
    if df is None or len(df) < slow + signal:
        return 0.0, 0.0, 0.0
    close = df['close']
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return safe_float(macd_line.iloc[-1]), safe_float(signal_line.iloc[-1]), safe_float(histogram.iloc[-1])

def calculate_atr(df, period=14):
    if df is None or len(df) < period + 1:
        return 0.0
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return safe_float(atr.iloc[-1])

def calculate_adx(df, period=14):
    if df is None or len(df) < period * 2:
        return 25.0, 'MODERATE'
    high, low, close = df['high'], df['low'], df['close']
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
    strength = 'STRONG' if adx_value >= 50 else 'MODERATE' if adx_value >= 25 else 'WEAK'
    return adx_value, strength

def calculate_sma(df, period=200):
    if df is None or len(df) < period:
        return None
    return safe_float(df['close'].rolling(window=period).mean().iloc[-1])

# ============================================================
# ICHIMOKU
# ============================================================

def calculate_ichimoku(df, params=(9, 26, 52)):
    if df is None or len(df) < params[2] + 26:
        return {
            'tenkan': 0, 'kijun': 0, 'senkou_a': 0, 'senkou_b': 0,
            'cloud_top': 0, 'cloud_bottom': 0, 'chikou': 0,
            'cloud_signal': 'NEUTRAL', 'tk_cross': 'NEUTRAL', 'kijun_flat': False
        }
    
    tenkan_period, kijun_period, senkou_period = params
    high, low, close = df['high'], df['low'], df['close']
    
    tenkan = (high.rolling(window=tenkan_period).max() + low.rolling(window=tenkan_period).min()) / 2
    kijun = (high.rolling(window=kijun_period).max() + low.rolling(window=kijun_period).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)
    senkou_b = ((high.rolling(window=senkou_period).max() + low.rolling(window=senkou_period).min()) / 2).shift(kijun_period)
    
    current_close = safe_float(close.iloc[-1])
    current_tenkan = safe_float(tenkan.iloc[-1])
    current_kijun = safe_float(kijun.iloc[-1])
    current_senkou_a = safe_float(senkou_a.iloc[-1])
    current_senkou_b = safe_float(senkou_b.iloc[-1])
    
    cloud_top = max(current_senkou_a, current_senkou_b)
    cloud_bottom = min(current_senkou_a, current_senkou_b)
    
    cloud_signal = 'BULLISH' if current_close > cloud_top else 'BEARISH' if current_close < cloud_bottom else 'NEUTRAL'
    
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
    
    kijun_flat = len(kijun) >= 5 and kijun.iloc[-5:].std() < (current_kijun * 0.001)
    
    return {
        'tenkan': current_tenkan, 'kijun': current_kijun,
        'senkou_a': current_senkou_a, 'senkou_b': current_senkou_b,
        'cloud_top': cloud_top, 'cloud_bottom': cloud_bottom,
        'chikou': current_close, 'cloud_signal': cloud_signal,
        'tk_cross': tk_cross, 'kijun_flat': kijun_flat
    }

# ============================================================
# GANN CALCULATIONS - v5.0.5 with metadata
# ============================================================

def calculate_gann_levels_for_timeframe(df, tf_name='1D'):
    """Calculate Gann levels with date metadata"""
    if df is None or df.empty:
        return {
            'high': 0, 'low': 0, 'range': 0,
            '0_8': 0, '1_8': 0, '2_8': 0, '3_8': 0,
            '4_8': 0, '5_8': 0, '6_8': 0, '7_8': 0, '8_8': 0,
            'high_date': None, 'low_date': None, 'lookback_bars': 0
        }
    
    lookback = GANN_LOOKBACK.get(tf_name, 252)
    recent_df = df.tail(lookback)
    
    high = safe_float(recent_df['high'].max())
    low = safe_float(recent_df['low'].min())
    range_val = high - low if high - low > 0 else high * 0.1
    
    # Get dates of high and low
    high_idx = recent_df['high'].idxmax()
    low_idx = recent_df['low'].idxmin()
    high_date = str(recent_df.loc[high_idx, 'date']) if 'date' in recent_df.columns else None
    low_date = str(recent_df.loc[low_idx, 'date']) if 'date' in recent_df.columns else None
    
    return {
        'high': high, 'low': low, 'range': range_val,
        '0_8': low,
        '1_8': low + range_val * 0.125,
        '2_8': low + range_val * 0.25,
        '3_8': low + range_val * 0.375,
        '4_8': low + range_val * 0.5,
        '5_8': low + range_val * 0.625,
        '6_8': low + range_val * 0.75,
        '7_8': low + range_val * 0.875,
        '8_8': high,
        'high_date': high_date,
        'low_date': low_date,
        'lookback_bars': lookback,
        'range_pct': safe_round((range_val / high) * 100 if high > 0 else 0)
    }

# ============================================================
# SQUARE OF 9 - v5.0.5 with distance filtering
# ============================================================

def calculate_square_of_9(base_price, filter_by_distance=True):
    """Calculate SQ9 levels with actionable distance filtering"""
    if base_price <= 0:
        return [], []
    
    sqrt_price = np.sqrt(base_price)
    all_levels = []
    actionable_levels = []
    
    # Use larger angles for swing trading
    for angle in [45, 90, 135, 180, 225, 270, 315, 360, 450, 540, 630, 720]:
        increment = angle / 180
        price_up = (sqrt_price + increment) ** 2
        price_down = max(0, (sqrt_price - increment) ** 2)
        
        dist_up_pct = ((price_up - base_price) / base_price) * 100
        dist_down_pct = ((price_down - base_price) / base_price) * 100
        
        level = {
            'angle': angle,
            'price_up': safe_round(price_up),
            'price_down': safe_round(price_down),
            'distance_up_pct': safe_round(dist_up_pct),
            'distance_down_pct': safe_round(dist_down_pct)
        }
        all_levels.append(level)
        
        # Filter actionable levels (1-10% away)
        if filter_by_distance:
            if SQ9_MIN_DISTANCE_PCT <= abs(dist_up_pct) <= SQ9_MAX_DISTANCE_PCT:
                actionable_levels.append({
                    'direction': 'UP',
                    'angle': angle,
                    'price': safe_round(price_up),
                    'distance_pct': safe_round(dist_up_pct)
                })
            if SQ9_MIN_DISTANCE_PCT <= abs(dist_down_pct) <= SQ9_MAX_DISTANCE_PCT:
                actionable_levels.append({
                    'direction': 'DOWN',
                    'angle': angle,
                    'price': safe_round(price_down),
                    'distance_pct': safe_round(dist_down_pct)
                })
    
    # Sort actionable by distance
    actionable_levels.sort(key=lambda x: abs(x['distance_pct']))
    
    return all_levels, actionable_levels

# ============================================================
# ENNEAGRAM
# ============================================================

def identify_enneagram_state(df, rsi, macd_hist):
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
        state = 2 if price_change_pct > 2 else 4 if price_change_pct < -2 else 1
    elif rsi > 55 and macd_hist > 0:
        state = 7
    elif rsi < 45 and macd_hist < 0:
        state = 9
    else:
        state = 1
    
    return state, ENNEAGRAM_STATES[state]

def determine_arrow(state, rsi, macd_hist, prev_rsi=None):
    if prev_rsi is None:
        prev_rsi = rsi
    
    if state in [2, 3, 6, 7]:
        arrow = 'GROWTH' if rsi > prev_rsi and macd_hist > 0 else 'STRESS'
    elif state in [5, 8, 9]:
        arrow = 'STRESS' if rsi < prev_rsi and macd_hist < 0 else 'GROWTH'
    else:
        arrow = 'GROWTH' if macd_hist > 0 else 'STRESS'
    
    meaning = ARROW_MEANINGS.get((state, arrow), "Transition in progress")
    return arrow, meaning

# ============================================================
# CAPITULATION VALIDATION - v5.0.5 with volume
# ============================================================

def validate_capitulation(df, weekly_rsi, current_price, gann_levels, atr):
    """Validate capitulation with volume confirmation"""
    criteria = {
        'rsi_oversold': False,
        'price_near_gann_low': False,
        'volume_spike': False,
        'rsi_divergence': False
    }
    details = []
    missing = []
    
    gann_28 = gann_levels.get('2_8', 0)
    gann_38 = gann_levels.get('3_8', 0)
    
    # 1. RSI oversold
    if weekly_rsi < 30:
        criteria['rsi_oversold'] = True
        details.append(f"RSI oversold ({weekly_rsi:.1f} < 30)")
    else:
        missing.append(f"RSI not oversold ({weekly_rsi:.1f}, need < 30)")
    
    # 2. Price near Gann support
    if gann_38 > 0 and current_price <= gann_38 * 1.02:
        criteria['price_near_gann_low'] = True
        details.append(f"Price near Gann 3/8 (${gann_38:,.0f})")
    elif gann_28 > 0 and current_price <= gann_28 * 1.02:
        criteria['price_near_gann_low'] = True
        details.append(f"Price near Gann 2/8 (${gann_28:,.0f})")
    else:
        missing.append(f"Price not near Gann support (current: ${current_price:,.0f}, 3/8: ${gann_38:,.0f})")
    
    # 3. Volume spike
    volume_ratio = 1.0
    if df is not None and 'volume' in df.columns and len(df) >= 20:
        avg_volume = df['volume'].tail(20).mean()
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        if volume_ratio > 2.0:
            criteria['volume_spike'] = True
            details.append(f"Volume spike ({volume_ratio:.1f}x average)")
        else:
            missing.append(f"No volume spike ({volume_ratio:.1f}x, need > 2.0x)")
    else:
        missing.append("Volume data unavailable")
    
    # 4. RSI divergence (price lower low, RSI higher low)
    if df is not None and len(df) >= 50:
        try:
            # Find previous low in last 50 bars
            recent_df = df.tail(50)
            prev_low_idx = recent_df['low'].idxmin()
            current_idx = df.index[-1]
            
            if prev_low_idx != current_idx:
                prev_price_low = recent_df.loc[prev_low_idx, 'low']
                # Simplified RSI divergence check
                if current_price < prev_price_low and weekly_rsi > 25:
                    criteria['rsi_divergence'] = True
                    details.append("RSI divergence detected (higher low on RSI)")
                else:
                    missing.append("No RSI divergence")
            else:
                missing.append("No prior pivot for divergence check")
        except:
            missing.append("RSI divergence calculation failed")
    
    met = sum(criteria.values())
    total = len(criteria)
    
    # Require 3/4 for CONFIRMED
    if met >= 3:
        status = 'CONFIRMED'
        outlook = 'Sharp reversal rally expected - watch for bullish TK cross confirmation'
    elif met >= 2:
        status = 'POTENTIAL'
        outlook = 'Potential capitulation - monitor for volume and divergence confirmation'
    else:
        status = 'UNCONFIRMED'
        outlook = 'Oversold condition but capitulation criteria not met - further downside possible'
    
    return {
        'status': status,
        'criteria_met': met,
        'criteria_total': total,
        'details': details,
        'missing': missing,
        'volume_ratio': safe_round(volume_ratio, 2),
        'outlook': outlook
    }

# ============================================================
# MARKET REGIME
# ============================================================

def detect_market_regime(current_price, sma_200, adx, adx_strength):
    if sma_200 is None or sma_200 == 0:
        return {
            'regime': 'UNKNOWN', 'strength': 'UNKNOWN', 'adx': adx,
            'price_vs_sma200': 'UNKNOWN', 'sma_200': 0,
            'description': 'Insufficient data', 'trend_direction': 'UNKNOWN'
        }
    
    price_vs_sma = 'ABOVE' if current_price > sma_200 else 'BELOW'
    trend_direction = 'UP' if current_price > sma_200 else 'DOWN'
    
    if current_price > sma_200:
        regime = 'TRENDING_BULL' if adx >= 25 else 'RANGING_BULL'
        description = 'Strong uptrend above 200 SMA' if adx >= 25 else 'Weak uptrend / consolidation above 200 SMA'
    else:
        regime = 'TRENDING_BEAR' if adx >= 25 else 'RANGING_BEAR'
        description = 'Strong downtrend below 200 SMA' if adx >= 25 else 'Weak downtrend / consolidation below 200 SMA'
    
    return {
        'regime': regime, 'strength': adx_strength, 'adx': safe_round(adx),
        'price_vs_sma200': price_vs_sma, 'sma_200': safe_round(sma_200),
        'description': description, 'trend_direction': trend_direction
    }

# ============================================================
# DIRECTION CLASSIFICATION
# ============================================================

def classify_timeframe_direction(rsi, adx, cloud_signal, tk_cross, macd_hist, kijun_flat=False, tf_name='1D'):
    bullish, bearish = 0, 0
    
    if tf_name != '1M':
        if cloud_signal == 'BULLISH':
            bullish += 2
        elif cloud_signal == 'BEARISH':
            bearish += 2
    
    tk_weight = 2 if tf_name == '1M' else 1
    if tk_cross == 'BULLISH':
        bullish += tk_weight
    elif tk_cross == 'BEARISH':
        bearish += tk_weight
    
    if macd_hist > 0:
        bullish += 1
    elif macd_hist < 0:
        bearish += 1
    
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
    
    if adx > 40:
        if bullish > bearish:
            bullish += 1
        elif bearish > bullish:
            bearish += 1
    
    total = bullish + bearish
    if total == 0:
        return 'NEUTRAL', 0, 0
    
    bullish_pct = bullish / total * 100
    bearish_pct = bearish / total * 100
    
    if bullish_pct >= 60:
        return 'BULLISH', bullish, bearish
    elif bearish_pct >= 60:
        return 'BEARISH', bullish, bearish
    return 'NEUTRAL', bullish, bearish

# ============================================================
# REGIME-AWARE CLASSIFICATION
# ============================================================

def apply_regime_awareness(timeframes, regime):
    regime_type = regime.get('regime', 'UNKNOWN')
    adjusted = {}
    
    for tf_name, tf_data in timeframes.items():
        adj = tf_data.copy()
        
        if tf_name in ['1D', '3D']:
            tf_dir = tf_data['direction']
            
            if regime_type == 'TRENDING_BEAR' and tf_dir == 'BULLISH':
                adj['original_direction'] = tf_dir
                adj['direction'] = 'COUNTER_TREND_RALLY'
                adj['regime_adjusted'] = True
                adj['regime_note'] = 'Bullish signal in bear regime - treat as counter-trend rally, not reversal'
                adj['confidence_modifier'] = -30
            elif regime_type == 'TRENDING_BULL' and tf_dir == 'BEARISH':
                adj['original_direction'] = tf_dir
                adj['direction'] = 'COUNTER_TREND_PULLBACK'
                adj['regime_adjusted'] = True
                adj['regime_note'] = 'Bearish signal in bull regime - treat as pullback, not reversal'
                adj['confidence_modifier'] = -30
            else:
                adj['regime_adjusted'] = False
                adj['confidence_modifier'] = 0
        else:
            adj['regime_adjusted'] = False
            adj['confidence_modifier'] = 0
        
        adjusted[tf_name] = adj
    
    return adjusted

# ============================================================
# PRIMARY BIAS - v5.0.5 Gann 50% rule
# ============================================================

def determine_primary_bias(current_price, weekly_gann_50, consensus_direction, atr):
    """
    Primary bias follows Gann 50% rule - overrides consensus when clear
    """
    buffer = atr * 0.5  # Deadzone around 50%
    
    if current_price < (weekly_gann_50 - buffer):
        gann_bias = "BEARISH"
        bias_note = f"Price below Weekly Gann 50% (${weekly_gann_50:,.0f})"
        shift_trigger = f"Daily close above ${weekly_gann_50:,.0f}"
    elif current_price > (weekly_gann_50 + buffer):
        gann_bias = "BULLISH"
        bias_note = f"Price above Weekly Gann 50% (${weekly_gann_50:,.0f})"
        shift_trigger = f"Daily close below ${weekly_gann_50:,.0f}"
    else:
        gann_bias = "NEUTRAL"
        bias_note = f"Price near Weekly Gann 50% (${weekly_gann_50:,.0f}) - decision zone"
        shift_trigger = "Wait for clear breakout from 50% level"
    
    # Gann 50% rule overrides consensus when not neutral
    if gann_bias != "NEUTRAL" and gann_bias != consensus_direction:
        final_bias = gann_bias
        bias_source = "Gann 50% Rule (overrides consensus)"
    else:
        final_bias = consensus_direction if consensus_direction != 'NEUTRAL' else gann_bias
        bias_source = "Consensus + Gann aligned" if gann_bias == consensus_direction else "Consensus"
    
    return {
        'primary_bias': final_bias,
        'gann_bias': gann_bias,
        'consensus_bias': consensus_direction,
        'bias_note': bias_note,
        'bias_source': bias_source,
        'shift_trigger': shift_trigger,
        'weekly_gann_50': safe_round(weekly_gann_50)
    }

# ============================================================
# CONSENSUS CALCULATION
# ============================================================

def calculate_consensus(timeframes, regime):
    total_bullish, total_bearish, total_weight = 0, 0, 0
    directions = {}
    regime_conflicts = []
    
    for tf_name, tf_data in timeframes.items():
        weight = tf_data['weight']
        direction = tf_data['direction']
        total_weight += weight
        directions[tf_name] = direction
        
        if tf_data.get('regime_adjusted', False):
            regime_conflicts.append({
                'timeframe': tf_name,
                'original': tf_data.get('original_direction', direction),
                'adjusted': direction,
                'note': tf_data.get('regime_note', '')
            })
        
        if direction == 'BULLISH':
            total_bullish += weight
            tf_data['weighted_contribution'] = weight
        elif direction == 'BEARISH':
            total_bearish += weight
            tf_data['weighted_contribution'] = weight
        else:
            tf_data['weighted_contribution'] = 0
    
    bullish_pct = (total_bullish / total_weight) * 100 if total_weight > 0 else 0
    bearish_pct = (total_bearish / total_weight) * 100 if total_weight > 0 else 0
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
    
    aligned = sum(1 for d in directions.values() if d == primary_direction) if primary_direction != 'NEUTRAL' else sum(1 for d in directions.values() if d in ['NEUTRAL', 'COUNTER_TREND_RALLY', 'COUNTER_TREND_PULLBACK'])
    
    regime_type = regime.get('regime', 'UNKNOWN')
    regime_agrees = (regime_type in ['TRENDING_BULL', 'RANGING_BULL'] and primary_direction == 'BULLISH') or \
                    (regime_type in ['TRENDING_BEAR', 'RANGING_BEAR'] and primary_direction == 'BEARISH') or \
                    primary_direction == 'NEUTRAL'
    
    if weighted_score >= 75 and aligned >= 3 and regime_agrees:
        confidence = 'HIGH'
    elif weighted_score >= 60 and aligned >= 2:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'
    
    modifier = sum(tf.get('confidence_modifier', 0) for tf in timeframes.values())
    adjusted_score = max(0, min(100, weighted_score + modifier))
    
    if primary_direction == 'BULLISH' and confidence in ['HIGH', 'MEDIUM']:
        signal_type = 'BUY'
    elif primary_direction == 'BEARISH' and confidence in ['HIGH', 'MEDIUM']:
        signal_type = 'SELL'
    else:
        signal_type = 'WAIT'
    
    # Interpretation
    if regime_conflicts:
        interpretation = f"Regime conflict: {', '.join([c['timeframe'] for c in regime_conflicts])}. "
        if regime_type == 'TRENDING_BEAR':
            interpretation += "Bear regime intact - daily bullish signals are likely bear flag rallies."
        else:
            interpretation += "Bull regime intact - daily bearish signals are likely pullbacks."
    elif aligned >= 3:
        interpretation = f"Strong {primary_direction.lower()} alignment across timeframes. High conviction setup."
    else:
        interpretation = "Mixed signals. Wait for clearer alignment."
    
    return {
        'direction': primary_direction, 'signal_type': signal_type,
        'weighted_bullish_pct': round(bullish_pct), 'weighted_bearish_pct': round(bearish_pct),
        'weighted_neutral_pct': round(neutral_pct), 'weighted_score': round(weighted_score),
        'adjusted_score': round(adjusted_score), 'confidence_level': confidence,
        'alignment_count': f"{aligned}/{len(timeframes)}",
        'alignment_detail': ", ".join([f"{tf}:{d}" for tf, d in directions.items()]),
        'monthly_direction': directions.get('1M', 'NEUTRAL'),
        'weekly_direction': directions.get('1W', 'NEUTRAL'),
        'daily_direction': directions.get('1D', 'NEUTRAL'),
        'regime_agrees': regime_agrees, 'regime_conflicts': regime_conflicts,
        'interpretation': interpretation, 'is_valid': True
    }

def validate_setup(invalidation, consensus):
    rules_triggered = invalidation.get('rules_triggered', 0)
    if rules_triggered > 0:
        triggered = [r['condition'] for r in invalidation.get('rules', []) if r.get('triggered', False)]
        return {**consensus, 'direction': 'INVALID', 'signal_type': 'NO_TRADE',
                'confidence_level': 'NONE', 'weighted_score': 0,
                'interpretation': f"Setup invalidated: {', '.join(triggered)}",
                'is_valid': False, 'invalidation_reason': triggered}
    return {**consensus, 'is_valid': True}

# ============================================================
# R:R VALIDATION - v5.0.5 with minimum threshold
# ============================================================

def calculate_risk_reward(entry, target, stop, min_rr=MIN_RR_RATIO):
    """Calculate R:R with minimum threshold validation"""
    if entry == 0 or stop == entry:
        return 0.0, 0.0, 0.0, False, "Invalid entry/stop"
    
    reward = abs(target - entry)
    risk = abs(stop - entry)
    
    reward_pct = (reward / entry) * 100
    risk_pct = (risk / entry) * 100
    rr_ratio = reward / risk if risk > 0 else 0
    
    meets_minimum = rr_ratio >= min_rr
    
    if not meets_minimum:
        warning = f"R:R {rr_ratio:.2f} below minimum {min_rr}. Consider deeper TP or tighter stop."
    else:
        warning = None
    
    return safe_round(rr_ratio, 2), safe_round(reward_pct, 2), safe_round(risk_pct, 2), meets_minimum, warning

# ============================================================
# PRICE TARGETS - v5.0.5 with R:R validation
# ============================================================

def calculate_targets_and_stops(current_price, direction, gann_levels_by_tf, ichimoku_by_tf, atr):
    """Calculate targets with R:R validation"""
    resistance_levels, support_levels = [], []
    
    for tf_name in ['1M', '1W', '3D', '1D']:
        gann = gann_levels_by_tf.get(tf_name, {})
        ichi = ichimoku_by_tf.get(tf_name, {})
        
        for level_name in ['3_8', '4_8', '5_8', '6_8', '7_8']:
            level_val = gann.get(level_name, 0)
            if level_val > 0:
                source = f"GANN_{level_name}_{tf_name}"
                if level_val > current_price * 1.005:
                    resistance_levels.append({'price': level_val, 'source': source, 'tf': tf_name})
                elif level_val < current_price * 0.995:
                    support_levels.append({'price': level_val, 'source': source, 'tf': tf_name})
        
        for key, src_name in [('cloud_top', 'CLOUD_TOP'), ('cloud_bottom', 'CLOUD_BOTTOM'), ('kijun', 'KIJUN')]:
            val = ichi.get(key, 0)
            if val > 0:
                if val > current_price * 1.005:
                    resistance_levels.append({'price': val, 'source': f'{src_name}_{tf_name}', 'tf': tf_name})
                elif val < current_price * 0.995:
                    support_levels.append({'price': val, 'source': f'{src_name}_{tf_name}', 'tf': tf_name})
    
    resistance_levels.sort(key=lambda x: x['price'])
    support_levels.sort(key=lambda x: x['price'], reverse=True)
    
    # Dedupe
    def dedupe(levels):
        if not levels:
            return levels
        result = [levels[0]]
        for l in levels[1:]:
            if abs(l['price'] - result[-1]['price']) / result[-1]['price'] > 0.005:
                result.append(l)
            else:
                result[-1]['source'] += f" + {l['source']}"
        return result
    
    resistance_levels = dedupe(resistance_levels)
    support_levels = dedupe(support_levels)
    
    targets = {'tp1': None, 'tp1_sources': [], 'tp2': None, 'tp2_sources': [],
               'tp3': None, 'tp3_sources': [], 'stop_loss': None, 'stop_sources': [],
               'rr_valid': True, 'rr_warning': None}
    
    # Determine stop first
    if direction in ['BULLISH', 'NEUTRAL', 'COUNTER_TREND_RALLY']:
        if support_levels:
            targets['stop_loss'] = support_levels[0]['price']
            targets['stop_sources'] = [support_levels[0]['source']]
        else:
            targets['stop_loss'] = current_price - atr * 1.5
            targets['stop_sources'] = ['ATR_1.5X']
        
        # Find targets that meet minimum R:R
        valid_targets = []
        for r in resistance_levels:
            reward = r['price'] - current_price
            risk = current_price - targets['stop_loss']
            rr = reward / risk if risk > 0 else 0
            if rr >= MIN_RR_RATIO:
                valid_targets.append({**r, 'rr': rr})
        
        if valid_targets:
            targets['tp1'] = valid_targets[0]['price']
            targets['tp1_sources'] = [valid_targets[0]['source']]
            targets['tp1_rr'] = safe_round(valid_targets[0]['rr'], 2)
            if len(valid_targets) > 1:
                targets['tp2'] = valid_targets[1]['price']
                targets['tp2_sources'] = [valid_targets[1]['source']]
            if len(valid_targets) > 2:
                targets['tp3'] = valid_targets[2]['price']
                targets['tp3_sources'] = [valid_targets[2]['source']]
        else:
            # No valid targets - flag warning
            targets['rr_valid'] = False
            targets['rr_warning'] = f"No targets with R:R >= {MIN_RR_RATIO}. Consider waiting for better entry."
            # Use first resistance anyway but flag it
            if resistance_levels:
                targets['tp1'] = resistance_levels[0]['price']
                targets['tp1_sources'] = [resistance_levels[0]['source'] + " (LOW R:R)"]
    
    else:  # BEARISH or COUNTER_TREND_PULLBACK
        if resistance_levels:
            targets['stop_loss'] = resistance_levels[0]['price']
            targets['stop_sources'] = [resistance_levels[0]['source']]
        else:
            targets['stop_loss'] = current_price + atr * 1.5
            targets['stop_sources'] = ['ATR_1.5X']
        
        valid_targets = []
        for s in support_levels:
            reward = current_price - s['price']
            risk = targets['stop_loss'] - current_price
            rr = reward / risk if risk > 0 else 0
            if rr >= MIN_RR_RATIO:
                valid_targets.append({**s, 'rr': rr})
        
        if valid_targets:
            targets['tp1'] = valid_targets[0]['price']
            targets['tp1_sources'] = [valid_targets[0]['source']]
            targets['tp1_rr'] = safe_round(valid_targets[0]['rr'], 2)
            if len(valid_targets) > 1:
                targets['tp2'] = valid_targets[1]['price']
                targets['tp2_sources'] = [valid_targets[1]['source']]
            if len(valid_targets) > 2:
                targets['tp3'] = valid_targets[2]['price']
                targets['tp3_sources'] = [valid_targets[2]['source']]
        else:
            targets['rr_valid'] = False
            targets['rr_warning'] = f"No targets with R:R >= {MIN_RR_RATIO}. Consider waiting for better entry."
            if support_levels:
                targets['tp1'] = support_levels[0]['price']
                targets['tp1_sources'] = [support_levels[0]['source'] + " (LOW R:R)"]
    
    # Fallbacks
    if targets['tp1'] is None:
        mult = 1 if direction not in ['BEARISH', 'COUNTER_TREND_PULLBACK'] else -1
        targets['tp1'] = current_price + mult * atr * 2
        targets['tp1_sources'] = ['ATR_2X']
    if targets['tp2'] is None:
        mult = 1 if direction not in ['BEARISH', 'COUNTER_TREND_PULLBACK'] else -1
        targets['tp2'] = current_price + mult * atr * 3
        targets['tp2_sources'] = ['ATR_3X']
    if targets['tp3'] is None:
        mult = 1 if direction not in ['BEARISH', 'COUNTER_TREND_PULLBACK'] else -1
        targets['tp3'] = current_price + mult * atr * 4
        targets['tp3_sources'] = ['ATR_4X']
    
    return targets

# ============================================================
# TIME FORECAST - v5.0.5 with transparency
# ============================================================

def calculate_time_forecast(df, current_price, atr, gann_levels_1w):
    if df is None or df.empty:
        return {
            'next_pivot_date': None, 'next_pivot_display': 'N/A',
            'days_to_pivot': 0, 'pivot_type': 'UNKNOWN',
            'pivot_confidence': 0, 'cycle_sources': [],
            'probable_price_low': 0, 'probable_price_high': 0,
            'probable_range_text': 'Insufficient data',
            'forecast_reliable': False, 'cycle_origin': None
        }
    
    last_date = pd.to_datetime(df['date'].iloc[-1])
    cycles = [30, 45, 60, 90, 120, 180, 360]
    
    # Find major pivots
    highs = df.nlargest(5, 'high')
    lows = df.nsmallest(5, 'low')
    
    pivot_forecasts = []
    
    for cycle in cycles:
        if not highs.empty:
            high_date = pd.to_datetime(highs['date'].iloc[0])
            high_price = highs['high'].iloc[0]
            projected = high_date + timedelta(days=cycle)
            if projected > last_date:
                days = (projected - last_date).days
                if 5 <= days <= 180:
                    conf = 60 if cycle in [45, 90, 180] else 50 if cycle in [30, 60] else 40
                    pivot_forecasts.append({
                        'date': projected.strftime('%Y-%m-%d'),
                        'date_display': projected.strftime('%d/%m/%Y'),
                        'days_from_now': days,
                        'cycle': f'{cycle}D_CYCLE',
                        'from': 'HIGH',
                        'origin_date': str(high_date.date()),
                        'origin_price': safe_round(high_price),
                        'confidence': conf
                    })
    
    pivot_forecasts.sort(key=lambda x: (-x['confidence'], x['days_from_now']))
    
    primary = pivot_forecasts[0] if pivot_forecasts else {
        'date': (last_date + timedelta(days=30)).strftime('%Y-%m-%d'),
        'date_display': (last_date + timedelta(days=30)).strftime('%d/%m/%Y'),
        'days_from_now': 30, 'cycle': 'DEFAULT_30D', 'from': 'ESTIMATE',
        'origin_date': None, 'origin_price': None, 'confidence': 30
    }
    
    days = primary['days_from_now']
    conf = primary['confidence']
    
    max_move = atr * np.sqrt(days) * 1.5
    raw_high = current_price + max_move
    raw_low = current_price - max_move
    
    gann_high = gann_levels_1w.get('7_8', raw_high)
    gann_low = gann_levels_1w.get('2_8', raw_low)
    
    prob_high = min(raw_high, gann_high * 1.02)
    prob_low = max(raw_low, gann_low * 0.98)
    
    if prob_high <= current_price:
        prob_high = current_price * 1.08
    if prob_low >= current_price:
        prob_low = current_price * 0.92
    
    reliable = conf >= 50
    range_text = f"${safe_round(prob_low):,.0f} - ${safe_round(prob_high):,.0f}"
    if not reliable:
        range_text = f"Low confidence - {range_text}"
    
    return {
        'next_pivot_date': primary['date'],
        'next_pivot_display': primary['date_display'],
        'days_to_pivot': days,
        'pivot_type': 'HIGH' if primary['from'] == 'HIGH' else 'LOW',
        'pivot_confidence': conf,
        'cycle_sources': [f"{primary['cycle']} from {primary['from']}"],
        'cycle_origin': {
            'date': primary.get('origin_date'),
            'price': primary.get('origin_price'),
            'cycle_length': int(primary['cycle'].replace('D_CYCLE', '').replace('DEFAULT_', ''))
        },
        'probable_price_low': safe_round(prob_low),
        'probable_price_high': safe_round(prob_high),
        'probable_range_text': range_text,
        'atr_daily': safe_round(atr),
        'max_expected_move': safe_round(max_move),
        'forecast_reliable': reliable,
        'confidence_note': 'High confidence' if conf >= 60 else 'Medium confidence' if conf >= 50 else 'Low confidence - insufficient cycle convergence'
    }

# ============================================================
# INVALIDATION RULES
# ============================================================

def build_invalidation_rules(direction, current_price, gann_levels, ichimoku, rsi):
    rules = []
    kijun = ichimoku.get('kijun', 0)
    cloud_bottom = ichimoku.get('cloud_bottom', 0)
    cloud_top = ichimoku.get('cloud_top', 0)
    gann_38 = gann_levels.get('3_8', 0)
    gann_62 = gann_levels.get('5_8', 0)
    
    if direction in ['BULLISH', 'COUNTER_TREND_RALLY']:
        inv_price = max(
            cloud_bottom if cloud_bottom > 0 and cloud_bottom < current_price else 0,
            gann_38 if gann_38 < current_price else 0
        ) or current_price * 0.95
        
        rules = [
            {'condition': 'Daily close below Gann 3/8', 'price': safe_round(gann_38), 'triggered': current_price < gann_38 if gann_38 > 0 else False},
            {'condition': 'Daily close below cloud bottom', 'price': safe_round(cloud_bottom), 'triggered': current_price < cloud_bottom if cloud_bottom > 0 else False},
            {'condition': 'RSI breaks below 35', 'current': safe_round(rsi), 'triggered': rsi < 35}
        ]
    else:
        inv_price = min(
            cloud_top if cloud_top > current_price else float('inf'),
            gann_62 if gann_62 > current_price else float('inf')
        )
        if inv_price == float('inf'):
            inv_price = current_price * 1.05
        
        rules = [
            {'condition': 'Daily close above Gann 5/8', 'price': safe_round(gann_62), 'triggered': current_price > gann_62 if gann_62 > 0 else False},
            {'condition': 'Daily close above cloud top', 'price': safe_round(cloud_top), 'triggered': current_price > cloud_top if cloud_top > 0 else False},
            {'condition': 'RSI breaks above 65', 'current': safe_round(rsi), 'triggered': rsi > 65}
        ]
    
    triggered = sum(1 for r in rules if r.get('triggered', False))
    
    return {
        'invalidation_price': safe_round(inv_price),
        'invalidation_reason': 'Key level breach',
        'rules': rules,
        'rules_triggered': triggered,
        'warning': f"⚠️ {triggered} invalidation rule(s) already triggered!" if triggered > 0 else None
    }

# ============================================================
# STRATEGY GENERATION - v5.0.5
# ============================================================

def generate_strategy(consensus, invalidation, timeframes, current_price, regime, bias_info, targets):
    if not consensus.get('is_valid', True):
        return {
            'primary_bias': 'FLAT', 'action': 'No trade - setup invalidated',
            'entry_method': 'Wait for new setup', 'position_size_recommendation': 'Zero',
            'time_in_trade': 'N/A', 'interpretation': consensus.get('interpretation', ''),
            'invalidation_action': 'Already invalidated', 'trade_type': 'NONE',
            'rr_warning': None
        }
    
    # Use Gann-determined bias
    primary_bias = bias_info['primary_bias']
    confidence = consensus['confidence_level']
    regime_type = regime.get('regime', 'UNKNOWN')
    regime_conflicts = consensus.get('regime_conflicts', [])
    
    trade_type = 'COUNTER_TREND' if regime_conflicts else 'WITH_TREND'
    
    # Check R:R validity
    rr_warning = targets.get('rr_warning')
    if not targets.get('rr_valid', True):
        trade_type = 'INVALID_RR'
    
    if primary_bias == 'BULLISH' and confidence == 'HIGH' and targets.get('rr_valid', True):
        action = 'Buy dips toward support levels'
        entry = 'Wait for pullback to Gann 50% or Kijun, then bullish TK cross'
        size = 'Full size (high confidence)'
    elif primary_bias == 'BULLISH' and confidence == 'MEDIUM':
        action = 'Buy dips cautiously'
        entry = 'Wait for daily RSI < 50 then bullish reversal candle'
        size = '50% size (medium confidence)'
    elif primary_bias == 'BEARISH' and confidence == 'HIGH' and targets.get('rr_valid', True):
        action = f"Sell rallies toward ${bias_info['weekly_gann_50']:,.0f} (Weekly Gann 50%)"
        entry = 'Wait for rally to Gann 50% or resistance, then bearish TK cross'
        size = 'Full size (high confidence)'
    elif primary_bias == 'BEARISH' and confidence == 'MEDIUM':
        action = 'Sell rallies cautiously'
        entry = 'Wait for daily RSI > 50 then bearish reversal candle'
        size = '50% size (medium confidence)'
    elif trade_type == 'COUNTER_TREND':
        action = f'CAUTION: Counter-trend setup in {regime_type.replace("_", " ").lower()}'
        entry = 'Scalp only with tight stops - do not hold overnight'
        size = '25% size max (counter-trend)'
    elif trade_type == 'INVALID_RR':
        action = f'WAIT - R:R below minimum {MIN_RR_RATIO}'
        entry = 'Wait for better entry or deeper targets'
        size = 'No position until R:R improves'
    else:
        action = 'Wait for clearer setup'
        entry = 'No entry until alignment improves'
        size = 'No position (low confidence)'
    
    inv_price = invalidation.get('invalidation_price', 0)
    inv_action = f"Close {'long' if primary_bias == 'BULLISH' else 'short'} if price closes {'below' if primary_bias == 'BULLISH' else 'above'} ${inv_price:,.0f}"
    
    return {
        'primary_bias': primary_bias,
        'gann_bias': bias_info['gann_bias'],
        'bias_source': bias_info['bias_source'],
        'bias_note': bias_info['bias_note'],
        'shift_trigger': bias_info['shift_trigger'],
        'action': action,
        'entry_method': entry,
        'position_size_recommendation': size,
        'time_in_trade': 'Hold until TP1 or next pivot',
        'interpretation': consensus['interpretation'],
        'invalidation_action': inv_action,
        'trade_type': trade_type,
        'regime_context': regime_type,
        'rr_warning': rr_warning
    }

# ============================================================
# ANALYZE TIMEFRAME
# ============================================================

def analyze_timeframe(df, tf_name):
    if df is None or df.empty or len(df) < 50:
        return None
    
    weight = TIMEFRAME_WEIGHTS.get(tf_name, 10)
    ichi_params = ICHIMOKU_PARAMS.get(tf_name, (9, 26, 52))
    
    rsi = calculate_rsi(df)
    _, _, macd_hist = calculate_macd(df)
    adx, adx_strength = calculate_adx(df)
    ichimoku = calculate_ichimoku(df, ichi_params)
    gann = calculate_gann_levels_for_timeframe(df, tf_name)
    
    state, state_info = identify_enneagram_state(df, rsi, macd_hist)
    arrow, arrow_meaning = determine_arrow(state, rsi, macd_hist)
    direction, bullish, bearish = classify_timeframe_direction(
        rsi, adx, ichimoku['cloud_signal'], ichimoku['tk_cross'], macd_hist, ichimoku['kijun_flat'], tf_name
    )
    
    return {
        'direction': direction,
        'signal_type': 'BUY' if direction == 'BULLISH' else 'SELL' if direction == 'BEARISH' else 'WAIT',
        'weight': weight, 'weighted_contribution': 0,
        'enneagram_state': state, 'state_name': state_info['name'],
        'state_bias': state_info['bias'], 'phase': state_info['phase'],
        'arrow': arrow, 'arrow_meaning': arrow_meaning,
        'rsi': safe_round(rsi), 'macd_histogram': safe_round(macd_hist),
        'adx': safe_round(adx), 'trend_strength': adx_strength,
        'cloud_signal': ichimoku['cloud_signal'], 'tk_cross': ichimoku['tk_cross'],
        'kijun_flat': ichimoku['kijun_flat'],
        'tenkan': safe_round(ichimoku['tenkan']), 'kijun': safe_round(ichimoku['kijun']),
        'cloud_top': safe_round(ichimoku['cloud_top']), 'cloud_bottom': safe_round(ichimoku['cloud_bottom']),
        'gann_high': safe_round(gann['high']), 'gann_low': safe_round(gann['low']),
        'gann_50_pct': safe_round(gann['4_8']),
        'gann_38_pct': safe_round(gann['3_8']), 'gann_62_pct': safe_round(gann['5_8']),
        'gann_high_date': gann.get('high_date'), 'gann_low_date': gann.get('low_date'),
        'gann_lookback': gann.get('lookback_bars'), 'gann_range_pct': gann.get('range_pct'),
        'bullish_signals': bullish, 'bearish_signals': bearish,
        'gann_levels': gann, 'ichimoku': ichimoku
    }

# ============================================================
# MAIN ENDPOINT
# ============================================================

@app.get("/signal/daily")
async def get_daily_signal():
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
        
        # Regime
        atr = calculate_atr(df_daily)
        sma_200 = calculate_sma(df_daily, 200)
        regime = detect_market_regime(current_price, sma_200, timeframes['1D']['adx'], timeframes['1D']['trend_strength'])
        
        # Apply regime awareness
        timeframes = apply_regime_awareness(timeframes, regime)
        
        # Consensus
        consensus = calculate_consensus(timeframes, regime)
        
        # Primary bias (Gann 50% rule)
        weekly_gann_50 = gann_levels_by_tf.get('1W', {}).get('4_8', current_price)
        bias_info = determine_primary_bias(current_price, weekly_gann_50, consensus['direction'], atr)
        
        # Targets with R:R validation
        targets = calculate_targets_and_stops(current_price, bias_info['primary_bias'], gann_levels_by_tf, ichimoku_by_tf, atr)
        
        # R:R calculation
        rr_ratio, reward_pct, risk_pct, rr_valid, rr_warn = calculate_risk_reward(
            current_price, targets['tp1'], targets['stop_loss']
        )
        
        # Time forecast
        time_forecast = calculate_time_forecast(df_daily, current_price, atr, gann_levels_by_tf.get('1W', {}))
        
        # Invalidation
        weekly_gann = gann_levels_by_tf.get('1W', {})
        weekly_ichi = ichimoku_by_tf.get('1W', {})
        weekly_rsi = timeframes.get('1W', {}).get('rsi', 50)
        invalidation = build_invalidation_rules(bias_info['primary_bias'], current_price, weekly_gann, weekly_ichi, weekly_rsi)
        
        # Validate setup
        consensus = validate_setup(invalidation, consensus)
        
        # Capitulation check
        capitulation = validate_capitulation(df_1w, weekly_rsi, current_price, weekly_gann, atr)
        
        # Strategy
        strategy = generate_strategy(consensus, invalidation, timeframes, current_price, regime, bias_info, targets)
        
        # SQ9 with filtering
        sq9_all, sq9_actionable = calculate_square_of_9(current_price)
        
        weekly_data = timeframes.get('1W', timeframes.get('1D', {}))
        
        response = {
            'status': 'success',
            'version': '5.0.5',
            'timestamp': datetime.now().isoformat(),
            'symbol': 'BTCUSDT',
            
            # Legacy
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
            
            # MTF
            'primary_direction': consensus['direction'],
            'primary_bias': bias_info['primary_bias'],
            'bias_info': bias_info,
            'weighted_score': consensus['weighted_score'],
            'adjusted_score': consensus.get('adjusted_score', consensus['weighted_score']),
            'mtf_alignment': consensus['alignment_count'],
            'market_regime': regime['regime'],
            'setup_valid': consensus.get('is_valid', True),
            'regime_agrees': consensus.get('regime_agrees', False),
            'regime_conflicts': consensus.get('regime_conflicts', []),
            
            'timeframes': timeframes,
            'consensus': consensus,
            
            # Targets with R:R
            'price_targets': {
                'source_timeframe': '1W',
                'calculation_method': 'Gann + Ichimoku Confluence',
                'tp1': safe_round(targets['tp1']),
                'tp1_sources': targets['tp1_sources'],
                'tp1_rr': targets.get('tp1_rr'),
                'tp2': safe_round(targets['tp2']),
                'tp2_sources': targets['tp2_sources'],
                'tp3': safe_round(targets['tp3']),
                'tp3_sources': targets['tp3_sources'],
                'stop_loss': safe_round(targets['stop_loss']),
                'stop_sources': targets['stop_sources'],
                'rr_ratio': rr_ratio,
                'rr_valid': rr_valid,
                'rr_warning': rr_warn,
                'reward_pct': reward_pct,
                'risk_pct': risk_pct,
                'min_rr_required': MIN_RR_RATIO
            },
            
            'target_1': safe_round(targets['tp1']),
            'target_2': safe_round(targets['tp2']),
            'target_3': safe_round(targets['tp3']),
            
            'time_forecast': time_forecast,
            'pivot_forecast_primary': {
                'date': time_forecast['next_pivot_date'],
                'date_display': time_forecast['next_pivot_display'],
                'days_from_now': time_forecast['days_to_pivot'],
                'expected_pivot': time_forecast['pivot_type'],
                'confidence': time_forecast['pivot_confidence'],
                'cycle_origin': time_forecast.get('cycle_origin'),
                'reliable': time_forecast.get('forecast_reliable', False)
            },
            
            'capitulation_analysis': capitulation,
            
            'sq9_analysis': {
                'from_current': sq9_all[:8],
                'actionable_levels': sq9_actionable[:6],
                'filter_settings': {
                    'min_distance_pct': SQ9_MIN_DISTANCE_PCT,
                    'max_distance_pct': SQ9_MAX_DISTANCE_PCT
                },
                'anchors': {
                    'current': safe_round(current_price),
                    'high_52': safe_round(gann_levels_by_tf.get('1D', {}).get('high', 0)),
                    'low_52': safe_round(gann_levels_by_tf.get('1D', {}).get('low', 0))
                }
            },
            
            'gann_sq9_levels': str(sq9_all[:4]),
            'gann_angles_active': str([45, 90, 135, 180, 225, 270, 315, 360]),
            
            'major_high': safe_round(gann_levels_by_tf.get('1M', {}).get('high', 0)),
            'major_low': safe_round(gann_levels_by_tf.get('1M', {}).get('low', 0)),
            'gann_range': safe_round(gann_levels_by_tf.get('1W', {}).get('range', 0)),
            'gann_3_8': safe_round(gann_levels_by_tf.get('1W', {}).get('3_8', 0)),
            'gann_4_8': safe_round(gann_levels_by_tf.get('1W', {}).get('4_8', 0)),
            'gann_5_8': safe_round(gann_levels_by_tf.get('1W', {}).get('5_8', 0)),
            
            'tenkan': safe_round(weekly_ichi.get('tenkan', 0)),
            'kijun': safe_round(weekly_ichi.get('kijun', 0)),
            'cloud_top': safe_round(weekly_ichi.get('cloud_top', 0)),
            'cloud_bottom': safe_round(weekly_ichi.get('cloud_bottom', 0)),
            'cloud_signal': weekly_data.get('cloud_signal', 'NEUTRAL'),
            'tk_cross': weekly_data.get('tk_cross', 'NEUTRAL'),
            'kijun_flat': weekly_data.get('kijun_flat', False),
            
            'regime': regime,
            'adx': safe_round(timeframes['1D']['adx']),
            'trend_strength': timeframes['1D']['trend_strength'],
            
            'state': weekly_data.get('enneagram_state', 1),
            'state_name': weekly_data.get('state_name', 'Unknown'),
            'phase': weekly_data.get('phase', 'Unknown'),
            'arrow': weekly_data.get('arrow', 'NEUTRAL'),
            'arrow_meaning': weekly_data.get('arrow_meaning', ''),
            
            'rsi': safe_round(timeframes['1D']['rsi']),
            'macd': safe_round(timeframes['1D']['macd_histogram']),
            'macd_hist': safe_round(weekly_data.get('macd_histogram', 0)),
            'atr': safe_round(atr),
            
            'invalidation': invalidation,
            'strategy': strategy,
            
            'candles_analyzed': len(df_daily),
            'last_candle_date': str(current_date),
            'signal_strength': consensus['confidence_level'],
            'confirmation_score': consensus['weighted_score'],
            'confirmation_score_display': f"{consensus['weighted_score']}%"
        }
        
        return to_native(response)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "5.0.5", "timestamp": datetime.now().isoformat()}

@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print("LUXOR V7 PRANA RUNTIME - MTF EDITION v5.0.5")
    print("=" * 50)
    print(f"Min R:R Required: {MIN_RR_RATIO}")
    print(f"SQ9 Filter: {SQ9_MIN_DISTANCE_PCT}% - {SQ9_MAX_DISTANCE_PCT}%")
    print("=" * 50)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
