"""
LUXOR V7 PRANA RUNTIME - MULTI-TIMEFRAME EDITION
Version: 5.0.0
Complete Gann-aligned multi-timeframe analysis system
"""

from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta
import uvicorn
import json
import math
import pandas as pd
import numpy as np
from luxor_v7_prana import LuxorV7PranaSystem
from config import INITIAL_CAPITAL, API_HOST, API_PORT

app = FastAPI(title="LUXOR V7 PRANA Runtime", version="5.0.0")

# Initialize system
luxor = LuxorV7PranaSystem(initial_capital=INITIAL_CAPITAL)

# ============================================================================
# CONFIGURATION
# ============================================================================

TIMEFRAME_WEIGHTS = {
    '1M': 40,
    '1W': 35,
    '3D': 15,
    '1D': 10
}

ICHIMOKU_PARAMS = {
    '1D': (9, 26, 52),
    '3D': (9, 26, 52),
    '1W': (9, 26, 26),  # Gann-aligned for weekly
    '1M': (9, 26, 52)
}

GANN_DAY_CYCLES = [7, 14, 21, 30, 45, 60, 90, 120, 135, 150, 180, 270, 360]
GANN_WEEK_CYCLES = [13, 26, 39, 52, 78]
GANN_MONTH_CYCLES = [3, 6, 12, 24]

# ============================================================================
# ENNEAGRAM STATE DEFINITIONS
# ============================================================================

ENNEAGRAM_STATES = {
    1: {"name": "Initiation", "phase": "Early accumulation / new up-cycle start", "stress": 4, "growth": 7, "bias": "BULLISH"},
    2: {"name": "Acceleration", "phase": "Momentum builds / trend strengthens", "stress": 8, "growth": 4, "bias": "BULLISH"},
    3: {"name": "Peak Formation", "phase": "Topping pattern emerging", "stress": 9, "growth": 6, "bias": "BEARISH"},
    4: {"name": "Retracement", "phase": "Healthy pullback / correction", "stress": 2, "growth": 1, "bias": "BEARISH"},
    5: {"name": "Capitulation", "phase": "Panic selling / washout phase", "stress": 7, "growth": 8, "bias": "BULLISH"},
    6: {"name": "Accumulation", "phase": "Smart money accumulating / base building", "stress": 3, "growth": 9, "bias": "BULLISH"},
    7: {"name": "Expansion", "phase": "Breakout / rapid expansion phase", "stress": 1, "growth": 5, "bias": "BULLISH"},
    8: {"name": "Distribution", "phase": "Late-stage markup / distribution begins", "stress": 5, "growth": 2, "bias": "BEARISH"},
    9: {"name": "Equilibrium", "phase": "Consolidation / range-bound balance", "stress": 6, "growth": 3, "bias": "NEUTRAL"}
}

ARROW_MEANINGS = {
    (1, "STRESS", 4): "Early accumulation interrupted - expect pullback before uptrend resumes",
    (1, "GROWTH", 7): "Accumulation complete - healthy expansion phase beginning",
    (2, "STRESS", 8): "Momentum stalling - distribution phase beginning, reduce exposure",
    (2, "GROWTH", 4): "Momentum maturing - healthy retracement before continuation",
    (3, "STRESS", 9): "Peak confirmed - consolidation/reversal imminent",
    (3, "GROWTH", 6): "Peak transitioning - orderly accumulation at lower levels",
    (4, "STRESS", 2): "Pullback accelerating - may become deeper correction",
    (4, "GROWTH", 1): "Retracement complete - new uptrend initiating",
    (5, "STRESS", 7): "Capitulation ending - sharp reversal rally expected",
    (5, "GROWTH", 8): "Capitulation bottom - strong recovery rally underway",
    (6, "STRESS", 3): "Accumulation failed - another leg down likely",
    (6, "GROWTH", 9): "Accumulation successful - equilibrium before breakout",
    (7, "STRESS", 1): "Expansion exhausted - return to accumulation phase",
    (7, "GROWTH", 5): "Expansion overheated - expect volatility spike/washout",
    (8, "STRESS", 5): "Distribution complete - capitulation phase starting",
    (8, "GROWTH", 2): "Distribution to momentum - trend continuation likely",
    (9, "STRESS", 6): "Balance broken - accumulation or breakdown ahead",
    (9, "GROWTH", 3): "Equilibrium ending - peak formation or breakdown"
}

# ============================================================================
# DATA RESAMPLING FUNCTIONS
# ============================================================================

def resample_ohlcv(df, timeframe):
    """
    Resample daily OHLCV data to higher timeframes using proper OHLC aggregation.
    Gann emphasized using true highs/lows within periods.
    """
    if df is None or len(df) == 0:
        return None
    
    # Ensure we have a datetime index
    df_copy = df.copy()
    if 'date' in df_copy.columns:
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy = df_copy.set_index('date')
    
    # Define resampling rules (compatible with pandas 1.x and 2.x)
    resample_map = {
        '1D': '1D',
        '3D': '3D',
        '1W': 'W',
        '1M': 'M'  # Changed from 'ME' to 'M' for pandas 1.x compatibility
    }
    
    if timeframe not in resample_map:
        return df_copy
    
    rule = resample_map[timeframe]
    
    try:
        # OHLC resampling - preserves true price action
        resampled = df_copy.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    except Exception as e:
        print(f"[WARNING] Resample failed for {timeframe}: {e}")
        return None
    
    # Reset index to have date as column
    resampled = resampled.reset_index()
    resampled = resampled.rename(columns={'index': 'date'})
    
    return resampled



def prepare_timeframe_data(df_daily):
    """
    Prepare data for all timeframes from daily data.
    Returns dict with DataFrames for each timeframe.
    """
    timeframes = {}
    
    # Daily - use as-is
    timeframes['1D'] = df_daily.copy()
    
    # 3-Day
    timeframes['3D'] = resample_ohlcv(df_daily, '3D')
    
    # Weekly
    timeframes['1W'] = resample_ohlcv(df_daily, '1W')
    
    # Monthly
    timeframes['1M'] = resample_ohlcv(df_daily, '1M')
    
    return timeframes


# ============================================================================
# INDICATOR CALCULATION FUNCTIONS
# ============================================================================

def calculate_rsi(df, period=14):
    """Calculate RSI"""
    if len(df) < period + 1:
        return 50.0
    
    close = df['close'].values
    deltas = np.diff(close)
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return round(rsi, 2)


def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD and histogram"""
    if len(df) < slow + signal:
        return {'macd': 0, 'signal': 0, 'histogram': 0}
    
    close = df['close']
    
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return {
        'macd': round(float(macd_line.iloc[-1]), 4),
        'signal': round(float(signal_line.iloc[-1]), 4),
        'histogram': round(float(histogram.iloc[-1]), 4)
    }


def calculate_adx(df, period=14):
    """Calculate ADX for trend strength"""
    if len(df) < period * 2:
        return {'adx': 25.0, 'trend_strength': 'MODERATE', 'plus_di': 0, 'minus_di': 0}
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # True Range
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(abs(high[1:] - close[:-1]), 
                              abs(low[1:] - close[:-1])))
    
    # Directional Movement
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]),
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                        np.maximum(low[:-1] - low[1:], 0), 0)
    
    # Smoothed averages
    atr = np.mean(tr[-period:])
    plus_di = 100 * np.mean(plus_dm[-period:]) / atr if atr > 0 else 0
    minus_di = 100 * np.mean(minus_dm[-period:]) / atr if atr > 0 else 0
    
    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
    adx = dx  # Simplified - would normally smooth
    
    # Trend strength classification
    if adx > 50:
        trend_strength = "STRONG"
    elif adx > 25:
        trend_strength = "MODERATE"
    else:
        trend_strength = "WEAK"
    
    return {
        'adx': round(adx, 2),
        'trend_strength': trend_strength,
        'plus_di': round(plus_di, 2),
        'minus_di': round(minus_di, 2)
    }


def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    if len(df) < period + 1:
        return float(df['close'].iloc[-1] * 0.025)
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(abs(high[1:] - close[:-1]),
                              abs(low[1:] - close[:-1])))
    
    atr = np.mean(tr[-period:])
    return round(float(atr), 2)


def calculate_sma(df, period=200):
    """Calculate Simple Moving Average"""
    if len(df) < period:
        return float(df['close'].mean())
    
    return round(float(df['close'].iloc[-period:].mean()), 2)


def calculate_ichimoku(df, tenkan_period=9, kijun_period=26, senkou_period=52):
    """Calculate Ichimoku Cloud components"""
    if len(df) < senkou_period:
        close = float(df['close'].iloc[-1])
        return {
            'tenkan': close,
            'kijun': close,
            'senkou_a': close,
            'senkou_b': close,
            'cloud_top': close,
            'cloud_bottom': close,
            'cloud_signal': 'NEUTRAL',
            'tk_cross': 'NEUTRAL',
            'price_vs_cloud': 'INSIDE',
            'kijun_flat': False,
            'chikou_signal': 'NEUTRAL'
        }
    
    high = df['high']
    low = df['low']
    close = df['close']
    current_price = float(close.iloc[-1])
    
    # Tenkan-sen (Conversion Line)
    tenkan = (high.iloc[-tenkan_period:].max() + low.iloc[-tenkan_period:].min()) / 2
    
    # Kijun-sen (Base Line)
    kijun = (high.iloc[-kijun_period:].max() + low.iloc[-kijun_period:].min()) / 2
    
    # Senkou Span A (Leading Span A) - projected 26 periods ahead
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B) - projected 26 periods ahead
    senkou_b = (high.iloc[-senkou_period:].max() + low.iloc[-senkou_period:].min()) / 2
    
    cloud_top = max(senkou_a, senkou_b)
    cloud_bottom = min(senkou_a, senkou_b)
    
    # Cloud signal
    if current_price > cloud_top:
        cloud_signal = "BULLISH"
        price_vs_cloud = "ABOVE"
    elif current_price < cloud_bottom:
        cloud_signal = "BEARISH"
        price_vs_cloud = "BELOW"
    else:
        cloud_signal = "NEUTRAL"
        price_vs_cloud = "INSIDE"
    
    # TK Cross
    if tenkan > kijun:
        tk_cross = "BULLISH"
    elif tenkan < kijun:
        tk_cross = "BEARISH"
    else:
        tk_cross = "NEUTRAL"
    
    # Kijun flat detection (price magnet)
    kijun_flat = False
    if len(df) >= kijun_period + 5:
        recent_kijun = []
        for i in range(5):
            idx = -(i + 1)
            k = (high.iloc[idx - kijun_period:idx].max() + low.iloc[idx - kijun_period:idx].min()) / 2
            recent_kijun.append(k)
        kijun_range = max(recent_kijun) - min(recent_kijun)
        kijun_flat = kijun_range < (kijun * 0.01)  # Less than 1% variation
    
    # Chikou Span signal (lagging)
    chikou_signal = "NEUTRAL"
    if len(df) >= 26:
        chikou_price = float(close.iloc[-1])
        price_26_ago = float(close.iloc[-26])
        if chikou_price > price_26_ago:
            chikou_signal = "BULLISH"
        elif chikou_price < price_26_ago:
            chikou_signal = "BEARISH"
    
    return {
        'tenkan': round(float(tenkan), 2),
        'kijun': round(float(kijun), 2),
        'senkou_a': round(float(senkou_a), 2),
        'senkou_b': round(float(senkou_b), 2),
        'cloud_top': round(float(cloud_top), 2),
        'cloud_bottom': round(float(cloud_bottom), 2),
        'cloud_signal': cloud_signal,
        'tk_cross': tk_cross,
        'price_vs_cloud': price_vs_cloud,
        'kijun_flat': kijun_flat,
        'chikou_signal': chikou_signal
    }


# ============================================================================
# GANN ANALYSIS FUNCTIONS
# ============================================================================

def calculate_gann_eighths(high_52, low_52):
    """
    Calculate Gann Rule of Eighths - key support/resistance levels.
    50% (4/8) is the most important - "center of gravity"
    """
    range_val = high_52 - low_52
    
    return {
        'major_low': round(low_52, 2),
        'major_high': round(high_52, 2),
        'range': round(range_val, 2),
        '1_8': round(low_52 + range_val * 0.125, 2),
        '2_8': round(low_52 + range_val * 0.250, 2),
        '3_8': round(low_52 + range_val * 0.375, 2),
        '4_8': round(low_52 + range_val * 0.500, 2),  # Most important
        '5_8': round(low_52 + range_val * 0.625, 2),
        '6_8': round(low_52 + range_val * 0.750, 2),
        '7_8': round(low_52 + range_val * 0.875, 2)
    }


def calculate_square_of_9(anchor_price, angles=[45, 90, 135, 180, 225, 270, 315, 360]):
    """
    Calculate Gann Square of 9 levels from anchor price.
    Returns both up and down projections for each angle.
    """
    if anchor_price <= 0:
        return []
    
    sqrt_price = math.sqrt(anchor_price)
    levels = []
    
    for angle in angles:
        # Gann increment: angle/180 gives the number of "steps"
        increment = angle / 180
        
        price_up = (sqrt_price + increment) ** 2
        price_down = (sqrt_price - increment) ** 2
        
        levels.append({
            'angle': angle,
            'price_up': round(price_up, 2),
            'price_down': round(max(0, price_down), 2),
            'distance_up_pct': round((price_up - anchor_price) / anchor_price * 100, 2),
            'distance_down_pct': round((price_down - anchor_price) / anchor_price * 100, 2) if price_down > 0 else -100
        })
    
    return levels


def calculate_multi_anchor_sq9(current_price, high_52, low_52):
    """
    Calculate Square of 9 from multiple anchor points as Gann recommended.
    Confluence of levels from different anchors = high probability zones.
    """
    midpoint = (high_52 + low_52) / 2
    
    return {
        'from_current': calculate_square_of_9(current_price),
        'from_high': calculate_square_of_9(high_52),
        'from_low': calculate_square_of_9(low_52),
        'from_midpoint': calculate_square_of_9(midpoint),
        'anchors': {
            'current': round(current_price, 2),
            'high_52': round(high_52, 2),
            'low_52': round(low_52, 2),
            'midpoint': round(midpoint, 2)
        }
    }


def calculate_gann_time_cycles(last_pivot_date, pivot_type, current_date=None):
    """
    Calculate Gann time cycles from a pivot point.
    Returns projected dates for each cycle.
    """
    if current_date is None:
        current_date = datetime.now()
    
    if isinstance(last_pivot_date, str):
        last_pivot_date = datetime.strptime(last_pivot_date, '%Y-%m-%d')
    
    days_since_pivot = (current_date - last_pivot_date).days
    
    cycles = []
    
    for cycle_days in GANN_DAY_CYCLES:
        target_date = last_pivot_date + timedelta(days=cycle_days)
        days_to_target = (target_date - current_date).days
        
        # Calculate cycle strength (strongest at exact cycle day)
        distance_from_cycle = abs(days_since_pivot - cycle_days)
        if distance_from_cycle == 0:
            strength = 100
        elif distance_from_cycle <= 2:
            strength = 85
        elif distance_from_cycle <= 5:
            strength = 60
        else:
            strength = 30
        
        # Only include future windows or very recent ones
        if days_to_target >= -3:
            cycles.append({
                'cycle_days': cycle_days,
                'target_date': target_date.strftime('%Y-%m-%d'),
                'target_display': target_date.strftime('%d/%m/%Y'),
                'days_to_target': days_to_target,
                'strength': strength,
                'is_future': days_to_target > 0,
                'from_pivot_type': pivot_type
            })
    
    return sorted(cycles, key=lambda x: abs(x['days_to_target']))


def find_anniversary_dates(pivot_date, pivot_type, current_date=None, lookback_years=3):
    """
    Find anniversary dates - Gann's principle of annual cycles.
    """
    if current_date is None:
        current_date = datetime.now()
    
    if isinstance(pivot_date, str):
        pivot_date = datetime.strptime(pivot_date, '%Y-%m-%d')
    
    anniversaries = []
    
    for years in range(1, lookback_years + 1):
        # Add years
        try:
            anniversary = pivot_date.replace(year=pivot_date.year + years)
        except ValueError:
            # Handle Feb 29
            anniversary = pivot_date.replace(year=pivot_date.year + years, day=28)
        
        days_away = (anniversary - current_date).days
        
        # Include if within 30 days (past or future)
        if -30 <= days_away <= 30:
            anniversaries.append({
                'date': anniversary.strftime('%Y-%m-%d'),
                'display': anniversary.strftime('%d/%m/%Y'),
                'type': f'{years}Y_ANNIVERSARY',
                'original_pivot_date': pivot_date.strftime('%Y-%m-%d'),
                'original_pivot_type': pivot_type,
                'days_away': days_away,
                'importance': 'HIGH' if years == 1 else 'MEDIUM'
            })
    
    return anniversaries


def calculate_cycle_strength(days_since_pivot):
    """
    Calculate how close we are to a Gann cycle day.
    Higher strength = more likely to see a turn.
    """
    closest_cycle = min(GANN_DAY_CYCLES, key=lambda x: abs(x - days_since_pivot))
    distance = abs(closest_cycle - days_since_pivot)
    
    if distance == 0:
        return {'strength': 100, 'nearest_cycle': closest_cycle, 'status': 'EXACT'}
    elif distance <= 2:
        return {'strength': 85, 'nearest_cycle': closest_cycle, 'status': 'VERY_CLOSE'}
    elif distance <= 5:
        return {'strength': 60, 'nearest_cycle': closest_cycle, 'status': 'APPROACHING'}
    elif distance <= 10:
        return {'strength': 40, 'nearest_cycle': closest_cycle, 'status': 'MODERATE'}
    else:
        return {'strength': 20, 'nearest_cycle': closest_cycle, 'status': 'DISTANT'}


# ============================================================================
# ENNEAGRAM STATE ANALYSIS
# ============================================================================

def identify_enneagram_state(df, timeframe='1D'):
    """
    Identify current Enneagram market state based on indicators.
    Returns state number (1-9) and confidence.
    """
    if len(df) < 50:
        return 9, 50, "Insufficient data"
    
    # Get indicators
    rsi = calculate_rsi(df)
    macd_data = calculate_macd(df)
    macd_hist = macd_data['histogram']
    adx_data = calculate_adx(df)
    adx = adx_data['adx']
    
    # Price position
    close = float(df['close'].iloc[-1])
    sma_50 = float(df['close'].iloc[-50:].mean())
    above_sma = close > sma_50
    
    # Volume trend (if available)
    volume_increasing = True
    if 'volume' in df.columns and len(df) >= 20:
        recent_vol = df['volume'].iloc[-10:].mean()
        prior_vol = df['volume'].iloc[-20:-10].mean()
        volume_increasing = recent_vol > prior_vol * 1.1
    
    # State identification logic
    if rsi < 30 and macd_hist < 0 and volume_increasing:
        state = 5  # Capitulation
        confidence = min(95, 70 + (30 - rsi))
        reason = "Extreme oversold with high volume - capitulation"
    elif rsi < 35 and macd_hist > 0 and not above_sma:
        state = 6  # Accumulation
        confidence = 75
        reason = "Oversold but momentum turning - accumulation"
    elif rsi > 70 and macd_hist > 0 and volume_increasing:
        state = 3  # Peak Formation
        confidence = min(90, 65 + (rsi - 70))
        reason = "Extreme overbought with climactic volume - peak forming"
    elif rsi > 65 and macd_hist < 0:
        state = 8  # Distribution
        confidence = 70
        reason = "Overbought with declining momentum - distribution"
    elif macd_hist > 0 and above_sma and rsi > 50 and rsi < 70:
        if adx > 30:
            state = 7  # Expansion
            confidence = 80
            reason = "Strong uptrend - expansion phase"
        else:
            state = 2  # Acceleration
            confidence = 70
            reason = "Building momentum - acceleration"
    elif macd_hist < 0 and rsi > 40 and rsi < 60:
        state = 4  # Retracement
        confidence = 65
        reason = "Pullback in neutral zone - retracement"
    elif rsi > 45 and rsi < 55 and abs(macd_hist) < 50:
        state = 9  # Equilibrium
        confidence = 60
        reason = "Neutral indicators - equilibrium"
    elif macd_hist > 0 and rsi > 40 and rsi < 55:
        state = 1  # Initiation
        confidence = 65
        reason = "Early momentum building - initiation"
    else:
        state = 9  # Default to Equilibrium
        confidence = 50
        reason = "Mixed signals - equilibrium"
    
    return state, confidence, reason


def determine_active_arrow(state, df):
    """
    Determine if stress or growth arrow is active based on momentum.
    """
    state_info = ENNEAGRAM_STATES[state]
    
    # Get momentum indicators
    macd_data = calculate_macd(df)
    macd_hist = macd_data['histogram']
    rsi = calculate_rsi(df)
    
    # Compare current vs recent momentum
    stress_score = 0
    growth_score = 0
    
    if macd_hist < 0:
        stress_score += 2
    else:
        growth_score += 2
    
    if rsi < 45:
        stress_score += 1
    elif rsi > 55:
        growth_score += 1
    
    # Check momentum direction
    if len(df) >= 5:
        recent_close = df['close'].iloc[-1]
        prior_close = df['close'].iloc[-5]
        if recent_close < prior_close:
            stress_score += 1
        else:
            growth_score += 1
    
    if stress_score > growth_score:
        arrow_type = "STRESS"
        target_state = state_info["stress"]
        confidence = min(85, 50 + stress_score * 10)
    else:
        arrow_type = "GROWTH"
        target_state = state_info["growth"]
        confidence = min(85, 50 + growth_score * 10)
    
    # Get meaning
    meaning_key = (state, arrow_type, target_state)
    meaning = ARROW_MEANINGS.get(meaning_key, "Market transitioning between phases")
    
    return {
        'arrow_type': arrow_type,
        'target_state': target_state,
        'target_state_name': ENNEAGRAM_STATES[target_state]['name'],
        'confidence': confidence,
        'meaning': meaning
    }


# ============================================================================
# MARKET REGIME DETECTION
# ============================================================================

def detect_market_regime(df):
    """
    Detect overall market regime - Gann's first step: identify the main trend.
    """
    if len(df) < 200:
        return {
            'regime': 'UNKNOWN',
            'strength': 'WEAK',
            'description': 'Insufficient data'
        }
    
    close = float(df['close'].iloc[-1])
    sma_200 = calculate_sma(df, 200)
    adx_data = calculate_adx(df)
    adx = adx_data['adx']
    
    # Determine regime
    if close > sma_200:
        if adx > 25:
            regime = "TRENDING_BULL"
            description = "Strong uptrend above 200 SMA"
        else:
            regime = "WEAK_BULL"
            description = "Above 200 SMA but weak trend"
    else:
        if adx > 25:
            regime = "TRENDING_BEAR"
            description = "Strong downtrend below 200 SMA"
        else:
            regime = "WEAK_BEAR"
            description = "Below 200 SMA but weak trend"
    
    # Check for ranging
    if adx < 20:
        regime = "RANGING"
        description = "Low ADX indicates ranging/consolidation"
    
    return {
        'regime': regime,
        'strength': adx_data['trend_strength'],
        'adx': adx,
        'price_vs_sma200': 'ABOVE' if close > sma_200 else 'BELOW',
        'sma_200': sma_200,
        'description': description
    }


# ============================================================================
# TIMEFRAME ANALYSIS
# ============================================================================

def analyze_single_timeframe(df, timeframe):
    """
    Complete analysis for a single timeframe.
    Returns all indicators and signals.
    """
    if df is None or len(df) < 30:
        return None
    
    current_price = float(df['close'].iloc[-1])
    
    # Get 52-period high/low for this timeframe
    lookback = min(52, len(df))
    high_52 = float(df['high'].iloc[-lookback:].max())
    low_52 = float(df['low'].iloc[-lookback:].min())
    
    # Calculate all indicators
    rsi = calculate_rsi(df)
    macd_data = calculate_macd(df)
    adx_data = calculate_adx(df)
    atr = calculate_atr(df)
    
    # Ichimoku with timeframe-specific parameters
    ichimoku_params = ICHIMOKU_PARAMS.get(timeframe, (9, 26, 52))
    ichimoku = calculate_ichimoku(df, *ichimoku_params)
    
    # Gann levels
    gann_eighths = calculate_gann_eighths(high_52, low_52)
    
    # Enneagram state
    state, state_confidence, state_reason = identify_enneagram_state(df, timeframe)
    state_info = ENNEAGRAM_STATES[state]
    arrow = determine_active_arrow(state, df)
    
    # Determine direction for this timeframe
    bullish_signals = 0
    bearish_signals = 0
    
    # RSI
    if rsi < 30:
        bullish_signals += 2  # Oversold = bullish reversal potential
    elif rsi > 70:
        bearish_signals += 2  # Overbought = bearish reversal potential
    elif rsi > 55:
        bullish_signals += 1
    elif rsi < 45:
        bearish_signals += 1
    
    # MACD
    if macd_data['histogram'] > 0:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    # Ichimoku
    if ichimoku['cloud_signal'] == 'BULLISH':
        bullish_signals += 2
    elif ichimoku['cloud_signal'] == 'BEARISH':
        bearish_signals += 2
    
    if ichimoku['tk_cross'] == 'BULLISH':
        bullish_signals += 1
    elif ichimoku['tk_cross'] == 'BEARISH':
        bearish_signals += 1
    
    # Enneagram bias
    if state_info['bias'] == 'BULLISH':
        bullish_signals += 1
    elif state_info['bias'] == 'BEARISH':
        bearish_signals += 1
    
    # Determine direction
    if bullish_signals > bearish_signals + 1:
        direction = "BULLISH"
        signal_type = "BUY"
    elif bearish_signals > bullish_signals + 1:
        direction = "BEARISH"
        signal_type = "SELL"
    else:
        direction = "NEUTRAL"
        signal_type = "WAIT"
    
    # Get nearest SQ9 level
    sq9_levels = calculate_square_of_9(current_price)
    nearest_sq9 = None
    min_distance = float('inf')
    for level in sq9_levels:
        for price_key in ['price_up', 'price_down']:
            dist = abs(level[price_key] - current_price)
            if dist < min_distance and level[price_key] > 0:
                min_distance = dist
                nearest_sq9 = {
                    'angle': level['angle'],
                    'price': level[price_key],
                    'type': 'resistance' if level[price_key] > current_price else 'support'
                }
    
    return {
        'timeframe': timeframe,
        'direction': direction,
        'signal_type': signal_type,
        'weight': TIMEFRAME_WEIGHTS[timeframe],
        'weighted_contribution': TIMEFRAME_WEIGHTS[timeframe] if direction != 'NEUTRAL' else 0,
        
        # Price data
        'current_price': current_price,
        'high_52': high_52,
        'low_52': low_52,
        
        # Enneagram
        'enneagram_state': state,
        'state_name': state_info['name'],
        'state_phase': state_info['phase'],
        'state_confidence': state_confidence,
        'state_reason': state_reason,
        'arrow_type': arrow['arrow_type'],
        'target_state': arrow['target_state'],
        'target_state_name': arrow['target_state_name'],
        'arrow_meaning': arrow['meaning'],
        
        # Momentum
        'rsi': rsi,
        'macd': macd_data['macd'],
        'macd_signal': macd_data['signal'],
        'macd_histogram': macd_data['histogram'],
        
        # Trend
        'adx': adx_data['adx'],
        'trend_strength': adx_data['trend_strength'],
        'plus_di': adx_data['plus_di'],
        'minus_di': adx_data['minus_di'],
        
        # Volatility
        'atr': atr,
        
        # Ichimoku
        'tenkan': ichimoku['tenkan'],
        'kijun': ichimoku['kijun'],
        'cloud_top': ichimoku['cloud_top'],
        'cloud_bottom': ichimoku['cloud_bottom'],
        'cloud_signal': ichimoku['cloud_signal'],
        'tk_cross': ichimoku['tk_cross'],
        'price_vs_cloud': ichimoku['price_vs_cloud'],
        'kijun_flat': ichimoku['kijun_flat'],
        'chikou_signal': ichimoku['chikou_signal'],
        
        # Gann
        'gann_50_pct': gann_eighths['4_8'],
        'gann_levels': gann_eighths,
        'sq9_nearest': nearest_sq9,
        
        # Signal counts
        'bullish_signals': bullish_signals,
        'bearish_signals': bearish_signals,
        'net_signals': bullish_signals - bearish_signals
    }


def analyze_all_timeframes(df_daily):
    """
    Analyze all timeframes and return comprehensive results.
    """
    # Prepare data for each timeframe
    tf_data = prepare_timeframe_data(df_daily)
    
    # Analyze each timeframe
    analyses = {}
    for tf in ['1D', '3D', '1W', '1M']:
        if tf_data[tf] is not None and len(tf_data[tf]) >= 30:
            analyses[tf] = analyze_single_timeframe(tf_data[tf], tf)
        else:
            analyses[tf] = None
    
    return analyses


# ============================================================================
# CONSENSUS CALCULATION
# ============================================================================

def calculate_consensus(timeframe_analyses):
    """
    Calculate weighted consensus across all timeframes.
    Implements Gann's principle: higher timeframes dominate.
    """
    weighted_bullish = 0
    weighted_bearish = 0
    aligned_count = 0
    aligned_direction = None
    details = []
    
    monthly_direction = None
    weekly_direction = None
    
    for tf in ['1M', '1W', '3D', '1D']:
        analysis = timeframe_analyses.get(tf)
        if analysis is None:
            continue
        
        weight = TIMEFRAME_WEIGHTS[tf]
        direction = analysis['direction']
        
        if direction == 'BULLISH':
            weighted_bullish += weight
        elif direction == 'BEARISH':
            weighted_bearish += weight
        
        # Track alignment
        if tf == '1M':
            monthly_direction = direction
            aligned_direction = direction
        elif tf == '1W':
            weekly_direction = direction
        
        if aligned_direction and direction == aligned_direction:
            aligned_count += 1
        
        details.append(f"{tf}:{direction}")
    
    # Calculate percentages
    total_weight = weighted_bullish + weighted_bearish
    if total_weight > 0:
        bullish_pct = int(weighted_bullish / total_weight * 100)
        bearish_pct = int(weighted_bearish / total_weight * 100)
    else:
        bullish_pct = 50
        bearish_pct = 50
    
    # Determine consensus direction
    if bullish_pct >= 60:
        consensus_direction = "BULLISH"
        confidence_level = "HIGH" if bullish_pct >= 75 else "MEDIUM"
    elif bearish_pct >= 60:
        consensus_direction = "BEARISH"
        confidence_level = "HIGH" if bearish_pct >= 75 else "MEDIUM"
    else:
        consensus_direction = "NEUTRAL"
        confidence_level = "LOW"
    
    # Determine signal type
    if confidence_level in ["HIGH", "MEDIUM"]:
        if consensus_direction == "BULLISH":
            signal_type = "BUY"
        elif consensus_direction == "BEARISH":
            signal_type = "SELL"
        else:
            signal_type = "WAIT"
    else:
        signal_type = "WAIT"
    
    # Check for divergences
    daily_analysis = timeframe_analyses.get('1D')
    daily_divergent = False
    if daily_analysis and monthly_direction:
        daily_divergent = daily_analysis['direction'] != monthly_direction and daily_analysis['direction'] != 'NEUTRAL'
    
    # Generate Gann interpretation
    interpretation = generate_gann_interpretation(
        consensus_direction,
        monthly_direction,
        weekly_direction,
        daily_divergent,
        timeframe_analyses
    )
    
    return {
        'direction': consensus_direction,
        'signal_type': signal_type,
        'weighted_bullish_pct': bullish_pct,
        'weighted_bearish_pct': bearish_pct,
        'weighted_score': max(bullish_pct, bearish_pct),
        'confidence_level': confidence_level,
        'alignment_count': f"{aligned_count}/4",
        'alignment_detail': ', '.join(details),
        'monthly_direction': monthly_direction,
        'weekly_direction': weekly_direction,
        'monthly_dominant': True,
        'weekly_confirms': weekly_direction == monthly_direction if monthly_direction else False,
        'daily_divergent': daily_divergent,
        'interpretation': interpretation
    }


def generate_gann_interpretation(direction, monthly, weekly, daily_divergent, analyses):
    """
    Generate Gann-style interpretation of the multi-timeframe analysis.
    """
    interpretations = []
    
    if direction == "BEARISH":
        if monthly == "BEARISH" and weekly == "BEARISH":
            interpretations.append("Strong downtrend confirmed on Monthly and Weekly.")
            if daily_divergent:
                interpretations.append("Daily showing oversold bounce - this is a SELLING OPPORTUNITY, not a reversal.")
                interpretations.append("Gann Rule: Never trade against the main trend.")
            else:
                interpretations.append("All timeframes aligned bearish - high probability sell setup.")
        elif monthly == "BEARISH":
            interpretations.append("Monthly trend is down. Weekly showing potential bounce.")
            interpretations.append("Wait for weekly to realign bearish before adding shorts.")
    
    elif direction == "BULLISH":
        if monthly == "BULLISH" and weekly == "BULLISH":
            interpretations.append("Strong uptrend confirmed on Monthly and Weekly.")
            if daily_divergent:
                interpretations.append("Daily showing pullback - this is a BUYING OPPORTUNITY.")
                interpretations.append("Gann: Buy the dips in an uptrend.")
            else:
                interpretations.append("All timeframes aligned bullish - high probability buy setup.")
        elif monthly == "BULLISH":
            interpretations.append("Monthly trend is up. Weekly showing correction.")
            interpretations.append("Wait for weekly to realign bullish before buying.")
    
    else:  # NEUTRAL
        interpretations.append("Timeframes conflicting - no clear trend.")
        interpretations.append("Gann Rule #6: When in doubt, stay out.")
        interpretations.append("Wait for alignment before committing capital.")
    
    return " ".join(interpretations)


# ============================================================================
# PRICE TARGET CALCULATION
# ============================================================================

def calculate_price_targets(timeframe_analyses, consensus, current_price):
    """
    Calculate price targets based on Gann confluence zones.
    Uses weekly timeframe as primary source (per Gann).
    """
    weekly = timeframe_analyses.get('1W')
    monthly = timeframe_analyses.get('1M')
    daily = timeframe_analyses.get('1D')
    
    direction = consensus['direction']
    targets = []
    stop_sources = []
    
    if weekly is None:
        # Fallback to daily
        weekly = daily
    
    if weekly is None:
        return {
            'source_timeframe': 'N/A',
            'tp1': current_price,
            'tp2': current_price,
            'tp3': current_price,
            'stop_loss': current_price,
            'calculation_method': 'No data available'
        }
    
    gann_levels = weekly['gann_levels']
    atr = weekly['atr']
    
    if direction == "BEARISH":
        # Targets below current price
        potential_targets = [
            {'price': gann_levels['4_8'], 'source': 'GANN_50%_1W', 'priority': 1},
            {'price': gann_levels['3_8'], 'source': 'GANN_3/8_1W', 'priority': 2},
            {'price': weekly['cloud_bottom'], 'source': 'CLOUD_BOTTOM_1W', 'priority': 1},
            {'price': weekly['kijun'], 'source': 'KIJUN_1W', 'priority': 2 if weekly['kijun_flat'] else 3},
        ]
        
        # Add monthly levels for TP3
        if monthly:
            potential_targets.append({'price': monthly['gann_levels']['4_8'], 'source': 'GANN_50%_1M', 'priority': 3})
        
        # Filter targets below current price
        valid_targets = [t for t in potential_targets if t['price'] < current_price * 0.995]
        valid_targets.sort(key=lambda x: (x['priority'], -x['price']))  # Closest first within priority
        
        # Stop loss above current price
        stop_price = max(
            gann_levels['5_8'],
            weekly['cloud_top'],
            weekly['kijun'] if weekly['kijun'] > current_price else current_price + atr * 1.5
        )
        stop_sources = ['GANN_5/8_1W' if stop_price == gann_levels['5_8'] else 'CLOUD_TOP_1W']
        
    else:  # BULLISH or NEUTRAL
        # Targets above current price
        potential_targets = [
            {'price': gann_levels['5_8'], 'source': 'GANN_5/8_1W', 'priority': 1},
            {'price': gann_levels['6_8'], 'source': 'GANN_6/8_1W', 'priority': 2},
            {'price': weekly['cloud_top'], 'source': 'CLOUD_TOP_1W', 'priority': 1},
            {'price': weekly['kijun'], 'source': 'KIJUN_1W', 'priority': 2 if weekly['kijun_flat'] else 3},
        ]
        
        if monthly:
            potential_targets.append({'price': monthly['gann_levels']['5_8'], 'source': 'GANN_5/8_1M', 'priority': 3})
        
        # Filter targets above current price
        valid_targets = [t for t in potential_targets if t['price'] > current_price * 1.005]
        valid_targets.sort(key=lambda x: (x['priority'], x['price']))  # Closest first within priority
        
        # Stop loss below current price
        stop_price = min(
            gann_levels['3_8'],
            weekly['cloud_bottom'],
            weekly['kijun'] if weekly['kijun'] < current_price else current_price - atr * 1.5
        )
        stop_sources = ['GANN_3/8_1W' if stop_price == gann_levels['3_8'] else 'CLOUD_BOTTOM_1W']
    
    # Assign targets
    tp1 = valid_targets[0]['price'] if len(valid_targets) > 0 else current_price + (atr * 2 if direction == "BULLISH" else -atr * 2)
    tp2 = valid_targets[1]['price'] if len(valid_targets) > 1 else tp1 + (atr * 2 if direction == "BULLISH" else -atr * 2)
    tp3 = valid_targets[2]['price'] if len(valid_targets) > 2 else tp2 + (atr * 2 if direction == "BULLISH" else -atr * 2)
    
    tp1_sources = [valid_targets[0]['source']] if len(valid_targets) > 0 else ['ATR_SYNTHETIC']
    tp2_sources = [valid_targets[1]['source']] if len(valid_targets) > 1 else ['ATR_SYNTHETIC']
    tp3_sources = [valid_targets[2]['source']] if len(valid_targets) > 2 else ['ATR_SYNTHETIC']
    
    return {
        'source_timeframe': '1W',
        'calculation_method': 'Gann Eighths + Ichimoku Confluence',
        'tp1': round(tp1, 2),
        'tp1_sources': tp1_sources,
        'tp1_confluence_strength': len(tp1_sources),
        'tp2': round(tp2, 2),
        'tp2_sources': tp2_sources,
        'tp2_confluence_strength': len(tp2_sources),
        'tp3': round(tp3, 2),
        'tp3_sources': tp3_sources,
        'tp3_confluence_strength': len(tp3_sources),
        'stop_loss': round(stop_price, 2),
        'stop_sources': stop_sources,
        'stop_reason': 'Weekly Kijun/Cloud resistance' if direction == "BEARISH" else 'Weekly Kijun/Cloud support'
    }


# ============================================================================
# TIME FORECAST
# ============================================================================

def calculate_time_forecast(timeframe_analyses, consensus, current_price):
    """
    Calculate time-based forecasts using Gann cycles.
    """
    today = datetime.now()
    
    # Get daily ATR for price projection
    daily = timeframe_analyses.get('1D')
    atr_daily = daily['atr'] if daily else current_price * 0.025
    
    # Find last significant pivot (simplified - use recent swing)
    # In production, this would come from actual pivot detection
    last_high_date = today - timedelta(days=30)  # Placeholder
    last_low_date = today - timedelta(days=45)   # Placeholder
    
    # Calculate cycles from both pivots
    cycles_from_high = calculate_gann_time_cycles(last_high_date, 'HIGH', today)
    cycles_from_low = calculate_gann_time_cycles(last_low_date, 'LOW', today)
    
    # Find the nearest future cycle
    all_future_cycles = [c for c in cycles_from_high + cycles_from_low if c['is_future']]
    all_future_cycles.sort(key=lambda x: x['days_to_target'])
    
    if all_future_cycles:
        nearest = all_future_cycles[0]
        days_to_pivot = nearest['days_to_target']
        pivot_date = nearest['target_date']
        pivot_display = nearest['target_display']
        pivot_confidence = nearest['strength']
        cycle_sources = [f"{nearest['cycle_days']}D_CYCLE from {nearest['from_pivot_type']}"]
    else:
        days_to_pivot = 7
        pivot_date = (today + timedelta(days=7)).strftime('%Y-%m-%d')
        pivot_display = (today + timedelta(days=7)).strftime('%d/%m/%Y')
        pivot_confidence = 50
        cycle_sources = ['DEFAULT_7D_PROJECTION']
    
    # Determine expected pivot type based on consensus
    if consensus['direction'] == 'BEARISH':
        expected_pivot = 'LOW'
    elif consensus['direction'] == 'BULLISH':
        expected_pivot = 'HIGH'
    else:
        expected_pivot = 'UNKNOWN'
    
    # Calculate probable price range at pivot
    efficiency_factor = 0.7
    max_expected_move = atr_daily * days_to_pivot * efficiency_factor
    
    if consensus['direction'] == 'BEARISH':
        probable_low = current_price - max_expected_move
        probable_high = current_price + (atr_daily * days_to_pivot * 0.3)
    else:
        probable_high = current_price + max_expected_move
        probable_low = current_price - (atr_daily * days_to_pivot * 0.3)
    
    # Check anniversary dates
    anniversaries = find_anniversary_dates(last_high_date, 'HIGH', today)
    anniversaries.extend(find_anniversary_dates(last_low_date, 'LOW', today))
    
    return {
        'next_pivot_date': pivot_date,
        'next_pivot_display': pivot_display,
        'days_to_pivot': days_to_pivot,
        'pivot_type': expected_pivot,
        'pivot_confidence': pivot_confidence,
        'cycle_sources': cycle_sources,
        'probable_price_low': round(probable_low, 2),
        'probable_price_high': round(probable_high, 2),
        'probable_range_text': f"${round(probable_low):,} - ${round(probable_high):,}",
        'atr_daily': round(atr_daily, 2),
        'max_expected_move': round(max_expected_move, 2),
        'efficiency_factor': efficiency_factor,
        'anniversary_dates': anniversaries[:3]  # Top 3 nearest
    }


# ============================================================================
# INVALIDATION RULES
# ============================================================================

def calculate_invalidation_rules(timeframe_analyses, consensus, current_price):
    """
    Define what would invalidate the current analysis.
    """
    weekly = timeframe_analyses.get('1W')
    
    if weekly is None:
        return {
            'invalidation_price': current_price,
            'rules': []
        }
    
    rules = []
    invalidation_price = current_price
    
    if consensus['direction'] == 'BEARISH':
        # What would invalidate bearish case
        invalidation_price = weekly['kijun'] if weekly['kijun'] > current_price else weekly['cloud_top']
        
        rules = [
            {'condition': 'Daily close above Weekly Kijun', 'price': round(weekly['kijun'], 2)},
            {'condition': 'Weekly RSI breaks above 65', 'current': weekly['rsi']},
            {'condition': 'Weekly close above cloud top', 'price': round(weekly['cloud_top'], 2)}
        ]
    
    elif consensus['direction'] == 'BULLISH':
        invalidation_price = weekly['kijun'] if weekly['kijun'] < current_price else weekly['cloud_bottom']
        
        rules = [
            {'condition': 'Daily close below Weekly Kijun', 'price': round(weekly['kijun'], 2)},
            {'condition': 'Weekly RSI breaks below 35', 'current': weekly['rsi']},
            {'condition': 'Weekly close below cloud bottom', 'price': round(weekly['cloud_bottom'], 2)}
        ]
    
    return {
        'invalidation_price': round(invalidation_price, 2),
        'invalidation_reason': 'Weekly Kijun breach' if weekly['kijun_flat'] else 'Cloud breach',
        'rules': rules
    }


# ============================================================================
# STRATEGY GENERATION
# ============================================================================

def generate_strategy(consensus, timeframe_analyses, targets, time_forecast, invalidation):
    """
    Generate complete trading strategy based on analysis.
    """
    direction = consensus['direction']
    confidence = consensus['confidence_level']
    daily_divergent = consensus['daily_divergent']
    
    # Primary bias
    primary_bias = direction if direction != 'NEUTRAL' else 'NO_TRADE'
    
    # Action
    if direction == 'BEARISH':
        if daily_divergent:
            action = "Sell rallies toward Weekly Kijun"
            entry_method = "Wait for daily RSI > 50 then bearish TK cross, or test of weekly resistance"
        else:
            action = "Sell at market or on minor bounces"
            entry_method = "Enter short on any rally with stop above weekly Kijun"
    elif direction == 'BULLISH':
        if daily_divergent:
            action = "Buy dips toward Weekly Kijun"
            entry_method = "Wait for daily RSI < 50 then bullish TK cross, or test of weekly support"
        else:
            action = "Buy at market or on minor dips"
            entry_method = "Enter long on any pullback with stop below weekly Kijun"
    else:
        action = "No trade - wait for clarity"
        entry_method = "Monitor for timeframe alignment"
    
    # Position size recommendation
    if confidence == 'HIGH':
        position_size = "Full size (high confidence)"
    elif confidence == 'MEDIUM':
        position_size = "Half size (medium confidence)"
    else:
        position_size = "No position (low confidence)"
    
    # Time in trade
    time_in_trade = f"Hold until TP1 or pivot date ({time_forecast['next_pivot_display']})"
    
    # Invalidation action
    invalidation_action = f"Close position if price closes {'above' if direction == 'BEARISH' else 'below'} ${invalidation['invalidation_price']:,.0f}"
    
    return {
        'primary_bias': primary_bias,
        'action': action,
        'entry_method': entry_method,
        'position_size_recommendation': position_size,
        'time_in_trade': time_in_trade,
        'interpretation': consensus['interpretation'],
        'invalidation_action': invalidation_action
    }


# ============================================================================
# MAIN ENDPOINT
# ============================================================================

@app.get("/signal/daily")
async def get_daily_signal():
    """Generate comprehensive multi-timeframe daily trading signal"""
    try:
        print("\n" + "="*70)
        print("  LUXOR V7 PRANA - MULTI-TIMEFRAME EDITION v5.0.0")
        print("="*70)
        
        # [1/10] Fetch data
        print("\n[1/10] Fetching BTCUSDT daily data...")
        df_daily = luxor.fetch_real_binance_data(use_cache=True)
        
        if df_daily is None or len(df_daily) < 100:
            raise HTTPException(status_code=500, detail="Insufficient data")
        
        print(f"       Loaded {len(df_daily)} daily candles")
        
        # Current price
        current_price = float(df_daily['close'].iloc[-1])
        signal_date = df_daily.iloc[-1]['date']
        if hasattr(signal_date, 'strftime'):
            signal_date_str = signal_date.strftime('%Y-%m-%d %H:%M:%S')
        else:
            signal_date_str = str(signal_date)
        
        # [2/10] Analyze all timeframes
        print("[2/10] Analyzing all timeframes...")
        tf_analyses = analyze_all_timeframes(df_daily)
        
        for tf in ['1D', '3D', '1W', '1M']:
            if tf_analyses[tf]:
                print(f"       {tf}: {tf_analyses[tf]['direction']} (State {tf_analyses[tf]['enneagram_state']})")
        
        # [3/10] Calculate consensus
        print("[3/10] Calculating weighted consensus...")
        consensus = calculate_consensus(tf_analyses)
        print(f"       Direction: {consensus['direction']} ({consensus['weighted_score']}% confidence)")
        
        # [4/10] Market regime
        print("[4/10] Detecting market regime...")
        regime = detect_market_regime(df_daily)
        print(f"       Regime: {regime['regime']}")
        
        # [5/10] Price targets
        print("[5/10] Calculating price targets...")
        targets = calculate_price_targets(tf_analyses, consensus, current_price)
        print(f"       TP1: ${targets['tp1']:,.0f} | TP2: ${targets['tp2']:,.0f} | SL: ${targets['stop_loss']:,.0f}")
        
        # [6/10] Time forecast
        print("[6/10] Calculating time forecast...")
        time_forecast = calculate_time_forecast(tf_analyses, consensus, current_price)
        print(f"       Next pivot: {time_forecast['next_pivot_display']} ({time_forecast['pivot_type']})")
        
        # [7/10] Invalidation rules
        print("[7/10] Setting invalidation rules...")
        invalidation = calculate_invalidation_rules(tf_analyses, consensus, current_price)
        
        # [8/10] Strategy generation
        print("[8/10] Generating trading strategy...")
        strategy = generate_strategy(consensus, tf_analyses, targets, time_forecast, invalidation)
        
        # [9/10] Square of 9 analysis
        print("[9/10] Calculating Square of 9 levels...")
        daily = tf_analyses.get('1D')
        high_52 = daily['high_52'] if daily else current_price * 1.5
        low_52 = daily['low_52'] if daily else current_price * 0.5
        sq9_analysis = calculate_multi_anchor_sq9(current_price, high_52, low_52)
        
        # [10/10] Build response
        print("[10/10] Building response...")
        
        # Get primary timeframe data for legacy fields
        weekly = tf_analyses.get('1W') or tf_analyses.get('1D')
        monthly = tf_analyses.get('1M') or weekly
        daily_tf = tf_analyses.get('1D')
        
        # Legacy Gann levels
        gann_levels = weekly['gann_levels'] if weekly else calculate_gann_eighths(high_52, low_52)
        
        # Prepare timeframes for JSON (ensure serializable)
        timeframes_clean = {}
        for tf, analysis in tf_analyses.items():
            if analysis:
                timeframes_clean[tf] = {
                    'direction': analysis['direction'],
                    'signal_type': analysis['signal_type'],
                    'weight': analysis['weight'],
                    'weighted_contribution': analysis['weighted_contribution'],
                    'enneagram_state': analysis['enneagram_state'],
                    'state_name': analysis['state_name'],
                    'rsi': analysis['rsi'],
                    'macd_histogram': analysis['macd_histogram'],
                    'adx': analysis['adx'],
                    'trend_strength': analysis['trend_strength'],
                    'cloud_signal': analysis['cloud_signal'],
                    'tk_cross': analysis['tk_cross'],
                    'kijun_flat': analysis['kijun_flat'],
                    'gann_50_pct': analysis['gann_50_pct'],
                    'bullish_signals': analysis['bullish_signals'],
                    'bearish_signals': analysis['bearish_signals']
                }
        
        # Build complete response
        response_data = {
            # System info
            "status": "success",
            "version": "5.0.0",
            "timestamp": datetime.now().isoformat(),
            
            # ================================================================
            # LEGACY FIELDS (DB COMPATIBLE)
            # ================================================================
            "symbol": "BTCUSDT",
            "signal_type": consensus['signal_type'],
            "signal_date": signal_date_str,
            "entry_price": round(current_price, 2),
            "take_profit": targets['tp1'],
            "stop_loss": targets['stop_loss'],
            "confidence": consensus['weighted_score'],
            "confluence_score": consensus['weighted_score'],
            
            # Integer fields
            "active_pivot_id": len(df_daily) - 1,
            "enneagram_state": weekly['enneagram_state'] if weekly else 9,
            "price_confluences": targets['tp1_confluence_strength'],
            "time_confluences": len(time_forecast['cycle_sources']),
            "gann_cycle_target": time_forecast['days_to_pivot'],
            
            # String fields
            "enneagram_arrow": weekly['arrow_type'] if weekly else "NEUTRAL",
            "macd_signal": str(round(weekly['macd'] if weekly else 0, 4)),
            "ichimoku_signal": weekly['cloud_signal'] if weekly else "NEUTRAL",
            
            # Numeric fields
            "rsi_value": round(weekly['rsi'] if weekly else 50, 2),
            
            # JSON/Text fields
            "gann_sq9_levels": json.dumps(sq9_analysis['from_current'][:4]),
            "gann_angles_active": json.dumps([l['angle'] for l in sq9_analysis['from_current'] if abs(l['distance_up_pct']) < 5 or abs(l['distance_down_pct']) < 5]),
            "confluence_details": json.dumps({
                'score': consensus['weighted_score'],
                'alignment': consensus['alignment_count'],
                'sources': targets['tp1_sources']
            }),
            
            # ================================================================
            # NEW MTF FIELDS
            # ================================================================
            "primary_direction": consensus['direction'],
            "weighted_score": consensus['weighted_score'],
            "mtf_alignment": consensus['alignment_count'],
            "market_regime": regime['regime'],
            
            # Full timeframe analysis
            "timeframes": timeframes_clean,
            
            # Consensus details
            "consensus": consensus,
            
            # Price targets
            "price_targets": targets,
            "target_1": targets['tp1'],
            "target_2": targets['tp2'],
            "target_3": targets['tp3'],
            
            # Time forecast
            "time_forecast": time_forecast,
            "pivot_forecast_primary": {
                "date": time_forecast['next_pivot_date'],
                "date_display": time_forecast['next_pivot_display'],
                "days_from_now": time_forecast['days_to_pivot'],
                "expected_pivot": time_forecast['pivot_type'],
                "confidence": time_forecast['pivot_confidence'],
                "cycle_type": time_forecast['cycle_sources'][0] if time_forecast['cycle_sources'] else "projection"
            },
            
            # Price-time forecast (legacy compatible)
            "price_time_forecast": {
                "days_to_pivot": time_forecast['days_to_pivot'],
                "atr_daily": time_forecast['atr_daily'],
                "max_expected_move": time_forecast['max_expected_move'],
                "probable_price_low": time_forecast['probable_price_low'],
                "probable_price_high": time_forecast['probable_price_high'],
                "probable_range_text": time_forecast['probable_range_text'],
                "target_1_reachable": abs(targets['tp1'] - current_price) < time_forecast['max_expected_move'],
                "target_2_reachable": abs(targets['tp2'] - current_price) < time_forecast['max_expected_move'],
                "target_3_reachable": abs(targets['tp3'] - current_price) < time_forecast['max_expected_move'],
                "direction": consensus['direction']
            },
            
            # Square of 9
            "sq9_analysis": sq9_analysis,
            
            # Invalidation
            "invalidation": invalidation,
            
            # Strategy
            "strategy": strategy,
            
            # ================================================================
            # LEGACY EXTENDED FIELDS
            # ================================================================
            "state": weekly['enneagram_state'] if weekly else 9,
            "state_name": weekly['state_name'] if weekly else "Equilibrium",
            "phase": weekly['state_phase'] if weekly else "Unknown",
            "target_state": weekly['target_state'] if weekly else 9,
            "target_state_name": weekly['target_state_name'] if weekly else "Equilibrium",
            "arrow": weekly['arrow_type'] if weekly else "NEUTRAL",
            "arrow_meaning": weekly['arrow_meaning'] if weekly else "",
            
            "price_direction": consensus['direction'],
            "direction_emoji": "⬇️" if consensus['direction'] == "BEARISH" else ("⬆️" if consensus['direction'] == "BULLISH" else "➡️"),
            "direction_probability": consensus['weighted_score'],
            "direction_reasoning": f"{consensus['alignment_count']} timeframes aligned",
            
            "bullish_signals": sum(tf['bullish_signals'] for tf in tf_analyses.values() if tf),
            "bearish_signals": sum(tf['bearish_signals'] for tf in tf_analyses.values() if tf),
            
            # Gann levels
            "major_low": gann_levels['major_low'],
            "major_high": gann_levels['major_high'],
            "gann_range": gann_levels['range'],
            "gann_3_8": gann_levels['3_8'],
            "gann_4_8": gann_levels['4_8'],
            "gann_5_8": gann_levels['5_8'],
            
            # Ichimoku
            "tenkan": weekly['tenkan'] if weekly else current_price,
            "kijun": weekly['kijun'] if weekly else current_price,
            "cloud_top": weekly['cloud_top'] if weekly else current_price,
            "cloud_bottom": weekly['cloud_bottom'] if weekly else current_price,
            "cloud_signal": weekly['cloud_signal'] if weekly else "NEUTRAL",
            "tk_cross": weekly['tk_cross'] if weekly else "NEUTRAL",
            "kijun_flat": weekly['kijun_flat'] if weekly else False,
            
            # Trend
            "adx": weekly['adx'] if weekly else 25,
            "trend_strength": weekly['trend_strength'] if weekly else "MODERATE",
            
            # Regime
            "regime": regime,
            
            # Additional indicators
            "rsi": round(daily_tf['rsi'] if daily_tf else 50, 2),
            "macd": round(weekly['macd'] if weekly else 0, 4),
            "macd_hist": round(weekly['macd_histogram'] if weekly else 0, 4),
            "atr": round(daily_tf['atr'] if daily_tf else current_price * 0.025, 2),
            
            # Metadata
            "candles_analyzed": len(df_daily),
            "last_candle_date": signal_date_str,
            
            # Legacy fields for Telegram
            "signal_strength": consensus['confidence_level'],
            "confirmation_score": consensus['weighted_score'],
            "confirmation_score_display": f"{consensus['weighted_score']}%",
            
            # Active pivot
            "active_pivot": {
                "id": len(df_daily) - 1,
                "date": signal_date_str,
                "price": current_price,
                "type": time_forecast['pivot_type']
            },
            
            # Strong confluence zones for Telegram
            "strong_confluence_zones": [
                {"price": targets['tp1'], "type": "TARGET", "sources": targets['tp1_sources']},
                {"price": targets['stop_loss'], "type": "STOP", "sources": targets['stop_sources']}
            ],
            
            # Gann time windows for legacy
            "gann_time_windows": [
                {
                    "cycle_length": time_forecast['days_to_pivot'],
                    "target_date": time_forecast['next_pivot_date'],
                    "date_display": time_forecast['next_pivot_display'],
                    "days_from_now": time_forecast['days_to_pivot']
                }
            ]
        }
        
        # Print summary
        print("\n" + "="*70)
        print("  SIGNAL SUMMARY")
        print("="*70)
        print(f"  Symbol: BTCUSDT")
        print(f"  Price:  ${current_price:,.2f}")
        print(f"  Signal: {consensus['signal_type']}")
        print(f"  Direction: {consensus['direction']} ({consensus['weighted_score']}%)")
        print(f"  Alignment: {consensus['alignment_count']}")
        print(f"  Regime: {regime['regime']}")
        print(f"")
        print(f"  Timeframe Breakdown:")
        print(f"    1M: {tf_analyses['1M']['direction'] if tf_analyses.get('1M') else 'N/A'} (40%)")
        print(f"    1W: {tf_analyses['1W']['direction'] if tf_analyses.get('1W') else 'N/A'} (35%)")
        print(f"    3D: {tf_analyses['3D']['direction'] if tf_analyses.get('3D') else 'N/A'} (15%)")
        print(f"    1D: {tf_analyses['1D']['direction'] if tf_analyses.get('1D') else 'N/A'} (10%)")
        print(f"")
        print(f"  Targets:")
        print(f"    TP1: ${targets['tp1']:,.2f}")
        print(f"    TP2: ${targets['tp2']:,.2f}")
        print(f"    TP3: ${targets['tp3']:,.2f}")
        print(f"    SL:  ${targets['stop_loss']:,.2f}")
        print(f"")
        print(f"  Time Forecast:")
        print(f"    Next Pivot: {time_forecast['next_pivot_display']} ({time_forecast['pivot_type']})")
        print(f"    Probable Range: {time_forecast['probable_range_text']}")
        print(f"")
        print(f"  Strategy: {strategy['action']}")
        print("="*70 + "\n")
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Signal generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "5.0.0",
        "timestamp": datetime.now().isoformat(),
        "system": "LUXOR V7 PRANA - MULTI-TIMEFRAME EDITION"
    }


@app.on_event("startup")
async def startup():
    print("\n" + "="*70)
    print("  LUXOR V7 PRANA RUNTIME - MULTI-TIMEFRAME EDITION")
    print("  Version 5.0.0")
    print("  ")
    print("  Timeframe Weights:")
    print(f"    Monthly (1M): {TIMEFRAME_WEIGHTS['1M']}%")
    print(f"    Weekly (1W):  {TIMEFRAME_WEIGHTS['1W']}%")
    print(f"    3-Day (3D):   {TIMEFRAME_WEIGHTS['3D']}%")
    print(f"    Daily (1D):   {TIMEFRAME_WEIGHTS['1D']}%")
    print("  ")
    print("  Gann Principle: Higher timeframes dominate")
    print("  Starting up...")
    print("="*70 + "\n")


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
