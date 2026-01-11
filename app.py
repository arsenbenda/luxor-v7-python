"""
LUXOR V7 PRANA RUNTIME - MULTI-TIMEFRAME EDITION
Version: 5.0.1
Complete Gann-aligned multi-timeframe analysis system
Fixed: numpy type serialization for JSON compatibility
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

app = FastAPI(title="LUXOR V7 PRANA Runtime", version="5.0.1")

# Initialize system
luxor = LuxorV7PranaSystem(initial_capital=INITIAL_CAPITAL)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def to_native(obj):
    """
    Recursively convert numpy/pandas types to native Python types for JSON serialization.
    This prevents FastAPI serialization errors with numpy.bool_, numpy.int64, etc.
    """
    if obj is None:
        return None
    elif isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(to_native(i) for i in obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [to_native(i) for i in obj.tolist()]
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
    elif isinstance(obj, pd.Series):
        return to_native(obj.tolist())
    elif hasattr(obj, 'item'):  # numpy scalar
        return to_native(obj.item())
    else:
        return obj


def safe_float(value, default=0.0):
    """Safely convert to float, handling NaN/Inf"""
    try:
        result = float(value)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    """Safely convert to int"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_round(value, decimals=2, default=0.0):
    """Safely round a value"""
    try:
        result = float(value)
        if np.isnan(result) or np.isinf(result):
            return default
        return round(result, decimals)
    except (TypeError, ValueError):
        return default


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
    """
    if df is None or len(df) == 0:
        return None
    
    try:
        df_copy = df.copy()
        
        # Ensure we have a datetime index
        if 'date' in df_copy.columns:
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            df_copy = df_copy.set_index('date')
        elif not isinstance(df_copy.index, pd.DatetimeIndex):
            return None
        
        # Define resampling rules (pandas 1.x compatible)
        resample_map = {
            '1D': '1D',
            '3D': '3D',
            '1W': 'W',
            '1M': 'M'
        }
        
        if timeframe not in resample_map:
            return df_copy.reset_index()
        
        rule = resample_map[timeframe]
        
        # Check required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df_copy.columns:
                print(f"[WARNING] Missing column: {col}")
                return None
        
        # Build aggregation dict based on available columns
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }
        
        if 'volume' in df_copy.columns:
            agg_dict['volume'] = 'sum'
        
        # OHLC resampling
        resampled = df_copy.resample(rule).agg(agg_dict).dropna()
        
        if len(resampled) == 0:
            return None
        
        # Reset index
        resampled = resampled.reset_index()
        if 'index' in resampled.columns:
            resampled = resampled.rename(columns={'index': 'date'})
        
        return resampled
        
    except Exception as e:
        print(f"[ERROR] Resample failed for {timeframe}: {e}")
        return None


def prepare_timeframe_data(df_daily):
    """
    Prepare data for all timeframes from daily data.
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
    if df is None or len(df) < period + 1:
        return 50.0
    
    try:
        close = df['close'].values
        deltas = np.diff(close)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return safe_round(rsi, 2, 50.0)
    except Exception as e:
        print(f"[ERROR] RSI calculation failed: {e}")
        return 50.0


def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD and histogram"""
    if df is None or len(df) < slow + signal:
        return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
    
    try:
        close = df['close']
        
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': safe_round(macd_line.iloc[-1], 4, 0.0),
            'signal': safe_round(signal_line.iloc[-1], 4, 0.0),
            'histogram': safe_round(histogram.iloc[-1], 4, 0.0)
        }
    except Exception as e:
        print(f"[ERROR] MACD calculation failed: {e}")
        return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}


def calculate_adx(df, period=14):
    """Calculate ADX for trend strength"""
    default_result = {'adx': 25.0, 'trend_strength': 'MODERATE', 'plus_di': 0.0, 'minus_di': 0.0}
    
    if df is None or len(df) < period * 2:
        return default_result
    
    try:
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # True Range
        tr = np.maximum(high[1:] - low[1:], 
                        np.maximum(np.abs(high[1:] - close[:-1]), 
                                  np.abs(low[1:] - close[:-1])))
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed averages
        atr = np.mean(tr[-period:])
        
        if atr == 0:
            return default_result
        
        plus_di = 100 * np.mean(plus_dm[-period:]) / atr
        minus_di = 100 * np.mean(minus_dm[-period:]) / atr
        
        # ADX
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx = 0
        else:
            dx = 100 * abs(plus_di - minus_di) / di_sum
        
        adx = dx
        
        # Trend strength classification
        if adx > 50:
            trend_strength = "STRONG"
        elif adx > 25:
            trend_strength = "MODERATE"
        else:
            trend_strength = "WEAK"
        
        return {
            'adx': safe_round(adx, 2, 25.0),
            'trend_strength': trend_strength,
            'plus_di': safe_round(plus_di, 2, 0.0),
            'minus_di': safe_round(minus_di, 2, 0.0)
        }
    except Exception as e:
        print(f"[ERROR] ADX calculation failed: {e}")
        return default_result


def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    if df is None or len(df) < period + 1:
        try:
            return safe_float(df['close'].iloc[-1] * 0.025) if df is not None else 1000.0
        except:
            return 1000.0
    
    try:
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                  np.abs(low[1:] - close[:-1])))
        
        atr = np.mean(tr[-period:])
        return safe_round(atr, 2, 1000.0)
    except Exception as e:
        print(f"[ERROR] ATR calculation failed: {e}")
        return 1000.0


def calculate_sma(df, period=200):
    """Calculate Simple Moving Average"""
    if df is None or len(df) < period:
        try:
            return safe_float(df['close'].mean()) if df is not None else 0.0
        except:
            return 0.0
    
    try:
        return safe_round(df['close'].iloc[-period:].mean(), 2, 0.0)
    except Exception as e:
        print(f"[ERROR] SMA calculation failed: {e}")
        return 0.0


def calculate_ichimoku(df, tenkan_period=9, kijun_period=26, senkou_period=52):
    """Calculate Ichimoku Cloud components"""
    if df is None or len(df) < max(tenkan_period, kijun_period, senkou_period):
        try:
            close = safe_float(df['close'].iloc[-1]) if df is not None else 0.0
        except:
            close = 0.0
        
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
    
    try:
        high = df['high']
        low = df['low']
        close = df['close']
        current_price = safe_float(close.iloc[-1])
        
        # Tenkan-sen (Conversion Line)
        tenkan = (high.iloc[-tenkan_period:].max() + low.iloc[-tenkan_period:].min()) / 2
        
        # Kijun-sen (Base Line)
        kijun = (high.iloc[-kijun_period:].max() + low.iloc[-kijun_period:].min()) / 2
        
        # Senkou Span A
        senkou_a = (tenkan + kijun) / 2
        
        # Senkou Span B
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
        
        # Kijun flat detection
        kijun_flat = False
        if len(df) >= kijun_period + 5:
            try:
                recent_kijun = []
                for i in range(5):
                    idx = -(i + 1)
                    start_idx = max(0, len(df) + idx - kijun_period)
                    end_idx = len(df) + idx
                    if start_idx < end_idx:
                        k = (high.iloc[start_idx:end_idx].max() + low.iloc[start_idx:end_idx].min()) / 2
                        recent_kijun.append(float(k))
                
                if recent_kijun:
                    kijun_range = max(recent_kijun) - min(recent_kijun)
                    kijun_flat = bool(kijun_range < (float(kijun) * 0.01))
            except:
                kijun_flat = False
        
        # Chikou signal
        chikou_signal = "NEUTRAL"
        if len(df) >= 26:
            try:
                chikou_price = safe_float(close.iloc[-1])
                price_26_ago = safe_float(close.iloc[-26])
                if chikou_price > price_26_ago:
                    chikou_signal = "BULLISH"
                elif chikou_price < price_26_ago:
                    chikou_signal = "BEARISH"
            except:
                pass
        
        return {
            'tenkan': safe_round(tenkan, 2),
            'kijun': safe_round(kijun, 2),
            'senkou_a': safe_round(senkou_a, 2),
            'senkou_b': safe_round(senkou_b, 2),
            'cloud_top': safe_round(cloud_top, 2),
            'cloud_bottom': safe_round(cloud_bottom, 2),
            'cloud_signal': cloud_signal,
            'tk_cross': tk_cross,
            'price_vs_cloud': price_vs_cloud,
            'kijun_flat': bool(kijun_flat),
            'chikou_signal': chikou_signal
        }
    except Exception as e:
        print(f"[ERROR] Ichimoku calculation failed: {e}")
        return {
            'tenkan': 0.0, 'kijun': 0.0, 'senkou_a': 0.0, 'senkou_b': 0.0,
            'cloud_top': 0.0, 'cloud_bottom': 0.0, 'cloud_signal': 'NEUTRAL',
            'tk_cross': 'NEUTRAL', 'price_vs_cloud': 'INSIDE',
            'kijun_flat': False, 'chikou_signal': 'NEUTRAL'
        }


# ============================================================================
# GANN ANALYSIS FUNCTIONS
# ============================================================================

def calculate_gann_eighths(high_52, low_52):
    """Calculate Gann Rule of Eighths"""
    high_52 = safe_float(high_52)
    low_52 = safe_float(low_52)
    range_val = high_52 - low_52
    
    if range_val <= 0:
        range_val = high_52 * 0.5
    
    return {
        'major_low': safe_round(low_52, 2),
        'major_high': safe_round(high_52, 2),
        'range': safe_round(range_val, 2),
        '1_8': safe_round(low_52 + range_val * 0.125, 2),
        '2_8': safe_round(low_52 + range_val * 0.250, 2),
        '3_8': safe_round(low_52 + range_val * 0.375, 2),
        '4_8': safe_round(low_52 + range_val * 0.500, 2),
        '5_8': safe_round(low_52 + range_val * 0.625, 2),
        '6_8': safe_round(low_52 + range_val * 0.750, 2),
        '7_8': safe_round(low_52 + range_val * 0.875, 2)
    }


def calculate_square_of_9(anchor_price, angles=[45, 90, 135, 180, 225, 270, 315, 360]):
    """Calculate Gann Square of 9 levels"""
    anchor_price = safe_float(anchor_price)
    
    if anchor_price <= 0:
        return []
    
    sqrt_price = math.sqrt(anchor_price)
    levels = []
    
    for angle in angles:
        increment = angle / 180
        
        price_up = (sqrt_price + increment) ** 2
        price_down = max(0, (sqrt_price - increment) ** 2)
        
        levels.append({
            'angle': int(angle),
            'price_up': safe_round(price_up, 2),
            'price_down': safe_round(price_down, 2),
            'distance_up_pct': safe_round((price_up - anchor_price) / anchor_price * 100, 2) if anchor_price > 0 else 0.0,
            'distance_down_pct': safe_round((price_down - anchor_price) / anchor_price * 100, 2) if anchor_price > 0 else 0.0
        })
    
    return levels


def calculate_multi_anchor_sq9(current_price, high_52, low_52):
    """Calculate Square of 9 from multiple anchors"""
    current_price = safe_float(current_price)
    high_52 = safe_float(high_52)
    low_52 = safe_float(low_52)
    midpoint = (high_52 + low_52) / 2
    
    return {
        'from_current': calculate_square_of_9(current_price),
        'from_high': calculate_square_of_9(high_52),
        'from_low': calculate_square_of_9(low_52),
        'from_midpoint': calculate_square_of_9(midpoint),
        'anchors': {
            'current': safe_round(current_price, 2),
            'high_52': safe_round(high_52, 2),
            'low_52': safe_round(low_52, 2),
            'midpoint': safe_round(midpoint, 2)
        }
    }


def calculate_gann_time_cycles(last_pivot_date, pivot_type, current_date=None):
    """Calculate Gann time cycles from a pivot point"""
    if current_date is None:
        current_date = datetime.now()
    
    if isinstance(last_pivot_date, str):
        try:
            last_pivot_date = datetime.strptime(last_pivot_date, '%Y-%m-%d')
        except:
            last_pivot_date = current_date - timedelta(days=30)
    
    days_since_pivot = (current_date - last_pivot_date).days
    cycles = []
    
    for cycle_days in GANN_DAY_CYCLES:
        target_date = last_pivot_date + timedelta(days=cycle_days)
        days_to_target = (target_date - current_date).days
        
        distance_from_cycle = abs(days_since_pivot - cycle_days)
        if distance_from_cycle == 0:
            strength = 100
        elif distance_from_cycle <= 2:
            strength = 85
        elif distance_from_cycle <= 5:
            strength = 60
        else:
            strength = 30
        
        if days_to_target >= -3:
            cycles.append({
                'cycle_days': int(cycle_days),
                'target_date': target_date.strftime('%Y-%m-%d'),
                'target_display': target_date.strftime('%d/%m/%Y'),
                'days_to_target': int(days_to_target),
                'strength': int(strength),
                'is_future': bool(days_to_target > 0),
                'from_pivot_type': str(pivot_type)
            })
    
    return sorted(cycles, key=lambda x: abs(x['days_to_target']))


def find_anniversary_dates(pivot_date, pivot_type, current_date=None, lookback_years=3):
    """Find anniversary dates"""
    if current_date is None:
        current_date = datetime.now()
    
    if isinstance(pivot_date, str):
        try:
            pivot_date = datetime.strptime(pivot_date, '%Y-%m-%d')
        except:
            return []
    
    anniversaries = []
    
    for years in range(1, lookback_years + 1):
        try:
            anniversary = pivot_date.replace(year=pivot_date.year + years)
        except ValueError:
            anniversary = pivot_date.replace(year=pivot_date.year + years, day=28)
        
        days_away = (anniversary - current_date).days
        
        if -30 <= days_away <= 30:
            anniversaries.append({
                'date': anniversary.strftime('%Y-%m-%d'),
                'display': anniversary.strftime('%d/%m/%Y'),
                'type': f'{years}Y_ANNIVERSARY',
                'original_pivot_date': pivot_date.strftime('%Y-%m-%d'),
                'original_pivot_type': str(pivot_type),
                'days_away': int(days_away),
                'importance': 'HIGH' if years == 1 else 'MEDIUM'
            })
    
    return anniversaries


# ============================================================================
# ENNEAGRAM STATE ANALYSIS
# ============================================================================

def identify_enneagram_state(df, timeframe='1D'):
    """Identify current Enneagram market state"""
    if df is None or len(df) < 50:
        return 9, 50, "Insufficient data"
    
    try:
        rsi = calculate_rsi(df)
        macd_data = calculate_macd(df)
        macd_hist = macd_data['histogram']
        adx_data = calculate_adx(df)
        adx = adx_data['adx']
        
        close = safe_float(df['close'].iloc[-1])
        sma_50 = safe_float(df['close'].iloc[-50:].mean())
        above_sma = close > sma_50
        
        volume_increasing = True
        if 'volume' in df.columns and len(df) >= 20:
            recent_vol = safe_float(df['volume'].iloc[-10:].mean())
            prior_vol = safe_float(df['volume'].iloc[-20:-10].mean())
            volume_increasing = recent_vol > prior_vol * 1.1
        
        # State identification
        if rsi < 30 and macd_hist < 0 and volume_increasing:
            state, confidence = 5, min(95, 70 + int(30 - rsi))
            reason = "Extreme oversold with high volume - capitulation"
        elif rsi < 35 and macd_hist > 0 and not above_sma:
            state, confidence = 6, 75
            reason = "Oversold but momentum turning - accumulation"
        elif rsi > 70 and macd_hist > 0 and volume_increasing:
            state, confidence = 3, min(90, 65 + int(rsi - 70))
            reason = "Extreme overbought with climactic volume - peak forming"
        elif rsi > 65 and macd_hist < 0:
            state, confidence = 8, 70
            reason = "Overbought with declining momentum - distribution"
        elif macd_hist > 0 and above_sma and 50 < rsi < 70:
            if adx > 30:
                state, confidence = 7, 80
                reason = "Strong uptrend - expansion phase"
            else:
                state, confidence = 2, 70
                reason = "Building momentum - acceleration"
        elif macd_hist < 0 and 40 < rsi < 60:
            state, confidence = 4, 65
            reason = "Pullback in neutral zone - retracement"
        elif 45 < rsi < 55 and abs(macd_hist) < 50:
            state, confidence = 9, 60
            reason = "Neutral indicators - equilibrium"
        elif macd_hist > 0 and 40 < rsi < 55:
            state, confidence = 1, 65
            reason = "Early momentum building - initiation"
        else:
            state, confidence = 9, 50
            reason = "Mixed signals - equilibrium"
        
        return int(state), int(confidence), str(reason)
    
    except Exception as e:
        print(f"[ERROR] Enneagram state identification failed: {e}")
        return 9, 50, "Error in calculation"


def determine_active_arrow(state, df):
    """Determine if stress or growth arrow is active"""
    state_info = ENNEAGRAM_STATES.get(state, ENNEAGRAM_STATES[9])
    
    try:
        macd_data = calculate_macd(df)
        macd_hist = macd_data['histogram']
        rsi = calculate_rsi(df)
        
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
        
        if df is not None and len(df) >= 5:
            recent_close = safe_float(df['close'].iloc[-1])
            prior_close = safe_float(df['close'].iloc[-5])
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
        
        meaning_key = (state, arrow_type, target_state)
        meaning = ARROW_MEANINGS.get(meaning_key, "Market transitioning between phases")
        
        return {
            'arrow_type': str(arrow_type),
            'target_state': int(target_state),
            'target_state_name': str(ENNEAGRAM_STATES[target_state]['name']),
            'confidence': int(confidence),
            'meaning': str(meaning)
        }
    
    except Exception as e:
        print(f"[ERROR] Arrow determination failed: {e}")
        return {
            'arrow_type': 'NEUTRAL',
            'target_state': 9,
            'target_state_name': 'Equilibrium',
            'confidence': 50,
            'meaning': 'Unable to determine'
        }


# ============================================================================
# MARKET REGIME DETECTION
# ============================================================================

def detect_market_regime(df):
    """Detect overall market regime"""
    if df is None or len(df) < 200:
        return {
            'regime': 'UNKNOWN',
            'strength': 'WEAK',
            'adx': 25.0,
            'price_vs_sma200': 'UNKNOWN',
            'sma_200': 0.0,
            'description': 'Insufficient data'
        }
    
    try:
        close = safe_float(df['close'].iloc[-1])
        sma_200 = calculate_sma(df, 200)
        adx_data = calculate_adx(df)
        adx = adx_data['adx']
        
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
        
        if adx < 20:
            regime = "RANGING"
            description = "Low ADX indicates ranging/consolidation"
        
        return {
            'regime': str(regime),
            'strength': str(adx_data['trend_strength']),
            'adx': safe_round(adx, 2),
            'price_vs_sma200': 'ABOVE' if close > sma_200 else 'BELOW',
            'sma_200': safe_round(sma_200, 2),
            'description': str(description)
        }
    
    except Exception as e:
        print(f"[ERROR] Regime detection failed: {e}")
        return {
            'regime': 'UNKNOWN',
            'strength': 'WEAK',
            'adx': 25.0,
            'price_vs_sma200': 'UNKNOWN',
            'sma_200': 0.0,
            'description': 'Error in calculation'
        }


# ============================================================================
# TIMEFRAME ANALYSIS
# ============================================================================

def analyze_single_timeframe(df, timeframe):
    """Complete analysis for a single timeframe"""
    if df is None or len(df) < 30:
        return None
    
    try:
        current_price = safe_float(df['close'].iloc[-1])
        
        lookback = min(52, len(df))
        high_52 = safe_float(df['high'].iloc[-lookback:].max())
        low_52 = safe_float(df['low'].iloc[-lookback:].min())
        
        # Calculate indicators
        rsi = calculate_rsi(df)
        macd_data = calculate_macd(df)
        adx_data = calculate_adx(df)
        atr = calculate_atr(df)
        
        ichimoku_params = ICHIMOKU_PARAMS.get(timeframe, (9, 26, 52))
        ichimoku = calculate_ichimoku(df, *ichimoku_params)
        
        gann_eighths = calculate_gann_eighths(high_52, low_52)
        
        state, state_confidence, state_reason = identify_enneagram_state(df, timeframe)
        state_info = ENNEAGRAM_STATES[state]
        arrow = determine_active_arrow(state, df)
        
        # Determine direction
        bullish_signals = 0
        bearish_signals = 0
        
        if rsi < 30:
            bullish_signals += 2
        elif rsi > 70:
            bearish_signals += 2
        elif rsi > 55:
            bullish_signals += 1
        elif rsi < 45:
            bearish_signals += 1
        
        if macd_data['histogram'] > 0:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        if ichimoku['cloud_signal'] == 'BULLISH':
            bullish_signals += 2
        elif ichimoku['cloud_signal'] == 'BEARISH':
            bearish_signals += 2
        
        if ichimoku['tk_cross'] == 'BULLISH':
            bullish_signals += 1
        elif ichimoku['tk_cross'] == 'BEARISH':
            bearish_signals += 1
        
        if state_info['bias'] == 'BULLISH':
            bullish_signals += 1
        elif state_info['bias'] == 'BEARISH':
            bearish_signals += 1
        
        if bullish_signals > bearish_signals + 1:
            direction = "BULLISH"
            signal_type = "BUY"
        elif bearish_signals > bullish_signals + 1:
            direction = "BEARISH"
            signal_type = "SELL"
        else:
            direction = "NEUTRAL"
            signal_type = "WAIT"
        
        # Nearest SQ9 level
        sq9_levels = calculate_square_of_9(current_price)
        nearest_sq9 = None
        min_distance = float('inf')
        for level in sq9_levels:
            for price_key in ['price_up', 'price_down']:
                dist = abs(level[price_key] - current_price)
                if dist < min_distance and level[price_key] > 0:
                    min_distance = dist
                    nearest_sq9 = {
                        'angle': int(level['angle']),
                        'price': safe_round(level[price_key], 2),
                        'type': 'resistance' if level[price_key] > current_price else 'support'
                    }
        
        return {
            'timeframe': str(timeframe),
            'direction': str(direction),
            'signal_type': str(signal_type),
            'weight': int(TIMEFRAME_WEIGHTS[timeframe]),
            'weighted_contribution': int(TIMEFRAME_WEIGHTS[timeframe]) if direction != 'NEUTRAL' else 0,
            'current_price': safe_round(current_price, 2),
            'high_52': safe_round(high_52, 2),
            'low_52': safe_round(low_52, 2),
            'enneagram_state': int(state),
            'state_name': str(state_info['name']),
            'state_phase': str(state_info['phase']),
            'state_confidence': int(state_confidence),
            'state_reason': str(state_reason),
            'arrow_type': str(arrow['arrow_type']),
            'target_state': int(arrow['target_state']),
            'target_state_name': str(arrow['target_state_name']),
            'arrow_meaning': str(arrow['meaning']),
            'rsi': safe_round(rsi, 2),
            'macd': safe_round(macd_data['macd'], 4),
            'macd_signal': safe_round(macd_data['signal'], 4),
            'macd_histogram': safe_round(macd_data['histogram'], 4),
            'adx': safe_round(adx_data['adx'], 2),
            'trend_strength': str(adx_data['trend_strength']),
            'plus_di': safe_round(adx_data['plus_di'], 2),
            'minus_di': safe_round(adx_data['minus_di'], 2),
            'atr': safe_round(atr, 2),
            'tenkan': safe_round(ichimoku['tenkan'], 2),
            'kijun': safe_round(ichimoku['kijun'], 2),
            'cloud_top': safe_round(ichimoku['cloud_top'], 2),
            'cloud_bottom': safe_round(ichimoku['cloud_bottom'], 2),
            'cloud_signal': str(ichimoku['cloud_signal']),
            'tk_cross': str(ichimoku['tk_cross']),
            'price_vs_cloud': str(ichimoku['price_vs_cloud']),
            'kijun_flat': bool(ichimoku['kijun_flat']),
            'chikou_signal': str(ichimoku['chikou_signal']),
            'gann_50_pct': safe_round(gann_eighths['4_8'], 2),
            'gann_levels': gann_eighths,
            'sq9_nearest': nearest_sq9,
            'bullish_signals': int(bullish_signals),
            'bearish_signals': int(bearish_signals),
            'net_signals': int(bullish_signals - bearish_signals)
        }
    
    except Exception as e:
        print(f"[ERROR] Single timeframe analysis failed for {timeframe}: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_all_timeframes(df_daily):
    """Analyze all timeframes"""
    tf_data = prepare_timeframe_data(df_daily)
    
    analyses = {}
    for tf in ['1D', '3D', '1W', '1M']:
        if tf_data.get(tf) is not None and len(tf_data[tf]) >= 30:
            analyses[tf] = analyze_single_timeframe(tf_data[tf], tf)
        else:
            analyses[tf] = None
    
    return analyses


# ============================================================================
# CONSENSUS CALCULATION
# ============================================================================

def calculate_consensus(timeframe_analyses):
    """Calculate weighted consensus across all timeframes"""
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
        
        if tf == '1M':
            monthly_direction = direction
            aligned_direction = direction
        elif tf == '1W':
            weekly_direction = direction
        
        if aligned_direction and direction == aligned_direction:
            aligned_count += 1
        
        details.append(f"{tf}:{direction}")
    
    total_weight = weighted_bullish + weighted_bearish
    if total_weight > 0:
        bullish_pct = int(weighted_bullish / total_weight * 100)
        bearish_pct = int(weighted_bearish / total_weight * 100)
    else:
        bullish_pct = 50
        bearish_pct = 50
    
    if bullish_pct >= 60:
        consensus_direction = "BULLISH"
        confidence_level = "HIGH" if bullish_pct >= 75 else "MEDIUM"
    elif bearish_pct >= 60:
        consensus_direction = "BEARISH"
        confidence_level = "HIGH" if bearish_pct >= 75 else "MEDIUM"
    else:
        consensus_direction = "NEUTRAL"
        confidence_level = "LOW"
    
    if confidence_level in ["HIGH", "MEDIUM"]:
        if consensus_direction == "BULLISH":
            signal_type = "BUY"
        elif consensus_direction == "BEARISH":
            signal_type = "SELL"
        else:
            signal_type = "WAIT"
    else:
        signal_type = "WAIT"
    
    daily_analysis = timeframe_analyses.get('1D')
    daily_divergent = False
    if daily_analysis and monthly_direction:
        daily_divergent = daily_analysis['direction'] != monthly_direction and daily_analysis['direction'] != 'NEUTRAL'
    
    interpretation = generate_gann_interpretation(
        consensus_direction, monthly_direction, weekly_direction,
        daily_divergent, timeframe_analyses
    )
    
    return {
        'direction': str(consensus_direction),
        'signal_type': str(signal_type),
        'weighted_bullish_pct': int(bullish_pct),
        'weighted_bearish_pct': int(bearish_pct),
        'weighted_score': int(max(bullish_pct, bearish_pct)),
        'confidence_level': str(confidence_level),
        'alignment_count': f"{aligned_count}/4",
        'alignment_detail': ', '.join(details),
        'monthly_direction': str(monthly_direction) if monthly_direction else 'N/A',
        'weekly_direction': str(weekly_direction) if weekly_direction else 'N/A',
        'monthly_dominant': True,
        'weekly_confirms': bool(weekly_direction == monthly_direction) if monthly_direction else False,
        'daily_divergent': bool(daily_divergent),
        'interpretation': str(interpretation)
    }


def generate_gann_interpretation(direction, monthly, weekly, daily_divergent, analyses):
    """Generate Gann-style interpretation"""
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
        else:
            interpretations.append("Bearish bias from lower timeframes. Monitor monthly for confirmation.")
    
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
        else:
            interpretations.append("Bullish bias from lower timeframes. Monitor monthly for confirmation.")
    
    else:
        interpretations.append("Timeframes conflicting - no clear trend.")
        interpretations.append("Gann Rule #6: When in doubt, stay out.")
        interpretations.append("Wait for alignment before committing capital.")
    
    return " ".join(interpretations)


# ============================================================================
# PRICE TARGET CALCULATION
# ============================================================================

def calculate_price_targets(timeframe_analyses, consensus, current_price):
    """Calculate price targets based on Gann confluence zones"""
    current_price = safe_float(current_price)
    weekly = timeframe_analyses.get('1W')
    monthly = timeframe_analyses.get('1M')
    daily = timeframe_analyses.get('1D')
    
    direction = consensus['direction']
    
    # Use weekly or fallback to daily
    source_tf = weekly if weekly else daily
    
    if source_tf is None:
        atr = current_price * 0.025
        return {
            'source_timeframe': 'N/A',
            'calculation_method': 'ATR fallback',
            'tp1': safe_round(current_price + atr * 2, 2),
            'tp1_sources': ['ATR_SYNTHETIC'],
            'tp1_confluence_strength': 1,
            'tp2': safe_round(current_price + atr * 4, 2),
            'tp2_sources': ['ATR_SYNTHETIC'],
            'tp2_confluence_strength': 1,
            'tp3': safe_round(current_price + atr * 6, 2),
            'tp3_sources': ['ATR_SYNTHETIC'],
            'tp3_confluence_strength': 1,
            'stop_loss': safe_round(current_price - atr * 1.5, 2),
            'stop_sources': ['ATR_SYNTHETIC'],
            'stop_reason': 'ATR-based stop'
        }
    
    gann_levels = source_tf['gann_levels']
    atr = source_tf['atr']
    
    if direction == "BEARISH":
        potential_targets = [
            {'price': gann_levels['4_8'], 'source': 'GANN_50%_1W', 'priority': 1},
            {'price': gann_levels['3_8'], 'source': 'GANN_3/8_1W', 'priority': 2},
            {'price': source_tf['cloud_bottom'], 'source': 'CLOUD_BOTTOM_1W', 'priority': 1},
            {'price': source_tf['kijun'], 'source': 'KIJUN_1W', 'priority': 2 if source_tf['kijun_flat'] else 3},
        ]
        
        if monthly:
            potential_targets.append({'price': monthly['gann_levels']['4_8'], 'source': 'GANN_50%_1M', 'priority': 3})
        
        valid_targets = [t for t in potential_targets if t['price'] < current_price * 0.995]
        valid_targets.sort(key=lambda x: (x['priority'], -x['price']))
        
        stop_candidates = [gann_levels['5_8'], source_tf['cloud_top']]
        if source_tf['kijun'] > current_price:
            stop_candidates.append(source_tf['kijun'])
        stop_candidates.append(current_price + atr * 1.5)
        stop_price = max(stop_candidates)
        stop_sources = ['GANN_5/8_1W']
        
    else:
        potential_targets = [
            {'price': gann_levels['5_8'], 'source': 'GANN_5/8_1W', 'priority': 1},
            {'price': gann_levels['6_8'], 'source': 'GANN_6/8_1W', 'priority': 2},
            {'price': source_tf['cloud_top'], 'source': 'CLOUD_TOP_1W', 'priority': 1},
            {'price': source_tf['kijun'], 'source': 'KIJUN_1W', 'priority': 2 if source_tf['kijun_flat'] else 3},
        ]
        
        if monthly:
            potential_targets.append({'price': monthly['gann_levels']['5_8'], 'source': 'GANN_5/8_1M', 'priority': 3})
        
        valid_targets = [t for t in potential_targets if t['price'] > current_price * 1.005]
        valid_targets.sort(key=lambda x: (x['priority'], x['price']))
        
        stop_candidates = [gann_levels['3_8'], source_tf['cloud_bottom']]
        if source_tf['kijun'] < current_price:
            stop_candidates.append(source_tf['kijun'])
        stop_candidates.append(current_price - atr * 1.5)
        stop_price = min(stop_candidates)
        stop_sources = ['GANN_3/8_1W']
    
    # Assign targets
    tp1 = valid_targets[0]['price'] if len(valid_targets) > 0 else current_price + (atr * 2 if direction != "BEARISH" else -atr * 2)
    tp2 = valid_targets[1]['price'] if len(valid_targets) > 1 else tp1 + (atr * 2 if direction != "BEARISH" else -atr * 2)
    tp3 = valid_targets[2]['price'] if len(valid_targets) > 2 else tp2 + (atr * 2 if direction != "BEARISH" else -atr * 2)
    
    tp1_sources = [valid_targets[0]['source']] if len(valid_targets) > 0 else ['ATR_SYNTHETIC']
    tp2_sources = [valid_targets[1]['source']] if len(valid_targets) > 1 else ['ATR_SYNTHETIC']
    tp3_sources = [valid_targets[2]['source']] if len(valid_targets) > 2 else ['ATR_SYNTHETIC']
    
    return {
        'source_timeframe': '1W',
        'calculation_method': 'Gann Eighths + Ichimoku Confluence',
        'tp1': safe_round(tp1, 2),
        'tp1_sources': tp1_sources,
        'tp1_confluence_strength': len(tp1_sources),
        'tp2': safe_round(tp2, 2),
        'tp2_sources': tp2_sources,
        'tp2_confluence_strength': len(tp2_sources),
        'tp3': safe_round(tp3, 2),
        'tp3_sources': tp3_sources,
        'tp3_confluence_strength': len(tp3_sources),
        'stop_loss': safe_round(stop_price, 2),
        'stop_sources': stop_sources,
        'stop_reason': 'Weekly Kijun/Cloud resistance' if direction == "BEARISH" else 'Weekly Kijun/Cloud support'
    }


# ============================================================================
# TIME FORECAST
# ============================================================================

def calculate_time_forecast(timeframe_analyses, consensus, current_price):
    """Calculate time-based forecasts"""
    today = datetime.now()
    current_price = safe_float(current_price)
    
    daily = timeframe_analyses.get('1D')
    atr_daily = daily['atr'] if daily else current_price * 0.025
    
    last_high_date = today - timedelta(days=30)
    last_low_date = today - timedelta(days=45)
    
    cycles_from_high = calculate_gann_time_cycles(last_high_date, 'HIGH', today)
    cycles_from_low = calculate_gann_time_cycles(last_low_date, 'LOW', today)
    
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
        future_date = today + timedelta(days=7)
        pivot_date = future_date.strftime('%Y-%m-%d')
        pivot_display = future_date.strftime('%d/%m/%Y')
        pivot_confidence = 50
        cycle_sources = ['DEFAULT_7D_PROJECTION']
    
    if consensus['direction'] == 'BEARISH':
        expected_pivot = 'LOW'
    elif consensus['direction'] == 'BULLISH':
        expected_pivot = 'HIGH'
    else:
        expected_pivot = 'UNKNOWN'
    
    efficiency_factor = 0.7
    max_expected_move = atr_daily * days_to_pivot * efficiency_factor
    
    if consensus['direction'] == 'BEARISH':
        probable_low = current_price - max_expected_move
        probable_high = current_price + (atr_daily * days_to_pivot * 0.3)
    else:
        probable_high = current_price + max_expected_move
        probable_low = current_price - (atr_daily * days_to_pivot * 0.3)
    
    anniversaries = find_anniversary_dates(last_high_date, 'HIGH', today)
    anniversaries.extend(find_anniversary_dates(last_low_date, 'LOW', today))
    
    return {
        'next_pivot_date': str(pivot_date),
        'next_pivot_display': str(pivot_display),
        'days_to_pivot': int(days_to_pivot),
        'pivot_type': str(expected_pivot),
        'pivot_confidence': int(pivot_confidence),
        'cycle_sources': cycle_sources,
        'probable_price_low': safe_round(probable_low, 2),
        'probable_price_high': safe_round(probable_high, 2),
        'probable_range_text': f"${int(probable_low):,} - ${int(probable_high):,}",
        'atr_daily': safe_round(atr_daily, 2),
        'max_expected_move': safe_round(max_expected_move, 2),
        'efficiency_factor': float(efficiency_factor),
        'anniversary_dates': anniversaries[:3]
    }


# ============================================================================
# INVALIDATION RULES
# ============================================================================

def calculate_invalidation_rules(timeframe_analyses, consensus, current_price):
    """Define invalidation rules"""
    current_price = safe_float(current_price)
    weekly = timeframe_analyses.get('1W')
    
    if weekly is None:
        return {
            'invalidation_price': safe_round(current_price, 2),
            'invalidation_reason': 'No weekly data',
            'rules': []
        }
    
    rules = []
    
    if consensus['direction'] == 'BEARISH':
        invalidation_price = weekly['kijun'] if weekly['kijun'] > current_price else weekly['cloud_top']
        rules = [
            {'condition': 'Daily close above Weekly Kijun', 'price': safe_round(weekly['kijun'], 2)},
            {'condition': 'Weekly RSI breaks above 65', 'current': safe_round(weekly['rsi'], 2)},
            {'condition': 'Weekly close above cloud top', 'price': safe_round(weekly['cloud_top'], 2)}
        ]
    elif consensus['direction'] == 'BULLISH':
        invalidation_price = weekly['kijun'] if weekly['kijun'] < current_price else weekly['cloud_bottom']
        rules = [
            {'condition': 'Daily close below Weekly Kijun', 'price': safe_round(weekly['kijun'], 2)},
            {'condition': 'Weekly RSI breaks below 35', 'current': safe_round(weekly['rsi'], 2)},
            {'condition': 'Weekly close below cloud bottom', 'price': safe_round(weekly['cloud_bottom'], 2)}
        ]
    else:
        invalidation_price = current_price
    
    return {
        'invalidation_price': safe_round(invalidation_price, 2),
        'invalidation_reason': 'Weekly Kijun breach' if weekly['kijun_flat'] else 'Cloud breach',
        'rules': rules
    }


# ============================================================================
# STRATEGY GENERATION
# ============================================================================

def generate_strategy(consensus, timeframe_analyses, targets, time_forecast, invalidation):
    """Generate trading strategy"""
    direction = consensus['direction']
    confidence = consensus['confidence_level']
    daily_divergent = consensus['daily_divergent']
    
    primary_bias = direction if direction != 'NEUTRAL' else 'NO_TRADE'
    
    if direction == 'BEARISH':
        if daily_divergent:
            action = "Sell rallies toward Weekly Kijun"
            entry_method = "Wait for daily RSI > 50 then bearish TK cross"
        else:
            action = "Sell at market or on minor bounces"
            entry_method = "Enter short on any rally with stop above weekly Kijun"
    elif direction == 'BULLISH':
        if daily_divergent:
            action = "Buy dips toward Weekly Kijun"
            entry_method = "Wait for daily RSI < 50 then bullish TK cross"
        else:
            action = "Buy at market or on minor dips"
            entry_method = "Enter long on any pullback with stop below weekly Kijun"
    else:
        action = "No trade - wait for clarity"
        entry_method = "Monitor for timeframe alignment"
    
    if confidence == 'HIGH':
        position_size = "Full size (high confidence)"
    elif confidence == 'MEDIUM':
        position_size = "Half size (medium confidence)"
    else:
        position_size = "No position (low confidence)"
    
    time_in_trade = f"Hold until TP1 or pivot date ({time_forecast['next_pivot_display']})"
    
    invalidation_action = f"Close position if price closes {'above' if direction == 'BEARISH' else 'below'} ${invalidation['invalidation_price']:,.0f}"
    
    return {
        'primary_bias': str(primary_bias),
        'action': str(action),
        'entry_method': str(entry_method),
        'position_size_recommendation': str(position_size),
        'time_in_trade': str(time_in_trade),
        'interpretation': str(consensus['interpretation']),
        'invalidation_action': str(invalidation_action)
    }


# ============================================================================
# MAIN ENDPOINT
# ============================================================================

@app.get("/signal/daily")
async def get_daily_signal():
    """Generate comprehensive multi-timeframe daily trading signal"""
    try:
        print("\n" + "="*70)
        print("  LUXOR V7 PRANA - MULTI-TIMEFRAME EDITION v5.0.1")
        print("="*70)
        
        # [1/10] Fetch data
        print("\n[1/10] Fetching BTCUSDT daily data...")
        df_daily = luxor.fetch_real_binance_data(use_cache=True)
        
        if df_daily is None or len(df_daily) < 100:
            raise HTTPException(status_code=500, detail="Insufficient data")
        
        print(f"       Loaded {len(df_daily)} daily candles")
        
        current_price = safe_float(df_daily['close'].iloc[-1])
        signal_date = df_daily.iloc[-1]['date']
        if hasattr(signal_date, 'strftime'):
            signal_date_str = signal_date.strftime('%Y-%m-%d %H:%M:%S')
        else:
            signal_date_str = str(signal_date)
        
        # [2/10] Analyze all timeframes
        print("[2/10] Analyzing all timeframes...")
        tf_analyses = analyze_all_timeframes(df_daily)
        
        for tf in ['1D', '3D', '1W', '1M']:
            if tf_analyses.get(tf):
                print(f"       {tf}: {tf_analyses[tf]['direction']} (State {tf_analyses[tf]['enneagram_state']})")
            else:
                print(f"       {tf}: No data")
        
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
        
        weekly = tf_analyses.get('1W') or tf_analyses.get('1D')
        monthly = tf_analyses.get('1M') or weekly
        daily_tf = tf_analyses.get('1D')
        
        gann_levels = weekly['gann_levels'] if weekly else calculate_gann_eighths(high_52, low_52)
        
        # Prepare clean timeframes dict
        timeframes_clean = {}
        for tf, analysis in tf_analyses.items():
            if analysis:
                timeframes_clean[tf] = {
                    'direction': str(analysis['direction']),
                    'signal_type': str(analysis['signal_type']),
                    'weight': int(analysis['weight']),
                    'weighted_contribution': int(analysis['weighted_contribution']),
                    'enneagram_state': int(analysis['enneagram_state']),
                    'state_name': str(analysis['state_name']),
                    'rsi': safe_round(analysis['rsi'], 2),
                    'macd_histogram': safe_round(analysis['macd_histogram'], 4),
                    'adx': safe_round(analysis['adx'], 2),
                    'trend_strength': str(analysis['trend_strength']),
                    'cloud_signal': str(analysis['cloud_signal']),
                    'tk_cross': str(analysis['tk_cross']),
                    'kijun_flat': bool(analysis['kijun_flat']),
                    'gann_50_pct': safe_round(analysis['gann_50_pct'], 2),
                    'bullish_signals': int(analysis['bullish_signals']),
                    'bearish_signals': int(analysis['bearish_signals'])
                }
        
        # Build response
        response_data = {
            "status": "success",
            "version": "5.0.1",
            "timestamp": datetime.now().isoformat(),
            
            # Legacy fields
            "symbol": "BTCUSDT",
            "signal_type": str(consensus['signal_type']),
            "signal_date": str(signal_date_str),
            "entry_price": safe_round(current_price, 2),
            "take_profit": safe_round(targets['tp1'], 2),
            "stop_loss": safe_round(targets['stop_loss'], 2),
            "confidence": int(consensus['weighted_score']),
            "confluence_score": int(consensus['weighted_score']),
            "active_pivot_id": int(len(df_daily) - 1),
            "enneagram_state": int(weekly['enneagram_state']) if weekly else 9,
            "price_confluences": int(targets['tp1_confluence_strength']),
            "time_confluences": int(len(time_forecast['cycle_sources'])),
            "gann_cycle_target": int(time_forecast['days_to_pivot']),
            "enneagram_arrow": str(weekly['arrow_type']) if weekly else "NEUTRAL",
            "macd_signal": str(safe_round(weekly['macd'], 4)) if weekly else "0",
            "ichimoku_signal": str(weekly['cloud_signal']) if weekly else "NEUTRAL",
            "rsi_value": safe_round(weekly['rsi'], 2) if weekly else 50.0,
            "gann_sq9_levels": json.dumps(sq9_analysis['from_current'][:4]),
            "gann_angles_active": json.dumps([int(l['angle']) for l in sq9_analysis['from_current'] if abs(l['distance_up_pct']) < 5 or abs(l['distance_down_pct']) < 5]),
            "confluence_details": json.dumps({
                'score': int(consensus['weighted_score']),
                'alignment': str(consensus['alignment_count']),
                'sources': targets['tp1_sources']
            }),
            
            # New MTF fields
            "primary_direction": str(consensus['direction']),
            "weighted_score": int(consensus['weighted_score']),
            "mtf_alignment": str(consensus['alignment_count']),
            "market_regime": str(regime['regime']),
            "timeframes": timeframes_clean,
            "consensus": consensus,
            "price_targets": targets,
            "target_1": safe_round(targets['tp1'], 2),
            "target_2": safe_round(targets['tp2'], 2),
            "target_3": safe_round(targets['tp3'], 2),
            "time_forecast": time_forecast,
            "pivot_forecast_primary": {
                "date": str(time_forecast['next_pivot_date']),
                "date_display": str(time_forecast['next_pivot_display']),
                "days_from_now": int(time_forecast['days_to_pivot']),
                "expected_pivot": str(time_forecast['pivot_type']),
                "confidence": int(time_forecast['pivot_confidence']),
                "cycle_type": str(time_forecast['cycle_sources'][0]) if time_forecast['cycle_sources'] else "projection"
            },
            "price_time_forecast": {
                "days_to_pivot": int(time_forecast['days_to_pivot']),
                "atr_daily": safe_round(time_forecast['atr_daily'], 2),
                "max_expected_move": safe_round(time_forecast['max_expected_move'], 2),
                "probable_price_low": safe_round(time_forecast['probable_price_low'], 2),
                "probable_price_high": safe_round(time_forecast['probable_price_high'], 2),
                "probable_range_text": str(time_forecast['probable_range_text']),
                "target_1_reachable": bool(abs(targets['tp1'] - current_price) < time_forecast['max_expected_move']),
                "target_2_reachable": bool(abs(targets['tp2'] - current_price) < time_forecast['max_expected_move']),
                "target_3_reachable": bool(abs(targets['tp3'] - current_price) < time_forecast['max_expected_move']),
                "direction": str(consensus['direction'])
            },
            "sq9_analysis": sq9_analysis,
            "invalidation": invalidation,
            "strategy": strategy,
            
            # Legacy extended fields
            "state": int(weekly['enneagram_state']) if weekly else 9,
            "state_name": str(weekly['state_name']) if weekly else "Equilibrium",
            "phase": str(weekly['state_phase']) if weekly else "Unknown",
            "target_state": int(weekly['target_state']) if weekly else 9,
            "target_state_name": str(weekly['target_state_name']) if weekly else "Equilibrium",
            "arrow": str(weekly['arrow_type']) if weekly else "NEUTRAL",
            "arrow_meaning": str(weekly['arrow_meaning']) if weekly else "",
            "price_direction": str(consensus['direction']),
            "direction_emoji": "⬇️" if consensus['direction'] == "BEARISH" else ("⬆️" if consensus['direction'] == "BULLISH" else "➡️"),
            "direction_probability": int(consensus['weighted_score']),
            "direction_reasoning": f"{consensus['alignment_count']} timeframes aligned",
            "bullish_signals": int(sum(tf['bullish_signals'] for tf in tf_analyses.values() if tf)),
            "bearish_signals": int(sum(tf['bearish_signals'] for tf in tf_analyses.values() if tf)),
            "major_low": safe_round(gann_levels['major_low'], 2),
            "major_high": safe_round(gann_levels['major_high'], 2),
            "gann_range": safe_round(gann_levels['range'], 2),
            "gann_3_8": safe_round(gann_levels['3_8'], 2),
            "gann_4_8": safe_round(gann_levels['4_8'], 2),
            "gann_5_8": safe_round(gann_levels['5_8'], 2),
            "tenkan": safe_round(weekly['tenkan'], 2) if weekly else 0.0,
            "kijun": safe_round(weekly['kijun'], 2) if weekly else 0.0,
            "cloud_top": safe_round(weekly['cloud_top'], 2) if weekly else 0.0,
            "cloud_bottom": safe_round(weekly['cloud_bottom'], 2) if weekly else 0.0,
            "cloud_signal": str(weekly['cloud_signal']) if weekly else "NEUTRAL",
            "tk_cross": str(weekly['tk_cross']) if weekly else "NEUTRAL",
            "kijun_flat": bool(weekly['kijun_flat']) if weekly else False,
            "adx": safe_round(weekly['adx'], 2) if weekly else 25.0,
            "trend_strength": str(weekly['trend_strength']) if weekly else "MODERATE",
            "regime": regime,
            "rsi": safe_round(daily_tf['rsi'], 2) if daily_tf else 50.0,
            "macd": safe_round(weekly['macd'], 4) if weekly else 0.0,
            "macd_hist": safe_round(weekly['macd_histogram'], 4) if weekly else 0.0,
            "atr": safe_round(daily_tf['atr'], 2) if daily_tf else 1000.0,
            "candles_analyzed": int(len(df_daily)),
            "last_candle_date": str(signal_date_str),
            "signal_strength": str(consensus['confidence_level']),
            "confirmation_score": int(consensus['weighted_score']),
            "confirmation_score_display": f"{consensus['weighted_score']}%",
            "active_pivot": {
                "id": int(len(df_daily) - 1),
                "date": str(signal_date_str),
                "price": safe_round(current_price, 2),
                "type": str(time_forecast['pivot_type'])
            },
            "strong_confluence_zones": [
                {"price": safe_round(targets['tp1'], 2), "type": "TARGET", "sources": targets['tp1_sources']},
                {"price": safe_round(targets['stop_loss'], 2), "type": "STOP", "sources": targets['stop_sources']}
            ],
            "gann_time_windows": [
                {
                    "cycle_length": int(time_forecast['days_to_pivot']),
                    "target_date": str(time_forecast['next_pivot_date']),
                    "date_display": str(time_forecast['next_pivot_display']),
                    "days_from_now": int(time_forecast['days_to_pivot'])
                }
            ]
        }
        
        # Convert all numpy types to native Python types
        response_data = to_native(response_data)
        
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
        for tf in ['1M', '1W', '3D', '1D']:
            if tf_analyses.get(tf):
                print(f"    {tf}: {tf_analyses[tf]['direction']} ({TIMEFRAME_WEIGHTS[tf]}%)")
            else:
                print(f"    {tf}: N/A")
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
        "version": "5.0.1",
        "timestamp": datetime.now().isoformat(),
        "system": "LUXOR V7 PRANA - MULTI-TIMEFRAME EDITION"
    }


@app.on_event("startup")
async def startup():
    print("\n" + "="*70)
    print("  LUXOR V7 PRANA RUNTIME - MULTI-TIMEFRAME EDITION")
    print("  Version 5.0.1")
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
