from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta
import uvicorn
from luxor_v7_prana import LuxorV7PranaSystem
from config import *
import traceback
import sys
import math
import json
import numpy as np
import pandas as pd

app = FastAPI(
    title="LUXOR V7 PRANA Runtime",
    version="4.0.5",
    description="Enneagram-Gann Integration System - INVINCIBLE Edition with Price Confluence"
)

# Initialize once
luxor = LuxorV7PranaSystem(initial_capital=INITIAL_CAPITAL)


# ============================================================================
# JSON SERIALIZATION HELPER - Fix numpy types
# ============================================================================

def sanitize_for_json(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if obj is None:
        return None
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_json(item) for item in obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif hasattr(obj, 'item'):
        try:
            return obj.item()
        except:
            return str(obj)
    else:
        return obj


# ============================================================================
# ENNEAGRAM-GANN SYSTEM CONSTANTS
# ============================================================================

ENNEAGRAM_ANGLES = {
    1: 0, 4: 40, 2: 80, 8: 120, 5: 160, 7: 200, 9: 240, 6: 280, 3: 320
}

TRANSITION_ARROWS = {
    1: (4, 7), 2: (8, 4), 3: (9, 6), 4: (2, 1), 5: (7, 8),
    6: (3, 9), 7: (1, 5), 8: (5, 2), 9: (6, 3)
}

TRANSITION_DIRECTION = {
    (1, 7): True, (2, 4): False, (3, 6): False, (4, 1): True,
    (5, 8): True, (6, 9): True, (7, 5): False, (8, 2): True,
    (9, 3): True, (1, 4): False, (2, 8): True, (3, 9): False,
    (4, 2): False, (5, 7): True, (6, 3): True, (7, 1): False,
    (8, 5): False, (9, 6): False
}

GANN_CYCLES = {
    'minor': [7, 14, 21],
    'intermediate': [30, 45, 60, 90],
    'major': [120, 144, 180, 270, 360]
}

CYCLE_TOLERANCE = {
    7: 1, 14: 1, 21: 2, 30: 2, 45: 3, 60: 3,
    90: 3, 120: 4, 144: 4, 180: 5, 270: 6, 360: 7
}

GANN_EIGHTHS = {
    '0/8': 0.000, '1/8': 0.125, '2/8': 0.250, '3/8': 0.375,
    '4/8': 0.500, '5/8': 0.625, '6/8': 0.750, '7/8': 0.875, '8/8': 1.000
}

EIGHTHS_IMPORTANCE = {
    '0/8': 100, '1/8': 40, '2/8': 70, '3/8': 85,
    '4/8': 100, '5/8': 85, '6/8': 70, '7/8': 40, '8/8': 100
}

MARKET_STATES = {
    1: {'name': 'Initiation', 'phase': 'Early accumulation / new up-cycle start', 'bias': 'bullish_early'},
    2: {'name': 'Early Distribution', 'phase': 'Early distribution / topping process', 'bias': 'bearish_early'},
    3: {'name': 'Completion', 'phase': 'Final exhaustion / major top', 'bias': 'reversal_imminent'},
    4: {'name': 'Retracement', 'phase': 'Pullback / corrective pause', 'bias': 'neutral'},
    5: {'name': 'Deep Correction', 'phase': 'Capitulation / major low zone', 'bias': 'bullish_opportunity'},
    6: {'name': 'Decision', 'phase': 'Sideways decision zone / coiling', 'bias': 'breakout_pending'},
    7: {'name': 'Expansion', 'phase': 'Healthy uptrend / sustained advance', 'bias': 'bullish_trend'},
    8: {'name': 'Strong Markup', 'phase': 'Late-stage steep markup', 'bias': 'bullish_late_caution'},
    9: {'name': 'Equilibrium', 'phase': 'Flat equilibrium / low-volatility range', 'bias': 'calm_before_storm'}
}


# ============================================================================
# GANN RULE OF EIGHTHS
# ============================================================================

def calculate_gann_eighths(major_high, major_low):
    """Calculate Gann's Rule of Eighths levels."""
    major_high = float(major_high)
    major_low = float(major_low)
    range_size = major_high - major_low
    
    eighths_levels = {}
    for name, ratio in GANN_EIGHTHS.items():
        price = major_low + (range_size * ratio)
        eighths_levels[name] = {
            'price': round(float(price), 2),
            'ratio': float(ratio),
            'percentage': f"{ratio * 100:.1f}%",
            'importance': int(EIGHTHS_IMPORTANCE[name]),
            'type': 'RESISTANCE' if ratio > 0.5 else 'SUPPORT' if ratio < 0.5 else 'EQUILIBRIUM'
        }
    
    return eighths_levels


def find_major_pivots(df, lookback=252):
    """Find major high and low for Gann Eighths calculation."""
    lookback = min(lookback, len(df) - 1)
    recent_data = df.iloc[-lookback:]
    
    major_high = float(recent_data['high'].max())
    major_low = float(recent_data['low'].min())
    
    return {
        'major_high': major_high,
        'major_low': major_low,
        'range': float(major_high - major_low),
        'lookback_days': int(lookback)
    }


# ============================================================================
# ICHIMOKU CLOUD
# ============================================================================

def calculate_ichimoku(df):
    """Calculate complete Ichimoku Cloud components."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou = close.shift(-26)
    cloud_thickness = abs(senkou_a - senkou_b)
    cloud_bullish = senkou_a > senkou_b
    
    return {
        'tenkan': tenkan,
        'kijun': kijun,
        'senkou_a': senkou_a,
        'senkou_b': senkou_b,
        'chikou': chikou,
        'cloud_thickness': cloud_thickness,
        'cloud_bullish': cloud_bullish
    }


def get_ichimoku_levels(df, idx):
    """Get current Ichimoku levels and signals."""
    ichi = calculate_ichimoku(df)
    
    current_price = float(df['close'].iloc[idx])
    
    tenkan_val = float(ichi['tenkan'].iloc[idx]) if not pd.isna(ichi['tenkan'].iloc[idx]) else current_price
    kijun_val = float(ichi['kijun'].iloc[idx]) if not pd.isna(ichi['kijun'].iloc[idx]) else current_price
    senkou_a_val = float(ichi['senkou_a'].iloc[idx]) if not pd.isna(ichi['senkou_a'].iloc[idx]) else current_price
    senkou_b_val = float(ichi['senkou_b'].iloc[idx]) if not pd.isna(ichi['senkou_b'].iloc[idx]) else current_price
    
    future_senkou_a = float((ichi['tenkan'].iloc[idx] + ichi['kijun'].iloc[idx]) / 2) if not pd.isna(ichi['tenkan'].iloc[idx]) else current_price
    high_52 = float(df['high'].iloc[max(0, idx-51):idx+1].max())
    low_52 = float(df['low'].iloc[max(0, idx-51):idx+1].min())
    future_senkou_b = float((high_52 + low_52) / 2)
    
    cloud_top = float(max(senkou_a_val, senkou_b_val))
    cloud_bottom = float(min(senkou_a_val, senkou_b_val))
    
    if current_price > cloud_top:
        price_position = 'ABOVE_CLOUD'
        cloud_signal = 'BULLISH'
    elif current_price < cloud_bottom:
        price_position = 'BELOW_CLOUD'
        cloud_signal = 'BEARISH'
    else:
        price_position = 'INSIDE_CLOUD'
        cloud_signal = 'NEUTRAL'
    
    if tenkan_val > kijun_val:
        tk_cross = 'BULLISH'
    elif tenkan_val < kijun_val:
        tk_cross = 'BEARISH'
    else:
        tk_cross = 'NEUTRAL'
    
    ichi_levels = []
    
    ichi_levels.append({
        'name': 'Tenkan-sen',
        'price': round(tenkan_val, 2),
        'type': 'SUPPORT' if current_price > tenkan_val else 'RESISTANCE',
        'strength': 60,
        'description': 'Short-term equilibrium (9-period)'
    })
    
    ichi_levels.append({
        'name': 'Kijun-sen',
        'price': round(kijun_val, 2),
        'type': 'SUPPORT' if current_price > kijun_val else 'RESISTANCE',
        'strength': 80,
        'description': 'Medium-term equilibrium (26-period)'
    })
    
    ichi_levels.append({
        'name': 'Cloud Top (Senkou)',
        'price': round(cloud_top, 2),
        'type': 'RESISTANCE' if current_price < cloud_top else 'SUPPORT',
        'strength': 85,
        'description': 'Upper cloud boundary'
    })
    
    ichi_levels.append({
        'name': 'Cloud Bottom (Senkou)',
        'price': round(cloud_bottom, 2),
        'type': 'SUPPORT' if current_price > cloud_bottom else 'RESISTANCE',
        'strength': 85,
        'description': 'Lower cloud boundary'
    })
    
    kijun_flat = False
    if idx >= 5:
        kijun_range = float(ichi['kijun'].iloc[idx-5:idx+1].max() - ichi['kijun'].iloc[idx-5:idx+1].min())
        if kijun_range < current_price * 0.005:
            kijun_flat = True
            ichi_levels.append({
                'name': 'Flat Kijun (Magnet)',
                'price': round(kijun_val, 2),
                'type': 'MAGNET',
                'strength': 90,
                'description': 'Price tends to return to flat Kijun'
            })
    
    return {
        'tenkan': round(tenkan_val, 2),
        'kijun': round(kijun_val, 2),
        'senkou_a': round(senkou_a_val, 2),
        'senkou_b': round(senkou_b_val, 2),
        'cloud_top': round(cloud_top, 2),
        'cloud_bottom': round(cloud_bottom, 2),
        'future_senkou_a': round(future_senkou_a, 2),
        'future_senkou_b': round(future_senkou_b, 2),
        'cloud_thickness': round(float(abs(senkou_a_val - senkou_b_val)), 2),
        'cloud_bullish': bool(senkou_a_val > senkou_b_val),
        'price_position': price_position,
        'cloud_signal': cloud_signal,
        'tk_cross': tk_cross,
        'kijun_flat': bool(kijun_flat),
        'levels': ichi_levels
    }


# ============================================================================
# SQUARE OF 9
# ============================================================================

def calculate_square_of_9_levels(price, direction='both', angles=[45, 90, 135, 180, 225, 270, 315, 360]):
    """Calculate Square of 9 price projections based on Gann's geometric principles."""
    price = float(price)
    sqrt_price = math.sqrt(price)
    sq9_levels = []
    
    for angle in angles:
        if direction in ['up', 'both']:
            target_up = (sqrt_price + angle / 180) ** 2
            distance_pct = ((target_up - price) / price) * 100
            sq9_levels.append({
                'angle': int(angle),
                'direction': 'UP',
                'price': round(float(target_up), 2),
                'type': 'RESISTANCE',
                'distance_pct': round(float(distance_pct), 2),
                'strength': 70 + (10 if angle in [90, 180, 270, 360] else 0),
                'active': abs(distance_pct) < 5
            })
        
        if direction in ['down', 'both']:
            target_down = (sqrt_price - angle / 180) ** 2
            if target_down > 0:
                distance_pct = ((target_down - price) / price) * 100
                sq9_levels.append({
                    'angle': int(angle),
                    'direction': 'DOWN',
                    'price': round(float(target_down), 2),
                    'type': 'SUPPORT',
                    'distance_pct': round(float(distance_pct), 2),
                    'strength': 70 + (10 if angle in [90, 180, 270, 360] else 0),
                    'active': abs(distance_pct) < 5
                })
    
    return sorted(sq9_levels, key=lambda x: x['price'])


# ============================================================================
# CONFLUENCE ZONE DETECTION
# ============================================================================

def find_confluence_zones(current_price, gann_eighths, ichimoku_levels, sq9_levels, tolerance_pct=1.0):
    """Find price zones where multiple systems agree."""
    current_price = float(current_price)
    all_levels = []
    
    for name, data in gann_eighths.items():
        all_levels.append({
            'source': 'GANN_EIGHTHS',
            'name': f"Gann {name}",
            'price': float(data['price']),
            'strength': int(data['importance']),
            'type': data['type']
        })
    
    for level in ichimoku_levels['levels']:
        all_levels.append({
            'source': 'ICHIMOKU',
            'name': level['name'],
            'price': float(level['price']),
            'strength': int(level['strength']),
            'type': level['type']
        })
    
    for level in sq9_levels:
        all_levels.append({
            'source': 'SQUARE_OF_9',
            'name': f"Sq9 {level['angle']}° {level['direction']}",
            'price': float(level['price']),
            'strength': int(level['strength']),
            'type': level['type']
        })
    
    all_levels.sort(key=lambda x: x['price'])
    
    confluence_zones = []
    used_indices = set()
    
    for i, level in enumerate(all_levels):
        if i in used_indices:
            continue
        
        zone_levels = [level]
        zone_price = level['price']
        tolerance = zone_price * (tolerance_pct / 100)
        
        for j, other_level in enumerate(all_levels):
            if j != i and j not in used_indices:
                if abs(other_level['price'] - zone_price) <= tolerance:
                    zone_levels.append(other_level)
                    used_indices.add(j)
        
        if len(zone_levels) >= 2:
            used_indices.add(i)
            
            prices = [l['price'] for l in zone_levels]
            avg_price = sum(prices) / len(prices)
            sources = set(l['source'] for l in zone_levels)
            
            base_strength = sum(l['strength'] for l in zone_levels)
            confluence_bonus = (len(zone_levels) - 1) * 15
            source_bonus = (len(sources) - 1) * 20
            
            total_strength = min(100, (base_strength / len(zone_levels)) + confluence_bonus + source_bonus)
            
            types = [l['type'] for l in zone_levels]
            if types.count('SUPPORT') > types.count('RESISTANCE'):
                zone_type = 'SUPPORT'
            elif types.count('RESISTANCE') > types.count('SUPPORT'):
                zone_type = 'RESISTANCE'
            else:
                zone_type = 'PIVOT'
            
            distance_pct = ((avg_price - current_price) / current_price) * 100
            
            confluence_zones.append({
                'price': round(float(avg_price), 2),
                'price_low': round(float(min(prices)), 2),
                'price_high': round(float(max(prices)), 2),
                'type': zone_type,
                'strength': round(float(total_strength), 1),
                'num_confluences': int(len(zone_levels)),
                'sources': list(sources),
                'levels': [{'source': l['source'], 'name': l['name'], 'price': float(l['price'])} for l in zone_levels],
                'distance_pct': round(float(distance_pct), 2),
                'is_nearby': bool(abs(distance_pct) <= 3)
            })
    
    confluence_zones.sort(key=lambda x: x['strength'], reverse=True)
    
    return confluence_zones


def identify_key_zones(confluence_zones, current_price, direction, atr_value):
    """
    Identify the most important zones for the expected move direction.
    Always provides targets using ATR-based fallbacks when confluence zones are insufficient.
    """
    current_price = float(current_price)
    atr = float(atr_value)
    
    supports = [z for z in confluence_zones if z['price'] < current_price]
    resistances = [z for z in confluence_zones if z['price'] > current_price]
    
    supports.sort(key=lambda x: x['price'], reverse=True)
    resistances.sort(key=lambda x: x['price'])
    
    key_zones = {
        'immediate_support': supports[0] if supports else None,
        'major_support': max(supports, key=lambda x: x['strength']) if supports else None,
        'immediate_resistance': resistances[0] if resistances else None,
        'major_resistance': max(resistances, key=lambda x: x['strength']) if resistances else None
    }
    
    def create_synthetic_target(base_price, multiplier, zone_type, label):
        """Create a synthetic target based on ATR multiples (Gann volatility principle)."""
        return {
            'price': round(float(base_price), 2),
            'type': zone_type,
            'strength': max(50, 70 - (abs(multiplier) - 1) * 10),
            'num_confluences': 1,
            'sources': ['ATR_PROJECTION'],
            'distance_pct': round(((base_price - current_price) / current_price) * 100, 2),
            'is_nearby': abs(multiplier) <= 2,
            'synthetic': True,
            'label': label
        }
    
    if direction == 'BULLISH':
        target_zones = resistances.copy()
        
        if len(target_zones) < 1:
            target_zones.append(create_synthetic_target(
                current_price + atr * 1.5, 1.5, 'RESISTANCE', 'ATR 1.5x'
            ))
        if len(target_zones) < 2:
            target_zones.append(create_synthetic_target(
                current_price + atr * 3.0, 3.0, 'RESISTANCE', 'ATR 3x'
            ))
        if len(target_zones) < 3:
            target_zones.append(create_synthetic_target(
                current_price + atr * 4.5, 4.5, 'RESISTANCE', 'ATR 4.5x'
            ))
        
        key_zones['target_1'] = target_zones[0]
        key_zones['target_2'] = target_zones[1] if len(target_zones) > 1 else None
        key_zones['target_3'] = target_zones[2] if len(target_zones) > 2 else None
        
        if supports:
            key_zones['invalidation'] = supports[0]
        else:
            key_zones['invalidation'] = create_synthetic_target(
                current_price - atr * 1.5, -1.5, 'SUPPORT', 'ATR Stop'
            )
    
    else:  # BEARISH
        target_zones = supports.copy()
        
        if len(target_zones) < 1:
            target_zones.append(create_synthetic_target(
                current_price - atr * 1.5, -1.5, 'SUPPORT', 'ATR 1.5x'
            ))
        if len(target_zones) < 2:
            target_zones.append(create_synthetic_target(
                current_price - atr * 3.0, -3.0, 'SUPPORT', 'ATR 3x'
            ))
        if len(target_zones) < 3:
            target_zones.append(create_synthetic_target(
                current_price - atr * 4.5, -4.5, 'SUPPORT', 'ATR 4.5x'
            ))
        
        key_zones['target_1'] = target_zones[0]
        key_zones['target_2'] = target_zones[1] if len(target_zones) > 1 else None
        key_zones['target_3'] = target_zones[2] if len(target_zones) > 2 else None
        
        if resistances:
            key_zones['invalidation'] = resistances[0]
        else:
            key_zones['invalidation'] = create_synthetic_target(
                current_price + atr * 1.5, 1.5, 'RESISTANCE', 'ATR Stop'
            )
    
    return key_zones


# ============================================================================
# ADX CALCULATION
# ============================================================================

def calculate_adx(df, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    try:
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        plus_dm = np.zeros(len(df))
        minus_dm = np.zeros(len(df))
        tr = np.zeros(len(df))
        
        for i in range(1, len(df)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
            
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        atr = pd.Series(tr).rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / (atr + 1e-10)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        
        return adx.fillna(20)
    except Exception as e:
        print(f"[WARNING] ADX calculation error: {e}")
        return pd.Series([20.0] * len(df))


# ============================================================================
# ENNEAGRAM STATE IDENTIFICATION
# ============================================================================

def identify_enneagram_state(df, idx):
    """Identify current Enneagram market state based on ADX, RSI, Volume, and price structure."""
    try:
        if idx < 50:
            return 9, 50, {}
        
        row = df.iloc[idx]
        rsi = float(row['rsi'])
        volume_ratio = float(row['volume_ratio'])
        adx = float(df['adx'].iloc[idx]) if 'adx' in df.columns else 20.0
        
        recent = df.iloc[max(0, idx-10):idx+1]
        higher_highs = int((recent['high'].diff() > 0).sum()) > 5
        higher_lows = int((recent['low'].diff() > 0).sum()) > 5
        lower_highs = int((recent['high'].diff() < 0).sum()) > 5
        lower_lows = int((recent['low'].diff() < 0).sum()) > 5
        
        scores = {}
        
        # State 1: Initiation
        score_1 = 0
        if 0 <= adx <= 25: score_1 += 25
        if 35 <= rsi <= 65: score_1 += 25
        if higher_lows and not higher_highs: score_1 += 30
        if volume_ratio > 1.0: score_1 += 20
        scores[1] = score_1
        
        # State 2: Early Distribution
        score_2 = 0
        if 20 <= adx <= 35: score_2 += 25
        if 50 <= rsi <= 75: score_2 += 25
        if lower_highs: score_2 += 30
        if volume_ratio > 1.3: score_2 += 20
        scores[2] = score_2
        
        # State 3: Completion
        score_3 = 0
        if adx > 35: score_3 += 25
        if rsi > 70: score_3 += 30
        if volume_ratio > 2.0: score_3 += 25
        if higher_highs and rsi > 75: score_3 += 20
        scores[3] = score_3
        
        # State 4: Retracement
        score_4 = 0
        if 15 <= adx <= 35: score_4 += 25
        if 30 <= rsi <= 55: score_4 += 25
        if lower_highs and higher_lows: score_4 += 30
        if volume_ratio < 1.2: score_4 += 20
        scores[4] = score_4
        
        # State 5: Deep Correction
        score_5 = 0
        if adx < 25: score_5 += 20
        if rsi < 35: score_5 += 35
        if lower_lows: score_5 += 25
        if volume_ratio > 1.5: score_5 += 20
        scores[5] = score_5
        
        # State 6: Decision
        score_6 = 0
        if 10 <= adx <= 25: score_6 += 30
        if 35 <= rsi <= 55: score_6 += 25
        if not higher_highs and not lower_lows: score_6 += 25
        if volume_ratio < 0.9: score_6 += 20
        scores[6] = score_6
        
        # State 7: Expansion
        score_7 = 0
        if 25 <= adx <= 45: score_7 += 25
        if 50 <= rsi <= 70: score_7 += 30
        if higher_highs and higher_lows: score_7 += 30
        if volume_ratio > 1.1: score_7 += 15
        scores[7] = score_7
        
        # State 8: Strong Markup
        score_8 = 0
        if adx > 35: score_8 += 25
        if rsi > 60: score_8 += 25
        if higher_highs: score_8 += 25
        if volume_ratio > 1.5: score_8 += 25
        scores[8] = score_8
        
        # State 9: Equilibrium
        score_9 = 0
        if adx < 20: score_9 += 35
        if 40 <= rsi <= 60: score_9 += 25
        if not higher_highs and not lower_lows: score_9 += 20
        if volume_ratio < 0.8: score_9 += 20
        scores[9] = score_9
        
        best_state = max(scores, key=scores.get)
        confidence = min(95, scores[best_state])
        
        return int(best_state), int(confidence), {'adx': adx, 'rsi': rsi, 'scores': {k: int(v) for k, v in scores.items()}}
        
    except Exception as e:
        print(f"[ERROR] identify_enneagram_state: {e}")
        return 9, 50, {}


# ============================================================================
# ARROW DETERMINATION
# ============================================================================

def determine_active_arrow(state, df, idx, market_regime):
    """Determine whether stress or growth arrow is active based on market conditions."""
    stress_target, growth_target = TRANSITION_ARROWS[state]
    
    if idx < 20:
        return 'growth', int(growth_target), True, 50.0
    
    recent_close = float(df['close'].iloc[idx])
    sma_20 = float(df['close'].iloc[idx-20:idx].mean())
    sma_50 = float(df['close'].iloc[max(0,idx-50):idx].mean()) if idx >= 50 else sma_20
    
    momentum_5 = float((df['close'].iloc[idx] - df['close'].iloc[idx-5]) / df['close'].iloc[idx-5] * 100)
    momentum_20 = float((df['close'].iloc[idx] - df['close'].iloc[idx-20]) / df['close'].iloc[idx-20] * 100)
    
    bullish_conditions = 0
    bearish_conditions = 0
    
    if market_regime == 'BULL': bullish_conditions += 2
    elif market_regime == 'BEAR': bearish_conditions += 2
    
    if momentum_5 > 2: bullish_conditions += 1
    elif momentum_5 < -2: bearish_conditions += 1
    
    if momentum_20 > 5: bullish_conditions += 1
    elif momentum_20 < -5: bearish_conditions += 1
    
    if recent_close > sma_20: bullish_conditions += 1
    else: bearish_conditions += 1
    
    if recent_close > sma_50: bullish_conditions += 1
    else: bearish_conditions += 1
    
    if state == 5: bullish_conditions += 2
    elif state == 8 and float(df['rsi'].iloc[idx]) > 70: bearish_conditions += 2
    elif state == 3: bearish_conditions += 2
    
    if bullish_conditions > bearish_conditions:
        arrow = 'growth'
        target = growth_target
    else:
        arrow = 'stress'
        target = stress_target
    
    is_bullish = TRANSITION_DIRECTION.get((state, target), True)
    
    total = bullish_conditions + bearish_conditions
    if arrow == 'growth':
        confidence = min(90.0, 50.0 + (bullish_conditions / max(1, total)) * 40)
    else:
        confidence = min(90.0, 50.0 + (bearish_conditions / max(1, total)) * 40)
    
    return arrow, int(target), bool(is_bullish), round(float(confidence), 1)


# ============================================================================
# GANN TIME WINDOWS
# ============================================================================

def calculate_gann_time_windows(current_state, target_state, pivot_date, cycles=None):
    """
    Calculate Gann time windows for state transition.
    Always returns future dates, recalculating from today if pivot is too old.
    """
    if cycles is None:
        cycles = [30, 90]
    
    current_angle = ENNEAGRAM_ANGLES[current_state]
    target_angle = ENNEAGRAM_ANGLES[target_state]
    
    angular_distance = target_angle - current_angle
    if angular_distance < 0:
        angular_distance += 360
    
    now = datetime.now()
    time_windows = []
    
    for cycle_length in cycles:
        day_target = (angular_distance / 360) * cycle_length
        tolerance = CYCLE_TOLERANCE.get(cycle_length, 3)
        
        target_date = pivot_date + timedelta(days=int(day_target))
        
        # If target date is in the past, recalculate from today
        if target_date < now:
            target_date = now + timedelta(days=int(day_target))
        
        window_start = target_date - timedelta(days=tolerance)
        window_end = target_date + timedelta(days=tolerance)
        days_from_now = (target_date - now).days
        
        time_windows.append({
            'cycle_length': int(cycle_length),
            'angular_distance': int(angular_distance),
            'day_offset': round(float(day_target), 1),
            'target_date': target_date.strftime('%Y-%m-%d'),
            'target_date_display': target_date.strftime('%d/%m/%Y'),
            'window_start': window_start.strftime('%Y-%m-%d'),
            'window_end': window_end.strftime('%Y-%m-%d'),
            'tolerance_days': int(tolerance),
            'days_from_now': int(days_from_now)
        })
    
    return time_windows


# ============================================================================
# ACTIVE PIVOT DETECTION
# ============================================================================

def calculate_active_pivot(df, current_price):
    """Calculate the most significant recent pivot point."""
    try:
        current_price = float(current_price)
        pivots = []
        
        for i in range(2, len(df) - 2):
            # Swing High detection
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i-2] and
                df['high'].iloc[i] > df['high'].iloc[i+1] and 
                df['high'].iloc[i] > df['high'].iloc[i+2]):
                
                vol_avg = float(df['volume'].iloc[max(0,i-5):i].mean())
                vol_ratio = float(df['volume'].iloc[i]) / vol_avg if vol_avg > 0 else 1.0
                
                pivots.append({
                    'id': int(i),
                    'price': float(df['high'].iloc[i]),
                    'type': 'HIGH',
                    'date': df['date'].iloc[i],
                    'distance': float(abs(df['high'].iloc[i] - current_price)),
                    'volume_ratio': float(vol_ratio)
                })
            
            # Swing Low detection
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                df['low'].iloc[i] < df['low'].iloc[i-2] and
                df['low'].iloc[i] < df['low'].iloc[i+1] and 
                df['low'].iloc[i] < df['low'].iloc[i+2]):
                
                vol_avg = float(df['volume'].iloc[max(0,i-5):i].mean())
                vol_ratio = float(df['volume'].iloc[i]) / vol_avg if vol_avg > 0 else 1.0
                
                pivots.append({
                    'id': int(i),
                    'price': float(df['low'].iloc[i]),
                    'type': 'LOW',
                    'date': df['date'].iloc[i],
                    'distance': float(abs(df['low'].iloc[i] - current_price)),
                    'volume_ratio': float(vol_ratio)
                })
        
        if not pivots:
            return None, []
        
        for p in pivots:
            try:
                if hasattr(p['date'], 'days'):
                    days_ago = (df['date'].iloc[-1] - p['date']).days
                else:
                    days_ago = 30
            except:
                days_ago = 30
            p['recency_score'] = float(max(0, 100 - days_ago))
            p['combined_score'] = float(p['volume_ratio'] * 0.4 + p['recency_score'] * 0.6)
        
        best_pivot = max(pivots, key=lambda x: x['combined_score'])
        recent_pivots = sorted(pivots, key=lambda x: x['date'], reverse=True)[:10]
        
        return best_pivot, recent_pivots
        
    except Exception as e:
        print(f"[WARNING] Error calculating pivot: {e}")
        return None, []


# ============================================================================
# CONFIRMATION SCORE
# ============================================================================

def generate_confirmation_score(state, arrow, is_bullish, df, idx, time_windows, confluence_zones, ichimoku):
    """Generate confirmation score based on multiple system alignment."""
    score = 0
    breakdown = {}
    
    # 1. Enneagram Arrow alignment (max 25)
    arrow_score = 20 if arrow else 10
    if (is_bullish and float(df['close'].iloc[idx]) > float(df['close'].iloc[idx-5])):
        arrow_score += 5
    elif (not is_bullish and float(df['close'].iloc[idx]) < float(df['close'].iloc[idx-5])):
        arrow_score += 5
    breakdown['enneagram_arrow'] = min(25, arrow_score)
    score += breakdown['enneagram_arrow']
    
    # 2. Price action structure (max 15)
    recent = df.iloc[max(0, idx-10):idx+1]
    higher_highs = int((recent['high'].diff() > 0).sum())
    higher_lows = int((recent['low'].diff() > 0).sum())
    
    if is_bullish and higher_highs > 5 and higher_lows > 5:
        breakdown['price_action'] = 15
    elif not is_bullish and higher_highs < 5 and higher_lows < 5:
        breakdown['price_action'] = 15
    else:
        breakdown['price_action'] = 8
    score += breakdown['price_action']
    
    # 3. Volume confirmation (max 10)
    vol_ratio = float(df['volume_ratio'].iloc[idx])
    breakdown['volume'] = min(10, int(vol_ratio * 5))
    score += breakdown['volume']
    
    # 4. ADX trend strength (max 10)
    adx = float(df['adx'].iloc[idx]) if 'adx' in df.columns else 20.0
    if 25 <= adx <= 45:
        breakdown['adx'] = 10
    elif 20 <= adx <= 50:
        breakdown['adx'] = 7
    else:
        breakdown['adx'] = 4
    score += breakdown['adx']
    
    # 5. RSI positioning (max 10)
    rsi = float(df['rsi'].iloc[idx])
    if is_bullish and 40 <= rsi <= 60:
        breakdown['rsi'] = 10
    elif not is_bullish and (rsi > 65 or rsi < 35):
        breakdown['rsi'] = 10
    else:
        breakdown['rsi'] = 5
    score += breakdown['rsi']
    
    # 6. Time window alignment (max 10)
    today = datetime.now()
    in_window = False
    for tw in time_windows:
        start = datetime.strptime(tw['window_start'], '%Y-%m-%d')
        end = datetime.strptime(tw['window_end'], '%Y-%m-%d')
        if start <= today <= end:
            in_window = True
            break
    breakdown['time_window'] = 10 if in_window else 3
    score += breakdown['time_window']
    
    # 7. Confluence zone strength (max 15)
    nearby_zones = [z for z in confluence_zones if z['is_nearby']]
    if nearby_zones:
        best_zone = max(nearby_zones, key=lambda x: x['strength'])
        breakdown['confluence'] = min(15, int(best_zone['strength'] * 0.15))
    else:
        breakdown['confluence'] = 5
    score += breakdown['confluence']
    
    # 8. Ichimoku alignment (max 5)
    ichi_score = 0
    if is_bullish:
        if ichimoku['cloud_signal'] == 'BULLISH': ichi_score += 2
        if ichimoku['tk_cross'] == 'BULLISH': ichi_score += 2
        if ichimoku['price_position'] == 'ABOVE_CLOUD': ichi_score += 1
    else:
        if ichimoku['cloud_signal'] == 'BEARISH': ichi_score += 2
        if ichimoku['tk_cross'] == 'BEARISH': ichi_score += 2
        if ichimoku['price_position'] == 'BELOW_CLOUD': ichi_score += 1
    breakdown['ichimoku'] = min(5, ichi_score)
    score += breakdown['ichimoku']
    
    return min(100, int(score)), {k: int(v) for k, v in breakdown.items()}


def get_signal_recommendation(score):
    """Signal strength based on confirmation score."""
    if score >= 70:
        return 'HIGH_CONFIDENCE', 'Full position size', '🟢'
    elif score >= 50:
        return 'MEDIUM_CONFIDENCE', 'Half position size', '🟡'
    else:
        return 'LOW_CONFIDENCE', 'No trade - wait', '🔴'


# ============================================================================
# PRICE DIRECTION CALCULATION
# ============================================================================

def calculate_price_direction(signals_list, signal_type, ichimoku_signal, rsi_val, is_bullish_arrow):
    """Calculate price direction with probability based on multiple factors."""
    bullish_signals = ['TREND_UP', 'RSI_OVERSOLD', 'RSI_WEAK', 'MACD_BULLISH', 
                       'ICHIMOKU_BULL', 'ABOVE_PIVOT', 'RSI_MILD_UP']
    bearish_signals = ['RSI_OVERBOUGHT', 'RSI_STRONG', 'MACD_BEARISH', 
                       'ICHIMOKU_BEAR', 'BELOW_PIVOT', 'RSI_MILD_DOWN']
    
    bullish_count = sum(1 for s in signals_list if s in bullish_signals)
    bearish_count = sum(1 for s in signals_list if s in bearish_signals)
    
    if ichimoku_signal == 'BULLISH':
        bullish_count += 1
    elif ichimoku_signal == 'BEARISH':
        bearish_count += 1
    
    if is_bullish_arrow:
        bullish_count += 1
    else:
        bearish_count += 1
    
    total = bullish_count + bearish_count
    if total == 0:
        total = 1
    
    bullish_ratio = bullish_count / total
    
    if bullish_ratio > 0.65:
        direction = 'BULLISH'
        emoji = '🟢 ↑'
        base_prob = 60 + (bullish_ratio - 0.5) * 60
    elif bullish_ratio < 0.35:
        direction = 'BEARISH'
        emoji = '🔴 ↓'
        base_prob = 60 + (0.5 - bullish_ratio) * 60
    elif bullish_ratio > 0.5:
        direction = 'LEAN_BULLISH'
        emoji = '🟡 ↗'
        base_prob = 50 + (bullish_ratio - 0.5) * 40
    elif bullish_ratio < 0.5:
        direction = 'LEAN_BEARISH'
        emoji = '🟡 ↘'
        base_prob = 50 + (0.5 - bullish_ratio) * 40
    else:
        direction = 'NEUTRAL'
        emoji = '⚪ →'
        base_prob = 50
    
    probability = min(95, max(35, base_prob))
    
    reasoning_parts = []
    if signal_type in ['BUY', 'SELL']:
        reasoning_parts.append(f"{signal_type} signal active")
    if ichimoku_signal != 'NEUTRAL':
        reasoning_parts.append(f"Ichimoku {ichimoku_signal.lower()}")
    if rsi_val < 35:
        reasoning_parts.append("RSI oversold")
    elif rsi_val > 65:
        reasoning_parts.append("RSI overbought")
    
    reasoning = " | ".join(reasoning_parts) if reasoning_parts else "Mixed signals"
    
    return {
        'price_direction': direction,
        'direction_emoji': emoji,
        'direction_probability': round(probability, 1),
        'direction_reasoning': reasoning,
        'bullish_signals_count': bullish_count,
        'bearish_signals_count': bearish_count
    }


# ============================================================================
# MAIN API ENDPOINT
# ============================================================================

@app.get("/signal/daily")
async def get_daily_signal():
    """Generate daily LUXOR V7 signal - INVINCIBLE Edition v4.0.5 with DB-Compatible Fields"""
    try:
        print("\n" + "="*80)
        print("[API] GET /signal/daily - INVINCIBLE Edition v4.0.5")
        print("="*80)
        sys.stdout.flush()
        
        # 1. Fetch data
        print("[1/10] Fetching data...")
        df = luxor.fetch_real_binance_data(use_cache=True)
        
        if df is None or len(df) < 100:
            raise Exception(f"Insufficient data: {len(df) if df is not None else 0}")
        
        print(f"[1/10] ✓ Data: {len(df)} candles")
        sys.stdout.flush()
        
        # 2. Calculate base indicators
        print("[2/10] Calculating base indicators...")
        output = luxor.get_daily_signal(df)
        
        if output.get('status') == 'error':
            raise Exception(output.get('detail'))
        
        df['adx'] = calculate_adx(df)
        idx = len(df) - 1
        current_price = float(df['close'].iloc[idx])
        atr_value = float(df['atr'].iloc[idx]) if 'atr' in df.columns else float(output.get('atr', current_price * 0.02))
        
        print(f"[2/10] ✓ Price: ${current_price:.2f}, RSI: {output['rsi']:.1f}, ATR: ${atr_value:.2f}")
        sys.stdout.flush()
        
        # 3. Find major pivots for Gann Eighths
        print("[3/10] Calculating Gann Rule of Eighths...")
        major_pivots = find_major_pivots(df, lookback=252)
        gann_eighths = calculate_gann_eighths(major_pivots['major_high'], major_pivots['major_low'])
        
        print(f"[3/10] ✓ Range: ${major_pivots['major_low']:.2f} - ${major_pivots['major_high']:.2f}")
        print(f"       50% Level (4/8): ${gann_eighths['4/8']['price']:.2f}")
        sys.stdout.flush()
        
        # 4. Calculate Ichimoku levels
        print("[4/10] Calculating Ichimoku Cloud levels...")
        ichimoku = get_ichimoku_levels(df, idx)
        
        print(f"[4/10] ✓ Cloud: {ichimoku['cloud_signal']}, TK Cross: {ichimoku['tk_cross']}")
        print(f"       Tenkan: ${ichimoku['tenkan']:.2f}, Kijun: ${ichimoku['kijun']:.2f}")
        sys.stdout.flush()
        
        # 5. Calculate Square of 9 levels
        print("[5/10] Calculating Square of 9 levels...")
        sq9_levels = calculate_square_of_9_levels(current_price, direction='both')
        
        print(f"[5/10] ✓ Generated {len(sq9_levels)} Sq9 levels")
        sys.stdout.flush()
        
        # 6. Find confluence zones
        print("[6/10] Finding confluence zones...")
        confluence_zones = find_confluence_zones(current_price, gann_eighths, ichimoku, sq9_levels)
        
        strong_zones = [z for z in confluence_zones if z['strength'] >= 70]
        print(f"[6/10] ✓ Found {len(confluence_zones)} zones, {len(strong_zones)} strong (≥70%)")
        for z in strong_zones[:3]:
            print(f"       ${z['price']:.2f} ({z['type']}) - {z['strength']}% - {z['num_confluences']} confluences")
        sys.stdout.flush()
        
        # 7. Identify Enneagram state
        print("[7/10] Identifying Enneagram state...")
        enneagram_state, state_confidence, state_details = identify_enneagram_state(df, idx)
        state_info = MARKET_STATES[enneagram_state]
        
        print(f"[7/10] ✓ State {enneagram_state}: {state_info['name']} ({state_confidence}%)")
        sys.stdout.flush()
        
        # 8. Determine market regime and arrow
        print("[8/10] Determining direction...")
        sma_200 = float(df['close'].rolling(200).mean().iloc[-1])
        
        if current_price > sma_200 * 1.02:
            market_regime = 'BULL'
        elif current_price < sma_200 * 0.98:
            market_regime = 'BEAR'
        else:
            market_regime = 'RANGE'
        
        arrow_type, target_state, is_bullish, arrow_confidence = determine_active_arrow(
            enneagram_state, df, idx, market_regime
        )
        
        target_info = MARKET_STATES[target_state]
        direction = 'BULLISH' if is_bullish else 'BEARISH'
        direction_emoji = '🟢 ↑' if is_bullish else '🔴 ↓'
        
        print(f"[8/10] ✓ {direction} ({arrow_confidence}%)")
        print(f"       Arrow: {arrow_type.upper()} → State {target_state} ({target_info['name']})")
        sys.stdout.flush()
        
        # 9. Calculate time windows and key zones
        print("[9/10] Calculating time windows and key zones...")
        active_pivot, recent_pivots = calculate_active_pivot(df, current_price)
        
        if active_pivot:
            pivot_date = active_pivot['date']
            if hasattr(pivot_date, 'to_pydatetime'):
                pivot_date = pivot_date.to_pydatetime()
            elif isinstance(pivot_date, str):
                pivot_date = datetime.strptime(pivot_date, '%Y-%m-%d')
        else:
            pivot_date = datetime.now() - timedelta(days=7)
        
        time_windows = calculate_gann_time_windows(enneagram_state, target_state, pivot_date, cycles=[30, 90])
        key_zones = identify_key_zones(confluence_zones, current_price, direction, atr_value)
        
        print(f"[9/10] ✓ Time windows calculated, key zones identified")
        if key_zones.get('target_1'):
            print(f"       Target 1: ${key_zones['target_1']['price']:.2f}")
        if key_zones.get('target_2'):
            print(f"       Target 2: ${key_zones['target_2']['price']:.2f}")
        sys.stdout.flush()
        
        # 10. Generate confirmation score and final signal
        print("[10/10] Generating confirmation score...")
        confirmation_score, score_breakdown = generate_confirmation_score(
            enneagram_state, arrow_type, is_bullish, df, idx,
            time_windows, confluence_zones, ichimoku
        )
        signal_strength, position_advice, strength_emoji = get_signal_recommendation(confirmation_score)
        
        # Calculate price direction
        price_direction_data = calculate_price_direction(
            output.get('signals', []),
            output.get('signal_type', 'WAIT'),
            ichimoku['cloud_signal'],
            output.get('rsi', 50),
            is_bullish
        )
        
        print(f"[10/10] ✓ Confirmation: {confirmation_score}% ({signal_strength})")
        print(f"        Direction: {price_direction_data['price_direction']} ({price_direction_data['direction_probability']}%)")
        sys.stdout.flush()
        
        # Select primary pivot forecast - always provide a future pivot
        future_windows = [tw for tw in time_windows if tw['days_from_now'] > 0]
        if future_windows:
            nearest_window = min(future_windows, key=lambda x: x['days_from_now'])
            primary_pivot = {
                'date': nearest_window['target_date'],
                'date_display': nearest_window['target_date_display'],
                'days_from_now': nearest_window['days_from_now'],
                'expected_pivot': 'HIGH' if is_bullish else 'LOW',
                'confidence': min(85, arrow_confidence),
                'cycle_type': f"{nearest_window['cycle_length']}-day cycle"
            }
        else:
            fallback_date = datetime.now() + timedelta(days=7)
            primary_pivot = {
                'date': fallback_date.strftime('%Y-%m-%d'),
                'date_display': fallback_date.strftime('%d/%m/%Y'),
                'days_from_now': 7,
                'expected_pivot': 'HIGH' if is_bullish else 'LOW',
                'confidence': 55,
                'cycle_type': 'Short-term projection'
            }
        
        # Extract numeric values for DB
        target_1 = key_zones.get('target_1')
        target_2 = key_zones.get('target_2')
        target_3 = key_zones.get('target_3')
        invalidation = key_zones.get('invalidation')
        
        take_profit_price = round(float(target_1['price']), 2) if target_1 else round(current_price + atr_value * 3, 2)
        stop_loss_price = round(float(invalidation['price']), 2) if invalidation else round(current_price - atr_value * 1.5, 2)
        
        # Prepare Sq9 levels for DB (key levels only) - as JSON string
        gann_sq9_levels_db = [
            {'angle': lvl['angle'], 'price': lvl['price'], 'direction': lvl['direction'], 'type': lvl['type']}
            for lvl in sq9_levels if lvl['angle'] in [90, 180, 270, 360]
        ]
        
        # Prepare active Gann angles - as JSON string
        gann_angles_active_db = [lvl['angle'] for lvl in sq9_levels if lvl.get('active', False)]
        
        # Prepare confluence details for DB - as JSON string
        confluence_details_db = {
            'score': confirmation_score,
            'breakdown': score_breakdown,
            'zones_count': len(confluence_zones),
            'strong_zones_count': len(strong_zones),
            'sources': list(set(s for z in strong_zones for s in z.get('sources', [])))
        }
        
        # Build response with ALL fields matching exact DB schema types
        response_data = {
            # ==================== CORE FIELDS ====================
            'status': 'success',
            'version': '4.0.5',
            'timestamp': datetime.now().isoformat(),
            'signal_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': 'BTCUSDT',
            'entry_price': round(current_price, 2),
            
            # ==================== DB FIELDS (EXACT TYPES) ====================
            # These match your PostgreSQL schema exactly
            
            # varchar fields
            'signal_type': str(output.get('signal_type', 'WAIT')),
            'enneagram_arrow': str(arrow_type),
            'macd_signal': str(round(output.get('macd', 0), 4)),
            'ichimoku_signal': str(ichimoku['cloud_signal']),
            
            # numeric fields
            'take_profit': float(take_profit_price),
            'stop_loss': float(stop_loss_price),
            'rsi_value': round(float(output.get('rsi', 50)), 2),
            
            # integer fields
            'confidence': int(output.get('confidence', 0)),
            'confluence_score': int(confirmation_score),
            'active_pivot_id': int(active_pivot['id']) if active_pivot else None,
            'enneagram_state': int(enneagram_state),
            'price_confluences': int(len(strong_zones)),
            'time_confluences': int(len(time_windows)),
            
            # text fields (JSON strings)
            'gann_sq9_levels': json.dumps(gann_sq9_levels_db),
            'gann_angles_active': json.dumps(gann_angles_active_db),
            'confluence_details': json.dumps(confluence_details_db),
            
            # ==================== EXTENDED DATA (for Telegram) ====================
            # These are NOT saved to DB but used for Telegram messages
            
            # Enneagram details
            'state': enneagram_state,
            'state_name': state_info['name'],
            'phase': state_info['phase'],
            'state_confidence': state_confidence,
            'target_state': target_state,
            'target_state_name': target_info['name'],
            'arrow': arrow_type,
            
            # Direction
            'price_direction': price_direction_data['price_direction'],
            'direction_emoji': price_direction_data['direction_emoji'],
            'direction_probability': price_direction_data['direction_probability'],
            'direction_reasoning': price_direction_data['direction_reasoning'],
            'bullish_signals_count': price_direction_data['bullish_signals_count'],
            'bearish_signals_count': price_direction_data['bearish_signals_count'],
            'market_regime': market_regime,
            
            # Gann Eighths
            'major_high': major_pivots['major_high'],
            'major_low': major_pivots['major_low'],
            'gann_range': major_pivots['range'],
            'gann_eighths': gann_eighths,
            'gann_3_8': gann_eighths['3/8']['price'],
            'gann_4_8': gann_eighths['4/8']['price'],
            'gann_5_8': gann_eighths['5/8']['price'],
            
            # Ichimoku
            'tenkan': ichimoku['tenkan'],
            'kijun': ichimoku['kijun'],
            'cloud_top': ichimoku['cloud_top'],
            'cloud_bottom': ichimoku['cloud_bottom'],
            'cloud_signal': ichimoku['cloud_signal'],
            'tk_cross': ichimoku['tk_cross'],
            'kijun_flat': ichimoku['kijun_flat'],
            
            # Confluence zones (for Telegram)
            'confluence_zones': confluence_zones[:10],
            'strong_confluence_zones': strong_zones[:5],
            'key_zones': key_zones,
            
            # Targets with zone details (for Telegram display)
            'take_profit_zone': target_1,
            'stop_loss_zone': invalidation,
            'target_1': target_1,
            'target_2': target_2,
            'target_3': target_3,
            
            # Time windows (for Telegram)
            'gann_time_windows': time_windows,
            'pivot_forecast_primary': primary_pivot,
            'active_pivot': {
                'id': active_pivot['id'] if active_pivot else None,
                'price': active_pivot['price'] if active_pivot else None,
                'type': active_pivot['type'] if active_pivot else None,
                'date': str(active_pivot['date']) if active_pivot else None
            } if active_pivot else None,
            
            # Confirmation (for Telegram)
            'confirmation_score_display': confirmation_score,
            'score_breakdown': score_breakdown,
            'signal_strength': signal_strength,
            'strength_emoji': strength_emoji,
            'position_advice': position_advice,
            
            # Original signals (for Telegram)
            'signals_list': output.get('signals', []),
            'signal_count': output.get('signal_count', 0),
            
            # Indicators (for Telegram)
            'adx_value': round(float(df['adx'].iloc[idx]), 2) if 'adx' in df.columns else 20.0,
            'macd_value': round(output.get('macd', 0), 4),
            'volume_ratio': round(output.get('volume_ratio', 1), 2),
            'atr': round(atr_value, 2),
            
            # Metadata
            'candles_analyzed': len(df),
            'last_candle_date': str(df['date'].iloc[-1].date())
        }
        
        print("\n" + "="*80)
        print("[SUCCESS] INVINCIBLE SIGNAL GENERATED v4.0.5")
        print(f"State: {enneagram_state} → {target_state} ({state_info['name']} to {target_info['name']})")
        print(f"Direction: {price_direction_data['price_direction']} {price_direction_data['direction_probability']}%")
        print(f"Confirmation: {confirmation_score}% ({signal_strength})")
        print(f"Target 1: ${take_profit_price:.2f}")
        print(f"Invalidation: ${stop_loss_price:.2f}")
        print("="*80 + "\n")
        sys.stdout.flush()
        
        return sanitize_for_json(response_data)
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        traceback.print_exc()
        sys.stdout.flush()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        'status': 'healthy',
        'service': 'LUXOR V7 PRANA Runtime',
        'version': '4.0.5',
        'edition': 'INVINCIBLE',
        'timestamp': datetime.now().isoformat()
    }


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    print("\n" + "="*80)
    print("  LUXOR V7 PRANA RUNTIME - INVINCIBLE EDITION v4.0.5")
    print("  Enneagram-Gann Integration System with Price Confluence")
    print("  DB Schema Compatible - All Field Types Matched")
    print("="*80)
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Endpoints:")
    print("    - GET /signal/daily  : Generate daily trading signal")
    print("    - GET /health        : Health check")
    print("="*80 + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
