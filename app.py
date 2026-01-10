from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta
import uvicorn
from luxor_v7_prana import LuxorV7PranaSystem
from config import *
import traceback
import sys
import json
import math
import numpy as np
import pandas as pd

app = FastAPI(
    title="LUXOR V7 PRANA Runtime",
    version="4.0.0",
    description="Enneagram-Gann Integration System - INVINCIBLE Edition with Price Confluence"
)

# Initialize once
luxor = LuxorV7PranaSystem(initial_capital=INITIAL_CAPITAL)


# ============================================================================
# ENNEAGRAM-GANN SYSTEM CONSTANTS
# ============================================================================

# Enneagram Point Angular Positions (degrees on the circle)
ENNEAGRAM_ANGLES = {
    1: 0,    # Initiation
    4: 40,   # Retracement
    2: 80,   # Early Distribution
    8: 120,  # Strong Markup
    5: 160,  # Deep Correction
    7: 200,  # Expansion
    9: 240,  # Equilibrium (Triangle)
    6: 280,  # Decision (Triangle)
    3: 320   # Completion (Triangle)
}

# Transition Arrows Matrix
TRANSITION_ARROWS = {
    1: (4, 7),   # 1 → 4 (stress/bearish), 1 → 7 (growth/bullish)
    2: (8, 4),   # 2 → 8 (stress), 2 → 4 (growth)
    3: (9, 6),   # 3 → 9 (stress), 3 → 6 (growth)
    4: (2, 1),   # 4 → 2 (stress), 4 → 1 (growth)
    5: (7, 8),   # 5 → 7 (stress), 5 → 8 (growth/BULLISH - major opportunity)
    6: (3, 9),   # 6 → 3 (stress), 6 → 9 (growth)
    7: (1, 5),   # 7 → 1 (stress), 7 → 5 (growth)
    8: (5, 2),   # 8 → 5 (stress/BEARISH - major correction), 8 → 2 (growth)
    9: (6, 3)    # 9 → 6 (stress), 9 → 3 (growth)
}

# Transition Directional Bias
TRANSITION_DIRECTION = {
    (1, 7): True, (2, 4): False, (3, 6): False, (4, 1): True,
    (5, 8): True, (6, 9): True, (7, 5): False, (8, 2): True,
    (9, 3): True, (1, 4): False, (2, 8): True, (3, 9): False,
    (4, 2): False, (5, 7): True, (6, 3): True, (7, 1): False,
    (8, 5): False, (9, 6): False
}

# Gann Time Cycles
GANN_CYCLES = {
    'minor': [7, 14, 21],
    'intermediate': [30, 45, 60, 90],
    'major': [120, 144, 180, 270, 360]
}

CYCLE_TOLERANCE = {
    7: 1, 14: 1, 21: 2, 30: 2, 45: 3, 60: 3,
    90: 3, 120: 4, 144: 4, 180: 5, 270: 6, 360: 7
}

# Gann Rule of Eighths - Division levels
GANN_EIGHTHS = {
    '0/8': 0.000,      # Major Low (100% retracement)
    '1/8': 0.125,      # Weak support/resistance
    '2/8': 0.250,      # Important - 25% level
    '3/8': 0.375,      # Strong S/R - First major retracement
    '4/8': 0.500,      # CRITICAL - 50% level (most important)
    '5/8': 0.625,      # Strong S/R - Golden zone
    '6/8': 0.750,      # Important - 75% level
    '7/8': 0.875,      # Weak support/resistance
    '8/8': 1.000       # Major High (0% retracement)
}

# Importance weights for each eighth
EIGHTHS_IMPORTANCE = {
    '0/8': 100, '1/8': 40, '2/8': 70, '3/8': 85,
    '4/8': 100, '5/8': 85, '6/8': 70, '7/8': 40, '8/8': 100
}

# Market State Definitions
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
# GANN RULE OF EIGHTHS - Price Structure Analysis
# ============================================================================

def calculate_gann_eighths(major_high, major_low):
    """
    Calculate Gann's Rule of Eighths levels from major high to major low.
    
    The range is divided into 8 equal parts creating 9 levels:
    - 0/8 (0%) = Major Low
    - 4/8 (50%) = Most important equilibrium
    - 8/8 (100%) = Major High
    
    Key levels: 3/8, 4/8, 5/8 are the most significant for reversals
    """
    range_size = major_high - major_low
    
    eighths_levels = {}
    for name, ratio in GANN_EIGHTHS.items():
        price = major_low + (range_size * ratio)
        eighths_levels[name] = {
            'price': round(price, 2),
            'ratio': ratio,
            'percentage': f"{ratio * 100:.1f}%",
            'importance': EIGHTHS_IMPORTANCE[name],
            'type': 'RESISTANCE' if ratio > 0.5 else 'SUPPORT' if ratio < 0.5 else 'EQUILIBRIUM'
        }
    
    return eighths_levels


def find_major_pivots(df, lookback=252):
    """
    Find major high and low for Gann Eighths calculation.
    Uses 52-week (252 trading days) or available data.
    """
    lookback = min(lookback, len(df) - 1)
    recent_data = df.iloc[-lookback:]
    
    major_high = recent_data['high'].max()
    major_low = recent_data['low'].min()
    
    high_idx = recent_data['high'].idxmax()
    low_idx = recent_data['low'].idxmin()
    
    high_date = df.loc[high_idx, 'date'] if high_idx in df.index else None
    low_date = df.loc[low_idx, 'date'] if low_idx in df.index else None
    
    return {
        'major_high': major_high,
        'major_low': major_low,
        'high_date': high_date,
        'low_date': low_date,
        'range': major_high - major_low,
        'lookback_days': lookback
    }


# ============================================================================
# ICHIMOKU CLOUD - Complete Implementation
# ============================================================================

def calculate_ichimoku(df):
    """
    Calculate complete Ichimoku Cloud components:
    
    - Tenkan-sen (Conversion Line): 9-period midpoint - SHORT-TERM equilibrium
    - Kijun-sen (Base Line): 26-period midpoint - MEDIUM-TERM equilibrium
    - Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted 26 periods ahead
    - Senkou Span B (Leading Span B): 52-period midpoint, shifted 26 periods ahead
    - Chikou Span (Lagging Span): Close shifted 26 periods back
    
    Cloud (Kumo): Area between Senkou A and Senkou B
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Tenkan-sen (9-period)
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    
    # Kijun-sen (26-period)
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    
    # Senkou Span A (shifted 26 periods forward)
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    
    # Senkou Span B (52-period, shifted 26 forward)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    
    # Chikou Span (close shifted 26 back)
    chikou = close.shift(-26)
    
    # Cloud thickness (for strength assessment)
    cloud_thickness = abs(senkou_a - senkou_b)
    
    # Cloud color (bullish if A > B)
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
    """
    Get current Ichimoku levels and signals for confluence analysis.
    """
    ichi = calculate_ichimoku(df)
    
    current_price = df['close'].iloc[idx]
    
    # Current values
    tenkan_val = ichi['tenkan'].iloc[idx] if not pd.isna(ichi['tenkan'].iloc[idx]) else current_price
    kijun_val = ichi['kijun'].iloc[idx] if not pd.isna(ichi['kijun'].iloc[idx]) else current_price
    
    # For Senkou, we need to look at current cloud (which was calculated 26 bars ago)
    senkou_a_val = ichi['senkou_a'].iloc[idx] if not pd.isna(ichi['senkou_a'].iloc[idx]) else current_price
    senkou_b_val = ichi['senkou_b'].iloc[idx] if not pd.isna(ichi['senkou_b'].iloc[idx]) else current_price
    
    # Future cloud (26 bars ahead projection) - use current calculation without shift
    future_senkou_a = ((ichi['tenkan'].iloc[idx] + ichi['kijun'].iloc[idx]) / 2) if not pd.isna(ichi['tenkan'].iloc[idx]) else current_price
    high_52 = df['high'].iloc[max(0, idx-51):idx+1].max()
    low_52 = df['low'].iloc[max(0, idx-51):idx+1].min()
    future_senkou_b = (high_52 + low_52) / 2
    
    # Cloud boundaries
    cloud_top = max(senkou_a_val, senkou_b_val)
    cloud_bottom = min(senkou_a_val, senkou_b_val)
    
    # Price position relative to cloud
    if current_price > cloud_top:
        price_position = 'ABOVE_CLOUD'
        cloud_signal = 'BULLISH'
    elif current_price < cloud_bottom:
        price_position = 'BELOW_CLOUD'
        cloud_signal = 'BEARISH'
    else:
        price_position = 'INSIDE_CLOUD'
        cloud_signal = 'NEUTRAL'
    
    # TK Cross signal
    if tenkan_val > kijun_val:
        tk_cross = 'BULLISH'
    elif tenkan_val < kijun_val:
        tk_cross = 'BEARISH'
    else:
        tk_cross = 'NEUTRAL'
    
    # Determine key S/R levels from Ichimoku
    ichi_levels = []
    
    # Tenkan as immediate S/R
    ichi_levels.append({
        'name': 'Tenkan-sen',
        'price': round(tenkan_val, 2),
        'type': 'SUPPORT' if current_price > tenkan_val else 'RESISTANCE',
        'strength': 60,
        'description': 'Short-term equilibrium (9-period)'
    })
    
    # Kijun as strong S/R
    ichi_levels.append({
        'name': 'Kijun-sen',
        'price': round(kijun_val, 2),
        'type': 'SUPPORT' if current_price > kijun_val else 'RESISTANCE',
        'strength': 80,
        'description': 'Medium-term equilibrium (26-period)'
    })
    
    # Cloud top
    ichi_levels.append({
        'name': 'Cloud Top (Senkou)',
        'price': round(cloud_top, 2),
        'type': 'RESISTANCE' if current_price < cloud_top else 'SUPPORT',
        'strength': 85,
        'description': 'Upper cloud boundary'
    })
    
    # Cloud bottom
    ichi_levels.append({
        'name': 'Cloud Bottom (Senkou)',
        'price': round(cloud_bottom, 2),
        'type': 'SUPPORT' if current_price > cloud_bottom else 'RESISTANCE',
        'strength': 85,
        'description': 'Lower cloud boundary'
    })
    
    # Flat Kijun/Senkou B levels (these act as magnets)
    # Check if Kijun is flat (hasn't changed much in 5 bars)
    kijun_flat = False
    if idx >= 5:
        kijun_range = ichi['kijun'].iloc[idx-5:idx+1].max() - ichi['kijun'].iloc[idx-5:idx+1].min()
        if kijun_range < current_price * 0.005:  # Less than 0.5% movement
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
        'cloud_thickness': round(abs(senkou_a_val - senkou_b_val), 2),
        'cloud_bullish': senkou_a_val > senkou_b_val,
        'price_position': price_position,
        'cloud_signal': cloud_signal,
        'tk_cross': tk_cross,
        'kijun_flat': kijun_flat,
        'levels': ichi_levels
    }


# ============================================================================
# SQUARE OF 9 - Enhanced Implementation
# ============================================================================

def calculate_square_of_9_levels(price, direction='both', angles=[45, 90, 135, 180, 225, 270, 315, 360]):
    """
    Calculate Square of 9 price projections.
    
    Formulas:
    - Upward: (√price + angle/180)²
    - Downward: (√price - angle/180)²
    """
    sqrt_price = math.sqrt(price)
    sq9_levels = []
    
    for angle in angles:
        if direction in ['up', 'both']:
            target_up = (sqrt_price + angle / 180) ** 2
            distance_pct = ((target_up - price) / price) * 100
            sq9_levels.append({
                'angle': angle,
                'direction': 'UP',
                'price': round(target_up, 2),
                'type': 'RESISTANCE',
                'distance_pct': round(distance_pct, 2),
                'strength': 70 + (10 if angle in [90, 180, 270, 360] else 0)
            })
        
        if direction in ['down', 'both']:
            target_down = (sqrt_price - angle / 180) ** 2
            if target_down > 0:
                distance_pct = ((target_down - price) / price) * 100
                sq9_levels.append({
                    'angle': angle,
                    'direction': 'DOWN',
                    'price': round(target_down, 2),
                    'type': 'SUPPORT',
                    'distance_pct': round(distance_pct, 2),
                    'strength': 70 + (10 if angle in [90, 180, 270, 360] else 0)
                })
    
    return sorted(sq9_levels, key=lambda x: x['price'])


# ============================================================================
# CONFLUENCE ZONE DETECTION - The Secret Weapon
# ============================================================================

def find_confluence_zones(current_price, gann_eighths, ichimoku_levels, sq9_levels, tolerance_pct=1.0):
    """
    Find price zones where multiple systems agree.
    
    A confluence zone is created when 2+ systems have levels within tolerance_pct of each other.
    
    Scoring:
    - Gann Eighths: Base importance (40-100)
    - Ichimoku: Strength (60-90)
    - Square of 9: Strength (70-80)
    
    Final zone strength = sum of individual strengths + bonus for # of confluences
    """
    all_levels = []
    
    # Add Gann Eighths
    for name, data in gann_eighths.items():
        all_levels.append({
            'source': 'GANN_EIGHTHS',
            'name': f"Gann {name}",
            'price': data['price'],
            'strength': data['importance'],
            'type': data['type']
        })
    
    # Add Ichimoku levels
    for level in ichimoku_levels['levels']:
        all_levels.append({
            'source': 'ICHIMOKU',
            'name': level['name'],
            'price': level['price'],
            'strength': level['strength'],
            'type': level['type']
        })
    
    # Add Square of 9 levels
    for level in sq9_levels:
        all_levels.append({
            'source': 'SQUARE_OF_9',
            'name': f"Sq9 {level['angle']}° {level['direction']}",
            'price': level['price'],
            'strength': level['strength'],
            'type': level['type']
        })
    
    # Sort by price
    all_levels.sort(key=lambda x: x['price'])
    
    # Find confluences
    confluence_zones = []
    used_indices = set()
    
    for i, level in enumerate(all_levels):
        if i in used_indices:
            continue
        
        zone_levels = [level]
        zone_price = level['price']
        tolerance = zone_price * (tolerance_pct / 100)
        
        # Find nearby levels
        for j, other_level in enumerate(all_levels):
            if j != i and j not in used_indices:
                if abs(other_level['price'] - zone_price) <= tolerance:
                    zone_levels.append(other_level)
                    used_indices.add(j)
        
        if len(zone_levels) >= 2:  # Confluence requires at least 2 agreeing systems
            used_indices.add(i)
            
            # Calculate zone properties
            prices = [l['price'] for l in zone_levels]
            avg_price = sum(prices) / len(prices)
            
            # Count unique sources
            sources = set(l['source'] for l in zone_levels)
            
            # Calculate strength
            base_strength = sum(l['strength'] for l in zone_levels)
            confluence_bonus = (len(zone_levels) - 1) * 15  # Bonus for each additional confluence
            source_bonus = (len(sources) - 1) * 20  # Bonus for different systems agreeing
            
            total_strength = min(100, (base_strength / len(zone_levels)) + confluence_bonus + source_bonus)
            
            # Determine zone type
            types = [l['type'] for l in zone_levels]
            if types.count('SUPPORT') > types.count('RESISTANCE'):
                zone_type = 'SUPPORT'
            elif types.count('RESISTANCE') > types.count('SUPPORT'):
                zone_type = 'RESISTANCE'
            else:
                zone_type = 'PIVOT'
            
            # Distance from current price
            distance_pct = ((avg_price - current_price) / current_price) * 100
            
            confluence_zones.append({
                'price_low': round(min(prices), 2),
                'price_high': round(max(prices), 2),
                'price_avg': round(avg_price, 2),
                'type': zone_type,
                'strength': round(total_strength, 1),
                'num_confluences': len(zone_levels),
                'sources': list(sources),
                'levels': [{'source': l['source'], 'name': l['name'], 'price': l['price']} for l in zone_levels],
                'distance_pct': round(distance_pct, 2),
                'is_nearby': abs(distance_pct) <= 3  # Within 3% of current price
            })
    
    # Sort by strength (strongest first)
    confluence_zones.sort(key=lambda x: x['strength'], reverse=True)
    
    return confluence_zones


def identify_key_zones(confluence_zones, current_price, direction):
    """
    Identify the most important zones for the expected move direction.
    
    Returns:
    - Immediate target (first major zone in direction of move)
    - Major target (strongest zone in direction)
    - Key support (if bullish) or resistance (if bearish)
    - Invalidation level
    """
    supports = [z for z in confluence_zones if z['price_avg'] < current_price]
    resistances = [z for z in confluence_zones if z['price_avg'] > current_price]
    
    # Sort by distance from current price
    supports.sort(key=lambda x: x['price_avg'], reverse=True)  # Nearest first
    resistances.sort(key=lambda x: x['price_avg'])  # Nearest first
    
    key_zones = {
        'immediate_support': supports[0] if supports else None,
        'major_support': max(supports, key=lambda x: x['strength']) if supports else None,
        'immediate_resistance': resistances[0] if resistances else None,
        'major_resistance': max(resistances, key=lambda x: x['strength']) if resistances else None
    }
    
    if direction == 'BULLISH':
        key_zones['target_1'] = resistances[0] if resistances else None
        key_zones['target_2'] = resistances[1] if len(resistances) > 1 else None
        key_zones['target_3'] = resistances[2] if len(resistances) > 2 else None
        key_zones['invalidation'] = supports[0] if supports else None
    else:
        key_zones['target_1'] = supports[0] if supports else None
        key_zones['target_2'] = supports[1] if len(supports) > 1 else None
        key_zones['target_3'] = supports[2] if len(supports) > 2 else None
        key_zones['invalidation'] = resistances[0] if resistances else None
    
    return key_zones


# ============================================================================
# ADX CALCULATION
# ============================================================================

def calculate_adx(df, period=14):
    """Calculate ADX (Average Directional Index)"""
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
        
        # Smoothed averages
        atr = pd.Series(tr).rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / (atr + 1e-10)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        
        return adx.fillna(20)
    except Exception as e:
        print(f"[WARNING] ADX calculation error: {e}")
        return pd.Series([20] * len(df))


# ============================================================================
# ENNEAGRAM STATE IDENTIFICATION
# ============================================================================

def identify_enneagram_state(df, idx):
    """
    Identify current Enneagram market state based on ADX, RSI, Volume criteria.
    """
    try:
        if idx < 50:
            return 9, 50, {}
        
        row = df.iloc[idx]
        rsi = row['rsi']
        volume_ratio = row['volume_ratio']
        adx = df['adx'].iloc[idx] if 'adx' in df.columns else 20
        
        # Price structure analysis
        recent = df.iloc[max(0, idx-10):idx+1]
        higher_highs = (recent['high'].diff() > 0).sum() > 5
        higher_lows = (recent['low'].diff() > 0).sum() > 5
        lower_highs = (recent['high'].diff() < 0).sum() > 5
        lower_lows = (recent['low'].diff() < 0).sum() > 5
        
        # Score each state
        scores = {}
        
        # Point 1 - Initiation
        score_1 = 0
        if 0 <= adx <= 25: score_1 += 25
        if 35 <= rsi <= 65: score_1 += 25
        if higher_lows and not higher_highs: score_1 += 30
        if volume_ratio > 1.0: score_1 += 20
        scores[1] = score_1
        
        # Point 2 - Early Distribution
        score_2 = 0
        if 20 <= adx <= 35: score_2 += 25
        if 50 <= rsi <= 75: score_2 += 25
        if lower_highs: score_2 += 30
        if volume_ratio > 1.3: score_2 += 20
        scores[2] = score_2
        
        # Point 3 - Completion
        score_3 = 0
        if adx > 35: score_3 += 25
        if rsi > 70: score_3 += 30
        if volume_ratio > 2.0: score_3 += 25
        if higher_highs and rsi > 75: score_3 += 20
        scores[3] = score_3
        
        # Point 4 - Retracement
        score_4 = 0
        if 15 <= adx <= 35: score_4 += 25
        if 30 <= rsi <= 55: score_4 += 25
        if lower_highs and higher_lows: score_4 += 30
        if volume_ratio < 1.2: score_4 += 20
        scores[4] = score_4
        
        # Point 5 - Deep Correction
        score_5 = 0
        if adx < 25: score_5 += 20
        if rsi < 35: score_5 += 35
        if lower_lows: score_5 += 25
        if volume_ratio > 1.5: score_5 += 20
        scores[5] = score_5
        
        # Point 6 - Decision
        score_6 = 0
        if 10 <= adx <= 25: score_6 += 30
        if 35 <= rsi <= 55: score_6 += 25
        if not higher_highs and not lower_lows: score_6 += 25
        if volume_ratio < 0.9: score_6 += 20
        scores[6] = score_6
        
        # Point 7 - Expansion
        score_7 = 0
        if 25 <= adx <= 45: score_7 += 25
        if 50 <= rsi <= 70: score_7 += 30
        if higher_highs and higher_lows: score_7 += 30
        if volume_ratio > 1.1: score_7 += 15
        scores[7] = score_7
        
        # Point 8 - Strong Markup
        score_8 = 0
        if adx > 35: score_8 += 25
        if rsi > 60: score_8 += 25
        if higher_highs: score_8 += 25
        if volume_ratio > 1.5: score_8 += 25
        scores[8] = score_8
        
        # Point 9 - Equilibrium
        score_9 = 0
        if adx < 20: score_9 += 35
        if 40 <= rsi <= 60: score_9 += 25
        if not higher_highs and not lower_lows: score_9 += 20
        if volume_ratio < 0.8: score_9 += 20
        scores[9] = score_9
        
        best_state = max(scores, key=scores.get)
        confidence = min(95, scores[best_state])
        
        return best_state, confidence, {'adx': adx, 'rsi': rsi, 'scores': scores}
        
    except Exception as e:
        print(f"[ERROR] identify_enneagram_state: {e}")
        return 9, 50, {}


# ============================================================================
# ARROW DETERMINATION
# ============================================================================

def determine_active_arrow(state, df, idx, market_regime):
    """Determine whether stress or growth arrow is active."""
    stress_target, growth_target = TRANSITION_ARROWS[state]
    
    if idx < 20:
        return 'growth', growth_target, True, 50
    
    recent_close = df['close'].iloc[idx]
    sma_20 = df['close'].iloc[idx-20:idx].mean()
    sma_50 = df['close'].iloc[max(0,idx-50):idx].mean() if idx >= 50 else sma_20
    
    momentum_5 = (df['close'].iloc[idx] - df['close'].iloc[idx-5]) / df['close'].iloc[idx-5] * 100
    momentum_20 = (df['close'].iloc[idx] - df['close'].iloc[idx-20]) / df['close'].iloc[idx-20] * 100
    
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
    
    # State-specific logic
    if state == 5: bullish_conditions += 2
    elif state == 8 and df['rsi'].iloc[idx] > 70: bearish_conditions += 2
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
        confidence = min(90, 50 + (bullish_conditions / max(1, total)) * 40)
    else:
        confidence = min(90, 50 + (bearish_conditions / max(1, total)) * 40)
    
    return arrow, target, is_bullish, round(confidence, 1)


# ============================================================================
# GANN TIME WINDOWS
# ============================================================================

def calculate_gann_time_windows(current_state, target_state, pivot_date, cycles=None):
    """Calculate Gann time windows for state transition."""
    if cycles is None:
        cycles = [30, 90]
    
    current_angle = ENNEAGRAM_ANGLES[current_state]
    target_angle = ENNEAGRAM_ANGLES[target_state]
    
    angular_distance = target_angle - current_angle
    if angular_distance < 0:
        angular_distance += 360
    
    time_windows = []
    
    for cycle_length in cycles:
        day_target = (angular_distance / 360) * cycle_length
        tolerance = CYCLE_TOLERANCE.get(cycle_length, 3)
        
        target_date = pivot_date + timedelta(days=int(day_target))
        window_start = target_date - timedelta(days=tolerance)
        window_end = target_date + timedelta(days=tolerance)
        
        time_windows.append({
            'cycle_length': cycle_length,
            'angular_distance': angular_distance,
            'day_offset': round(day_target, 1),
            'target_date': target_date.strftime('%Y-%m-%d'),
            'target_date_display': target_date.strftime('%d/%m/%Y'),
            'window_start': window_start.strftime('%Y-%m-%d'),
            'window_end': window_end.strftime('%Y-%m-%d'),
            'tolerance_days': tolerance,
            'days_from_now': (target_date - datetime.now()).days
        })
    
    return time_windows


# ============================================================================
# ACTIVE PIVOT DETECTION
# ============================================================================

def calculate_active_pivot(df, current_price):
    """Calculate the most significant recent pivot."""
    try:
        pivots = []
        
        for i in range(2, len(df) - 2):
            # Swing High
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i-2] and
                df['high'].iloc[i] > df['high'].iloc[i+1] and 
                df['high'].iloc[i] > df['high'].iloc[i+2]):
                
                vol_avg = df['volume'].iloc[max(0,i-5):i].mean()
                vol_ratio = df['volume'].iloc[i] / vol_avg if vol_avg > 0 else 1
                
                pivots.append({
                    'id': i,
                    'price': df['high'].iloc[i],
                    'type': 'HIGH',
                    'date': df['date'].iloc[i],
                    'distance': abs(df['high'].iloc[i] - current_price),
                    'volume_ratio': vol_ratio
                })
            
            # Swing Low
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                df['low'].iloc[i] < df['low'].iloc[i-2] and
                df['low'].iloc[i] < df['low'].iloc[i+1] and 
                df['low'].iloc[i] < df['low'].iloc[i+2]):
                
                vol_avg = df['volume'].iloc[max(0,i-5):i].mean()
                vol_ratio = df['volume'].iloc[i] / vol_avg if vol_avg > 0 else 1
                
                pivots.append({
                    'id': i,
                    'price': df['low'].iloc[i],
                    'type': 'LOW',
                    'date': df['date'].iloc[i],
                    'distance': abs(df['low'].iloc[i] - current_price),
                    'volume_ratio': vol_ratio
                })
        
        if not pivots:
            return None, []
        
        # Score pivots
        for p in pivots:
            days_ago = (df['date'].iloc[-1] - p['date']).days if hasattr(p['date'], 'days') else 30
            p['recency_score'] = max(0, 100 - days_ago)
            p['combined_score'] = p['volume_ratio'] * 0.4 + p['recency_score'] * 0.6
        
        best_pivot = max(pivots, key=lambda x: x['combined_score'])
        recent_pivots = sorted(pivots, key=lambda x: x['date'], reverse=True)[:10]
        
        return best_pivot, recent_pivots
        
    except Exception as e:
        print(f"[WARNING] Error calculating pivot: {e}")
        return None, []


# ============================================================================
# CONFIRMATION SCORE - Enhanced with Confluence
# ============================================================================

def generate_confirmation_score(state, arrow, is_bullish, df, idx, time_windows, confluence_zones, ichimoku):
    """
    Enhanced confirmation score including confluence analysis.
    
    Weights:
    - Enneagram arrow alignment: 25%
    - Price action: 15%
    - Volume pattern: 10%
    - ADX trend: 10%
    - RSI zone: 10%
    - Time window alignment: 10%
    - Confluence zone strength: 15%
    - Ichimoku alignment: 5%
    """
    score = 0
    breakdown = {}
    
    # 1. Enneagram arrow (25%)
    arrow_score = 20 if arrow else 10
    if (is_bullish and df['close'].iloc[idx] > df['close'].iloc[idx-5]):
        arrow_score += 5
    elif (not is_bullish and df['close'].iloc[idx] < df['close'].iloc[idx-5]):
        arrow_score += 5
    breakdown['enneagram_arrow'] = min(25, arrow_score)
    score += breakdown['enneagram_arrow']
    
    # 2. Price action (15%)
    recent = df.iloc[max(0, idx-10):idx+1]
    higher_highs = (recent['high'].diff() > 0).sum()
    higher_lows = (recent['low'].diff() > 0).sum()
    
    if is_bullish and higher_highs > 5 and higher_lows > 5:
        breakdown['price_action'] = 15
    elif not is_bullish and higher_highs < 5 and higher_lows < 5:
        breakdown['price_action'] = 15
    else:
        breakdown['price_action'] = 8
    score += breakdown['price_action']
    
    # 3. Volume (10%)
    vol_ratio = df['volume_ratio'].iloc[idx]
    breakdown['volume'] = min(10, int(vol_ratio * 5))
    score += breakdown['volume']
    
    # 4. ADX (10%)
    adx = df['adx'].iloc[idx] if 'adx' in df.columns else 20
    if 25 <= adx <= 45:
        breakdown['adx'] = 10
    elif 20 <= adx <= 50:
        breakdown['adx'] = 7
    else:
        breakdown['adx'] = 4
    score += breakdown['adx']
    
    # 5. RSI (10%)
    rsi = df['rsi'].iloc[idx]
    if is_bullish and 40 <= rsi <= 60:
        breakdown['rsi'] = 10
    elif not is_bullish and (rsi > 65 or rsi < 35):
        breakdown['rsi'] = 10
    else:
        breakdown['rsi'] = 5
    score += breakdown['rsi']
    
    # 6. Time window (10%)
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
    
    # 7. Confluence zone strength (15%) - NEW
    nearby_zones = [z for z in confluence_zones if z['is_nearby']]
    if nearby_zones:
        best_zone = max(nearby_zones, key=lambda x: x['strength'])
        # Higher score if we're near a strong confluence zone
        breakdown['confluence'] = min(15, int(best_zone['strength'] * 0.15))
    else:
        breakdown['confluence'] = 5
    score += breakdown['confluence']
    
    # 8. Ichimoku alignment (5%) - NEW
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
    
    return min(100, score), breakdown


def get_signal_recommendation(score):
    """Signal strength based on confirmation score."""
    if score >= 70:
        return 'HIGH_CONFIDENCE', 'Full position size', '🟢'
    elif score >= 50:
        return 'MEDIUM_CONFIDENCE', 'Half position size', '🟡'
    else:
        return 'LOW_CONFIDENCE', 'No trade - wait', '🔴'


# ============================================================================
# MAIN API ENDPOINT
# ============================================================================

@app.get("/signal/daily")
async def get_daily_signal():
    """Generate daily LUXOR V7 signal - INVINCIBLE Edition with Price Confluence"""
    try:
        print("\n" + "="*80)
        print("[API] GET /signal/daily - INVINCIBLE Edition v4.0")
        print("="*80)
        sys.stdout.flush()
        
        # 1. Fetch data
        print("[1/10] Fetching data...")
        df = luxor.fetch_real_binance_data(use_cache=True)
        
        if df is None or len(df) < 100:
            raise Exception(f"Insufficient data: {len(df) if df is not None else 0}")
        
        print(f"[1/10] ✅ Data: {len(df)} candles")
        sys.stdout.flush()
        
        # 2. Calculate base indicators
        print("[2/10] Calculating base indicators...")
        output = luxor.get_daily_signal(df)
        
        if output.get('status') == 'error':
            raise Exception(output.get('detail'))
        
        df['adx'] = calculate_adx(df)
        idx = len(df) - 1
        current_price = df['close'].iloc[idx]
        
        print(f"[2/10] ✅ Price: ${current_price:.2f}, RSI: {output['rsi']:.1f}")
        sys.stdout.flush()
        
        # 3. Find major pivots for Gann Eighths
        print("[3/10] Calculating Gann Rule of Eighths...")
        major_pivots = find_major_pivots(df, lookback=252)
        gann_eighths = calculate_gann_eighths(major_pivots['major_high'], major_pivots['major_low'])
        
        print(f"[3/10] ✅ Range: ${major_pivots['major_low']:.2f} - ${major_pivots['major_high']:.2f}")
        print(f"       50% Level (4/8): ${gann_eighths['4/8']['price']:.2f}")
        sys.stdout.flush()
        
        # 4. Calculate Ichimoku levels
        print("[4/10] Calculating Ichimoku Cloud levels...")
        ichimoku = get_ichimoku_levels(df, idx)
        
        print(f"[4/10] ✅ Cloud: {ichimoku['cloud_signal']}, TK Cross: {ichimoku['tk_cross']}")
        print(f"       Tenkan: ${ichimoku['tenkan']:.2f}, Kijun: ${ichimoku['kijun']:.2f}")
        print(f"       Cloud: ${ichimoku['cloud_bottom']:.2f} - ${ichimoku['cloud_top']:.2f}")
        sys.stdout.flush()
        
        # 5. Calculate Square of 9 levels
        print("[5/10] Calculating Square of 9 levels...")
        sq9_levels = calculate_square_of_9_levels(current_price, direction='both')
        
        print(f"[5/10] ✅ Generated {len(sq9_levels)} Sq9 levels")
        sys.stdout.flush()
        
        # 6. Find confluence zones
        print("[6/10] Finding confluence zones...")
        confluence_zones = find_confluence_zones(current_price, gann_eighths, ichimoku, sq9_levels)
        
        strong_zones = [z for z in confluence_zones if z['strength'] >= 70]
        print(f"[6/10] ✅ Found {len(confluence_zones)} zones, {len(strong_zones)} strong (≥70)")
        for z in strong_zones[:3]:
            print(f"       ${z['price_avg']:.2f} ({z['type']}) - {z['strength']}% - {z['num_confluences']} confluences")
        sys.stdout.flush()
        
        # 7. Identify Enneagram state
        print("[7/10] Identifying Enneagram state...")
        enneagram_state, state_confidence, state_details = identify_enneagram_state(df, idx)
        state_info = MARKET_STATES[enneagram_state]
        
        print(f"[7/10] ✅ State {enneagram_state}: {state_info['name']} ({state_confidence}%)")
        sys.stdout.flush()
        
        # 8. Determine market regime and arrow
        print("[8/10] Determining direction...")
        sma_200 = df['close'].rolling(200).mean().iloc[-1]
        
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
        
        print(f"[8/10] ✅ {direction_emoji} {direction} ({arrow_confidence}%)")
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
        key_zones = identify_key_zones(confluence_zones, current_price, direction)
        
        print(f"[9/10] ✅ Time windows and key zones calculated")
        if key_zones['target_1']:
            print(f"       Target 1: ${key_zones['target_1']['price_avg']:.2f}")
        if key_zones['invalidation']:
            print(f"       Invalidation: ${key_zones['invalidation']['price_avg']:.2f}")
        sys.stdout.flush()
        
        # 10. Generate confirmation score
        print("[10/10] Generating final confirmation score...")
        confirmation_score, score_breakdown = generate_confirmation_score(
            enneagram_state, arrow_type, is_bullish, df, idx, time_windows, confluence_zones, ichimoku
        )
        
        signal_strength, position_advice, strength_emoji = get_signal_recommendation(confirmation_score)
        
        print(f"[10/10] ✅ Confirmation: {confirmation_score}% → {signal_strength}")
        sys.stdout.flush()
        
        # Build response
        now = datetime.now()
        
        # Primary pivot forecast
        primary_pivot = None
        if time_windows:
            for tw in sorted(time_windows, key=lambda x: abs(x['days_from_now'])):
                if tw['days_from_now'] >= 0:
                    primary_pivot = {
                        'date': tw['target_date'],
                        'date_display': tw['target_date_display'],
                        'days_from_now': tw['days_from_now'],
                        'cycle': tw['cycle_length'],
                        'expected_pivot': 'HIGH' if is_bullish else 'LOW',
                        'confidence': arrow_confidence
                    }
                    break
        
        # Format confluence zones for response
        formatted_zones = []
        for z in confluence_zones[:10]:
            formatted_zones.append({
                'price': z['price_avg'],
                'price_range': f"${z['price_low']:.2f} - ${z['price_high']:.2f}",
                'type': z['type'],
                'strength': z['strength'],
                'confluences': z['num_confluences'],
                'sources': z['sources'],
                'distance_pct': z['distance_pct'],
                'is_nearby': z['is_nearby']
            })
        
        # Format key zones for Telegram
        key_zones_formatted = {}
        for key, zone in key_zones.items():
            if zone:
                key_zones_formatted[key] = {
                    'price': zone['price_avg'],
                    'strength': zone['strength'],
                    'type': zone['type']
                }
        
        response_data = {
            # Basic info
            'symbol': 'BTCUSDT',
            'signal_date': now.isoformat() + 'Z',
            'timestamp': now.isoformat(),
            'last_date': output['last_date'],
            'candles_analyzed': output['candles_analyzed'],
            
            # Current price
            'entry_price': float(current_price),
            'atr': float(output['atr']),
            
            # Enneagram State
            'enneagram_state': enneagram_state,
            'enneagram_state_name': state_info['name'],
            'enneagram_phase': state_info['phase'],
            'enneagram_state_confidence': state_confidence,
            'enneagram_arrow': arrow_type,
            'enneagram_target_state': target_state,
            'enneagram_target_name': target_info['name'],
            
            # Direction
            'price_direction': direction,
            'direction_emoji': direction_emoji,
            'direction_probability': arrow_confidence,
            'direction_reasoning': f"State {enneagram_state} ({state_info['name']}) → {arrow_type} → State {target_state} ({target_info['name']})",
            
            # Market Regime
            'market_regime': market_regime,
            'sma_200': float(sma_200),
            
            # Confirmation
            'confirmation_score': confirmation_score,
            'signal_strength': signal_strength,
            'position_advice': position_advice,
            'strength_emoji': strength_emoji,
            'score_breakdown': score_breakdown,
            
            # Signal
            'signal_type': 'BUY' if is_bullish and confirmation_score >= 50 else 'SELL' if not is_bullish and confirmation_score >= 50 else 'WAIT',
            'confidence': confirmation_score,
            
            # ===== GANN RULE OF EIGHTHS =====
            'gann_eighths': {
                'major_high': major_pivots['major_high'],
                'major_low': major_pivots['major_low'],
                'range': major_pivots['range'],
                'levels': {k: v['price'] for k, v in gann_eighths.items()},
                'current_position': next(
                    (k for k, v in sorted(gann_eighths.items(), key=lambda x: x[1]['price']) 
                     if v['price'] > current_price), '8/8'
                )
            },
            
            # ===== ICHIMOKU =====
            'ichimoku': {
                'tenkan': ichimoku['tenkan'],
                'kijun': ichimoku['kijun'],
                'senkou_a': ichimoku['senkou_a'],
                'senkou_b': ichimoku['senkou_b'],
                'cloud_top': ichimoku['cloud_top'],
                'cloud_bottom': ichimoku['cloud_bottom'],
                'cloud_signal': ichimoku['cloud_signal'],
                'tk_cross': ichimoku['tk_cross'],
                'price_position': ichimoku['price_position'],
                'kijun_flat': ichimoku['kijun_flat']
            },
            
            # ===== SQUARE OF 9 =====
            'sq9_levels': [
                {'angle': l['angle'], 'direction': l['direction'], 'price': l['price'], 'distance_pct': l['distance_pct']}
                for l in sq9_levels[:12]
            ],
            
            # ===== CONFLUENCE ZONES (THE SECRET WEAPON) =====
            'confluence_zones': formatted_zones,
            'strong_confluence_zones': [z for z in formatted_zones if z['strength'] >= 70],
            'key_zones': key_zones_formatted,
            
            # Targets based on confluence
            'target_1': key_zones_formatted.get('target_1'),
            'target_2': key_zones_formatted.get('target_2'),
            'target_3': key_zones_formatted.get('target_3'),
            'stop_loss': key_zones_formatted.get('invalidation', {'price': float(output['stop_loss'])}),
            'take_profit': key_zones_formatted.get('target_1', {'price': float(output['take_profit'])}),
            
            # Time Windows
            'gann_time_windows': time_windows,
            'pivot_forecast_primary': primary_pivot,
            
            # Technical Indicators
            'rsi_value': float(output['rsi']),
            'macd_signal': float(output['macd']),
            'volume_ratio': float(output['volume_ratio']),
            'adx_value': float(df['adx'].iloc[-1]),
            
            # Signal counts
            'bullish_signals_count': sum(1 for s in output['signals'] if s in ['TREND_UP', 'RSI_OVERSOLD', 'RSI_WEAK', 'MACD_BULLISH', 'ICHIMOKU_BULL', 'ABOVE_PIVOT', 'HIGH_VOLUME']),
            'bearish_signals_count': sum(1 for s in output['signals'] if s in ['RSI_OVERBOUGHT', 'RSI_STRONG', 'MACD_BEARISH', 'ICHIMOKU_BEAR', 'BELOW_PIVOT']),
            'signals_list': output['signals'],
            
            # Legacy compatibility
            'status': 'PENDING',
            'confluence_score': confirmation_score,
            'price_confluences': len([z for z in confluence_zones if z['is_nearby']]),
            'time_confluences': len(time_windows)
        }
        
        # Final summary
        print("\n" + "="*80)
        print("🏆 INVINCIBLE SIGNAL GENERATED")
        print("="*80)
        print(f"   State: {enneagram_state} ({state_info['name']}) → {target_state} ({target_info['name']})")
        print(f"   Direction: {direction_emoji} {direction} ({arrow_confidence}%)")
        print(f"   Confirmation: {strength_emoji} {confirmation_score}%")
        print(f"")
        print(f"   📐 GANN EIGHTHS:")
        print(f"      Range: ${major_pivots['major_low']:.2f} - ${major_pivots['major_high']:.2f}")
        print(f"      50% (4/8): ${gann_eighths['4/8']['price']:.2f}")
        print(f"")
        print(f"   ☁️ ICHIMOKU:")
        print(f"      Cloud: {ichimoku['cloud_signal']} | TK: {ichimoku['tk_cross']}")
        print(f"      Kijun: ${ichimoku['kijun']:.2f}")
        print(f"")
        print(f"   🎯 KEY CONFLUENCE ZONES:")
        for i, z in enumerate(strong_zones[:3], 1):
            print(f"      {i}. ${z['price_avg']:.2f} ({z['type']}) - {z['strength']}% [{z['num_confluences']} confluences]")
        print(f"")
        if primary_pivot:
            print(f"   📅 NEXT PIVOT: {primary_pivot['date_display']} ({primary_pivot['expected_pivot']})")
        print("="*80 + "\n")
        sys.stdout.flush()
        
        return response_data
    
    except Exception as e:
        error_msg = f"[ERROR] /signal/daily: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        sys.stdout.flush()
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/health")
async def health():
    """Health check"""
    return {
        'status': 'healthy',
        'service': SERVICE_NAME,
        'version': '4.0.0 INVINCIBLE',
        'features': [
            'Enneagram-Gann Integration',
            'Rule of Eighths',
            'Ichimoku Cloud',
            'Square of 9',
            'Confluence Zone Detection'
        ],
        'timestamp': datetime.now().isoformat()
    }


@app.on_event("startup")
async def startup_event():
    print("\n" + "="*80)
    print(f"🏆 {SERVICE_NAME} v4.0.0 - INVINCIBLE EDITION")
    print("="*80)
    print(f"")
    print(f"   ⚡ DIRECTION ENGINE:")
    print(f"      • 9 Enneagram Market States")
    print(f"      • Stress/Growth Arrow Detection")
    print(f"      • Angular Time Mapping")
    print(f"")
    print(f"   📐 PRICE ENGINE:")
    print(f"      • Gann Rule of Eighths (8 divisions)")
    print(f"      • Ichimoku Cloud (5 lines)")
    print(f"      • Square of 9 Projections")
    print(f"")
    print(f"   🎯 CONFLUENCE ENGINE:")
    print(f"      • Multi-System Zone Detection")
    print(f"      • Strength Scoring Algorithm")
    print(f"      • Key S/R Identification")
    print(f"")
    print(f"   ⏰ TIME ENGINE:")
    print(f"      • Gann Cycle Windows (30/90/180 days)")
    print(f"      • Pivot Date Forecasting")
    print(f"")
    print(f"🔗 Endpoints:")
    print(f"   • GET /signal/daily → Complete INVINCIBLE signal")
    print(f"   • GET /health → System status")
    print(f"")
    print(f"⏰ Started: {datetime.now().isoformat()}")
    print("="*80 + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info"
    )
