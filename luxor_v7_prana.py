import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

class LuxorV7PranaSystem:
    """LUXOR V7 PRANA - GANN EGYPT-INDIA UNIFIED SYSTEM - OPTIMIZED"""
    
    # Cache per evitare re-download
    CACHE = {
        'df': None,
        'last_fetch': None,
        'cache_duration': 3600  # 1 ora
    }
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.sidereal_epoch = pd.Timestamp('1900-01-01')
        print("[INIT] LuxorV7PranaSystem initialized")
    
    def fetch_real_binance_data(self, use_cache=True):
        """Scarica dati con CACHING"""
        try:
            # Check cache
            if use_cache and self.CACHE['df'] is not None:
                cache_age = (datetime.now() - self.CACHE['last_fetch']).total_seconds()
                if cache_age < self.CACHE['cache_duration']:
                    print(f"[CACHE] Using cached data (age: {cache_age:.0f}s)")
                    return self.CACHE['df'].copy()
            
            print("[FETCH] Downloading BTCUSDT data from CryptoDataDownload...")
            url = "https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_d.csv"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            lines = response.text.split('\n')
            header_idx = 0
            for i, line in enumerate(lines):
                if 'Unix' in line and 'Date' in line:
                    header_idx = i
                    break
            
            data_text = '\n'.join(lines[header_idx:])
            from io import StringIO
            df = pd.read_csv(StringIO(data_text))
            
            # Map columns
            cols_lower = {c: c.lower() for c in df.columns}
            date_col = [c for c in df.columns if 'date' in c.lower() and 'unix' not in c.lower()][0]
            open_col = [c for c in df.columns if 'open' in c.lower()][0]
            high_col = [c for c in df.columns if 'high' in c.lower()][0]
            low_col = [c for c in df.columns if 'low' in c.lower()][0]
            close_col = [c for c in df.columns if 'close' in c.lower()][0]
            vol_col = [c for c in df.columns if 'volume' in c.lower() and 'btc' in c.lower()][0]
            
            df = df[[date_col, open_col, high_col, low_col, close_col, vol_col]].copy()
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            
            # Convert types
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            df = df.sort_values('date').reset_index(drop=True)
            df = df[(df['date'] >= '2017-01-01') & (df['date'] <= '2026-01-09')]
            
            print(f"[FETCH] Downloaded {len(df)} candles")
            
            # Save to cache
            self.CACHE['df'] = df.copy()
            self.CACHE['last_fetch'] = datetime.now()
            
            return df
            
        except Exception as e:
            print(f"[ERROR] fetch_real_binance_data: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Calcola SOLO gli indicatori essenziali (VELOCE)"""
        try:
            # Price levels (Gann)
            df['high_52'] = df['high'].rolling(52).max()
            df['low_52'] = df['low'].rolling(52).min()
            df['range_52'] = df['high_52'] - df['low_52']
            df['pivot_50'] = (df['high_52'] + df['low_52']) / 2
            
            # ATR (veloce)
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift()),
                    abs(df['low'] - df['close'].shift())
                )
            )
            df['atr'] = df['tr'].rolling(14).mean().fillna(df['tr'].mean())
            
            # RSI (veloce)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)
            
            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Ichimoku (semplificato)
            df['tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
            df['kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
            df['senkou_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(26).fillna(df['close'])
            df['senkou_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26).fillna(df['close'])
            
            # Volume
            df['volume_ma'] = df['volume'].rolling(20).mean().fillna(df['volume'])
            df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
            
            # Trend (SMA200)
            df['sma200'] = df['close'].rolling(200).mean().fillna(df['close'])
            df['above_sma200'] = (df['close'] > df['sma200']).astype(int)
            
            # Time-based (Gann cycles)
            df['day_of_month'] = df['date'].dt.day
            df['day_of_year'] = df['date'].dt.dayofyear
            df['near_pivot_cycle'] = ((df['day_of_year'] % 21) <= 2).astype(int)
            
            print("[CALC] Indicators calculated successfully")
            return df
            
        except Exception as e:
            print(f"[ERROR] calculate_indicators: {e}")
            return df
    
    def evaluate_signals_optimized(self, df, idx):
        """Logica segnale MIGLIORATA"""
        try:
            if idx < 100:
                return {
                    'action': 'WAIT',
                    'signal_count': 0,
                    'signals': [],
                    'strength': 0
                }
            
            row = df.iloc[idx]
            close = row['close']
            signals = []
            strength = 0
            
            # ===== ENTRY CONDITIONS =====
            
            # 1. Trend filter (SMA200) - IMPORTANTE
            if row['above_sma200'] == 1:
                signals.append('TREND_UP')
                strength += 2
            
            # 2. RSI extremes
            if row['rsi'] < 30:
                signals.append('RSI_OVERSOLD')
                strength += 3
            elif row['rsi'] < 40:
                signals.append('RSI_WEAK')
                strength += 1
            
            if row['rsi'] > 70:
                signals.append('RSI_OVERBOUGHT')
                strength += 2  # For shorts
            elif row['rsi'] > 60:
                signals.append('RSI_STRONG')
                strength += 1
            
            # 3. MACD bullish
            if row['macd'] > row['macd_signal'] and row['macd_hist'] > 0:
                signals.append('MACD_BULLISH')
                strength += 2
            elif row['macd'] < row['macd_signal'] and row['macd_hist'] < 0:
                signals.append('MACD_BEARISH')
                strength += 1
            
            # 4. Ichimoku cloud
            if close > row['senkou_a'] and close > row['senkou_b']:
                signals.append('ICHIMOKU_BULL')
                strength += 2
            
            # 5. Price support/resistance
            if close > row['pivot_50']:
                signals.append('ABOVE_PIVOT')
                strength += 1
            
            # 6. Volume confirmation
            if row['volume_ratio'] > 1.3:
                signals.append('HIGH_VOLUME')
                strength += 1
            
            # 7. Time cycle (Gann)
            if row['near_pivot_cycle'] == 1:
                signals.append('GANN_CYCLE')
                strength += 1
            
            # ===== DECISION LOGIC =====
            
            action = 'WAIT'
            
            # BUY signal
            if (row['rsi'] < 35 and row['above_sma200'] == 1 and 
                row['macd'] > row['macd_signal'] and row['volume_ratio'] > 1.2):
                action = 'BUY'
            elif (row['rsi'] < 30 and row['above_sma200'] == 1 and strength >= 5):
                action = 'BUY'
            
            # SELL signal
            if (row['rsi'] > 75 and row['above_sma200'] == 0 and 
                row['macd'] < row['macd_signal'] and strength >= 4):
                action = 'SELL'
            elif (row['rsi'] > 70 and row['macd'] < row['macd_signal']):
                action = 'SELL'
            
            return {
                'action': action,
                'signal_count': len(signals),
                'signals': signals,
                'strength': strength,
                'rsi': float(row['rsi']),
                'macd': float(row['macd']),
                'volume_ratio': float(row['volume_ratio'])
            }
        
        except Exception as e:
            print(f"[ERROR] evaluate_signals: {e}")
            return {
                'action': 'WAIT',
                'signal_count': 0,
                'signals': [],
                'strength': 0
            }
    
    def calculate_risk_management(self, row, atr_val):
        """Risk management con ATR"""
        entry = float(row['close'])
        
        # SL e TP dinamici basati su volatilità (ATR)
        sl = entry - (atr_val * 1.5)
        tp = entry + (atr_val * 3.0)
        
        return {
            'entry': entry,
            'sl': sl,
            'tp': tp,
            'risk': entry - sl,
            'reward': tp - entry
        }
    
    def get_daily_signal(self, df):
        """Main: ottieni segnale giornaliero"""
        try:
            print("[SIGNAL] Generating daily signal...")
            
            # Calcola indicatori
            df = self.calculate_indicators(df)
            
            # Valuta segnale
            signal = self.evaluate_signals_optimized(df, len(df) - 1)
            
            # Risk management
            row = df.iloc[-1]
            risk = self.calculate_risk_management(row, row['atr'])
            
            # Build output
            output = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'signal_type': signal['action'],
                'entry_price': risk['entry'],
                'stop_loss': risk['sl'],
                'take_profit': risk['tp'],
                'confidence': min(100, signal['strength'] * 15),
                'signal_count': signal['signal_count'],
                'signals': signal['signals'],
                'rsi': signal['rsi'],
                'macd': signal['macd'],
                'volume_ratio': signal['volume_ratio'],
                'atr': float(row['atr']),
                'last_date': str(row['date'].date()),
                'candles_analyzed': len(df)
            }
            
            print(f"[SIGNAL] Generated: {output['signal_type']} (strength: {signal['strength']})")
            return output
        
        except Exception as e:
            print(f"[ERROR] get_daily_signal: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'detail': str(e)
            }
