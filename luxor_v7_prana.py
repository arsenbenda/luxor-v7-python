import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class LuxorV7PranaSystem:
    """LUXOR V7 PRANA - GANN EGYPT-INDIA UNIFIED SYSTEM"""
    
    def __init__(self, initial_capital=10000, commission=0.001, slippage=0.0005):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.trades = []
        self.equity_curve = []
        self.signals_log = []
        self.sidereal_epoch = pd.Timestamp('1900-01-01')
    
    def fetch_real_binance_data(self):
        """Scarica dati reali Binance BTCUSDT 2017-2026"""
    try:
        print("⏳ Downloading BTCUSDT data...")
        url = "https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_d.csv"
        df = pd.read_csv(url, skiprows=1)  # Skip header row
        
        # Seleziona solo le colonne che servono
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume BTC']]
        
        # Rinomina colonne
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        
        # Converti date e valori numerici
        df['date'] = pd.to_datetime(df['date'])
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        df = df.sort_values('date').reset_index(drop=True)
        df = df[(df['date'] >= '2017-01-01') & (df['date'] <= '2026-01-09')]
        df = df.dropna()
        
        print(f"✅ Downloaded {len(df)} candles")
        return df
    except Exception as e:
        print(f"❌ Error in fetch_real_binance_data: {e}")
        import traceback
        traceback.print_exc()
        return None

    
    def calculate_law_of_three_four(self, df, lookback=52):
        """Law of 3 & 4: 90°, 120°, 240° angles"""
        df['high_52'] = df['high'].rolling(window=lookback).max()
        df['low_52'] = df['low'].rolling(window=lookback).min()
        df['range_52'] = df['high_52'] - df['low_52']
        df['angle_90_support'] = df['low_52'] + (df['range_52'] * 0.25)
        df['angle_90_resistance'] = df['low_52'] + (df['range_52'] * 0.75)
        df['angle_120_lower'] = df['low_52'] + (df['range_52'] * 0.333)
        df['angle_120_upper'] = df['low_52'] + (df['range_52'] * 0.667)
        df['angle_240_lower'] = df['low_52'] + (df['range_52'] * 0.667)
        df['angle_240_upper'] = df['low_52'] + (df['range_52'] * 1.333)
        return df
    
    def calculate_gann_hexagon(self, df, lookback=52):
        """Gann Hexagon: 6/12 subdivision"""
        df['hex_high'] = df['high'].rolling(window=lookback).max()
        df['hex_low'] = df['low'].rolling(window=lookback).min()
        df['hex_range'] = df['hex_high'] - df['hex_low']
        for i in range(1, 6):
            level = i / 6
            df[f'hex_level_{i}'] = df['hex_low'] + (df['hex_range'] * level)
        for i in range(1, 12):
            level = i / 12
            df[f'hex_12_level_{i}'] = df['hex_low'] + (df['hex_range'] * level)
        return df
    
    def calculate_pyramid_price_levels(self, df, lookback=52):
        """Pyramid: 9 livelli egizi"""
        df['pyr_high'] = df['high'].rolling(window=lookback).max()
        df['pyr_low'] = df['low'].rolling(window=lookback).min()
        df['pyr_range'] = df['pyr_high'] - df['pyr_low']
        for i in range(1, 10):
            df[f'pyr_level_{i}'] = df['pyr_low'] + (df['pyr_range'] * (i/9))
        return df
    
    def calculate_60_year_master_cycle(self, df):
        """60-Year Cycle: Jupiter-Saturn conjunction"""
        df['years_since_1900'] = (df['date'].dt.year - 1900)
        df['60_year_cycle_position'] = df['years_since_1900'] % 60
        df['jupiter_position'] = (df['years_since_1900'] % 12)
        df['saturn_position'] = (df['years_since_1900'] % 30)
        df['jupiter_saturn_conjunction'] = ((df['jupiter_position'] - df['saturn_position']) % 20 == 0).astype(int)
        df['cycle_power'] = 1 - (abs(df['60_year_cycle_position'] - 30) / 30)
        return df
    
    def calculate_sidereal_astrology(self, df):
        """Sidereal Astrology: Nakshatras (27 stazioni lunari)"""
        df['days_since_epoch'] = (df['date'] - self.sidereal_epoch).dt.days
        df['nakshatra'] = (df['days_since_epoch'] % 27).astype(int)
        df['nakshatra_phase'] = (df['days_since_epoch'] % 27) / 27
        influential_nakshatras = [0, 5, 13, 17, 22]
        df['influential_nakshatra'] = df['nakshatra'].isin(influential_nakshatras).astype(int)
        df['nakshatra_strength'] = 1 - abs(df['nakshatra_phase'] - 0.5) * 2
        return df
    
    def calculate_chakra_meridians(self, df, lookback=52):
        """Chakra: 7 energy zones"""
        df['chakra_high'] = df['high'].rolling(window=lookback).max()
        df['chakra_low'] = df['low'].rolling(window=lookback).min()
        df['chakra_range'] = df['chakra_high'] - df['chakra_low']
        chakra_names = ['muladhara', 'svadhisthana', 'manipura', 'anahata', 'vishuddha', 'ajna', 'sahasrara']
        for i, name in enumerate(chakra_names, 1):
            level = i / 7
            df[f'chakra_{name}'] = df['chakra_low'] + (df['chakra_range'] * level)
        return df
    
    def calculate_ghatika_intraday_rhythms(self, df):
        """Ghatika: Intraday breathing rhythms"""
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['hour_proxy'] = (df['day_of_month'] % 24)
        df['ghatika_4_cycle'] = (df['day_of_month'] % 4)
        df['ghatika_60_cycle'] = (df['day_of_month'] % 12)
        df['ghatika_240_cycle'] = (df['day_of_month'] % 30)
        df['breath_phase'] = (df['day_of_month'] % 30) / 30
        df['inhale_exhale'] = np.where(df['breath_phase'] < 0.5, 'inhale', 'exhale')
        return df
    
    def calculate_pivot_time_dates(self, df):
        """Pivot Time Dates: Gann + Fibonacci cycles"""
        pivot_2020 = pd.Timestamp('2020-03-23')
        df['days_from_pivot'] = (df['date'] - pivot_2020).dt.days
        pivot_cycles = [90, 144, 180, 233, 252, 360, 377]
        df['days_to_next_pivot'] = df['days_from_pivot'].apply(
            lambda x: min([cycle - (x % cycle) for cycle in pivot_cycles]) if x >= 0 else 999
        )
        df['near_pivot_time'] = (df['days_to_next_pivot'] <= 10).astype(int)
        return df
    
    def calculate_algol_safety_check(self, df):
        """Algol Safety: Fixed star filter"""
        df['days_since_epoch'] = (df['date'] - self.sidereal_epoch).dt.days
        df['algol_phase'] = (df['days_since_epoch'] % 27) / 27
        df['algol_active'] = ((df['algol_phase'] >= 0.3) & (df['algol_phase'] <= 0.7)).astype(int)
        df['algol_safety'] = 0
        df.loc[df['algol_active'] == 1, 'algol_safety'] = 1
        return df
    
    def calculate_time_price_sync_matrix(self, df):
        """Synchronization: Time × Price"""
        time_factors = (
            (df['influential_nakshatra'] * 0.25) +
            (df['near_pivot_time'] * 0.25) +
            ((1 - df['algol_active']) * 0.25) +
            (df['cycle_power'] * 0.25)
        )
        price_factors = (
            ((df['close'] > df['pyr_level_5']) * 0.25) +
            ((df['close'] > df['chakra_anahata']) * 0.25) +
            ((df['close'] > df['hex_level_3']) * 0.25) +
            (((df['close'] > df['angle_120_lower']) & (df['close'] < df['angle_120_upper'])) * 0.25)
        )
        df['time_sync_score'] = time_factors
        df['price_sync_score'] = price_factors
        df['sync_matrix'] = (time_factors * price_factors)
        return df
    
    def calculate_atr(self, df, period=14):
        """ATR Wilder's"""
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift()))
        )
        atr = [np.nan] * period
        atr.append(df['tr'][:period+1].mean())
        for i in range(period + 1, len(df)):
            atr.append((atr[-1] * (period - 1) + df['tr'].iloc[i]) / period)
        return pd.Series(atr, index=df.index)
    
    def calculate_rsi(self, series, period=14):
        """RSI Wilder's"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = [np.nan] * period
        avg_loss = [np.nan] * period
        avg_gain.append(gain[:period+1].mean())
        avg_loss.append(loss[:period+1].mean())
        for i in range(period + 1, len(series)):
            avg_gain.append((avg_gain[-1] * (period - 1) + gain.iloc[i]) / period)
            avg_loss.append((avg_loss[-1] * (period - 1) + loss.iloc[i]) / period)
        rs = np.array(avg_gain) / (np.array(avg_loss) + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return pd.Series(rsi, index=series.index)
    
    def calculate_adx(self, df, period=14):
        """ADX Wilder's DMI"""
        df['plus_dm'] = 0.0
        df['minus_dm'] = 0.0
        for i in range(1, len(df)):
            up = df['high'].iloc[i] - df['high'].iloc[i-1]
            down = df['low'].iloc[i-1] - df['low'].iloc[i]
            if up > down and up > 0:
                df.loc[i, 'plus_dm'] = up
            if down > up and down > 0:
                df.loc[i, 'minus_dm'] = down
        tr = self.calculate_atr(df, period)
        df['plus_di'] = 100 * self._wilder_smooth(df['plus_dm'], period) / (tr + 1e-10)
        df['minus_di'] = 100 * self._wilder_smooth(df['minus_dm'], period) / (tr + 1e-10)
        dx = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-10)
        adx = self._wilder_smooth(dx, period)
        return adx
    
    def _wilder_smooth(self, series, period):
        """Wilder's Smoothing"""
        smooth = [np.nan] * period
        smooth.append(series[:period+1].mean())
        for i in range(period + 1, len(series)):
            if not np.isnan(smooth[-1]):
                smooth.append((smooth[-1] * (period - 1) + series.iloc[i]) / period)
            else:
                smooth.append(np.nan)
        return pd.Series(smooth, index=series.index)
    
    def calculate_ichimoku(self, df):
        """Ichimoku Kinko Hyo"""
        tenkan_high = df['high'].rolling(window=9).max()
        tenkan_low = df['low'].rolling(window=9).min()
        df['tenkan'] = (tenkan_high + tenkan_low) / 2
        
        kijun_high = df['high'].rolling(window=26).max()
        kijun_low = df['low'].rolling(window=26).min()
        df['kijun'] = (kijun_high + kijun_low) / 2
        
        df['senkou_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(26)
        
        senkou_b_high = df['high'].rolling(window=52).max()
        senkou_b_low = df['low'].rolling(window=52).min()
        df['senkou_b'] = ((senkou_b_high + senkou_b_low) / 2).shift(26)
        
        df['chikou'] = df['close'].shift(-26)
        
        return df
    
    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """MACD"""
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        return df
    
    def calculate_all_indicators(self, df):
        """Calcola TUTTI gli indicatori"""
        print("\n🔄 Calculating EGYPT indicators...")
        df = self.calculate_law_of_three_four(df)
        df = self.calculate_gann_hexagon(df)
        df = self.calculate_pyramid_price_levels(df)
        
        print("⏱️  Calculating INDIA indicators...")
        df = self.calculate_60_year_master_cycle(df)
        df = self.calculate_sidereal_astrology(df)
        df = self.calculate_chakra_meridians(df)
        df = self.calculate_ghatika_intraday_rhythms(df)
        df = self.calculate_pivot_time_dates(df)
        df = self.calculate_algol_safety_check(df)
        
        print("🔗 Calculating SYNCHRONIZATION...")
        df = self.calculate_time_price_sync_matrix(df)
        
        print("📊 Calculating CLASSIC indicators...")
        df['atr'] = self.calculate_atr(df, 14)
        df['rsi'] = self.calculate_rsi(df['close'], 14)
        df['adx'] = self.calculate_adx(df, 14)
        df = self.calculate_ichimoku(df)
        df = self.calculate_macd(df)
        df['ma200'] = df['close'].rolling(window=200).mean()
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['v_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
        
        return df
    
    def evaluate_signals(self, df, idx):
        """Valuta segnali OLISTICI"""
        if idx < 100:
            return {'action': 'WAIT', 'signal_count': 0, 'signals': []}
        
        row = df.iloc[idx]
        close = row['close']
        signals = []
        
        # EGITTO
        if close > row['pyr_level_5']:
            signals.append('PYRAMID_ABOVE_MIDPOINT')
        if close > row['angle_120_upper']:
            signals.append('GANN_ANGLE_120_RESIST')
        if close > row['hex_level_3']:
            signals.append('HEXAGON_MID_LEVEL')
        
        # INDIA
        if row['influential_nakshatra'] == 1:
            signals.append('NAKSHATRA_INFLUENTIAL')
        if row['near_pivot_time'] == 1:
            signals.append('PIVOT_TIME_ACTIVE')
        if row['algol_safety'] == 0:
            signals.append('ALGOL_SAFE')
        if row['cycle_power'] > 0.6:
            signals.append('60YEAR_CYCLE_STRONG')
        
        # SYNC
        if row['sync_matrix'] > 0.5:
            signals.append('TIME_PRICE_SYNC_STRONG')
        
        # CLASSIC
        if close > row['senkou_a'] and close > row['senkou_b']:
            signals.append('ICHIMOKU_BULLISH')
        if row['rsi'] < 50:
            signals.append('RSI_BULLISH')
        if row['adx'] > 15:
            signals.append('ADX_TRENDING')
        if row['v_ratio'] > 1.5:
            signals.append('VOLUME_SPIKE')
        if row['macd'] > row['macd_signal']:
            signals.append('MACD_BULLISH')
        
        signal_count = len(signals)
        action = 'BUY' if signal_count >= 7 else 'WAIT'
        
        return {
            'action': action,
            'signal_count': signal_count,
            'signals': signals,
            'sync_matrix': row['sync_matrix'],
            'algol_safety': row['algol_safety'],
            'nakshatra': row['nakshatra'],
            'cycle_power': row['cycle_power']
        }
    
    def run_backtest(self, df):
        """Backtest COMPLETO"""
        print("\n" + "="*100)
        print("🎯 LUXOR V7 PRANA BACKTEST")
        print("="*100)
        
        df = self.calculate_all_indicators(df)
        print("\n🚀 Running backtest...")
        
        capital = self.initial_capital
        position = None
        entry_price = None
        entry_atr = None
        bars_held = 0
        
        tp_mult = 4.5
        sl_mult = 0.5
        time_exit_days = 45
        rsi_overbought = 75
        
        for idx in range(100, len(df)):
            row = df.iloc[idx]
            date = row['date']
            close = row['close']
            
            signal_eval = self.evaluate_signals(df, idx)
            
            # EXIT
            if position:
                bars_held += 1
                adx_factor = 1.0 + (row['adx'] - 15) / 50 if row['adx'] > 15 else 1.0
                trailing_sl = entry_price - (entry_atr * sl_mult * adx_factor)
                tp = entry_price + (entry_atr * tp_mult)
                
                exit_reason = None
                pnl_pct = 0
                
                if close >= tp:
                    exit_reason = 'TP_HIT'
                    pnl_pct = (tp - entry_price) / entry_price * 100
                elif close <= trailing_sl:
                    exit_reason = 'TRAILING_SL'
                    pnl_pct = (close - entry_price) / entry_price * 100
                elif bars_held >= time_exit_days:
                    exit_reason = 'TIME_EXIT'
                    pnl_pct = (close - entry_price) / entry_price * 100
                elif row['rsi'] > rsi_overbought:
                    exit_reason = 'RSI_OVERBOUGHT'
                    pnl_pct = (close - entry_price) / entry_price * 100
                elif close < row['ma200']:
                    exit_reason = 'REGIME_BREAKDOWN'
                    pnl_pct = (close - entry_price) / entry_price * 100
                
                if exit_reason:
                    pnl_amount = capital * (pnl_pct / 100)
                    capital += pnl_amount
                    
                    self.trades.append({
                        'entry_date': df.iloc[idx - bars_held]['date'],
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': close,
                        'bars_held': bars_held,
                        'pnl_pct': pnl_pct,
                        'pnl_amount': pnl_amount,
                        'exit_reason': exit_reason,
                        'signal_count': signal_eval['signal_count']
                    })
                    
                    position = None
                    entry_price = None
                    entry_atr = None
                    bars_held = 0
            
            # ENTRY
            if not position and signal_eval['action'] == 'BUY':
                if (row['adx'] > 15 and row['algol_safety'] == 0 and signal_eval['sync_matrix'] > 0.3):
                    position = True
                    entry_price = close
                    entry_atr = row['atr']
                    bars_held = 0
                    
                    self.signals_log.append({
                        'date': date,
                        'action': 'BUY',
                        'price': close,
                        'signal_count': signal_eval['signal_count'],
                        'signals': signal_eval['signals'],
                        'nakshatra': signal_eval['nakshatra'],
                        'cycle_power': signal_eval['cycle_power'],
                        'sync_matrix': signal_eval['sync_matrix']
                    })
            
            self.equity_curve.append({
                'date': date,
                'capital': capital,
                'position': position
            })
        
        return capital
    
    def print_report(self, final_capital):
        """Report dettagliato"""
        initial = self.initial_capital
        returns = ((final_capital - initial) / initial) * 100
        
        if self.trades:
            wins = len([t for t in self.trades if t['pnl_pct'] > 0])
            losses = len([t for t in self.trades if t['pnl_pct'] <= 0])
            win_rate = (wins / len(self.trades)) * 100 if self.trades else 0
            avg_win = np.mean([t['pnl_pct'] for t in self.trades if t['pnl_pct'] > 0]) if wins > 0 else 0
            avg_loss = np.mean([t['pnl_pct'] for t in self.trades if t['pnl_pct'] <= 0]) if losses > 0 else 0
            
            winning_pnl = sum([t['pnl_amount'] for t in self.trades if t['pnl_pct'] > 0])
            losing_pnl = abs(sum([t['pnl_amount'] for t in self.trades if t['pnl_pct'] <= 0]))
            profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else (999 if winning_pnl > 0 else 0)
            
            equity = [e['capital'] for e in self.equity_curve]
            running_max = np.maximum.accumulate(equity)
            drawdowns = (np.array(equity) - running_max) / (running_max + 1e-10) * 100
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        else:
            wins = losses = win_rate = avg_win = avg_loss = profit_factor = max_drawdown = 0
        
        print("\n" + "="*100)
        print("📊 LUXOR V7 PRANA - EGYPT-INDIA UNIFIED")
        print("="*100)
        print(f"\n💰 CAPITAL")
        print(f"   Initial: ${initial:,.2f}")
        print(f"   Final: ${final_capital:,.2f}")
        print(f"   Return: {returns:.2f}%")
        print(f"\n🎯 TRADES")
        print(f"   Total: {len(self.trades)}")
        print(f"   Wins: {wins} ({win_rate:.1f}%)")
        print(f"   Losses: {losses}")
        print(f"   Avg Win: {avg_win:.2f}%")
        print(f"   Profit Factor: {profit_factor:.2f}x")
        print(f"\n⚠️  RISK")
        print(f"   Max Drawdown: {max_drawdown:.2f}%")
        print("\n" + "="*100)
        
        return {
            'initial_capital': initial,
            'final_capital': final_capital,
            'total_return_pct': returns,
            'num_trades': len(self.trades),
            'wins': wins,
            'losses': losses,
            'win_rate_pct': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown,
            'verdict': '✅ EXCELLENT' if returns > 1200 and win_rate > 82 else '✅ GOOD' if returns > 800 else '⚠️ TUNE'
        }
