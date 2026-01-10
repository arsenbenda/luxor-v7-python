from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta
import uvicorn
from luxor_v7_prana import LuxorV7PranaSystem
from config import *
import traceback
import sys
import json

app = FastAPI(
    title="LUXOR V7 PRANA Runtime",
    version="2.0.0",
    description="Egypt-India Unified Trading System - Optimized"
)

# Initialize once
luxor = LuxorV7PranaSystem(initial_capital=INITIAL_CAPITAL)

@app.get("/signal/daily")
async def get_daily_signal():
    """Generate daily LUXOR V7 signal - COMPLETE WITH POSITION FIELDS"""
    try:
        print("\n" + "="*80)
        print("[API] GET /signal/daily")
        print("="*80)
        sys.stdout.flush()
        
        # Fetch data (with caching)
        print("[API-1/3] Fetching data...")
        df = luxor.fetch_real_binance_data(use_cache=True)
        
        if df is None or len(df) < 100:
            raise Exception(f"Insufficient data: {len(df) if df is not None else 0}")
        
        print(f"[API-1/3] ✅ Data: {len(df)} candles")
        sys.stdout.flush()
        
        # Calculate and evaluate
        print("[API-2/3] Calculating signal...")
        output = luxor.get_daily_signal(df)
        print(f"[API-2/3] ✅ Signal: {output.get('signal_type', 'ERROR')}")
        sys.stdout.flush()
        
        if output.get('status') == 'error':
            raise Exception(output.get('detail'))
        
        # Format COMPLETE response with all DB fields
        print("[API-3/3] Formatting complete response...")
        
        # Calculate confluence score
        confluence_score = output['signal_count'] * 10
        
        # Determine Ichimoku signal
        ichimoku_signal = "BULLISH" if output['signal_type'] == 'BUY' else "BEARISH" if output['signal_type'] == 'SELL' else "NEUTRAL"
        
        # Enneagram state mapping (based on signal type and RSI)
        rsi_val = output['rsi']
        if output['signal_type'] == 'BUY':
            if rsi_val < 30:
                enneagram_state = 5
                enneagram_arrow = "Growth"
            elif rsi_val < 50:
                enneagram_state = 7
                enneagram_arrow = "Growth"
            else:
                enneagram_state = 8
                enneagram_arrow = "Growth"
        elif output['signal_type'] == 'SELL':
            if rsi_val > 70:
                enneagram_state = 3
                enneagram_arrow = "Stress"
            elif rsi_val > 60:
                enneagram_state = 4
                enneagram_arrow = "Stress"
            else:
                enneagram_state = 5
                enneagram_arrow = "Stress"
        else:  # WAIT
            if rsi_val < 40:
                enneagram_state = 6
                enneagram_arrow = "Neutral"
            elif rsi_val > 60:
                enneagram_state = 2
                enneagram_arrow = "Neutral"
            else:
                enneagram_state = 9
                enneagram_arrow = "Neutral"
        
        # Gann Square of 9 levels (calculate from entry price)
        entry_price = output['entry_price']
        sq9_base = int(entry_price / 1000) * 1000
        gann_sq9_levels = {
            "base": sq9_base,
            "level_1": sq9_base + 1000,
            "level_2": sq9_base + 2000,
            "level_3": sq9_base + 3000,
            "support_1": sq9_base - 1000,
            "support_2": sq9_base - 2000
        }
        
        # Gann angles (calculate from entry)
        gann_angles_active = {
            "1x1_up": entry_price + (output['atr'] * 1),
            "1x1_down": entry_price - (output['atr'] * 1),
            "2x1_up": entry_price + (output['atr'] * 2),
            "2x1_down": entry_price - (output['atr'] * 2)
        }
        
        # Confluence details
        confluence_details = {
            "price_confluences": output['signal_count'],
            "time_confluences": 0,
            "signals": output['signals'],
            "rsi_confluences": 1 if (rsi_val < 30 or rsi_val > 70) else 0,
            "macd_confluences": 1 if output['macd'] > 0 else 0,
            "volume_confluences": 1 if output['volume_ratio'] > 1.1 else 0
        }
        
        # ===== POSITION FIELDS (For Save Position to DB) =====
        
        now = datetime.now()
        entry_date = now
        
        # Calculate exit date (21 giorni = 3 settimane = Gann cycle standard)
        close_date_calculated = entry_date + timedelta(days=21)
        
        # Close price = TP (target profit)
        close_price_calculated = output['take_profit']
        
        # Calculate P&L
        position_size = 15  # 15%
        pnl_amount = (close_price_calculated - entry_price) * (position_size / 100) * (10000 / 100)  # Basato su $10k capitale
        pnl_pct_calculated = ((close_price_calculated - entry_price) / entry_price) * 100
        
        # Exit reason (intelligente)
        if output['signal_type'] == 'BUY':
            exit_reason_calculated = "TP_HIT"
        elif output['signal_type'] == 'SELL':
            exit_reason_calculated = "TP_HIT"
        else:
            exit_reason_calculated = "TIME_EXIT"  # Esce dopo 21 giorni se WAIT
        
        # Signal response (per Save Signal to DB)
        response_signal = {
            'symbol': 'BTCUSDT',
            'signal_date': now.isoformat() + 'Z',
            'signal_type': output['signal_type'],
            'entry_price': float(output['entry_price']),
            'stop_loss': float(output['stop_loss']),
            'take_profit': float(output['take_profit']),
            'confidence': int(output['confidence']),
            'confluence_score': confluence_score,
            'active_pivot_id': None,
            'gann_sq9_levels': gann_sq9_levels,
            'gann_angles_active': gann_angles_active,
            'enneagram_state': enneagram_state,
            'enneagram_arrow': enneagram_arrow,
            'price_confluences': output['signal_count'],
            'time_confluences': 0,
            'confluence_details': json.dumps(confluence_details),
            'rsi_value': float(output['rsi']),
            'macd_signal': float(output['macd']),
            'ichimoku_signal': ichimoku_signal,
            'status': 'PENDING',
            'timestamp': now.isoformat(),
            'last_date': output['last_date'],
            'candles_analyzed': output['candles_analyzed'],
            'atr': float(output['atr']),
            'volume_ratio': float(output['volume_ratio']),
            'signals_list': output['signals']
        }
        
        # Position response (per Save Position to DB)
        response_position = {
            'symbol': 'BTCUSDT',
            'entry_date': entry_date.isoformat() + 'Z',
            'entry_price': float(output['entry_price']),
            'stop_loss': float(output['stop_loss']),
            'take_profit': float(output['take_profit']),
            'position_size_pct': 15,
            'enneagram_entry_state': enneagram_state,
            'gann_cycle_target': 21,  # 21 giorni = 3 settimane
            'confluence_score_entry': confluence_score,
            'status': 'OPEN',
            'close_date': close_date_calculated.isoformat() + 'Z',
            'close_price': float(close_price_calculated),
            'pnl': float(pnl_amount),
            'pnl_pct': float(pnl_pct_calculated),
            'exit_reason': exit_reason_calculated,
            'signal_type': output['signal_type'],
            'rsi_entry': float(output['rsi']),
            'macd_entry': float(output['macd'])
        }
        
        # Combined response (per N8N passthrough)
        response_data = {
            **response_signal,
            **response_position
        }
        
        print(f"[API-3/3] ✅ Response ready with SIGNAL + POSITION fields")
        print(f"   Entry: ${response_position['entry_price']:.2f}")
        print(f"   Target Close: ${response_position['close_price']:.2f}")
        print(f"   Expected P&L: ${response_position['pnl']:.2f} ({response_position['pnl_pct']:.2f}%)")
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
        'version': '2.0.0',
        'timestamp': datetime.now().isoformat()
    }

@app.on_event("startup")
async def startup_event():
    print("\n" + "="*80)
    print(f"🚀 {SERVICE_NAME} v2.0.0 - OPTIMIZED RUNTIME")
    print("="*80)
    print(f"📊 System: LUXOR V7 PRANA Egypt-India Unified")
    print(f"⚡ Features: Complete Signal+Position Fields, Smart Exit Calc")
    print(f"🔗 Endpoints:")
    print(f"   • GET /signal/daily → Daily signal generation (COMPLETE)")
    print(f"   • GET /health → Health check")
    print(f"⏰ Timestamp: {datetime.now().isoformat()}")
    print("="*80 + "\n")
    sys.stdout.flush()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info"
    )
