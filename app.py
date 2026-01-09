from fastapi import FastAPI, HTTPException
from datetime import datetime
import uvicorn
from luxor_v7_prana import LuxorV7PranaSystem
from config import *
import traceback
import sys

app = FastAPI(
    title="LUXOR V7 PRANA Runtime",
    version="2.0.0",
    description="Egypt-India Unified Trading System - Optimized"
)

# Initialize once
luxor = LuxorV7PranaSystem(initial_capital=INITIAL_CAPITAL)

@app.get("/signal/daily")
async def get_daily_signal():
    """Generate daily LUXOR V7 signal - OPTIMIZED"""
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
        
        # Format response
        print("[API-3/3] Formatting response...")
        response = {
            'status': 'success',
            'timestamp': output['timestamp'],
            'signal_type': output['signal_type'],
            'entry_price': float(output['entry_price']),
            'stop_loss': float(output['stop_loss']),
            'take_profit': float(output['take_profit']),
            'confidence': int(output['confidence']),
            'signal_count': output['signal_count'],
            'signals': output['signals'],
            'rsi': float(output['rsi']),
            'macd': float(output['macd']),
            'volume_ratio': float(output['volume_ratio']),
            'atr': float(output['atr']),
            'last_date': output['last_date'],
            'candles_analyzed': output['candles_analyzed']
        }
        
        print(f"[API-3/3] ✅ Response ready")
        print("="*80 + "\n")
        sys.stdout.flush()
        
        return response
    
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
    print(f"⚡ Features: Caching, Fast Indicators, Improved Signals")
    print(f"🔗 Endpoints:")
    print(f"   • GET /signal/daily → Daily signal generation")
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
