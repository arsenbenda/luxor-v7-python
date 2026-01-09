from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import json
from datetime import datetime
import asyncio
import requests
from luxor_v7_prana import LuxorV7PranaSystem
import uvicorn
from config import *
import traceback
import sys

app = FastAPI(
    title="LUXOR V7 PRANA Runtime",
    version="1.0.0",
    description="Egypt-India Unified Trading System"
)

luxor = LuxorV7PranaSystem(initial_capital=INITIAL_CAPITAL)

@app.get("/signal/daily")
async def get_daily_signal():
    """Generate daily LUXOR V7 signal"""
    try:
        print("\n" + "="*80)
        print("🔄 [1/5] Starting /signal/daily endpoint...")
        print("="*80)
        sys.stdout.flush()
        
        # Step 1: Fetch data
        print("\n[2/5] 📥 Fetching BTCUSDT data...")
        sys.stdout.flush()
        df = luxor.fetch_real_binance_data()
        
        if df is None:
            print("❌ [2/5] Data fetch returned None")
            sys.stdout.flush()
            raise Exception("fetch_real_binance_data returned None")
        
        if len(df) < 100:
            print(f"❌ [2/5] Insufficient data: {len(df)} candles")
            sys.stdout.flush()
            raise Exception(f"Insufficient data: {len(df)} candles")
        
        print(f"✅ [2/5] Data fetched: {len(df)} candles")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")
        sys.stdout.flush()
        
        # Step 2: Calculate indicators
        print("\n[3/5] 📊 Calculating indicators...")
        sys.stdout.flush()
        
        try:
            df = luxor.calculate_all_indicators(df)
            print("✅ [3/5] All indicators calculated successfully")
        except Exception as e:
            print(f"❌ [3/5] Error calculating indicators: {e}")
            print(traceback.format_exc())
            sys.stdout.flush()
            raise
        
        sys.stdout.flush()
        
        # Step 3: Check dataframe
        print("\n[4/5] 🔍 Checking dataframe columns...")
        print(f"   Total columns: {len(df.columns)}")
        print(f"   Sample columns: {list(df.columns[:10])}")
        sys.stdout.flush()
        
        # Step 4: Get last row
        print("\n[5/5] 📈 Preparing output...")
        sys.stdout.flush()
        
        try:
            row = df.iloc[-1]
            print(f"✅ Got last row")
            print(f"   Date: {row['date']}")
            print(f"   Close: {row['close']}")
            sys.stdout.flush()
        except Exception as e:
            print(f"❌ Error getting last row: {e}")
            print(traceback.format_exc())
            sys.stdout.flush()
            raise
        
        # Evaluate signals
        try:
            print("   Evaluating signals...")
            signal = luxor.evaluate_signals(df, len(df) - 1)
            print(f"✅ Signal evaluated: {signal.get('action', 'N/A')}")
            sys.stdout.flush()
        except Exception as e:
            print(f"❌ Error evaluating signals: {e}")
            print(traceback.format_exc())
            signal = {
                'action': 'WAIT',
                'signal_count': 0,
                'signals': [],
                'rsi': 50.0,
                'atr': 100.0
            }
            sys.stdout.flush()
        
        # Build response
        try:
            print("   Building response...")
            entry = float(row['close'])
            atr_val = float(row.get('atr', 100))
            sl = entry - (atr_val * 0.5)
            tp = entry + (atr_val * 4.5)
            
            response_data = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'signal_type': signal.get('action', 'WAIT'),
                'entry_price': entry,
                'stop_loss': sl,
                'take_profit': tp,
                'confidence': min(100, signal.get('signal_count', 0) * 15),
                'signal_count': signal.get('signal_count', 0),
                'signals': signal.get('signals', []),
                'rsi': float(signal.get('rsi', 50)),
                'atr': float(atr_val),
                'last_date': str(row['date'].date()),
                'candles_analyzed': len(df)
            }
            
            print(f"✅ Response built successfully")
            print(f"   Signal: {response_data['signal_type']}")
            print(f"   Entry: {response_data['entry_price']}")
            sys.stdout.flush()
            
            return response_data
        
        except Exception as e:
            print(f"❌ Error building response: {e}")
            print(traceback.format_exc())
            sys.stdout.flush()
            raise
    
    except Exception as e:
        error_msg = f"Error in /signal/daily: {str(e)}"
        print(f"\n❌ {error_msg}")
        print(traceback.format_exc())
        sys.stdout.flush()
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/health")
async def health():
    """Health check"""
    return {
        'status': 'healthy',
        'service': SERVICE_NAME,
        'version': SERVICE_VERSION,
        'timestamp': datetime.now().isoformat()
    }

@app.on_event("startup")
async def startup_event():
    print("\n" + "="*80)
    print(f"🚀 {SERVICE_NAME} - PYTHON RUNTIME STARTED")
    print("="*80)
    print(f"📊 System: LUXOR V7 PRANA Egypt-India Unified")
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
