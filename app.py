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
        print("\n🔄 Starting /signal/daily endpoint...")
        
        # Step 1: Fetch data
        print("📥 Fetching data...")
        df = luxor.fetch_real_binance_data()
        
        if df is None or len(df) < 100:
            print("❌ Data fetch failed or insufficient")
            raise HTTPException(status_code=400, detail="Insufficient data")
        
        print(f"✅ Data fetched: {len(df)} candles")
        
        # Step 2: Calculate indicators
        print("📊 Calculating indicators...")
        df = luxor.calculate_all_indicators(df)
        print("✅ Indicators calculated")
        
        # Step 3: Evaluate signals
        print("🎯 Evaluating signals...")
        signal = luxor.evaluate_signals(df, len(df) - 1)
        print(f"✅ Signal evaluated: {signal['action']}")
        
        # Step 4: Get last row
        print("📈 Preparing output...")
        row = df.iloc[-1]
        
        # Entry/Exit
        entry = float(row['close'])
        atr_val = float(row.get('atr', 100))
        sl = entry - (atr_val * 0.5)
        tp = entry + (atr_val * 4.5)
        
        print(f"   Entry: {entry}")
        print(f"   SL: {sl}")
        print(f"   TP: {tp}")
        
        # Step 5: Build response
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
            'rsi': signal.get('rsi', 50),
            'atr': atr_val,
            'last_date': str(row['date'].date()),
            'candles_analyzed': len(df)
        }
        
        print(f"✅ Response ready: {response_data['signal_type']}")
        return response_data
    
    except HTTPException as he:
        print(f"❌ HTTP Error: {he.detail}")
        raise he
    
    except Exception as e:
        print(f"❌ Error in /signal/daily: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/backtest/optimize")
async def backtest_optimize():
    """Run backtest optimizer"""
    try:
        df = luxor.fetch_real_binance_data()
        
        if df is None or len(df) < 100:
            raise HTTPException(status_code=400, detail="Insufficient data")
        
        final_capital = luxor.run_backtest(df)
        report = luxor.print_report(final_capital)
        
        return {
            'status': 'success',
            'backtest_report': report
        }
    
    except Exception as e:
        print(f"❌ Error in /backtest/optimize: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
    print(f"   • GET /backtest/optimize → Run optimizer")
    print(f"   • GET /health → Health check")
    print(f"⏰ Timestamp: {datetime.now().isoformat()}")
    print("="*80 + "\n")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info" if DEBUG else "warning"
    )
