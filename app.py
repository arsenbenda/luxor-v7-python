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
        df = luxor.fetch_real_binance_data()
        
        if df is None or len(df) < 100:
            raise HTTPException(status_code=400, detail="Insufficient data")
        
        df = luxor.calculate_all_indicators(df)
        signal = luxor.evaluate_signals(df, len(df) - 1)
        
        row = df.iloc[-1]
        entry = row['close']
        sl = entry - (row['atr'] * 0.5)
        tp = entry + (row['atr'] * 4.5)
        
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'signal_type': signal['action'],
            'entry_price': float(entry),
            'stop_loss': float(sl),
            'take_profit': float(tp),
            'confidence': min(100, signal['signal_count'] * 12),
            'signal_count': signal['signal_count'],
            'signals': signal['signals'],
            'rsi': float(row['rsi']),
            'atr': float(row['atr']),
            'cycle_power': float(row['cycle_power']),
            'nakshatra': int(row['nakshatra']),
            'near_pivot': int(row['near_pivot']),
            'algol_safe': int(row['algol_safe']),
            'last_date': str(row['date'].date()),
            'candles_analyzed': len(df)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
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
