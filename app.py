# LUXOR V7 PRANA - FastAPI v5.0.10

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import logging
import traceback
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

VERSION = "5.0.10"

app = FastAPI(title="LUXOR V7 PRANA", version=VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

luxor_system = None
import_error = None

try:
    from luxor_v7_prana import LuxorV7PranaSystem
    luxor_system = LuxorV7PranaSystem()
    logger.info(f"LUXOR v{VERSION} ready")
except Exception as e:
    import_error = str(e)
    logger.error(f"Import failed: {e}")

@app.exception_handler(Exception)
async def handler(req, exc):
    return JSONResponse(status_code=500, content={"status": "error", "detail": str(exc), "version": VERSION})

@app.get("/")
async def root():
    return {"name": "LUXOR V7 PRANA", "version": VERSION, "ready": luxor_system is not None}

@app.get("/health")
async def health():
    return {"status": "healthy" if luxor_system else "degraded", "version": VERSION}

@app.get("/signal/daily")
async def daily(symbol: str = Query(default="BTCUSDT")):
    if not luxor_system:
        raise HTTPException(500, f"Not initialized: {import_error}")
    try:
        sig = luxor_system.generate_mtf_signal(symbol)
        if sig.get("status") == "error":
            raise HTTPException(500, sig.get("detail"))
        return sig
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/signal/quick")
async def quick(symbol: str = Query(default="BTCUSDT")):
    if not luxor_system:
        raise HTTPException(500, f"Not initialized: {import_error}")
    try:
        df = luxor_system.fetch_real_binance_data(symbol=symbol)
        if df is None:
            raise HTTPException(500, "No data")
        p = float(df['close'].iloc[-1])
        rsi = float(luxor_system.calculate_rsi(df['close']).iloc[-1])
        sma = float(df['close'].rolling(200).mean().iloc[-1]) if len(df) >= 200 else p
        d = "BULLISH" if p > sma and rsi > 50 else "BEARISH" if p < sma and rsi < 50 else "NEUTRAL"
        return {"status": "success", "symbol": symbol, "price": round(p, 2), "direction": d, "rsi": round(rsi, 2), "sma_200": round(sma, 2)}
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
