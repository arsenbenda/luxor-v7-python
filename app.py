# ============================================================
# LUXOR V7 PRANA RUNTIME - FastAPI v5.1.2
# ============================================================

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import logging
import traceback
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

VERSION = "5.1.2"

app = FastAPI(
    title="LUXOR V7 PRANA",
    description="MTF Gann Trading System v5.1.2 - R:R Fix",
    version=VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

luxor_system = None
import_error = None

try:
    from luxor_v7_prana import LuxorV7PranaSystem, TIMEFRAME_CONFIGS
    luxor_system = LuxorV7PranaSystem()
    logger.info(f"[INIT] LUXOR V7 PRANA v{VERSION} ready")
    logger.info(f"[INIT] min_bars: 1M={TIMEFRAME_CONFIGS['1M'].min_bars}, 1W={TIMEFRAME_CONFIGS['1W'].min_bars}, 3D={TIMEFRAME_CONFIGS['3D'].min_bars}, 1D={TIMEFRAME_CONFIGS['1D'].min_bars}")
except Exception as e:
    import_error = str(e)
    logger.error(f"[INIT] Failed: {e}")
    logger.error(traceback.format_exc())


@app.exception_handler(Exception)
async def handler(req, exc):
    return JSONResponse(status_code=500, content={"status": "error", "detail": str(exc), "version": VERSION})


@app.get("/")
async def root():
    return {
        "name": "LUXOR V7 PRANA",
        "version": VERSION,
        "ready": luxor_system is not None,
        "fixes": ["R:R calculation", "min_bars reduced", "Stop validation"],
        "min_bars": {"1M": 12, "1W": 40, "3D": 60, "1D": 150}
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy" if luxor_system else "degraded",
        "version": VERSION,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/config")
async def config():
    if not luxor_system:
        raise HTTPException(500, f"Not initialized: {import_error}")
    
    from luxor_v7_prana import TIMEFRAME_CONFIGS, MIN_RR_RATIO
    
    return {
        "version": VERSION,
        "timeframes": {
            tf: {"min_bars": cfg.min_bars, "gann_lookback": cfg.gann_lookback}
            for tf, cfg in TIMEFRAME_CONFIGS.items()
        },
        "min_rr_ratio": MIN_RR_RATIO
    }


@app.get("/signal/daily")
async def daily(symbol: str = Query(default="BTCUSDT")):
    if not luxor_system:
        raise HTTPException(500, f"Not initialized: {import_error}")
    try:
        signal = luxor_system.generate_mtf_signal(symbol=symbol)
        if signal.get("status") == "error":
            raise HTTPException(500, signal.get("detail"))
        return signal
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(500, str(e))


@app.get("/signal/quick")
async def quick(symbol: str = Query(default="BTCUSDT")):
    if not luxor_system:
        raise HTTPException(500, f"Not initialized: {import_error}")
    try:
        df = luxor_system.fetch_real_binance_data(symbol=symbol)
        p = float(df['close'].iloc[-1])
        rsi = float(luxor_system.calculate_rsi(df['close']).iloc[-1])
        sma = float(df['close'].rolling(200).mean().iloc[-1]) if len(df) >= 200 else p
        d = "BULLISH" if p > sma and rsi > 50 else "BEARISH" if p < sma and rsi < 50 else "NEUTRAL"
        return {"status": "success", "symbol": symbol, "price": round(p, 2), "direction": d, "rsi": round(rsi, 2), "sma_200": round(sma, 2), "version": VERSION}
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
