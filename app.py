# ============================================================
# LUXOR V7 PRANA RUNTIME - FastAPI v5.0.6
# With detailed error logging
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import logging
import traceback
import sys

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="LUXOR V7 PRANA",
    description="Multi-Timeframe Trading Signal System",
    version="5.0.6"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# IMPORT SYSTEM
# ============================================================

luxor_system = None
import_error = None

try:
    logger.info("Importing LuxorV7PranaSystem...")
    from luxor_v7_prana import LuxorV7PranaSystem
    luxor_system = LuxorV7PranaSystem()
    logger.info("LuxorV7PranaSystem initialized successfully")
except Exception as e:
    import_error = str(e)
    logger.error(f"Failed to import LuxorV7PranaSystem: {e}")
    logger.error(traceback.format_exc())

# ============================================================
# ROUTES
# ============================================================

@app.get("/")
async def root():
    return {
        "message": "LUXOR V7 PRANA - MTF Signal System",
        "version": "5.0.6",
        "system_loaded": luxor_system is not None,
        "import_error": import_error
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy" if luxor_system else "degraded",
        "version": "5.0.6",
        "timestamp": datetime.now().isoformat(),
        "system_loaded": luxor_system is not None,
        "import_error": import_error
    }


@app.get("/debug")
async def debug():
    """Debug endpoint."""
    debug_info = {
        "system_loaded": luxor_system is not None,
        "import_error": import_error,
        "python_version": sys.version,
    }
    
    imports = {}
    for module in ['pandas', 'numpy', 'ccxt', 'fastapi']:
        try:
            mod = __import__(module)
            imports[module] = getattr(mod, '__version__', 'OK')
        except ImportError as e:
            imports[module] = f"FAILED: {e}"
    
    debug_info["imports"] = imports
    return debug_info


@app.get("/signal/daily")
async def get_daily_signal(symbol: str = "BTCUSDT"):
    """Main MTF signal endpoint."""
    logger.info(f"[SIGNAL] Request for {symbol}")
    
    if luxor_system is None:
        logger.error(f"System not loaded. Import error: {import_error}")
        raise HTTPException(status_code=500, detail=f"System not initialized: {import_error}")
    
    try:
        logger.info("[SIGNAL] Calling generate_mtf_signal...")
        signal = luxor_system.generate_mtf_signal(symbol)
        
        if signal.get("status") == "error":
            error_detail = signal.get("detail", "Unknown error")
            logger.error(f"[SIGNAL] Error: {error_detail}")
            raise HTTPException(status_code=500, detail=error_detail)
        
        logger.info(f"[SIGNAL] Success: {signal.get('consensus', {}).get('primary_direction', 'N/A')}")
        return signal
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        logger.error(f"[SIGNAL] Exception: {error_msg}")
        logger.error(f"[SIGNAL] Traceback:\n{tb}")
        raise HTTPException(status_code=500, detail=f"Error: {error_msg}")


@app.get("/signal/quick")
async def get_quick_signal(symbol: str = "BTCUSDT"):
    """Quick signal endpoint."""
    logger.info(f"[QUICK] Request for {symbol}")
    
    if luxor_system is None:
        raise HTTPException(status_code=500, detail=f"System not initialized: {import_error}")
    
    try:
        df = luxor_system.fetch_real_binance_data(symbol=symbol)
        
        if df is None or len(df) == 0:
            raise HTTPException(status_code=500, detail="Failed to fetch data")
        
        logger.info(f"[QUICK] Got {len(df)} candles")
        
        current_price = float(df['close'].iloc[-1])
        rsi_series = luxor_system.calculate_rsi(df['close'])
        rsi = float(rsi_series.iloc[-1]) if not rsi_series.isna().iloc[-1] else 50.0
        sma_200 = float(df['close'].rolling(200).mean().iloc[-1]) if len(df) >= 200 else current_price
        
        if current_price > sma_200 and rsi > 50:
            direction = "BULLISH"
        elif current_price < sma_200 and rsi < 50:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"
        
        return {
            "status": "success",
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "direction": direction,
            "rsi": round(rsi, 2),
            "sma_200": round(sma_200, 2),
            "candles": len(df),
            "version": "5.0.6"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[QUICK] Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
