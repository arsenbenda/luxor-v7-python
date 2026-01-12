# ============================================================
# LUXOR V7 PRANA RUNTIME - FastAPI v5.0.8
# COMPLETE VERSION - All endpoints and error handling
# ============================================================

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

VERSION = "5.0.8"

app = FastAPI(
    title="LUXOR V7 PRANA",
    description="Multi-Timeframe Gann Trading Signal System",
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
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
    logger.info(f"LuxorV7PranaSystem v{VERSION} initialized successfully")
except Exception as e:
    import_error = str(e)
    logger.error(f"Failed to import LuxorV7PranaSystem: {e}")
    logger.error(traceback.format_exc())

# ============================================================
# ERROR HANDLER
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": str(exc),
            "version": VERSION,
            "timestamp": datetime.now().isoformat()
        }
    )

# ============================================================
# ROUTES
# ============================================================

@app.get("/")
async def root():
    """Root endpoint - system info."""
    return {
        "name": "LUXOR V7 PRANA",
        "description": "Multi-Timeframe Gann Trading Signal System",
        "version": VERSION,
        "system_loaded": luxor_system is not None,
        "import_error": import_error,
        "endpoints": {
            "health": "/health",
            "daily_signal": "/signal/daily",
            "quick_signal": "/signal/quick",
            "debug": "/debug",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if luxor_system else "degraded",
        "version": VERSION,
        "timestamp": datetime.now().isoformat(),
        "system_loaded": luxor_system is not None,
        "import_error": import_error
    }


@app.get("/debug")
async def debug():
    """Debug endpoint - system diagnostics."""
    debug_info = {
        "system_loaded": luxor_system is not None,
        "import_error": import_error,
        "python_version": sys.version,
        "version": VERSION,
        "timestamp": datetime.now().isoformat()
    }
    
    # Check imports
    imports = {}
    for module in ['pandas', 'numpy', 'ccxt', 'fastapi']:
        try:
            mod = __import__(module)
            imports[module] = {
                "status": "OK",
                "version": getattr(mod, '__version__', 'unknown')
            }
        except ImportError as e:
            imports[module] = {
                "status": "FAILED",
                "error": str(e)
            }
    
    debug_info["imports"] = imports
    
    # Cache status
    if luxor_system:
        cache = luxor_system.CACHE
        debug_info["cache"] = {
            "has_data": cache.get('df') is not None,
            "last_fetch": str(cache.get('last_fetch')) if cache.get('last_fetch') else None,
            "cache_duration": cache.get('cache_duration')
        }
    
    return debug_info


@app.get("/signal/daily")
async def get_daily_signal(
    symbol: str = Query(default="BTCUSDT", description="Trading pair symbol")
):
    """
    Main MTF signal endpoint.
    
    Returns complete multi-timeframe analysis including:
    - Regime detection
    - 4 timeframe analysis (1D, 3D, 1W, 1M)
    - MTF consensus
    - Capitulation detection
    - Time forecast
    - Trade setups with R:R
    - Support/Resistance levels
    - Gann interpretation
    """
    logger.info(f"[SIGNAL] Request for {symbol}")
    
    if luxor_system is None:
        logger.error(f"System not loaded. Import error: {import_error}")
        raise HTTPException(
            status_code=500,
            detail=f"System not initialized: {import_error}"
        )
    
    try:
        logger.info("[SIGNAL] Calling generate_mtf_signal...")
        signal = luxor_system.generate_mtf_signal(symbol)
        
        if signal.get("status") == "error":
            error_detail = signal.get("detail", "Unknown error")
            logger.error(f"[SIGNAL] Error: {error_detail}")
            raise HTTPException(status_code=500, detail=error_detail)
        
        direction = signal.get('consensus', {}).get('primary_direction', 'N/A')
        confidence = signal.get('consensus', {}).get('confidence_level', 'N/A')
        logger.info(f"[SIGNAL] Success: {direction} ({confidence})")
        
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
async def get_quick_signal(
    symbol: str = Query(default="BTCUSDT", description="Trading pair symbol")
):
    """
    Quick signal endpoint - simplified response.
    
    Returns basic directional bias without full MTF analysis.
    Useful for quick checks or high-frequency polling.
    """
    logger.info(f"[QUICK] Request for {symbol}")
    
    if luxor_system is None:
        raise HTTPException(
            status_code=500,
            detail=f"System not initialized: {import_error}"
        )
    
    try:
        df = luxor_system.fetch_real_binance_data(symbol=symbol)
        
        if df is None or len(df) == 0:
            raise HTTPException(status_code=500, detail="Failed to fetch data")
        
        logger.info(f"[QUICK] Got {len(df)} candles")
        
        current_price = float(df['close'].iloc[-1])
        
        # Quick calculations
        rsi_series = luxor_system.calculate_rsi(df['close'])
        rsi = float(rsi_series.iloc[-1]) if not rsi_series.isna().iloc[-1] else 50.0
        
        sma_200 = float(df['close'].rolling(200).mean().iloc[-1]) if len(df) >= 200 else current_price
        sma_50 = float(df['close'].rolling(50).mean().iloc[-1]) if len(df) >= 50 else current_price
        
        # Quick direction
        bullish_count = 0
        bearish_count = 0
        
        if current_price > sma_200:
            bullish_count += 1
        else:
            bearish_count += 1
        
        if current_price > sma_50:
            bullish_count += 1
        else:
            bearish_count += 1
        
        if rsi > 50:
            bullish_count += 1
        else:
            bearish_count += 1
        
        if bullish_count > bearish_count:
            direction = "BULLISH"
        elif bearish_count > bullish_count:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"
        
        return {
            "status": "success",
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "direction": direction,
            "rsi": round(rsi, 2),
            "sma_50": round(sma_50, 2),
            "sma_200": round(sma_200, 2),
            "price_vs_sma_200": "ABOVE" if current_price > sma_200 else "BELOW",
            "candles": len(df),
            "timestamp": datetime.now().isoformat(),
            "version": VERSION
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[QUICK] Exception: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    logger.info(f"Starting LUXOR V7 PRANA v{VERSION}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
