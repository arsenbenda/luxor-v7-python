# ============================================================
# LUXOR V7 PRANA RUNTIME - FastAPI v5.1.1
# Complete API with Historical DataFrame Support
# ============================================================

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import logging
import traceback
import sys
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

VERSION = "5.1.1"

app = FastAPI(
    title="LUXOR V7 PRANA",
    description="Multi-Timeframe Gann Trading System v5.1.1 - Increased min_bars for Reliable Backtests",
    version=VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global system instance
luxor_system = None
import_error = None

try:
    from luxor_v7_prana import LuxorV7PranaSystem, TIMEFRAME_CONFIGS
    luxor_system = LuxorV7PranaSystem()
    logger.info(f"[INIT] LUXOR V7 PRANA v{VERSION} initialized successfully")
    logger.info(f"[INIT] min_bars config: 1M={TIMEFRAME_CONFIGS['1M'].min_bars}, 1W={TIMEFRAME_CONFIGS['1W'].min_bars}, 3D={TIMEFRAME_CONFIGS['3D'].min_bars}, 1D={TIMEFRAME_CONFIGS['1D'].min_bars}")
except Exception as e:
    import_error = str(e)
    logger.error(f"[INIT] Import failed: {e}")
    logger.error(traceback.format_exc())


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"[ERROR] Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": str(exc),
            "version": VERSION,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.get("/")
async def root():
    """Root endpoint with system info."""
    return {
        "name": "LUXOR V7 PRANA",
        "version": VERSION,
        "ready": luxor_system is not None,
        "error": import_error,
        "features": [
            "Multi-Timeframe Analysis (1M/1W/3D/1D)",
            "Gann 50% Rule with Override",
            "Capitulation Detection",
            "Adaptive MTF Weights",
            "Historical DataFrame Support",
            "Backtest Module Integration",
            "Increased min_bars for Reliability"
        ],
        "min_bars": {
            "1M": 24,
            "1W": 52,
            "3D": 90,
            "1D": 200
        } if luxor_system else None
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    status = "healthy" if luxor_system else "degraded"
    
    return {
        "status": status,
        "version": VERSION,
        "system_ready": luxor_system is not None,
        "import_error": import_error,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/config")
async def get_config():
    """Get current system configuration."""
    if not luxor_system:
        raise HTTPException(status_code=500, detail=f"System not initialized: {import_error}")
    
    from luxor_v7_prana import TIMEFRAME_CONFIGS, RSI_OVERSOLD, RSI_OVERBOUGHT, ADX_STRONG_TREND, MIN_RR_RATIO, GANN_CYCLES
    
    return {
        "version": VERSION,
        "timeframes": {
            tf: {
                "name": cfg.name,
                "base_weight": cfg.base_weight,
                "gann_lookback": cfg.gann_lookback,
                "min_bars": cfg.min_bars,
                "trend_weight": cfg.trend_weight,
                "range_weight": cfg.range_weight
            }
            for tf, cfg in TIMEFRAME_CONFIGS.items()
        },
        "thresholds": {
            "rsi_oversold": RSI_OVERSOLD,
            "rsi_overbought": RSI_OVERBOUGHT,
            "adx_strong_trend": ADX_STRONG_TREND,
            "min_rr_ratio": MIN_RR_RATIO
        },
        "gann_cycles": GANN_CYCLES
    }


@app.get("/signal/daily")
async def get_daily_signal(symbol: str = Query(default="BTCUSDT")):
    """
    Get complete MTF signal for live trading.
    
    Args:
        symbol: Trading symbol (default: BTCUSDT)
    
    Returns:
        Complete signal with all MTF analysis
    """
    if not luxor_system:
        raise HTTPException(status_code=500, detail=f"System not initialized: {import_error}")
    
    try:
        logger.info(f"[SIGNAL] Generating MTF signal for {symbol}")
        
        signal = luxor_system.generate_mtf_signal(symbol=symbol)
        
        if signal.get("status") == "error":
            logger.error(f"[SIGNAL] Generation failed: {signal.get('detail')}")
            raise HTTPException(status_code=500, detail=signal.get("detail", "Signal generation failed"))
        
        logger.info(f"[SIGNAL] Success: {symbol} @ ${signal.get('current_price', 0):,.2f}, Bias: {signal.get('primary_bias', {}).get('primary_bias', 'N/A')}")
        
        return signal
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SIGNAL] Error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signal/quick")
async def get_quick_signal(symbol: str = Query(default="BTCUSDT")):
    """
    Quick signal check without full MTF analysis.
    Faster but less comprehensive.
    
    Args:
        symbol: Trading symbol (default: BTCUSDT)
    
    Returns:
        Quick directional signal
    """
    if not luxor_system:
        raise HTTPException(status_code=500, detail=f"System not initialized: {import_error}")
    
    try:
        logger.info(f"[QUICK] Generating quick signal for {symbol}")
        
        # Fetch data
        df = luxor_system.fetch_real_binance_data(symbol=symbol)
        
        if df is None or len(df) == 0:
            raise HTTPException(status_code=500, detail="No data available")
        
        price = float(df['close'].iloc[-1])
        rsi = float(luxor_system.calculate_rsi(df['close']).iloc[-1])
        
        # SMA calculations
        sma_50 = float(df['close'].rolling(50).mean().iloc[-1]) if len(df) >= 50 else price
        sma_200 = float(df['close'].rolling(200).mean().iloc[-1]) if len(df) >= 200 else price
        
        # ADX
        adx = float(luxor_system.calculate_adx(df).iloc[-1])
        
        # Gann 50%
        gann = luxor_system.calculate_gann_levels(df, min(252, len(df)))
        gann_50 = gann.get("gann_50", price)
        
        # Quick direction
        bull_signals = 0
        bear_signals = 0
        
        if price > sma_200:
            bull_signals += 1
        else:
            bear_signals += 1
        
        if price > gann_50:
            bull_signals += 1.5  # Gann has higher weight
        else:
            bear_signals += 1.5
        
        if rsi > 50:
            bull_signals += 0.5
        else:
            bear_signals += 0.5
        
        if bull_signals > bear_signals:
            direction = "BULLISH"
        elif bear_signals > bull_signals:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"
        
        # Strength
        if adx > 50:
            strength = "VERY_STRONG"
        elif adx > 25:
            strength = "STRONG"
        else:
            strength = "WEAK"
        
        return {
            "status": "success",
            "version": VERSION,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "current_price": round(price, 2),
            "direction": direction,
            "strength": strength,
            "indicators": {
                "rsi": round(rsi, 2),
                "adx": round(adx, 2),
                "sma_50": round(sma_50, 2),
                "sma_200": round(sma_200, 2),
                "gann_50": round(gann_50, 2),
                "price_vs_gann_50": "ABOVE" if price > gann_50 else "BELOW",
                "price_vs_sma_200": "ABOVE" if price > sma_200 else "BELOW"
            },
            "candles_used": len(df)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[QUICK] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signal/historical")
async def get_historical_signal(
    symbol: str = Query(default="BTCUSDT"),
    days_back: int = Query(default=0, ge=0, le=365)
):
    """
    Get signal for a historical date (for backtesting verification).
    
    Args:
        symbol: Trading symbol
        days_back: Number of days back from today (0 = today)
    
    Returns:
        Signal as of that date
    """
    if not luxor_system:
        raise HTTPException(status_code=500, detail=f"System not initialized: {import_error}")
    
    try:
        # Fetch full data
        df = luxor_system.fetch_real_binance_data(symbol=symbol, use_cache=False)
        
        if df is None or len(df) == 0:
            raise HTTPException(status_code=500, detail="No data available")
        
        # Slice to historical point
        if days_back > 0:
            if days_back >= len(df):
                raise HTTPException(status_code=400, detail=f"days_back ({days_back}) exceeds available data ({len(df)} days)")
            
            df_historical = df.iloc[:-days_back].copy()
            reference_date = datetime.now() - pd.Timedelta(days=days_back)
        else:
            df_historical = df.copy()
            reference_date = datetime.now()
        
        logger.info(f"[HISTORICAL] Generating signal for {symbol} @ {reference_date.date()} ({len(df_historical)} candles)")
        
        # Generate signal with historical data
        signal = luxor_system.generate_mtf_signal(
            symbol=symbol,
            df_historical=df_historical,
            reference_date=reference_date
        )
        
        if signal.get("status") == "error":
            raise HTTPException(status_code=500, detail=signal.get("detail"))
        
        # Add historical context
        signal["historical_context"] = {
            "days_back": days_back,
            "reference_date": reference_date.strftime("%Y-%m-%d"),
            "candles_used": len(df_historical)
        }
        
        return signal
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[HISTORICAL] Error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/ohlcv")
async def get_ohlcv_data(
    symbol: str = Query(default="BTCUSDT"),
    limit: int = Query(default=100, ge=10, le=750)
):
    """
    Get raw OHLCV data.
    
    Args:
        symbol: Trading symbol
        limit: Number of candles (max 750)
    
    Returns:
        OHLCV data as list
    """
    if not luxor_system:
        raise HTTPException(status_code=500, detail=f"System not initialized: {import_error}")
    
    try:
        ccxt_sym = symbol[:-4] + '/USDT' if symbol.endswith('USDT') and '/' not in symbol else symbol
        df = luxor_system.fetch_ohlcv_ccxt(ccxt_sym, "1d", limit)
        
        # Convert to list of dicts
        data = []
        for _, row in df.iterrows():
            data.append({
                "timestamp": row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                "open": round(float(row['open']), 2),
                "high": round(float(row['high']), 2),
                "low": round(float(row['low']), 2),
                "close": round(float(row['close']), 2),
                "volume": round(float(row['volume']), 2)
            })
        
        return {
            "status": "success",
            "symbol": symbol,
            "interval": "1d",
            "count": len(data),
            "data": data
        }
        
    except Exception as e:
        logger.error(f"[DATA] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis/timeframe/{tf}")
async def get_timeframe_analysis(
    tf: str,
    symbol: str = Query(default="BTCUSDT")
):
    """
    Get analysis for a specific timeframe.
    
    Args:
        tf: Timeframe (1D, 3D, 1W, 1M)
        symbol: Trading symbol
    
    Returns:
        Detailed timeframe analysis
    """
    if not luxor_system:
        raise HTTPException(status_code=500, detail=f"System not initialized: {import_error}")
    
    tf = tf.upper()
    if tf not in ["1D", "3D", "1W", "1M"]:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe: {tf}. Use 1D, 3D, 1W, or 1M")
    
    try:
        # Fetch and resample data
        df_1d = luxor_system.fetch_real_binance_data(symbol=symbol)
        
        if tf == "1D":
            df = df_1d
        elif tf == "3D":
            df = luxor_system.resample_ohlcv(df_1d, "3D")
        elif tf == "1W":
            df = luxor_system.resample_ohlcv(df_1d, "1W")
        elif tf == "1M":
            df = luxor_system.resample_ohlcv(df_1d, "1M")
        
        # Analyze
        analysis = luxor_system.analyze_timeframe(df, tf)
        
        return {
            "status": "success",
            "symbol": symbol,
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"[TF] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug")
async def debug_info():
    """Debug endpoint with system info."""
    import sys
    
    modules = {}
    for mod in ['pandas', 'numpy', 'fastapi', 'ccxt']:
        try:
            m = __import__(mod)
            modules[mod] = getattr(m, '__version__', 'installed')
        except ImportError:
            modules[mod] = 'not installed'
    
    return {
        "version": VERSION,
        "python_version": sys.version,
        "modules": modules,
        "system_initialized": luxor_system is not None,
        "import_error": import_error,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
