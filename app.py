# ============================================================
# LUXOR V7 PRANA RUNTIME - FastAPI v5.0.6
# Imports core logic from luxor_v7_prana.py
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import logging

# Import the main system
from luxor_v7_prana import LuxorV7PranaSystem

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="LUXOR V7 PRANA",
    description="Multi-Timeframe Trading Signal System with Gann & Ichimoku",
    version="5.0.6"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the system
luxor_system = LuxorV7PranaSystem()

# ============================================================
# ROUTES
# ============================================================

@app.get("/")
async def root():
    return {"message": "LUXOR V7 PRANA - MTF Signal System", "version": "5.0.6"}


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "5.0.6", "timestamp": datetime.now().isoformat()}


@app.get("/signal/daily")
async def get_daily_signal(symbol: str = "BTCUSDT"):
    """Main MTF signal endpoint."""
    try:
        logger.info(f"Generating signal for {symbol}")
        signal = luxor_system.generate_mtf_signal(symbol)
        
        if signal.get("status") == "error":
            raise HTTPException(status_code=500, detail=signal.get("detail"))
        
        return signal
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signal/quick")
async def get_quick_signal(symbol: str = "BTCUSDT"):
    """Quick signal endpoint."""
    try:
        df = luxor_system.fetch_real_binance_data(symbol=symbol)
        if df is None:
            raise HTTPException(status_code=500, detail="Failed to fetch data")
        
        current_price = float(df['close'].iloc[-1])
        rsi = float(luxor_system.calculate_rsi(df['close']).iloc[-1])
        sma_200 = df['close'].rolling(200).mean().iloc[-1] if len(df) >= 200 else current_price
        
        direction = "BULLISH" if current_price > sma_200 and rsi > 50 else "BEARISH" if current_price < sma_200 and rsi < 50 else "NEUTRAL"
        
        return {
            "status": "success",
            "symbol": symbol,
            "current_price": current_price,
            "direction": direction,
            "rsi": round(rsi, 2),
            "sma_200": round(sma_200, 2),
            "version": "5.0.6"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
