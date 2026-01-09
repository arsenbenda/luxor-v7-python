import os
from datetime import datetime

ENV = os.getenv("ENVIRONMENT", "development")
DEBUG = ENV == "development"

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/luxor")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "317166443")

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

INITIAL_CAPITAL = 10000
POSITION_SIZE_PCT = 2
MIN_CONFLUENCE_SCORE = 7
MIN_CONFIDENCE = 65

SERVICE_NAME = "LUXOR V7 PRANA"
SERVICE_VERSION = "1.0.0"
START_TIME = datetime.now().isoformat()

API_HOST = "0.0.0.0"
API_PORT = int(os.getenv("PORT", "8000"))
