"""
Configuration file for AI Trading Bot
Store all API keys, model settings, and risk parameters here.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Alpaca API
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', 'YOUR_ALPACA_KEY_HERE')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', 'YOUR_ALPACA_SECRET_HERE')
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'  # Paper trading endpoint


# Ollama config
OLLAMA_MODEL = "mixtral"  # Model running on RTX 5090
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama endpoint
OLLAMA_TIMEOUT = 60  # Timeout in seconds for AI inference

# Trading Parameters
TICKER = "SPY"  # Change this to trade different symbols (e.g., "QQQ", "AAPL")
POSITION_SIZE = 1000  # Dollar amount to invest per trade (in USD)
MAX_POSITIONS = 1  # Maximum number of concurrent positions

# Portfolio Universe
# Curated list of liquid stocks across S&P 500 sectors for portfolio optimization
TICKER_UNIVERSE = [
    # Technology
    "AAPL",  
    "MSFT",   
    "NVDA",   
    "GOOGL",  
    "AMD",    
    "META",   
    "TSLA",   

    # Finance
    "JPM",    
    "BAC",    
    "V",     
    "MA",     
    "GS",     

    # Healthcare
    "JNJ",    
    "PFE",    
    "UNH",    
    "LLY",    
    "ABBV",   

    # Consumer
    "AMZN",   
    "HD",     
    "PG",     
    "KO",     
    "WMT",    
    "MCD",    

    # Energy & Industrials
    "XOM",   
    "CVX",   
    "CAT",    
    "BA",     
    "UPS",    

    # ETFs (Optional - for diversification)
    "SPY",
    "QQQ",
]

# Default portfolio for quick testing
DEFAULT_PORTFOLIO_TICKERS = [
    "AAPL", "MSFT", "NVDA", "JPM", "JNJ",
    "AMZN", "XOM", "V", "PG", "HD"
]

# The "All-Weather" Quant Basket (16 Tickers)
DEFAULT_MULTI_ASSET_BASKET = [
    # US Equities (The Growth Engine)
    "SPY",       
    "QQQ",       
    "IWM",       
    # Sector Specifics 
    "XLE",       # Energy 
    "XLF",       # Financials 
    "XLV",       # Healthcare 
    # Fixed Income 
    "TLT",       # 20+ Year Treasuries
    "LQD",       # Corporate Bonds 
    # Commodities 
    "GLD",       # Gold 
    "USO",       # Oil Fund (
    "DBC",       # Broad Commodities 
    # Global 
    "EEM",       # Emerging Markets
    "EFA",       # Developed Markets
    # Crypto 
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
]


# Crypto Universe
# Curated list of top cryptocurrencies for portfolio optimization
# Note: All crypto tickers must end with -USD for Yahoo Finance compatibility
CRYPTO_UNIVERSE = [
    "BTC-USD",    # Bitcoin - Market Leader
    "ETH-USD",    # Ethereum - Smart Contracts
    "BNB-USD",    # Binance Coin
    "SOL-USD",    # Solana - High Performance
    "XRP-USD",    # Ripple - Cross-border Payments
    "ADA-USD",    # Cardano - Proof of Stake
    "DOGE-USD",   # Dogecoin - Meme Coin
    "MATIC-USD",  # Polygon - Layer 2 Scaling
    "DOT-USD",    # Polkadot - Interoperability
    "AVAX-USD",   # Avalanche - DeFi Platform
    "LINK-USD",   # Chainlink - Oracles
    "UNI-USD",    # Uniswap - DEX
]

# Default crypto portfolio for quick testing
DEFAULT_CRYPTO_PORTFOLIO = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD"
]


# Risk Management
STOP_LOSS_PCT = 0.01  # 1% stop loss
TAKE_PROFIT_PCT = 0.02  # 2% take profit
TRAILING_STOP_PCT = 0.015  # 1.5% trailing stop (optional)

# Indicator settings
SMA_PERIOD = 200  # Simple Moving Average lookback
RSI_PERIOD = 14  # RSI lookback period
RSI_OVERSOLD = 55  # RSI threshold for "oversold" condition
BB_LENGTH = 20
BB_STD = 2.0

# SmartTrend Strategy
ADX_THRESHOLD = 25  # ADX minimum for strong trend

# VolatilitySqueeze Strategy
KELTNER_MULT = 1.5  # Keltner Channel ATR multiplier

# MACD Scalp Strategy
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bidirectional Strategy
LONG_RSI_THRESHOLD = 30   # RSI < value = Long entry (oversold dip)
SHORT_RSI_THRESHOLD = 70  # RSI > value = Short entry (overbought rally)
LONG_STOP_LOSS = 2.0      # 2% stop loss for longs
SHORT_STOP_LOSS = 1.5     # 1.5% stop loss for shorts (tighter)
ADX_CHOP_THRESHOLD = 20   # ADX < value = choppy market (no entries)


# AI decision thresholds
MIN_AI_CONFIDENCE = 0.7  # Minimum confidence to execute trade (0.0 - 1.0)


# Data settings

TIMEFRAME = "1h"  # Candle timeframe: "1m", "5m", "15m", "1h", "1d"
LOOKBACK_BARS = 200  # Number of historical bars to fetch
BACKTEST_BARS = 500  # Number of bars to use in backtesting

# Execution settings
LIVE_LOOP_INTERVAL = 60  # Seconds between live trading checks
DRY_RUN = True  # Set to False to enable real order execution

# Logging
LOG_FILE = "logs/brain_activity.log"
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# Validation
def validate_config():
    """Validate that required configuration is set"""
    errors = []

    if ALPACA_API_KEY == 'YOUR_ALPACA_KEY_HERE':
        errors.append("ALPACA_API_KEY not set. Please update .env or config.py")

    if ALPACA_SECRET_KEY == 'YOUR_ALPACA_SECRET_HERE':
        errors.append("ALPACA_SECRET_KEY not set. Please update .env or config.py")

    if MIN_AI_CONFIDENCE < 0 or MIN_AI_CONFIDENCE > 1:
        errors.append("MIN_AI_CONFIDENCE must be between 0.0 and 1.0")

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    return True

if __name__ == "__main__":
    # Test configuration
    try:
        validate_config()
        print("Configuration validated successfully")
        print(f"Trading: {TICKER}")
        print(f"Model: {OLLAMA_MODEL}")
        print(f"Risk: {STOP_LOSS_PCT*100}% SL / {TAKE_PROFIT_PCT*100}% TP")
    except ValueError as e:
        print(f"Configuration error:\n{e}")
