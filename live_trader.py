"""
Live Trading Engine - The Execution Layer
Connects to Alpaca Paper Trading and executes trades based on AI + strategy signals.
"""

import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict
import alpaca_trade_api as tradeapi
import yfinance as yf

import config
from strategy import (
    HybridStrategy, BreakoutStrategy, MeanReversionStrategy,
    SmartTrendStrategy, VolatilitySqueezeStrategy, MacdScalpStrategy,
    MacroReversalStrategy, DmiReversalStrategy, BidirectionalStrategy
)
from ai_brain import AIBrain

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/live_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LiveTrader:
    """
    Live trading bot that monitors markets and executes trades.

    Flow:
    1. Every N seconds, fetch latest market data
    2. Calculate indicators
    3. Check hard filters
    4. If passed, ask AI for decision
    5. If AI says BUY with high confidence, place bracket order
    """

    def __init__(self, dry_run: bool = None, strategy_class=None, strategy_params: dict = None):
        """
        Initialize the live trading engine.

        Args:
            dry_run: If True, log trades but don't execute. If None, use config.DRY_RUN
            strategy_class: Strategy class to use (default: HybridStrategy)
            strategy_params: Dict of strategy parameter overrides (e.g., from optimization)
        """
        self.dry_run = dry_run if dry_run is not None else config.DRY_RUN

        # Initialize components
        strategy_class = strategy_class or HybridStrategy
        strategy_params = strategy_params or {}
        self.strategy = strategy_class(**strategy_params)
        self.strategy_name = strategy_class.__name__
        self.ai_brain = AIBrain()

        # Initialize Alpaca API
        try:
            self.api = tradeapi.REST(
                key_id=config.ALPACA_API_KEY,
                secret_key=config.ALPACA_SECRET_KEY,
                base_url=config.ALPACA_BASE_URL,
                api_version='v2'
            )
            # Test connection
            account = self.api.get_account()
            logger.info(f"✓ Connected to Alpaca Paper Trading")
            logger.info(f"  Account: {account.account_number}")
            logger.info(f"  Buying Power: ${float(account.buying_power):,.2f}")
            logger.info(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")

        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            logger.error("Please check your API keys in config.py or .env")
            raise

        self.ticker = config.TICKER
        self.loop_interval = config.LIVE_LOOP_INTERVAL
        self.last_check_time = None
        self.trade_count = 0

        logger.info("="*60)
        logger.info("LIVE TRADER INITIALIZED")
        logger.info("="*60)
        logger.info(f"  Strategy:     {self.strategy_name}")
        logger.info(f"  Mode:         {'DRY RUN (No real orders)' if self.dry_run else 'LIVE (Real orders!)'}")
        logger.info(f"  Ticker:       {self.ticker}")
        logger.info(f"  Check Every:  {self.loop_interval}s")
        logger.info(f"  Position Size: ${config.POSITION_SIZE}")
        logger.info(f"  Stop Loss:    {config.STOP_LOSS_PCT*100:.1f}%")
        logger.info(f"  Take Profit:  {config.TAKE_PROFIT_PCT*100:.1f}%")
        if strategy_params:
            logger.info(f"  Params:       {strategy_params}")
        logger.info("="*60)

    def get_market_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch recent market data for analysis.

        Returns:
            DataFrame with OHLCV data, or None on error
        """
        try:
            logger.debug(f"Fetching market data for {self.ticker}...")

            # Use yfinance for historical data (more reliable than Alpaca for this)
            # Get enough data to calculate indicators (need 200+ bars for SMA 200)
            df = yf.download(
                self.ticker,
                period="3mo",  # 3 months of data
                interval=config.TIMEFRAME,
                progress=False
            )

            if isinstance(df.columns, pd.MultiIndex):
                # Try to find the level with 'Close'
                if 'Close' in df.columns.get_level_values(0):
                    df.columns = df.columns.get_level_values(0)
                else:
                    df.columns = df.columns.get_level_values(1)

            if df.empty:
                logger.warning(f"No data returned for {self.ticker}")
                return None

            logger.debug(f"Fetched {len(df)} bars, latest: {df.index[-1]}")
            return df

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None

    def get_current_position(self) -> Optional[Dict]:
        """
        Check if we have an open position in the ticker.

        Returns:
            Dict with position info, or None if no position
        """
        try:
            positions = self.api.list_positions()

            for position in positions:
                if position.symbol == self.ticker:
                    return {
                        'symbol': position.symbol,
                        'qty': float(position.qty),
                        'avg_entry_price': float(position.avg_entry_price),
                        'current_price': float(position.current_price),
                        'unrealized_pl': float(position.unrealized_pl),
                        'unrealized_plpc': float(position.unrealized_plpc)
                    }

            return None

        except Exception as e:
            logger.error(f"Error checking position: {e}")
            return None

    def place_bracket_order(self, symbol: str, qty: int, stop_loss_price: float, take_profit_price: float):
        """
        Place a bracket order (market buy with stop loss and take profit).

        Args:
            symbol: Stock symbol
            qty: Number of shares
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
        """
        try:
            if self.dry_run:
                logger.warning("DRY RUN MODE - Order NOT executed")
                logger.info(f"  [SIMULATED] BUY {qty} shares of {symbol}")
                logger.info(f"  [SIMULATED] Stop Loss: ${stop_loss_price:.2f}")
                logger.info(f"  [SIMULATED] Take Profit: ${take_profit_price:.2f}")
                return

            # Place bracket order via Alpaca
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='gtc',
                order_class='bracket',
                stop_loss={'stop_price': stop_loss_price},
                take_profit={'limit_price': take_profit_price}
            )

            logger.info(f"✓ Order placed successfully!")
            logger.info(f"  Order ID: {order.id}")
            logger.info(f"  Symbol: {symbol}")
            logger.info(f"  Qty: {qty}")
            logger.info(f"  Stop Loss: ${stop_loss_price:.2f}")
            logger.info(f"  Take Profit: ${take_profit_price:.2f}")

        except Exception as e:
            logger.error(f"Failed to place order: {e}")

    def run_trading_cycle(self):
        """
        Execute one complete trading cycle:
        1. Fetch data
        2. Calculate indicators
        3. Check hard filters
        4. Ask AI
        5. Execute if conditions met
        """
        logger.info("\n" + "="*60)
        logger.info(f"TRADING CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)

        # Check if we already have a position
        current_position = self.get_current_position()

        if current_position:
            logger.info(f"✓ Existing position detected:")
            logger.info(f"  Qty: {current_position['qty']}")
            logger.info(f"  Entry: ${current_position['avg_entry_price']:.2f}")
            logger.info(f"  Current: ${current_position['current_price']:.2f}")
            logger.info(f"  P&L: ${current_position['unrealized_pl']:+.2f} ({current_position['unrealized_plpc']*100:+.2f}%)")
            logger.info("  → Skipping new trade (position already open)")
            return

        # Fetch market data
        df = self.get_market_data()
        if df is None or len(df) < 210:
            logger.warning("Insufficient data (need 200+ bars for indicators), skipping cycle")
            return

        # Calculate indicators
        df = self.strategy.calculate_indicators(df)

        # Get the most recent bar
        latest_bar = df.iloc[-1]
        logger.info(f"\nLatest Data ({df.index[-1].strftime('%Y-%m-%d %H:%M')}):")
        logger.info(f"  Price: ${latest_bar['Close']:.2f}")
        # Log key indicators dynamically based on strategy
        for col in df.columns:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                val = latest_bar[col]
                if pd.notna(val) and isinstance(val, (int, float)):
                    logger.info(f"  {col}: {val:.2f}")

        # Check hard filters
        logger.info("\n[1/3] Checking hard filters...")
        if not self.strategy.check_hard_filters(latest_bar):
            logger.info("Hard filters FAILED. No trade.")
            return

        logger.info("-> Hard filters PASSED!")

        # Step 4: Ask AI
        logger.info("\n[2/3] Asking AI for decision...")
        prompt = self.strategy.generate_prompt(latest_bar, df.tail(10))
        ai_decision = self.ai_brain.analyze_market(prompt)

        logger.info(f"  AI Signal: {ai_decision['signal']}")
        logger.info(f"  Confidence: {ai_decision['confidence']:.1%}")
        logger.info(f"  Reasoning: {ai_decision['reasoning']}")

        # Step 5: Execute trade if conditions met
        if ai_decision['signal'] != 'BUY':
            logger.info(f"  → AI says {ai_decision['signal']}, no trade.")
            return

        if ai_decision['confidence'] < config.MIN_AI_CONFIDENCE:
            logger.info(f"  → Confidence {ai_decision['confidence']:.1%} < {config.MIN_AI_CONFIDENCE:.1%}, no trade.")
            return

        logger.info("\n[3/3] Executing trade...")
        logger.info(" -> All conditions met")

        # Calculate order parameters
        current_price = latest_bar['Close']
        stop_loss_price = current_price * (1 - config.STOP_LOSS_PCT)
        take_profit_price = current_price * (1 + config.TAKE_PROFIT_PCT)

        # Get account info for position sizing
        account = self.api.get_account()
        account_value = float(account.portfolio_value)
        qty = self.strategy.get_position_size(current_price, account_value)

        logger.info(f"\nOrder Details:")
        logger.info(f"  Entry Price: ${current_price:.2f}")
        logger.info(f"  Quantity: {qty} shares")
        logger.info(f"  Position Value: ${qty * current_price:.2f}")
        logger.info(f"  Stop Loss: ${stop_loss_price:.2f} (-{config.STOP_LOSS_PCT*100:.1f}%)")
        logger.info(f"  Take Profit: ${take_profit_price:.2f} (+{config.TAKE_PROFIT_PCT*100:.1f}%)")

        # Place the order
        self.place_bracket_order(
            symbol=self.ticker,
            qty=qty,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price
        )

        self.trade_count += 1
        logger.info(f"\n✓ Trade #{self.trade_count} completed!")

    def run(self):
        """
        Main loop: Run trading cycles continuously.
        """
        logger.info("\nStarting live trading loop...")
        logger.info(f"   Press Ctrl+C to stop\n")

        try:
            while True:
                # Run a trading cycle
                try:
                    self.run_trading_cycle()
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}", exc_info=True)

                # Wait before next cycle
                logger.info(f"\n Sleeping for {self.loop_interval}s...")
                logger.info("="*60)
                time.sleep(self.loop_interval)

        except KeyboardInterrupt:
            logger.info("\n\n  Shutdown signal received")
            logger.info("="*60)
            logger.info("TRADING SESSION SUMMARY")
            logger.info("="*60)
            logger.info(f"  Total Cycles: {self.trade_count}")

            # Show final account status
            try:
                account = self.api.get_account()
                logger.info(f"  Final Portfolio Value: ${float(account.portfolio_value):,.2f}")
                logger.info(f"  Buying Power: ${float(account.buying_power):,.2f}")

                position = self.get_current_position()
                if position:
                    logger.info(f"\n  Open Position:")
                    logger.info(f"    {position['qty']} shares @ ${position['avg_entry_price']:.2f}")
                    logger.info(f"    P&L: ${position['unrealized_pl']:+.2f} ({position['unrealized_plpc']*100:+.2f}%)")
            except:
                pass

            logger.info("="*60)
            logger.info("✓ Trader stopped safely")


# Main execution
if __name__ == "__main__":
    import argparse
    import json

    STRATEGY_MAP = {
        'hybrid': HybridStrategy,
        'breakout': BreakoutStrategy,
        'mean_reversion': MeanReversionStrategy,
        'smart_trend': SmartTrendStrategy,
        'volatility_squeeze': VolatilitySqueezeStrategy,
        'macd_scalp': MacdScalpStrategy,
        'macro_reversal': MacroReversalStrategy,
        'dmi_reversal': DmiReversalStrategy,
        'bidirectional': BidirectionalStrategy,
    }

    parser = argparse.ArgumentParser(description="Live AI Trading Bot")
    parser.add_argument(
        '--live',
        action='store_true',
        help='DANGER: Execute real orders (default is dry-run)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='hybrid',
        choices=list(STRATEGY_MAP.keys()),
        help='Strategy to use (default: hybrid)'
    )
    parser.add_argument(
        '--params',
        type=str,
        default='{}',
        help='JSON string of strategy params, e.g. \'{"RSI_OVERSOLD": 35}\''
    )

    args = parser.parse_args()

    # Parse strategy params
    try:
        strategy_params = json.loads(args.params)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid --params JSON: {e}")
        exit(1)

    strategy_class = STRATEGY_MAP[args.strategy]

    # Validate config
    try:
        config.validate_config()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        exit(1)

    # Initialize trader
    dry_run = not args.live

    if not dry_run:
        logger.warning("\n" + "!"*60)
        logger.warning("⚠️  LIVE MODE ENABLED - REAL ORDERS WILL BE EXECUTED!")
        logger.warning("!"*60)
        response = input("\nType 'YES' to confirm: ")
        if response != 'YES':
            logger.info("Cancelled by user.")
            exit(0)

    trader = LiveTrader(
        dry_run=dry_run,
        strategy_class=strategy_class,
        strategy_params=strategy_params
    )
    trader.run()
