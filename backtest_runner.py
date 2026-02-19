import pandas as pd
import yfinance as yf
import logging
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from typing import Optional
import inspect
import config
from strategy import MeanReversionStrategy, SmartTrendStrategy, BidirectionalStrategy
from ai_brain import AIBrain
import backtesting._plotting
# Note: backtest_runner.py is used in main terminal for single ticker quick backtests
# unlike grid search which uses vectorbt

# Default strategy (can be overridden)
HybridStrategy = MeanReversionStrategy

backtesting._plotting._add_popcon = lambda: None
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_backtest_strategy(strategy_class):
    """
    Factory function to create a backtest strategy with a specific strategy class.
    Automatically detects bidirectional strategies (IS_BIDIRECTIONAL=True)
    and uses the appropriate long/short execution logic.
    """
    class AIBacktestStrategy(Strategy):
        """
        Universal Backtest Strategy.
        Automatically adapts to whatever indicators the strategy calculates.
        Supports both long-only and bidirectional (long+short) strategies.
        """
        use_real_ai = False
        stop_loss_pct = config.STOP_LOSS_PCT
        take_profit_pct = config.TAKE_PROFIT_PCT

        def init(self):
            self.hybrid_strategy = strategy_class()
            self.ai_brain = None
            self._is_bidirectional = getattr(self.hybrid_strategy, 'IS_BIDIRECTIONAL', False)

            # Reconstruct the DataFrame for the strategy
            df = pd.DataFrame({
                'Open': self.data.Open,
                'High': self.data.High,
                'Low': self.data.Low,
                'Close': self.data.Close,
                'Volume': self.data.Volume
            })

            # Calculate all indicators using the imported Strategy
            try:
                self.full_df = self.hybrid_strategy.calculate_indicators(df)
                logger.info(f"Indicators calculated: {list(self.full_df.columns)}")
            except Exception as e:
                logger.error(f"Failed to calculate indicators: {e}")
                raise

            # Register main lines for plotting
            if 'BB_UPPER' in self.full_df.columns:
                self.I(lambda: self.full_df['BB_UPPER'].values, name='BB_Upper')
                self.I(lambda: self.full_df['BB_LOWER'].values, name='BB_Lower')
            elif f'SMA_{config.SMA_PERIOD}' in self.full_df.columns:
                self.I(lambda: self.full_df[f'SMA_{config.SMA_PERIOD}'].values, name='SMA')
            elif 'EMA_200' in self.full_df.columns:
                self.I(lambda: self.full_df['EMA_200'].values, name='EMA_200')

            if self.use_real_ai:
                self.ai_brain = AIBrain()
                logger.warning("⚠️  Real AI mode enabled - backtest will be SLOW!")

            self.trade_count = 0

            if self._is_bidirectional:
                logger.info("🔄 Bidirectional mode enabled (Long + Short)")

        def next(self):
            i = len(self.data) - 1
            row = self.full_df.iloc[i]

            # Skip if indicators are NaN (warmup period)
            if pd.isna(row.iloc[-1]):
                return

            # Dispatch to the correct execution path
            if self._is_bidirectional:
                self._next_bidirectional(row)
            else:
                self._next_standard(row)

        # Standard path - long only
        def _next_standard(self, row):
            # Check for exit signals (Trailing Stops / Disaster Stops)
            if self.position:
                if hasattr(self.hybrid_strategy, 'check_exit_signal'):
                    try:
                        sig = inspect.signature(self.hybrid_strategy.check_exit_signal)
                        exit_args = {'row': row}

                        if 'entry_price' in sig.parameters:
                            entry = getattr(self.position, 'entry_price',
                                    getattr(self.position, 'price',
                                    getattr(self.position, 'avg_price', self.data.Close[-1])))
                            exit_args['entry_price'] = entry

                        should_exit = self.hybrid_strategy.check_exit_signal(**exit_args)
                        if should_exit:
                            logger.info(f"[Exit] Closing position at ${self.data.Close[-1]:.2f} (Exit signal triggered)")
                            self.position.close()

                    except Exception as e:
                        if not hasattr(self, '_logged_error'):
                            logger.warning(f"CRITICAL: Exit signal check failed: {e}")
                            logger.warning(f"Available attributes on Position: {dir(self.position)}")
                            self._logged_error = True

                return  # Don't enter new position if we already have one

            # Check hard filters
            try:
                if not self.hybrid_strategy.check_hard_filters(row):
                    return
            except (KeyError, AttributeError) as e:
                logger.debug(f"[Debug] Hard filter check failed (likely missing indicator): {e}")
                return

            # Get AI decision
            try:
                if self.use_real_ai and self.ai_brain:
                    prompt = self.hybrid_strategy.generate_prompt(row)
                    ai_decision = self.ai_brain.analyze_market(prompt)
                else:
                    ai_decision = self._mock_ai_decision(row)
            except Exception as e:
                logger.warning(f"[Warning] AI decision failed: {e}")
                return

            # Execute
            if ai_decision['signal'] == 'BUY' and ai_decision['confidence'] >= config.MIN_AI_CONFIDENCE:
                entry_price = self.data.Close[-1]
                sl_price = entry_price * (1 - self.stop_loss_pct)
                tp_price = entry_price * (1 + self.take_profit_pct)

                self.buy(sl=sl_price, tp=tp_price)
                self.trade_count += 1

                if self.use_real_ai:
                    logger.info(f"[Trade #{self.trade_count}] BUY at ${entry_price:.2f} | AI: {ai_decision['reasoning']}")


        # Bidirectional Long and Short with flip logic
        def _next_bidirectional(self, row):
            signal = self.hybrid_strategy.get_signal(row)
            entry_price = self.data.Close[-1]

            # Holding a position
            if self.position:
                # Check exit signal 
                if hasattr(self.hybrid_strategy, 'check_exit_signal'):
                    try:
                        sig = inspect.signature(self.hybrid_strategy.check_exit_signal)
                        exit_args = {'row': row}
                        if 'entry_price' in sig.parameters:
                            entry = getattr(self.position, 'entry_price',
                                    getattr(self.position, 'price',
                                    getattr(self.position, 'avg_price', entry_price)))
                            exit_args['entry_price'] = entry

                        if self.hybrid_strategy.check_exit_signal(**exit_args):
                            logger.info(f"[Exit] Closing at ${entry_price:.2f} (Regime exit)")
                            self.position.close()
                            return
                    except Exception:
                        pass

                # Flip logic: Opposite signal -> close current position.
                # The opposite entry will open on the next bar when the signal persists.
                # This avoids backtesting.py exclusive_orders conflicts.
                if self.position.is_long and signal == -1:
                    logger.info(f"[Flip] Closing LONG at ${entry_price:.2f} (Short signal detected)")
                    self.position.close()
                    return
                elif self.position.is_short and signal == 1:
                    logger.info(f"[Flip] Closing SHORT at ${entry_price:.2f} (Long signal detected)")
                    self.position.close()
                    return

                return  # Hold current position

            # Flat: Check for new entry
            if signal == 0:
                return

            # AI confidence gate
            try:
                if self.use_real_ai and self.ai_brain:
                    prompt = self.hybrid_strategy.generate_prompt(row)
                    ai_decision = self.ai_brain.analyze_market(prompt)
                else:
                    ai_decision = self._mock_ai_decision(row)
            except Exception as e:
                logger.warning(f"[Warning] AI decision failed: {e}")
                return

            if ai_decision['confidence'] < config.MIN_AI_CONFIDENCE:
                return

            # Execute: Long or Short with separate risk parameters
            if signal == 1:
                sl_pct = self.hybrid_strategy.long_stop_loss / 100
                sl_price = entry_price * (1 - sl_pct)
                tp_price = entry_price * (1 + self.take_profit_pct)
                self.buy(sl=sl_price, tp=tp_price)
                self.trade_count += 1
                logger.info(f"[Trade #{self.trade_count}] LONG at ${entry_price:.2f} | SL=${sl_price:.2f} TP=${tp_price:.2f}")

            elif signal == -1:
                sl_pct = self.hybrid_strategy.short_stop_loss / 100
                sl_price = entry_price * (1 + sl_pct)
                tp_price = entry_price * (1 - self.take_profit_pct)
                self.sell(sl=sl_price, tp=tp_price)
                self.trade_count += 1
                logger.info(f"[Trade #{self.trade_count}] SHORT at ${entry_price:.2f} | SL=${sl_price:.2f} TP=${tp_price:.2f}")

        def _mock_ai_decision(self, row):
            # If the hard filters (math) passed, we ALWAYS take the trade.
            # This removes all randomness from the backtest.
            return {
                'signal': 'BUY',
                'confidence': 1,
                'reasoning': 'Hard Filters Passed (Deterministic)'
            }

    return AIBacktestStrategy


def download_and_resample(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Download historical data with smart resampling for custom timeframes.

    Args:
        ticker: Stock/Crypto symbol (e.g., "SPY", "BTC-USD")
        period: Historical period (e.g., "1y", "5y", "10y", "max")
        interval: Timeframe (e.g., "1m", "5m", "15m", "1h", "4h", "1d", "1wk")

    Returns:
        DataFrame with OHLCV data
    """
    # Auto-cap period for short intervals (yfinance limits)
    if interval == "1m" and period not in ["1d", "5d"]:
        logger.warning(f"  → 1m interval: capping period from {period} to 5d (yfinance limit: ~7 days)")
        period = "5d"
    elif interval in ["5m", "15m"] and period not in ["1d", "5d", "1mo"]:
        logger.warning(f"  → {interval} interval: capping period from {period} to 1mo (yfinance limit: ~60 days)")
        period = "1mo"

    logger.info(f"Downloading {ticker} data (period={period}, interval={interval})...")

    # Special handling for 4h interval (not natively supported by yfinance)
    if interval == "4h":
        logger.info("  → 4h interval requested. Downloading 1h data and resampling...")

        # Download 1h data
        df = yf.download(ticker, period=period, interval="1h", progress=False)

        if df.empty:
            logger.error("Failed to download 1h data for resampling!")
            return df

        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                df.columns = df.columns.get_level_values(0)
            else:
                df.columns = df.columns.get_level_values(1)

        # Resample to 4h candles
        # Aggregation rules: Open=first, High=max, Low=min, Close=last, Volume=sum
        df_resampled = df.resample('4h').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        logger.info(f"  → Resampled from {len(df)} 1h bars to {len(df_resampled)} 4h bars")
        return df_resampled

    else:
        # Standard download for all other intervals
        df = yf.download(ticker, period=period, interval=interval, progress=False)

        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                df.columns = df.columns.get_level_values(0)
            else:
                df.columns = df.columns.get_level_values(1)

        return df


def run_backtest(
    ticker: str = None,
    period: str = None,
    interval: str = None,
    strategy_class = None,
    use_real_ai: bool = False,
    plot: bool = True
) -> dict:
    """
    Run a backtest on historical data with support for crypto and custom timeframes.

    Args:
        ticker: Symbol to trade (e.g., "SPY", "BTC-USD"). Defaults to config.TICKER.
        period: Historical period (e.g., "1y", "5y", "10y", "max"). Defaults to "1y".
        interval: Timeframe (e.g., "1h", "4h", "1d", "1wk"). Defaults to config.TIMEFRAME.
        strategy_class: Strategy class to use (MeanReversionStrategy or SmartTrendStrategy).
        use_real_ai: If True, use real AI model (SLOW). If False, use mock AI (FAST).
        plot: If True, display interactive backtest chart.

    Returns:
        Dict with backtest results
    """
    # Use defaults from config if not provided
    ticker = ticker or config.TICKER
    period = period or "1y"
    interval = interval or config.TIMEFRAME
    strategy_class = strategy_class or HybridStrategy

    logger.info("="*60)
    logger.info("STARTING BACKTEST")
    logger.info("="*60)
    logger.info(f"  Ticker:   {ticker}")
    logger.info(f"  Period:   {period}")
    logger.info(f"  Interval: {interval}")

    # Download historical data with smart resampling
    df = download_and_resample(ticker, period, interval)

    if df.empty:
        logger.error("Failed to download data!")
        return {}

    # Use only recent bars to speed up backtest 
    if hasattr(config, 'BACKTEST_BARS') and config.BACKTEST_BARS > 0:
        num_bars = min(config.BACKTEST_BARS, len(df))
        df = df.tail(num_bars)
        logger.info(f"  Limited to last {num_bars} bars (config.BACKTEST_BARS)")

    logger.info(f"  Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    logger.info(f"  Strategy: {strategy_class.__name__}")

    # Create backtest strategy with selected strategy class
    BacktestStrategy = create_backtest_strategy(strategy_class)

    # Initialize backtest
    bt = Backtest(
        df,
        BacktestStrategy,
        cash=10000,  # Starting capital
        commission=0.001,  # 0.1% commission per trade
        exclusive_orders=True  # Only one order type (SL or TP) can execute
    )

    # Run with chosen AI mode
    logger.info(f"Running backtest with {'REAL' if use_real_ai else 'MOCK'} AI...")
    stats = bt.run(use_real_ai=use_real_ai)

    # Calculate strategy return (not buy & hold)
    equity_curve = stats._equity_curve
    initial_cash = 10000.0
    final_equity = equity_curve['Equity'].iloc[-1]
    strategy_return_pct = ((final_equity - initial_cash) / initial_cash) * 100

    # Calculate Buy & Hold return for comparison
    initial_price = df['Close'].iloc[0]
    final_price = df['Close'].iloc[-1]
    buy_hold_return_pct = ((final_price - initial_price) / initial_price) * 100

    # Override the potentially incorrect backtesting.py return with the calculated one
    stats['Strategy Return [%]'] = strategy_return_pct
    stats['Buy & Hold Return [%]'] = buy_hold_return_pct
    stats['Return [%]'] = strategy_return_pct  # Fix the main return metric

    # Print results
    logger.info("\n" + "="*60)
    logger.info("BACKTEST RESULTS")
    logger.info("="*60)
    print(stats)

    logger.info("\nKey Metrics:")
    logger.info(f"  Strategy Return: {strategy_return_pct:.2f}%")
    logger.info(f"  Buy & Hold:      {buy_hold_return_pct:.2f}%")
    logger.info(f"  Sharpe Ratio:    {stats['Sharpe Ratio']:.2f}")
    logger.info(f"  Max Drawdown:    {stats['Max. Drawdown [%]']:.2f}%")
    logger.info(f"  Total Trades:    {stats['# Trades']}")
    logger.info(f"  Win Rate:        {stats['Win Rate [%]']:.1f}%")

    # Plot results
    if plot:
        logger.info("\nGenerating backtest chart...")
        bt.plot()

    return stats


def optimize_parameters(ticker: str = None, period: str = "1y", interval: str = None):
    """
    Optimize strategy parameters using grid search.

    WARNING: This is SLOW with real AI. Only use with mock AI.

    Args:
        ticker: Symbol to trade
        period: Historical period
        interval: Timeframe
    """
    ticker = ticker or config.TICKER
    interval = interval or config.TIMEFRAME

    logger.info("="*60)
    logger.info("PARAMETER OPTIMIZATION")
    logger.info("="*60)

    # Download data
    df = download_and_resample(ticker, period, interval)

    if hasattr(config, 'BACKTEST_BARS') and config.BACKTEST_BARS > 0:
        df = df.tail(config.BACKTEST_BARS)

    # Initialize backtest
    bt = Backtest(df, AIBacktestStrategy, cash=10000, commission=0.001)

    # Optimize parameters
    logger.info("Optimizing stop_loss_pct and take_profit_pct...")
    logger.info("This may take several minutes...\n")

    stats = bt.optimize(
        stop_loss_pct=[0.005, 0.01, 0.015, 0.02],  # 0.5%, 1%, 1.5%, 2%
        take_profit_pct=[0.01, 0.02, 0.03, 0.04],  # 1%, 2%, 3%, 4%
        maximize='Sharpe Ratio',
        constraint=lambda p: p.take_profit_pct > p.stop_loss_pct  # TP must be > SL
    )

    logger.info("\nOptimal Parameters:")
    logger.info(f"  Stop Loss:    {stats._strategy.stop_loss_pct*100:.1f}%")
    logger.info(f"  Take Profit:  {stats._strategy.take_profit_pct*100:.1f}%")
    logger.info(f"  Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")

    return stats


# Main Execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest the AI Trading Strategy")
    parser.add_argument('--ticker', type=str, help='Ticker symbol (e.g., SPY, BTC-USD)')
    parser.add_argument('--period', type=str, default='1y', help='Historical period (1y, 5y, 10y, max)')
    parser.add_argument('--interval', type=str, help='Timeframe (1m, 5m, 15m, 1h, 4h, 1d, 1wk)')
    parser.add_argument('--real-ai', action='store_true', help='Use real AI model (SLOW)')
    parser.add_argument('--optimize', action='store_true', help='Run parameter optimization')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting results')

    args = parser.parse_args()

    try:
        if args.optimize:
            optimize_parameters(
                ticker=args.ticker,
                period=args.period,
                interval=args.interval
            )
        else:
            run_backtest(
                ticker=args.ticker,
                period=args.period,
                interval=args.interval,
                use_real_ai=args.real_ai,
                plot=not args.no_plot
            )

    except KeyboardInterrupt:
        logger.info("\n\nBacktest interrupted by user.")
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
