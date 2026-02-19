import pandas as pd
import numpy as np
import pandas_ta_classic as ta
import logging
from typing import Optional, Dict
import config

logger = logging.getLogger(__name__)


class HybridStrategy:
    """
    Hybrid trading strategy combining technical indicators (hard filters)
    with AI analysis (soft filters) for trade decisions.

    HARD FILTERS (Math-based):
    - Price must be above SMA 200 (uptrend)
    - RSI must be below 45 (oversold pullback)

    SOFT FILTERS (AI-based):
    - AI analyzes the context and makes final BUY/HOLD decision
    """

    @staticmethod
    def get_optimizable_params():
        """Return optimizable parameters for this strategy."""
        return {
            "RSI_OVERSOLD": {
                "name": "RSI Threshold",
                "type": "int",
                "min": 10,
                "max": 100,
                "step": 1,
                "default": 45,
                "optimize_min": 10,
                "optimize_max": 100,
                "optimize_step": 5,
                "description": "RSI < value = Oversold signal"
            },
            "SMA_PERIOD": {
                "name": "SMA Period",
                "type": "int",
                "min": 20,
                "max": 600,
                "step": 10,
                "default": 200,
                "optimize_min": 50,
                "optimize_max": 400,
                "optimize_step": 25,
                "description": "Simple Moving Average period for trend filter"
            }
        }

    @staticmethod
    def generate_vectorized_signals(close, high, low, volume, param_ranges):
        rsi = ta.rsi(close, length=14)
        sma_period_range = param_ranges.get('SMA_PERIOD', np.array([200]))
        rsi_oversold_range = param_ranges.get('RSI_OVERSOLD', np.array([45]))

        all_entries, all_exits, col_sma, col_rsi = [], [], [], []
        rsi_vals = rsi.values[:, None]
        close_vals = close.values[:, None]

        for sma_period in sma_period_range:
            sma = ta.sma(close, length=int(sma_period))
            price_above = (close_vals > sma.values[:, None])
            entries = price_above & (rsi_vals < rsi_oversold_range)
            exits = np.broadcast_to(~price_above[:, :1], entries.shape)
            all_entries.append(entries)
            all_exits.append(exits)
            n = len(rsi_oversold_range)
            col_sma.extend([sma_period] * n)
            col_rsi.extend(rsi_oversold_range.tolist())

        T = len(close)
        zeros = np.zeros((T, len(col_sma)), dtype=bool)
        return {
            'long_entries': np.hstack(all_entries) if all_entries else zeros,
            'long_exits': np.hstack(all_exits) if all_exits else zeros,
            'short_entries': zeros,
            'short_exits': zeros,
            'param_columns': {'SMA Period': np.array(col_sma), 'RSI Threshold': np.array(col_rsi)},
        }

    def __init__(self, **kwargs):
        """Initialize the strategy with parameters from config or kwargs."""
        self.sma_period = kwargs.get('SMA_PERIOD', getattr(config, 'SMA_PERIOD', 200))
        self.rsi_period = kwargs.get('RSI_PERIOD', getattr(config, 'RSI_PERIOD', 14))
        self.rsi_oversold = kwargs.get('RSI_OVERSOLD', getattr(config, 'RSI_OVERSOLD', 45))

        logger.info(
            f"HybridStrategy initialized: SMA={self.sma_period}, "
            f"RSI={self.rsi_period}, RSI_Threshold={self.rsi_oversold}"
        )

    @staticmethod
    def get_chart_config():
        """Define what indicators to plot on the chart."""
        return [
            {"name": "SMA 200", "column": "SMA_200", "color": "orange", "panel": 1, "width": 2},
            {"name": "RSI", "column": "RSI_14", "color": "purple", "panel": 2, "width": 2, "hlines": [30, 45, 70]},
        ]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators and add them as columns to the dataframe.

        Args:
            df: DataFrame with OHLCV data (requires: Open, High, Low, Close, Volume)

        Returns:
            DataFrame with added indicator columns: SMA_200, RSI_14
        """
        if df is None or len(df) == 0:
            logger.warning("Empty dataframe provided to calculate_indicators")
            return df

        # Make a copy to avoid modifying original
        df = df.copy()

        # Calculate Simple Moving Average
        df[f'SMA_{self.sma_period}'] = ta.sma(df['Close'], length=self.sma_period)

        # Calculate RSI
        df[f'RSI_{self.rsi_period}'] = ta.rsi(df['Close'], length=self.rsi_period)

        # Calculate additional useful metrics
        df['Price_vs_SMA'] = ((df['Close'] - df[f'SMA_{self.sma_period}']) /
                               df[f'SMA_{self.sma_period}'] * 100)  # Percentage above/below SMA

        # Calculate recent volatility (ATR)
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['ATR_14'] = atr
        df['ATR_Pct'] = (atr / df['Close'] * 100)  # ATR as % of price

        # Calculate volume ratio (current vs 20-period average)
        df['Volume_SMA_20'] = ta.sma(df['Volume'], length=20)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']

        logger.debug(f"Indicators calculated for {len(df)} bars")

        return df

    def check_hard_filters(self, row: pd.Series) -> bool:
        """
        Check if the current bar passes hard mathematical filters.

        RULES:
        1. Price must be ABOVE SMA 200 (confirmed uptrend)
        2. RSI must be BELOW 45 (oversold condition = pullback in uptrend)

        Args:
            row: Single row from DataFrame (must contain Close, SMA_200, RSI_14)

        Returns:
            True if ALL hard filters pass, False otherwise
        """
        try:
            price = row['Close']
            sma = row[f'SMA_{self.sma_period}']
            rsi = row[f'RSI_{self.rsi_period}']

            # Check for NaN values (early in dataset before indicators are calculated)
            if pd.isna(sma) or pd.isna(rsi):
                logger.debug("Hard filters: FAILED - Indicators not yet available (NaN)")
                return False

            # FILTER 1: Price above SMA (uptrend)
            uptrend = price > sma

            # FILTER 2: RSI oversold (pullback opportunity)
            oversold = rsi < self.rsi_oversold

            # Both conditions must be true
            passed = uptrend and oversold

            if passed:
                logger.info(
                    f"Hard filters PASSED: "
                    f"Price=${price:.2f} > SMA=${sma:.2f} ({(price/sma-1)*100:+.2f}%), "
                    f"RSI={rsi:.1f} < {self.rsi_oversold}"
                )
            else:
                reasons = []
                if not uptrend:
                    reasons.append(f"Price ${price:.2f} ≤ SMA ${sma:.2f}")
                if not oversold:
                    reasons.append(f"RSI {rsi:.1f} ≥ {self.rsi_oversold}")

                logger.debug(f"✗ Hard filters FAILED: {', '.join(reasons)}")

            return passed

        except KeyError as e:
            logger.error(f"Missing required column in hard filter check: {e}")
            return False
        except Exception as e:
            logger.error(f"Error in hard filter check: {e}")
            return False

    def generate_prompt(self, row: pd.Series, lookback_rows: Optional[pd.DataFrame] = None) -> str:
        """
        Generate a natural language prompt for the AI based on current market data.

        Args:
            row: Current bar data (must contain OHLCV + indicators)
            lookback_rows: Optional DataFrame with recent bars for context

        Returns:
            Formatted string prompt for AI analysis
        """
        try:
            # Extract key metrics
            symbol = config.TICKER
            price = row['Close']
            sma = row[f'SMA_{self.sma_period}']
            rsi = row[f'RSI_{self.rsi_period}']
            price_vs_sma = row.get('Price_vs_SMA', (price/sma - 1) * 100)
            atr_pct = row.get('ATR_Pct', 0)
            volume_ratio = row.get('Volume_Ratio', 1.0)

            # Build the prompt
            prompt = f"""Symbol: {symbol}
Current Price: ${price:.2f}
SMA {self.sma_period}: ${sma:.2f}
RSI {self.rsi_period}: {rsi:.1f}

Technical Analysis:
- Price is {price_vs_sma:+.2f}% {'ABOVE' if price > sma else 'BELOW'} the {self.sma_period}-day moving average
- RSI at {rsi:.1f} indicates {'OVERSOLD' if rsi < 30 else 'NEUTRAL' if rsi < 70 else 'OVERBOUGHT'} conditions
- Recent volatility (ATR): {atr_pct:.2f}% of price
- Volume is {volume_ratio:.1f}x the 20-period average

Market Context:
"""

            # Add recent price action if available
            if lookback_rows is not None and len(lookback_rows) >= 5:
                recent = lookback_rows.tail(5)
                price_change = (price / recent.iloc[0]['Close'] - 1) * 100
                high_5d = recent['High'].max()
                low_5d = recent['Low'].min()
                price_position = (price - low_5d) / (high_5d - low_5d) * 100

                prompt += f"- Price change over last 5 bars: {price_change:+.2f}%\n"
                prompt += f"- Current price is {price_position:.0f}% through 5-bar range (${low_5d:.2f} - ${high_5d:.2f})\n"

                # Trend direction
                if len(recent) >= 3:
                    recent_closes = recent['Close'].values
                    if recent_closes[-1] > recent_closes[-2] > recent_closes[-3]:
                        prompt += "- Short-term trend: BULLISH (3 consecutive higher closes)\n"
                    elif recent_closes[-1] < recent_closes[-2] < recent_closes[-3]:
                        prompt += "- Short-term trend: BEARISH (3 consecutive lower closes)\n"
                    else:
                        prompt += "- Short-term trend: CHOPPY (no clear direction)\n"

            # Add risk/reward context
            prompt += f"""
Risk Management:
- Proposed Stop Loss: {config.STOP_LOSS_PCT*100:.1f}% (${price * (1-config.STOP_LOSS_PCT):.2f})
- Proposed Take Profit: {config.TAKE_PROFIT_PCT*100:.1f}% (${price * (1+config.TAKE_PROFIT_PCT):.2f})
- Risk/Reward Ratio: 1:{config.TAKE_PROFIT_PCT/config.STOP_LOSS_PCT:.1f}

IMPORTANT: This setup has already passed hard filters (uptrend + oversold RSI).
Your job: Assess the quality of this setup. Is the risk/reward favorable?
"""

            return prompt.strip()

        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            # Return minimal prompt on error
            return f"Symbol: {config.TICKER}, Price: ${row.get('Close', 0):.2f}, Analysis failed."

    def get_position_size(self, price: float, account_value: float) -> int:
        """
        Calculate position size based on risk parameters.

        Args:
            price: Current stock price
            account_value: Total account value

        Returns:
            Number of shares to trade
        """
        # Use fixed dollar amount from config
        dollar_amount = min(config.POSITION_SIZE, account_value * 0.95)  # Don't use more than 95% of account
        shares = int(dollar_amount / price)

        return max(1, shares)  # At least 1 share

    def format_signal_summary(self, row: pd.Series, ai_decision: Dict) -> str:
        """
        Format a human-readable summary of the trading signal.

        Args:
            row: Current market data
            ai_decision: AI decision dict with signal, confidence, reasoning

        Returns:
            Formatted string summary
        """
        summary = f"""
{'='*60}
TRADING SIGNAL SUMMARY
{'='*60}
Symbol:      {config.TICKER}
Price:       ${row['Close']:.2f}
SMA {self.sma_period}:     ${row[f'SMA_{self.sma_period}']:.2f} ({row.get('Price_vs_SMA', 0):+.2f}%)
RSI {self.rsi_period}:      {row[f'RSI_{self.rsi_period}']:.1f}

AI Decision: {ai_decision['signal']}
Confidence:  {ai_decision['confidence']:.1%}
Reasoning:   {ai_decision['reasoning']}

Risk Mgmt:   SL={config.STOP_LOSS_PCT*100:.1f}% / TP={config.TAKE_PROFIT_PCT*100:.1f}%
{'='*60}
"""
        return summary

class BreakoutStrategy:
    """
    STRATEGY 2: Volatility Breakout

    HARD FILTERS:
    1. Price closes ABOVE the Upper Bollinger Band.
    2. Volume is > 1.5x the average volume (Volume Breakout).

    SOFT FILTERS (AI):
    - Analyzes if the breakout candle is strong (Marubozu) or weak (doji/wick).
    - Checks if the move is overextended.
    """

    @staticmethod
    def get_optimizable_params():
        """Return optimizable parameters for this strategy."""
        return {
            "BB_STD": {
                "name": "BB Std Dev",
                "type": "float",
                "min": 1.0,
                "max": 8.0,
                "step": 0.1,
                "default": 2.0,
                "optimize_min": 1.5,
                "optimize_max": 5.0,
                "optimize_step": 0.25,
                "description": "Bollinger Bands standard deviation multiplier"
            },
            "VOL_RATIO": {
                "name": "Volume Ratio",
                "type": "float",
                "min": 1.0,
                "max": 6.0,
                "step": 0.1,
                "default": 1.5,
                "optimize_min": 1.2,
                "optimize_max": 4.0,
                "optimize_step": 0.2,
                "description": "Volume must be > ratio × average for breakout confirmation"
            }
        }

    @staticmethod
    def generate_vectorized_signals(close, high, low, volume, param_ranges):
        bb_std_range = param_ranges.get('BB_STD', np.array([2.0]))
        vol_ratio_range = param_ranges.get('VOL_RATIO', np.array([1.5]))

        sma_20 = ta.sma(close, length=20)
        rolling_std = close.rolling(20).std()
        vol_sma = volume.rolling(20).mean()
        vol_ratio_series = (volume / vol_sma).values[:, None]

        grid_bb, grid_vol = np.meshgrid(bb_std_range, vol_ratio_range)
        flat_bb = grid_bb.flatten()
        flat_vol = grid_vol.flatten()

        bb_upper = sma_20.values[:, None] + flat_bb * rolling_std.values[:, None]
        entries = (close.values[:, None] > bb_upper) & (vol_ratio_series > flat_vol)
        T = len(close)
        N = len(flat_bb)
        return {
            'long_entries': entries,
            'long_exits': np.zeros((T, N), dtype=bool),
            'short_entries': np.zeros((T, N), dtype=bool),
            'short_exits': np.zeros((T, N), dtype=bool),
            'param_columns': {'BB Std Dev': flat_bb, 'Volume Ratio': flat_vol},
        }

    def __init__(self, **kwargs):
        self.bb_length = kwargs.get('BB_LENGTH', getattr(config, 'BB_LENGTH', 20))
        self.bb_std = kwargs.get('BB_STD', getattr(config, 'BB_STD', 2.0))
        self.vol_ratio = kwargs.get('VOL_RATIO', getattr(config, 'VOL_RATIO', 1.5))
        logger.info(f"BreakoutStrategy initialized: BB({self.bb_length},{self.bb_std}), VolRatio={self.vol_ratio}")

    @staticmethod
    def get_chart_config():
        """Define what indicators to plot on the chart."""
        return [
            {"name": "BB Upper", "column": "BB_UPPER", "color": "rgba(255,255,255,0.3)", "panel": 1, "width": 1},
            {"name": "BB Lower", "column": "BB_LOWER", "color": "rgba(255,255,255,0.3)", "panel": 1, "width": 1, "fill": "tonexty"},
        ]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Calculate Bollinger Bands
        # Note: pandas_ta returns 3 columns: BBL (Lower), BBM (Mid), BBU (Upper)
        bb = ta.bbands(df['Close'], length=self.bb_length, std=self.bb_std)
        
        # Rename columns to be easier to use
        # The default names are like BBL_20_2.0, BBU_20_2.0
        df['BB_LOWER'] = bb.iloc[:, 0]
        df['BB_MID'] = bb.iloc[:, 1]
        df['BB_UPPER'] = bb.iloc[:, 2]

        # Volume SMA
        df['Volume_SMA'] = ta.sma(df['Volume'], length=20)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

        # Bandwidth (Volatility Metric)
        df['Bandwidth'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MID'] * 100

        return df

    def check_hard_filters(self, row: pd.Series) -> bool:
        try:
            price = row['Close']
            upper_band = row['BB_UPPER']
            vol_ratio = row.get('Volume_Ratio', 0)

            # RULE 1: Price broke the upper band
            breakout = price > upper_band

            # RULE 2: Volume must be high (confirming the move)
            volume_confirm = vol_ratio > self.vol_ratio

            if breakout and volume_confirm:
                logger.info(f"✓ Breakout PASSED: Price ${price:.2f} > BB Upper ${upper_band:.2f}, Vol Ratio {vol_ratio:.1f}x (threshold: {self.vol_ratio}x)")
                return True

            return False

        except Exception as e:
            logger.error(f"Error in breakout filter: {e}")
            return False

    def generate_prompt(self, row: pd.Series, lookback_rows: Optional[pd.DataFrame] = None) -> str:
        # Custom AI prompt specifically for Breakouts
        price = row['Close']
        upper_band = row['BB_UPPER']
        vol_ratio = row.get('Volume_Ratio', 1.0)
        bandwidth = row.get('Bandwidth', 0)
        
        prompt = f"""
Symbol: {config.TICKER}
Current Price: ${price:.2f}
Strategy: Volatility Breakout

Technical Context:
- Price has BROKEN OUT above the Upper Bollinger Band (${upper_band:.2f}).
- Volume is {vol_ratio:.1f}x the average (Strong buying pressure).
- Bollinger Bandwidth is {bandwidth:.2f}% (Volatility measurement).

Your Job:
Analyze this breakout. 
- Is this a "fake out" (price extended too far too fast)?
- Or is this the start of a new momentum trend?
- If the bandwidth is extremely wide (>5%), be cautious of exhaustion.
- If the bandwidth was narrow before this (<1%), this is a "Squeeze Breakout" (Very Bullish).

Reply with BUY if you believe the momentum will continue.
"""
        return prompt.strip()
    
    def get_position_size(self, price: float, account_value: float) -> int:
        # Same sizing logic as before
        dollar_amount = min(config.POSITION_SIZE, account_value * 0.95)
        return max(1, int(dollar_amount / price))

class MeanReversionStrategy:
    """
    STRATEGY 3: Mean Reversion (Buy the Fear)

    HARD FILTERS:
    1. Price is BELOW the Lower Bollinger Band (Extreme fear).
    2. RSI is BELOW 30 (Oversold).

    SOFT FILTERS (AI):
    - AI checks news sentiment (simulation) or market structure.
    - AI looks for "Hammer" candles or rejection wicks indicating a bottom.
    """

    @staticmethod
    def get_optimizable_params():
        """Return optimizable parameters for this strategy."""
        return {
            "RSI_OVERSOLD": {
                "name": "RSI Threshold",
                "type": "int",
                "min": 10,
                "max": 100,
                "step": 1,
                "default": 55,
                "optimize_min": 10,
                "optimize_max": 100,
                "optimize_step": 5,
                "description": "RSI < value = Oversold signal"
            },
            "BB_LENGTH": {
                "name": "BB Length",
                "type": "int",
                "min": 10,
                "max": 100,
                "step": 5,
                "default": 20,
                "optimize_min": 15,
                "optimize_max": 60,
                "optimize_step": 5,
                "description": "Bollinger Bands period"
            }
        }

    @staticmethod
    def generate_vectorized_signals(close, high, low, volume, param_ranges):
        rsi = ta.rsi(close, length=14)
        bb_length_range = param_ranges.get('BB_LENGTH', np.array([20]))
        rsi_oversold_range = param_ranges.get('RSI_OVERSOLD', np.array([55]))

        all_entries, all_exits, col_bb, col_rsi = [], [], [], []
        rsi_vals = rsi.values[:, None]
        close_vals = close.values[:, None]

        for bb_len in bb_length_range:
            bb = ta.bbands(close, length=int(bb_len), std=2.0)
            bb_lower = bb.iloc[:, 0].values[:, None]
            bb_mid = bb.iloc[:, 1].values[:, None]

            entries = (close_vals < bb_lower) & (rsi_vals < rsi_oversold_range)
            exits = np.broadcast_to(close_vals > bb_mid, entries.shape)

            all_entries.append(entries)
            all_exits.append(exits)
            n = len(rsi_oversold_range)
            col_bb.extend([bb_len] * n)
            col_rsi.extend(rsi_oversold_range.tolist())

        T = len(close)
        zeros = np.zeros((T, len(col_bb)), dtype=bool)
        return {
            'long_entries': np.hstack(all_entries) if all_entries else zeros,
            'long_exits': np.hstack(all_exits) if all_exits else zeros,
            'short_entries': zeros,
            'short_exits': zeros,
            'param_columns': {'BB Length': np.array(col_bb), 'RSI Threshold': np.array(col_rsi)},
        }

    def __init__(self, **kwargs):
        # Use kwargs if provided, otherwise fall back to config
        self.bb_length = kwargs.get('BB_LENGTH', getattr(config, 'BB_LENGTH', 20))
        self.bb_std = kwargs.get('BB_STD', getattr(config, 'BB_STD', 2.0))
        self.rsi_period = kwargs.get('RSI_PERIOD', getattr(config, 'RSI_PERIOD', 14))
        self.rsi_oversold = kwargs.get('RSI_OVERSOLD', getattr(config, 'RSI_OVERSOLD', 55))
        logger.info(f"MeanReversionStrategy initialized: BB={self.bb_length}, RSI_Threshold={self.rsi_oversold}")

    @staticmethod
    def get_chart_config():
        """Define what indicators to plot on the chart."""
        return [
            {"name": "BB Upper", "column": "BB_UPPER", "color": "rgba(255,255,255,0.3)", "panel": 1, "width": 1},
            {"name": "BB Lower", "column": "BB_LOWER", "color": "rgba(255,255,255,0.3)", "panel": 1, "width": 1, "fill": "tonexty"},
            {"name": "RSI", "column": "RSI", "color": "purple", "panel": 2, "width": 2, "hlines": [30, 70]},
        ]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Bollinger Bands
        bb = ta.bbands(df['Close'], length=self.bb_length, std=self.bb_std)
        df['BB_LOWER'] = bb.iloc[:, 0]
        df['BB_UPPER'] = bb.iloc[:, 2]
        
        # RSI
        df['RSI'] = ta.rsi(df['Close'], length=self.rsi_period)
        
        # Distance from Lower Band (Negative means below band)
        df['Band_Dist'] = (df['Close'] - df['BB_LOWER']) / df['BB_LOWER'] * 100
        
        return df

    def check_hard_filters(self, row: pd.Series) -> bool:
        try:
            price = row['Close']
            lower_band = row['BB_LOWER']
            rsi = row['RSI']

            # RULE 1: Price panic (below lower band)
            panic_mode = price < lower_band

            # RULE 2: RSI oversold (use self.rsi_oversold for dynamic threshold)
            oversold = rsi < self.rsi_oversold

            if panic_mode and oversold:
                logger.info(f"✓ Panic Detected: Price ${price:.2f} < BB Low ${lower_band:.2f}, RSI {rsi:.1f}")
                return True

            return False

        except Exception:
            return False

    def generate_prompt(self, row: pd.Series, lookback_rows: Optional[pd.DataFrame] = None) -> str:
        price = row['Close']
        rsi = row['RSI']
        
        prompt = f"""
Symbol: SPY
Current Price: ${price:.2f}
Strategy: Mean Reversion (Catching the Knife)

Technical Context:
- Price has CRASHED below the Lower Bollinger Band (Statistical outlier).
- RSI is {rsi:.1f} (Heavily Oversold).

Your Job:
We are looking to buy this dip. Be critical:
- Is this a temporary panic (BUY)? 
- Or is the asset collapsing (HOLD)?
- If RSI is below 20, the bounce is highly likely (Strong Signal).

Reply with BUY if you think the panic is overdone.
"""
        return prompt.strip()
    
    def get_position_size(self, price: float, account_value: float) -> int:
        dollar_amount = min(config.POSITION_SIZE, account_value * 0.95)
        return max(1, int(dollar_amount / price))

class SmartTrendStrategy:
    """
    STRATEGY 4: Smart Trend Following (Outperform Buy & Hold)

    HARD FILTERS:
    1. Price > EMA_200 (Long-term uptrend)
    2. EMA_50 > EMA_200 (Golden Cross alignment)
    3. ADX > 25 (Strong trend, not noise)

    EXIT SIGNAL:
    - Price drops below EMA_50 (Trailing Stop)

    SOFT FILTERS (AI):
    - Validates Volume Trend (volume rising with price = good)
    - Rejects if volume dropping while price rising (trap)
    """

    @staticmethod
    def get_optimizable_params():
        """Return optimizable parameters for this strategy."""
        return {
            "ADX_THRESHOLD": {
                "name": "ADX Threshold",
                "type": "int",
                "min": 10,
                "max": 100,
                "step": 5,
                "default": 25,
                "optimize_min": 15,
                "optimize_max": 80,
                "optimize_step": 5,
                "description": "ADX > value = Strong trend required"
            },
            "EMA_FAST": {
                "name": "EMA Fast Period",
                "type": "int",
                "min": 20,
                "max": 200,
                "step": 10,
                "default": 50,
                "optimize_min": 30,
                "optimize_max": 140,
                "optimize_step": 10,
                "description": "Fast EMA period for Golden Cross"
            }
        }

    @staticmethod
    def generate_vectorized_signals(close, high, low, volume, param_ranges):
        ema_fast_range = param_ranges.get('EMA_FAST', np.array([50]))
        adx_threshold_range = param_ranges.get('ADX_THRESHOLD', np.array([25]))

        ema_200 = ta.ema(close, length=200)
        adx_data = ta.adx(high, low, close, length=14)
        adx = adx_data['ADX_14']

        all_entries, all_exits, col_ema, col_adx = [], [], [], []
        close_vals = close.values[:, None]
        ema200_vals = ema_200.values[:, None]
        adx_vals = adx.values[:, None]

        for ema_fast in ema_fast_range:
            ema_f = ta.ema(close, length=int(ema_fast))
            ema_f_vals = ema_f.values[:, None]

            price_above = close_vals > ema200_vals
            golden_cross = ema_f_vals > ema200_vals
            strong_trend = adx_vals > adx_threshold_range

            entries = price_above & golden_cross & strong_trend
            exits = np.broadcast_to(close_vals < ema_f_vals, entries.shape)

            all_entries.append(entries)
            all_exits.append(exits)
            n = len(adx_threshold_range)
            col_ema.extend([ema_fast] * n)
            col_adx.extend(adx_threshold_range.tolist())

        T = len(close)
        zeros = np.zeros((T, len(col_ema)), dtype=bool)
        return {
            'long_entries': np.hstack(all_entries) if all_entries else zeros,
            'long_exits': np.hstack(all_exits) if all_exits else zeros,
            'short_entries': zeros,
            'short_exits': zeros,
            'param_columns': {'EMA Fast Period': np.array(col_ema), 'ADX Threshold': np.array(col_adx)},
        }

    def __init__(self, **kwargs):
        self.ema_fast = kwargs.get('EMA_FAST', 50)
        self.ema_slow = kwargs.get('EMA_SLOW', 200)
        self.adx_period = kwargs.get('ADX_PERIOD', 14)
        self.adx_threshold = kwargs.get('ADX_THRESHOLD', getattr(config, 'ADX_THRESHOLD', 25))
        logger.info(f"SmartTrendStrategy initialized: EMA({self.ema_fast},{self.ema_slow}), ADX>{self.adx_threshold}")

    @staticmethod
    def get_chart_config():
        """Define what indicators to plot on the chart."""
        return [
            {"name": "EMA 50", "column": "EMA_50", "color": "orange", "panel": 1, "width": 2},
            {"name": "EMA 200", "column": "EMA_200", "color": "cyan", "panel": 1, "width": 2},
            {"name": "ADX", "column": "ADX", "color": "purple", "panel": 2, "width": 2, "hlines": [25]},
        ]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Calculate EMAs
        df['EMA_50'] = ta.ema(df['Close'], length=self.ema_fast)
        df['EMA_200'] = ta.ema(df['Close'], length=self.ema_slow)

        # Calculate ADX (Trend Strength)
        adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=self.adx_period)
        df['ADX'] = adx_data[f'ADX_{self.adx_period}']

        # Calculate Volume metrics
        df['Volume_SMA_20'] = ta.sma(df['Volume'], length=20)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']

        # Calculate volume trend (5-period slope)
        df['Volume_Trend'] = df['Volume'].rolling(window=5).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False
        )

        # EMA alignment score
        df['EMA_Alignment'] = (df['EMA_50'] > df['EMA_200']).astype(int)

        return df

    def check_hard_filters(self, row: pd.Series) -> bool:
        """
        Entry Filters (Golden Cross + Strong Trend):
        1. Price > EMA_200 (Long-term uptrend)
        2. EMA_50 > EMA_200 (Golden Cross alignment)
        3. ADX > 25 (Strong trend)
        """
        try:
            price = row['Close']
            ema_50 = row['EMA_50']
            ema_200 = row['EMA_200']
            adx = row['ADX']

            # Check for NaN
            if pd.isna(ema_50) or pd.isna(ema_200) or pd.isna(adx):
                return False

            # RULE 1: Price above long-term trend
            price_above_trend = price > ema_200

            # RULE 2: Golden Cross alignment (fast > slow)
            golden_cross = ema_50 > ema_200

            # RULE 3: Strong trend (ADX > threshold)
            strong_trend = adx > self.adx_threshold

            passed = price_above_trend and golden_cross and strong_trend

            if passed:
                logger.info(
                    f"✓ Trend Signal: Price=${price:.2f} > EMA200=${ema_200:.2f}, "
                    f"EMA50=${ema_50:.2f} > EMA200 (Golden Cross), ADX={adx:.1f} > {self.adx_threshold}"
                )

            return passed

        except Exception as e:
            logger.error(f"Error in trend filter: {e}")
            return False

    def check_exit_signal(self, row: pd.Series) -> bool:
        """
        Exit Signal (Trailing Stop):
        Close position if price drops below EMA_50

        Returns:
            True if should exit position
        """
        try:
            price = row['Close']
            ema_50 = row['EMA_50']

            if pd.isna(ema_50):
                return False

            # Exit if price breaks below fast EMA (trend reversal)
            should_exit = price < ema_50

            if should_exit:
                logger.info(f"✗ Exit Signal: Price ${price:.2f} < EMA50 ${ema_50:.2f} (Trailing Stop)")

            return should_exit

        except Exception:
            return False

    def generate_prompt(self, row: pd.Series, lookback_rows: Optional[pd.DataFrame] = None) -> str:
        """
        Generate AI prompt that validates Volume Trend.
        AI should reject if volume dropping while price rising (trap).
        """
        price = row['Close']
        ema_50 = row['EMA_50']
        ema_200 = row['EMA_200']
        adx = row['ADX']
        volume_ratio = row.get('Volume_Ratio', 1.0)
        volume_trend = row.get('Volume_Trend', 0)

        # Calculate price momentum
        price_change_pct = 0
        if lookback_rows is not None and len(lookback_rows) >= 5:
            old_price = lookback_rows.iloc[-5]['Close']
            price_change_pct = (price / old_price - 1) * 100

        prompt = f"""Symbol: {config.TICKER}
Current Price: ${price:.2f}
Strategy: Smart Trend Following

Technical Setup:
- Price: ${price:.2f}
- EMA 50: ${ema_50:.2f}
- EMA 200: ${ema_200:.2f}
- Golden Cross: (EMA50 > EMA200)
- ADX (Trend Strength): {adx:.1f} (Strong trend confirmed)

Price Action:
- 5-bar momentum: {price_change_pct:+.2f}%
- Price is riding the trend above both EMAs

Volume Analysis (CRITICAL):
- Current Volume: {volume_ratio:.1f}x the 20-period average
- Volume Trend: {'RISING' if volume_trend > 0 else 'FALLING'}

YOUR CRITICAL JOB:
This is a Golden Cross trend setup. The math says BUY.
BUT you must validate the VOLUME TREND:

GOOD SIGNAL: Volume is RISING as price rises (healthy accumulation)
BAD SIGNAL: Volume is FALLING as price rises (TRAP - lack of conviction)

If volume is declining while price climbs, this is a weak move that will reverse.
Reply BUY only if volume confirms the trend.
"""
        return prompt.strip()

    def get_position_size(self, price: float, account_value: float) -> int:
        dollar_amount = min(config.POSITION_SIZE, account_value * 0.95)
        return max(1, int(dollar_amount / price))

class VolatilitySqueezeStrategy:
    """
    STRATEGY 5: Volatility Squeeze (TTM Squeeze Variant)

    LOGIC:
    1. SQUEEZE: Volatility falls. Bollinger Bands (2.0) go INSIDE Keltner Channels (2.0).
    2. TRIGGER: Price breaks out above the Upper Bollinger Band.
    3. CONFIRMATION: Momentum is positive.
    """

    @staticmethod
    def get_optimizable_params():
        """Return optimizable parameters for this strategy."""
        return {
            "KELTNER_MULT": {
                "name": "Keltner Multiplier",
                "type": "float",
                "min": 1.0,
                "max": 8.0,
                "step": 0.1,
                "default": 1.5,
                "optimize_min": 1.0,
                "optimize_max": 6.0,
                "optimize_step": 0.25,
                "description": "ATR × value = Keltner Channel width"
            },
            "BB_LENGTH": {
                "name": "BB Length",
                "type": "int",
                "min": 10,
                "max": 100,
                "step": 5,
                "default": 20,
                "optimize_min": 15,
                "optimize_max": 60,
                "optimize_step": 5,
                "description": "Bollinger Bands period"
            }
        }

    @staticmethod
    def generate_vectorized_signals(close, high, low, volume, param_ranges):
        bb_length_range = param_ranges.get('BB_LENGTH', np.array([20]))
        kc_mult_range = param_ranges.get('KELTNER_MULT', np.array([1.5]))

        all_entries, all_exits, col_bb, col_kc = [], [], [], []
        close_vals = close.values[:, None]
        T = len(close)

        for bb_len in bb_length_range:
            bb_len_int = int(bb_len)
            bb = ta.bbands(close, length=bb_len_int, std=2.0)
            bb_lower = bb.iloc[:, 0].values[:, None]
            bb_upper = bb.iloc[:, 2].values[:, None]

            kc_mid = ta.ema(close, length=bb_len_int)
            atr = ta.atr(high, low, close, length=bb_len_int)
            kc_mid_vals = kc_mid.values[:, None]
            atr_vals = atr.values[:, None]

            kc_upper = kc_mid_vals + kc_mult_range * atr_vals
            kc_lower = kc_mid_vals - kc_mult_range * atr_vals

            in_squeeze = (bb_lower > kc_lower) & (bb_upper < kc_upper)

            N = len(kc_mult_range)
            was_squeezed = np.zeros((T, N), dtype=bool)
            for t in range(T):
                start = max(0, t - 4)
                was_squeezed[t] = in_squeeze[start:t+1].any(axis=0)

            momentum = (close - close.rolling(bb_len_int).mean()).values[:, None]

            entries = was_squeezed & (close_vals > bb_upper) & (momentum > 0)
            all_entries.append(entries)
            all_exits.append(np.zeros((T, N), dtype=bool))
            col_bb.extend([bb_len] * N)
            col_kc.extend(kc_mult_range.tolist())

        total = len(col_bb)
        zeros = np.zeros((T, total), dtype=bool)
        return {
            'long_entries': np.hstack(all_entries) if all_entries else zeros,
            'long_exits': np.hstack(all_exits) if all_exits else zeros,
            'short_entries': zeros,
            'short_exits': zeros,
            'param_columns': {'BB Length': np.array(col_bb), 'Keltner Multiplier': np.array(col_kc)},
        }

    def __init__(self, **kwargs):
        # Use kwargs if provided, otherwise fall back to config
        self.bb_length = kwargs.get('BB_LENGTH', getattr(config, 'BB_LENGTH', 20))
        self.bb_std = kwargs.get('BB_STD', getattr(config, 'BB_STD', 2.0))
        self.kc_length = kwargs.get('KC_LENGTH', 20)
        self.kc_mult = kwargs.get('KELTNER_MULT', getattr(config, 'KELTNER_MULT', 1.5))

        logger.info(f"VolatilitySqueeze initialized: KC Mult={self.kc_mult}, BB Length={self.bb_length}")

    @staticmethod
    def get_chart_config():
        """Define what indicators to plot on the chart."""
        return [
            {"name": "BB Upper", "column": "BB_UPPER", "color": "rgba(255,255,255,0.3)", "panel": 1, "width": 1},
            {"name": "BB Lower", "column": "BB_LOWER", "color": "rgba(255,255,255,0.3)", "panel": 1, "width": 1, "fill": "tonexty"},
            {"name": "KC Upper", "column": "KC_UPPER", "color": "rgba(255,165,0,0.5)", "panel": 1, "width": 2, "dash": "dash"},
            {"name": "KC Lower", "column": "KC_LOWER", "color": "rgba(255,165,0,0.5)", "panel": 1, "width": 2, "dash": "dash"},
        ]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Bollinger Bands (20, 2.0)
        bb = ta.bbands(df['Close'], length=self.bb_length, std=self.bb_std)
        df['BB_LOWER'] = bb.iloc[:, 0]
        df['BB_MID'] = bb.iloc[:, 1]
        df['BB_UPPER'] = bb.iloc[:, 2]
        
        # 2. Keltner Channels (20, 2.0 ATR)
        # KC Middle is usually EMA 20
        kc_mid = ta.ema(df['Close'], length=self.kc_length)
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=self.kc_length)
        
        df['KC_UPPER'] = kc_mid + (self.kc_mult * atr)
        df['KC_LOWER'] = kc_mid - (self.kc_mult * atr)
        
        # 3. Squeeze Metric (For Visualization)
        # If BB Upper < KC Upper, we are in a squeeze
        df['In_Squeeze'] = (df['BB_UPPER'] < df['KC_UPPER']) & (df['BB_LOWER'] > df['KC_LOWER'])

        # 4. Was Squeezed Recently (Fix for instant band expansion)
        # Check if In_Squeeze was True in any of the last 5 bars
        df['Was_Squeezed'] = df['In_Squeeze'].rolling(window=5, min_periods=1).max().astype(bool)

        # 5. Momentum (Delta from average)
        df['Momentum'] = df['Close'] - df['Close'].rolling(window=20).mean()

        return df

    def check_hard_filters(self, row: pd.Series) -> bool:
        try:
            # Check if we WERE in a squeeze recently (last 5 bars)
            was_squeezed = row['Was_Squeezed']

            # Check for Breakout (Price exploding above BB Upper)
            breakout = row['Close'] > row['BB_UPPER']

            # Check Momentum
            positive_mom = row['Momentum'] > 0

            # LOGIC:
            # Only trigger if there was a recent squeeze AND we're breaking out with momentum
            # This prevents false signals during normal volatility
            passed = was_squeezed and breakout and positive_mom

            if passed:
                logger.info(f"Squeeze Breakout: Was_Squeezed=True, Price ${row['Close']:.2f} > BB ${row['BB_UPPER']:.2f}, Momentum={row['Momentum']:.2f}")

            return passed

        except Exception as e:
            logger.error(f"Error in Squeeze filter: {e}")
            return False

    def generate_prompt(self, row: pd.Series, lookback_rows: Optional[pd.DataFrame] = None) -> str:
        return f"""
Symbol: {config.TICKER}
Price: ${row['Close']:.2f}
Strategy: Volatility Squeeze

Context:
- Price has BROKEN OUT above the Upper Bollinger Band.
- Momentum is POSITIVE.
- Volatility was recently compressed (The Squeeze).

Your Job:
Is this a fake-out? 
- If volume is rising, BUY.
- If price is just drifting up with low volume, HOLD.
"""
    
    def get_position_size(self, price: float, account_value: float) -> int:
        dollar_amount = min(config.POSITION_SIZE, account_value * 0.95)
        return max(1, int(dollar_amount / price))

class MacdScalpStrategy:
    """
    STRATEGY 6: MACD Scalping (Fast Momentum Trades)

    CONCEPT: Catch early momentum shifts using MACD crossover + Stochastic RSI filter

    HARD FILTERS:
    1. MACD Line crosses above Signal Line (Bullish crossover)
    2. Stochastic RSI < 80 (Not overbought)
    3. MACD Histogram positive (Confirming momentum)

    SOFT FILTERS (AI):
    - Identifies support/resistance levels
    - Validates if breakout is clean or near resistance
    """

    @staticmethod
    def get_optimizable_params():
        """Return optimizable parameters for this strategy."""
        return {
            "MACD_FAST": {
                "name": "MACD Fast Period",
                "type": "int",
                "min": 8,
                "max": 100,
                "step": 1,
                "default": 12,
                "optimize_min": 8,
                "optimize_max": 64,
                "optimize_step": 2,
                "description": "Lower = More sensitive to price changes"
            },
            "MACD_SLOW": {
                "name": "MACD Slow Period",
                "type": "int",
                "min": 20,
                "max": 150,
                "step": 2,
                "default": 26,
                "optimize_min": 22,
                "optimize_max": 100,
                "optimize_step": 2,
                "description": "Higher = Smoother signal"
            }
        }

    @staticmethod
    def generate_vectorized_signals(close, high, low, volume, param_ranges):
        macd_fast_range = param_ranges.get('MACD_FAST', np.array([12]))
        macd_slow_range = param_ranges.get('MACD_SLOW', np.array([26]))

        stoch = ta.stochrsi(close, length=14)
        stochrsi_k = stoch['STOCHRSIk_14_14_3_3'].values

        all_entries, col_fast, col_slow = [], [], []
        T = len(close)

        for fast in macd_fast_range:
            for slow in macd_slow_range:
                if fast >= slow:
                    continue
                f, s, sig = int(fast), int(slow), 9
                macd_data = ta.macd(close, fast=f, slow=s, signal=sig)
                macd_vals = macd_data[f'MACD_{f}_{s}_{sig}'].values
                signal_vals = macd_data[f'MACDs_{f}_{s}_{sig}'].values
                hist_vals = macd_data[f'MACDh_{f}_{s}_{sig}'].values

                macd_above = macd_vals > signal_vals
                macd_above_prev = np.roll(macd_above, 1)
                macd_above_prev[0] = False
                crossover = macd_above & ~macd_above_prev

                entry = crossover & (stochrsi_k < 80) & (hist_vals > 0)
                all_entries.append(entry[:, None])
                col_fast.append(fast)
                col_slow.append(slow)

        if all_entries:
            entries = np.hstack(all_entries)
        else:
            entries = np.zeros((T, 1), dtype=bool)
            col_fast, col_slow = [12], [26]

        N = entries.shape[1]
        zeros = np.zeros((T, N), dtype=bool)
        return {
            'long_entries': entries,
            'long_exits': zeros,
            'short_entries': zeros,
            'short_exits': zeros,
            'param_columns': {'MACD Fast Period': np.array(col_fast), 'MACD Slow Period': np.array(col_slow)},
        }

    def __init__(self, **kwargs):
        self.macd_fast = kwargs.get('MACD_FAST', getattr(config, 'MACD_FAST', 12))
        self.macd_slow = kwargs.get('MACD_SLOW', getattr(config, 'MACD_SLOW', 26))
        self.macd_signal = kwargs.get('MACD_SIGNAL', getattr(config, 'MACD_SIGNAL', 9))
        self.stoch_rsi_period = kwargs.get('STOCH_RSI_PERIOD', 14)
        logger.info(f"MacdScalpStrategy initialized: MACD({self.macd_fast},{self.macd_slow},{self.macd_signal}), StochRSI({self.stoch_rsi_period})")

    @staticmethod
    def get_chart_config():
        """Define what indicators to plot on the chart."""
        return [
            {"name": "MACD", "column": "MACD", "color": "blue", "panel": 2, "width": 2},
            {"name": "MACD Signal", "column": "MACD_Signal", "color": "orange", "panel": 2, "width": 2},
            {"name": "StochRSI", "column": "StochRSI_K", "color": "purple", "panel": 2, "width": 2, "hlines": [20, 80]},
        ]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # MACD
        macd = ta.macd(df['Close'], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        df['MACD'] = macd[f'MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
        df['MACD_Signal'] = macd[f'MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
        df['MACD_Hist'] = macd[f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']

        # Stochastic RSI
        stoch = ta.stochrsi(df['Close'], length=self.stoch_rsi_period)
        df['StochRSI_K'] = stoch[f'STOCHRSIk_{self.stoch_rsi_period}_14_3_3']

        # MACD Crossover detection
        df['MACD_Cross'] = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))

        # Support/Resistance (Simple: 20-period high/low)
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support'] = df['Low'].rolling(window=20).min()

        return df

    def check_hard_filters(self, row: pd.Series) -> bool:
        """
        Entry: MACD bullish crossover + StochRSI not overbought
        """
        try:
            macd = row['MACD']
            macd_signal = row['MACD_Signal']
            macd_hist = row['MACD_Hist']
            stoch_rsi = row['StochRSI_K']
            macd_cross = row['MACD_Cross']

            if pd.isna(macd) or pd.isna(stoch_rsi):
                return False

            # RULE 1: MACD crossover (bullish)
            bullish_cross = macd_cross

            # RULE 2: Stochastic RSI not overbought
            not_overbought = stoch_rsi < 80

            # RULE 3: MACD Histogram positive (momentum confirmation)
            positive_momentum = macd_hist > 0

            passed = bullish_cross and not_overbought and positive_momentum

            if passed:
                logger.info(
                    f"✓ MACD Scalp Signal: MACD {macd:.2f} > Signal {macd_signal:.2f}, "
                    f"Histogram {macd_hist:+.2f}, StochRSI {stoch_rsi:.1f}"
                )

            return passed

        except Exception as e:
            logger.error(f"Error in MACD filter: {e}")
            return False

    def generate_prompt(self, row: pd.Series, lookback_rows: Optional[pd.DataFrame] = None) -> str:
        price = row['Close']
        macd = row['MACD']
        macd_signal = row['MACD_Signal']
        stoch_rsi = row['StochRSI_K']
        resistance = row['Resistance']
        support = row['Support']

        distance_to_resistance = ((resistance - price) / price) * 100

        prompt = f"""Symbol: {config.TICKER}
Current Price: ${price:.2f}
Strategy: MACD Scalping

Technical Setup:
- MACD Line: {macd:.2f}
- MACD Signal: {macd_signal:.2f}
- MACD just crossed ABOVE Signal (Bullish momentum)
- Stochastic RSI: {stoch_rsi:.1f} (Not overbought)

Support/Resistance Context:
- Nearest Resistance: ${resistance:.2f} ({distance_to_resistance:+.1f}% away)
- Nearest Support: ${support:.2f}

YOUR CRITICAL JOB:
This is a fresh MACD crossover. The math says BUY.
BUT you must check the RESISTANCE LEVEL:

GOOD SIGNAL: Price has 2%+ room to resistance (clean runway)
BAD SIGNAL: Price is < 1% from resistance (will hit ceiling)

If resistance is too close, this trade has no room to run.
Reply BUY only if there's sufficient upside room.
"""
        return prompt.strip()

    def get_position_size(self, price: float, account_value: float) -> int:
        dollar_amount = min(config.POSITION_SIZE, account_value * 0.95)
        return max(1, int(dollar_amount / price))


class MacroReversalStrategy:
    """
    STRATEGY 7: Macro Reversal (The Golden Goose)
    Catches V-shape bottoms on Daily/4H charts.
    """

    @staticmethod
    def get_optimizable_params():
        return {
            "RSI_OVERSOLD": {
                "name": "RSI Oversold Threshold",
                "type": "int",
                "min": 10, "max": 100, "step": 5, "default": 30,
                "optimize_min": 10, "optimize_max": 100, "optimize_step": 5,
                "description": "RSI level to define the 'Crash'"
            },
            "RSI_EXIT": {
                "name": "RSI Overbought Exit",
                "type": "int",
                "min": 50, "max": 100, "step": 5, "default": 70,
                "optimize_min": 50, "optimize_max": 100, "optimize_step": 5,
                "description": "RSI level to take profit"
            },
            "DISASTER_STOP": {
                "name": "Disaster Stop (%)",
                "type": "float",
                "min": 1.0, "max": 100.0, "step": 0.5, "default": 5.0,
                "optimize_min": 3.0, "optimize_max": 20.0, "optimize_step": 1.0,
                "description": "Emergency exit if price drops X% below entry",
                "category": "long_sl"
            },
            "LOOKBACK": {
                "name": "Lookback Period",
                "type": "int",
                "min": 5, "max": 60, "step": 5, "default": 10,
                "optimize_min": 5, "optimize_max": 40, "optimize_step": 5,
                "description": "Bars to look back for the crash"
            }
        }

    @staticmethod
    def generate_vectorized_signals(close, high, low, volume, param_ranges):
        lookback_range = param_ranges.get('LOOKBACK', np.array([10]))
        rsi_oversold_range = param_ranges.get('RSI_OVERSOLD', np.array([30]))
        rsi_exit_range = param_ranges.get('RSI_EXIT', np.array([70]))

        rsi = ta.rsi(close, length=14)
        rsi_vals = rsi.values[:, None]
        close_vals = close.values[:, None]
        high_prev = high.shift(1).values[:, None]

        grid_oversold, grid_exit = np.meshgrid(rsi_oversold_range, rsi_exit_range)
        flat_oversold = grid_oversold.flatten()
        flat_exit = grid_exit.flatten()

        all_entries, all_exits = [], []
        col_lookback, col_oversold, col_exit = [], [], []

        for lookback in lookback_range:
            lb = int(lookback)
            lowest_rsi = rsi.rolling(window=lb).min().values[:, None]

            was_oversold = lowest_rsi < flat_oversold
            recovery = rsi_vals > flat_oversold
            momentum = close_vals > high_prev

            entries = was_oversold & recovery & momentum
            exits = rsi_vals > flat_exit

            all_entries.append(entries)
            all_exits.append(exits)
            n = len(flat_oversold)
            col_lookback.extend([lookback] * n)
            col_oversold.extend(flat_oversold.tolist())
            col_exit.extend(flat_exit.tolist())

        T = len(close)
        total = len(col_lookback)
        zeros = np.zeros((T, total), dtype=bool)
        return {
            'long_entries': np.hstack(all_entries) if all_entries else zeros,
            'long_exits': np.hstack(all_exits) if all_exits else zeros,
            'short_entries': zeros,
            'short_exits': zeros,
            'param_columns': {
                'Lookback Period': np.array(col_lookback),
                'RSI Oversold Threshold': np.array(col_oversold),
                'RSI Overbought Exit': np.array(col_exit),
            },
        }

    @staticmethod
    def get_chart_config():
        return [
            {"name": "RSI", "column": "RSI", "panel": 2, "color": "purple", "hlines": [30, 70]}
        ]

    def __init__(self, **kwargs):
        self.rsi_period = 14
        self.rsi_oversold = kwargs.get('RSI_OVERSOLD', 30)
        self.rsi_exit = kwargs.get('RSI_EXIT', 70)
        self.lookback = kwargs.get('LOOKBACK', 10)
        # Convert percentage (e.g., 5.0) to decimal (0.05)
        self.disaster_stop = kwargs.get('DISASTER_STOP', 5.0) / 100.0

        logger.info(f"MacroReversal initialized: Stop={self.disaster_stop*100}%")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['RSI'] = ta.rsi(df['Close'], length=self.rsi_period)

        # "Was Oversold": Did RSI drop below threshold in the last X bars?
        # rolling().min() checks the lowest RSI in the window
        df['Lowest_RSI'] = df['RSI'].rolling(window=self.lookback).min()
        df['Was_Oversold'] = df['Lowest_RSI'] < self.rsi_oversold

        # Momentum: Previous High (for breakout confirmation)
        df['High_Prev'] = df['High'].shift(1)

        return df

    def check_hard_filters(self, row: pd.Series) -> bool:
        try:
            # 1. SETUP: We were in a crash recently
            was_oversold = row['Was_Oversold']

            # 2. TRIGGER: We have recovered (RSI back above threshold)
            current_rsi = row['RSI']
            recovery = current_rsi > self.rsi_oversold

            # 3. MOMENTUM: Price is breaking above previous candle's high
            momentum = row['Close'] > row['High_Prev']

            if was_oversold and recovery and momentum:
                return True

            return False
        except Exception:
            return False

    def check_exit_signal(self, row: pd.Series, entry_price: float = None) -> bool:
        """
        Exit if:
        1. RSI hits Overbought (Profit Take)
        2. Price hits Disaster Stop (Safety Net)
        """
        try:
            current_rsi = row['RSI']
            current_price = row['Close']

            # 1. Profit Take (RSI Overbought)
            if current_rsi > self.rsi_exit:
                logger.info(f"Exit: RSI Overbought ({current_rsi:.1f} > {self.rsi_exit})")
                return True

            # 2. Disaster Stop (Safety Net)
            if entry_price and entry_price > 0:
                stop_price = entry_price * (1 - self.disaster_stop)
                if current_price < stop_price:
                    logger.info(f"Exit: Disaster Stop Triggered! Price ${current_price:.2f} < ${stop_price:.2f}")
                    return True

            return False
        except Exception:
            return False

    def generate_prompt(self, row: pd.Series, lookback_rows: Optional[pd.DataFrame] = None) -> str:
        return f"Strategy: Macro Reversal. RSI Recovery from {self.rsi_oversold}. Confirm uptrend?"

    def get_position_size(self, price: float, account_value: float) -> int:
        dollar_amount = min(config.POSITION_SIZE, account_value * 0.95)
        return max(1, int(dollar_amount / price))


class DmiReversalStrategy:
    """
    STRATEGY 8: DMI Reversal (Discrepancy Hunter)
    Catches bottoms when bears are exhausted (-DI high) but bulls haven't shown up yet (+DI low).
    Trades the massive spread (discrepancy) between +DI and -DI.
    """

    @staticmethod
    def get_optimizable_params():
        return {
            "DI_LENGTH": {
                "name": "DMI Length",
                "type": "int",
                "min": 7, "max": 42, "step": 1, "default": 14,
                "optimize_min": 10, "optimize_max": 36, "optimize_step": 2,
                "description": "Period for ADX/DMI calculation"
            },
            "SPREAD_THRESHOLD": {
                "name": "Spread Threshold",
                "type": "float",
                "min": 10.0, "max": 100.0, "step": 1.0, "default": 20.0,
                "optimize_min": 15.0, "optimize_max": 80.0, "optimize_step": 5.0,
                "description": "Minimum gap between +DI and -DI to enter"
            },
            "DI_LOW_LIMIT": {
                "name": "+DI Low Limit",
                "type": "float",
                "min": 5.0, "max": 50.0, "step": 1.0, "default": 15.0,
                "optimize_min": 5.0, "optimize_max": 40.0, "optimize_step": 5.0,
                "description": "+DI must be below this (bulls beaten)"
            },
            "DI_HIGH_LIMIT": {
                "name": "-DI High Limit",
                "type": "float",
                "min": 20.0, "max": 100.0, "step": 1.0, "default": 30.0,
                "optimize_min": 25.0, "optimize_max": 90.0, "optimize_step": 5.0,
                "description": "-DI must be above this (bears exhausted)"
            }
        }

    @staticmethod
    def generate_vectorized_signals(close, high, low, volume, param_ranges):
        di_length_range = param_ranges.get('DI_LENGTH', np.array([14]))
        spread_range = param_ranges.get('SPREAD_THRESHOLD', np.array([20.0]))
        di_low_range = param_ranges.get('DI_LOW_LIMIT', np.array([15.0]))
        di_high_range = param_ranges.get('DI_HIGH_LIMIT', np.array([30.0]))

        grid = np.meshgrid(spread_range, di_low_range, di_high_range)
        flat_spread = grid[0].flatten()
        flat_low = grid[1].flatten()
        flat_high = grid[2].flatten()

        all_entries, all_exits = [], []
        col_length, col_spread, col_low, col_high = [], [], [], []
        T = len(close)

        for di_len in di_length_range:
            dl = int(di_len)
            adx_data = ta.adx(high, low, close, length=dl)
            di_plus = adx_data[f'DMP_{dl}'].values[:, None]
            di_minus = adx_data[f'DMN_{dl}'].values[:, None]

            bears_exhausted = di_minus > flat_high
            bulls_beaten = di_plus < flat_low
            spread_large = np.abs(di_plus - di_minus) > flat_spread

            entries = bears_exhausted & bulls_beaten & spread_large

            di_cross = (di_plus > di_minus)
            di_cross_prev = np.roll(di_cross, 1, axis=0)
            di_cross_prev[0] = False
            exits = di_cross & ~di_cross_prev
            exits = np.broadcast_to(exits, entries.shape).copy()

            all_entries.append(entries)
            all_exits.append(exits)
            n = len(flat_spread)
            col_length.extend([di_len] * n)
            col_spread.extend(flat_spread.tolist())
            col_low.extend(flat_low.tolist())
            col_high.extend(flat_high.tolist())

        total = len(col_length)
        zeros = np.zeros((T, total), dtype=bool)
        return {
            'long_entries': np.hstack(all_entries) if all_entries else zeros,
            'long_exits': np.hstack(all_exits) if all_exits else zeros,
            'short_entries': zeros,
            'short_exits': zeros,
            'param_columns': {
                'DMI Length': np.array(col_length),
                'Spread Threshold': np.array(col_spread),
                '+DI Low Limit': np.array(col_low),
                '-DI High Limit': np.array(col_high),
            },
        }

    @staticmethod
    def get_chart_config():
        return [
            {"name": "+DI", "column": "DI_Plus", "panel": 2, "color": "blue"},
            {"name": "-DI", "column": "DI_Minus", "panel": 2, "color": "orange"},
            {"name": "ADX", "column": "ADX", "panel": 2, "color": "gray", "dash": "dash"}
        ]

    def __init__(self, **kwargs):
        self.di_length = kwargs.get('DI_LENGTH', 14)
        self.spread_threshold = kwargs.get('SPREAD_THRESHOLD', 20.0)
        self.di_low_limit = kwargs.get('DI_LOW_LIMIT', 15.0)
        self.di_high_limit = kwargs.get('DI_HIGH_LIMIT', 30.0)
        self.disaster_stop = 0.05  # 5% hard stop
        logger.info(f"DmiReversalStrategy initialized: Length={self.di_length}, Spread={self.spread_threshold}")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Calculate ADX and Directional Indicators using pandas_ta
        adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=self.di_length)

        # pandas_ta returns a DataFrame with columns: ADX_<length>, DMP_<length>, DMN_<length>
        df['ADX'] = adx_data[f'ADX_{self.di_length}']
        df['DI_Plus'] = adx_data[f'DMP_{self.di_length}']
        df['DI_Minus'] = adx_data[f'DMN_{self.di_length}']

        # Calculate the Spread (Discrepancy)
        df['DI_Spread'] = abs(df['DI_Plus'] - df['DI_Minus'])

        # Track previous +DI for crossover detection
        df['DI_Plus_Prev'] = df['DI_Plus'].shift(1)
        df['DI_Minus_Prev'] = df['DI_Minus'].shift(1)

        return df

    def check_hard_filters(self, row: pd.Series) -> bool:
        try:
            # 1. Bears Exhausted: -DI is very high
            bears_exhausted = row['DI_Minus'] > self.di_high_limit

            # 2. Bulls Beaten: +DI is very low
            bulls_beaten = row['DI_Plus'] < self.di_low_limit

            # 3. The Discrepancy: Massive spread between them
            spread_large = row['DI_Spread'] > self.spread_threshold

            passed = bears_exhausted and bulls_beaten and spread_large

            if passed:
                logger.info(f"✓ DmiReversalStrategy PASSED | +DI={row['DI_Plus']:.1f} -DI={row['DI_Minus']:.1f} Spread={row['DI_Spread']:.1f}")

            return passed
        except Exception as e:
            logger.error(f"Error in DmiReversalStrategy filter: {e}")
            return False

    def check_exit_signal(self, row: pd.Series, entry_price: float = None) -> bool:
        """
        Exit when:
        1. Trend Change: +DI crosses back above -DI (bulls taking control)
        2. Disaster Stop: Price drops 5% below entry
        """
        try:
            # Exit 1: +DI crosses above -DI (trend reversal confirmed)
            if row['DI_Plus'] > row['DI_Minus'] and row.get('DI_Plus_Prev', 0) <= row.get('DI_Minus_Prev', 999):
                logger.info(f"EXIT: +DI crossed above -DI (+DI={row['DI_Plus']:.1f}, -DI={row['DI_Minus']:.1f})")
                return True

            # Exit 2: Disaster Stop
            if entry_price and entry_price > 0:
                stop_price = entry_price * (1 - self.disaster_stop)
                if row['Close'] < stop_price:
                    logger.info(f"EXIT: Disaster Stop Triggered! Price ${row['Close']:.2f} < ${stop_price:.2f}")
                    return True

            return False
        except Exception:
            return False

    def generate_prompt(self, row: pd.Series, lookback_rows: Optional[pd.DataFrame] = None) -> str:
        return f"Strategy: DMI Reversal. Bears exhausted (-DI={row['DI_Minus']:.1f}), Bulls beaten (+DI={row['DI_Plus']:.1f}). Spread={row['DI_Spread']:.1f}. Confirm reversal?"

    def get_position_size(self, price: float, account_value: float) -> int:
        dollar_amount = min(config.POSITION_SIZE, account_value * 0.95)
        return max(1, int(dollar_amount / price))


class BidirectionalStrategy:
    """
    STRATEGY: Bidirectional Trading (Long & Short)

    Trades both directions with separate parameters for each side.
    Uses an ADX regime filter to avoid choppy, directionless markets.

    LONG SIGNAL:
    - Price > EMA 200 (uptrend confirmed)
    - RSI < long_rsi_threshold (oversold dip in an uptrend)

    SHORT SIGNAL:
    - Price < EMA 200 (downtrend confirmed)
    - RSI > short_rsi_threshold (overbought rally in a downtrend)

    REGIME FILTER (ADX):
    - ADX < threshold = choppy market -> NO new entries, only exits.

    FLIP LOGIC:
    - If holding Long and a Short signal fires: Close Long, then Open Short (next bar).
    - If holding Short and a Long signal fires: Close Short, then Open Long (next bar).
    """

    IS_BIDIRECTIONAL = True

    @staticmethod
    def get_optimizable_params():
        return {
            "LONG_RSI_THRESHOLD": {
                "name": "Long RSI Threshold",
                "type": "int",
                "min": 10, "max": 100, "step": 1, "default": 30,
                "optimize_min": 15, "optimize_max": 90, "optimize_step": 5,
                "description": "RSI < value = Oversold (Long entry)",
                "category": "signal"
            },
            "SHORT_RSI_THRESHOLD": {
                "name": "Short RSI Threshold",
                "type": "int",
                "min": 50, "max": 100, "step": 1, "default": 70,
                "optimize_min": 55, "optimize_max": 100, "optimize_step": 5,
                "description": "RSI > value = Overbought (Short entry)",
                "category": "signal"
            },
            "LONG_STOP_LOSS": {
                "name": "Long Stop Loss (%)",
                "type": "float",
                "min": 0.5, "max": 20.0, "step": 0.1, "default": 2.0,
                "optimize_min": 1.0, "optimize_max": 10.0, "optimize_step": 0.5,
                "description": "Stop loss percentage for long positions",
                "category": "long_sl"
            },
            "SHORT_STOP_LOSS": {
                "name": "Short Stop Loss (%)",
                "type": "float",
                "min": 0.5, "max": 20.0, "step": 0.1, "default": 1.5,
                "optimize_min": 0.5, "optimize_max": 8.0, "optimize_step": 0.5,
                "description": "Stop loss percentage for short positions (tighter)",
                "category": "short_sl"
            },
            "ADX_CHOP_THRESHOLD": {
                "name": "ADX Chop Filter",
                "type": "int",
                "min": 10, "max": 70, "step": 1, "default": 20,
                "optimize_min": 15, "optimize_max": 60, "optimize_step": 5,
                "description": "ADX < value = Choppy market (no new entries)",
                "category": "signal_filter"
            }
        }

    @staticmethod
    def generate_vectorized_signals(close, high, low, volume, param_ranges):
        long_rsi_range = param_ranges.get('LONG_RSI_THRESHOLD', np.array([30]))
        short_rsi_range = param_ranges.get('SHORT_RSI_THRESHOLD', np.array([70]))

        ema_200 = ta.ema(close, length=200)
        rsi = ta.rsi(close, length=14)

        grid_long, grid_short = np.meshgrid(long_rsi_range, short_rsi_range)
        flat_long = grid_long.flatten()
        flat_short = grid_short.flatten()

        close_vals = close.values[:, None]
        ema_vals = ema_200.values[:, None]
        rsi_vals = rsi.values[:, None]

        long_entries = (close_vals > ema_vals) & (rsi_vals < flat_long)
        short_entries = (close_vals < ema_vals) & (rsi_vals > flat_short)

        T = len(close)
        N = len(flat_long)
        zeros = np.zeros((T, N), dtype=bool)
        return {
            'long_entries': long_entries,
            'long_exits': zeros,
            'short_entries': short_entries,
            'short_exits': zeros,
            'param_columns': {
                'Long RSI Threshold': flat_long,
                'Short RSI Threshold': flat_short,
            },
        }

    def __init__(self, **kwargs):
        self.ema_period = kwargs.get('EMA_PERIOD', 200)
        self.rsi_period = kwargs.get('RSI_PERIOD', getattr(config, 'RSI_PERIOD', 14))
        self.long_rsi_threshold = kwargs.get('LONG_RSI_THRESHOLD', getattr(config, 'LONG_RSI_THRESHOLD', 30))
        self.short_rsi_threshold = kwargs.get('SHORT_RSI_THRESHOLD', getattr(config, 'SHORT_RSI_THRESHOLD', 70))
        self.long_stop_loss = kwargs.get('LONG_STOP_LOSS', getattr(config, 'LONG_STOP_LOSS', 2.0))
        self.short_stop_loss = kwargs.get('SHORT_STOP_LOSS', getattr(config, 'SHORT_STOP_LOSS', 1.5))
        self.adx_chop_threshold = kwargs.get('ADX_CHOP_THRESHOLD', getattr(config, 'ADX_CHOP_THRESHOLD', 20))
        logger.info(
            f"BidirectionalStrategy initialized: "
            f"Long RSI<{self.long_rsi_threshold}, Short RSI>{self.short_rsi_threshold}, "
            f"Long SL={self.long_stop_loss}%, Short SL={self.short_stop_loss}%, "
            f"ADX Chop<{self.adx_chop_threshold}"
        )

    @staticmethod
    def get_chart_config():
        return [
            {"name": "EMA 200", "column": "EMA_200", "color": "cyan", "panel": 1, "width": 2},
            {"name": "RSI", "column": "RSI", "color": "purple", "panel": 2, "width": 2, "hlines": [30, 70]},
            {"name": "ADX", "column": "ADX", "color": "orange", "panel": 3, "width": 2, "hlines": [20]},
        ]

    @staticmethod
    def is_choppy(row):
        """Check if market is in a choppy regime based on ADX."""
        adx = row.get('ADX', 0)
        if pd.isna(adx):
            return True
        return adx < 20

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # EMA for trend direction
        df['EMA_200'] = ta.ema(df['Close'], length=self.ema_period)

        # RSI for overbought/oversold
        df['RSI'] = ta.rsi(df['Close'], length=self.rsi_period)

        # ADX for regime filter (trend strength)
        adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        df['ADX'] = adx_data['ADX_14']
        df['DI_Plus'] = adx_data['DMP_14']
        df['DI_Minus'] = adx_data['DMN_14']

        # Volume context
        df['Volume_SMA_20'] = ta.sma(df['Volume'], length=20)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']

        return df

    def get_signal(self, row) -> int:
        """
        Determine trade direction based on indicators.

        Returns:
             1 = Long signal
            -1 = Short signal
             0 = No signal (flat / choppy)
        """
        try:
            price = row['Close']
            ema_200 = row['EMA_200']
            rsi = row['RSI']
            adx = row['ADX']

            if pd.isna(ema_200) or pd.isna(rsi) or pd.isna(adx):
                return 0

            # REGIME FILTER: Choppy market = no new entries
            if adx < self.adx_chop_threshold:
                return 0

            # LONG: Uptrend + oversold dip
            if price > ema_200 and rsi < self.long_rsi_threshold:
                logger.info(
                    f"✓ LONG Signal: Price ${price:.2f} > EMA200 ${ema_200:.2f}, "
                    f"RSI {rsi:.1f} < {self.long_rsi_threshold}, ADX {adx:.1f}"
                )
                return 1

            # SHORT: Downtrend + overbought rally
            if price < ema_200 and rsi > self.short_rsi_threshold:
                logger.info(
                    f"SHORT Signal: Price ${price:.2f} < EMA200 ${ema_200:.2f}, "
                    f"RSI {rsi:.1f} > {self.short_rsi_threshold}, ADX {adx:.1f}"
                )
                return -1

            return 0

        except Exception:
            return 0

    def check_hard_filters(self, row: pd.Series) -> bool:
        """Compatibility wrapper: returns True if any signal (long or short)."""
        return self.get_signal(row) != 0

    def check_exit_signal(self, row: pd.Series, entry_price: float = None) -> bool:
        """Exit when the market regime becomes choppy (ADX collapse)."""
        try:
            adx = row['ADX']
            if pd.isna(adx):
                return False
            if adx < self.adx_chop_threshold:
                logger.info(f"EXIT: ADX {adx:.1f} < {self.adx_chop_threshold} (Choppy regime)")
                return True
            return False
        except Exception:
            return False

    def generate_prompt(self, row: pd.Series, lookback_rows: Optional[pd.DataFrame] = None) -> str:
        signal = self.get_signal(row)
        direction = "LONG" if signal == 1 else "SHORT" if signal == -1 else "FLAT"
        price = row['Close']
        ema = row['EMA_200']
        rsi = row['RSI']
        adx = row['ADX']

        prompt = f"""Symbol: {config.TICKER}
Current Price: ${price:.2f}
Strategy: Bidirectional (Long & Short)
Proposed Direction: {direction}

Technical Context:
- EMA 200: ${ema:.2f} (Price is {'ABOVE' if price > ema else 'BELOW'} trend)
- RSI: {rsi:.1f}
- ADX: {adx:.1f} (Trend strength: {'Strong' if adx > 25 else 'Moderate' if adx > 20 else 'Weak'})

Your Job:
Confirm or reject this {direction} entry.
- For LONG: Is the dip likely to bounce?
- For SHORT: Is the rally likely to fade?
- ADX confirms a directional trend exists.

Reply with BUY to confirm, HOLD to reject.
"""
        return prompt.strip()

    def get_position_size(self, price: float, account_value: float) -> int:
        dollar_amount = min(config.POSITION_SIZE, account_value * 0.95)
        return max(1, int(dollar_amount / price))


# Standalone test
if __name__ == "__main__":
    import yfinance as yf

    print("Testing HybridStrategy...")
    print("=" * 60)

    # Download sample data
    print(f"Downloading {config.TICKER} data...")
    df = yf.download(config.TICKER, period="1y", interval="1h", progress=False)

    # Initialize strategy
    strategy = HybridStrategy()

    # Calculate indicators
    df = strategy.calculate_indicators(df)

    # Test on recent bars
    print(f"\nTesting hard filters on last 10 bars:\n")

    for i in range(-10, 0):
        row = df.iloc[i]
        timestamp = df.index[i]
        passed = strategy.check_hard_filters(row)

        status = "PASS" if passed else "FAIL"
        print(f"{timestamp.strftime('%Y-%m-%d %H:%M')} | {status}")

        if passed:
            print("\nGenerating AI prompt for this setup:")
            print("-" * 60)
            prompt = strategy.generate_prompt(row, df.iloc[i-5:i])
            print(prompt)
            print("-" * 60)
            break

    print("\n" + "=" * 60)
    print("Strategy test complete!")
