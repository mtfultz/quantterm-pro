import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import vectorbt as vbt
import pandas_ta_classic as ta
import config
from backtest_runner import run_backtest
from strategy import (
    HybridStrategy,
    BreakoutStrategy,
    MeanReversionStrategy,
    SmartTrendStrategy,
    VolatilitySqueezeStrategy,
    MacdScalpStrategy,
    MacroReversalStrategy,
    DmiReversalStrategy,
    BidirectionalStrategy
)
import requests
import alpaca_trade_api as tradeapi
from datetime import datetime
from joblib import Parallel, delayed
import multiprocessing
import importlib
import itertools
import inspect
import time
from portfolio_manager import PortfolioOptimizer
from options_manager import OptionsManager
import matplotlib.pyplot as plt

# Import professional UI components
from ui_components import (
    metric_card,
    metric_row,
    status_badge,
    render_status_badge,
    section_card_start,
    section_card_end,
    labeled_divider,
    page_header,
    get_nav_menu_styles,
    get_plotly_layout,
    get_chart_colors,
    inject_custom_css,
    COLORS,
)

# Import custom strategies module
try:
    import custom_strategies
    # Force reload to pick up any new strategies added during session
    importlib.reload(custom_strategies)
except (ImportError, SyntaxError, Exception) as e:
    custom_strategies = None
    custom_strategies_error = str(e)

# Import Macro Intelligence module
try:
    from macro_intelligence import (
        render_macro_page, get_macro_data, classify_macro_regime,
        align_macro_to_prices, SAFE_HAVENS
    )
    MACRO_AVAILABLE = True
except ImportError:
    MACRO_AVAILABLE = False


# Strategy Registry
def sanitize_strategy_name(name: str) -> str:
    """
    Convert user input into a valid Python class name.

    Rules:
    - Remove all non-alphanumeric characters except spaces
    - Convert to title case (capitalize each word)
    - Remove spaces
    - Ensure it starts with a letter (prepend 'Strategy' if needed)
    - Append 'Strategy' suffix

    Examples:
        "ema + rsi" -> "EmaRsiStrategy"
        "123 momentum" -> "Strategy123MomentumStrategy"
        "My-Cool_Strategy!" -> "MyCoolStrategyStrategy"
    """
    import re

    # Remove all non-alphanumeric characters except spaces
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', name)

    # Split on spaces, capitalize each word, and join
    words = clean.split()
    class_name = ''.join(word.capitalize() for word in words)

    # Ensure it starts with a letter
    if class_name and not class_name[0].isalpha():
        class_name = 'Strategy' + class_name

    # Append 'Strategy' if not already present
    if not class_name.endswith('Strategy'):
        class_name += 'Strategy'

    # Handle empty input
    if not class_name or class_name == 'Strategy':
        class_name = 'CustomStrategy'

    return class_name


def load_custom_strategies():
    """
    Dynamically load custom strategies from custom_strategies.py
    Returns dict of strategy_name -> {class, description}
    """
    custom_strats = {}

    if custom_strategies is None:
        return custom_strats

    try:
        # Force reload to pick up new strategies
        importlib.reload(custom_strategies)

        # Inspect the custom_strategies module for classes ending with "Strategy"
        for name, obj in inspect.getmembers(custom_strategies):
            if inspect.isclass(obj) and name.endswith("Strategy"):
                # Skip if the class is imported from another module
                if obj.__module__ != "custom_strategies":
                    continue

                # Get docstring for description
                description = obj.__doc__.split('\n')[0] if obj.__doc__ else "Custom strategy"

                # Format the display name (remove "Strategy" suffix, add spaces)
                display_name = name.replace("Strategy", "")
                # Add spaces before capital letters
                import re
                display_name = re.sub(r'([A-Z])', r' \1', display_name).strip()
                display_name = f"🎨 {display_name} (Custom)"

                custom_strats[display_name] = {
                    "class": obj,
                    "description": description,
                    "class_name": name  # Store original class name for deletion
                }

    except Exception as e:
        # Silently fail if reload fails
        pass

    return custom_strats


def delete_custom_strategy(class_name: str) -> bool:
    """
    Delete a strategy class from custom_strategies.py

    Args:
        class_name: The exact class name (e.g., "SuperRsiStrategy")

    Returns:
        bool: True if successful, False otherwise
    """
    import os

    custom_file = "custom_strategies.py"

    if not os.path.exists(custom_file):
        return False

    try:
        # Read the file
        with open(custom_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Find the class definition
        class_start = None
        class_end = None
        class_indent = None

        for i, line in enumerate(lines):
            # Look for the class definition
            if f"class {class_name}:" in line or f"class {class_name}(" in line:
                class_start = i
                # Determine the indentation of this class
                class_indent = len(line) - len(line.lstrip())
                continue

            # If we found the class, look for where it ends
            if class_start is not None and class_indent is not None:
                # Class ends when we find another class at same or lower indentation
                # or a non-empty line at the same indentation that isn't part of the class
                stripped = line.lstrip()
                if stripped and not line.startswith(' ' * (class_indent + 1)):
                    # Check if this is another class or end of file
                    if stripped.startswith('class ') or i == len(lines) - 1:
                        class_end = i
                        break

        # If we reached end of file and still in class
        if class_start is not None and class_end is None:
            class_end = len(lines)

        # If class not found
        if class_start is None:
            return False

        # Remove the class (including empty lines immediately after)
        # But preserve the header comment
        new_lines = lines[:class_start]

        # Skip to next non-empty line or next class
        skip_to = class_end
        while skip_to < len(lines) and lines[skip_to].strip() == '':
            skip_to += 1

        new_lines.extend(lines[skip_to:])

        # Write back
        with open(custom_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

        return True

    except Exception as e:
        st.error(f"Error deleting strategy: {e}")
        return False


# Define core strategy names and their corresponding classes
CORE_STRATEGIES = {
    "Smart Trend (EMA + ADX)": {
        "class": SmartTrendStrategy,
        "description": "Follows Golden Cross (EMA50 > EMA200) with ADX > 25. Trailing stops on EMA50."
    },
    "Mean Reversion (Buy the Fear)": {
        "class": MeanReversionStrategy,
        "description": "Buys when price drops below Lower Bollinger Band + RSI oversold"
    },
    "Volatility Squeeze (Breakouts)": {
        "class": VolatilitySqueezeStrategy,
        "description": "Trades explosive breakouts from Bollinger Band compression inside Keltner Channels"
    },
    "MACD Scalping (Fast Momentum)": {
        "class": MacdScalpStrategy,
        "description": "Catches MACD crossovers with StochRSI filter for quick momentum trades"
    },
    "Hybrid (Dip Buyer - Original)": {
        "class": HybridStrategy,
        "description": "Original strategy: Price > SMA 200 + RSI < 45 oversold pullback"
    },
    "Breakout (Bollinger + Volume)": {
        "class": BreakoutStrategy,
        "description": "Price breaks above Upper Bollinger Band with 1.5x+ volume confirmation"
    },
    "Macro Reversal (Bottom Buyer)": {
        "class": MacroReversalStrategy,
        "description": "Catches major reversals after crashes. Buys confirmed bottoms and holds for long swings."
    },
    "DMI Reversal (Discrepancy Hunter)": {
        "class": DmiReversalStrategy,
        "description": "Trades the spread when bears are exhausted (-DI high) but bulls haven't arrived (+DI low)."
    },
    "Bidirectional (Long & Short)": {
        "class": BidirectionalStrategy,
        "description": "Trades both directions with ADX regime filter. Longs oversold dips in uptrends, shorts overbought rallies in downtrends."
    }
}

# Merge core strategies with custom strategies
STRATEGY_REGISTRY = {**CORE_STRATEGIES, **load_custom_strategies()}


def get_strategy_params(strategy_class):
    """
    Dynamically fetch optimizable parameters from a strategy class.

    Returns:
        Dict of parameters with their metadata
    """
    if hasattr(strategy_class, 'get_optimizable_params'):
        return strategy_class.get_optimizable_params()
    return {}


def get_primary_param(strategy_class):
    """
    Get the primary (first) optimizable parameter for a strategy.
    Used for backward compatibility with single-parameter optimization.

    Returns:
        Tuple of (param_name, param_config) or (None, None)
    """
    params = get_strategy_params(strategy_class)
    if params:
        # Return first parameter as primary
        param_name = list(params.keys())[0]
        return param_name, params[param_name]
    return None, None

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    layout="wide",
    page_title="QuantTerm Pro - Professional Trading Terminal",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS INJECTION - LOAD PROFESSIONAL THEME
# ============================================================================
def load_css():
    """Load and inject custom CSS from assets/styles.css"""
    # Load Google Fonts via @import (works in Streamlit)
    fonts = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    </style>
    """
    st.markdown(fonts, unsafe_allow_html=True)

    try:
        with open("assets/styles.css", "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Custom CSS file not found. Using default styling.")

# Load the professional theme
load_css()

# ============================================================================
# TICKER PRESETS
# ============================================================================
TICKER_CATEGORIES = {
    "Indices": ["SPY", "QQQ", "IWM", "DIA"],
    "Tech": ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "GOOGL", "META", "AMZN"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "ADA-USD", "MATIC-USD"]
}

# Flatten for selectbox
ALL_TICKERS = []
for category, tickers in TICKER_CATEGORIES.items():
    ALL_TICKERS.extend(tickers)

# ============================================================================
# PARALLEL PROCESSING WRAPPER FOR GRID SEARCH
# ============================================================================

def run_optimization_case(params):
    """
    Standalone function for parallel grid search optimization.

    Args:
        params: Dict with keys:
            - 'ticker': Symbol to trade
            - 'period': Historical period
            - 'interval': Timeframe
            - 'strategy_class': Strategy class to use
            - 'sl': Stop loss percentage (decimal)
            - 'tp': Take profit percentage (decimal)
            - 'strategy_params': Dict mapping config param names to values
            - 'strategy_param_labels': Dict mapping config param names to display labels

    Returns:
        Dict with backtest results + parameters, or None if failed
    """
    try:
        # Import config inside the worker to ensure thread safety
        import config

        # Override config with parameters
        config.TICKER = params['ticker']
        config.TIMEFRAME = params['interval']
        config.STOP_LOSS_PCT = params['sl']
        config.TAKE_PROFIT_PCT = params['tp']

        # Dynamically set ALL strategy-specific parameters
        for param_name, param_value in params['strategy_params'].items():
            setattr(config, param_name, param_value)

        # Run backtest
        stats = run_backtest(
            ticker=params['ticker'],
            period=params['period'],
            interval=params['interval'],
            strategy_class=params['strategy_class'],
            use_real_ai=False,  # Always use mock AI for speed
            plot=False
        )

        # Check if backtest generated valid results
        if stats is not None and stats.get('# Trades', 0) > 0:
            result = {
                'Stop Loss (%)': params['sl'] * 100,
                'Take Profit (%)': params['tp'] * 100,
                'Return (%)': stats['Return [%]'],
                'Win Rate (%)': stats['Win Rate [%]'],
                'Sharpe Ratio': stats['Sharpe Ratio'],
                'Max Drawdown (%)': stats['Max. Drawdown [%]'],
                'Total Trades': stats['# Trades']
            }
            # Add all strategy-specific parameters to results
            for p_name, p_value in params['strategy_params'].items():
                label = params['strategy_param_labels'].get(p_name, p_name)
                result[label] = p_value
            return result
        else:
            # No trades generated - return safe default
            return None

    except Exception as e:
        # If backtest fails, return None (will be filtered out)
        return None


# Helper functions

def create_price_chart(df_plot, trades, equity_series, ticker):
    """Create interactive Plotly chart with price action and equity."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{ticker} Price Action', 'Account Value'),
        row_heights=[0.7, 0.3]
    )

    # Layer 1: Candlesticks
    fig.add_trace(go.Candlestick(
        x=df_plot['Date'],
        open=df_plot['Open'],
        high=df_plot['High'],
        low=df_plot['Low'],
        close=df_plot['Close'],
        name='Price',
        increasing_line_color='#00e676',
        decreasing_line_color='#ff1744'
    ), row=1, col=1)

    # Layer 2: Trade Markers
    if not trades.empty:
        fig.add_trace(go.Scatter(
            x=trades['EntryTime'],
            y=trades['EntryPrice'],
            mode='markers',
            marker=dict(symbol='triangle-up', color='#00e676', size=12, line=dict(color='white', width=1)),
            name='Buy Entry',
            hovertemplate='<b>BUY</b><br>Price: $%{y:.2f}<br>%{x}<extra></extra>'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=trades['ExitTime'],
            y=trades['ExitPrice'],
            mode='markers',
            marker=dict(symbol='triangle-down', color='#ff1744', size=12, line=dict(color='white', width=1)),
            name='Sell/Exit',
            hovertemplate='<b>SELL</b><br>Price: $%{y:.2f}<br>%{x}<extra></extra>'
        ), row=1, col=1)

    # Layer 3: Equity Curve
    fig.add_trace(go.Scatter(
        x=equity_series.index,
        y=equity_series.values,
        line=dict(color='#2962ff', width=2),
        fill='tozeroy',
        fillcolor='rgba(41, 98, 255, 0.2)',
        name='Account Value'
    ), row=2, col=1)

    # Styling
    fig.update_layout(
        template="plotly_dark",
        height=700,
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#121212",
        plot_bgcolor="#1E1E1E",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)')

    return fig


def generate_multi_asset_signals(
    price_data, tickers, strategy_class, signal_params,
    filter_params=None, progress_callback=None, macro_regime=None,
    safe_haven_action='cash',
):
    """Generate trading signals for multiple tickers by iterating per-ticker.

    pandas_ta functions expect 1D Series, so we loop over each ticker column,
    generate signals individually, and combine into aligned DataFrames.

    Args:
        macro_regime: Optional Series of bool (True = Risk Off). When provided,
            suppresses equity long entries and forces long exits for non-safe-haven
            assets during Risk Off periods. GLD/TLT are unaffected.
        safe_haven_action: 'cash' (default), 'TLT', or 'GLD'. During Risk Off,
            if a safe haven ticker is specified and present in the portfolio,
            force long entries for that asset as a defensive rotation.

    Returns:
        dict with long_entries, short_entries, long_exits, short_exits (DataFrames),
        skipped_entries (DataFrame of trades blocked by macro filter),
        and failed_tickers (list).
    """
    long_entries_dict = {}
    short_entries_dict = {}
    long_exits_dict = {}
    short_exits_dict = {}
    skipped_entries_dict = {}
    failed_tickers = []

    close_df = price_data.get('Close')
    high_df = price_data.get('High')
    low_df = price_data.get('Low')
    volume_df = price_data.get('Volume')

    # Determine safe haven set
    safe_havens = SAFE_HAVENS if MACRO_AVAILABLE else {'GLD', 'TLT'}

    # Determine safe haven rotation target (must be in the portfolio)
    rotation_target = None
    if safe_haven_action in ('TLT', 'GLD') and safe_haven_action in tickers:
        rotation_target = safe_haven_action

    for i, tkr in enumerate(tickers):
        try:
            close = close_df[tkr]
            high = high_df[tkr]
            low = low_df[tkr]
            volume = volume_df[tkr]

            sig = strategy_class.generate_vectorized_signals(
                close, high, low, volume, signal_params
            )

            le = sig['long_entries'].squeeze()
            se = sig['short_entries'].squeeze()
            lx = sig['long_exits'].squeeze()
            sx = sig['short_exits'].squeeze()

            # Apply signal_filter (ADX mask) if applicable
            if filter_params:
                adx_data = ta.adx(high, low, close, length=14)
                adx_series = adx_data['ADX_14']
                for fname, fval in filter_params.items():
                    adx_mask = adx_series.values >= fval
                    le = le & adx_mask
                    se = se & adx_mask

            # Track pre-macro entries for "skipped trades" visualization
            le_before_macro = le.copy() if macro_regime is not None else None

            # Apply macro regime filter (suppress equity longs during Risk Off)
            if macro_regime is not None and tkr.upper() not in safe_havens:
                # Align macro regime (daily) to price index (may be hourly/etc.)
                risk_off = macro_regime.reindex(close.index, method='ffill').fillna(False).values
                risk_on = ~risk_off
                le = le & risk_on        # Suppress long entries during Risk Off
                lx = lx | risk_off       # Force long exits when Risk Off starts

            # Safe haven rotation: force long entries for rotation target during Risk Off
            if macro_regime is not None and rotation_target is not None and tkr == rotation_target:
                risk_off = macro_regime.reindex(close.index, method='ffill').fillna(False).values
                le = le | risk_off       # Force long entries during Risk Off
                lx = lx & ~risk_off      # Don't exit haven during Risk Off

            # Record skipped entries (blocked by macro filter)
            if le_before_macro is not None and tkr.upper() not in safe_havens:
                skipped = le_before_macro & ~le
                skipped_entries_dict[tkr] = pd.Series(skipped, index=close.index)
            else:
                skipped_entries_dict[tkr] = pd.Series(False, index=close.index)

            long_entries_dict[tkr] = pd.Series(le, index=close.index)
            short_entries_dict[tkr] = pd.Series(se, index=close.index)
            long_exits_dict[tkr] = pd.Series(lx, index=close.index)
            short_exits_dict[tkr] = pd.Series(sx, index=close.index)

        except Exception:
            failed_tickers.append(tkr)

        if progress_callback:
            progress_callback((i + 1) / len(tickers))

    return {
        'long_entries': pd.DataFrame(long_entries_dict).fillna(False),
        'short_entries': pd.DataFrame(short_entries_dict).fillna(False),
        'long_exits': pd.DataFrame(long_exits_dict).fillna(False),
        'short_exits': pd.DataFrame(short_exits_dict).fillna(False),
        'skipped_entries': pd.DataFrame(skipped_entries_dict).fillna(False),
        'failed_tickers': failed_tickers,
    }


def apply_ai_filter(long_entries, short_entries, close, high, low, volume,
                    selected_strategy_class, param_values, confidence_threshold,
                    open_prices=None, progress_callback=None):
    """
    Apply AI confirmation as a soft filter on vectorbt entry signals.
    For each True entry, consults AIBrain (Mixtral via Ollama) and removes
    entries the AI rejects.

    Args:
        long_entries: 1D boolean array of long entry signals
        short_entries: 1D boolean array of short entry signals
        close, high, low, volume: Price series with DatetimeIndex
        selected_strategy_class: Strategy class to instantiate
        param_values: Dict of strategy parameter values from sidebar
        confidence_threshold: Minimum AI confidence to keep a signal
        open_prices: Optional Open price series (falls back to close)
        progress_callback: Optional callable(float) for progress updates

    Returns:
        (filtered_long_entries, filtered_short_entries, ai_log)
    """
    from ai_brain import AIBrain

    # Build DataFrame for indicator calculation
    open_col = open_prices if open_prices is not None else close
    df = pd.DataFrame({
        'Open': open_col.values,
        'High': high.values,
        'Low': low.values,
        'Close': close.values,
        'Volume': volume.values,
    }, index=close.index)

    # Instantiate strategy with current params and calculate indicators
    strategy_instance = selected_strategy_class(**param_values)
    df_ind = strategy_instance.calculate_indicators(df)

    # Initialize AI brain (single instance for all calls)
    ai_brain = AIBrain()

    # Find entry indices
    long_entry_indices = np.where(long_entries)[0]
    short_entry_indices = np.where(short_entries)[0]
    total_entries = len(long_entry_indices) + len(short_entry_indices)

    ai_log = []
    filtered_long = long_entries.copy()
    filtered_short = short_entries.copy()

    processed = 0

    # Filter long entries through AI
    for idx in long_entry_indices:
        row = df_ind.iloc[idx]
        lookback = df_ind.iloc[max(0, idx - 10):idx]
        prompt = strategy_instance.generate_prompt(row, lookback)
        decision = ai_brain.analyze_market(prompt)

        keep = (decision['signal'] == 'BUY' and
                decision['confidence'] >= confidence_threshold)
        if not keep:
            filtered_long[idx] = False

        ai_log.append({
            'Timestamp': close.index[idx],
            'Direction': 'LONG',
            'Price': f"${row['Close']:.2f}",
            'AI Signal': decision['signal'],
            'Confidence': f"{decision['confidence']:.0%}",
            'Reasoning': decision['reasoning'],
            'Kept': 'Yes' if keep else 'No',
        })

        processed += 1
        if progress_callback and total_entries > 0:
            progress_callback(processed / total_entries)

    # Filter short entries through AI
    for idx in short_entry_indices:
        row = df_ind.iloc[idx]
        lookback = df_ind.iloc[max(0, idx - 10):idx]
        prompt = strategy_instance.generate_prompt(row, lookback)
        decision = ai_brain.analyze_market(prompt)

        keep = (decision['signal'] == 'BUY' and
                decision['confidence'] >= confidence_threshold)
        if not keep:
            filtered_short[idx] = False

        ai_log.append({
            'Timestamp': close.index[idx],
            'Direction': 'SHORT',
            'Price': f"${row['Close']:.2f}",
            'AI Signal': decision['signal'],
            'Confidence': f"{decision['confidence']:.0%}",
            'Reasoning': decision['reasoning'],
            'Kept': 'Yes' if keep else 'No',
        })

        processed += 1
        if progress_callback and total_entries > 0:
            progress_callback(processed / total_entries)

    return filtered_long, filtered_short, ai_log


def create_multi_asset_equity_chart(combined_equity, per_asset_equity, benchmark_equity,
                                    top_n=10, optimized_equity=None, drifting_equity=None):
    """Create Plotly chart: combined portfolio equity vs benchmark + top individual assets."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Portfolio Equity vs Benchmark', 'Individual Asset Performance (%)'),
        row_heights=[0.6, 0.4]
    )

    # Row 1: Combined portfolio equity (blue, bold)
    fig.add_trace(go.Scatter(
        x=combined_equity.index, y=combined_equity.values,
        line=dict(color='#2962ff', width=3),
        name='Portfolio (Equal Weight)',
        fill='tozeroy', fillcolor='rgba(41, 98, 255, 0.15)'
    ), row=1, col=1)

    # Benchmark (gray, dashed)
    if benchmark_equity is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_equity.index, y=benchmark_equity.values,
            line=dict(color='#6E7681', width=2, dash='dash'),
            name='SPY Benchmark'
        ), row=1, col=1)

    # Optimized portfolio (orange, dotted) — rebalanced at selected frequency
    if optimized_equity is not None:
        fig.add_trace(go.Scatter(
            x=optimized_equity.index, y=optimized_equity.values,
            line=dict(color='#ff6d00', width=2, dash='dot'),
            name='Optimized (Rebalanced)'
        ), row=1, col=1)

    # Drifting portfolio (purple, dash-dot) — same weights, never rebalanced
    if drifting_equity is not None:
        fig.add_trace(go.Scatter(
            x=drifting_equity.index, y=drifting_equity.values,
            line=dict(color='#ab47bc', width=1.5, dash='dashdot'),
            name='Optimized (Drifting)'
        ), row=1, col=1)

    # Row 2: Top N individual tickers by final return
    if per_asset_equity is not None and not per_asset_equity.empty:
        final_returns = (per_asset_equity.iloc[-1] / per_asset_equity.iloc[0] - 1).sort_values(ascending=False)
        top_tickers = final_returns.head(top_n).index.tolist()

        colors = ['#00e676', '#ff6d00', '#aa00ff', '#00b0ff', '#ffea00',
                  '#ff1744', '#64ffda', '#f50057', '#76ff03', '#e040fb']

        for i, tkr in enumerate(top_tickers):
            normalized = (per_asset_equity[tkr] / per_asset_equity[tkr].iloc[0] - 1) * 100
            fig.add_trace(go.Scatter(
                x=normalized.index, y=normalized.values,
                line=dict(color=colors[i % len(colors)], width=1.5),
                name=tkr, opacity=0.8
            ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark", height=800,
        paper_bgcolor="#121212", plot_bgcolor="#1E1E1E",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)')

    return fig


def create_correlation_heatmap(returns_df):
    """Create Plotly correlation heatmap from per-asset returns DataFrame."""
    corr = returns_df.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale='RdBu_r',
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{x} vs %{y}: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        template="plotly_dark",
        height=max(400, len(corr) * 25 + 200),
        paper_bgcolor="#121212", plot_bgcolor="#1E1E1E",
        title="Return Correlation Matrix",
        xaxis=dict(tickangle=45),
    )

    return fig


def compute_markowitz_optimization(per_asset_returns, close_df, risk_free_rate=0.02, rebalance_freq='Monthly'):
    """Compute Markowitz max-Sharpe weights from strategy returns with periodic rebalancing.

    Args:
        per_asset_returns: DataFrame of daily strategy returns per asset
        close_df: DataFrame of close prices (for expected returns & covariance)
        risk_free_rate: Risk-free rate for Sharpe calculation
        rebalance_freq: 'Never', 'Monthly', 'Quarterly', 'Yearly'

    Returns:
        dict with weights, performance metrics, rebalanced/drifting equity, and frontier data
    """
    from pypfopt import EfficientFrontier, risk_models, expected_returns

    # Use close prices for expected returns & covariance (standard for Markowitz)
    mu = expected_returns.mean_historical_return(close_df)
    S = risk_models.sample_cov(close_df)

    ef = EfficientFrontier(mu, S)
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    weights = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)

    # Filter zero-weight tickers
    weights_nonzero = {k: v for k, v in weights.items() if v > 0.0001}

    # Compute rebalanced and drifting portfolio equity
    rebal_result = compute_rebalanced_equity(
        per_asset_returns, weights, rebalance_freq=rebalance_freq, fees=0.001
    )

    # The "optimized" equity is the rebalanced version
    optimized_returns = rebal_result['rebalanced_returns']
    optimized_equity = rebal_result['rebalanced_equity']

    # Efficient frontier simulation for Plotly chart
    n_assets = len(close_df.columns)
    n_sims = 5000
    sim_returns = np.zeros(n_sims)
    sim_vols = np.zeros(n_sims)
    sim_sharpes = np.zeros(n_sims)
    for i in range(n_sims):
        w = np.random.dirichlet(np.ones(n_assets))
        r = float(np.dot(w, mu))
        v = float(np.sqrt(np.dot(w.T, np.dot(S, w))))
        sim_returns[i] = r * 100
        sim_vols[i] = v * 100
        sim_sharpes[i] = r / v if v > 0 else 0

    # Per asset volatilities for individual asset plotting
    asset_vols = np.sqrt(np.diag(S)) * 100
    asset_rets = mu.values * 100

    return {
        'weights': weights,
        'weights_nonzero': weights_nonzero,
        'expected_return': perf[0],
        'annual_volatility': perf[1],
        'sharpe_ratio': perf[2],
        'optimized_returns': optimized_returns,
        'optimized_equity': optimized_equity,
        'drifting_equity': rebal_result['drifting_equity'],
        'drifting_returns': rebal_result['drifting_returns'],
        'total_turnover': rebal_result['total_turnover'],
        'num_rebalances': rebal_result['num_rebalances'],
        'rebalance_freq': rebalance_freq,
        'frontier_returns': sim_returns,
        'frontier_vols': sim_vols,
        'frontier_sharpes': sim_sharpes,
        'mu': mu,
        'S': S,
        'asset_vols': asset_vols,
        'asset_rets': asset_rets,
    }


def compute_hrp_optimization(per_asset_returns, close_df, rebalance_freq='Monthly'):
    """Compute Hierarchical Risk Parity portfolio weights.

    HRP clusters assets by correlation structure and allocates inversely
    to cluster variance — no expected returns needed, more robust than MVO.

    Args:
        per_asset_returns: DataFrame of daily strategy returns per asset
        close_df: DataFrame of close prices
        rebalance_freq: 'Never', 'Monthly', 'Quarterly', 'Yearly'

    Returns:
        dict with weights, equity curves, and clustering data
    """
    from pypfopt import HRPOpt

    returns_df = close_df.pct_change().dropna()
    hrp = HRPOpt(returns_df)
    hrp.optimize()
    weights = hrp.clean_weights()
    perf = hrp.portfolio_performance(verbose=False)

    weights_nonzero = {k: v for k, v in weights.items() if v > 0.0001}

    rebal_result = compute_rebalanced_equity(
        per_asset_returns, weights, rebalance_freq=rebalance_freq, fees=0.001
    )

    optimized_equity = rebal_result['rebalanced_equity']
    opt_rets = rebal_result['rebalanced_returns']

    return {
        'weights': weights,
        'weights_nonzero': weights_nonzero,
        'expected_return': perf[0],
        'annual_volatility': perf[1],
        'sharpe_ratio': perf[2],
        'optimized_returns': opt_rets,
        'optimized_equity': optimized_equity,
        'drifting_equity': rebal_result['drifting_equity'],
        'drifting_returns': rebal_result['drifting_returns'],
        'total_turnover': rebal_result['total_turnover'],
        'num_rebalances': rebal_result['num_rebalances'],
        'rebalance_freq': rebalance_freq,
        'returns_df': returns_df,
    }


def compute_rebalanced_equity(per_asset_returns, weights, rebalance_freq='Monthly', fees=0.001):
    """Simulate portfolio with periodic rebalancing and drifting weights.

    Between rebalance dates, each asset compounds independently (weights drift).
    At rebalance dates, portfolio resets to target weights with transaction fees.

    Args:
        per_asset_returns: DataFrame of daily strategy returns per asset
        weights: dict of {ticker: weight} from Markowitz optimization
        rebalance_freq: 'Never', 'Monthly', 'Quarterly', 'Yearly'
        fees: transaction fee rate per trade (0.001 = 0.1%)

    Returns:
        dict with rebalanced/drifting equity and returns, turnover, rebalance count
    """
    columns = per_asset_returns.columns
    dates = per_asset_returns.index
    returns_arr = per_asset_returns.values  # (T, N)
    target_weights = np.array([weights.get(c, 0) for c in columns], dtype=float)
    T = len(dates)

    # Identify rebalance dates
    if rebalance_freq == 'Never':
        rebal_mask = np.zeros(T, dtype=bool)
    else:
        date_series = dates.to_series()
        if rebalance_freq == 'Monthly':
            periods = date_series.dt.to_period('M')
        elif rebalance_freq == 'Quarterly':
            periods = date_series.dt.to_period('Q')
        elif rebalance_freq == 'Yearly':
            periods = date_series.dt.to_period('Y')
        else:
            periods = date_series.dt.to_period('M')
        # First trading day of each period
        first_of_period = ~periods.duplicated(keep='first')
        rebal_mask = first_of_period.values
        # Don't rebalance on the very first day (initial allocation)
        rebal_mask[0] = False

    # Simulate rebalanced portfolio
    rebal_holdings = target_weights * 10000.0
    rebal_equity = np.zeros(T)
    total_turnover = 0.0
    num_rebalances = 0

    for t in range(T):
        # Apply daily returns
        rebal_holdings = rebal_holdings * (1 + returns_arr[t])
        portfolio_value = rebal_holdings.sum()
        rebal_equity[t] = portfolio_value

        # Rebalance if needed
        if rebal_mask[t] and portfolio_value > 0:
            current_weights = rebal_holdings / portfolio_value
            turnover = np.abs(current_weights - target_weights).sum() / 2
            total_turnover += turnover
            num_rebalances += 1
            # Deduct fees (buy + sell sides)
            fee_cost = turnover * portfolio_value * fees * 2
            portfolio_value -= fee_cost
            rebal_holdings = target_weights * portfolio_value
            rebal_equity[t] = portfolio_value

    # Simulate drifting portfolio (never rebalance)
    drift_holdings = target_weights * 10000.0
    drift_equity = np.zeros(T)

    for t in range(T):
        drift_holdings = drift_holdings * (1 + returns_arr[t])
        drift_equity[t] = drift_holdings.sum()

    # Convert to Series
    rebal_equity_series = pd.Series(rebal_equity, index=dates)
    drift_equity_series = pd.Series(drift_equity, index=dates)

    # Compute return series
    rebal_returns = rebal_equity_series.pct_change().fillna(0)
    drift_returns = drift_equity_series.pct_change().fillna(0)

    # Annualized turnover
    if T > 0:
        years = T / 252
        annual_turnover = total_turnover / years * 100 if years > 0 else 0
    else:
        annual_turnover = 0

    return {
        'rebalanced_equity': rebal_equity_series,
        'drifting_equity': drift_equity_series,
        'rebalanced_returns': rebal_returns,
        'drifting_returns': drift_returns,
        'total_turnover': annual_turnover,
        'num_rebalances': num_rebalances,
    }


def create_weights_bar_chart(weights_nonzero):
    """Create Plotly horizontal bar chart showing optimized allocation percentages."""
    sorted_weights = dict(sorted(weights_nonzero.items(), key=lambda x: x[1], reverse=True))
    tickers = list(sorted_weights.keys())
    pcts = [v * 100 for v in sorted_weights.values()]

    fig = go.Figure(go.Bar(
        x=pcts, y=tickers,
        orientation='h',
        marker=dict(
            color=pcts,
            colorscale=[[0, '#2962ff'], [0.5, '#00e676'], [1, '#ff6d00']],
            line=dict(width=0),
        ),
        text=[f'{p:.1f}%' for p in pcts],
        textposition='outside',
        textfont=dict(color='white', size=12),
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#121212", plot_bgcolor="#1E1E1E",
        title=dict(text="Optimized Allocation Weights", font=dict(size=16)),
        xaxis_title="Weight (%)",
        yaxis=dict(autorange="reversed"),
        height=max(300, len(tickers) * 35 + 100),
        margin=dict(l=80, r=80, t=50, b=40),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)')

    return fig


def create_efficient_frontier_plotly(opt_result):
    """Create Plotly efficient frontier scatter plot with optimal portfolio marked."""
    fig = go.Figure()

    # Simulated portfolios colored by Sharpe ratio
    fig.add_trace(go.Scatter(
        x=opt_result['frontier_vols'],
        y=opt_result['frontier_returns'],
        mode='markers',
        marker=dict(
            size=4, opacity=0.4,
            color=opt_result['frontier_sharpes'],
            colorscale='Viridis',
            colorbar=dict(title="Sharpe", thickness=15, len=0.6),
        ),
        name='Simulated Portfolios',
        hovertemplate='Vol: %{x:.1f}%<br>Ret: %{y:.1f}%<extra></extra>',
    ))

    # Individual assets
    mu = opt_result['mu']
    fig.add_trace(go.Scatter(
        x=opt_result['asset_vols'].tolist(),
        y=opt_result['asset_rets'].tolist(),
        mode='markers+text',
        marker=dict(size=10, color='white', symbol='diamond', line=dict(width=1, color='gray')),
        text=mu.index.tolist(),
        textposition='top center',
        textfont=dict(size=9, color='rgba(255,255,255,0.7)'),
        name='Individual Assets',
        hovertemplate='%{text}<br>Vol: %{x:.1f}%<br>Ret: %{y:.1f}%<extra></extra>',
    ))

    # Optimal portfolio (star)
    fig.add_trace(go.Scatter(
        x=[opt_result['annual_volatility'] * 100],
        y=[opt_result['expected_return'] * 100],
        mode='markers',
        marker=dict(size=18, color='#ff6d00', symbol='star', line=dict(width=2, color='white')),
        name=f"Max Sharpe ({opt_result['sharpe_ratio']:.2f})",
        hovertemplate='Optimal<br>Vol: %{x:.1f}%<br>Ret: %{y:.1f}%<extra></extra>',
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#121212", plot_bgcolor="#1E1E1E",
        title=dict(text="Efficient Frontier — Mean-Variance Optimization", font=dict(size=16)),
        xaxis_title="Annual Volatility (%)",
        yaxis_title="Expected Annual Return (%)",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='closest',
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)')

    return fig


@st.cache_data(ttl=30)
def _check_ollama_online() -> bool:
    """Poll the Ollama /api/tags endpoint. Result cached for 30 s."""
    try:
        r = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def run_walk_forward_backtest(
    close_df, high_df, low_df, volume_df,
    strategy_class, signal_param_ranges,
    sl_range, tp_range, ann_factor, vbt_freq,
    grid_valid_tickers, train_years=3, test_years=1,
    macro_regime=None, safe_haven_action='cash',
    progress_cb=None,
):
    """Walk-forward (out-of-sample) backtest using rolling train/test windows.

    For each fold:
      - Runs the full grid search on the train window to find the max-Sharpe
        signal-param + SL/TP combination (in-sample).
      - Applies those frozen parameters to the subsequent test window (out-of-sample).
      - Stitches all test-window returns into a continuous OOS equity curve.

    If macro_regime is provided (a boolean Series, True = Risk Off), the regime
    filter is sliced to each fold's date range and applied to entries/exits before
    any Sharpe calculation — so the WFO respects the same filter as the main backtest.

    Returns a dict with OOS/IS equity curves and metrics, or None if there is
    not enough data for at least one complete fold.
    """
    _SAFE_HAVENS = {'GLD', 'TLT'}
    M = len(grid_valid_tickers)

    # Build rolling fold windows
    folds = []
    fold_start = close_df.index[0]
    data_end = close_df.index[-1]

    while True:
        train_end = pd.Timestamp(fold_start) + pd.DateOffset(years=train_years)
        test_end  = train_end + pd.DateOffset(years=test_years)
        if test_end > pd.Timestamp(data_end):
            break
        folds.append({
            'train_start': pd.Timestamp(fold_start),
            'train_end':   train_end,
            'test_start':  train_end,
            'test_end':    test_end,
        })
        fold_start = pd.Timestamp(fold_start) + pd.DateOffset(years=test_years)

    if not folds:
        return None

    # If sl_range is a placeholder (per-side SL mode), fall back to a 2% default
    _sl_range = sl_range if (len(sl_range) > 0 and sl_range[0] > 0) else np.array([0.02])
    _tp_range = tp_range if len(tp_range) > 0 else np.array([0.04])

    oos_returns_list       = []
    is_returns_list        = []
    fold_records           = []
    oos_gross_returns_list = []   # Task 3: gross (fee-free) OOS returns
    oos_base_signals_list  = []   # Task 4: pre-macro-mask signals for sensitivity sweep

    for fi, fold in enumerate(folds):
        if progress_cb:
            progress_cb(
                fi / len(folds),
                f"Fold {fi+1}/{len(folds)}: Training "
                f"{fold['train_start'].strftime('%Y')}–{fold['train_end'].strftime('%Y')}…"
            )

        # Slice data for this fold
        tr_mask = (close_df.index >= fold['train_start']) & (close_df.index < fold['train_end'])
        te_mask = (close_df.index >= fold['test_start'])  & (close_df.index < fold['test_end'])

        if tr_mask.sum() < 60 or te_mask.sum() < 20:
            continue

        close_tr = close_df[tr_mask]; high_tr = high_df[tr_mask]
        low_tr   = low_df[tr_mask];   vol_tr  = volume_df[tr_mask]

        # Extend the test slice back by 200 bars so that long-period indicators
        # (SMA-200, RSI, etc.) are fully warmed up before the test window begins.
        # Warmup returns are stripped after VBT evaluation — only bars from
        # test_start onward are included in the stitched OOS equity curve.
        warmup_bars      = 200
        te_start_pos     = close_df.index.searchsorted(fold['test_start'])
        warmup_start_pos = max(0, te_start_pos - warmup_bars)
        warmup_start     = close_df.index[warmup_start_pos]

        te_ext_mask = (close_df.index >= warmup_start) & (close_df.index < fold['test_end'])
        close_te = close_df[te_ext_mask]; high_te = high_df[te_ext_mask]
        low_te   = low_df[te_ext_mask];   vol_te  = volume_df[te_ext_mask]

        T_tr = len(close_tr)
        T_te = len(close_te)

        # Generate vectorized signals for training window
        train_sigs = {}
        for tkr in grid_valid_tickers:
            try:
                train_sigs[tkr] = strategy_class.generate_vectorized_signals(
                    close_tr[tkr], high_tr[tkr], low_tr[tkr], vol_tr[tkr],
                    signal_param_ranges
                )
            except Exception:
                pass

        if len(train_sigs) < M:
            continue

        first_sig = train_sigs[grid_valid_tickers[0]]
        n_combos  = first_sig['long_entries'].shape[1] if first_sig['long_entries'].ndim > 1 else 1

        # Build super matrices for training: shape (T_tr, M*N)
        le_tr_sup = np.column_stack([train_sigs[t]['long_entries'].reshape(T_tr, -1) for t in grid_valid_tickers])
        lx_tr_sup = np.column_stack([train_sigs[t]['long_exits'].reshape(T_tr, -1)   for t in grid_valid_tickers])
        se_tr_sup = np.column_stack([train_sigs[t]['short_entries'].reshape(T_tr, -1) for t in grid_valid_tickers])
        sx_tr_sup = np.column_stack([train_sigs[t]['short_exits'].reshape(T_tr, -1)   for t in grid_valid_tickers])

        close_super_arr = np.column_stack([
            np.tile(close_tr[t].values[:, None], (1, n_combos)) for t in grid_valid_tickers
        ])
        close_super_tr = pd.DataFrame(close_super_arr, index=close_tr.index)

        # Apply macro regime filter to train super matrices.
        # This is applied in-place before the grid search, so both the Sharpe
        # ranking and the IS equity automatically obey the regime filter.
        if macro_regime is not None:
            risk_off_tr = macro_regime.reindex(close_tr.index, method='ffill').fillna(False).values
            risk_on_tr  = ~risk_off_tr
            col_offset  = 0
            for tkr in grid_valid_tickers:
                if tkr.upper() not in _SAFE_HAVENS:
                    le_tr_sup[:, col_offset:col_offset + n_combos] &= risk_on_tr[:, None]
                    lx_tr_sup[:, col_offset:col_offset + n_combos] |= risk_off_tr[:, None]
                if safe_haven_action in ('TLT', 'GLD') and tkr == safe_haven_action:
                    le_tr_sup[:, col_offset:col_offset + n_combos] |= risk_off_tr[:, None]
                    lx_tr_sup[:, col_offset:col_offset + n_combos] &= ~risk_off_tr[:, None]
                col_offset += n_combos

        # Grid search on training window: find best (signal_idx, sl, tp) by Sharpe
        best_sharpe_fold = -np.inf
        best_signal_idx  = 0
        best_sl_fold     = _sl_range[0]
        best_tp_fold     = _tp_range[-1]

        for sl in _sl_range:
            for tp in _tp_range:
                if tp <= sl:
                    continue
                try:
                    pf_tr = vbt.Portfolio.from_signals(
                        close_super_tr, entries=le_tr_sup, exits=lx_tr_sup,
                        short_entries=se_tr_sup, short_exits=sx_tr_sup,
                        sl_stop=sl, tp_stop=tp,
                        freq=vbt_freq, init_cash=10000, fees=0.001,
                    )
                    ret_arr  = pf_tr.returns().values         # (T_tr, M*N)
                    ret_3d   = ret_arr.reshape(T_tr, M, n_combos)
                    comb_2d  = ret_3d.mean(axis=1)           # (T_tr, N)
                    means    = comb_2d.mean(axis=0)
                    stds     = comb_2d.std(axis=0)
                    sharpes  = np.divide(means, stds, out=np.zeros_like(means), where=stds > 0) * np.sqrt(ann_factor)

                    idx = int(np.argmax(sharpes))
                    if sharpes[idx] > best_sharpe_fold:
                        best_sharpe_fold = float(sharpes[idx])
                        best_signal_idx  = idx
                        best_sl_fold     = sl
                        best_tp_fold     = tp
                except Exception:
                    continue

        # IS equity: best combo applied to the train window (shows in-sample fit)
        try:
            is_le = pd.DataFrame({
                t: le_tr_sup[:, ti * n_combos + best_signal_idx]
                for ti, t in enumerate(grid_valid_tickers)
            }, index=close_tr.index)
            is_lx = pd.DataFrame({
                t: lx_tr_sup[:, ti * n_combos + best_signal_idx]
                for ti, t in enumerate(grid_valid_tickers)
            }, index=close_tr.index)
            is_se = pd.DataFrame({
                t: se_tr_sup[:, ti * n_combos + best_signal_idx]
                for ti, t in enumerate(grid_valid_tickers)
            }, index=close_tr.index)
            is_sx = pd.DataFrame({
                t: sx_tr_sup[:, ti * n_combos + best_signal_idx]
                for ti, t in enumerate(grid_valid_tickers)
            }, index=close_tr.index)

            pf_is = vbt.Portfolio.from_signals(
                close_tr, entries=is_le, exits=is_lx,
                short_entries=is_se, short_exits=is_sx,
                sl_stop=best_sl_fold, tp_stop=best_tp_fold,
                freq=vbt_freq, init_cash=10000, fees=0.001,
            )
            is_returns_list.append(pf_is.returns().mean(axis=1))
        except Exception:
            pass

        # OOS equity: frozen params applied to the test window (out-of-sample)
        test_sigs = {}
        for tkr in grid_valid_tickers:
            try:
                test_sigs[tkr] = strategy_class.generate_vectorized_signals(
                    close_te[tkr], high_te[tkr], low_te[tkr], vol_te[tkr],
                    signal_param_ranges
                )
            except Exception:
                pass

        if len(test_sigs) < M:
            continue

        try:
            oos_le = pd.DataFrame({
                t: test_sigs[t]['long_entries'].reshape(T_te, -1)[:, best_signal_idx]
                for t in grid_valid_tickers if t in test_sigs
            }, index=close_te.index)
            oos_lx = pd.DataFrame({
                t: test_sigs[t]['long_exits'].reshape(T_te, -1)[:, best_signal_idx]
                for t in grid_valid_tickers if t in test_sigs
            }, index=close_te.index)
            oos_se = pd.DataFrame({
                t: test_sigs[t]['short_entries'].reshape(T_te, -1)[:, best_signal_idx]
                for t in grid_valid_tickers if t in test_sigs
            }, index=close_te.index)
            oos_sx = pd.DataFrame({
                t: test_sigs[t]['short_exits'].reshape(T_te, -1)[:, best_signal_idx]
                for t in grid_valid_tickers if t in test_sigs
            }, index=close_te.index)

            # Save pre-mask base signals for gross portfolio (Task 3) and
            # hysteresis sensitivity sweep (Task 4)
            oos_le_base = oos_le.copy()
            oos_lx_base = oos_lx.copy()
            oos_se_base = oos_se.copy()
            oos_sx_base = oos_sx.copy()

            # Apply macro regime filter to OOS test-window signals.
            # Slice macro_regime to the extended test window (including warmup),
            # then apply bitwise masks per ticker — identical logic to the main grid search.
            if macro_regime is not None:
                risk_off_te = macro_regime.reindex(close_te.index, method='ffill').fillna(False)
                risk_on_te  = ~risk_off_te
                for tkr in list(oos_le.columns):
                    if tkr.upper() not in _SAFE_HAVENS:
                        oos_le[tkr] = oos_le[tkr] & risk_on_te.values
                        oos_lx[tkr] = oos_lx[tkr] | risk_off_te.values
                    if safe_haven_action in ('TLT', 'GLD') and tkr == safe_haven_action:
                        oos_le[tkr] = oos_le[tkr] | risk_off_te.values
                        oos_lx[tkr] = oos_lx[tkr] & ~risk_off_te.values

            pf_oos = vbt.Portfolio.from_signals(
                close_te, entries=oos_le, exits=oos_lx,
                short_entries=oos_se, short_exits=oos_sx,
                sl_stop=best_sl_fold, tp_stop=best_tp_fold,
                freq=vbt_freq, init_cash=10000, fees=0.001,
            )
            # Strip the 200-bar warmup prefix — only keep returns from test_start onward
            oos_raw      = pf_oos.returns().mean(axis=1)
            oos_stripped = oos_raw[oos_raw.index >= fold['test_start']]
            oos_returns_list.append(oos_stripped)

            # --- Per-fold OOS diagnostics (Task 1) ---
            if len(oos_stripped) > 1 and oos_stripped.std() > 0:
                fold_oos_sharpe = float(oos_stripped.mean() / oos_stripped.std()) * np.sqrt(ann_factor)
            else:
                fold_oos_sharpe = 0.0
            fold_eq = (1 + oos_stripped).cumprod() * 10000
            fold_oos_maxdd = float((fold_eq / fold_eq.cummax() - 1).min() * 100) if len(fold_eq) > 0 else 0.0

            # Turnover: entry signals per (bars × tickers) in the actual test window
            _test_mask    = oos_le.index >= fold['test_start']
            _n_entries    = int(oos_le[_test_mask].values.sum())
            _n_bars       = int(_test_mask.sum())
            fold_turnover = (_n_entries / max(_n_bars * max(len(oos_le.columns), 1), 1)) * 100

            # Hit rate from VBT trade records
            try:
                _trade_rets   = pf_oos.trades.records['return']
                fold_hit_rate = float((_trade_rets > 0).mean() * 100) if len(_trade_rets) > 0 else 0.0
            except Exception:
                fold_hit_rate = 0.0

            # --- Gross OOS portfolio (Task 3) ---
            # Use the SAME post-macro-mask signals as the net portfolio — the only
            # difference between gross and net must be fees=0.0 vs fees=0.001.
            # Using _base signals here would give different trade patterns (macro
            # filter not applied), causing the curves to diverge in shape.
            try:
                pf_oos_gross = vbt.Portfolio.from_signals(
                    close_te, entries=oos_le, exits=oos_lx,
                    short_entries=oos_se, short_exits=oos_sx,
                    sl_stop=best_sl_fold, tp_stop=best_tp_fold,
                    freq=vbt_freq, init_cash=10000, fees=0.0,
                )
                gross_raw = pf_oos_gross.returns().mean(axis=1)
                oos_gross_returns_list.append(gross_raw[gross_raw.index >= fold['test_start']])
            except Exception:
                pass

            # --- Base signals for sensitivity sweep (Task 4) ---
            oos_base_signals_list.append({
                'close':      close_te,
                'le':         oos_le_base,
                'lx':         oos_lx_base,
                'se':         oos_se_base,
                'sx':         oos_sx_base,
                'sl':         best_sl_fold,
                'tp':         best_tp_fold,
                'test_start': fold['test_start'],
            })

        except Exception:
            continue

        fold_records.append({
            'Fold':         fi + 1,
            'Train Period': (
                f"{fold['train_start'].strftime('%Y-%m')} → "
                f"{fold['train_end'].strftime('%Y-%m')}"
            ),
            'Test Period': (
                f"{fold['test_start'].strftime('%Y-%m')} → "
                f"{fold['test_end'].strftime('%Y-%m')}"
            ),
            'IS Sharpe':    round(best_sharpe_fold, 2),
            'OOS Sharpe':   round(fold_oos_sharpe, 2),
            'OOS Max DD %': round(fold_oos_maxdd, 1),
            'Turnover %':   round(fold_turnover, 1),
            'Hit Rate %':   round(fold_hit_rate, 1),
            'Best SL %':    round(float(best_sl_fold) * 100, 1),
            'Best TP %':    round(float(best_tp_fold) * 100, 1),
        })

    if progress_cb:
        progress_cb(1.0, "Walk-forward complete.")

    if not oos_returns_list:
        return None

    def _equity_metrics(equity):
        ret_s   = equity.pct_change().dropna()
        total   = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        sharpe  = (ret_s.mean() / ret_s.std()) * np.sqrt(ann_factor) if ret_s.std() > 0 else 0.0
        run_max = equity.cummax()
        max_dd  = ((equity - run_max) / run_max).min() * 100
        return round(float(total), 2), round(float(sharpe), 2), round(float(max_dd), 2)

    oos_combined = pd.concat(oos_returns_list).sort_index()
    oos_equity   = (1 + oos_combined).cumprod() * 10000
    oos_ret, oos_sharpe, oos_dd = _equity_metrics(oos_equity)

    if is_returns_list:
        is_combined = pd.concat(is_returns_list).sort_index()
        is_equity   = (1 + is_combined).cumprod() * 10000
        is_ret, is_sharpe, is_dd = _equity_metrics(is_equity)
    else:
        is_equity = oos_equity
        is_ret, is_sharpe, is_dd = oos_ret, oos_sharpe, oos_dd

    # Task 3: stitch gross (fee-free) OOS equity
    if oos_gross_returns_list:
        gross_combined   = pd.concat(oos_gross_returns_list).sort_index()
        oos_equity_gross = (1 + gross_combined).cumprod() * 10000
    else:
        oos_equity_gross = None

    return {
        'folds':             fold_records,
        'oos_equity':        oos_equity,
        'is_equity':         is_equity,
        'oos_return':        oos_ret,    'is_return':  is_ret,
        'oos_sharpe':        oos_sharpe, 'is_sharpe':  is_sharpe,
        'oos_max_dd':        oos_dd,     'is_max_dd':  is_dd,
        'n_folds':           len(fold_records),
        'oos_equity_gross':  oos_equity_gross,   # Task 3
        'oos_base_signals':  oos_base_signals_list,  # Task 4
    }


def compute_hysteresis_sensitivity(
    macro_df, oos_base_signals, ann_factor, vbt_freq,
    lower_range=None, upper_range=None,
):
    """Sweep (lower_threshold, upper_threshold) hysteresis params and return a 2D OOS-Sharpe matrix.

    Args:
        macro_df: DataFrame of FRED macro data (from get_macro_data())
        oos_base_signals: List of per-fold dicts with pre-macro-mask signals
        ann_factor: Annualisation factor (e.g. 252 for daily)
        vbt_freq: VectorBT frequency string (e.g. '1D')
        lower_range: 1-D array of lower_threshold values (entry band, negative)
        upper_range: 1-D array of upper_threshold values (exit band, positive)

    Returns:
        (matrix, lower_range, upper_range) — matrix shape (len(upper_range), len(lower_range))
    """
    _SAFE_HAVENS = {'GLD', 'TLT'}
    if lower_range is None:
        lower_range = np.array([-0.20, -0.15, -0.10, -0.05, -0.02])
    if upper_range is None:
        upper_range = np.array([0.02, 0.05, 0.10, 0.15, 0.20])

    results = np.full((len(upper_range), len(lower_range)), np.nan)

    for i, ut in enumerate(upper_range):
        for j, lt in enumerate(lower_range):
            regime_v = classify_macro_regime(macro_df, upper_threshold=float(ut), lower_threshold=float(lt))
            oos_rets = []
            for fs in oos_base_signals:
                close_te = fs['close']
                le = fs['le'].copy()
                lx = fs['lx'].copy()
                se = fs['se'].copy()
                sx = fs['sx'].copy()
                risk_off = regime_v.reindex(close_te.index, method='ffill').fillna(False)
                risk_on  = ~risk_off
                for tkr in le.columns:
                    if tkr.upper() not in _SAFE_HAVENS:
                        le[tkr] = le[tkr] & risk_on.values
                        lx[tkr] = lx[tkr] | risk_off.values
                try:
                    pf = vbt.Portfolio.from_signals(
                        close_te, entries=le, exits=lx,
                        short_entries=se, short_exits=sx,
                        sl_stop=fs['sl'], tp_stop=fs['tp'],
                        freq=vbt_freq, init_cash=10000, fees=0.001,
                    )
                    r = pf.returns().mean(axis=1)
                    oos_rets.append(r[r.index >= fs['test_start']])
                except Exception:
                    pass
            if oos_rets:
                all_r = pd.concat(oos_rets)
                std = all_r.std()
                results[i, j] = float(all_r.mean() / std * np.sqrt(ann_factor)) if std > 0 else 0.0

    return results, lower_range, upper_range


def create_hysteresis_heatmap(matrix, lower_range, upper_range,
                               current_lt=-0.10, current_ut=0.10):
    """Create a Plotly heatmap of OOS Sharpe over hysteresis threshold grid.

    Args:
        matrix: 2-D numpy array (rows=upper_range, cols=lower_range)
        lower_range: 1-D array of lower_threshold values
        upper_range: 1-D array of upper_threshold values
        current_lt: Currently configured lower_threshold (marked with ★)
        current_ut: Currently configured upper_threshold (marked with ★)

    Returns:
        Plotly Figure
    """
    x_labels = [f"{v:.2f}" for v in lower_range]
    y_labels = [f"{v:.2f}" for v in upper_range]
    text     = [[f"{v:.2f}" if not np.isnan(v) else "" for v in row] for row in matrix]

    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=x_labels, y=y_labels,
        colorscale='RdYlGn',
        colorbar=dict(title='OOS Sharpe'),
        text=text, texttemplate='%{text}',
        hovertemplate='Entry Band: %{x}<br>Exit Band: %{y}<br>OOS Sharpe: %{z:.3f}<extra></extra>',
    ))

    # Mark the currently configured thresholds
    xi = int(np.argmin(np.abs(lower_range - current_lt)))
    yi = int(np.argmin(np.abs(upper_range - current_ut)))
    fig.add_annotation(
        x=x_labels[xi], y=y_labels[yi],
        text="★", showarrow=False,
        font=dict(size=22, color='white'),
    )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#121212',
        plot_bgcolor='#1E1E1E',
        font=dict(color='#E6EDF3'),
        title=dict(
            text='Hysteresis Parameter Sensitivity — OOS Sharpe  (★ = current setting)',
            font=dict(size=14, color='#E6EDF3'),
        ),
        xaxis=dict(title='Entry Band (lower_threshold)'),
        yaxis=dict(title='Exit Band (upper_threshold)'),
        height=420,
    )
    return fig


def create_dendrogram_chart(returns_df):
    """Create HRP asset clustering dendrogram using correlation distance."""
    from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendro
    from scipy.spatial.distance import squareform

    corr = returns_df.corr()
    dist = ((1 - corr) / 2.0) ** 0.5
    condensed = squareform(dist.values)
    linkage_matrix = linkage(condensed, method='single')

    dendro_data = scipy_dendro(linkage_matrix, labels=corr.columns.tolist(), no_plot=True)

    fig = go.Figure()
    for xs, ys in zip(dendro_data['icoord'], dendro_data['dcoord']):
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(color='#00e676', width=1.5),
            showlegend=False, hoverinfo='skip',
        ))

    n_labels = len(dendro_data['ivl'])
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#121212', plot_bgcolor='#1E1E1E',
        font=dict(color='#E6EDF3'),
        xaxis=dict(
            tickvals=[5 + 10 * i for i in range(n_labels)],
            ticktext=dendro_data['ivl'], tickangle=45,
        ),
        yaxis=dict(title='Correlation Distance'),
        title=dict(text='HRP Asset Clustering (Dendrogram)', font=dict(size=14, color='#E6EDF3')),
        height=400, margin=dict(b=80),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    return fig


def create_model_comparison_bar_chart(mvo_weights, hrp_weights):
    """Create grouped bar chart comparing MVO vs HRP allocation weights."""
    all_tickers = sorted(
        set(mvo_weights) | set(hrp_weights),
        key=lambda t: mvo_weights.get(t, 0), reverse=True
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Markowitz (MVO)', x=all_tickers,
        y=[mvo_weights.get(t, 0) * 100 for t in all_tickers],
        marker_color='#2962ff',
    ))
    fig.add_trace(go.Bar(
        name='HRP', x=all_tickers,
        y=[hrp_weights.get(t, 0) * 100 for t in all_tickers],
        marker_color='#00e676',
    ))

    fig.update_layout(
        barmode='group',
        template='plotly_dark',
        paper_bgcolor='#121212', plot_bgcolor='#1E1E1E',
        font=dict(color='#E6EDF3'),
        title=dict(text='Weight Comparison: Markowitz vs HRP', font=dict(size=14, color='#E6EDF3')),
        yaxis_title='Allocation (%)',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, tickangle=45)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    return fig


def create_strategy_manifold(df_results):
    """Create UMAP 2D manifold of grid search parameter space, colored by Sharpe Ratio."""
    from sklearn.preprocessing import StandardScaler
    import umap

    metric_cols = {'Return (%)', 'Win Rate (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Total Trades'}
    param_cols = [c for c in df_results.columns if c not in metric_cols]

    if len(param_cols) < 2 or len(df_results) < 15:
        return None

    features = df_results[param_cols].copy()
    features = features.apply(pd.to_numeric, errors='coerce').fillna(0)

    scaled = StandardScaler().fit_transform(features)

    n_neighbors = min(15, len(df_results) - 1)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
    embedding = reducer.fit_transform(scaled)

    plot_df = df_results.copy()
    plot_df['UMAP-1'] = embedding[:, 0]
    plot_df['UMAP-2'] = embedding[:, 1]

    color_col = 'Sharpe Ratio' if 'Sharpe Ratio' in plot_df.columns else 'Return (%)'

    fig = px.scatter(
        plot_df, x='UMAP-1', y='UMAP-2',
        color=color_col, color_continuous_scale='Viridis',
        hover_data=param_cols + [color_col, 'Return (%)'],
        opacity=0.7,
    )

    # Mark the best point (row 0, since df_results is sorted by best metric)
    best = plot_df.iloc[0]
    fig.add_trace(go.Scatter(
        x=[best['UMAP-1']], y=[best['UMAP-2']],
        mode='markers',
        marker=dict(size=16, color='red', symbol='star', line=dict(width=2, color='white')),
        name='Best Config', showlegend=True,
        hovertemplate='Best Config<extra></extra>',
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#121212', plot_bgcolor='#1E1E1E',
        font=dict(color='#E6EDF3'),
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def display_multi_asset_results(results):
    """Display multi-asset backtest results with portfolio-level metrics."""
    if results is None:
        st.warning("No results to display.")
        return False

    # Macro Filter Badge
    skipped_entries = results.get('skipped_entries')
    skipped_count = 0
    if skipped_entries is not None and not skipped_entries.empty:
        skipped_count = int(skipped_entries.any(axis=1).sum())
    if results.get('macro_filtered'):
        badge_msg = "Macro Regime Filter active: equity longs suppressed during Risk Off periods."
        if skipped_count > 0:
            badge_msg += f" **{skipped_count} trades blocked** by macro filter."
        st.info(badge_msg)

    # KPI Row
    n_kpi_cols = 7 if results.get('macro_filtered') else 6
    kpi_cols = st.columns(n_kpi_cols)

    with kpi_cols[0]:
        ret = results['portfolio_return']
        st.metric(
            "Portfolio Return",
            f"{ret:.2f}%",
            delta=f"{ret:.2f}%" if ret > 0 else None,
            help="Equal-weight portfolio total return"
        )
    with kpi_cols[1]:
        bm = results['benchmark_return']
        st.metric(
            "SPY Benchmark",
            f"{bm:.2f}%",
            delta=f"{bm - ret:.2f}% vs Portfolio",
            delta_color="inverse",
            help="SPY buy-and-hold benchmark"
        )
    with kpi_cols[2]:
        st.metric("Avg Win Rate", f"{results['avg_win_rate']:.1f}%")
    with kpi_cols[3]:
        st.metric("Portfolio Sharpe", f"{results['portfolio_sharpe']:.2f}")
    with kpi_cols[4]:
        st.metric("Max Drawdown", f"{results['portfolio_max_dd']:.2f}%")
    with kpi_cols[5]:
        st.metric("Assets Traded", f"{results['num_assets']}")
    if results.get('macro_filtered'):
        with kpi_cols[6]:
            st.metric("Trades Blocked", f"{skipped_count}",
                      help="Long entries suppressed by macro regime filter")

    # Equity Chart
    opt = results.get('optimization')
    opt_equity = opt['optimized_equity'] if opt is not None else None
    # Only show drifting trace when rebalancing is active 
    drift_equity = None
    if opt is not None and opt.get('rebalance_freq', 'Never') != 'Never':
        drift_equity = opt.get('drifting_equity')
    fig = create_multi_asset_equity_chart(
        results['combined_equity'],
        results['per_asset_equity'],
        results['benchmark_equity'],
        optimized_equity=opt_equity,
        drifting_equity=drift_equity,
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Per Asset Performance Table
    with st.expander("Per-Asset Performance", expanded=True):
        stats_df = results['per_asset_stats']
        if stats_df is not None and not stats_df.empty:
            st.dataframe(
                stats_df.sort_values('Return (%)', ascending=False),
                use_container_width=True,
                column_config={
                    "Return (%)": st.column_config.NumberColumn(format="%.2f%%"),
                    "Win Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
                    "Max DD (%)": st.column_config.NumberColumn(format="%.2f%%"),
                    "Trades": st.column_config.NumberColumn(format="%d"),
                }
            )

    # Correlation Heatmap
    per_asset_returns = results.get('per_asset_returns')
    if per_asset_returns is not None and not per_asset_returns.empty:
        with st.expander("Correlation Heatmap", expanded=True):
            # Cap at top 20 for readability when many tickers
            if len(per_asset_returns.columns) > 30:
                top_20 = results['per_asset_stats'].nlargest(20, 'Return (%)')['Ticker'].tolist()
                corr_returns = per_asset_returns[top_20]
                st.caption(f"Showing top 20 assets by return (out of {len(per_asset_returns.columns)})")
            else:
                corr_returns = per_asset_returns
            fig_corr = create_correlation_heatmap(corr_returns)
            st.plotly_chart(fig_corr, use_container_width=True, config={'displayModeBar': False})

    # Portfolio Allocation Models
    hrp = results.get('hrp_optimization')
    has_mvo = opt is not None
    has_hrp = hrp is not None
    has_both = has_mvo and has_hrp

    def _compute_model_metrics(model_result, label):
        """Helper to compute return, sharpe, max DD, top-3 concentration for a model."""
        eq = model_result['optimized_equity']
        ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
        rets = model_result['optimized_returns']
        sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0.0
        running_max = eq.cummax()
        max_dd = ((eq - running_max) / running_max).min() * 100
        # Top 3 concentration
        sorted_w = sorted(model_result['weights_nonzero'].values(), reverse=True)
        top3_pct = sum(sorted_w[:3]) * 100
        return {'label': label, 'return': ret, 'sharpe': sharpe, 'max_dd': max_dd,
                'top3_pct': top3_pct, 'positions': len(model_result['weights_nonzero'])}

    if has_both:
        # Side by Side Model Comparison
        mvo_m = _compute_model_metrics(opt, 'Markowitz (MVO)')
        hrp_m = _compute_model_metrics(hrp, 'HRP')

        with st.expander("Portfolio Allocation — Model Comparison", expanded=True):
            # Comparison metrics table
            comp_df = pd.DataFrame([
                {'Metric': 'Return (%)', 'Markowitz (MVO)': f"{mvo_m['return']:.2f}",
                 'HRP': f"{hrp_m['return']:.2f}"},
                {'Metric': 'Sharpe Ratio', 'Markowitz (MVO)': f"{mvo_m['sharpe']:.2f}",
                 'HRP': f"{hrp_m['sharpe']:.2f}"},
                {'Metric': 'Max Drawdown (%)', 'Markowitz (MVO)': f"{mvo_m['max_dd']:.2f}",
                 'HRP': f"{hrp_m['max_dd']:.2f}"},
                {'Metric': 'Top 3 Holdings (%)', 'Markowitz (MVO)': f"{mvo_m['top3_pct']:.1f}",
                 'HRP': f"{hrp_m['top3_pct']:.1f}"},
                {'Metric': 'Positions', 'Markowitz (MVO)': f"{mvo_m['positions']}",
                 'HRP': f"{hrp_m['positions']}"},
            ])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Grouped bar chart: weight comparison
            fig_comp = create_model_comparison_bar_chart(opt['weights_nonzero'], hrp['weights_nonzero'])
            st.plotly_chart(fig_comp, use_container_width=True, config={'displayModeBar': False})

            # Two columns: MVO left, HRP right
            col_mvo, col_hrp = st.columns(2)
            with col_mvo:
                st.markdown("#### Markowitz (MVO)")
                fig_weights = create_weights_bar_chart(opt['weights_nonzero'])
                st.plotly_chart(fig_weights, use_container_width=True, config={'displayModeBar': False})
                fig_frontier = create_efficient_frontier_plotly(opt)
                st.plotly_chart(fig_frontier, use_container_width=True, config={'displayModeBar': False})

            with col_hrp:
                st.markdown("#### Hierarchical Risk Parity")
                fig_hrp_weights = create_weights_bar_chart(hrp['weights_nonzero'])
                st.plotly_chart(fig_hrp_weights, use_container_width=True, config={'displayModeBar': False})
                if hrp.get('returns_df') is not None:
                    fig_dendro = create_dendrogram_chart(hrp['returns_df'])
                    st.plotly_chart(fig_dendro, use_container_width=True, config={'displayModeBar': False})

    elif has_mvo:
        # Markowitz Only
        rebal_freq = opt.get('rebalance_freq', 'Never')
        rebal_label = f" — {rebal_freq} Rebalancing" if rebal_freq != 'Never' else " — Buy & Hold"
        with st.expander(f"Portfolio Allocation (Markowitz{rebal_label})", expanded=True):
            mvo_m = _compute_model_metrics(opt, 'Markowitz')
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                delta_ret = mvo_m['return'] - results['portfolio_return']
                st.metric("Optimized Return", f"{mvo_m['return']:.2f}%",
                          delta=f"{delta_ret:+.2f}% vs EW")
            with col2:
                delta_sharpe = mvo_m['sharpe'] - results['portfolio_sharpe']
                st.metric("Optimized Sharpe", f"{mvo_m['sharpe']:.2f}",
                          delta=f"{delta_sharpe:+.2f} vs EW")
            with col3:
                st.metric("Optimized Max DD", f"{mvo_m['max_dd']:.2f}%")
            with col4:
                st.metric("Positions", f"{mvo_m['positions']} / {results['num_assets']}")

            fig_weights = create_weights_bar_chart(opt['weights_nonzero'])
            st.plotly_chart(fig_weights, use_container_width=True, config={'displayModeBar': False})
            fig_frontier = create_efficient_frontier_plotly(opt)
            st.plotly_chart(fig_frontier, use_container_width=True, config={'displayModeBar': False})

    elif has_hrp:
        # HRP Only
        rebal_freq = hrp.get('rebalance_freq', 'Never')
        rebal_label = f" — {rebal_freq} Rebalancing" if rebal_freq != 'Never' else " — Buy & Hold"
        with st.expander(f"Portfolio Allocation (HRP{rebal_label})", expanded=True):
            hrp_m = _compute_model_metrics(hrp, 'HRP')
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                delta_ret = hrp_m['return'] - results['portfolio_return']
                st.metric("HRP Return", f"{hrp_m['return']:.2f}%",
                          delta=f"{delta_ret:+.2f}% vs EW")
            with col2:
                delta_sharpe = hrp_m['sharpe'] - results['portfolio_sharpe']
                st.metric("HRP Sharpe", f"{hrp_m['sharpe']:.2f}",
                          delta=f"{delta_sharpe:+.2f} vs EW")
            with col3:
                st.metric("HRP Max DD", f"{hrp_m['max_dd']:.2f}%")
            with col4:
                st.metric("Positions", f"{hrp_m['positions']} / {results['num_assets']}")

            fig_weights = create_weights_bar_chart(hrp['weights_nonzero'])
            st.plotly_chart(fig_weights, use_container_width=True, config={'displayModeBar': False})
            if hrp.get('returns_df') is not None:
                fig_dendro = create_dendrogram_chart(hrp['returns_df'])
                st.plotly_chart(fig_dendro, use_container_width=True, config={'displayModeBar': False})

    # Failed tickers warning
    if results.get('failed_tickers'):
        st.warning(f"Failed to process {len(results['failed_tickers'])} tickers: {', '.join(results['failed_tickers'])}")

    return True


def display_backtest_results(results, ticker):
    """Display backtest results with metrics and charts.

    Args:
        results: dict with keys: strategy_return, buy_hold_return, win_rate,
                 sharpe, max_drawdown, total_trades, df_plot, trades_df, equity_series
        ticker: str
    """
    if results is None or results.get('total_trades', 0) == 0:
        st.warning("⚠️ No trades generated. Try adjusting parameters (e.g., higher RSI limit, wider SL/TP).")
        return False

    strategy_ret = results['strategy_return']
    buy_hold = results['buy_hold_return']

    # METRICS ROW
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric(
            "Strategy Return",
            f"{strategy_ret:.2f}%",
            delta=f"{strategy_ret:.2f}%" if strategy_ret > 0 else None,
            help="Your bot's actual performance"
        )

    with col2:
        st.metric(
            "Buy & Hold",
            f"{buy_hold:.2f}%",
            delta=f"{buy_hold - strategy_ret:.2f}% vs Strategy",
            delta_color="inverse",
            help="Market benchmark (passive investing)"
        )

    with col3:
        st.metric("Win Rate", f"{results['win_rate']:.1f}%")

    with col4:
        st.metric("Sharpe Ratio", f"{results['sharpe']:.2f}")

    with col5:
        st.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")

    with col6:
        st.metric("Total Trades", f"{results['total_trades']}")

    # Plotly Chart
    fig = create_price_chart(results['df_plot'], results['trades_df'], results['equity_series'], ticker)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Trade Journal
    st.subheader("📝 Trade Log")
    trades_df = results['trades_df']
    if not trades_df.empty:
        trade_log = trades_df.copy()
        trade_log['ReturnPct'] = trade_log['ReturnPct'] * 100
        st.dataframe(
            trade_log[['EntryTime', 'EntryPrice', 'ExitTime', 'ExitPrice', 'ReturnPct']].head(20),
            use_container_width=True,
            column_config={
                "ReturnPct": st.column_config.NumberColumn("Return %", format="%.2f%%"),
                "EntryPrice": st.column_config.NumberColumn("Entry", format="$%.2f"),
                "ExitPrice": st.column_config.NumberColumn("Exit", format="$%.2f"),
            }
        )

    return True


def check_data_warning(interval: str, period: str) -> str:
    """
    Check if the interval/period combination is problematic.
    Returns warning message if needed.
    """
    # Yahoo Finance strict limits for very short intervals
    if interval == "1m" and period not in ["1d", "5d"]:
        return f"⚠️ Yahoo Finance only provides ~7 days of 1m data. Period will be auto-capped to 5d."

    if interval in ["5m", "15m"] and period not in ["1d", "5d", "1mo"]:
        return f"⚠️ Yahoo Finance only provides ~60 days of {interval} data. Period will be auto-capped to 1mo."

    # Yahoo Finance limits intraday data to ~730 days
    intraday_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "4h"]
    long_periods = ["5y", "10y", "max"]

    if interval in intraday_intervals and period in long_periods:
        return f"⚠️ Yahoo Finance only provides ~730 days of {interval} data. Requesting {period} will return less data than expected."

    return None



# Sidebar Global Settings
with st.sidebar:
    # Sidebar header
    st.markdown("""
    <div style="padding-bottom: 1rem; margin-bottom: 1rem; border-bottom: 1px solid #21262D;">
        <h3 style="font-size: 1rem; font-weight: 600; color: #E6EDF3; margin: 0;">Settings</h3>
    </div>
    """, unsafe_allow_html=True)

    # AI Engine status indicator
    _ollama_online = _check_ollama_online()
    _ollama_host_label = config.OLLAMA_BASE_URL.split("://")[-1].split("/")[0]
    render_status_badge(
        f"AI Engine — {_ollama_host_label}",
        status="online" if _ollama_online else "offline",
    )
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

    st.markdown('<p style="font-size: 0.75rem; font-weight: 600; color: #8B949E; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem;">Market Data</p>', unsafe_allow_html=True)

    # Backtest Mode Toggle
    backtest_mode = st.radio(
        "Backtest Mode",
        ["Single Ticker", "Multi-Asset Portfolio"],
        index=0,
        horizontal=True,
        key="backtest_mode_toggle"
    )

    if backtest_mode == "Single Ticker":
        # Existing single-ticker selectbox
        ticker_option = st.selectbox(
            "Select Ticker",
            ALL_TICKERS,
            index=ALL_TICKERS.index(config.TICKER) if config.TICKER in ALL_TICKERS else 0
        )
        custom_ticker = st.text_input("Or enter custom ticker:", "")
        ticker = custom_ticker if custom_ticker else ticker_option

        for category, tickers in TICKER_CATEGORIES.items():
            if ticker in tickers:
                st.caption(f"Category: {category}")
                break

        multi_asset_mode = False
        selected_tickers_multi = []
        rebalance_freq = "Never"
        opt_models = ["Mean-Variance (Markowitz)"]
    else:
        # Multi-Asset Portfolio mode
        multi_asset_mode = True

        # Build full universe for multiselect options (include default basket so all defaults are valid)
        multi_universe = sorted(set(
            ALL_TICKERS + config.TICKER_UNIVERSE + config.CRYPTO_UNIVERSE
            + config.DEFAULT_MULTI_ASSET_BASKET
        ))

        selected_tickers_multi = st.multiselect(
            "Select Tickers (2-200)",
            options=multi_universe,
            default=config.DEFAULT_MULTI_ASSET_BASKET,
            help="Select assets for portfolio backtest. Supports 100+ tickers.",
            key="multi_asset_tickers"
        )

        # Custom tickers input for adding unlisted symbols
        custom_tickers_str = st.text_input(
            "Add custom tickers (comma-separated):",
            "",
            key="multi_custom_tickers"
        )
        if custom_tickers_str.strip():
            extras = [t.strip().upper() for t in custom_tickers_str.split(",") if t.strip()]
            selected_tickers_multi = list(dict.fromkeys(selected_tickers_multi + extras))

        st.caption(f"Selected: {len(selected_tickers_multi)} assets")
        ticker = f"Portfolio ({len(selected_tickers_multi)} assets)"

        rebalance_freq = st.selectbox(
            "Rebalance Frequency",
            ["Never", "Monthly", "Quarterly", "Yearly"],
            index=1,
            help="How often to reset portfolio to optimal weights. 'Never' = buy-and-hold drift.",
            key="rebalance_freq"
        )

        opt_models = st.multiselect(
            "Allocation Models",
            ["Mean-Variance (Markowitz)", "Hierarchical Risk Parity (HRP)"],
            default=["Mean-Variance (Markowitz)", "Hierarchical Risk Parity (HRP)"],
            help="Select one or both models. Both enables side-by-side comparison.",
            key="opt_models"
        )

    # Timeframe Selector 
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d", "1wk", "1mo"], index=3)

    # Data Period Selector 
    period = st.selectbox("Data Period", ["1d", "5d", "1mo", "1y", "2y", "5y", "10y", "max"], index=3)

    # Show warning if needed
    warning = check_data_warning(timeframe, period)
    if warning:
        st.warning(warning)

    st.markdown('<p style="font-size: 0.75rem; font-weight: 600; color: #8B949E; text-transform: uppercase; letter-spacing: 0.05em; margin: 1.5rem 0 0.75rem 0;">Strategy</p>', unsafe_allow_html=True)

    # Show error if custom strategies failed to load
    if custom_strategies is None and 'custom_strategies_error' in globals():
        st.error(f"Custom strategies failed to load: {custom_strategies_error}")

    # Strategy selector
    strategy_name = st.selectbox(
        "Trading Strategy",
        list(STRATEGY_REGISTRY.keys()),
        index=1  # Default to Smart Trend
    )

    selected_strategy_class = STRATEGY_REGISTRY[strategy_name]["class"]
    strategy_description = STRATEGY_REGISTRY[strategy_name]["description"]

    # Show strategy description
    st.caption(strategy_description)

    # Dynamic Strategy Parameters
    st.markdown('<p style="font-size: 0.75rem; font-weight: 600; color: #8B949E; text-transform: uppercase; letter-spacing: 0.05em; margin: 1.5rem 0 0.75rem 0;">Parameters</p>', unsafe_allow_html=True)

    # Get all optimizable parameters from the strategy class
    strategy_params = get_strategy_params(selected_strategy_class)

    # Store parameter values for passing to strategy
    param_values = {}

    if strategy_params:
        for param_name, param_config in strategy_params.items():
            param_label = param_config["name"]
            param_desc = param_config["description"]
            param_min = param_config["min"]
            param_max = param_config["max"]
            param_default = param_config["default"]
            param_step = param_config["step"]
            param_type = param_config["type"]

            # Get current value from config
            current_value = getattr(config, param_name, param_default)

            # Create the dynamic slider based on type
            if param_type == "float":
                param_value = st.slider(
                    param_label,
                    float(param_min),
                    float(param_max),
                    float(current_value),
                    float(param_step),
                    key=f"sidebar_{param_name}"
                )
            else:  # int
                param_value = st.slider(
                    param_label,
                    int(param_min),
                    int(param_max),
                    int(current_value),
                    int(param_step),
                    key=f"sidebar_{param_name}"
                )

            # Update config with the new value
            setattr(config, param_name, param_value)
            param_values[param_name] = param_value

            # Show description
            st.caption(f"ℹ{param_desc}")
    else:
        st.info("This strategy has no optimizable parameters")

    st.markdown('<p style="font-size: 0.75rem; font-weight: 600; color: #8B949E; text-transform: uppercase; letter-spacing: 0.05em; margin: 1.5rem 0 0.75rem 0;">AI Engine</p>', unsafe_allow_html=True)
    ai_mode = st.selectbox(
        "AI Mode",
        ["Disabled (Fast)", "AI Confirmation (Backtest Filter)"],
        index=0,
        help="When enabled, AI (Mixtral) reviews each entry signal during Quick Backtest",
        key="ai_mode_select"
    )
    use_real_ai = ai_mode != "Disabled (Fast)"
    if use_real_ai:
        ai_confidence_threshold = st.slider(
            "Min AI Confidence",
            min_value=0.0, max_value=1.0,
            value=float(config.MIN_AI_CONFIDENCE),
            step=0.05,
            key="ai_confidence_slider",
            help="Entries below this confidence are filtered out by the AI"
        )
        st.warning("AI will review each entry signal (~5-15s per signal)")
    else:
        ai_confidence_threshold = config.MIN_AI_CONFIDENCE

    # Macro Regime Filter 
    st.markdown('<p style="font-size: 0.75rem; font-weight: 600; color: #8B949E; text-transform: uppercase; letter-spacing: 0.05em; margin: 1.5rem 0 0.75rem 0;">Macro Regime</p>', unsafe_allow_html=True)
    use_macro_filter = st.checkbox(
        "Filter by Macro Regime",
        value=False,
        help="When Risk Off (credit spreads rising or yield curve inverted), suppress equity longs. Safe havens (GLD, TLT) unaffected.",
        key="macro_regime_filter"
    )
    safe_haven_action = 'cash'
    if use_macro_filter:
        safe_haven_action = st.selectbox(
            "Risk Off Action",
            ["Cash (Default)", "Rotate to TLT", "Rotate to GLD"],
            index=0,
            help="What to do during Risk Off: hold cash, or rotate into a safe-haven asset.",
            key="safe_haven_action"
        )
        # Map display label to action key
        safe_haven_action = {'Cash (Default)': 'cash', 'Rotate to TLT': 'TLT', 'Rotate to GLD': 'GLD'}[safe_haven_action]
        if safe_haven_action == 'cash':
            st.caption("Equity longs suppressed during Risk Off. GLD/TLT unaffected.")
        else:
            st.caption(f"Risk Off: equity longs suppressed, rotate into {safe_haven_action}.")

    # Walk-Forward Testing
    st.markdown('<p style="font-size: 0.75rem; font-weight: 600; color: #8B949E; text-transform: uppercase; letter-spacing: 0.05em; margin: 1.5rem 0 0.75rem 0;">Walk-Forward Testing</p>', unsafe_allow_html=True)
    enable_walk_forward = st.checkbox(
        "Enable Walk-Forward Optimization",
        value=False,
        help=(
            "Splits history into rolling train/test windows. "
            "Finds the best parameters in-sample, then validates them out-of-sample. "
            "Proves whether the strategy edge holds on unseen data."
        ),
        key="enable_walk_forward"
    )
    if enable_walk_forward:
        wf_train_years = st.slider(
            "Train Window (years)", min_value=1, max_value=7, value=3, step=1,
            key="wf_train_years",
            help="Number of years used to optimize parameters per fold."
        )
        wf_test_years = st.slider(
            "Test Window (years)", min_value=1, max_value=3, value=1, step=1,
            key="wf_test_years",
            help="Number of years applied out-of-sample per fold."
        )
        st.caption(
            f"Rolling: {wf_train_years}yr train → {wf_test_years}yr test. "
            f"Requires ≥ {wf_train_years + wf_test_years} years of data."
        )
        st.warning("Walk-forward re-runs the grid search per fold — may take several minutes.")
    else:
        wf_train_years = 3
        wf_test_years  = 1

    st.markdown("---")
    st.markdown(f'<p style="color: #484F58; font-size: 0.75rem; font-family: \'JetBrains Mono\', monospace;">Updated {datetime.now().strftime("%H:%M:%S")}</p>', unsafe_allow_html=True)



# Main Dashboard horizontal top navigation

# Initialize navigation state if not exists
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Main Terminal"

# Define navigation options
nav_options = ["Main Terminal", "Grid Search", "Strategy Builder", "Live Terminal", "Portfolio Opt", "Options Lab", "Global Macro"]
nav_icons = ["house", "search", "tools", "broadcast", "pie-chart", "graph-up", "globe"]

# Get current index from session state
current_index = nav_options.index(st.session_state.current_page) if st.session_state.current_page in nav_options else 0

# Horizontal Navigation Bar - Modern Top Navigation
selected_page = option_menu(
    menu_title=None,
    options=nav_options,
    icons=nav_icons,
    menu_icon="cast",
    default_index=current_index,
    orientation="horizontal",
    key="top_nav_menu",
    styles={
        "container": {"padding": "0!important", "background-color": "transparent"},
        "icon": {"color": "#2962FF", "font-size": "14px"},
        "nav-link": {
            "font-size": "14px",
            "text-align": "center",
            "margin": "0px",
            "padding": "10px 15px",
            "color": "#8B949E",
            "--hover-color": "#262730"
        },
        "nav-link-selected": {"background-color": "#2962FF", "color": "#FFFFFF"},
    }
)

# Update session state when selection changes
if selected_page != st.session_state.current_page:
    st.session_state.current_page = selected_page

# App branding - clean, professional header
st.markdown("""
<div style="margin-bottom: 1.5rem; margin-top: 1rem;">
    <h1 style="font-size: 1.5rem; font-weight: 700; color: #E6EDF3; margin: 0; letter-spacing: -0.02em;">
        QuantTerm Pro
    </h1>
    <p style="color: #6E7681; font-size: 0.875rem; margin: 0.25rem 0 0 0;">
        Professional Trading Terminal
    </p>
</div>
""", unsafe_allow_html=True)


# Page - Main Terminal (Home)
if selected_page == "Main Terminal":
    # Professional Page Header
    page_header(
        title="Main Terminal",
        subtitle="Your professional trading command center",
        badge=("Online", "online")
    )

    # Top Metrics Row 
    custom_strategies_count = len([k for k in STRATEGY_REGISTRY.keys() if '🎨' in k])

    metric_row([
        {
            "label": "System Status",
            "value": "Online",
            "delta": "All systems operational",
            "delta_type": "positive",
        },
        {
            "label": "Strategies Loaded",
            "value": str(len(STRATEGY_REGISTRY)),
            "delta": f"{custom_strategies_count} custom",
            "delta_type": "neutral",
        },
        {
            "label": "Portfolio" if multi_asset_mode else "Selected Asset",
            "value": f"{len(selected_tickers_multi)} Assets" if multi_asset_mode else ticker,
            "delta": f"EW · {timeframe} · {period}" if multi_asset_mode else f"{timeframe} · {period}",
            "delta_type": "neutral",
        },
        {
            "label": "AI Engine",
            "value": "Ready" if use_real_ai else "Mock",
            "delta": "Mixtral" if use_real_ai else "Fast Mode",
            "delta_type": "positive" if use_real_ai else "neutral",
        },
    ])

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Main Content - Two Column Layout
    main_col, side_col = st.columns([2, 1])

    with main_col:
        # Quick Backtest Card
        st.markdown("""
        <div class="section-card">
            <div class="section-title">Quick Backtest</div>
        """, unsafe_allow_html=True)

        st.markdown("Run a backtest with your current configuration settings.", unsafe_allow_html=True)
        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

        sl_col, tp_col = st.columns(2)
        with sl_col:
            quick_sl_pct = st.number_input("Stop Loss (%)", min_value=0.1, max_value=100.0, value=float(config.STOP_LOSS_PCT * 100), step=0.1, key="quick_sl") / 100
        with tp_col:
            quick_tp_pct = st.number_input("Take Profit (%)", min_value=0.1, max_value=1000.0, value=float(config.TAKE_PROFIT_PCT * 100), step=0.1, key="quick_tp") / 100

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

        if st.button("Run Backtest", type="primary", use_container_width=True, key="quick_backtest"):
            # Classify params by category 
            strategy_params = get_strategy_params(selected_strategy_class)
            signal_params = {k: np.array([param_values[k]])
                             for k, v in strategy_params.items()
                             if v.get('category', 'signal') == 'signal'}
            filter_params = {k: param_values[k]
                             for k, v in strategy_params.items()
                             if v.get('category') == 'signal_filter'}
            long_sl_params = {k: param_values[k]
                              for k, v in strategy_params.items()
                              if v.get('category') == 'long_sl'}
            short_sl_params = {k: param_values[k]
                               for k, v in strategy_params.items()
                               if v.get('category') == 'short_sl'}

            if multi_asset_mode:
                # Multi Asset Portfolio Path
                if len(selected_tickers_multi) < 2:
                    st.error("Please select at least 2 tickers for multi-asset backtest.")
                else:
                    try:
                        # Batch download all tickers
                        with st.spinner(f"Downloading data for {len(selected_tickers_multi)} assets..."):
                            price_data = vbt.YFData.download(
                                selected_tickers_multi,
                                period=period,
                                interval=timeframe,
                                missing_index='nan',
                                missing_columns='drop'
                            )

                            close_df = price_data.get('Close')
                            high_df = price_data.get('High')
                            low_df = price_data.get('Low')
                            volume_df = price_data.get('Volume')

                            # Flatten MultiIndex columns from VBT download
                            # (VBT creates tuples like ('GLD','GLD') instead of 'GLD')
                            if isinstance(close_df.columns, pd.MultiIndex):
                                close_df.columns = close_df.columns.get_level_values(0)
                                high_df.columns = high_df.columns.get_level_values(0)
                                low_df.columns = low_df.columns.get_level_values(0)
                                volume_df.columns = volume_df.columns.get_level_values(0)

                            # Forward-fill NaN gaps from different trading calendars
                            close_df = close_df.ffill()
                            high_df = high_df.ffill()
                            low_df = low_df.ffill()
                            volume_df = volume_df.ffill().fillna(0)

                            # Drop any remaining leading NaN rows (before first valid data)
                            close_df = close_df.dropna()
                            high_df = high_df.loc[close_df.index]
                            low_df = low_df.loc[close_df.index]
                            volume_df = volume_df.loc[close_df.index]

                            # Drop tickers that are entirely NaN (no data at all)
                            valid_cols = close_df.columns[close_df.notna().any()]
                            close_df = close_df[valid_cols]
                            high_df = high_df[valid_cols]
                            low_df = low_df[valid_cols]
                            volume_df = volume_df[valid_cols]

                            # Identify valid tickers (those that returned data)
                            valid_tickers = close_df.columns.tolist()
                            dropped = set(selected_tickers_multi) - set(valid_tickers)
                            if dropped:
                                st.warning(f"No data returned for: {', '.join(dropped)}")

                            if len(valid_tickers) < 2:
                                st.error("Need at least 2 tickers with valid data.")
                                st.stop()

                        # Fetch macro regime data if filter enabled
                        macro_regime = None
                        if use_macro_filter and MACRO_AVAILABLE:
                            try:
                                start_date = close_df.index[0]
                                end_date = close_df.index[-1]
                                macro_df = get_macro_data(start_date, end_date)
                                macro_regime = classify_macro_regime(macro_df)
                            except Exception as macro_err:
                                st.warning(f"Macro data unavailable: {macro_err}. Running without regime filter.")

                        # 3. Generate signals per ticker
                        signal_progress = st.progress(0, text="Generating signals...")
                        # Pass the cleaned DataFrames, not raw price_data
                        cleaned_price_data = {
                            'Close': close_df, 'High': high_df,
                            'Low': low_df, 'Volume': volume_df,
                        }
                        signals = generate_multi_asset_signals(
                            cleaned_price_data, valid_tickers,
                            selected_strategy_class, signal_params, filter_params,
                            progress_callback=lambda pct: signal_progress.progress(
                                pct, text=f"Signals: {pct*100:.0f}%"
                            ),
                            macro_regime=macro_regime,
                            safe_haven_action=safe_haven_action,
                        )
                        signal_progress.empty()

                        # Remove failed tickers
                        failed = signals['failed_tickers']
                        if failed:
                            st.warning(f"Signal generation failed for: {', '.join(failed)}")
                            valid_tickers = [t for t in valid_tickers if t not in failed]
                            close_df = close_df[valid_tickers]

                        if len(valid_tickers) < 2:
                            st.error("Not enough tickers with valid signals.")
                            st.stop()

                        long_entries_df = signals['long_entries'][valid_tickers]
                        short_entries_df = signals['short_entries'][valid_tickers]
                        long_exits_df = signals['long_exits'][valid_tickers]
                        short_exits_df = signals['short_exits'][valid_tickers]

                        # 3. Build sl_stop
                        has_per_side_sl = bool(long_sl_params) and bool(short_sl_params)
                        if has_per_side_sl:
                            long_sl_val = list(long_sl_params.values())[0] / 100.0
                            short_sl_val = list(short_sl_params.values())[0] / 100.0
                            sl_stop = np.where(short_entries_df.values, short_sl_val, long_sl_val)
                        else:
                            sl_stop = quick_sl_pct

                        # Build VBT Portfolio (one column per ticker)
                        with st.spinner("Running portfolio simulation..."):
                            vbt_freq = {'1m':'1T','5m':'5T','15m':'15T','1h':'1H',
                                        '4h':'4H','1d':'1D','1wk':'1W','1mo':'1MS'}.get(timeframe, timeframe)

                            pf = vbt.Portfolio.from_signals(
                                close_df,
                                entries=long_entries_df,
                                exits=long_exits_df,
                                short_entries=short_entries_df,
                                short_exits=short_exits_df,
                                sl_stop=sl_stop,
                                tp_stop=quick_tp_pct,
                                freq=vbt_freq,
                                init_cash=10000,
                                fees=0.001,
                            )

                        # Compute combined equal-weight portfolio
                        per_asset_returns = pf.returns()
                        per_asset_equity = pf.value()

                        # Flatten MultiIndex columns from VBT portfolio outputs
                        # (VBT can produce tuples like ('GLD','GLD') internally)
                        if isinstance(per_asset_returns.columns, pd.MultiIndex):
                            per_asset_returns.columns = per_asset_returns.columns.get_level_values(0)
                        if isinstance(per_asset_equity.columns, pd.MultiIndex):
                            per_asset_equity.columns = per_asset_equity.columns.get_level_values(0)

                        combined_returns = per_asset_returns.mean(axis=1)
                        combined_equity = (1 + combined_returns).cumprod() * 10000

                        # Portfolio-level metrics
                        portfolio_return = (combined_equity.iloc[-1] / combined_equity.iloc[0] - 1) * 100

                        # Annualized Sharpe
                        ann_factor = {'1d': 252, '1h': 252*6.5, '4h': 252*2,
                                      '5m': 252*6.5*12, '15m': 252*6.5*4,
                                      '1wk': 52, '1mo': 12}.get(timeframe, 252)
                        if combined_returns.std() > 0:
                            portfolio_sharpe = (combined_returns.mean() / combined_returns.std()) * np.sqrt(ann_factor)
                        else:
                            portfolio_sharpe = 0.0

                        # Max drawdown
                        running_max = combined_equity.cummax()
                        drawdown = (combined_equity - running_max) / running_max
                        portfolio_max_dd = drawdown.min() * 100

                        # Benchmark (SPY buy-and-hold)
                        benchmark_ticker = "SPY"
                        benchmark_equity = None
                        benchmark_return = 0.0
                        if benchmark_ticker in close_df.columns:
                            bm_close = close_df[benchmark_ticker]
                            benchmark_equity = (bm_close / bm_close.iloc[0]) * 10000
                            benchmark_return = (bm_close.iloc[-1] / bm_close.iloc[0] - 1) * 100
                        else:
                            try:
                                bm_data = vbt.YFData.download(benchmark_ticker, period=period, interval=timeframe)
                                bm_close = bm_data.get('Close')
                                common_idx = close_df.index.intersection(bm_close.index)
                                if len(common_idx) > 0:
                                    bm_close = bm_close.loc[common_idx]
                                    benchmark_equity = (bm_close / bm_close.iloc[0]) * 10000
                                    benchmark_return = (bm_close.iloc[-1] / bm_close.iloc[0] - 1) * 100
                            except Exception:
                                pass

                        # Per-asset stats
                        # Build a map from clean ticker name -> original pf column key
                        pf_columns = pf.wrapper.columns
                        pf_col_map = {}
                        for col in pf_columns:
                            clean = col[0] if isinstance(col, tuple) else col
                            pf_col_map[clean] = col

                        per_asset_total_returns = pf.total_return()
                        if isinstance(per_asset_total_returns, pd.Series) and isinstance(per_asset_total_returns.index, pd.MultiIndex):
                            per_asset_total_returns.index = per_asset_total_returns.index.get_level_values(0)

                        per_asset_stats_rows = []
                        for tkr in valid_tickers:
                            try:
                                tkr_return = (per_asset_total_returns[tkr] if isinstance(per_asset_total_returns, pd.Series)
                                              else per_asset_total_returns) * 100
                                pf_key = pf_col_map.get(tkr, tkr)
                                tkr_pf = pf[pf_key] if hasattr(pf, '__getitem__') else pf
                                tkr_trades = tkr_pf.trades.count()
                                tkr_winrate = 0
                                if tkr_trades > 0:
                                    wr = tkr_pf.trades.win_rate()
                                    tkr_winrate = wr * 100 if not np.isnan(wr) else 0
                                tkr_dd = tkr_pf.max_drawdown() * 100
                                per_asset_stats_rows.append({
                                    'Ticker': tkr, 'Return (%)': tkr_return,
                                    'Trades': int(tkr_trades), 'Win Rate (%)': tkr_winrate,
                                    'Max DD (%)': tkr_dd,
                                })
                            except Exception:
                                per_asset_stats_rows.append({
                                    'Ticker': tkr, 'Return (%)': 0,
                                    'Trades': 0, 'Win Rate (%)': 0, 'Max DD (%)': 0,
                                })

                        per_asset_stats = pd.DataFrame(per_asset_stats_rows)
                        traded_assets = per_asset_stats[per_asset_stats['Trades'] > 0]
                        avg_win_rate = traded_assets['Win Rate (%)'].mean() if len(traded_assets) > 0 else 0

                        # Portfolio optimization models with fallback
                        optimization = None
                        hrp_optimization = None
                        if 'Mean-Variance (Markowitz)' in opt_models:
                            try:
                                optimization = compute_markowitz_optimization(per_asset_returns, close_df, rebalance_freq=rebalance_freq)
                            except Exception as opt_err:
                                st.warning(f"Markowitz optimization skipped: {opt_err}")
                        if 'Hierarchical Risk Parity (HRP)' in opt_models:
                            try:
                                hrp_optimization = compute_hrp_optimization(per_asset_returns, close_df, rebalance_freq=rebalance_freq)
                            except Exception as hrp_err:
                                st.warning(f"HRP optimization skipped: {hrp_err}")

                        # Display results
                        results = {
                            'portfolio_return': portfolio_return,
                            'benchmark_return': benchmark_return,
                            'portfolio_sharpe': portfolio_sharpe if not np.isnan(portfolio_sharpe) else 0,
                            'portfolio_max_dd': portfolio_max_dd,
                            'num_assets': len(valid_tickers),
                            'total_trades': int(per_asset_stats['Trades'].sum()),
                            'avg_win_rate': avg_win_rate,
                            'combined_equity': combined_equity,
                            'per_asset_equity': per_asset_equity,
                            'benchmark_equity': benchmark_equity,
                            'per_asset_returns': per_asset_returns,
                            'per_asset_stats': per_asset_stats,
                            'failed_tickers': failed,
                            'optimization': optimization,
                            'hrp_optimization': hrp_optimization,
                            'macro_filtered': use_macro_filter and macro_regime is not None,
                            'macro_regime': macro_regime,
                            'skipped_entries': signals.get('skipped_entries'),
                        }

                        display_multi_asset_results(results)

                    except Exception as e:
                        st.error(f"Multi-Asset Backtest Error: {e}")

            else:
                # Existing single ticker path
                with st.spinner("Running backtest..."):
                    try:
                        # Download data
                        price_data = vbt.YFData.download(ticker, period=period, interval=timeframe)
                        close = price_data.get('Close')
                        high = price_data.get('High')
                        low = price_data.get('Low')
                        volume = price_data.get('Volume')

                        # Generate signals (single-value arrays -> (T,1) matrices)
                        sig = selected_strategy_class.generate_vectorized_signals(
                            close, high, low, volume, signal_params
                        )
                        long_entries  = sig['long_entries'].squeeze()
                        short_entries = sig['short_entries'].squeeze()
                        long_exits    = sig['long_exits'].squeeze()
                        short_exits   = sig['short_exits'].squeeze()

                        # Apply signal_filter
                        if filter_params:
                            adx_data = ta.adx(high, low, close, length=14)
                            adx_series = adx_data['ADX_14']
                            for fname, fval in filter_params.items():
                                adx_mask = adx_series.values >= fval
                                long_entries  = long_entries & adx_mask
                                short_entries = short_entries & adx_mask

                        # Optional AI confirmation filter
                        ai_log = []
                        if use_real_ai:
                            entry_count = int(long_entries.sum() + short_entries.sum())
                            if entry_count > 0:
                                st.info(f"AI (Mixtral) reviewing {entry_count} entry signals...")
                                ai_progress = st.progress(0, text="AI analyzing entries...")

                                open_prices = price_data.get('Open')
                                long_entries, short_entries, ai_log = apply_ai_filter(
                                    long_entries, short_entries,
                                    close, high, low, volume,
                                    selected_strategy_class, param_values,
                                    ai_confidence_threshold,
                                    open_prices=open_prices,
                                    progress_callback=lambda pct: ai_progress.progress(
                                        pct, text=f"AI analysis: {pct*100:.0f}%"
                                    ),
                                )

                                ai_progress.empty()
                                kept = sum(1 for r in ai_log if r['Kept'] == 'Yes')
                                rejected = sum(1 for r in ai_log if r['Kept'] == 'No')
                                st.success(
                                    f"AI filter complete: {kept} entries confirmed, "
                                    f"{rejected} entries rejected"
                                )
                            else:
                                st.info("No entry signals to review with AI")

                        # 4. Build sl_stop (per-side or global)
                        has_per_side_sl = bool(long_sl_params) and bool(short_sl_params)
                        if has_per_side_sl:
                            long_sl_val = list(long_sl_params.values())[0] / 100.0
                            short_sl_val = list(short_sl_params.values())[0] / 100.0
                            sl_stop = np.where(short_entries, short_sl_val, long_sl_val)
                        else:
                            sl_stop = quick_sl_pct

                        # 5. VectorBT Portfolio
                        vbt_freq = {'1m':'1T','5m':'5T','15m':'15T','1h':'1H',
                                    '4h':'4H','1d':'1D','1wk':'1W','1mo':'1MS'}.get(timeframe, timeframe)
                        pf = vbt.Portfolio.from_signals(
                            close,
                            entries=long_entries, exits=long_exits,
                            short_entries=short_entries, short_exits=short_exits,
                            sl_stop=sl_stop, tp_stop=quick_tp_pct,
                            freq=vbt_freq, init_cash=10000, fees=0.001,
                        )

                        # 6. Extract results
                        strategy_return = pf.total_return() * 100
                        buy_hold_return = ((close.iloc[-1] / close.iloc[0]) - 1) * 100
                        total_trades = pf.trades.count()
                        win_rate = pf.trades.win_rate() * 100 if total_trades > 0 else 0
                        sharpe = pf.sharpe_ratio() if total_trades > 0 else 0
                        max_dd = pf.max_drawdown() * 100

                        # Build trades_df from VectorBT records
                        trades_df = pd.DataFrame()
                        if total_trades > 0:
                            records = pf.trades.records_readable
                            trades_df = pd.DataFrame({
                                'EntryTime': records['Entry Timestamp'],
                                'EntryPrice': records['Avg Entry Price'],
                                'ExitTime': records['Exit Timestamp'],
                                'ExitPrice': records['Avg Exit Price'],
                                'ReturnPct': records['Return'],
                            })

                        # Build df_plot for candlestick chart
                        df_plot = pd.DataFrame({
                            'Date': close.index,
                            'Open': price_data.get('Open').values,
                            'High': high.values,
                            'Low': low.values,
                            'Close': close.values,
                            'Volume': volume.values,
                        })

                        equity_series = pf.value()

                        results = {
                            'strategy_return': strategy_return,
                            'buy_hold_return': buy_hold_return,
                            'win_rate': win_rate,
                            'sharpe': sharpe if not np.isnan(sharpe) else 0,
                            'max_drawdown': max_dd,
                            'total_trades': total_trades,
                            'df_plot': df_plot,
                            'trades_df': trades_df,
                            'equity_series': equity_series,
                        }

                        display_backtest_results(results, ticker)

                        # Show AI decision log if AI was used
                        if use_real_ai and ai_log:
                            with st.expander("AI Decision Log", expanded=False):
                                ai_df = pd.DataFrame(ai_log)
                                st.dataframe(ai_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Backtest Error: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Features Card
        st.markdown("""
        <div class="section-card" style="margin-top: 1rem;">
            <div class="section-title">Platform Features</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-top: 0.5rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #3FB950;">✓</span>
                    <span style="color: #8B949E; font-size: 0.875rem;">Multi-Strategy Backtesting</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #3FB950;">✓</span>
                    <span style="color: #8B949E; font-size: 0.875rem;">Parallel Grid Search</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #3FB950;">✓</span>
                    <span style="color: #8B949E; font-size: 0.875rem;">Visual Strategy Builder</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #3FB950;">✓</span>
                    <span style="color: #8B949E; font-size: 0.875rem;">Live Paper Trading</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #3FB950;">✓</span>
                    <span style="color: #8B949E; font-size: 0.875rem;">Portfolio Optimization (MPT)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #3FB950;">✓</span>
                    <span style="color: #8B949E; font-size: 0.875rem;">Options Analysis & Scanner</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with side_col:
        # Current Configuration Card
        st.markdown(f"""
        <div class="section-card">
            <div class="section-title">Current Configuration</div>
            <div style="margin-top: 0.5rem;">
                <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #21262D;">
                    <span style="color: #6E7681; font-size: 0.8125rem;">{"Portfolio" if multi_asset_mode else "Asset"}</span>
                    <span style="font-family: 'JetBrains Mono', monospace; color: #E6EDF3; font-size: 0.875rem;">{f"{len(selected_tickers_multi)} Assets (EW)" if multi_asset_mode else ticker}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #21262D;">
                    <span style="color: #6E7681; font-size: 0.8125rem;">Timeframe</span>
                    <span style="font-family: 'JetBrains Mono', monospace; color: #E6EDF3; font-size: 0.875rem;">{timeframe}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #21262D;">
                    <span style="color: #6E7681; font-size: 0.8125rem;">Period</span>
                    <span style="font-family: 'JetBrains Mono', monospace; color: #E6EDF3; font-size: 0.875rem;">{period}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #21262D;">
                    <span style="color: #6E7681; font-size: 0.8125rem;">Stop Loss</span>
                    <span style="font-family: 'JetBrains Mono', monospace; color: #F85149; font-size: 0.875rem;">{quick_sl_pct*100:.1f}%</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.5rem 0;">
                    <span style="color: #6E7681; font-size: 0.8125rem;">Take Profit</span>
                    <span style="font-family: 'JetBrains Mono', monospace; color: #3FB950; font-size: 0.875rem;">{quick_tp_pct*100:.1f}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Strategy Card
        st.markdown(f"""
        <div class="section-card" style="margin-top: 1rem;">
            <div class="section-title">Active Strategy</div>
            <div style="margin-top: 0.75rem;">
                <div style="font-family: 'JetBrains Mono', monospace; color: #E6EDF3; font-size: 0.875rem; word-break: break-word;">
                    {strategy_name}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # System Info Card
        st.markdown(f"""
        <div class="section-card" style="margin-top: 1rem;">
            <div class="section-title">System Info</div>
            <div style="margin-top: 0.5rem;">
                <div style="display: flex; justify-content: space-between; padding: 0.375rem 0;">
                    <span style="color: #6E7681; font-size: 0.8125rem;">CPU Cores</span>
                    <span style="font-family: 'JetBrains Mono', monospace; color: #E6EDF3; font-size: 0.875rem;">{multiprocessing.cpu_count()}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.375rem 0;">
                    <span style="color: #6E7681; font-size: 0.8125rem;">Core Strategies</span>
                    <span style="font-family: 'JetBrains Mono', monospace; color: #E6EDF3; font-size: 0.875rem;">8</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.375rem 0;">
                    <span style="color: #6E7681; font-size: 0.8125rem;">Custom Strategies</span>
                    <span style="font-family: 'JetBrains Mono', monospace; color: #E6EDF3; font-size: 0.875rem;">{custom_strategies_count}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# Page - Grid Search Optimizer
elif selected_page == "Grid Search":
    page_header(
        title="Grid Search Optimizer",
        subtitle="Find optimal parameters by testing multiple combinations"
    )

    # In multi-asset mode, grid search optimizes on the full portfolio
    if multi_asset_mode:
        grid_multi_asset = True
        grid_ticker = f"Portfolio ({len(selected_tickers_multi)} assets)"
        st.info(f"Optimizing parameters for **equal-weight portfolio** across "
                f"**{len(selected_tickers_multi)} assets**. "
                f"Each parameter combo is tested on the combined portfolio.")
    else:
        grid_multi_asset = False
        grid_ticker = ticker

    # Get ALL optimizable parameters from the strategy
    all_strategy_params = get_strategy_params(selected_strategy_class)

    # Check if this strategy uses per-side stop losses 
    _has_per_side_sl_ui = (
        any(p.get('category') == 'long_sl' for p in all_strategy_params.values()) and
        any(p.get('category') == 'short_sl' for p in all_strategy_params.values())
    )

    # Row 1 - SL and TP
    col1, col2 = st.columns(2)

    with col1:
        if _has_per_side_sl_ui:
            st.markdown("### Stop Loss Range")
            st.info("Per-side Stop Loss is optimized via strategy parameters below. Global SL is disabled.")
            sl_min = sl_max = sl_step = 0
        else:
            st.markdown("### Stop Loss Range")
            sl_min = st.number_input("Min SL (%)", 0.1, 100.0, 0.5, 0.1, key="opt_sl_min")
            sl_max = st.number_input("Max SL (%)", 0.1, 100.0, 2.0, 0.1, key="opt_sl_max")
            sl_step = st.number_input("Step (%)", 0.1, 10.0, 0.5, 0.1, key="opt_sl_step")

    with col2:
        st.markdown("### Take Profit Range")
        tp_min = st.number_input("Min TP (%)", 0.1, 1000.0, 1.0, 0.1, key="opt_tp_min")
        tp_max = st.number_input("Max TP (%)", 0.1, 1000.0, 4.0, 0.1, key="opt_tp_max")
        tp_step = st.number_input("Step (%)", 0.1, 20.0, 0.5, 0.1, key="opt_tp_step")

    # Row 2 - all strategy specific parameters (dynamic)
    param_opt_configs = {}  # Stores optimization config for each enabled param

    if all_strategy_params:
        param_cols = st.columns(len(all_strategy_params))

        for idx, (param_name, param_config) in enumerate(all_strategy_params.items()):
            with param_cols[idx]:
                label = param_config["name"]
                enabled = st.checkbox(f"{label} Range", value=True, key=f"opt_enable_{param_name}")

                if enabled:
                    opt_min_default = param_config["optimize_min"]
                    opt_max_default = param_config["optimize_max"]
                    opt_step_default = param_config["optimize_step"]
                    abs_min = param_config["min"]
                    abs_max = param_config["max"]

                    if param_config["type"] == "float":
                        p_min = st.number_input(
                            f"Min {label}",
                            float(abs_min), float(abs_max),
                            float(opt_min_default), float(opt_step_default),
                            key=f"opt_{param_name}_min"
                        )
                        p_max = st.number_input(
                            f"Max {label}",
                            float(abs_min), float(abs_max),
                            float(opt_max_default), float(opt_step_default),
                            key=f"opt_{param_name}_max"
                        )
                        step_max = max(float(abs_max - abs_min), float(opt_step_default))
                        p_step = st.number_input(
                            "Step",
                            float(opt_step_default), step_max,
                            float(opt_step_default), float(opt_step_default),
                            key=f"opt_{param_name}_step"
                        )
                    else:
                        p_min = st.number_input(
                            f"Min {label}",
                            int(abs_min), int(abs_max),
                            int(opt_min_default), int(opt_step_default),
                            key=f"opt_{param_name}_min"
                        )
                        p_max = st.number_input(
                            f"Max {label}",
                            int(abs_min), int(abs_max),
                            int(opt_max_default), int(opt_step_default),
                            key=f"opt_{param_name}_max"
                        )
                        step_max = max(int(abs_max - abs_min), int(opt_step_default))
                        p_step = st.number_input(
                            "Step",
                            int(opt_step_default), step_max,
                            int(opt_step_default), int(opt_step_default),
                            key=f"opt_{param_name}_step"
                        )

                    param_opt_configs[param_name] = {
                        'label': label,
                        'min': p_min,
                        'max': p_max,
                        'step': p_step,
                        'type': param_config["type"],
                        'category': param_config.get("category", "signal")
                    }
    else:
        st.info("This strategy has no optimizable parameters. Only Stop Loss and Take Profit will be optimized.")

    optimize_btn = st.button("START OPTIMIZATION", type="primary", key="optimize_run")

    if optimize_btn:
        # Generate SL/TP ranges (convert from % to decimal for vectorbt)
        tp_range = np.arange(tp_min, tp_max + tp_step, tp_step) / 100.0

        # Generate ranges for each enabled strategy parameter
        strategy_param_ranges = {}
        strategy_param_labels = {}
        for p_name, cfg in param_opt_configs.items():
            strategy_param_ranges[p_name] = np.arange(cfg['min'], cfg['max'] + cfg['step'], cfg['step'])
            strategy_param_labels[p_name] = cfg['label']

        # Classify parameters by category
        signal_cfgs = {k: v for k, v in param_opt_configs.items()
                       if v.get('category', 'signal') == 'signal'}
        filter_cfgs = {k: v for k, v in param_opt_configs.items()
                       if v.get('category') == 'signal_filter'}
        long_sl_cfgs = {k: v for k, v in param_opt_configs.items()
                        if v.get('category') == 'long_sl'}
        short_sl_cfgs = {k: v for k, v in param_opt_configs.items()
                         if v.get('category') == 'short_sl'}

        has_per_side_sl = bool(long_sl_cfgs) and bool(short_sl_cfgs)
        has_signal_filter = bool(filter_cfgs)

        # Signal params for meshgrid
        signal_param_names = list(signal_cfgs.keys())
        signal_param_arrays = [strategy_param_ranges[name] for name in signal_param_names]

        # Extra (non-signal) params for outer loop
        extra_param_names = []
        extra_param_arrays = []
        for k in filter_cfgs:
            extra_param_names.append(k)
            extra_param_arrays.append(strategy_param_ranges[k])
        for k in long_sl_cfgs:
            extra_param_names.append(k)
            extra_param_arrays.append(strategy_param_ranges[k])
        for k in short_sl_cfgs:
            extra_param_names.append(k)
            extra_param_arrays.append(strategy_param_ranges[k])

        extra_combos = list(itertools.product(*extra_param_arrays)) if extra_param_arrays else [()]
        n_extra_combos = len(extra_combos)

        # Global SL range (only used when no per-side SL)
        if has_per_side_sl:
            sl_range = np.array([0.0])  # placeholder, not used
        else:
            sl_range = np.arange(sl_min, sl_max + sl_step, sl_step) / 100.0

        # Build optimization summary
        opt_names_display = []
        if not has_per_side_sl:
            opt_names_display.append("Stop Loss")
        opt_names_display.append("Take Profit")
        opt_names_display += [param_opt_configs[p]['label'] for p in param_opt_configs]
        opt_summary = ", ".join(opt_names_display)

        # Map Yahoo Finance intervals to pandas frequency strings for VectorBT
        vbt_freq_map = {
            '1m': '1T', '5m': '5T', '15m': '15T',
            '1h': '1H', '4h': '4H',
            '1d': '1D', '1wk': '1W', '1mo': '1MS',
        }
        vbt_freq = vbt_freq_map.get(timeframe, timeframe)
        all_results = []

        try:
          if grid_multi_asset:
            # Multi Asset Portfolio Grid Search

            # Fetch data
            with st.spinner(f"Downloading {len(selected_tickers_multi)} assets ({period}, {timeframe})..."):
                price_data = vbt.YFData.download(
                    selected_tickers_multi, period=period, interval=timeframe,
                    missing_index='nan', missing_columns='drop'
                )
                close_df = price_data.get('Close')
                high_df = price_data.get('High')
                low_df = price_data.get('Low')
                volume_df = price_data.get('Volume')

                # Flatten MultiIndex columns
                if isinstance(close_df.columns, pd.MultiIndex):
                    close_df.columns = close_df.columns.get_level_values(0)
                    high_df.columns = high_df.columns.get_level_values(0)
                    low_df.columns = low_df.columns.get_level_values(0)
                    volume_df.columns = volume_df.columns.get_level_values(0)

                # Forward-fill and clean
                close_df = close_df.ffill()
                high_df = high_df.ffill()
                low_df = low_df.ffill()
                volume_df = volume_df.ffill().fillna(0)
                close_df = close_df.dropna()
                high_df = high_df.loc[close_df.index]
                low_df = low_df.loc[close_df.index]
                volume_df = volume_df.loc[close_df.index]

                valid_cols = close_df.columns[close_df.notna().any()]
                close_df = close_df[valid_cols]
                high_df = high_df[valid_cols]
                low_df = low_df[valid_cols]
                volume_df = volume_df[valid_cols]
                grid_valid_tickers = close_df.columns.tolist()

                dropped = set(selected_tickers_multi) - set(grid_valid_tickers)
                if dropped:
                    st.warning(f"No data for: {', '.join(dropped)}")
                if len(grid_valid_tickers) < 2:
                    st.error("Need at least 2 tickers with valid data.")
                    st.stop()

            T = len(close_df)
            M = len(grid_valid_tickers)

            # Fetch macro regime data if filter is enabled
            grid_macro_regime = None
            if use_macro_filter and MACRO_AVAILABLE:
                try:
                    macro_df = get_macro_data(close_df.index[0], close_df.index[-1])
                    grid_macro_regime = classify_macro_regime(macro_df)
                except Exception as macro_err:
                    st.warning(f"Macro data unavailable for Grid Search: {macro_err}. Running without regime filter.")

            # Generate signal matrices per ticker
            signal_param_ranges = {k: strategy_param_ranges[k] for k in signal_param_names}
            signal_progress = st.progress(0, text="Generating signal matrices...")
            per_ticker_signals = {}
            grid_failed_tickers = []

            for idx_t, tkr in enumerate(grid_valid_tickers):
                try:
                    sig = selected_strategy_class.generate_vectorized_signals(
                        close_df[tkr], high_df[tkr], low_df[tkr], volume_df[tkr],
                        signal_param_ranges
                    )
                    per_ticker_signals[tkr] = sig
                except Exception:
                    grid_failed_tickers.append(tkr)
                signal_progress.progress((idx_t + 1) / M, text=f"Signals: {tkr}")
            signal_progress.empty()

            if grid_failed_tickers:
                st.warning(f"Signal gen failed for: {', '.join(grid_failed_tickers)}")
                grid_valid_tickers = [t for t in grid_valid_tickers if t not in grid_failed_tickers]
                close_df = close_df[grid_valid_tickers]
                M = len(grid_valid_tickers)

            # Get param_columns from first ticker's signals
            first_sig = per_ticker_signals[grid_valid_tickers[0]]
            param_columns = first_sig.get('param_columns', {})
            n_combos = first_sig['long_entries'].shape[1] if first_sig['long_entries'].ndim > 1 else 1

            # Build super matrices (T, M*N)
            # Columns ordered: [tkr0_combo0, tkr0_combo1, ..., tkr1_combo0, ...]
            # So reshape(T, M, N) groups by ticker first, combo second
            base_long_entries_super = np.column_stack([
                per_ticker_signals[tkr]['long_entries'].reshape(T, -1) for tkr in grid_valid_tickers
            ])
            base_short_entries_super = np.column_stack([
                per_ticker_signals[tkr]['short_entries'].reshape(T, -1) for tkr in grid_valid_tickers
            ])
            long_exits_super = np.column_stack([
                per_ticker_signals[tkr]['long_exits'].reshape(T, -1) for tkr in grid_valid_tickers
            ])
            short_exits_super = np.column_stack([
                per_ticker_signals[tkr]['short_exits'].reshape(T, -1) for tkr in grid_valid_tickers
            ])

            # Build super close: repeat each ticker's close n_combos times
            close_super_arr = np.column_stack([
                np.tile(close_df[tkr].values[:, None], (1, n_combos)) for tkr in grid_valid_tickers
            ])
            close_super = pd.DataFrame(close_super_arr, index=close_df.index)

            # ADX per-ticker for signal filter
            adx_per_ticker = {}
            if has_signal_filter:
                for tkr in grid_valid_tickers:
                    adx_data = ta.adx(high_df[tkr], low_df[tkr], close_df[tkr], length=14)
                    adx_per_ticker[tkr] = adx_data['ADX_14']

            # Annualization factor for Sharpe
            ann_factor = {'1d': 252, '1h': 252*6.5, '4h': 252*2,
                          '5m': 252*6.5*12, '15m': 252*6.5*4,
                          '1wk': 52, '1mo': 12}.get(timeframe, 252)

            n_signal_combos = n_combos  # for results table
            if has_per_side_sl:
                valid_risk_combos = len(tp_range)
            else:
                valid_risk_combos = sum(1 for sl in sl_range for tp in tp_range if tp > sl)
            total_iterations = n_extra_combos * valid_risk_combos
            total_combos = n_signal_combos * total_iterations

            st.info(f"Testing **{total_combos:,}** portfolio parameter combinations "
                    f"({n_signal_combos:,} signal x {n_extra_combos} extra x {valid_risk_combos} risk) "
                    f"across **{M} assets**")

            # Loop extra paramaters and risk parameters
            progress_bar = st.progress(0)
            status_text = st.empty()
            counter = 0
            start_time = time.time()

            for extra_vals in extra_combos:
                extra_dict = dict(zip(extra_param_names, extra_vals)) if extra_param_names else {}

                # Copy base super entries
                long_entries = base_long_entries_super.copy()
                short_entries = base_short_entries_super.copy()

                # Apply ADX filter per-ticker (each ticker's columns in super matrix)
                if has_signal_filter:
                    col_offset = 0
                    for tkr in grid_valid_tickers:
                        for fname, fval in extra_dict.items():
                            if param_opt_configs.get(fname, {}).get('category') == 'signal_filter':
                                adx_vals = adx_per_ticker[tkr].values[:, None]
                                adx_mask = adx_vals >= fval
                                long_entries[:, col_offset:col_offset+n_combos] &= adx_mask
                                short_entries[:, col_offset:col_offset+n_combos] &= adx_mask
                        col_offset += n_combos

                # Apply macro regime filter to super matrices
                long_exits = long_exits_super.copy()
                if grid_macro_regime is not None:
                    grid_safe_havens = SAFE_HAVENS if MACRO_AVAILABLE else {'GLD', 'TLT'}
                    risk_off_aligned = grid_macro_regime.reindex(close_df.index, method='ffill').fillna(False).values
                    risk_on_aligned = ~risk_off_aligned
                    col_offset = 0
                    for tkr in grid_valid_tickers:
                        if tkr.upper() not in grid_safe_havens:
                            risk_on_col = risk_on_aligned[:, None]  # (T, 1) broadcasts to (T, n_combos)
                            risk_off_col = risk_off_aligned[:, None]
                            long_entries[:, col_offset:col_offset+n_combos] &= risk_on_col
                            long_exits[:, col_offset:col_offset+n_combos] |= risk_off_col
                        # Safe haven rotation in grid search
                        if safe_haven_action in ('TLT', 'GLD') and tkr == safe_haven_action:
                            risk_off_col = risk_off_aligned[:, None]
                            long_entries[:, col_offset:col_offset+n_combos] |= risk_off_col
                            long_exits[:, col_offset:col_offset+n_combos] &= ~risk_off_col
                        col_offset += n_combos
                else:
                    long_exits = long_exits_super

                if has_per_side_sl:
                    long_sl_val = next(extra_dict[k] / 100.0 for k in long_sl_cfgs if k in extra_dict)
                    short_sl_val = next(extra_dict[k] / 100.0 for k in short_sl_cfgs if k in extra_dict)
                    sl_stop_arr = np.full(long_entries.shape, long_sl_val)
                    sl_stop_arr[short_entries] = short_sl_val

                    for tp in tp_range:
                        pf = vbt.Portfolio.from_signals(
                            close_super, entries=long_entries, exits=long_exits,
                            short_entries=short_entries, short_exits=short_exits_super,
                            sl_stop=sl_stop_arr, tp_stop=tp,
                            freq=vbt_freq, init_cash=10000, fees=0.001,
                        )

                        try:
                            # Reshape returns to (T, M, N) and compute portfolio metrics
                            returns_arr = pf.returns().values  # (T, M*N)
                            returns_3d = returns_arr.reshape(T, M, n_combos)
                            combined_2d = returns_3d.mean(axis=1)  # (T, N)

                            portfolio_ret = (np.prod(1 + combined_2d, axis=0) - 1) * 100
                            means = combined_2d.mean(axis=0)
                            stds = combined_2d.std(axis=0)
                            portfolio_sharpe = np.where(stds > 0, (means / stds) * np.sqrt(ann_factor), 0.0)

                            cum_ret = np.cumprod(1 + combined_2d, axis=0)
                            run_max = np.maximum.accumulate(cum_ret, axis=0)
                            dd = (cum_ret - run_max) / run_max
                            portfolio_max_dd = dd.min(axis=0) * 100

                            trades_arr = pf.trades.count().values.reshape(M, n_combos)
                            portfolio_trades = trades_arr.sum(axis=0).astype(int)

                            wr_arr = np.nan_to_num(pf.trades.win_rate().values, nan=0.0).reshape(M, n_combos)
                            portfolio_wr = np.where(
                                trades_arr.sum(axis=0) > 0,
                                (wr_arr * trades_arr).sum(axis=0) / trades_arr.sum(axis=0),
                                0
                            ) * 100
                        except (ValueError, Exception):
                            counter += 1
                            progress_bar.progress(min(counter / total_iterations, 1.0))
                            continue

                        batch_df = pd.DataFrame({
                            'Stop Loss (%)': np.nan,
                            'Take Profit (%)': np.round(tp * 100, 2),
                            'Return (%)': portfolio_ret,
                            'Win Rate (%)': portfolio_wr,
                            'Sharpe Ratio': portfolio_sharpe,
                            'Max Drawdown (%)': portfolio_max_dd,
                            'Total Trades': portfolio_trades,
                        })
                        for col_label, col_values in param_columns.items():
                            batch_df[col_label] = col_values
                        for ep_name, ep_val in extra_dict.items():
                            batch_df[strategy_param_labels[ep_name]] = ep_val

                        all_results.append(batch_df)
                        counter += 1
                        progress_bar.progress(min(counter / total_iterations, 1.0))
                        status_text.caption(f"Processed {counter}/{total_iterations} (TP={tp*100:.1f}%)")

                else:
                    for sl in sl_range:
                        for tp in tp_range:
                            if tp <= sl:
                                continue

                            pf = vbt.Portfolio.from_signals(
                                close_super, entries=long_entries, exits=long_exits,
                                short_entries=short_entries, short_exits=short_exits_super,
                                sl_stop=sl, tp_stop=tp,
                                freq=vbt_freq, init_cash=10000, fees=0.001,
                            )

                            try:
                                returns_arr = pf.returns().values
                                returns_3d = returns_arr.reshape(T, M, n_combos)
                                combined_2d = returns_3d.mean(axis=1)

                                portfolio_ret = (np.prod(1 + combined_2d, axis=0) - 1) * 100
                                means = combined_2d.mean(axis=0)
                                stds = combined_2d.std(axis=0)
                                portfolio_sharpe = np.divide(means, stds, out=np.zeros_like(means), where=stds > 0) * np.sqrt(ann_factor)

                                cum_ret = np.cumprod(1 + combined_2d, axis=0)
                                run_max = np.maximum.accumulate(cum_ret, axis=0)
                                dd = (cum_ret - run_max) / run_max
                                portfolio_max_dd = dd.min(axis=0) * 100

                                trades_arr = pf.trades.count().values.reshape(M, n_combos)
                                portfolio_trades = trades_arr.sum(axis=0).astype(int)

                                wr_arr = np.nan_to_num(pf.trades.win_rate().values, nan=0.0).reshape(M, n_combos)
                                trades_sum = trades_arr.sum(axis=0)
                                portfolio_wr = np.divide(
                                    (wr_arr * trades_arr).sum(axis=0), trades_sum,
                                    out=np.zeros_like(trades_sum, dtype=float), where=trades_sum > 0
                                ) * 100
                            except (ValueError, Exception):
                                counter += 1
                                progress_bar.progress(min(counter / total_iterations, 1.0))
                                continue

                            batch_df = pd.DataFrame({
                                'Stop Loss (%)': np.round(sl * 100, 2),
                                'Take Profit (%)': np.round(tp * 100, 2),
                                'Return (%)': portfolio_ret,
                                'Win Rate (%)': portfolio_wr,
                                'Sharpe Ratio': portfolio_sharpe,
                                'Max Drawdown (%)': portfolio_max_dd,
                                'Total Trades': portfolio_trades,
                            })
                            for col_label, col_values in param_columns.items():
                                batch_df[col_label] = col_values
                            for ep_name, ep_val in extra_dict.items():
                                batch_df[strategy_param_labels[ep_name]] = ep_val

                            all_results.append(batch_df)
                            counter += 1
                            progress_bar.progress(min(counter / total_iterations, 1.0))
                            status_text.caption(f"Processed {counter}/{total_iterations} (SL={sl*100:.1f}%, TP={tp*100:.1f}%)")

            progress_bar.empty()
            status_text.empty()
            elapsed = time.time() - start_time

            # Aggregate and Display
            if all_results:
                df_results = pd.concat(all_results, ignore_index=True)
                df_results = df_results[df_results['Total Trades'] > 0]
                df_results = df_results.dropna(subset=['Return (%)'])
                df_results = df_results.sort_values('Return (%)', ascending=False)

                st.success(
                    f"Portfolio optimization complete in **{elapsed:.1f}s**! "
                    f"{len(df_results):,} valid configurations found "
                    f"(out of {total_combos:,} tested across {M} assets)."
                )

                if len(df_results) > 0:
                    st.markdown("### Top 10 Portfolio Configurations")

                    def highlight_best(row):
                        if row.name == 0:
                            return ['background-color: #1b5e20; font-weight: bold'] * len(row)
                        elif row.name < 3:
                            return ['background-color: #2e7d32'] * len(row)
                        return [''] * len(row)

                    display_cols = [c for c in df_results.columns if c != 'Stop Loss (%)'] if has_per_side_sl else df_results.columns.tolist()
                    styled = df_results[display_cols].head(10).reset_index(drop=True).style.apply(highlight_best, axis=1)
                    format_dict = {
                        'Return (%)': '{:.2f}', 'Win Rate (%)': '{:.1f}',
                        'Sharpe Ratio': '{:.2f}', 'Max Drawdown (%)': '{:.2f}',
                    }
                    if not has_per_side_sl:
                        format_dict['Stop Loss (%)'] = '{:.2f}'
                    format_dict['Take Profit (%)'] = '{:.2f}'
                    styled = styled.format(format_dict, na_rep='-')
                    st.dataframe(styled, use_container_width=True)

                    # Optimal Parameters
                    best = df_results.iloc[0]
                    st.markdown("### Optimal Portfolio Parameters")
                    opt_cols = st.columns(min(6, len(best)))
                    for i, (col_name, val) in enumerate(best.items()):
                        if col_name == 'Stop Loss (%)' and has_per_side_sl:
                            continue
                        with opt_cols[i % len(opt_cols)]:
                            fmt_val = f"{val:.2f}" if isinstance(val, float) else str(val)
                            st.metric(col_name, fmt_val)

                    # Robustness Heatmap (requires 2 parameters)
                    if len(signal_param_names) >= 2:
                        label_a = strategy_param_labels[signal_param_names[0]]
                        label_b = strategy_param_labels[signal_param_names[1]]
                        viable = df_results[df_results['Total Trades'] >= 10]
                        if len(viable) > 0:
                            heat_data = viable.groupby([label_a, label_b])['Return (%)'].mean().unstack()
                            if len(heat_data) > 1:
                                fig_heatmap = go.Figure(data=go.Heatmap(
                                    z=heat_data.values, x=[str(c) for c in heat_data.columns],
                                    y=[str(r) for r in heat_data.index],
                                    colorscale='Viridis', text=np.round(heat_data.values, 1),
                                    texttemplate='%{text}%', hovertemplate='%{y}, %{x}: %{z:.2f}%<extra></extra>'
                                ))
                                fig_heatmap.update_layout(
                                    template='plotly_dark',
                                    title=dict(text=f'Portfolio Robustness — {M} assets ({period}, {timeframe})', font=dict(color='#E6EDF3')),
                                    xaxis_title=label_b, yaxis_title=label_a,
                                    paper_bgcolor='#121212', plot_bgcolor='#1E1E1E', height=500,
                                )
                                st.plotly_chart(fig_heatmap, use_container_width=True, config={'displayModeBar': False})

                    # Strategy Manifold using UMAP
                    try:
                        fig_manifold = create_strategy_manifold(df_results)
                        if fig_manifold is not None:
                            st.markdown("### Strategy Manifold (UMAP)")
                            st.plotly_chart(fig_manifold, use_container_width=True, config={'displayModeBar': False})
                            st.caption("Clusters of bright dots indicate robust parameter zones. Isolated bright dots may indicate overfitting.")
                    except Exception as e:
                        st.warning(f"UMAP manifold skipped: {e}")

                    # Backtest with Optimal Parameters
                    st.subheader("Backtest with Optimal Parameters")

                    try:
                        opt_signal_params = {}
                        for p_name in signal_param_names:
                            label = param_opt_configs[p_name]['label']
                            opt_signal_params[p_name] = np.array([best[label]])

                        # Generate signals per-ticker with optimal params
                        opt_filter_params = {}
                        if has_signal_filter:
                            for fname, fcfg in filter_cfgs.items():
                                opt_filter_params[fname] = best[fcfg['label']]

                        cleaned_price_data = {
                            'Close': close_df, 'High': high_df,
                            'Low': low_df, 'Volume': volume_df,
                        }
                        opt_signals = generate_multi_asset_signals(
                            cleaned_price_data, grid_valid_tickers,
                            selected_strategy_class, opt_signal_params, opt_filter_params,
                            macro_regime=grid_macro_regime,
                            safe_haven_action=safe_haven_action,
                        )

                        opt_le = opt_signals['long_entries'][grid_valid_tickers]
                        opt_se = opt_signals['short_entries'][grid_valid_tickers]
                        opt_lx = opt_signals['long_exits'][grid_valid_tickers]
                        opt_sx = opt_signals['short_exits'][grid_valid_tickers]

                        opt_tp_pct = best['Take Profit (%)'] / 100.0
                        if has_per_side_sl:
                            opt_long_sl = best[list(long_sl_cfgs.values())[0]['label']] / 100.0
                            opt_short_sl = best[list(short_sl_cfgs.values())[0]['label']] / 100.0
                            opt_sl_stop = np.where(opt_se.values, opt_short_sl, opt_long_sl)
                        else:
                            opt_sl_stop = best['Stop Loss (%)'] / 100.0

                        opt_pf = vbt.Portfolio.from_signals(
                            close_df, entries=opt_le, exits=opt_lx,
                            short_entries=opt_se, short_exits=opt_sx,
                            sl_stop=opt_sl_stop, tp_stop=opt_tp_pct,
                            freq=vbt_freq, init_cash=10000, fees=0.001,
                        )

                        # Compute combined portfolio
                        opt_per_asset_returns = opt_pf.returns()
                        opt_per_asset_equity = opt_pf.value()
                        if isinstance(opt_per_asset_returns.columns, pd.MultiIndex):
                            opt_per_asset_returns.columns = opt_per_asset_returns.columns.get_level_values(0)
                        if isinstance(opt_per_asset_equity.columns, pd.MultiIndex):
                            opt_per_asset_equity.columns = opt_per_asset_equity.columns.get_level_values(0)

                        opt_combined_returns = opt_per_asset_returns.mean(axis=1)
                        opt_combined_equity = (1 + opt_combined_returns).cumprod() * 10000

                        opt_portfolio_return = (opt_combined_equity.iloc[-1] / opt_combined_equity.iloc[0] - 1) * 100
                        if opt_combined_returns.std() > 0:
                            opt_portfolio_sharpe = (opt_combined_returns.mean() / opt_combined_returns.std()) * np.sqrt(ann_factor)
                        else:
                            opt_portfolio_sharpe = 0.0
                        opt_run_max = opt_combined_equity.cummax()
                        opt_portfolio_max_dd = ((opt_combined_equity - opt_run_max) / opt_run_max).min() * 100

                        # Benchmark
                        bm_ticker = "SPY"
                        opt_bm_equity = None
                        opt_bm_return = 0.0
                        if bm_ticker in close_df.columns:
                            bm_c = close_df[bm_ticker]
                            opt_bm_equity = (bm_c / bm_c.iloc[0]) * 10000
                            opt_bm_return = (bm_c.iloc[-1] / bm_c.iloc[0] - 1) * 100

                        # Per asset stats
                        pf_columns = opt_pf.wrapper.columns
                        pf_col_map = {}
                        for col in pf_columns:
                            clean = col[0] if isinstance(col, tuple) else col
                            pf_col_map[clean] = col

                        opt_total_returns = opt_pf.total_return()
                        if isinstance(opt_total_returns, pd.Series) and isinstance(opt_total_returns.index, pd.MultiIndex):
                            opt_total_returns.index = opt_total_returns.index.get_level_values(0)

                        opt_stats_rows = []
                        for tkr in grid_valid_tickers:
                            try:
                                tr = (opt_total_returns[tkr] if isinstance(opt_total_returns, pd.Series) else opt_total_returns) * 100
                                pf_key = pf_col_map.get(tkr, tkr)
                                tp = opt_pf[pf_key] if hasattr(opt_pf, '__getitem__') else opt_pf
                                tc = tp.trades.count()
                                wr = tp.trades.win_rate() * 100 if tc > 0 and not np.isnan(tp.trades.win_rate()) else 0
                                dd = tp.max_drawdown() * 100
                                opt_stats_rows.append({'Ticker': tkr, 'Return (%)': tr, 'Trades': int(tc), 'Win Rate (%)': wr, 'Max DD (%)': dd})
                            except Exception:
                                opt_stats_rows.append({'Ticker': tkr, 'Return (%)': 0, 'Trades': 0, 'Win Rate (%)': 0, 'Max DD (%)': 0})

                        opt_stats = pd.DataFrame(opt_stats_rows)
                        traded = opt_stats[opt_stats['Trades'] > 0]
                        avg_wr = traded['Win Rate (%)'].mean() if len(traded) > 0 else 0

                        # Portfolio optimization models with fallback
                        grid_optimization = None
                        grid_hrp_optimization = None
                        if 'Mean-Variance (Markowitz)' in opt_models:
                            try:
                                grid_optimization = compute_markowitz_optimization(opt_per_asset_returns, close_df, rebalance_freq='Monthly')
                            except Exception:
                                grid_optimization = None
                        if 'Hierarchical Risk Parity (HRP)' in opt_models:
                            try:
                                grid_hrp_optimization = compute_hrp_optimization(opt_per_asset_returns, close_df, rebalance_freq='Monthly')
                            except Exception:
                                grid_hrp_optimization = None

                        opt_multi_results = {
                            'portfolio_return': opt_portfolio_return,
                            'benchmark_return': opt_bm_return,
                            'portfolio_sharpe': opt_portfolio_sharpe if not np.isnan(opt_portfolio_sharpe) else 0,
                            'portfolio_max_dd': opt_portfolio_max_dd,
                            'num_assets': M,
                            'total_trades': int(opt_stats['Trades'].sum()),
                            'avg_win_rate': avg_wr,
                            'combined_equity': opt_combined_equity,
                            'per_asset_equity': opt_per_asset_equity,
                            'benchmark_equity': opt_bm_equity,
                            'per_asset_returns': opt_per_asset_returns,
                            'per_asset_stats': opt_stats,
                            'failed_tickers': grid_failed_tickers,
                            'optimization': grid_optimization,
                            'hrp_optimization': grid_hrp_optimization,
                            'macro_filtered': use_macro_filter and grid_macro_regime is not None,
                            'macro_regime': grid_macro_regime,
                            'skipped_entries': opt_signals.get('skipped_entries'),
                        }
                        display_multi_asset_results(opt_multi_results)

                        # ============================================================
                        # WALK-FORWARD OUT-OF-SAMPLE TESTING
                        # ============================================================
                        if enable_walk_forward and len(grid_valid_tickers) >= 2:
                            st.markdown("---")
                            st.subheader("Walk-Forward Out-of-Sample Testing")
                            st.caption(
                                f"Rolling **{wf_train_years}-year train** / **{wf_test_years}-year test** windows. "
                                f"Each fold finds the max-Sharpe parameter set in-sample, then applies it "
                                f"frozen to the next unseen period. "
                                f"Green = realistic OOS performance. Orange = in-sample (overfitted) fit."
                            )

                            wf_prog = st.progress(0, text="Initializing walk-forward…")

                            # Use actual SL range only when not in per-side SL mode
                            wf_sl_range = (
                                np.array([0.02]) if has_per_side_sl
                                else sl_range
                            )

                            wf_results = run_walk_forward_backtest(
                                close_df, high_df, low_df, volume_df,
                                selected_strategy_class,
                                {k: strategy_param_ranges[k] for k in signal_param_names},
                                wf_sl_range, tp_range,
                                ann_factor, vbt_freq, grid_valid_tickers,
                                train_years=wf_train_years,
                                test_years=wf_test_years,
                                macro_regime=grid_macro_regime if use_macro_filter else None,
                                safe_haven_action=safe_haven_action,
                                progress_cb=lambda p, t: wf_prog.progress(p, text=t),
                            )
                            wf_prog.empty()

                            if wf_results and wf_results['n_folds'] > 0:
                                # Fold summary table
                                fold_df = pd.DataFrame(wf_results['folds'])
                                st.dataframe(fold_df, use_container_width=True, hide_index=True)

                                # IS vs OOS metrics side-by-side
                                st.markdown("#### In-Sample vs Out-of-Sample Performance")
                                col_is, col_oos = st.columns(2)

                                with col_is:
                                    st.markdown(
                                        "<p style='color:#ff6d00;font-weight:700;'>In-Sample (Train Windows)</p>",
                                        unsafe_allow_html=True
                                    )
                                    st.metric("Total Return", f"{wf_results['is_return']:.2f}%")
                                    st.metric("Sharpe Ratio", f"{wf_results['is_sharpe']:.2f}")
                                    st.metric("Max Drawdown",  f"{wf_results['is_max_dd']:.2f}%")

                                with col_oos:
                                    st.markdown(
                                        "<p style='color:#00e676;font-weight:700;'>Out-of-Sample (Test Windows)</p>",
                                        unsafe_allow_html=True
                                    )
                                    delta_ret    = wf_results['oos_return']  - wf_results['is_return']
                                    delta_sharpe = wf_results['oos_sharpe'] - wf_results['is_sharpe']
                                    st.metric(
                                        "Total Return", f"{wf_results['oos_return']:.2f}%",
                                        delta=f"{delta_ret:+.2f}% vs IS"
                                    )
                                    st.metric(
                                        "Sharpe Ratio", f"{wf_results['oos_sharpe']:.2f}",
                                        delta=f"{delta_sharpe:+.2f} vs IS"
                                    )
                                    st.metric("Max Drawdown", f"{wf_results['oos_max_dd']:.2f}%")

                                # IS vs OOS equity curve chart
                                fig_wf = go.Figure()
                                fig_wf.add_trace(go.Scatter(
                                    x=wf_results['is_equity'].index,
                                    y=wf_results['is_equity'].values,
                                    name=f"In-Sample (Sharpe: {wf_results['is_sharpe']:.2f})",
                                    line=dict(color='#ff6d00', width=2, dash='dash'),
                                ))
                                fig_wf.add_trace(go.Scatter(
                                    x=wf_results['oos_equity'].index,
                                    y=wf_results['oos_equity'].values,
                                    name=f"Out-of-Sample Net (Sharpe: {wf_results['oos_sharpe']:.2f})",
                                    line=dict(color='#00e676', width=2.5),
                                    fill='tozeroy',
                                    fillcolor='rgba(0, 230, 118, 0.05)',
                                ))
                                # Task 3: gross (fee-free) overlay
                                if wf_results.get('oos_equity_gross') is not None:
                                    fig_wf.add_trace(go.Scatter(
                                        x=wf_results['oos_equity_gross'].index,
                                        y=wf_results['oos_equity_gross'].values,
                                        name='OOS Gross (0% fees)',
                                        line=dict(color='#b0bec5', width=1.5, dash='dot'),
                                    ))
                                fig_wf.update_layout(
                                    template='plotly_dark',
                                    paper_bgcolor='#121212',
                                    plot_bgcolor='#1E1E1E',
                                    font=dict(color='#E6EDF3'),
                                    title=dict(
                                        text=(
                                            f"Walk-Forward: IS vs OOS Equity "
                                            f"({wf_results['n_folds']} folds, "
                                            f"{wf_train_years}yr train / {wf_test_years}yr test)"
                                        ),
                                        font=dict(size=15, color='#E6EDF3')
                                    ),
                                    height=450,
                                    hovermode='x unified',
                                    legend=dict(
                                        orientation='h', yanchor='bottom',
                                        y=1.02, xanchor='right', x=1
                                    ),
                                    yaxis=dict(title='Portfolio Value ($)'),
                                )
                                fig_wf.update_xaxes(
                                    showgrid=True, gridwidth=1,
                                    gridcolor='rgba(255,255,255,0.1)'
                                )
                                fig_wf.update_yaxes(
                                    showgrid=True, gridwidth=1,
                                    gridcolor='rgba(255,255,255,0.1)'
                                )
                                st.plotly_chart(
                                    fig_wf, use_container_width=True,
                                    config={'displayModeBar': False}
                                )

                                # Edge retention verdict
                                if wf_results['is_sharpe'] > 0:
                                    edge_pct = (
                                        wf_results['oos_sharpe'] /
                                        wf_results['is_sharpe'] * 100
                                    )
                                    if edge_pct >= 70:
                                        st.success(
                                            f"Edge retention: OOS Sharpe is **{edge_pct:.0f}%** "
                                            f"of IS Sharpe across {wf_results['n_folds']} folds — "
                                            f"strategy shows good out-of-sample robustness."
                                        )
                                    elif edge_pct >= 40:
                                        st.warning(
                                            f"Partial edge retention: OOS Sharpe is **{edge_pct:.0f}%** "
                                            f"of IS Sharpe — some parameter sensitivity detected. "
                                            f"Consider widening optimization ranges."
                                        )
                                    else:
                                        st.error(
                                            f"Edge degradation: OOS Sharpe is only **{edge_pct:.0f}%** "
                                            f"of IS Sharpe — high overfitting risk. "
                                            f"Consider fewer parameters, shorter optimization windows, "
                                            f"or a simpler strategy."
                                        )
                                else:
                                    st.info(
                                        "In-sample Sharpe ≤ 0 — walk-forward cannot assess edge retention."
                                    )

                                # Task 4: Hysteresis sensitivity heatmap
                                if (use_macro_filter and MACRO_AVAILABLE
                                        and wf_results.get('oos_base_signals')
                                        and grid_macro_regime is not None):
                                    st.markdown("#### Hysteresis Parameter Sensitivity")
                                    st.caption(
                                        "OOS Sharpe across a grid of entry-band × exit-band values. "
                                        "A broad green plateau around ★ confirms the chosen ±0.10 "
                                        "thresholds are in a robust region — not a lucky single point."
                                    )
                                    with st.spinner("Running sensitivity sweep (25 combos)…"):
                                        sens_matrix, sens_lt, sens_ut = compute_hysteresis_sensitivity(
                                            macro_df,
                                            wf_results['oos_base_signals'],
                                            ann_factor, vbt_freq,
                                        )
                                    fig_sens = create_hysteresis_heatmap(
                                        sens_matrix, sens_lt, sens_ut,
                                        current_lt=-0.10, current_ut=0.10,
                                    )
                                    st.plotly_chart(
                                        fig_sens, use_container_width=True,
                                        config={'displayModeBar': False},
                                    )

                            else:
                                st.warning(
                                    f"Not enough historical data for walk-forward with "
                                    f"{wf_train_years}yr train + {wf_test_years}yr test windows. "
                                    f"Select a longer data period or reduce window sizes."
                                )

                    except Exception as e:
                        st.error(f"Optimal portfolio backtest failed: {e}")
                        st.exception(e)

                else:
                    st.warning("No valid results found. Try widening parameter ranges.")

            else:
                st.warning("No valid results found. Try widening parameter ranges or changing assets/timeframe.")

          else:
            # Single ticker grid search

            # Fetch Data 
            with st.spinner(f"Downloading {grid_ticker} data ({period}, {timeframe})..."):
                price_data = vbt.YFData.download(grid_ticker, period=period, interval=timeframe)
                close = price_data.get('Close')
                high = price_data.get('High')
                low = price_data.get('Low')
                volume = price_data.get('Volume')

            # ADX for signal_filter params 
            adx_series = None
            if has_signal_filter:
                adx_data = ta.adx(high, low, close, length=14)
                adx_series = adx_data['ADX_14']

            # Build Signal Matrix
            signal_param_ranges = {k: strategy_param_ranges[k] for k in signal_param_names}

            if hasattr(selected_strategy_class, 'generate_vectorized_signals'):
                sig = selected_strategy_class.generate_vectorized_signals(
                    close, high, low, volume, signal_param_ranges
                )
                base_long_entries  = sig['long_entries']
                base_short_entries = sig['short_entries']
                long_exits         = sig['long_exits']
                short_exits        = sig['short_exits']
                param_columns      = sig['param_columns']
            else:
                # Fallback: RSI+EMA logic for strategies without vectorized signals
                rsi = vbt.RSI.run(close, window=14).rsi
                ema = vbt.MA.run(close, window=200, ewm=True).ma
                trend_up = close > ema
                trend_down = close < ema

                if len(signal_param_arrays) >= 2:
                    grid_a, grid_b = np.meshgrid(signal_param_arrays[0], signal_param_arrays[1])
                    flat_a = grid_a.flatten()
                    flat_b = grid_b.flatten()
                elif len(signal_param_arrays) == 1:
                    flat_a = signal_param_arrays[0]
                    flat_b = np.full_like(flat_a, 70.0)
                else:
                    flat_a = np.array([30.0])
                    flat_b = np.array([70.0])

                rsi_2d        = rsi.values[:, None]
                trend_up_2d   = trend_up.values[:, None]
                trend_down_2d = trend_down.values[:, None]

                base_long_entries  = trend_up_2d  & (rsi_2d < flat_a)
                base_short_entries = trend_down_2d & (rsi_2d > flat_b)
                long_exits    = trend_down_2d
                short_exits   = trend_up_2d
                param_columns = {}
                if len(signal_param_names) >= 2:
                    param_columns[strategy_param_labels[signal_param_names[0]]] = flat_a
                    param_columns[strategy_param_labels[signal_param_names[1]]] = flat_b
                elif len(signal_param_names) == 1:
                    param_columns[strategy_param_labels[signal_param_names[0]]] = flat_a

            n_signal_combos = base_long_entries.shape[1]

            # Count total combos for progress
            if has_per_side_sl:
                valid_risk_combos = len(tp_range)
            else:
                valid_risk_combos = sum(1 for sl in sl_range for tp in tp_range if tp > sl)
            total_iterations = n_extra_combos * valid_risk_combos
            total_combos = n_signal_combos * total_iterations

            st.info(f"Testing **{total_combos:,}** parameter combinations "
                    f"({n_signal_combos:,} signal × {n_extra_combos} extra × {valid_risk_combos} risk)")
            st.caption(f"VectorBT Accelerated | Optimizing: {opt_summary}")

            # Loop: extra params × risk params over signal matrix 
            progress_bar = st.progress(0)
            status_text = st.empty()
            counter = 0
            start_time = time.time()

            for extra_vals in extra_combos:
                # Unpack extra param values
                extra_dict = dict(zip(extra_param_names, extra_vals)) if extra_vals else {}

                # Apply ADX signal filter (mask entries in choppy markets)
                long_entries = base_long_entries.copy()
                short_entries = base_short_entries.copy()

                for fname, fval in extra_dict.items():
                    if param_opt_configs[fname]['category'] == 'signal_filter':
                        adx_mask = (adx_series.values[:, None] >= fval)
                        long_entries = long_entries & adx_mask
                        short_entries = short_entries & adx_mask

                if has_per_side_sl:
                    # Extract per-side SL values
                    long_sl_val = next(extra_dict[k] / 100.0 for k in long_sl_cfgs if k in extra_dict)
                    short_sl_val = next(extra_dict[k] / 100.0 for k in short_sl_cfgs if k in extra_dict)

                    # Build 2D sl_stop array: long_sl at long entries, short_sl at short entries
                    sl_stop_arr = np.full(long_entries.shape, long_sl_val)
                    sl_stop_arr[short_entries] = short_sl_val

                    for tp in tp_range:
                        pf = vbt.Portfolio.from_signals(
                            close,
                            entries=long_entries,
                            exits=long_exits,
                            short_entries=short_entries,
                            short_exits=short_exits,
                            sl_stop=sl_stop_arr,
                            tp_stop=tp,
                            freq=vbt_freq,
                            init_cash=10000,
                            fees=0.001,
                        )

                        try:
                            total_return  = pf.total_return()
                            sharpe        = pf.sharpe_ratio()
                            trades_count  = pf.trades.count()
                            win_rate      = pf.trades.win_rate()
                            max_dd        = pf.max_drawdown()
                        except (ValueError, Exception):
                            counter += 1
                            progress_bar.progress(min(counter / total_iterations, 1.0))
                            continue

                        batch_df = pd.DataFrame({
                            'Stop Loss (%)':    np.nan,
                            'Take Profit (%)':  np.round(tp * 100, 2),
                            'Return (%)':       total_return.values * 100,
                            'Win Rate (%)':     win_rate.values * 100,
                            'Sharpe Ratio':     sharpe.values,
                            'Max Drawdown (%)': max_dd.values * 100,
                            'Total Trades':     trades_count.values.astype(int),
                        })

                        # Attach signal param columns
                        for col_label, col_values in param_columns.items():
                            batch_df[col_label] = col_values

                        # Attach extra param columns
                        for ep_name, ep_val in extra_dict.items():
                            batch_df[strategy_param_labels[ep_name]] = ep_val

                        all_results.append(batch_df)

                        counter += 1
                        progress_bar.progress(min(counter / total_iterations, 1.0))
                        status_text.caption(
                            f"Processed {counter}/{total_iterations} "
                            f"(TP={tp*100:.1f}%, "
                            + ", ".join(f"{strategy_param_labels[k]}={v}" for k, v in extra_dict.items())
                            + ")"
                        )

                else:
                    # Original behavior: global SL × TP loop
                    for sl in sl_range:
                        for tp in tp_range:
                            if tp <= sl:
                                continue

                            pf = vbt.Portfolio.from_signals(
                                close,
                                entries=long_entries,
                                exits=long_exits,
                                short_entries=short_entries,
                                short_exits=short_exits,
                                sl_stop=sl,
                                tp_stop=tp,
                                freq=vbt_freq,
                                init_cash=10000,
                                fees=0.001,
                            )

                            try:
                                total_return  = pf.total_return()
                                sharpe        = pf.sharpe_ratio()
                                trades_count  = pf.trades.count()
                                win_rate      = pf.trades.win_rate()
                                max_dd        = pf.max_drawdown()
                            except (ValueError, Exception):
                                counter += 1
                                progress_bar.progress(min(counter / total_iterations, 1.0))
                                continue

                            batch_df = pd.DataFrame({
                                'Stop Loss (%)':    np.round(sl * 100, 2),
                                'Take Profit (%)':  np.round(tp * 100, 2),
                                'Return (%)':       total_return.values * 100,
                                'Win Rate (%)':     win_rate.values * 100,
                                'Sharpe Ratio':     sharpe.values,
                                'Max Drawdown (%)': max_dd.values * 100,
                                'Total Trades':     trades_count.values.astype(int),
                            })

                            # Attach signal param columns
                            for col_label, col_values in param_columns.items():
                                batch_df[col_label] = col_values

                            # Attach extra param columns
                            for ep_name, ep_val in extra_dict.items():
                                batch_df[strategy_param_labels[ep_name]] = ep_val

                            all_results.append(batch_df)

                            counter += 1
                            progress_bar.progress(min(counter / total_iterations, 1.0))
                            status_text.caption(
                                f"Processed {counter}/{total_iterations} "
                                f"(SL={sl*100:.1f}%, TP={tp*100:.1f}%)"
                            )

            progress_bar.empty()
            status_text.empty()
            elapsed = time.time() - start_time

            # Aggregate and Display
            if all_results:
                df_results = pd.concat(all_results, ignore_index=True)
                df_results = df_results[df_results['Total Trades'] > 0]
                df_results = df_results.dropna(subset=['Return (%)'])
                df_results = df_results.sort_values('Return (%)', ascending=False)

                st.success(
                    f"VectorBT optimization complete in **{elapsed:.1f}s**! "
                    f"{len(df_results):,} valid configurations found "
                    f"(out of {total_combos:,} tested)."
                )

                if len(df_results) > 0:
                    st.markdown("### Top 10 Configurations")

                    def highlight_best(row):
                        if row.name == 0:
                            return ['background-color: #1b5e20; font-weight: bold'] * len(row)
                        elif row.name < 3:
                            return ['background-color: #2e7d32'] * len(row)
                        else:
                            return [''] * len(row)

                    format_dict = {
                        'Return (%)': '{:.2f}',
                        'Win Rate (%)': '{:.1f}',
                        'Sharpe Ratio': '{:.2f}',
                        'Max Drawdown (%)': '{:.2f}',
                        'Take Profit (%)': '{:.1f}'
                    }
                    if not has_per_side_sl:
                        format_dict['Stop Loss (%)'] = '{:.1f}'
                    else:
                        if 'Stop Loss (%)' in df_results.columns:
                            df_results = df_results.drop(columns=['Stop Loss (%)'])

                    # Add format for each optimized strategy parameter
                    for p_name, cfg in param_opt_configs.items():
                        if cfg['type'] == 'float':
                            format_dict[cfg['label']] = '{:.2f}'
                        else:
                            format_dict[cfg['label']] = '{:.0f}'

                    styled_df = df_results.head(10).style.apply(highlight_best, axis=1).format(format_dict)
                    st.dataframe(styled_df, use_container_width=True)

                    # Best parameters - dynamic columns
                    best = df_results.iloc[0]
                    st.markdown("### Optimal Parameters")

                    # Only show params that were actually optimized (have columns in df_results)s
                    active_params = {p: cfg for p, cfg in param_opt_configs.items() if cfg['label'] in df_results.columns}

                    if has_per_side_sl:
                        n_metric_cols = 1 + len(active_params) + 1  # TP + each param + Return
                        metric_cols = st.columns(n_metric_cols)
                        metric_cols[0].metric("Take Profit", f"{best['Take Profit (%)']:.1f}%")

                        for i, (p_name, cfg) in enumerate(active_params.items()):
                            label = cfg['label']
                            val = best[label]
                            if cfg['type'] == 'float':
                                metric_cols[1 + i].metric(label, f"{val:.2f}")
                            else:
                                metric_cols[1 + i].metric(label, f"{int(val)}")

                        metric_cols[-1].metric("Expected Return", f"{best['Return (%)']:.2f}%")
                    else:
                        n_metric_cols = 2 + len(active_params) + 1  # SL + TP + each param + Return
                        metric_cols = st.columns(n_metric_cols)
                        metric_cols[0].metric("Stop Loss", f"{best['Stop Loss (%)']:.1f}%")
                        metric_cols[1].metric("Take Profit", f"{best['Take Profit (%)']:.1f}%")

                        for i, (p_name, cfg) in enumerate(active_params.items()):
                            label = cfg['label']
                            val = best[label]
                            if cfg['type'] == 'float':
                                metric_cols[2 + i].metric(label, f"{val:.2f}")
                            else:
                                metric_cols[2 + i].metric(label, f"{int(val)}")

                        metric_cols[-1].metric("Expected Return", f"{best['Return (%)']:.2f}%")

                    
                    # Requires two strategy parameters for the X and Y axes
                    if len(signal_param_names) >= 2:
                        st.markdown("---")
                        st.markdown("### Strategy Robustness Heatmap")
                        st.caption(
                            "Average return for each signal-parameter pair, "
                            "aggregated across **all** SL/TP combinations. "
                            "Bright zones are robust — they win regardless of risk settings."
                        )

                        label_a = strategy_param_labels[signal_param_names[0]]
                        label_b = strategy_param_labels[signal_param_names[1]]

                        # Filter for viable strategies provided enough trades
                        viable = df_results[df_results['Total Trades'] > 10].copy()

                        if not viable.empty:
                            heatmap_data = (
                                viable
                                .groupby([label_a, label_b])['Return (%)']
                                .mean()
                                .unstack()
                            )

                            fig_heatmap = go.Figure(data=go.Heatmap(
                                z=heatmap_data.values,
                                x=heatmap_data.columns,
                                y=heatmap_data.index,
                                colorscale='Viridis',
                                colorbar=dict(
                                    title=dict(text='Avg Return (%)', font=dict(color='#E6EDF3')),
                                    tickfont=dict(color='#E6EDF3'),
                                ),
                                hovertemplate=(
                                    f'{label_b}: %{{x}}<br>'
                                    f'{label_a}: %{{y}}<br>'
                                    'Avg Return: %{z:.2f}%<extra></extra>'
                                ),
                            ))

                            fig_heatmap.update_layout(
                                template='plotly_dark',
                                title=dict(
                                    text=f'Robustness Heatmap — {grid_ticker} ({period}, {timeframe})',
                                    font=dict(color='#E6EDF3'),
                                ),
                                xaxis_title=label_b,
                                yaxis_title=label_a,
                                width=600,
                                height=600,
                                paper_bgcolor='#121212',
                                plot_bgcolor='#1E1E1E',
                                xaxis=dict(
                                    tickfont=dict(color='#8B949E'),
                                    titlefont=dict(color='#E6EDF3'),
                                    constrain='domain',
                                ),
                                yaxis=dict(
                                    tickfont=dict(color='#8B949E'),
                                    titlefont=dict(color='#E6EDF3'),
                                    scaleanchor='x',
                                    constrain='domain',
                                ),
                            )

                            st.plotly_chart(
                                fig_heatmap,
                                use_container_width=False,
                                config={'displayModeBar': False},
                            )
                        else:
                            st.info("Not enough strategies with >10 trades to build a heatmap.")

                    # Parallel Coordinates Plot
                    st.markdown("---")
                    st.subheader("Parallel Coordinates Plot")

                    parcoord_cols = (
                        [cfg['label'] for cfg in param_opt_configs.values()]
                        + ['Stop Loss (%)', 'Take Profit (%)', 'Return (%)']
                    )
                    parcoord_cols = [c for c in parcoord_cols if c in df_results.columns]

                    dimensions = [
                        dict(
                            range=[df_results[col].min(), df_results[col].max()],
                            label=col,
                            values=df_results[col],
                        )
                        for col in parcoord_cols
                    ]

                    fig_parcoord = go.Figure(
                        data=go.Parcoords(
                            line=dict(
                                color=df_results['Return (%)'],
                                colorscale='Viridis',
                                showscale=True,
                                cmin=df_results['Return (%)'].min(),
                                cmax=df_results['Return (%)'].max(),
                            ),
                            dimensions=dimensions,
                        )
                    )

                    fig_parcoord.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='#121212',
                        plot_bgcolor='#1E1E1E',
                        font=dict(color='#E6EDF3'),
                        margin=dict(l=80, r=80, t=80, b=30),
                    )

                    st.plotly_chart(fig_parcoord, use_container_width=True)

                    # Strategy Manifold using UMAPs
                    try:
                        fig_manifold = create_strategy_manifold(df_results)
                        if fig_manifold is not None:
                            st.markdown("### Strategy Manifold (UMAP)")
                            st.plotly_chart(fig_manifold, use_container_width=True, config={'displayModeBar': False})
                            st.caption("Clusters of bright dots indicate robust parameter zones. Isolated bright dots may indicate overfitting.")
                    except Exception as e:
                        st.warning(f"UMAP manifold skipped: {e}")

                    # Backtest with Optimal Parameters
                    st.markdown("---")
                    st.subheader("Backtest with Optimal Parameters")

                    try:
                        # Extract optimal signal params as single-value arrays
                        opt_signal_params = {}
                        for p_name in signal_param_names:
                            label = param_opt_configs[p_name]['label']
                            opt_signal_params[p_name] = np.array([best[label]])

                        # Generate signals with optimal params
                        if hasattr(selected_strategy_class, 'generate_vectorized_signals'):
                            opt_sig = selected_strategy_class.generate_vectorized_signals(
                                close, high, low, volume, opt_signal_params
                            )
                            opt_long_entries  = opt_sig['long_entries'].squeeze()
                            opt_short_entries = opt_sig['short_entries'].squeeze()
                            opt_long_exits    = opt_sig['long_exits'].squeeze()
                            opt_short_exits   = opt_sig['short_exits'].squeeze()
                        else:
                            # Fallback uses RSI and EMA for custom strategies
                            rsi = vbt.RSI.run(close, window=14).rsi
                            ema = vbt.MA.run(close, window=200, ewm=True).ma
                            trend_up = close > ema
                            trend_down = close < ema

                            opt_val_a = list(opt_signal_params.values())[0][0] if opt_signal_params else 30.0
                            opt_val_b = list(opt_signal_params.values())[1][0] if len(opt_signal_params) > 1 else 70.0

                            opt_long_entries  = (trend_up.values  & (rsi.values < opt_val_a))
                            opt_short_entries = (trend_down.values & (rsi.values > opt_val_b))
                            opt_long_exits    = trend_down.values
                            opt_short_exits   = trend_up.values

                        # Apply signal filter if applicable
                        if has_signal_filter and adx_series is not None:
                            for fname, fcfg in filter_cfgs.items():
                                fval = best[fcfg['label']]
                                adx_mask = adx_series.values >= fval
                                opt_long_entries  = opt_long_entries & adx_mask
                                opt_short_entries = opt_short_entries & adx_mask

                        # Build SL stop
                        opt_tp_pct = best['Take Profit (%)'] / 100.0
                        if has_per_side_sl:
                            opt_long_sl_val = best[list(long_sl_cfgs.values())[0]['label']] / 100.0
                            opt_short_sl_val = best[list(short_sl_cfgs.values())[0]['label']] / 100.0
                            opt_sl_stop = np.where(opt_short_entries, opt_short_sl_val, opt_long_sl_val)
                        else:
                            opt_sl_stop = best['Stop Loss (%)'] / 100.0

                        # Run VectorBT portfolio
                        opt_pf = vbt.Portfolio.from_signals(
                            close,
                            entries=opt_long_entries, exits=opt_long_exits,
                            short_entries=opt_short_entries, short_exits=opt_short_exits,
                            sl_stop=opt_sl_stop, tp_stop=opt_tp_pct,
                            freq=vbt_freq, init_cash=10000, fees=0.001,
                        )

                        # Extract results
                        opt_strategy_return = opt_pf.total_return() * 100
                        opt_buy_hold_return = ((close.iloc[-1] / close.iloc[0]) - 1) * 100
                        opt_total_trades = opt_pf.trades.count()
                        opt_win_rate = opt_pf.trades.win_rate() * 100 if opt_total_trades > 0 else 0
                        opt_sharpe = opt_pf.sharpe_ratio() if opt_total_trades > 0 else 0
                        opt_max_dd = opt_pf.max_drawdown() * 100

                        # Build trades_df
                        opt_trades_df = pd.DataFrame()
                        if opt_total_trades > 0:
                            opt_records = opt_pf.trades.records_readable
                            opt_trades_df = pd.DataFrame({
                                'EntryTime': opt_records['Entry Timestamp'],
                                'EntryPrice': opt_records['Avg Entry Price'],
                                'ExitTime': opt_records['Exit Timestamp'],
                                'ExitPrice': opt_records['Avg Exit Price'],
                                'ReturnPct': opt_records['Return'],
                            })

                        # Build df_plot for candlestick chart
                        opt_df_plot = pd.DataFrame({
                            'Date': close.index,
                            'Open': price_data.get('Open').values,
                            'High': high.values,
                            'Low': low.values,
                            'Close': close.values,
                            'Volume': volume.values,
                        })

                        opt_equity = opt_pf.value()

                        opt_results = {
                            'strategy_return': opt_strategy_return,
                            'buy_hold_return': opt_buy_hold_return,
                            'win_rate': opt_win_rate,
                            'sharpe': opt_sharpe if not np.isnan(opt_sharpe) else 0,
                            'max_drawdown': opt_max_dd,
                            'total_trades': opt_total_trades,
                            'df_plot': opt_df_plot,
                            'trades_df': opt_trades_df,
                            'equity_series': opt_equity,
                        }

                        display_backtest_results(opt_results, grid_ticker)

                    except Exception as e:
                        st.error(f"Optimal backtest failed: {e}")
                        st.exception(e)

                else:
                    st.warning("No valid results found. Try widening parameter ranges or changing the asset/timeframe.")

            else:
                st.warning("No valid results found. Try widening parameter ranges or changing the asset/timeframe.")

        except Exception as e:
            st.error(f"VectorBT optimization failed: {e}")
            st.exception(e)

            # Aggregate and Display (fallback on partial results)
            if all_results:
                df_results = pd.concat(all_results, ignore_index=True)
                df_results = df_results[df_results['Total Trades'] > 0]
                df_results = df_results.dropna(subset=['Return (%)'])
                df_results = df_results.sort_values('Return (%)', ascending=False)

                st.success(
                    f"Partial results recovered! "
                    f"{len(df_results):,} valid configurations found."
                )

                if len(df_results) > 0:
                    st.markdown("### Top 10 Configurations")

                    def highlight_best(row):
                        if row.name == 0:
                            return ['background-color: #1b5e20; font-weight: bold'] * len(row)
                        elif row.name < 3:
                            return ['background-color: #2e7d32'] * len(row)
                        else:
                            return [''] * len(row)

                    format_dict = {
                        'Return (%)': '{:.2f}',
                        'Win Rate (%)': '{:.1f}',
                        'Sharpe Ratio': '{:.2f}',
                        'Max Drawdown (%)': '{:.2f}',
                        'Take Profit (%)': '{:.1f}'
                    }
                    if not has_per_side_sl:
                        format_dict['Stop Loss (%)'] = '{:.1f}'
                    else:
                        if 'Stop Loss (%)' in df_results.columns:
                            df_results = df_results.drop(columns=['Stop Loss (%)'])

                    for p_name, cfg in param_opt_configs.items():
                        if cfg['type'] == 'float':
                            format_dict[cfg['label']] = '{:.2f}'
                        else:
                            format_dict[cfg['label']] = '{:.0f}'

                    styled_df = df_results.head(10).style.apply(highlight_best, axis=1).format(format_dict)
                    st.dataframe(styled_df, use_container_width=True)

                    # Best parameters - dynamic columns
                    best = df_results.iloc[0]
                    st.markdown("### Optimal Parameters")

                    active_params = {p: cfg for p, cfg in param_opt_configs.items() if cfg['label'] in df_results.columns}

                    if has_per_side_sl:
                        n_metric_cols = 1 + len(active_params) + 1
                        metric_cols = st.columns(n_metric_cols)
                        metric_cols[0].metric("Take Profit", f"{best['Take Profit (%)']:.1f}%")

                        for i, (p_name, cfg) in enumerate(active_params.items()):
                            label = cfg['label']
                            val = best[label]
                            if cfg['type'] == 'float':
                                metric_cols[1 + i].metric(label, f"{val:.2f}")
                            else:
                                metric_cols[1 + i].metric(label, f"{int(val)}")

                        metric_cols[-1].metric("Expected Return", f"{best['Return (%)']:.2f}%")
                    else:
                        n_metric_cols = 2 + len(active_params) + 1
                        metric_cols = st.columns(n_metric_cols)
                        metric_cols[0].metric("Stop Loss", f"{best['Stop Loss (%)']:.1f}%")
                        metric_cols[1].metric("Take Profit", f"{best['Take Profit (%)']:.1f}%")

                        for i, (p_name, cfg) in enumerate(active_params.items()):
                            label = cfg['label']
                            val = best[label]
                            if cfg['type'] == 'float':
                                metric_cols[2 + i].metric(label, f"{val:.2f}")
                            else:
                                metric_cols[2 + i].metric(label, f"{int(val)}")

                        metric_cols[-1].metric("Expected Return", f"{best['Return (%)']:.2f}%")

                else:
                    st.warning("No valid results found. Try widening parameter ranges or changing the asset/timeframe.")

            else:
                st.warning("No valid results found. Try widening parameter ranges or changing the asset/timeframe.")

        except Exception as e:
            st.error(f"VectorBT optimization failed: {e}")
            st.exception(e)


# ============================================================================
# PAGE: STRATEGY BUILDER
# ============================================================================
elif selected_page == "Strategy Builder":
    page_header(
        title="Strategy Builder",
        subtitle="Build custom strategies visually — no coding required"
    )

    st.success("✨ **Fully Functional!** Create strategies with:")
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.write("- Visual strategy designer")
        st.write("- Indicator-based entry rules")
    with col_info2:
        st.write("- Optimizable parameters")
        st.write("- Auto-generated Python code")
    with col_info3:
        st.write("- Save to `custom_strategies.py`")
        st.write("- Instant reload in dropdown")

    st.markdown("---")

    # Placeholder for future implementation
    st.markdown("### 🎨 Strategy Designer")

    col1, col2 = st.columns([2, 1])

    with col1:
        strategy_name = st.text_input("Strategy Name", placeholder="e.g., Super RSI Momentum")

        st.markdown("#### Entry Rules")

        # Expandable sections for adding rules
        with st.expander("Add Entry Rule #1"):
            indicator1 = st.selectbox("Indicator", ["RSI", "MACD", "SMA_Cross", "EMA_Trend", "Volume_Spike"], key="ind1")
            operator1 = st.selectbox("Operator", ["<", ">", "Crosses Above", "Crosses Below"], key="op1")
            threshold1 = st.number_input("Threshold", value=30.0, key="thresh1")

        with st.expander("Add Entry Rule #2 (Optional)"):
            indicator2 = st.selectbox("Indicator", ["RSI", "MACD", "SMA_Cross", "EMA_Trend", "Volume_Spike"], key="ind2")
            operator2 = st.selectbox("Operator", ["<", ">", "Crosses Above", "Crosses Below"], key="op2")
            threshold2 = st.number_input("Threshold", value=50.0, key="thresh2")

        st.markdown("#### Optimizable Parameters")
        st.caption("Define which parameters users can optimize in Grid Search")

        with st.expander("Add Optimizable Parameter"):
            param_name = st.text_input("Parameter Name", placeholder="e.g., RSI_THRESHOLD")
            param_label = st.text_input("Display Label", placeholder="e.g., RSI Threshold")
            param_min = st.number_input("Min Value", value=10.0)
            param_max = st.number_input("Max Value", value=90.0)
            param_default = st.number_input("Default Value", value=30.0)

    with col2:
        st.markdown("#### Generated Code Preview")

        if strategy_name:
            # Sanitize strategy name to create valid Python class name
            class_name = sanitize_strategy_name(strategy_name)

            # Generate strategy code template (used for both preview and save)
            strategy_code = f'''

class {class_name}:
    """
    {strategy_name} - Custom Strategy
    Auto-generated by Strategy Builder
    """

    @staticmethod
    def get_optimizable_params():
        return {{
            "{param_name if param_name else 'RSI_THRESHOLD'}": {{
                "name": "{param_label if param_label else 'RSI Threshold'}",
                "type": "{'float' if isinstance(param_default, float) else 'int'}",
                "min": {int(param_min)},
                "max": {int(param_max)},
                "step": 1,
                "default": {int(param_default)},
                "optimize_min": {int(param_min)},
                "optimize_max": {int(param_max)},
                "optimize_step": 5,
                "description": "Primary optimization parameter"
            }}
        }}

    def __init__(self, **kwargs):
        self.rsi_period = 14
        self.{param_name.lower() if param_name else 'rsi_threshold'} = kwargs.get('{param_name if param_name else 'RSI_THRESHOLD'}', {int(param_default)})
        logger.info(f"{class_name} initialized")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Calculate {indicator1}
        if "{indicator1}" == "RSI":
            df['RSI'] = ta.rsi(df['Close'], length=self.rsi_period)
        elif "{indicator1}" == "MACD":
            macd = ta.macd(df['Close'])
            df['MACD'] = macd.iloc[:, 0]
            df['MACD_Signal'] = macd.iloc[:, 1]
        elif "{indicator1}" == "SMA_Cross":
            df['SMA_20'] = ta.sma(df['Close'], length=20)
            df['SMA_50'] = ta.sma(df['Close'], length=50)
        elif "{indicator1}" == "EMA_Trend":
            df['EMA_20'] = ta.ema(df['Close'], length=20)
        elif "{indicator1}" == "Volume_Spike":
            df['Volume_SMA'] = ta.sma(df['Volume'], length=20)
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

        return df

    def check_hard_filters(self, row: pd.Series) -> bool:
        try:
            # Rule 1: {indicator1} {operator1} {threshold1}
            if "{indicator1}" == "RSI":
                condition1 = row['RSI'] {'<' if operator1 == '<' else '>'} {threshold1}
            elif "{indicator1}" == "MACD":
                condition1 = row['MACD'] {'<' if operator1 == '<' else '>'} row['MACD_Signal']
            elif "{indicator1}" == "SMA_Cross":
                condition1 = row['SMA_20'] {'<' if operator1 == '<' else '>'} row['SMA_50']
            elif "{indicator1}" == "EMA_Trend":
                condition1 = row['Close'] {'<' if operator1 == '<' else '>'} row['EMA_20']
            elif "{indicator1}" == "Volume_Spike":
                condition1 = row['Volume_Ratio'] {'<' if operator1 == '<' else '>'} {threshold1 / 100}
            else:
                condition1 = True

            # Rule 2 (optional)
            condition2 = True

            passed = condition1 and condition2

            if passed:
                logger.info(f"✓ {class_name} filters PASSED")

            return passed

        except Exception as e:
            logger.error(f"Error in {class_name} filter: {{e}}")
            return False

    def generate_prompt(self, row: pd.Series, lookback_rows: Optional[pd.DataFrame] = None) -> str:
        return f"""Symbol: {{config.TICKER}}
Price: ${{row['Close']:.2f}}
Strategy: {strategy_name}

This is a custom strategy. Analyze the setup and decide BUY or HOLD.
"""

    def get_position_size(self, price: float, account_value: float) -> int:
        dollar_amount = min(config.POSITION_SIZE, account_value * 0.95)
        return max(1, int(dollar_amount / price))
'''

            st.code(strategy_code, language="python")
        else:
            st.info("Enter a strategy name to see code preview")

    st.markdown("---")

    # Save button
    if st.button("Save Strategy", type="primary", disabled=not strategy_name):
        # Validation
        if not strategy_name:
            st.error("Please enter a strategy name!")
        else:
            try:
                # Sanitize strategy name to create valid Python class name
                class_name = sanitize_strategy_name(strategy_name)

                # Check if strategy already exists
                import os
                custom_file = "custom_strategies.py"

                # Read existing file to check for duplicates
                if os.path.exists(custom_file):
                    with open(custom_file, 'r') as f:
                        existing_content = f.read()
                        if f"class {class_name}:" in existing_content:
                            st.warning(f"Strategy '{class_name}' already exists! Please use a different name.")
                            st.stop()

                # Append to custom_strategies.py
                with open(custom_file, 'a', encoding='utf-8') as f:
                    f.write(strategy_code)

                st.success(f"Strategy '{class_name}' saved successfully!")
                st.toast("Strategy Saved! Reloading dashboard...", icon="🎉")

                # Wait a moment and reload
                time.sleep(1)
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error saving strategy: {e}")

    st.markdown("---")

    # Manage existing custom strategies
    st.markdown("### Manage Custom Strategies")

    # Get list of custom strategies only
    custom_strategy_list = []
    for display_name, strategy_info in STRATEGY_REGISTRY.items():
        if "🎨" in display_name and "class_name" in strategy_info:
            custom_strategy_list.append({
                "display_name": display_name,
                "class_name": strategy_info["class_name"]
            })

    if custom_strategy_list:
        with st.expander("Delete a Custom Strategy"):
            # Create dropdown with display names
            strategy_to_delete = st.selectbox(
                "Select strategy to delete",
                options=[s["display_name"] for s in custom_strategy_list],
                key="delete_strategy_select"
            )

            # Find the class name for the selected strategy
            selected_class_name = None
            for s in custom_strategy_list:
                if s["display_name"] == strategy_to_delete:
                    selected_class_name = s["class_name"]
                    break

            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("Delete Strategy", type="secondary", key="delete_btn"):
                    if selected_class_name:
                        with st.spinner(f"Deleting {selected_class_name}..."):
                            success = delete_custom_strategy(selected_class_name)

                        if success:
                            st.success(f"Successfully deleted {selected_class_name}!")
                            st.toast("Strategy deleted! Reloading...", icon="🗑️")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Failed to delete {selected_class_name}")

            with col2:
                st.caption(f"This will permanently delete `{selected_class_name}` from custom_strategies.py")
    else:
        st.info("No custom strategies found. Create one above to get started!")

    st.markdown("---")
    st.caption("Your custom strategies are saved to custom_strategies.py and automatically appear in the dropdown")


# Page - Live Terminal
elif selected_page == "Live Terminal":
    page_header(
        title="Live Terminal",
        subtitle="Real-time connection to Alpaca Paper Trading",
        badge=("Live", "warning")
    )

    # Connect to Alpaca
    try:
        api = tradeapi.REST(
            key_id=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
            base_url=config.ALPACA_BASE_URL,
            api_version='v2'
        )

        # Get account info
        account = api.get_account()

        st.success("Connected to Alpaca")

        # Account metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            portfolio_value = float(account.portfolio_value)
            st.metric("Portfolio Value", f"${portfolio_value:,.2f}")

        with col2:
            equity = float(account.equity)
            st.metric("Equity", f"${equity:,.2f}")

        with col3:
            buying_power = float(account.buying_power)
            st.metric("Buying Power", f"${buying_power:,.2f}")

        with col4:
            cash = float(account.cash)
            st.metric("Cash", f"${cash:,.2f}")

        # Instant Order Execution
        st.markdown("### Instant Order Execution")

        with st.form("order_entry_ticket"):
            # Row 1 - Symbol, Side, Order Type
            r1c1, r1c2, r1c3 = st.columns(3)
            with r1c1:
                order_symbol = st.text_input("Symbol", value=ticker)
            with r1c2:
                order_side = st.selectbox("Side", ["BUY", "SELL"])
            with r1c3:
                order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"])

            # Row 2: Quantity, Limit Price, Time in Force
            r2c1, r2c2, r2c3 = st.columns(3)
            with r2c1:
                order_qty = st.number_input("Quantity", min_value=1, value=1, step=1)
            with r2c2:
                order_limit_price = st.number_input(
                    "Limit Price",
                    min_value=0.01,
                    value=1.00,
                    step=0.01,
                    format="%.2f",
                    disabled=(order_type != "LIMIT"),
                    help="Only used for LIMIT orders"
                )
            with r2c3:
                order_tif = st.selectbox("Time in Force", ["DAY", "GTC"])

            submitted = st.form_submit_button("SUBMIT ORDER", use_container_width=True)

        if submitted:
            try:
                order_kwargs = dict(
                    symbol=order_symbol.upper().strip(),
                    qty=int(order_qty),
                    side=order_side.lower(),
                    type=order_type.lower(),
                    time_in_force=order_tif.lower(),
                )
                if order_type == "LIMIT":
                    order_kwargs["limit_price"] = float(order_limit_price)

                order = api.submit_order(**order_kwargs)
                st.toast(f"Order Submitted! ID: {order.id}", icon="✅")
                time.sleep(1)
                st.rerun()
            except Exception as exc:
                st.error(f"Order Rejected: {exc}")

        # Display account status
        st.markdown("### Account Status")
        status_col1, status_col2 = st.columns(2)

        with status_col1:
            st.write(f"**Account Number:** {account.account_number}")
            st.write(f"**Status:** {account.status}")
            st.write(f"**Pattern Day Trader:** {account.pattern_day_trader}")

        with status_col2:
            st.write(f"**Day Trade Count:** {account.daytrade_count}")
            st.write(f"**Trading Blocked:** {account.trading_blocked}")
            st.write(f"**Account Blocked:** {account.account_blocked}")

        # Get open positions
        st.markdown("### Open Positions")

        positions = api.list_positions()

        if positions:
            positions_data = []
            for pos in positions:
                positions_data.append({
                    'Symbol': pos.symbol,
                    'Qty': float(pos.qty),
                    'Entry Price': f"${float(pos.avg_entry_price):.2f}",
                    'Current Price': f"${float(pos.current_price):.2f}",
                    'Market Value': f"${float(pos.market_value):.2f}",
                    'P&L': f"${float(pos.unrealized_pl):.2f}",
                    'P&L %': f"{float(pos.unrealized_plpc) * 100:.2f}%",
                    'Side': pos.side
                })

            df_positions = pd.DataFrame(positions_data)

            # Color code P&L
            def color_pnl(val):
                if isinstance(val, str) and '$' in val:
                    num = float(val.replace('$', '').replace(',', ''))
                    color = '#00e676' if num >= 0 else '#ff1744'
                    return f'color: {color}; font-weight: bold'
                elif isinstance(val, str) and '%' in val:
                    num = float(val.replace('%', ''))
                    color = '#00e676' if num >= 0 else '#ff1744'
                    return f'color: {color}; font-weight: bold'
                return ''

            styled_positions = df_positions.style.applymap(color_pnl, subset=['P&L', 'P&L %'])
            st.dataframe(styled_positions, use_container_width=True)

        else:
            st.info("No open positions")

        # Recent orders
        st.markdown("### Recent Orders (Last 10)")
        orders = api.list_orders(status='all', limit=10)

        if orders:
            orders_data = []
            for order in orders:
                orders_data.append({
                    'Time': order.submitted_at,
                    'Symbol': order.symbol,
                    'Side': order.side.upper(),
                    'Qty': order.qty,
                    'Type': order.type,
                    'Status': order.status,
                    'Filled': order.filled_qty if order.filled_qty else 0
                })

            df_orders = pd.DataFrame(orders_data)
            st.dataframe(df_orders, use_container_width=True)
        else:
            st.info("No recent orders")

        # Auto-refresh button
        if st.button("Refresh Data"):
            st.rerun()

        # ── AI Copilot Market Evaluation ────────────────────────────────
        st.markdown("---")
        st.markdown("### 🤖 AI Copilot Market Evaluation")
        st.caption(
            "Run a real-time, single-cycle analysis using the Hybrid Strategy "
            "and local Mixtral LLM."
        )

        if st.button("Evaluate Current Market", type="primary", use_container_width=True):
            with st.spinner("AI is analyzing the charts (this may take a few seconds)..."):
                try:
                    import yfinance as yf
                    from ai_brain import AIBrain

                    # Fetch last 3 months at the currently selected timeframe
                    df_ai = yf.download(
                        ticker, period="3mo", interval=timeframe, progress=False
                    )

                    # Flatten MultiIndex columns (yfinance multi-ticker download quirk)
                    if isinstance(df_ai.columns, pd.MultiIndex):
                        df_ai.columns = df_ai.columns.get_level_values(0)

                    if df_ai.empty:
                        st.error(
                            f"No data returned for **{ticker}** at `{timeframe}` interval. "
                            "Try a longer timeframe."
                        )
                    else:
                        strategy = HybridStrategy(**param_values)
                        df_ai = strategy.calculate_indicators(df_ai)
                        latest_bar = df_ai.iloc[-1]

                        ai_brain = AIBrain()
                        prompt = strategy.generate_prompt(latest_bar, df_ai.tail(10))
                        decision = ai_brain.analyze_market(prompt)

                        # Result metrics
                        col_sig, col_conf, col_price = st.columns(3)
                        with col_sig:
                            st.metric("AI Signal", decision.get("signal", "UNKNOWN"))
                        with col_conf:
                            st.metric(
                                "Confidence", f"{decision.get('confidence', 0):.0%}"
                            )
                        with col_price:
                            st.metric(
                                "Current Price", f"${float(latest_bar['Close']):.2f}"
                            )

                        reasoning = decision.get("reasoning", "")
                        if decision.get("signal") == "BUY":
                            st.success(reasoning)
                        else:
                            st.info(reasoning)

                except Exception as copilot_err:
                    st.error(
                        f"AI Copilot error: {copilot_err}. "
                        f"Ensure Ollama is running and reachable at {config.OLLAMA_BASE_URL}."
                    )

    except Exception as e:
        st.error(f"Failed to connect to Alpaca: {e}")
        st.info("Please check your API credentials in config.py or .env file")


# Page - Portfolio Optimizer
elif selected_page == "Portfolio Opt":
    page_header(
        title="Portfolio Optimizer",
        subtitle="Modern Portfolio Theory — Find the optimal allocation"
    )

    st.info("""
    **How it works:**
    - Select stocks from multiple sectors
    - Algorithm calculates optimal weights using **Maximum Sharpe Ratio**
    - Get exact share counts based on your investment amount
    - View Expected Return, Volatility, and Efficient Frontier
    """)

    st.markdown("---")

    # Step 1 - Universe selection
    st.markdown("### Step 1: Load Asset Universe")

    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        load_sp500_btn = st.button("Load S&P 500 Tickers (500+ stocks)", use_container_width=True)

    with col_btn2:
        load_crypto_btn = st.button("Load Top 100 Crypto", use_container_width=True)

    with col_btn3:
        use_default_btn = st.button("Use Default 30 Stocks", use_container_width=True)

    # Initialize session state for available tickers
    if 'available_tickers' not in st.session_state:
        st.session_state.available_tickers = config.TICKER_UNIVERSE
        st.session_state.ticker_source = "Default 30 Stocks"

    # Load S&P 500 tickers
    if load_sp500_btn:
        with st.spinner("Fetching S&P 500 list from Wikipedia..."):
            try:
                sp500_tickers = PortfolioOptimizer.get_sp500_tickers()
                st.session_state.available_tickers = sp500_tickers
                st.session_state.ticker_source = f"S&P 500 ({len(sp500_tickers)} stocks)"
                st.success(f"Loaded {len(sp500_tickers)} S&P 500 tickers")
            except Exception as e:
                st.error(f"Failed to load S&P 500 tickers: {e}")
                st.info("Using default ticker list as fallback")
                st.session_state.available_tickers = config.TICKER_UNIVERSE
                st.session_state.ticker_source = "Default 30 Stocks (Fallback)"

    # Load crypto tickers (merges with existing)
    if load_crypto_btn:
        with st.spinner("Fetching top 100 cryptocurrencies from Yahoo Finance..."):
            try:
                crypto_tickers = PortfolioOptimizer.get_top_crypto_tickers()
                existing_tickers = st.session_state.available_tickers
                combined_tickers = list(set(existing_tickers + crypto_tickers))  # Remove duplicates
                combined_tickers.sort()  # Sort alphabetically

                st.session_state.available_tickers = combined_tickers
                crypto_count = len(crypto_tickers)
                total_count = len(combined_tickers)
                st.session_state.ticker_source = f"Mixed Portfolio (Stocks + Crypto, {total_count} total)"
                st.success(f"Added {crypto_count} crypto tickers! Total universe: {total_count} assets")
            except Exception as e:
                st.error(f"Failed to load crypto tickers: {e}")
                st.info("Crypto tickers not added")

    # Use default tickers
    if use_default_btn:
        st.session_state.available_tickers = config.TICKER_UNIVERSE
        st.session_state.ticker_source = "Default 30 Stocks"
        st.success("Using default curated stock list!")

    # Display current universe
    st.caption(f"**Current Universe**: {st.session_state.ticker_source}")

    st.markdown("---")

    # Step 2: stock selection and investment
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Step 2: Select Stocks to Optimize")

        # Dynamic multiselect based on loaded universe
        default_selection = config.DEFAULT_PORTFOLIO_TICKERS if st.session_state.ticker_source.startswith("Default") else []

        selected_tickers = st.multiselect(
            f"Choose 5-30 stocks from {len(st.session_state.available_tickers)} available tickers",
            options=st.session_state.available_tickers,
            default=default_selection[:10] if default_selection else [],
            help="Select at least 5 stocks for meaningful diversification. More stocks = better diversification.",
            max_selections=30
        )

        st.caption(f"Selected: {len(selected_tickers)} stocks")

    with col2:
        st.markdown("### Step 3: Investment Amount")

        capital = st.number_input(
            "Total Capital ($)",
            min_value=1000.0,
            max_value=10000000.0,
            value=10000.0,
            step=1000.0,
            help="Total amount to invest across all positions"
        )

        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="Used for Sharpe Ratio calculation"
        ) / 100

        lookback_period = st.selectbox(
            "Lookback Period",
            options=["1y", "2y", "5y", "10y", "max"],
            index=2,  # Default to "5y"
            help="Historical period for correlation & return analysis. Longer = more robust but excludes young assets."
        )

    st.markdown("---")

    # Optimize button
    st.markdown("### Step 4: Optimize Portfolio")
    optimize_btn = st.button("Find Optimal Allocation", type="primary", use_container_width=True)

    # Optimization Execution
    if optimize_btn:
        # Validation
        if len(selected_tickers) < 5:
            st.error("Please select at least 5 stocks for meaningful portfolio optimization!")
        else:
            with st.spinner(f"Optimizing portfolio with {len(selected_tickers)} stocks..."):
                try:
                    # Create optimizer
                    optimizer = PortfolioOptimizer(selected_tickers, capital)

                    # Fetch data
                    st.info(f"Downloading {lookback_period} of price data for {len(selected_tickers)} tickers...")
                    prices = optimizer.fetch_data(period=lookback_period)

                    if len(optimizer.tickers) < 2:
                        st.error("Not enough valid tickers with data. Please select different stocks.")
                        st.stop()

                    # Optimize
                    st.info("Calculating optimal weights using Modern Portfolio Theory...")
                    weights = optimizer.optimize_portfolio(risk_free_rate=risk_free_rate)

                    # Get allocation
                    st.info("Converting to discrete share allocations...")
                    allocation_df = optimizer.get_allocation()

                    # Display results
                    st.success("Optimization Complete!")

                    st.markdown("---")

                    # Metrics row
                    st.markdown("### Portfolio Performance Metrics")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Expected Return",
                            f"{optimizer.expected_return * 100:.2f}%",
                            help="Annualized expected return based on historical data"
                        )

                    with col2:
                        st.metric(
                            "Annual Volatility",
                            f"{optimizer.annual_volatility * 100:.2f}%",
                            help="Standard deviation of returns (risk measure)"
                        )

                    with col3:
                        st.metric(
                            "Sharpe Ratio",
                            f"{optimizer.sharpe_ratio:.2f}",
                            help="Risk-adjusted return (higher is better)"
                        )

                    with col4:
                        st.metric(
                            "Leftover Cash",
                            f"${optimizer.leftover:,.2f}",
                            help="Cash remaining after discrete allocation"
                        )

                    st.markdown("---")

                    # --- ALLOCATION TABLE ---
                    st.markdown("### Optimal Allocation")
                    st.markdown("**Buy these exact shares to create your optimized portfolio:**")

                    # Format the dataframe
                    allocation_display = allocation_df.copy()
                    allocation_display['Price'] = allocation_display['Price'].apply(lambda x: f"${x:.2f}")
                    allocation_display['Value'] = allocation_display['Value'].apply(lambda x: f"${x:,.2f}")
                    allocation_display['Weight (%)'] = allocation_display['Weight (%)'].apply(lambda x: f"{x:.2f}%")

                    st.dataframe(
                        allocation_display,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Summary stats
                    total_invested = capital - optimizer.leftover
                    st.caption(f"Total Invested: ${total_invested:,.2f} | Leftover: ${optimizer.leftover:,.2f}")

                    st.markdown("---")

                    # Efficient frontier plot
                    st.markdown("### Efficient Frontier")
                    st.markdown("The optimal portfolio maximizes return per unit of risk.")

                    with st.spinner("Generating Efficient Frontier visualization..."):
                        fig = optimizer.plot_efficient_frontier(num_portfolios=5000)
                        st.pyplot(fig)
                        plt.close(fig)

                    st.markdown("---")

                    # positon weights
                    st.markdown("### Portfolio Weights")

                    # Create pie chart
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=allocation_df['Ticker'],
                        values=allocation_df['Weight (%)'],
                        hole=0.4,
                        marker=dict(colors=plt.cm.viridis(np.linspace(0, 1, len(allocation_df))))
                    )])

                    fig_pie.update_layout(
                        template="plotly_dark",
                        height=500,
                        paper_bgcolor="#121212",
                        plot_bgcolor="#1E1E1E",
                        showlegend=True,
                        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1)
                    )

                    st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

                    st.markdown("---")

                    st.success("**Your optimized portfolio is ready!** Copy the allocation table above to execute your trades.")

                except Exception as e:
                    st.error(f"Optimization failed: {e}")
                    st.exception(e)

    else:
        # Show placeholder when not optimizing
        st.info("👆 Select your stocks and investment amount, then click **Find Optimal Allocation** to begin.")

        # Show available universe
        with st.expander("📋 View Full Stock Universe"):
            st.markdown("**Available stocks organized by sector:**")

            sectors = {
                "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMD", "META", "TSLA"],
                "Finance": ["JPM", "BAC", "V", "MA", "GS"],
                "Healthcare": ["JNJ", "PFE", "UNH", "LLY", "ABBV"],
                "Consumer": ["AMZN", "HD", "PG", "KO", "WMT", "MCD"],
                "Energy & Industrials": ["XOM", "CVX", "CAT", "BA", "UPS"],
                "ETFs": ["SPY", "QQQ"]
            }

            for sector, tickers in sectors.items():
                st.markdown(f"**{sector}**")
                st.write(", ".join(tickers))


# PAGE - options lab
elif selected_page == "Options Lab":
    page_header(
        title="Options Lab",
        subtitle="Find high-leverage plays with advanced Greeks analysis"
    )

    st.info("""
    **How it works:**
    - Scan for "Lotto Tickets": OTM options with explosive potential
    - Analyze Greeks: Gamma (acceleration), Delta (directionality), Vega (volatility sensitivity)
    - Profit Calculator: Simulate option value at different stock prices
    - Risk Warning: Options are leveraged instruments - you can lose 100% of your investment
    """)

    st.markdown("---")

    # expiration selection
    st.markdown("### Step 1: Select Asset & Expiration")

    col1, col2 = st.columns(2)

    with col1:
        options_ticker = st.text_input(
            "Ticker Symbol",
            value="SPY",
            help="Enter stock ticker (e.g., SPY, TSLA, AAPL)"
        ).upper()

    with col2:
        # Initialize session state for options manager
        if 'options_manager' not in st.session_state:
            st.session_state.options_manager = None
            st.session_state.options_expirations = []

    # Fetch expirations button
    if st.button("Load Expirations", use_container_width=True):
        with st.spinner(f"Fetching options data for {options_ticker}..."):
            try:
                om = OptionsManager(options_ticker)
                om.fetch_stock_data()
                expirations = om.get_expirations()

                st.session_state.options_manager = om
                st.session_state.options_expirations = expirations

                st.success(f"Loaded {len(expirations)} expiration dates for {options_ticker}")
                st.metric("Current Stock Price", f"${om.current_price:.2f}")

            except Exception as e:
                st.error(f"Failed to load options data: {e}")

    # Expiration selector if expirations loaded
    if st.session_state.options_expirations:
        expiration_date = st.selectbox(
            "Select Expiration Date",
            options=st.session_state.options_expirations,
            help="Choose when the option expires"
        )

        # Display days to expiration
        exp_datetime = datetime.strptime(expiration_date, '%Y-%m-%d')
        days_to_exp = (exp_datetime - datetime.now()).days
        st.caption(f"📅 {days_to_exp} days to expiration")

        st.markdown("---")

        # Scanner finds speculative plays
        st.markdown("### Step 2: Scan for Speculative Plays")

        col1, col2, col3 = st.columns(3)

        with col1:
            max_price = st.number_input(
                "Max Option Price ($)",
                min_value=0.10,
                max_value=10.0,
                value=1.00,
                step=0.10,
                help="Maximum price per option (filter for cheap plays)"
            )

        with col2:
            min_gamma_rank = st.slider(
                "Min Gamma Rank (%)",
                min_value=0,
                max_value=100,
                value=70,
                help="Minimum gamma percentile (higher = more explosive)"
            )

        with col3:
            dte_min, dte_max = st.slider(
                "Days to Expiration Range",
                min_value=1,
                max_value=90,
                value=(7, 45),
                help="Filter by expiration timeframe"
            )

        # Scan button
        scan_btn = st.button("Find Speculative Plays", type="primary", use_container_width=True)

        if scan_btn:
            with st.spinner("Scanning options chain..."):
                try:
                    om = st.session_state.options_manager

                    # Fetch chain
                    calls, puts = om.fetch_chain(expiration_date)

                    # Calculate metrics
                    calls_with_metrics = om.calculate_spec_metrics(calls, 'call')
                    puts_with_metrics = om.calculate_spec_metrics(puts, 'put')

                    # Find lotto tickets
                    lottos = om.find_lotto_tickets(
                        max_price=max_price,
                        min_gamma_rank=min_gamma_rank,
                        dte_range=(dte_min, dte_max)
                    )

                    # Store in session state
                    st.session_state.lotto_calls = lottos['calls']
                    st.session_state.lotto_puts = lottos['puts']
                    st.session_state.all_calls = calls_with_metrics
                    st.session_state.all_puts = puts_with_metrics

                    st.success(f"Found {len(lottos['calls'])} call plays and {len(lottos['puts'])} put plays!")

                except Exception as e:
                    st.error(f"Scan failed: {e}")
                    st.exception(e)

        # Step 3 - display results
        if 'lotto_calls' in st.session_state and 'lotto_puts' in st.session_state:
            st.markdown("---")
            st.markdown("### Speculative Plays Found")

            # Tabs for Calls and Puts
            call_tab, put_tab = st.tabs(["CALL Lottos", "PUT Lottos"])

            with call_tab:
                lotto_calls = st.session_state.lotto_calls

                if not lotto_calls.empty:
                    st.markdown(f"**{len(lotto_calls)} High-Leverage Call Plays**")

                    # Display columns
                    display_cols = [
                        'expiration', 'strike', 'lastPrice', 'volume', 'openInterest',
                        'impliedVolatility', 'leverage', 'breakEvenMove'
                    ]

                    # Filter to existing columns
                    available_cols = [col for col in display_cols if col in lotto_calls.columns]

                    calls_display = lotto_calls[available_cols].head(20).copy()

                    # Format columns
                    calls_display['lastPrice'] = calls_display['lastPrice'].apply(lambda x: f"${x:.2f}")
                    calls_display['impliedVolatility'] = calls_display['impliedVolatility'].apply(lambda x: f"{x*100:.1f}%")

                    if 'leverage' in calls_display.columns:
                        calls_display['leverage'] = calls_display['leverage'].apply(lambda x: f"{x:.1f}x")

                    if 'breakEvenMove' in calls_display.columns:
                        calls_display['breakEvenMove'] = calls_display['breakEvenMove'].apply(lambda x: f"{x:.1f}%")

                    st.dataframe(calls_display, use_container_width=True, hide_index=True)

                    st.caption("💡 **Leverage**: How much the option moves vs stock | **Break-even**: % move needed to profit")

                else:
                    st.info("No call plays match your criteria. Try widening the filters.")

            with put_tab:
                lotto_puts = st.session_state.lotto_puts

                if not lotto_puts.empty:
                    st.markdown(f"**{len(lotto_puts)} High-Leverage Put Plays**")

                    # Display columns
                    display_cols = [
                        'expiration', 'strike', 'lastPrice', 'volume', 'openInterest',
                        'impliedVolatility', 'leverage', 'breakEvenMove'
                    ]

                    available_cols = [col for col in display_cols if col in lotto_puts.columns]
                    puts_display = lotto_puts[available_cols].head(20).copy()

                    # Format columns
                    puts_display['lastPrice'] = puts_display['lastPrice'].apply(lambda x: f"${x:.2f}")
                    puts_display['impliedVolatility'] = puts_display['impliedVolatility'].apply(lambda x: f"{x*100:.1f}%")

                    if 'leverage' in puts_display.columns:
                        puts_display['leverage'] = puts_display['leverage'].apply(lambda x: f"{x:.1f}x")

                    if 'breakEvenMove' in puts_display.columns:
                        puts_display['breakEvenMove'] = puts_display['breakEvenMove'].apply(lambda x: f"{x:.1f}%")

                    st.dataframe(puts_display, use_container_width=True, hide_index=True)

                    st.caption("💡 **Leverage**: How much the option moves vs stock | **Break-even**: % move needed to profit")

                else:
                    st.info("No put plays match your criteria. Try widening the filters.")

            st.markdown("---")

            # ================================================================
            # STEP 4: PROFIT CALCULATOR
            # ================================================================
            st.markdown("### Profit Calculator")
            st.markdown("**Simulate your profit at different stock prices**")

            calc_col1, calc_col2 = st.columns(2)

            with calc_col1:
                calc_type = st.selectbox("Option Type", ["Call", "Put"])

                if calc_type == "Call" and not lotto_calls.empty:
                    available_strikes = sorted(lotto_calls['strike'].unique())
                elif calc_type == "Put" and not lotto_puts.empty:
                    available_strikes = sorted(lotto_puts['strike'].unique())
                else:
                    available_strikes = []

                if available_strikes:
                    selected_strike = st.selectbox("Strike Price", available_strikes)

                    # Get option price for this strike
                    if calc_type == "Call":
                        option_row = lotto_calls[lotto_calls['strike'] == selected_strike].iloc[0]
                    else:
                        option_row = lotto_puts[lotto_puts['strike'] == selected_strike].iloc[0]

                    option_price = option_row['lastPrice']
                    st.metric("Option Price", f"${option_price:.2f}")

            with calc_col2:
                contracts = st.number_input(
                    "Number of Contracts",
                    min_value=1,
                    max_value=1000,
                    value=10,
                    help="Each contract = 100 shares"
                )

                cost_basis = option_price * 100 * contracts
                st.metric("Total Cost", f"${cost_basis:,.2f}")

            # Profit slider
            om = st.session_state.options_manager
            current_price = om.current_price

            price_range_pct = 50  # ±50% from current price
            min_price = current_price * (1 - price_range_pct / 100)
            max_price = current_price * (1 + price_range_pct / 100)

            new_stock_price = st.slider(
                f"If {options_ticker} moves to...",
                min_value=float(min_price),
                max_value=float(max_price),
                value=float(current_price),
                step=0.50,
                format="$%.2f"
            )

            # Calculate profit
            if available_strikes:
                profit_calc = om.calculate_profit(
                    strike=selected_strike,
                    option_price=option_price,
                    option_type=calc_type.lower(),
                    new_stock_price=new_stock_price,
                    contracts=contracts
                )

                # Display results
                st.markdown("#### 💰 Profit/Loss at New Price")

                metric_col1, metric_col2, metric_col3 = st.columns(3)

                with metric_col1:
                    profit_color = "normal" if profit_calc['total_profit'] >= 0 else "inverse"
                    st.metric(
                        "Total Profit/Loss",
                        f"${profit_calc['total_profit']:,.2f}",
                        delta=f"{profit_calc['roi']:.1f}% ROI"
                    )

                with metric_col2:
                    st.metric(
                        "Intrinsic Value",
                        f"${profit_calc['intrinsic_value']:.2f}",
                        help="Option value at expiration (no time value)"
                    )

                with metric_col3:
                    move_pct = ((new_stock_price - current_price) / current_price) * 100
                    st.metric(
                        "Stock Move",
                        f"{move_pct:+.1f}%",
                        help="Percentage change from current price"
                    )

                # Visual profit chart
                st.markdown("####Profit Curve")

                # Generate profit curve data
                price_points = np.linspace(min_price, max_price, 50)
                profits = []

                for price in price_points:
                    p = om.calculate_profit(
                        strike=selected_strike,
                        option_price=option_price,
                        option_type=calc_type.lower(),
                        new_stock_price=price,
                        contracts=contracts
                    )
                    profits.append(p['total_profit'])

                # Create plotly chart
                fig_profit = go.Figure()

                fig_profit.add_trace(go.Scatter(
                    x=price_points,
                    y=profits,
                    mode='lines',
                    name='Profit/Loss',
                    line=dict(color='#00e676', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(0, 230, 118, 0.1)'
                ))

                # Add zero line
                fig_profit.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)

                # Add current price line
                fig_profit.add_vline(x=current_price, line_dash="dot", line_color="yellow", opacity=0.7)

                # Add selected price marker
                fig_profit.add_trace(go.Scatter(
                    x=[new_stock_price],
                    y=[profit_calc['total_profit']],
                    mode='markers',
                    name='Selected Price',
                    marker=dict(size=15, color='red', symbol='diamond')
                ))

                fig_profit.update_layout(
                    template="plotly_dark",
                    height=400,
                    xaxis_title=f"{options_ticker} Stock Price",
                    yaxis_title="Profit/Loss ($)",
                    paper_bgcolor="#121212",
                    plot_bgcolor="#1E1E1E",
                    showlegend=True
                )

                st.plotly_chart(fig_profit, use_container_width=True, config={'displayModeBar': False})

                st.caption(f"💡 Yellow line = Current price (${current_price:.2f}) | Red diamond = Your selected price")

    else:
        st.info("👆 Click **Load Expirations** to get started!")

    st.markdown("---")
    st.warning("**Risk Warning**: Options trading is highly speculative. You can lose 100% of your investment. This tool is for educational purposes only.")


elif selected_page == "Global Macro":
    if MACRO_AVAILABLE:
        render_macro_page()
    else:
        page_header(
            title="Global Macro Dashboard",
            subtitle="Economic regime analysis — Treasury yields, credit spreads, yield curve",
            badge=("Error", "error")
        )
        st.error("Macro Intelligence module not available. Run: `pip install pandas-datareader`")


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("AI Trading Terminal | Dynamic Parameter Optimization | Built with Streamlit + Plotly + Alpaca + Ollama")
