"""
Macro Intelligence Module
Fetches economic data from FRED, classifies macro regimes,
and provides a Global Macro Dashboard visualization tab.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# FRED series IDs
FRED_SERIES = {
    'DGS10': '10-Year Treasury Yield',
    'BAMLH0A0HYM2': 'High Yield Credit Spread',
    'T10Y2Y': 'Yield Curve (10Y-2Y)',
}

SAFE_HAVENS = {'GLD', 'TLT'}


@st.cache_data(ttl=3600, show_spinner=False)
def get_macro_data(start=None, end=None):
    """Fetch macro economic data from FRED.

    Args:
        start: Start date (default: 20 years ago)
        end: End date (default: today)

    Returns:
        DataFrame with columns: DGS10, BAMLH0A0HYM2, T10Y2Y
    """
    import pandas_datareader.data as pdr

    if start is None:
        start = datetime.now() - timedelta(days=365 * 20)
    if end is None:
        end = datetime.now()

    # Strip to tz-naive date objects to avoid tz-naive vs tz-aware comparison errors
    if hasattr(start, 'date'):
        start = start.date()
    if hasattr(end, 'date'):
        end = end.date()

    series_ids = list(FRED_SERIES.keys())
    frames = {}

    for sid in series_ids:
        try:
            df = pdr.DataReader(sid, 'fred', start, end)
            frames[sid] = df[sid]
        except Exception as e:
            logger.warning(f"Failed to fetch {sid} from FRED: {e}")

    if not frames:
        raise ConnectionError("Could not fetch any FRED data. Check internet connection.")

    macro_df = pd.DataFrame(frames)
    macro_df = macro_df.ffill()

    # Localize to UTC so reindex works with yfinance's tz-aware (UTC) timestamps
    if macro_df.index.tz is None:
        macro_df.index = macro_df.index.tz_localize('UTC')

    return macro_df


def classify_macro_regime(macro_df, upper_threshold=0.10, lower_threshold=-0.10, persistence=5):
    """Classify market regime using a hysteresis state machine.

    Applies a 10-day EMA to the credit spread deviation from its 50-day SMA to
    kill daily noise, then uses a two-threshold band with a persistence requirement
    to prevent regime churn from short-lived market blips.

    State only flips when the smoothed indicator stays outside a threshold for
    `persistence` consecutive bars — eliminating false signals during brief
    volatility spikes within an otherwise healthy bull market.

    Args:
        macro_df: DataFrame from get_macro_data()
        upper_threshold: Smoothed deviation above this triggers Risk-Off countdown (default: +0.10)
        lower_threshold: Smoothed deviation below this triggers Risk-On countdown (default: -0.10)
        persistence: Consecutive bars required beyond threshold before state flips (default: 5)

    Returns:
        Series of bool (True = Risk Off) indexed like macro_df
    """
    risk_off = pd.Series(False, index=macro_df.index)

    if 'BAMLH0A0HYM2' not in macro_df.columns:
        return risk_off

    spread = macro_df['BAMLH0A0HYM2']

    # Step 1: Deviation from 50-day SMA (trend-following: rising spreads = stress)
    sma50 = spread.rolling(50, min_periods=15).mean()
    deviation = spread - sma50

    # Step 2: 10-day EMA smoothing to suppress daily noise
    smoothed = deviation.ewm(span=10, adjust=False).mean()

    # Step 3: Hysteresis state machine
    vals = smoothed.values
    states = np.zeros(len(vals), dtype=bool)
    current_state = False   # False = Risk On, True = Risk Off
    pending_flip = None     # Target state we're counting toward
    consec_count = 0

    for i, v in enumerate(vals):
        if np.isnan(v):
            states[i] = current_state
            continue

        # Determine what direction the signal is pushing
        if v > upper_threshold:
            signal_target = True    # Risk Off signal
        elif v < lower_threshold:
            signal_target = False   # Risk On signal
        else:
            signal_target = None    # Inside dead band — no pressure

        # Persistence logic: only count consecutive bars pushing in the same direction
        if signal_target is not None and signal_target != current_state:
            if signal_target == pending_flip:
                consec_count += 1
            else:
                pending_flip = signal_target
                consec_count = 1

            if consec_count >= persistence:
                current_state = signal_target
                pending_flip = None
                consec_count = 0
        else:
            # Signal agrees with current state or in dead band — reset countdown
            pending_flip = None
            consec_count = 0

        states[i] = current_state

    return pd.Series(states, index=macro_df.index)


def align_macro_to_prices(macro_series, price_index):
    """Align daily FRED data to any price index (hourly, daily, weekly).

    Args:
        macro_series: Series with DatetimeIndex (daily FRED data)
        price_index: DatetimeIndex from price data

    Returns:
        Series aligned to price_index with forward-fill
    """
    # Normalize both to date-only for matching if price is daily
    aligned = macro_series.reindex(price_index, method='ffill')
    # If reindex didn't match (e.g., hourly vs daily), try date-based join
    if aligned.isna().all():
        macro_daily = macro_series.copy()
        macro_daily.index = macro_daily.index.normalize()
        price_dates = pd.Series(price_index).dt.normalize()
        aligned = macro_daily.reindex(price_dates.values, method='ffill')
        aligned.index = price_index
    return aligned.fillna(method='ffill').fillna(False)


def _dark_layout(fig, title="", height=450):
    """Apply consistent dark theme to a figure."""
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#121212',
        plot_bgcolor='#1E1E1E',
        font=dict(color='#E6EDF3'),
        title=dict(text=title, font=dict(size=16, color='#E6EDF3')),
        height=height,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')


def render_macro_page():
    """Render the Global Macro Dashboard tab."""
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h2 style="color: #E6EDF3; margin: 0;">Global Macro Dashboard</h2>
        <p style="color: #6E7681; font-size: 0.875rem;">
            Economic regime analysis — Treasury yields, credit spreads, yield curve
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Period selector
    col_period, col_spacer = st.columns([1, 3])
    with col_period:
        lookback = st.selectbox(
            "Lookback Period",
            ["5 Years", "10 Years", "20 Years", "Max (2000+)"],
            index=1,
            key="macro_lookback"
        )

    years_map = {"5 Years": 5, "10 Years": 10, "20 Years": 20, "Max (2000+)": 25}
    years = years_map[lookback]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    # Fetch data
    with st.spinner("Fetching FRED macro data..."):
        try:
            macro_df = get_macro_data(start_date, end_date)
        except Exception as e:
            st.error(f"Failed to fetch macro data: {e}")
            st.info("Ensure `pandas-datareader` is installed and you have internet access.")
            return

    regime = classify_macro_regime(macro_df)

    # Compute regime transition statistics
    _prev, _count = None, 0
    _ro_runs, _rf_runs = [], []
    for v in regime:
        if _prev is None:
            _prev, _count = v, 1
        elif v == _prev:
            _count += 1
        else:
            (_rf_runs if _prev else _ro_runs).append(_count)
            _prev, _count = v, 1
    if _prev is not None:
        (_rf_runs if _prev else _ro_runs).append(_count)

    avg_risk_on_days  = float(np.mean(_ro_runs))  if _ro_runs  else 0.0
    avg_risk_off_days = float(np.mean(_rf_runs))  if _rf_runs  else 0.0
    n_flips           = max(len(_ro_runs) + len(_rf_runs) - 1, 0)
    flips_per_month   = n_flips / (len(regime) / 21) if len(regime) > 0 else 0.0

    # Current Regime Banner
    latest_regime = regime.iloc[-1] if len(regime) > 0 else False
    risk_off_pct = regime.rolling(252).mean().iloc[-1] * 100 if len(regime) > 252 else regime.mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if latest_regime:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #b71c1c, #e53935); padding: 1rem; border-radius: 8px; text-align: center;">
                <p style="color: white; font-size: 1.5rem; font-weight: 700; margin: 0;">RISK OFF</p>
                <p style="color: rgba(255,255,255,0.7); font-size: 0.8rem; margin: 0;">Macro Stress Detected</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1b5e20, #43a047); padding: 1rem; border-radius: 8px; text-align: center;">
                <p style="color: white; font-size: 1.5rem; font-weight: 700; margin: 0;">RISK ON</p>
                <p style="color: rgba(255,255,255,0.7); font-size: 0.8rem; margin: 0;">Macro Conditions Favorable</p>
            </div>
            """, unsafe_allow_html=True)
    with col2:
        st.metric("Risk Off (1Y %)", f"{risk_off_pct:.0f}%",
                  help="Percentage of the last year spent in Risk Off regime")
    with col3:
        if 'DGS10' in macro_df.columns:
            latest_10y = macro_df['DGS10'].dropna().iloc[-1]
            st.metric("10Y Treasury", f"{latest_10y:.2f}%")
    with col4:
        if 'T10Y2Y' in macro_df.columns:
            latest_curve = macro_df['T10Y2Y'].dropna().iloc[-1]
            st.metric("Yield Curve", f"{latest_curve:.2f}%",
                      delta="Inverted" if latest_curve < 0 else "Normal",
                      delta_color="inverse")

    # Regime Transition Statistics
    st.markdown("#### Regime Transition Statistics")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric(
            "Avg Risk-On Holding", f"{avg_risk_on_days:.0f} days",
            help="Average consecutive trading days spent in Risk-On state per run",
        )
    with col_b:
        st.metric(
            "Avg Risk-Off Holding", f"{avg_risk_off_days:.0f} days",
            help="Average consecutive trading days spent in Risk-Off state per run",
        )
    with col_c:
        st.metric(
            "Total Regime Flips", f"{n_flips}",
            help="Number of state transitions over the selected lookback period",
        )
    with col_d:
        st.metric(
            "Flips / Month", f"{flips_per_month:.2f}",
            help="Low value (< 0.5) confirms hysteresis is suppressing regime churn",
        )

    # Yield Curve Chart
    if 'T10Y2Y' in macro_df.columns:
        st.markdown("### Yield Curve (10Y - 2Y Spread)")
        curve = macro_df['T10Y2Y'].dropna()

        fig_curve = go.Figure()

        # Split into positive (green) and negative (red) regions
        pos = curve.where(curve >= 0)
        neg = curve.where(curve < 0)

        fig_curve.add_trace(go.Scatter(
            x=pos.index, y=pos.values,
            fill='tozeroy', fillcolor='rgba(67, 160, 71, 0.3)',
            line=dict(color='#43a047', width=1.5),
            name='Normal (Positive)'
        ))
        fig_curve.add_trace(go.Scatter(
            x=neg.index, y=neg.values,
            fill='tozeroy', fillcolor='rgba(229, 57, 53, 0.3)',
            line=dict(color='#e53935', width=1.5),
            name='Inverted (Negative)'
        ))

        # Zero line
        fig_curve.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)

        _dark_layout(fig_curve, "10Y-2Y Treasury Spread — Recession Signal", height=400)
        fig_curve.update_yaxes(title_text="Spread (%)")
        st.plotly_chart(fig_curve, use_container_width=True, config={'displayModeBar': False})
        st.caption("Red zones = yield curve inverted (10Y yield < 2Y yield). Historically precedes recessions by 12-18 months.")

    # Credit Spreads vs SPY
    if 'BAMLH0A0HYM2' in macro_df.columns:
        st.markdown("### Credit Spreads vs SPY")

        # Fetch SPY for comparison
        try:
            import yfinance as yf
            spy = yf.download('SPY', start=start_date, end=end_date, progress=False)['Close']
            if isinstance(spy, pd.DataFrame):
                spy = spy.iloc[:, 0]
        except Exception:
            spy = None

        spread = macro_df['BAMLH0A0HYM2'].dropna()
        spread_avg = spread.rolling(63, min_periods=21).mean()

        fig_spread = make_subplots(specs=[[{"secondary_y": True}]])

        # Credit spread (left Y)
        fig_spread.add_trace(go.Scatter(
            x=spread.index, y=spread.values,
            line=dict(color='#ff6d00', width=2),
            name='HY Credit Spread'
        ), secondary_y=False)

        # Rolling average
        fig_spread.add_trace(go.Scatter(
            x=spread_avg.index, y=spread_avg.values,
            line=dict(color='#ff6d00', width=1, dash='dash'),
            name='63d Average',
            opacity=0.6
        ), secondary_y=False)

        # SPY (right Y)
        if spy is not None and len(spy) > 0:
            fig_spread.add_trace(go.Scatter(
                x=spy.index, y=spy.values,
                line=dict(color='#2962ff', width=1.5),
                name='SPY'
            ), secondary_y=True)

        _dark_layout(fig_spread, "High Yield Credit Spread vs S&P 500", height=450)
        fig_spread.update_yaxes(title_text="Credit Spread (%)", secondary_y=False)
        fig_spread.update_yaxes(title_text="SPY Price ($)", secondary_y=True)
        st.plotly_chart(fig_spread, use_container_width=True, config={'displayModeBar': False})
        st.caption("Spikes in credit spreads (orange) often precede or coincide with equity sell-offs (blue). "
                   "Spread above its 63-day average = elevated stress.")

    # 10Y Treasury Yield
    if 'DGS10' in macro_df.columns:
        st.markdown("### 10-Year Treasury Yield")
        yield_10y = macro_df['DGS10'].dropna()

        fig_yield = go.Figure()
        fig_yield.add_trace(go.Scatter(
            x=yield_10y.index, y=yield_10y.values,
            line=dict(color='#00e676', width=2),
            fill='tozeroy', fillcolor='rgba(0, 230, 118, 0.1)',
            name='10Y Yield'
        ))

        _dark_layout(fig_yield, "U.S. 10-Year Treasury Yield", height=350)
        fig_yield.update_yaxes(title_text="Yield (%)")
        st.plotly_chart(fig_yield, use_container_width=True, config={'displayModeBar': False})

    # Regime Timeline
    st.markdown("### Macro Regime Timeline")

    fig_regime = go.Figure()

    # Create colored segments for Risk On / Risk Off
    risk_on_dates = regime.index[~regime]
    risk_off_dates = regime.index[regime]

    if len(risk_on_dates) > 0:
        fig_regime.add_trace(go.Scatter(
            x=risk_on_dates, y=[1] * len(risk_on_dates),
            mode='markers', marker=dict(color='#43a047', size=3, symbol='square'),
            name='Risk On'
        ))
    if len(risk_off_dates) > 0:
        fig_regime.add_trace(go.Scatter(
            x=risk_off_dates, y=[1] * len(risk_off_dates),
            mode='markers', marker=dict(color='#e53935', size=3, symbol='square'),
            name='Risk Off'
        ))

    # Rolling risk-off percentage
    risk_pct = regime.astype(float).rolling(63, min_periods=21).mean() * 100
    fig_regime.add_trace(go.Scatter(
        x=risk_pct.index, y=risk_pct.values,
        line=dict(color='#ffea00', width=1.5),
        name='Risk Off % (63d rolling)',
        yaxis='y2'
    ))

    _dark_layout(fig_regime, "Risk On / Risk Off Regime Classification", height=300)
    fig_regime.update_layout(
        yaxis=dict(showticklabels=False, range=[0.5, 1.5]),
        yaxis2=dict(title="Risk Off %", overlaying='y', side='right',
                    range=[0, 100], showgrid=False),
    )
    st.plotly_chart(fig_regime, use_container_width=True, config={'displayModeBar': False})
    st.caption("Green = Risk On (favorable conditions). Red = Risk Off (credit stress or yield curve inverted). "
               "Yellow line = rolling 63-day Risk Off percentage.")
