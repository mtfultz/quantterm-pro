"""
QuantTerm Pro - Professional UI Components
Reusable Streamlit components with modern fintech styling.
"""

import streamlit as st
from typing import Optional, Literal


# Color Palette

COLORS = {
    # Backgrounds
    "bg_primary": "#0E1117",
    "bg_secondary": "#161B22",
    "bg_card": "#1C2128",
    "bg_card_hover": "#21262D",
    "bg_elevated": "#262C36",

    # Borders
    "border_default": "#30363D",
    "border_subtle": "#21262D",
    "border_focus": "#388BFD",

    # Text
    "text_primary": "#E6EDF3",
    "text_secondary": "#8B949E",
    "text_tertiary": "#6E7681",
    "text_muted": "#484F58",

    # Accent
    "accent_primary": "#388BFD",
    "accent_hover": "#58A6FF",

    # Semantic
    "success": "#3FB950",
    "success_subtle": "rgba(63, 185, 80, 0.15)",
    "danger": "#F85149",
    "danger_subtle": "rgba(248, 81, 73, 0.15)",
    "warning": "#D29922",
    "warning_subtle": "rgba(210, 153, 34, 0.15)",
    "info": "#388BFD",
    "info_subtle": "rgba(56, 139, 253, 0.15)",
}


# Metric Card

def metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_type: Literal["positive", "negative", "neutral"] = "neutral",
    icon: Optional[str] = None,
) -> None:
    """
    Render a professional KPI metric card.

    Args:
        label: The metric label (e.g., "Total Revenue")
        value: The metric value (e.g., "$1,234.56")
        delta: Optional delta/change text (e.g., "+12.5%")
        delta_type: "positive", "negative", or "neutral"
        icon: Optional icon emoji
    """
    delta_class = delta_type if delta_type in ["positive", "negative"] else ""
    delta_color = {
        "positive": COLORS["success"],
        "negative": COLORS["danger"],
        "neutral": COLORS["text_secondary"],
    }.get(delta_type, COLORS["text_secondary"])

    icon_html = f'<span style="margin-right: 6px;">{icon}</span>' if icon else ""
    delta_html = f'<div class="metric-delta {delta_class}" style="color: {delta_color};">{delta}</div>' if delta else ""

    html = f"""
    <div class="metric-card">
        <div class="metric-label">{icon_html}{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def metric_row(metrics: list[dict]) -> None:
    """
    Render a row of metric cards.

    Args:
        metrics: List of dicts with keys: label, value, delta (optional), delta_type (optional), icon (optional)
    """
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            metric_card(
                label=metric.get("label", ""),
                value=metric.get("value", ""),
                delta=metric.get("delta"),
                delta_type=metric.get("delta_type", "neutral"),
                icon=metric.get("icon"),
            )



# Status Badge

def status_badge(
    text: str,
    status: Literal["online", "offline", "warning"] = "online",
) -> str:
    """
    Return HTML for a status badge.

    Args:
        text: Badge text
        status: "online", "offline", or "warning"

    Returns:
        HTML string for the badge
    """
    return f"""
    <span class="status-badge {status}">
        <span class="status-dot"></span>
        {text}
    </span>
    """


def render_status_badge(
    text: str,
    status: Literal["online", "offline", "warning"] = "online",
) -> None:
    """Render a status badge directly."""
    st.markdown(status_badge(text, status), unsafe_allow_html=True)



# Section Card

def section_card_start(title: str, icon: Optional[str] = None) -> None:
    """
    Start a section card container.

    Args:
        title: Section title
        icon: Optional icon emoji
    """
    icon_html = f'{icon} ' if icon else ""
    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">{icon_html}{title}</div>
    """, unsafe_allow_html=True)


def section_card_end() -> None:
    """End a section card container."""
    st.markdown("</div>", unsafe_allow_html=True)


def section_card(title: str, icon: Optional[str] = None):
    """
    Context manager for section cards.
    Usage:
        with section_card("My Section"):
            st.write("Content here")

    Note: Due to Streamlit limitations, this uses markdown injection
    which may not perfectly contain all Streamlit components.
    For best results, use section_card_start/end or custom containers.
    """
    class SectionCardContext:
        def __enter__(self):
            section_card_start(title, icon)
            return self

        def __exit__(self, *args):
            section_card_end()

    return SectionCardContext()


# Card Wrapper

def card_html(content: str, title: Optional[str] = None) -> None:
    """
    Wrap HTML content in a card.

    Args:
        content: HTML content
        title: Optional card title
    """
    title_html = f'<div class="card-header"><h4 class="card-title">{title}</h4></div>' if title else ""
    st.markdown(f"""
    <div class="card">
        {title_html}
        {content}
    </div>
    """, unsafe_allow_html=True)


# Ticker symbol 

def ticker_symbol(symbol: str) -> str:
    """
    Return HTML for a styled ticker symbol.

    Args:
        symbol: Ticker symbol (e.g., "AAPL", "BTC/USD")

    Returns:
        HTML string
    """
    return f'<span class="ticker-symbol">{symbol}</span>'


def render_ticker(symbol: str) -> None:
    """Render a ticker symbol directly."""
    st.markdown(ticker_symbol(symbol), unsafe_allow_html=True)


# Price value

def price_value(value: float, prefix: str = "$", is_change: bool = False) -> str:
    """
    Return HTML for a styled price value.

    Args:
        value: Numeric value
        prefix: Currency prefix
        is_change: If True, color based on positive/negative

    Returns:
        HTML string
    """
    formatted = f"{prefix}{value:,.2f}"
    if is_change:
        color_class = "price-positive" if value >= 0 else "price-negative"
        sign = "+" if value > 0 else ""
        formatted = f"{sign}{prefix}{value:,.2f}"
    else:
        color_class = ""

    return f'<span class="price-value {color_class}">{formatted}</span>'


def render_price(value: float, prefix: str = "$", is_change: bool = False) -> None:
    """Render a price value directly."""
    st.markdown(price_value(value, prefix, is_change), unsafe_allow_html=True)


# Info row

def info_row(items: list[tuple[str, str]], use_mono_values: bool = True) -> None:
    """
    Render a horizontal row of label-value pairs.

    Args:
        items: List of (label, value) tuples
        use_mono_values: If True, values use monospace font
    """
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            value_style = 'font-family: "JetBrains Mono", monospace;' if use_mono_values else ""
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 0.75rem; color: {COLORS['text_tertiary']}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
                    {label}
                </div>
                <div style="font-size: 1rem; color: {COLORS['text_primary']}; font-weight: 500; {value_style}">
                    {value}
                </div>
            </div>
            """, unsafe_allow_html=True)


# Divider with Label

def labeled_divider(label: str) -> None:
    """
    Render a divider with a centered label.

    Args:
        label: Text to display in the divider
    """
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin: 1.5rem 0;">
        <div style="flex: 1; height: 1px; background-color: {COLORS['border_subtle']};"></div>
        <div style="padding: 0 1rem; color: {COLORS['text_tertiary']}; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em;">
            {label}
        </div>
        <div style="flex: 1; height: 1px; background-color: {COLORS['border_subtle']};"></div>
    </div>
    """, unsafe_allow_html=True)


# Page header

def page_header(
    title: str,
    subtitle: Optional[str] = None,
    badge: Optional[tuple[str, str]] = None,  # (text, status)
) -> None:
    """
    Render a professional page header.

    Args:
        title: Main title
        subtitle: Optional subtitle text
        badge: Optional (text, status) tuple for a status badge
    """
    # Build badge HTML inline to avoid escaping issues
    badge_html = ""
    if badge:
        badge_text, badge_status = badge
        badge_colors = {
            "online": (COLORS["success"], COLORS["success_subtle"]),
            "offline": (COLORS["danger"], COLORS["danger_subtle"]),
            "warning": (COLORS["warning"], COLORS["warning_subtle"]),
        }
        color, bg_color = badge_colors.get(badge_status, (COLORS["text_secondary"], COLORS["bg_elevated"]))
        badge_html = f'''<span style="margin-left: 12px; display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 9999px; background-color: {bg_color}; color: {color}; font-size: 0.75rem; font-weight: 500;"><span style="width: 6px; height: 6px; border-radius: 50%; background-color: {color};"></span>{badge_text}</span>'''

    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<p style="color: {COLORS["text_secondary"]}; margin: 0; font-size: 0.9375rem;">{subtitle}</p>'

    html = f'''<div style="margin-bottom: 1.5rem;"><div style="display: flex; align-items: center; margin-bottom: 0.5rem;"><h1 style="margin: 0; font-size: 1.75rem; font-weight: 700; color: {COLORS["text_primary"]};">{title}</h1>{badge_html}</div>{subtitle_html}</div>'''

    st.markdown(html, unsafe_allow_html=True)


# Navigation menu styles

def get_nav_menu_styles() -> dict:
    """
    Return professional styles for streamlit_option_menu.

    Usage:
        from streamlit_option_menu import option_menu
        selected = option_menu(..., styles=get_nav_menu_styles())
    """
    return {
        "container": {
            "padding": "0",
            "background-color": "transparent",
        },
        "icon": {
            "color": COLORS["text_secondary"],
            "font-size": "16px",
        },
        "nav-link": {
            "font-family": "'Inter', sans-serif",
            "font-size": "0.875rem",
            "font-weight": "500",
            "text-align": "left",
            "margin": "2px 0",
            "padding": "0.625rem 0.875rem",
            "color": COLORS["text_secondary"],
            "background-color": "transparent",
            "border-radius": "8px",
            "--hover-color": COLORS["bg_card_hover"],
        },
        "nav-link-selected": {
            "background-color": COLORS["bg_card"],
            "color": COLORS["text_primary"],
            "font-weight": "600",
            "border": f"1px solid {COLORS['border_default']}",
        },
    }


# Plotly chart theme

def get_plotly_layout() -> dict:
    """
    Return a professional Plotly layout configuration.

    Usage:
        fig.update_layout(**get_plotly_layout())
    """
    return {
        "paper_bgcolor": COLORS["bg_card"],
        "plot_bgcolor": COLORS["bg_card"],
        "font": {
            "family": "Inter, sans-serif",
            "color": COLORS["text_secondary"],
            "size": 12,
        },
        "title": {
            "font": {
                "family": "Inter, sans-serif",
                "size": 16,
                "color": COLORS["text_primary"],
            },
            "x": 0,
            "xanchor": "left",
        },
        "xaxis": {
            "gridcolor": COLORS["border_subtle"],
            "linecolor": COLORS["border_default"],
            "tickfont": {"family": "JetBrains Mono, monospace", "size": 11},
            "title_font": {"family": "Inter, sans-serif", "size": 12},
        },
        "yaxis": {
            "gridcolor": COLORS["border_subtle"],
            "linecolor": COLORS["border_default"],
            "tickfont": {"family": "JetBrains Mono, monospace", "size": 11},
            "title_font": {"family": "Inter, sans-serif", "size": 12},
        },
        "legend": {
            "font": {"family": "Inter, sans-serif", "size": 11},
            "bgcolor": "rgba(0,0,0,0)",
        },
        "margin": {"l": 60, "r": 20, "t": 40, "b": 40},
    }


def get_chart_colors() -> list[str]:
    """
    Return a professional color palette for charts.

    Usage:
        fig = px.line(..., color_discrete_sequence=get_chart_colors())
    """
    return [
        "#388BFD",  # Primary blue
        "#3FB950",  # Success green
        "#F85149",  # Danger red
        "#D29922",  # Warning amber
        "#A371F7",  # Purple
        "#79C0FF",  # Light blue
        "#7EE787",  # Light green
        "#FFA657",  # Orange
    ]


# quick styles injection
def inject_custom_css() -> None:
    """
    Inject additional custom CSS that can't be in the main stylesheet.
    Call this early in your app if needed.
    """
    st.markdown("""
    <style>
    /* Additional runtime CSS overrides */

    /* Fix for Streamlit's default container padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Ensure sidebar doesn't have excessive padding */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important;
    }

    /* Make sidebar scrollable when content overflows */
    section[data-testid="stSidebar"] > div:first-child {
        overflow-y: auto !important;
        max-height: 100vh !important;
    }

    /* Clean up metric styling in columns */
    div[data-testid="metric-container"] {
        background-color: transparent;
    }

    /* SIDEBAR PROTECTION - Prevent card/container styles from leaking */
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div,
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"],
    section[data-testid="stSidebar"] div[data-testid="element-container"],
    section[data-testid="stSidebar"] .nav-container {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Ensure sidebar text stays visible */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span:not(.nav-link-selected span) {
        color: #8B949E !important;
    }
    </style>
    """, unsafe_allow_html=True)
