# QuantTerm Pro

**An institutional-grade research and execution terminal featuring an asymmetric Neuro-Symbolic architecture, Walk-Forward optimization, and Modern Portfolio Theory (MPT) allocation.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Alpaca](https://img.shields.io/badge/Alpaca-API-00C805?style=flat-square&logo=alpaca&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-0D0D0D?style=flat-square&logo=ollama&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> **[Watch the 3-Minute System Architecture Demo Here] (Loom link)**

---

## Architecture: The Asymmetric Neuro-Symbolic Edge

Most algorithmic systems fail in one of two ways: purely mathematical systems **overfit** to historical noise; purely AI-driven systems **hallucinate** in live markets. QuantTerm Pro solves this with a two-layer pipeline that assigns each task to the engine best suited for it.

### Layer 1 — The Math Engine (Vectorized Backtesting)

Sub-second, fully vectorized computation (VectorBT + NumPy) processes **10+ years of tick data** in a single matrix operation. This layer is responsible for:

- **Grid Search Optimization** — exhaustive sweep over all parameter combinations (SL %, TP %, signal thresholds) to find the empirically dominant regime.
- **Rolling Walk-Forward Optimization** — data is never seen all at once. The optimizer trains on an in-sample window, freezes the best parameters, then evaluates on a strictly unseen out-of-sample window. This is the only statistically honest measure of edge retention.
- **Macro Regime Hysteresis Filter** — live FRED economic data (HY credit spreads, yield curve) is processed through a state machine with a dead band and persistence requirement to suppress equity longs during macro stress — without generating false signals from noise.

No LLM token is wasted on a task that algebra handles perfectly.

### Layer 2 — The Agent (Local LLM Execution Manager)

Once the mathematical edge is confirmed and parameters are frozen, a **local Mixtral LLM (via Ollama)** acts as a live Portfolio Manager. It does not discover edge — it _validates execution context_. On each evaluation cycle the agent:

1. Receives a structured prompt containing the last 10 bars of price action, indicator state (RSI, SMA, ATR), and volume profile.
2. Verifies qualitative conditions a regression cannot encode: volume confirmation, proximity to key support/resistance, macro narrative alignment.
3. Returns a deterministic **JSON execution thesis** that either confirms or overrides the mathematical signal.

```json
{
  "signal": "BUY",
  "confidence": 0.82,
  "reasoning": "RSI recovering from oversold at 200-SMA support on above-average volume. HY spreads stable."
}
```

The result is an asymmetric advantage: the speed and rigour of systematic quant research, augmented by the contextual judgment of an AI that runs entirely offline with zero API costs.

---

## 🖥️ System Previews

<table align="center" style="border: none; background-color: transparent;">
<tr>
<td align="center" width="50%"><b>Portfolio Optimizer & Efficient Frontier</b></td>
<td align="center" width="50%"><b>Live Terminal & AI Copilot</b></td>
</tr>
<tr>
<td align="center"><img src="https://github.com/user-attachments/assets/0018a99e-d8e0-41a4-8551-91fcd68cc840" alt="Portfolio Optimizer" width="100%"/></td>
<td align="center"><img src="https://github.com/user-attachments/assets/c7eb8c1c-c9b9-4173-adc4-98bcb25d22f0" alt="Live Terminal" width="100%"/></td>
</tr>
<tr>
<td align="center" width="50%"><b>Walk-Forward OOS Equity</b></td>
<td align="center" width="50%"><b>Global Macro Dashboard</b></td>
</tr>
<tr>
<td align="center"><img src="https://github.com/user-attachments/assets/5c75a3ad-099c-4d6f-8ad8-39aef850b9a5" alt="Walk-Forward" width="100%"/></td>
<td align="center"><img src="https://github.com/user-attachments/assets/2d02e492-0fb6-44c3-bb2b-77e5f1b8da84" alt="Macro Dashboard" width="100%"/></td>
</tr>
</table>

---

## Core Platform Features

### Advanced Portfolio Allocation (MPT)

Two independent optimization models run in parallel so their allocations can be compared side-by-side:

- **Mean-Variance Optimization (Markowitz)** — uses [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/) to calculate the full Markowitz Efficient Frontier and select the maximum-Sharpe portfolio. The covariance matrix is estimated from historical returns; expected returns are derived via mean historical return (`expected_returns.mean_historical_return`).
- **Hierarchical Risk Parity (HRP)** — uses PyPortfolioOpt's `HRPOpt` to optimize allocation, with SciPy's agglomerative clustering used separately to build and visualize the asset correlation dendrogram. Capital is allocated inversely proportional to cluster variance, producing diversified portfolios that are robust to covariance estimation error — a known structural failure mode of classical Markowitz.
- **Discrete Share Allocation** — portfolio percentage weights are automatically converted to whole-share allocations at the current market price, producing a buy list that is directly executable in a brokerage account without manual calculation.
- **Rebalancing Simulation** — configurable drift-or-rebalance engine (Monthly / Quarterly / Yearly / Never) models the real compounding impact of transaction costs at each rebalance event.

### Institutional Risk Management

- **Walk-Forward Out-of-Sample Testing** — rolling train/test windows with configurable lengths (e.g. 4-year train / 2-year test). Each fold independently optimizes parameters, then freezes them for the subsequent unseen window. The OOS equity curve is stitched exclusively from frozen-parameter test windows — the only intellectually honest backtesting methodology.
- **Net-of-Costs Friction** — every simulated portfolio deducts a 10 bps round-trip transaction cost. The OOS equity chart overlays both gross and net curves to make fee drag explicit and measurable.
- **Per-Fold Diagnostics Table** — each walk-forward fold independently reports OOS Sharpe, OOS Max Drawdown, Turnover %, and Hit Rate % in a structured comparison table.
- **Hysteresis Sensitivity Heatmap** — a 5×5 parameter sweep over entry/exit band thresholds generates a full OOS Sharpe surface. A broad green plateau around the chosen ±0.10 thresholds is the quantitative proof that the configuration is robust, not coincidentally optimal.
- **Macro Regime Hysteresis Filter** — live data from the [FRED API](https://fred.stlouisfed.org/) (`BAMLH0A0HYM2` HY credit spread, `DGS10` 10Y yield, `T10Y2Y` yield curve) feeds a state machine with:
  - 10-day EMA smoothing on the credit spread deviation from its 50-day SMA
  - A dual-threshold dead band (±0.10) defining the transition zone
  - A 5-bar persistence requirement before any regime flip is committed
  - Safe-haven rotation: GLD and TLT automatically receive long signals during risk-off periods

### Options Speculation Lab

- **Black-Scholes Pricing Engine** — custom implementation using `scipy.stats.norm` to price European calls and puts from first principles, outputting the primary Greeks: Delta, Gamma, Theta, and Vega.
- **Leverage Profiling** — models asymmetric return profiles across a configurable range of underlying price moves, making the risk/reward structure of an options position visually explicit before capital is committed.
- **Profit Curve Simulation** — interactive Plotly chart of theoretical P&L vs. underlying price at expiry.

### Live Order Management System (OMS)

- **Asynchronous Alpaca Integration** — connects to the Alpaca brokerage API to display live account equity, open positions with real-time unrealised P&L, and a full timestamped order history.
- **AI Copilot Override** — a single button in the Live Terminal tab fetches the most recent 3 months of live market data for the selected ticker, runs the full indicator pipeline, and dispatches structured context to the local Mixtral LLM for an instant JSON execution thesis — designed for final human-in-the-loop confirmation before order submission.

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally

### Step 1 — Install the Local LLM

```bash
# Install Ollama (macOS / Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the Mixtral model (~26 GB, runs fully offline after download)
ollama pull mixtral
```

### Step 2 — Clone & Install Dependencies

```bash
git clone https://github.com/YOUR_USERNAME/quantterm-pro.git
cd quantterm-pro

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Step 3 — Configure API Keys

Create a `.env` file in the project root:

```env
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
```

> **Note:** Use Alpaca **Paper Trading** keys for development. The base URL in `config.py` defaults to `https://paper-api.alpaca.markets`.

### Step 4 — Launch the Terminal

```bash
streamlit run dashboard.py
```

Navigate to **http://localhost:8501**. All computation — backtesting, LLM inference, and portfolio optimization — runs locally. No cloud inference costs.

---

## Project Structure

```
quantterm-pro/
│
├── dashboard.py            # Main Streamlit application — all page routing and UI layout
├── strategy.py             # Strategy classes (HybridStrategy, Breakout, MeanReversion, etc.)
├── backtest_runner.py      # Single-pass backtest executor (used by Quick Backtest tab)
├── macro_intelligence.py   # FRED data fetcher, hysteresis state machine, Macro tab renderer
├── portfolio_manager.py    # Markowitz & HRP optimization, rebalancing simulation engine
├── options_manager.py      # Black-Scholes pricing, Greeks calculation, leverage profiling
├── ai_brain.py             # Ollama/Mixtral client — analyze_market() returns structured JSON
├── live_trader.py          # Alpaca order execution logic and real-time position management
├── ui_components.py        # Reusable Streamlit UI primitives (metric cards, badges, CSS)
├── config.py               # Global constants, API base URLs, default strategy parameters
├── custom_strategies.py    # User-defined strategy subclasses for rapid experimentation
│
├── requirements.txt        # Pinned Python dependencies
├── .env                    # API credentials (git-ignored)
└── README.md
```

---

## About the Developer

Michael Fultz is a Quantitative Software Engineer bridging a foundation in traditional Finance with advanced Computer Science and Agentic AI. _QuantTerm Pro_ was architected to demonstrate the translation of complex market microstructure and macro regime hysteresis into robust, production-ready Python pipelines that dynamically manage tail risk and preserve capital.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/michael-fultz-601aa0262/)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-000000?style=flat-square&logo=github&logoColor=white)](portfolio url)

---

## Disclaimer

This project is for **educational and research purposes only**. Nothing in this repository constitutes financial advice or a solicitation to buy or sell any security. All backtests are conducted on historical data and past performance is not indicative of future results. All live trading features are designed for use with **paper trading accounts only**. The authors accept no liability for financial losses incurred through the use of this software.

---

<p align="center">
  <sub>Built with Streamlit · VectorBT · PyPortfolioOpt · Alpaca · Ollama · Plotly</sub>
</p>
