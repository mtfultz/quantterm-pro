"""
Portfolio Management System
Professional portfolio optimization using Modern Portfolio Theory
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Professional Portfolio Optimizer using Modern Portfolio Theory.

    Features:
    - Fetches historical data for multiple tickers
    - Calculates optimal portfolio weights (Max Sharpe Ratio)
    - Converts percentage weights to discrete share allocations
    - Generates Efficient Frontier visualization
    - Dynamic S&P 500 ticker fetching from Wikipedia
    """

    @staticmethod
    def get_sp500_tickers():
        """
        Fetch the current S&P 500 ticker list from Wikipedia.
        Uses proper User-Agent header to avoid 403 Forbidden errors.

        Returns:
            list: List of S&P 500 ticker symbols

        Raises:
            Exception: If Wikipedia scraping fails
        """
        try:
            import requests
            from io import StringIO

            logger.info("Fetching S&P 500 tickers from Wikipedia...")

            # Scrape S&P 500 list from Wikipedia with proper headers
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            # Fetch the page with proper User-Agent
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise exception for HTTP errors

            # Parse HTML tables from response
            tables = pd.read_html(StringIO(response.text))

            # First table contains the S&P 500 list
            sp500_table = tables[0]

            # Extract ticker symbols (column is named 'Symbol')
            tickers = sp500_table['Symbol'].tolist()

            # Clean tickers (replace dots with dashes for compatibility with yfinance)
            # Example: BRK.B -> BRK-B
            tickers = [ticker.replace('.', '-') for ticker in tickers]

            logger.info(f"Successfully fetched {len(tickers)} S&P 500 tickers")

            return sorted(tickers)

        except Exception as e:
            logger.error(f"Failed to fetch S&P 500 tickers: {e}")
            raise Exception(f"Could not load S&P 500 tickers from Wikipedia: {e}")

    @staticmethod
    def get_top_crypto_tickers():
        """
        Fetch the top 100 cryptocurrency tickers from Yahoo Finance.
        Uses proper User-Agent header to avoid 403 Forbidden errors.
        Paginates through multiple pages to collect 100+ tickers.

        Returns:
            list: List of crypto ticker symbols (e.g., BTC-USD, ETH-USD)

        Raises:
            Exception: If Yahoo Finance scraping fails
        """
        try:
            import requests
            from io import StringIO
            import time

            logger.info("Fetching top 100 crypto tickers from Yahoo Finance...")

            all_tickers = []
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            # Paginate through Yahoo Finance crypto pages (25 per page)
            for offset in [0, 25, 50, 75]:
                try:
                    # Construct URL with offset
                    url = f"https://finance.yahoo.com/crypto?offset={offset}&count=25"

                    logger.info(f"Fetching page at offset {offset}...")

                    # Fetch the page with proper User-Agent
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()  # Raise exception for HTTP errors

                    # Parse HTML tables from response
                    tables = pd.read_html(StringIO(response.text))

                    # First table usually contains the crypto list
                    if tables:
                        crypto_table = tables[0]

                        # Extract ticker symbols (column is named 'Symbol')
                        if 'Symbol' in crypto_table.columns:
                            page_tickers = crypto_table['Symbol'].tolist()
                            # Filter to only include valid crypto tickers (contain -USD, -USDT, etc.)
                            page_tickers = [t for t in page_tickers if '-' in str(t)]
                            all_tickers.extend(page_tickers)
                            logger.info(f"Found {len(page_tickers)} tickers on page (offset {offset})")

                    # Sleep between requests to avoid rate limiting
                    if offset < 75:  # Don't sleep after last request
                        time.sleep(1)

                except Exception as page_error:
                    logger.warning(f"Failed to fetch page at offset {offset}: {page_error}")
                    continue

            # Remove duplicates and sort
            unique_tickers = sorted(list(set(all_tickers)))

            if len(unique_tickers) > 0:
                logger.info(f"Successfully fetched {len(unique_tickers)} unique crypto tickers")
                return unique_tickers
            else:
                # If no tickers found, raise exception to trigger fallback
                raise Exception("No crypto tickers found on any page")

        except Exception as e:
            logger.error(f"Failed to fetch crypto tickers: {e}")
            logger.info("Using fallback crypto list")

            # Fallback list of top cryptocurrencies
            fallback_cryptos = [
                "BTC-USD",    # Bitcoin
                "ETH-USD",    # Ethereum
                "BNB-USD",    # Binance Coin
                "SOL-USD",    # Solana
                "XRP-USD",    # Ripple
                "ADA-USD",    # Cardano
                "DOGE-USD",   # Dogecoin
                "MATIC-USD",  # Polygon
                "DOT-USD",    # Polkadot
                "AVAX-USD",   # Avalanche
                "LINK-USD",   # Chainlink
                "UNI-USD",    # Uniswap
            ]
            return fallback_cryptos

    def __init__(self, tickers, capital):
        """
        Initialize Portfolio Optimizer.

        Args:
            tickers (list): List of stock ticker symbols
            capital (float): Total investment capital in dollars
        """
        self.tickers = tickers
        self.capital = capital
        self.prices = None
        self.weights = None
        self.cleaned_weights = None
        self.expected_return = None
        self.annual_volatility = None
        self.sharpe_ratio = None
        self.allocation = None
        self.leftover = None

        logger.info(f"PortfolioOptimizer initialized with {len(tickers)} tickers and ${capital:,.2f} capital")

    def fetch_data(self, period="1y"):
        """
        Download historical price data for all tickers.

        Args:
            period (str): Time period for historical data (default: 1y)

        Returns:
            pd.DataFrame: Closing prices for all tickers
        """
        logger.info(f"Fetching {period} of price data for {len(self.tickers)} tickers...")

        try:
            # Download data for all tickers
            data = yf.download(
                self.tickers,
                period=period,
                progress=False
            )

            # Handle single ticker case
            if len(self.tickers) == 1:
                self.prices = pd.DataFrame({self.tickers[0]: data['Close']})
            else:
                self.prices = data['Close']

            # Drop tickers with no data
            self.prices = self.prices.dropna(axis=1, how='all')

            # Store initial date range before alignment
            initial_start = self.prices.index[0] if len(self.prices) > 0 else None
            initial_end = self.prices.index[-1] if len(self.prices) > 0 else None

            # When mixing stocks/crypto or young assets (e.g., COIN, SOL-USD) with older assets,
            # we MUST align to the period where ALL assets have data to prevent NaN errors
            # in the covariance matrix calculation.
            initial_rows = len(self.prices)
            self.prices = self.prices.dropna(axis=0, how='any')
            dropped_rows = initial_rows - len(self.prices)

            # Update tickers list to only include valid ones
            valid_tickers = list(self.prices.columns)
            invalid_tickers = set(self.tickers) - set(valid_tickers)

            if invalid_tickers:
                logger.warning(f"Could not fetch data for: {invalid_tickers}")
                self.tickers = valid_tickers

            if len(self.prices) == 0:
                raise ValueError("No common history found for selected tickers. Try a shorter period or different assets.")

            # Get final aligned date range
            final_start = self.prices.index[0]
            final_end = self.prices.index[-1]

            logger.info(f"Successfully fetched data for {len(self.tickers)} tickers")

            # Show prominent warning if data was aligned to common history
            if dropped_rows > 0:
                logger.warning(f"COMMON HISTORY ALIGNMENT: Data aligned to shortest asset history")
                logger.warning(f"   Requested period: {period}")
                logger.warning(f"   Actual date range: {final_start.date()} to {final_end.date()}")
                logger.warning(f"   Dropped {dropped_rows} rows to ensure all assets have complete data")
                logger.warning(f"   This prevents covariance matrix errors from young assets (e.g., COIN, SOL-USD)")
            else:
                logger.info(f"Date range: {final_start.date()} to {final_end.date()}")

            return self.prices

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    def optimize_portfolio(self, risk_free_rate=0.02):
        """
        Calculate optimal portfolio weights using Maximum Sharpe Ratio.

        Args:
            risk_free_rate (float): Risk-free rate for Sharpe calculation (default: 0.02)

        Returns:
            dict: Cleaned portfolio weights (only non-zero positions)
        """
        if self.prices is None:
            raise ValueError("Must call fetch_data() first!")

        logger.info("Calculating expected returns and covariance matrix...")

        try:
            # Calculate expected returns (mean historical return)
            mu = expected_returns.mean_historical_return(self.prices)

            # Calculate sample covariance matrix
            S = risk_models.sample_cov(self.prices)

            # Optimize for maximum Sharpe ratio
            logger.info("Optimizing portfolio for Maximum Sharpe Ratio...")
            ef = EfficientFrontier(mu, S)
            raw_weights = ef.max_sharpe(risk_free_rate=risk_free_rate)

            # Clean weights (remove tiny positions below 0.01%)
            self.cleaned_weights = ef.clean_weights()

            # Get performance metrics
            performance = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            self.expected_return = performance[0]
            self.annual_volatility = performance[1]
            self.sharpe_ratio = performance[2]

            logger.info(f"Optimization complete!")
            logger.info(f"Expected Annual Return: {self.expected_return * 100:.2f}%")
            logger.info(f"Annual Volatility: {self.annual_volatility * 100:.2f}%")
            logger.info(f"Sharpe Ratio: {self.sharpe_ratio:.2f}")

            # Store weights as dict (only non-zero)
            self.weights = {k: v for k, v in self.cleaned_weights.items() if v > 0.0001}

            return self.weights

        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            raise

    def get_allocation(self):
        """
        Convert percentage weights to discrete share allocations.

        Returns:
            pd.DataFrame: Allocation table with ticker, price, shares, value, weight
        """
        if self.weights is None:
            raise ValueError("Must call optimize_portfolio() first!")

        logger.info("Converting weights to discrete share allocations...")

        try:
            # Get latest prices
            latest_prices = get_latest_prices(self.prices)

            # Calculate discrete allocation
            da = DiscreteAllocation(
                self.weights,
                latest_prices,
                total_portfolio_value=self.capital
            )

            allocation, leftover = da.greedy_portfolio()
            self.allocation = allocation
            self.leftover = leftover

            # Build allocation DataFrame
            allocation_data = []
            total_value = 0

            for ticker, shares in allocation.items():
                price = latest_prices[ticker]
                value = shares * price
                total_value += value
                weight = value / self.capital * 100

                allocation_data.append({
                    'Ticker': ticker,
                    'Price': price,
                    'Shares': shares,
                    'Value': value,
                    'Weight (%)': weight
                })

            df_allocation = pd.DataFrame(allocation_data)

            # Sort by value descending
            df_allocation = df_allocation.sort_values('Value', ascending=False).reset_index(drop=True)

            logger.info(f"Allocation complete: {len(allocation)} positions")
            logger.info(f"Total invested: ${total_value:,.2f}")
            logger.info(f"Leftover cash: ${leftover:,.2f}")

            return df_allocation

        except Exception as e:
            logger.error(f"Error calculating allocation: {e}")
            raise

    def plot_efficient_frontier(self, num_portfolios=10000):
        """
        Generate Efficient Frontier plot with optimal portfolio marked.

        Args:
            num_portfolios (int): Number of random portfolios to simulate

        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if self.prices is None:
            raise ValueError("Must call fetch_data() first!")

        logger.info(f"Generating Efficient Frontier with {num_portfolios} simulations...")

        try:
            # Calculate expected returns and covariance
            mu = expected_returns.mean_historical_return(self.prices)
            S = risk_models.sample_cov(self.prices)

            # Simulate random portfolios
            results = np.zeros((3, num_portfolios))

            for i in range(num_portfolios):
                # Random weights
                weights = np.random.random(len(self.tickers))
                weights /= np.sum(weights)

                # Calculate return and volatility
                portfolio_return = np.dot(weights, mu)
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(S, weights)))

                # Calculate Sharpe ratio
                sharpe = portfolio_return / portfolio_std

                results[0, i] = portfolio_return
                results[1, i] = portfolio_std
                results[2, i] = sharpe

            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot all simulated portfolios
            scatter = ax.scatter(
                results[1, :] * 100,
                results[0, :] * 100,
                c=results[2, :],
                cmap='viridis',
                marker='o',
                s=10,
                alpha=0.3
            )

            # Plot optimal portfolio
            if self.annual_volatility and self.expected_return:
                ax.scatter(
                    self.annual_volatility * 100,
                    self.expected_return * 100,
                    marker='*',
                    color='red',
                    s=500,
                    edgecolors='black',
                    label=f'Optimal Portfolio\n(Sharpe: {self.sharpe_ratio:.2f})',
                    zorder=5
                )

            # Styling
            ax.set_xlabel('Annual Volatility (%)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Expected Annual Return (%)', fontsize=12, fontweight='bold')
            ax.set_title('Efficient Frontier - Modern Portfolio Theory', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='upper left')

            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Sharpe Ratio', fontsize=10, fontweight='bold')

            plt.tight_layout()

            logger.info("Efficient Frontier plot generated successfully")

            return fig

        except Exception as e:
            logger.error(f"Error generating Efficient Frontier: {e}")
            raise

    def get_summary(self):
        """
        Get a summary of the optimized portfolio.

        Returns:
            dict: Summary metrics
        """
        return {
            'Number of Positions': len(self.allocation) if self.allocation else 0,
            'Total Capital': self.capital,
            'Invested Amount': self.capital - (self.leftover if self.leftover else 0),
            'Leftover Cash': self.leftover if self.leftover else 0,
            'Expected Annual Return': f"{self.expected_return * 100:.2f}%" if self.expected_return else "N/A",
            'Annual Volatility': f"{self.annual_volatility * 100:.2f}%" if self.annual_volatility else "N/A",
            'Sharpe Ratio': f"{self.sharpe_ratio:.2f}" if self.sharpe_ratio else "N/A"
        }


# Standalone test
if __name__ == "__main__":
    # Test the optimizer
    logging.basicConfig(level=logging.INFO)

    test_tickers = ['AAPL', 'MSFT', 'NVDA', 'JPM', 'JNJ']
    test_capital = 10000

    optimizer = PortfolioOptimizer(test_tickers, test_capital)
    optimizer.fetch_data()
    optimizer.optimize_portfolio()
    allocation = optimizer.get_allocation()

    print("\n" + "="*60)
    print("OPTIMIZED PORTFOLIO ALLOCATION")
    print("="*60)
    print(allocation.to_string(index=False))
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for key, value in optimizer.get_summary().items():
        print(f"{key:.<30} {value}")
    print("="*60)
