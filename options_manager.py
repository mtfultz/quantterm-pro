"""
Options Speculation Lab
Advanced options analysis for identifying high-leverage speculative plays
"""

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class OptionsManager:
    """
    Professional Options Manager for speculative trading.

    Features:
    - Fetch real-time options chains
    - Calculate Greeks (Delta, Gamma, Theta, Vega, IV)
    - Identify "Lotto Tickets" (high leverage, low cost OTM options)
    - Profit calculator for theoretical pricing
    """

    def __init__(self, ticker):
        """
        Initialize Options Manager.

        Args:
            ticker (str): Stock ticker symbol (e.g., "SPY", "TSLA")
        """
        self.ticker = ticker
        self.stock = None
        self.current_price = None
        self.options_chain = None
        self.expirations = []

        logger.info(f"OptionsManager initialized for {ticker}")

    def fetch_stock_data(self):
        """
        Fetch current stock price and basic info.

        Returns:
            dict: Stock information
        """
        try:
            logger.info(f"Fetching stock data for {self.ticker}...")
            self.stock = yf.Ticker(self.ticker)

            # Get current price
            hist = self.stock.history(period="1d")
            if not hist.empty:
                self.current_price = hist['Close'].iloc[-1]
            else:
                # Fallback to info
                info = self.stock.info
                self.current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))

            logger.info(f"Current price for {self.ticker}: ${self.current_price:.2f}")

            return {
                'ticker': self.ticker,
                'current_price': self.current_price,
                'name': self.stock.info.get('longName', self.ticker)
            }

        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            raise

    def get_expirations(self):
        """
        Get available option expiration dates.

        Returns:
            list: Available expiration dates
        """
        try:
            if self.stock is None:
                self.fetch_stock_data()

            self.expirations = self.stock.options
            logger.info(f"Found {len(self.expirations)} expiration dates")

            return self.expirations

        except Exception as e:
            logger.error(f"Error fetching expirations: {e}")
            raise

    def fetch_chain(self, expiration_date):
        """
        Fetch options chain for a specific expiration date.

        Args:
            expiration_date (str): Expiration date in YYYY-MM-DD format

        Returns:
            tuple: (calls_df, puts_df)
        """
        try:
            if self.stock is None:
                self.fetch_stock_data()

            logger.info(f"Fetching options chain for {expiration_date}...")

            # Get options chain
            opt = self.stock.option_chain(expiration_date)
            calls = opt.calls
            puts = opt.puts

            # Add current stock price for reference
            calls['stockPrice'] = self.current_price
            puts['stockPrice'] = self.current_price

            # Add expiration date
            calls['expiration'] = expiration_date
            puts['expiration'] = expiration_date

            # Calculate days to expiration
            exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
            today = datetime.now()
            days_to_exp = (exp_date - today).days
            calls['daysToExpiration'] = days_to_exp
            puts['daysToExpiration'] = days_to_exp

            logger.info(f"Fetched {len(calls)} calls and {len(puts)} puts")

            self.options_chain = {'calls': calls, 'puts': puts, 'expiration': expiration_date}

            return calls, puts

        except Exception as e:
            logger.error(f"Error fetching options chain: {e}")
            raise

    @staticmethod
    def black_scholes(S, K, T, r, sigma, option_type='call'):
        """
        Calculate Black-Scholes option price and Greeks.

        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration (years)
            r (float): Risk-free rate
            sigma (float): Volatility (annualized)
            option_type (str): 'call' or 'put'

        Returns:
            dict: Price, Delta, Gamma, Theta, Vega
        """
        if T <= 0:
            # Option expired
            if option_type == 'call':
                return {
                    'price': max(S - K, 0),
                    'delta': 1.0 if S > K else 0.0,
                    'gamma': 0.0,
                    'theta': 0.0,
                    'vega': 0.0
                }
            else:
                return {
                    'price': max(K - S, 0),
                    'delta': -1.0 if S < K else 0.0,
                    'gamma': 0.0,
                    'theta': 0.0,
                    'vega': 0.0
                }

        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = -norm.cdf(-d1)

        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

        # Theta
        if option_type == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                     - r * K * np.exp(-r * T) * norm.cdf(d2))
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                     + r * K * np.exp(-r * T) * norm.cdf(-d2))

        # Vega (same for calls and puts)
        vega = S * norm.pdf(d1) * np.sqrt(T)

        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365,  # Daily theta
            'vega': vega / 100  # Vega per 1% change in IV
        }

    def calculate_spec_metrics(self, options_df, option_type='call'):
        """
        Calculate speculation metrics for options.

        Args:
            options_df (pd.DataFrame): Options dataframe
            option_type (str): 'call' or 'put'

        Returns:
            pd.DataFrame: Enhanced dataframe with spec metrics
        """
        df = options_df.copy()

        # Calculate moneyness
        df['moneyness'] = (df['strike'] / df['stockPrice'] - 1) * 100

        # Calculate Delta using Black-Scholes if not provided by API
        if 'delta' not in df.columns or df['delta'].isna().all():
            logger.info("Delta not provided by API, calculating via Black-Scholes...")
            deltas = []
            gammas = []
            for idx, row in df.iterrows():
                S = row['stockPrice']
                K = row['strike']
                T = max(row['daysToExpiration'] / 365.0, 0.001)  # Time in years, avoid zero
                r = 0.05  # Risk-free rate assumption
                # Use implied volatility from the chain, default to 30% if missing
                sigma = row.get('impliedVolatility', 0.30)
                if pd.isna(sigma) or sigma <= 0:
                    sigma = 0.30

                greeks = self.black_scholes(S, K, T, r, sigma, option_type)
                deltas.append(greeks['delta'])
                gammas.append(greeks['gamma'])

            df['delta'] = deltas
            df['gamma'] = gammas

        # Calculate leverage ratio (Omega/Gearing): how much option moves vs stock
        # Leverage = (Delta * Stock Price) / Option Price
        # Ensure option price is not zero to avoid division errors
        option_price_safe = df['lastPrice'].replace(0, np.nan).fillna(0.01)
        option_price_safe = option_price_safe.apply(lambda x: max(x, 0.01))
        df['leverage'] = (abs(df['delta']) * df['stockPrice']) / option_price_safe

        # Calculate break-even move (percentage stock needs to move to break even)
        if option_type == 'call':
            df['breakEvenMove'] = ((df['strike'] + df['lastPrice'] - df['stockPrice']) / df['stockPrice']) * 100
        else:
            df['breakEvenMove'] = ((df['stockPrice'] - df['strike'] + df['lastPrice']) / df['stockPrice']) * 100

        # Gamma rank (percentile ranking of gamma)
        if 'gamma' in df.columns:
            df['gammaRank'] = df['gamma'].rank(pct=True) * 100

        # Speculation score (composite metric)
        # High gamma + high leverage + low cost = high spec score
        if 'gamma' in df.columns and 'leverage' in df.columns:
            gamma_norm = (df['gamma'] - df['gamma'].min()) / (df['gamma'].max() - df['gamma'].min() + 1e-10)
            leverage_norm = (df['leverage'] - df['leverage'].min()) / (df['leverage'].max() - df['leverage'].min() + 1e-10)
            cost_norm = 1 - ((df['lastPrice'] - df['lastPrice'].min()) / (df['lastPrice'].max() - df['lastPrice'].min() + 1e-10))

            df['specScore'] = (gamma_norm * 0.4 + leverage_norm * 0.4 + cost_norm * 0.2) * 100

        return df

    def find_lotto_tickets(self, max_price=1.00, min_gamma_rank=70, dte_range=(7, 45)):
        """
        Find "Lotto Ticket" options - high leverage, low cost speculative plays.

        Args:
            max_price (float): Maximum option price (default $1.00)
            min_gamma_rank (float): Minimum gamma percentile rank (default 70)
            dte_range (tuple): Days to expiration range (default 7-45 days)

        Returns:
            dict: {'calls': DataFrame, 'puts': DataFrame}
        """
        if self.options_chain is None:
            raise ValueError("Must call fetch_chain() first!")

        logger.info("Searching for Lotto Tickets...")

        calls = self.options_chain['calls'].copy()
        puts = self.options_chain['puts'].copy()

        # Calculate spec metrics
        calls = self.calculate_spec_metrics(calls, 'call')
        puts = self.calculate_spec_metrics(puts, 'put')

        # Filter criteria
        lotto_calls = calls[
            (calls['lastPrice'] <= max_price) &
            (calls['lastPrice'] > 0.01) &  # Must have some value
            (calls['volume'] > 0) &  # Must have volume
            (calls['daysToExpiration'] >= dte_range[0]) &
            (calls['daysToExpiration'] <= dte_range[1])
        ]

        lotto_puts = puts[
            (puts['lastPrice'] <= max_price) &
            (puts['lastPrice'] > 0.01) &
            (puts['volume'] > 0) &
            (puts['daysToExpiration'] >= dte_range[0]) &
            (puts['daysToExpiration'] <= dte_range[1])
        ]

        # Filter by gamma rank if available
        if 'gammaRank' in lotto_calls.columns:
            lotto_calls = lotto_calls[lotto_calls['gammaRank'] >= min_gamma_rank]

        if 'gammaRank' in lotto_puts.columns:
            lotto_puts = lotto_puts[lotto_puts['gammaRank'] >= min_gamma_rank]

        # Sort by spec score
        if 'specScore' in lotto_calls.columns:
            lotto_calls = lotto_calls.sort_values('specScore', ascending=False)

        if 'specScore' in lotto_puts.columns:
            lotto_puts = lotto_puts.sort_values('specScore', ascending=False)

        logger.info(f"Found {len(lotto_calls)} call lottos and {len(lotto_puts)} put lottos")

        return {
            'calls': lotto_calls,
            'puts': lotto_puts
        }

    def calculate_profit(self, strike, option_price, option_type, new_stock_price, contracts=1):
        """
        Calculate profit/loss for an option position at a new stock price.

        Args:
            strike (float): Strike price
            option_price (float): Option purchase price
            option_type (str): 'call' or 'put'
            new_stock_price (float): Hypothetical new stock price
            contracts (int): Number of contracts (default 1)

        Returns:
            dict: Profit/loss metrics
        """
        # Calculate intrinsic value at new price
        if option_type == 'call':
            intrinsic_value = max(new_stock_price - strike, 0)
        else:
            intrinsic_value = max(strike - new_stock_price, 0)

        # Profit per contract (100 shares per contract)
        profit_per_contract = (intrinsic_value - option_price) * 100
        total_profit = profit_per_contract * contracts

        # ROI
        cost_basis = option_price * 100 * contracts
        roi = (total_profit / cost_basis * 100) if cost_basis > 0 else 0

        return {
            'intrinsic_value': intrinsic_value,
            'profit_per_contract': profit_per_contract,
            'total_profit': total_profit,
            'roi': roi,
            'cost_basis': cost_basis
        }


# ============================================================================
# STANDALONE TESTING
# ============================================================================
if __name__ == "__main__":
    # Test the options manager
    logging.basicConfig(level=logging.INFO)

    ticker = "SPY"
    om = OptionsManager(ticker)

    # Fetch stock data
    stock_info = om.fetch_stock_data()
    print(f"\n{stock_info['name']}")
    print(f"Current Price: ${stock_info['current_price']:.2f}")

    # Get expirations
    expirations = om.get_expirations()
    print(f"\nAvailable expirations: {len(expirations)}")
    print(f"Next 5 expirations: {expirations[:5]}")

    # Fetch chain for first expiration
    if expirations:
        calls, puts = om.fetch_chain(expirations[0])
        print(f"\nOptions for {expirations[0]}:")
        print(f"Calls: {len(calls)}")
        print(f"Puts: {len(puts)}")

        # Find lotto tickets
        lottos = om.find_lotto_tickets(max_price=1.00)
        print(f"\nLotto Tickets (< $1.00):")
        print(f"Call Lottos: {len(lottos['calls'])}")
        print(f"Put Lottos: {len(lottos['puts'])}")

        if not lottos['calls'].empty:
            print("\nTop 3 Call Lottos:")
            top_calls = lottos['calls'].head(3)[['strike', 'lastPrice', 'volume', 'impliedVolatility', 'moneyness']]
            print(top_calls.to_string(index=False))
