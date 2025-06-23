import numpy as np
from mixvol.forward_tools import model_forward_components

class VNTree:
    """
    Vectorized Leisen-Reimer binomial tree with time-varying forward components.
    Computes price and Greeks via finite-difference on time-varying grid.
    """
    def __init__(self, S0, K, T, r, b, cash_dividends, sigma, steps=100, cp=1):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.b = b
        self.cash_dividends = cash_dividends
        self.sigma = sigma
        self.steps = steps
        self.cp = cp

    def price(self, S0=None, T=None, sigma=None):
        """
        Compute option price on a time-varying grid. If S0, T, or sigma are provided, they override instance values.
        """
        S0 = self.S0 if S0 is None else S0
        T = self.T if T is None else T
        sigma = self.sigma if sigma is None else sigma
        # Build time grid
        times = np.linspace(T, 0, self.steps + 1)
        dt = times[0] - times[1]
        # Forward components
        S_adj, K_adj, F_market = model_forward_components(
            S0, self.K, times, self.r, self.b, self.cash_dividends
        )
        # Implied dividend yield
        times_safe = np.where(times > 0, times, times[1])
        q_implied = self.r(T) - np.log(F_market / S_adj) / times_safe
        # Discount & growth
        discount = np.exp(-self.r(T) * dt)
        growth = np.exp((self.r(T) - q_implied) * dt)
        # Asset grid at maturity
        asset_prices = S_adj * growth**self.steps
        # Payoff
        option_values = np.maximum(self.cp * (asset_prices - K_adj), 0)
        # Rollback
        for i in range(self.steps):
            option_values = (
                0.5 * (option_values[:-1] + option_values[1:])
                * discount * growth[i]
            )
        return option_values[0]

    def delta(self, h=1e-4):
        P_up = self.price(S0=self.S0*(1+h))
        P_down = self.price(S0=self.S0*(1-h))
        return (P_up - P_down) / (2 * self.S0 * h)

    def gamma(self, h=1e-4):
        P_up = self.price(S0=self.S0*(1+h))
        P_mid = self.price()
        P_down = self.price(S0=self.S0*(1-h))
        return (P_up - 2*P_mid + P_down) / (self.S0**2 * h**2)

    def theta(self, dt=1/365):
        P_now = self.price()
        P_future = self.price(T=self.T - dt)
        return (P_future - P_now) / dt

    def vega(self, dv=1e-4):
        P_up = self.price(sigma=self.sigma + dv)
        P_down = self.price(sigma=self.sigma - dv)
        return (P_up - P_down) / (2 * dv)

    def vanna(self, h=1e-4, dv=1e-4):
        P_up = self.price(S0=self.S0*(1+h), sigma=self.sigma + dv)
        P_down = self.price(S0=self.S0*(1-h), sigma=self.sigma - dv)
        return (P_up - P_down) / (4 * self.S0 * h * dv)

    def volga(self, dv=1e-4):
        P_up = self.price(sigma=self.sigma + dv)
        P_mid = self.price()
        P_down = self.price(sigma=self.sigma - dv)
        return (P_up - 2*P_mid + P_down) / (dv**2)
