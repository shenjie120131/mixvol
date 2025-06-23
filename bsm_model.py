from typing import Callable, Tuple, Union, List, Tuple
import numpy as np
from scipy.stats import norm
from .forward_tools import model_forward_components

Array = Union[float, np.ndarray]

def norm_pdf(x):
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

def norm_cdf(x):
    return norm.cdf(x)

class BSMModel:
    def __init__(self, S0: float, r, b, cash_dividends):
        """Initialize BSMModel, allowing scalar or callable r and b."""
        self.S0 = S0
        # wrap scalar r into a callable
        if callable(r):
            self.r = r
        else:
            self.r = lambda T: r
        # wrap scalar b into a callable
        if callable(b):
            self.b = b
        else:
            self.b = lambda T: b
        self.cash_dividends = cash_dividends
    def _adjusted_inputs(self, K: Array, T: Array):
        F_market, S_adj, K_adj, q_implied, _ = model_forward_components(
            self.S0, np.atleast_1d(T), self.r, self.b, self.cash_dividends
        )
        return S_adj, K + K_adj, F_market, q_implied

    def price(self, K: Array, T: Array, cp: Array, sigma: Array) -> Array:
        S_adj, K_adj, F, _ = self._adjusted_inputs(K, T)
        T = np.asarray(T)
        sigma = np.asarray(sigma)
        cp = np.asarray(cp)
        df = np.exp(-self.r(T) * T)
        sigma_sqrt_T = sigma * np.sqrt(T)
        d1 = (np.log(F / K_adj) + 0.5 * sigma**2 * T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T
        return df * cp * (F * norm_cdf(cp * d1) - K_adj * norm_cdf(cp * d2))

    def implied_vol(self, K: Array, T: Array, cp: Array, price: Array, tol: float = 1e-6, max_iter: int = 100) -> Array:
        K = np.asarray(K)
        T = np.asarray(T)
        cp = np.asarray(cp)
        price = np.asarray(price)
        sigma = np.full_like(K, 0.2, dtype=np.float64)
        for _ in range(max_iter):
            price_est = self.price(K, T, cp, sigma)
            vega_est = self.vega(K, T, cp, sigma)
            update = (price_est - price) / np.maximum(vega_est, 1e-8)
            sigma -= update
            if np.all(np.abs(update) < tol):
                break
        sigma[sigma <= 0] = np.nan
        return sigma

    def vega(self, K: Array, T: Array, cp: Array, sigma: Array) -> Array:
        S_adj, K_adj, F, _ = self._adjusted_inputs(K, T)
        T = np.asarray(T)
        d1 = (np.log(F / K_adj) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        df = np.exp(-self.r(T) * T)
        return S_adj * df * norm_pdf(d1) * np.sqrt(T)

    def delta(self, K: Array, T: Array, cp: Array, sigma: Array) -> Array:
        S_adj, K_adj, F, _ = self._adjusted_inputs(K, T)
        T = np.asarray(T)
        sigma = np.asarray(sigma)
        cp = np.asarray(cp)
        d1 = (np.log(F / K_adj) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        df = np.exp(-self.r(T) * T)
        return cp * df * norm_cdf(cp * d1) * F / S_adj

    def gamma(self, K: Array, T: Array, cp: Array, sigma: Array) -> Array:
        S_adj, K_adj, F, _ = self._adjusted_inputs(K, T)
        T = np.asarray(T)
        sigma = np.asarray(sigma)
        d1 = (np.log(F / K_adj) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        df = np.exp(-self.r(T) * T)
        return df * norm_pdf(d1) / (S_adj * sigma * np.sqrt(T))

    def theta(self, K: Array, T: Array, cp: Array, sigma: Array) -> Array:
        S_adj, K_adj, F, _ = self._adjusted_inputs(K, T)
        T = np.asarray(T)
        rT = self.r(T)
        sigma_sqrt_T = sigma * np.sqrt(T)
        d1 = (np.log(F / K_adj) + 0.5 * sigma**2 * T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T
        df = np.exp(-rT * T)
        term1 = -S_adj * norm_pdf(d1) * sigma / (2 * np.sqrt(T))
        term2 = -cp * rT * K_adj * df * norm_cdf(cp * d2)
        return term1 + term2

    def vanna(self, K: Array, T: Array, cp: Array, sigma: Array) -> Array:
        S_adj, K_adj, F, _ = self._adjusted_inputs(K, T)
        T = np.asarray(T)
        sigma = np.asarray(sigma)
        d1 = (np.log(F / K_adj) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        df = np.exp(-self.r(T) * T)
        return -df * norm_pdf(d1) * d2 / sigma

    def volga(self, K: Array, T: Array, cp: Array, sigma: Array) -> Array:
        S_adj, K_adj, F, _ = self._adjusted_inputs(K, T)
        T = np.asarray(T)
        sigma = np.asarray(sigma)
        d1 = (np.log(F / K_adj) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        df = np.exp(-self.r(T) * T)
        return df * norm_pdf(d1) * d1 * d2 / sigma