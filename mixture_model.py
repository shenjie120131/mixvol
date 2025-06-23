import numpy as np
from scipy.optimize import least_squares
from .bsm_model import BSMModel
from typing import Tuple, Union, List, Tuple

Array = Union[float, np.ndarray]

class MixtureLognormalModel:
    def __init__(self, S0: float, r, b, cash_dividends: List[Tuple[float, float]]):
        self.S0 = S0
        self.r = r
        self.b = b
        self.cash_dividends = cash_dividends
        self.weights = None
        self.sigmas = None

    def fit(self, K: Array, T: Array, cp: Array, market_prices: Array,
            N: int, initial_sigma: float = 0.2):
        K = np.asarray(K)
        T = np.asarray(T)
        cp = np.asarray(cp)
        market_prices = np.asarray(market_prices, dtype=float).ravel()

        def transform(params):
            logit_weights = params[:N]
            raw_sigmas = params[N:]
            weights = np.exp(logit_weights)
            weights /= np.sum(weights)
            sigmas = np.exp(raw_sigmas)
            return weights, sigmas

        def residuals(params):
            weights, sigmas = transform(params)
            comp_prices = np.stack([
                BSMModel(self.S0, self.r, self.b, self.cash_dividends).price(K, T, cp, sigma)
                for sigma in sigmas
            ], axis=-1)
            price = np.sum(comp_prices * weights, axis=-1)
            return price - market_prices

        x0 = np.concatenate([np.zeros(N), np.log(initial_sigma) * np.ones(N)])
        res = least_squares(residuals, x0, method='trf')
        self.weights, self.sigmas = transform(res.x)
        return res

    def price(self, K: Array, T: Array, cp: Array) -> np.ndarray:
        K = np.asarray(K)
        T = np.asarray(T)
        cp = np.asarray(cp)
        comp_prices = np.stack([
            BSMModel(self.S0, self.r, self.b, self.cash_dividends).price(K, T, cp, sigma)
            for sigma in self.sigmas
        ], axis=-1)
        return np.sum(comp_prices * self.weights, axis=-1)

    def vega(self, K: Array, T: Array, cp: Array) -> np.ndarray:
        K = np.asarray(K)
        T = np.asarray(T)
        cp = np.asarray(cp)
        comp_vega = np.stack([
            BSMModel(self.S0, self.r, self.b, self.cash_dividends).vega(K, T, cp, sigma)
            for sigma in self.sigmas
        ], axis=-1)
        return np.sum(comp_vega * self.weights, axis=-1)

    def delta(self, K: Array, T: Array, cp: Array) -> np.ndarray:
        K = np.asarray(K)
        T = np.asarray(T)
        cp = np.asarray(cp)
        comp_delta = np.stack([
            BSMModel(self.S0, self.r, self.b, self.cash_dividends).delta(K, T, cp, sigma)
            for sigma in self.sigmas
        ], axis=-1)
        return np.sum(comp_delta * self.weights, axis=-1)

    def gamma(self, K: Array, T: Array, cp: Array) -> np.ndarray:
        K = np.asarray(K)
        T = np.asarray(T)
        cp = np.asarray(cp)
        comp_gamma = np.stack([
            BSMModel(self.S0, self.r, self.b, self.cash_dividends).gamma(K, T, cp, sigma)
            for sigma in self.sigmas
        ], axis=-1)
        return np.sum(comp_gamma * self.weights, axis=-1)

    def theta(self, K: Array, T: Array, cp: Array) -> np.ndarray:
        K = np.asarray(K)
        T = np.asarray(T)
        cp = np.asarray(cp)
        comp_theta = np.stack([
            BSMModel(self.S0, self.r, self.b, self.cash_dividends).theta(K, T, cp, sigma)
            for sigma in self.sigmas
        ], axis=-1)
        return np.sum(comp_theta * self.weights, axis=-1)

    def vanna(self, K: Array, T: Array, cp: Array) -> np.ndarray:
        K = np.asarray(K)
        T = np.asarray(T)
        cp = np.asarray(cp)
        comp_vanna = np.stack([
            BSMModel(self.S0, self.r, self.b, self.cash_dividends).vanna(K, T, cp, sigma)
            for sigma in self.sigmas
        ], axis=-1)
        return np.sum(comp_vanna * self.weights, axis=-1)

    def volga(self, K: Array, T: Array, cp: Array) -> np.ndarray:
        K = np.asarray(K)
        T = np.asarray(T)
        cp = np.asarray(cp)
        comp_volga = np.stack([
            BSMModel(self.S0, self.r, self.b, self.cash_dividends).volga(K, T, cp, sigma)
            for sigma in self.sigmas
        ], axis=-1)
        return np.sum(comp_volga * self.weights, axis=-1)

    def fit_per_expiry(self, K: Array, T: Array, cp: Array, market_prices: Array,
                       N: int, initial_sigma: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit mixture model separately for each unique expiry in T.
        Returns:
            unique_T: array of unique expiries
            weights: array of shape (len(unique_T), N)
            sigmas: array of shape (len(unique_T), N)
        """
        K = np.asarray(K)
        T = np.asarray(T)
        cp = np.asarray(cp)
        market_prices = np.asarray(market_prices, dtype=float).ravel()

        unique_T = np.unique(T)
        all_weights = []
        all_sigmas = []
        for Ti in unique_T:
            mask = (T == Ti)
            mix = MixtureLognormalModel(self.S0, self.r, self.b, self.cash_dividends)
            mix.fit(K[mask], T[mask], cp[mask], market_prices[mask], N, initial_sigma)
            all_weights.append(mix.weights)
            all_sigmas.append(mix.sigmas)
        return unique_T, np.vstack(all_weights), np.vstack(all_sigmas)
    def implied_vol(self, K, T, cp, price, initial_sigma=0.2, tol=1e-8, max_iter=100):
        """
        Compute the equivalent single Black implied volatility matching the mixture price.
        """
        # Use BSM inversion on the mixture price
        bsm = BSMModel(self.S0, self.r, self.b, self.cash_dividends)
        return bsm.implied_vol(K, T, cp, price, tol, max_iter)

