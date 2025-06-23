
import numpy as np
from mixvol.forward_tools import model_forward_components

class VNTree:
    """Minimal VNTree for testing: includes price and theta only."""

    def __init__(self, S0, K, T, r_curve, b_curve, cash_divs, sigma, steps=51, cp=1):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r_curve = r_curve
        self.b_curve = b_curve
        self.cash_divs = cash_divs
        self.sigma = sigma
        self.steps = steps if steps % 2 == 1 else steps + 1
        self.cp = cp
        _, S_adj_arr, K_off_arr, q_arr, _ = model_forward_components(
            self.S0, np.array([self.T]), self.r_curve, self.b_curve, self.cash_divs
        )
        self.S_adj = S_adj_arr[0]
        self.K_off = K_off_arr[0]
        self.q = q_arr[0]

    def price(self, S0=None, sigma=None, T=None):
        S0_eff = self.S0 if S0 is None else S0
        sigma_eff = self.sigma if sigma is None else sigma
        T_val = self.T if T is None else T

        _, S_adj_arr, K_off_arr, q_arr, _ = model_forward_components(
            S0_eff, np.array([T_val]), self.r_curve, self.b_curve, self.cash_divs
        )
        S_adj = S_adj_arr[0]
        K_off = K_off_arr[0]
        q = q_arr[0]

        dt = T_val / self.steps
        u = np.exp(sigma_eff * np.sqrt(dt))
        d = 1 / u
        drift = np.exp((self.r_curve(T_val) - q) * dt)
        p = (drift - d) / (u - d)

        prices = np.array([S_adj * (u**i) * (d**(self.steps - i)) for i in range(self.steps + 1)])
        payoffs = np.maximum(self.cp * (prices - (self.K + K_off)), 0)

        disc = np.exp(-self.r_curve(T_val) * dt)
        for n in range(self.steps, 0, -1):
            payoffs = disc * (p * payoffs[1:n+1] + (1 - p) * payoffs[0:n])
        return payoffs[0]

    def theta(self, eps=None):
        h = eps if eps is not None else 1/365.0
        cd_arr = np.asarray(self.cash_divs)
        if cd_arr.ndim == 1 or len(cd_arr) == 0:
            cd_trunc = []
        else:
            cd_trunc = [tuple(row) for row in cd_arr if row[0] < self.T]
        tree_nd = VNTree(self.S0, self.K, self.T, self.r_curve, self.b_curve,
                         cd_trunc, self.sigma, steps=self.steps, cp=self.cp)
        f1 = tree_nd.price(T=self.T + 2*h)
        f2 = tree_nd.price(T=self.T + h)
        f3 = tree_nd.price(T=self.T - h)
        f4 = tree_nd.price(T=self.T - 2*h)
        return -(-f1 + 8*f2 - 8*f3 + f4) / (12 * h)

    def delta(self, eps=None):
        h = eps if eps is not None else 1e-3 * self.S0
        f1 = self.price(S0=self.S0 + 2*h)
        f2 = self.price(S0=self.S0 + h)
        f3 = self.price(S0=self.S0 - h)
        f4 = self.price(S0=self.S0 - 2*h)
        return (-f1 + 8*f2 - 8*f3 + f4) / (12 * h)

    def gamma(self, eps=None):
        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        h = self.S0 * (u - d) if eps is None else eps
        f_up = self.price(S0=self.S0 + h)
        f0 = self.price()
        f_dn = self.price(S0=self.S0 - h)
        return (f_up - 2 * f0 + f_dn) / (h * h)

    def vega(self, eps=None):
        h = eps if eps is not None else 1e-4
        f1 = self.price(sigma=self.sigma + 2*h)
        f2 = self.price(sigma=self.sigma + h)
        f0 = self.price()
        f3 = self.price(sigma=self.sigma - h)
        f4 = self.price(sigma=self.sigma - 2*h)
        return (-f1 + 8*f2 - 8*f3 + f4) / (12 * h)

    def vanna(self, eps_s=None, eps_v=None):
        hs = eps_s if eps_s is not None else 1e-4 * self.S0
        hv = eps_v if eps_v is not None else 1e-4
        f_pp = self.price(S0=self.S0 + hs, sigma=self.sigma + hv)
        f_pm = self.price(S0=self.S0 + hs, sigma=self.sigma - hv)
        f_mp = self.price(S0=self.S0 - hs, sigma=self.sigma + hv)
        f_mm = self.price(S0=self.S0 - hs, sigma=self.sigma - hv)
        return (f_pp - f_pm - f_mp + f_mm) / (4 * hs * hv)

    def volga(self, eps=None):
        h = eps if eps is not None else 1e-3
        P0 = self.price()
        tree_plus = VNTree(
            S0=self.S0, K=self.K, T=self.T,
            r_curve=self.r_curve, b_curve=self.b_curve,
            cash_divs=self.cash_divs, sigma=self.sigma + h,
            steps=self.steps, cp=self.cp
        )
        P_plus = tree_plus.price()
        tree_minus = VNTree(
            S0=self.S0, K=self.K, T=self.T,
            r_curve=self.r_curve, b_curve=self.b_curve,
            cash_divs=self.cash_divs, sigma=self.sigma - h,
            steps=self.steps, cp=self.cp
        )
        P_minus = tree_minus.price()
        return (P_plus + P_minus - 2 * P0) / (h ** 2)
