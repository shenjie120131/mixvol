# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True

"""
Cythonized VNTree V2 with inline forward component logic and full Greeks.
Translated from binomial_tree_v2.py, removing external calls.
"""
import numpy as np
cimport numpy as np
from libc.math cimport exp, log

# Inline forward component computation
def _compute_forward(double S0, double[:] times,
                     double[:] div_times, double[:] div_amounts,
                     object r_func, object b_func,
                     int steps,
                     np.ndarray[np.double_t, ndim=1] PV_divs,
                     np.ndarray[np.double_t, ndim=1] F_market,
                     np.ndarray[np.double_t, ndim=1] S_adj,
                     np.ndarray[np.double_t, ndim=1] K_adj,
                     np.ndarray[np.double_t, ndim=1] q_implied):
    cdef int n = steps + 1
    cdef double dt = times[0] - times[1]
    cdef int i, j
    cdef int m = div_times.shape[0]
    # PV of dividends
    for i in range(n):
        cdef double Ti = times[i]
        cdef double pv = 0.0
        for j in range(m):
            cdef double t = div_times[j]
            if t <= Ti:
                pv += div_amounts[j] * exp(-(r_func(t) - b_func(t)) * t)
        PV_divs[i] = pv
    # Forward market and adj
    for i in range(n):
        cdef double Ti = times[i]
        cdef double rr = r_func(Ti) - b_func(Ti)
        F_market[i] = (S0 - PV_divs[i]) * exp(rr * Ti)
        cdef double weight = Ti / (times[0] if times[0] > 0 else 1.0)
        S_adj[i] = S0 - (1-weight) * PV_divs[i]
        K_adj[i] = weight * PV_divs[i]
        q_implied[i] = r_func(Ti) - log(F_market[i] / S_adj[i]) / Ti if Ti>0 else q_implied[1]

cdef double compute_price_c(double S0, double K, double[:] times,
                            double[:] div_times, double[:] div_amounts,
                            object r_func, object b_func,
                            double sigma, int steps, int cp):
    cdef int n = steps + 1
    # allocate arrays
    cdef np.ndarray[np.double_t, ndim=1] PV_divs   = np.empty(n, dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] F_market  = np.empty(n, dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] S_adj     = np.empty(n, dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] K_adj     = np.empty(n, dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] q_implied = np.empty(n, dtype=np.double)
    # forward
    _compute_forward(S0, times, div_times, div_amounts, r_func, b_func,
                     steps, PV_divs, F_market, S_adj, K_adj, q_implied)
    # discount & growth
    cdef double dt = times[0] - times[1]
    cdef double r0 = r_func(times[0])
    cdef double discount = exp(-r0 * dt)
    cdef np.ndarray[np.double_t, ndim=1] growth = np.empty(n, dtype=np.double)
    for i in range(n): growth[i] = exp((r0 - q_implied[i]) * dt)
    # payoff
    cdef np.ndarray[np.double_t, ndim=1] asset_prices = np.empty(n, dtype=np.double)
    for i in range(n): asset_prices[i] = S_adj[i] * (growth[n-1] ** steps)
    cdef np.ndarray[np.double_t, ndim=1] option_vals = np.maximum(cp * (asset_prices - K_adj), 0)
    # rollback
    for i in range(steps):
        for j in range(n-1):
            option_vals[j] = 0.5 * (option_vals[j] + option_vals[j+1]) * discount * growth[i]
    return option_vals[0]

cdef class VNTree:
    """Cython VNTree V2 wrapper with inline forward logic and Greeks"""
    cdef double S0, K, sigma
    cdef object r, b
    cdef double[:] div_times, div_amounts, times
    cdef int steps, cp

    def __init__(self, double S0, double K, double T,
                 object r, object b,
                 list cash_dividends,
                 double sigma, int steps=100, int cp=1):
        self.S0, self.K, self.r, self.b = S0, K, r, b
        self.sigma, self.steps, self.cp = sigma, steps, cp
        self.times = np.linspace(T, 0, steps+1)
        cdef int m = len(cash_dividends)
        self.div_times   = np.empty(m, dtype=np.double)
        self.div_amounts = np.empty(m, dtype=np.double)
        cdef int i
        for i in range(m): self.div_times[i], self.div_amounts[i] = cash_dividends[i]

    def price(self):
        cdef double[:] tv = self.times
        return compute_price_c(self.S0, self.K, tv,
                               self.div_times, self.div_amounts,
                               self.r, self.b,
                               self.sigma, self.steps, self.cp)

    def delta(self, double h=1e-4):
        cdef double[:] tv = self.times
        return (compute_price_c(self.S0*(1+h), self.K, tv,
            self.div_times, self.div_amounts, self.r, self.b,
            self.sigma, self.steps, self.cp)
            - compute_price_c(self.S0*(1-h), self.K, tv,
            self.div_times, self.div_amounts, self.r, self.b,
            self.sigma, self.steps, self.cp)) / (2 * self.S0 * h)

    def gamma(self, double h=1e-4):
        cdef double[:] tv = self.times
        cdef double up   = compute_price_c(self.S0*(1+h), self.K, tv,
            self.div_times, self.div_amounts, self.r, self.b,
            self.sigma, self.steps, self.cp)
        cdef double mid  = compute_price_c(self.S0, self.K, tv,
            self.div_times, self.div_amounts, self.r, self.b,
            self.sigma, self.steps, self.cp)
        cdef double down = compute_price_c(self.S0*(1-h), self.K, tv,
            self.div_times, self.div_amounts, self.r, self.b,
            self.sigma, self.steps, self.cp)
        return (up - 2*mid + down) / (self.S0**2 * h**2)

    def theta(self, double dt=1/365):
        cdef double[:] tv = self.times
        cdef double now = compute_price_c(self.S0, self.K, tv,
            self.div_times, self.div_amounts, self.r, self.b,
            self.sigma, self.steps, self.cp)
        # rebuild times for T-dt
        cdef np.ndarray[np.double_t, ndim=1] times2 = np.linspace(
            tv[0]-dt, 0, self.steps+1)
        cdef double fut = compute_price_c(self.S0, self.K, times2,
            self.div_times, self.div_amounts, self.r, self.b,
            self.sigma, self.steps, self.cp)
        return (fut - now) / dt

    def vega(self, double dv=1e-4):
        cdef double[:] tv = self.times
        return (compute_price_c(self.S0, self.K, tv,
            self.div_times, self.div_amounts, self.r, self.b,
            self.sigma+dv, self.steps, self.cp)
            - compute_price_c(self.S0, self.K, tv,
            self.div_times, self.div_amounts, self.r, self.b,
            self.sigma-dv, self.steps, self.cp)) / (2*dv)

    def vanna(self, double h=1e-4, double dv=1e-4):
        cdef double[:] tv = self.times
        cdef double up   = compute_price_c(self.S0*(1+h), self.K, tv,
            self.div_times, self.div_amounts, self.r, self.b,
            self.sigma+dv, self.steps, self.cp)
        cdef double down = compute_price_c(self.S0*(1-h), self.K, tv,
            self.div_times, self.div_amounts, self.r, self.b,
            self.sigma-dv, self.steps, self.cp)
        return (up - down) / (4 * self.S0 * h * dv)

    def volga(self, double dv=1e-4):
        cdef double[:] tv = self.times
        cdef double up   = compute_price_c(self.S0, self.K, tv,
            self.div_times, self.div_amounts, self.r, self.b,
            self.sigma+dv, self.steps, self.cp)
        cdef double mid  = compute_price_c(self.S0, self.K, tv,
            self.div_times, self.div_amounts, self.r, self.b,
            self.sigma, self.steps, self.cp)
        cdef double down = compute_price_c(self.S0, self.K, tv,
            self.div_times, self.div_amounts, self.r, self.b,
            self.sigma-dv, self.steps, self.cp)
        return (up - 2*mid + down) / (dv*dv)
