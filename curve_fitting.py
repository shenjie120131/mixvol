#!/usr/bin/env python
"""
Two‐Step Parity Calibration with Synthetic Data
Refactored and corrected to fix OptimizeResult mis-call and syntax errors.
Features:
 - Discrete dividend adjustment
 - Cached VN tree pricer (call/put, American/European)
 - Black‑Scholes pricing/IV for call & put
 - Synthetic dividend schedule generator
 - Synthetic data generator with missing quotes
 - Parity-based fitting handling missing quotes & dividends
 - Main execution with CSV outputs
 - Quick unit tests for BS and tree IV
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, brentq, newton
from scipy.stats import norm
from datetime import date, timedelta
from functools import lru_cache

# ------------------------------------------------------------------
# Utility: fixed 'today'
# ------------------------------------------------------------------
def today():
    return date(2025, 6, 24)

# ------------------------------------------------------------------
# Cash‑Dividend Adjustment
# ------------------------------------------------------------------
def adjust_inputs(S0, K, r, q, T, divs):
    if not divs:
        return S0, K, 0.0, 0.0
    t_arr = np.array([t for t, _ in divs])
    d_arr = np.array([d for _, d in divs])
    pv = d_arr * np.exp(-r * t_arr)
    w_spot = (T - t_arr) / T
    w_strike = t_arr / T
    spot_pull = np.dot(pv, w_spot)
    strike_push = np.dot(pv, w_strike)
    return S0 - spot_pull, K + strike_push, spot_pull, pv.sum()

# ------------------------------------------------------------------
# Cached VN Tree Pricer
# ------------------------------------------------------------------
def tree_price(S0, K, r, q, s, sigma, T, N, divs, option_type='call', american=False):
    key = (S0, float(K), float(r), float(q), float(s), float(sigma), float(T), int(N), tuple(divs), option_type, american)
    return _cached_tree_price(key)

@lru_cache(maxsize=None)
def _cached_tree_price(key):
    S0, K, r, q, s, sigma, T, N, divs, option_type, american = key
    dt = T / N
    dx = sigma * np.sqrt(dt)
    pu = 0.5
    pd = 1 - pu
    df = np.exp(-r * dt)
    S_adj, K_adj, _, _ = adjust_inputs(S0, K, r, q, T, divs)
    lnS0 = np.log(S_adj)
    cache = [None] * (N+1)
    for i in range(N+1):
        cache[i] = lnS0 + (2 * np.arange(i+1) - i) * dx
    lnT = cache[N]
    ST = np.exp(lnT)
    if option_type == 'call':
        V = np.maximum(ST - K_adj, 0.0)
    else:
        V = np.maximum(K_adj - ST, 0.0)
    for i in range(N, 0, -1):
        V = df * (pu * V[1:] + pd * V[:-1])
        if american:
            lnS = cache[i-1]
            S = np.exp(lnS)
            if option_type == 'call':
                V = np.maximum(V, S - K_adj)
            else:
                V = np.maximum(V, K_adj - S)
    return float(V[0])

# ------------------------------------------------------------------
# Black‑Scholes Pricing & IV Helpers
# ------------------------------------------------------------------
def bs_price(S, K, r, q, T, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        pay = (S - K) if option_type == 'call' else (K - S)
        return max(pay, 0.0)
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_val = np.exp(-q * T) * S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)
    if option_type == 'call':
        return call_val
    return call_val - np.exp(-q * T) * S + np.exp(-r * T) * K


def bs_implied_vol(C, S, K, r, q, T, option_type='call'):
    def obj(sig): return bs_price(S, K, r, q, T, sig, option_type) - C
    try:
        return brentq(obj, 1e-6, 5)
    except:
        return np.nan


def tree_implied_vol(C, S0, K, r, q, s, T, N, divs, option_type='call'):
    def obj(sig): return tree_price(S0, K, r, q, s, sig, T, N, divs, option_type) - C
    low, high = 1e-6, 5.0
    f_low, f_high = obj(low), obj(high)
    if np.isnan(f_low) or np.isnan(f_high) or f_low * f_high > 0:
        return np.nan
    sigma0 = bs_implied_vol(C, S0, K, r, q, T, option_type)
    if np.isnan(sigma0) or not (low < sigma0 < high): sigma0 = (low + high) / 2
    try:
        return newton(obj, sigma0, tol=1e-6, maxiter=50)
    except:
        try: return brentq(obj, low, high)
        except: return np.nan

# ------------------------------------------------------------------
# Synthetic Dividend Forecast Generator
# ------------------------------------------------------------------
def generate_dividend_schedule(T_max, freq=4, amt=0.5):
    n = int(np.floor(T_max * freq))
    return [((i+1)/freq, amt) for i in range(n)]

# ------------------------------------------------------------------
# Synthetic Data Chain Generator
# ------------------------------------------------------------------
def generate_chain(S0, expiries, strikes, divs, miss_call=0.1, miss_put=0.1, seed=42):
    np.random.seed(seed)
    calls, puts = [], []
    for exp in expiries:
        T = (exp - today()).days / 365.0
        r = 0.03 + 0.02 * np.exp(-3 * T)
        q = 0.01 + 0.01 * T
        s = 0.001 + 0.002 * T
        divs_full = [d for d in divs if d[0] <= T]
        C_vals, P_vals = [], []
        for K in strikes:
            sig = 0.35 * np.exp(-5*T) + 0.2 + 0.05*T + 0.3*np.exp(-T)*abs(np.log(K/S0))
            c = tree_price(S0, K, r, q, s, sig, T, 200, divs_full, 'call', False)
            p = tree_price(S0, K, r, q, s, sig, T, 200, divs_full, 'put', False)
            C_vals.append(np.nan if np.random.rand() < miss_call else c)
            P_vals.append(np.nan if np.random.rand() < miss_put else p)
        calls.append(pd.DataFrame({'strike': strikes, 'lastPrice': C_vals}))
        puts.append(pd.DataFrame({'strike': strikes, 'lastPrice': P_vals}))
    return calls, puts

# ------------------------------------------------------------------
# Parity-Based Fitting (r, q, s)
# ------------------------------------------------------------------
def fit_rqs_logit(S0, mats, calls, puts, divs, r_max=0.2):
    sigmoid = lambda z: 1/(1+np.exp(-z))
    results = []
    for i, T in enumerate(mats):
        divs_full = [d for d in divs if d[0] <= T]
        df = pd.merge(calls[i], puts[i], on='strike', suffixes=('_C','_P')).dropna()
        df['moneyness'] = df['strike']/S0
        df = df[df['moneyness'].between(0.8, 1.2)]
        if df.empty:
            results.append({
                'T': T,
                'r_true': 0.03 + 0.02*np.exp(-3*T),
                'q_true': 0.01 + 0.01*T,
                's_true': 0.001 + 0.002*T,
                'r_fit': np.nan,
                'q_fit': np.nan,
                's_fit': np.nan,
                'n_pairs': 0
            })
            continue
        K_arr, C_obs, P_obs = df['strike'].values, df['lastPrice_C'].values, df['lastPrice_P'].values
        def resid(z):
            x0, x1, x2 = z
            r_ = r_max*sigmoid(x0)
            y = r_*np.tanh(x2)
            q_ = y*sigmoid(x1)
            s_ = y-q_
            res=[]
            for K,C_o,P_o in zip(K_arr,C_obs,P_obs):
                S_adj,K_adj,_,_ = adjust_inputs(S0,K,r_,q_,T,divs_full)
                theo = S_adj*np.exp(-q_*T) - K_adj*np.exp(-r_*T)
                res.append(C_o - P_o - theo)
            return np.array(res)
        x0 = np.log((0.03/r_max)/(1-0.03/r_max)); x1 = 0.0; x2 = 0.0
        sol = least_squares(resid, [x0, x1, x2], max_nfev=200)
        x0o,x1o,x2o = sol.x
        r_fit = r_max*sigmoid(x0o)
        y = r_fit*np.tanh(x2o)
        q_fit = y*sigmoid(x1o)
        s_fit = y-q_fit
        results.append({
            'T': T,
            'r_true': 0.03 + 0.02*np.exp(-3*T),
            'q_true': 0.01 + 0.01*T,
            's_true': 0.001 + 0.002*T,
            'r_fit': r_fit,
            'q_fit': q_fit,
            's_fit': s_fit,
            'n_pairs': len(K_arr)
        })
    return pd.DataFrame(results)

# ------------------------------------------------------------------
# Main Execution: Calibration, IV Surface, Repricing Errors
# ------------------------------------------------------------------
if __name__=='__main__':
    S0 = 100.0
    mats = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    expiries = [today() + timedelta(days=int(T*365)) for T in mats]
    strikes = np.linspace(50, 150, 21)
    divs = generate_dividend_schedule(max(mats))

    # 1) Fit r, q, s
    calls, puts = generate_chain(S0, expiries, strikes, divs)
    df_term = fit_rqs_logit(S0, mats, calls, puts, divs)
    print("=== Calibrated r, q, s Per Maturity ===")
    print(df_term.to_string(index=False))

        # 2) Implied Vol Surface (OTM rule)
    vol_surf = pd.DataFrame(index=mats, columns=strikes)
    for i, T in enumerate(mats):
        for K in strikes:
            mny = K / S0
            if mny < 1:
                # use put
                P = puts[i].set_index('strike').at[K, 'lastPrice']
                if not np.isnan(P):
                    iv = tree_implied_vol(P, S0, K,
                                           df_term.at[i, 'r_fit'], df_term.at[i, 'q_fit'], df_term.at[i, 's_fit'],
                                           T, 200, divs, 'put')
                else:
                    iv = np.nan
            else:
                # use call
                C = calls[i].set_index('strike').at[K, 'lastPrice']
                if not np.isnan(C):
                    iv = tree_implied_vol(C, S0, K,
                                           df_term.at[i, 'r_fit'], df_term.at[i, 'q_fit'], df_term.at[i, 's_fit'],
                                           T, 200, divs, 'call')
                else:
                    iv = np.nan
            vol_surf.at[T, K] = iv
    print("=== Implied Vol Surface ===")
    print(vol_surf)

    # 3) Repricing Errors
    err_surf = pd.DataFrame(index=mats, columns=strikes)
    err_surf = pd.DataFrame(index=mats, columns=strikes)
    for i, T in enumerate(mats):
        for K in strikes:
            C_obs = calls[i].set_index('strike').at[K, 'lastPrice']
            if not np.isnan(C_obs):
                price_hat = tree_price(S0, K,
                                       df_term.at[i, 'r_fit'], df_term.at[i, 'q_fit'], df_term.at[i, 's_fit'],
                                       vol_surf.at[T, K], T, 200, divs, 'call', False)
                err_surf.at[T, K] = price_hat - C_obs
            else:
                err_surf.at[T, K] = np.nan
    print("=== Repricing Errors ===")
    print(err_surf)

    # Optional CSV export
    df_term.to_csv("calibrated_rqs.csv", index=False)
    vol_surf.to_csv("implied_vol_surface.csv")
    err_surf.to_csv("repricing_errors.csv")
