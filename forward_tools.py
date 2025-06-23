
import numpy as np
from typing import Callable, List, Tuple, Union

Array = Union[float, np.ndarray]

def discount_cashflows(cash_dividends: List[Tuple[float, float]],
                       r: Callable[[Array], Array], b: Callable[[Array], Array]) -> float:
    """Compute present value of cashflows using (r - b) as discount rate."""
    cash_dividends = np.asarray(cash_dividends)
    if cash_dividends.ndim == 1 or len(cash_dividends) == 0:
        return 0.0
    times = cash_dividends[:, 0]
    amounts = cash_dividends[:, 1]
    discounts = np.exp(-(r(times) - b(times)) * times)
    return np.sum(amounts * discounts)

def model_forward_components(S0: float, T: np.ndarray,
                             r: Callable[[Array], Array],
                             b: Callable[[Array], Array],
                             cash_dividends: List[Tuple[float, float]]
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute arbitrage-free forward (F_market), implied q(T), and time-weighted adjusted S, K.
    Returns:
        F_market: np.ndarray of market forward for each T
        S_adj:     np.ndarray of adjusted spot
        K_adj:     np.ndarray of strike adjustment (to be added to strike)
        q_implied: np.ndarray of implied continuous dividend yields
        PV_divs:   np.ndarray of PV of discrete dividends
    """
    cash_dividends = np.asarray(cash_dividends)
    if cash_dividends.ndim == 1 or len(cash_dividends) == 0:
        times = np.array([])
        amounts = np.array([])
    else:
        times = cash_dividends[:, 0]
        amounts = cash_dividends[:, 1]
    T = np.atleast_1d(T).astype(float)
    PV_divs = np.array([
        discount_cashflows([(t, a) for t, a in zip(times, amounts) if t <= Ti], r, b)
        for Ti in T
    ])
    F_market = (S0 - PV_divs) * np.exp((r(T) - b(T)) * T)
    T_max = np.max(T) if np.max(T) > 0 else 1.0
    weight = T / T_max
    S_adj = S0 - (1 - weight) * PV_divs
    K_adj = weight * PV_divs
    q_implied = r(T) - np.log(F_market / S_adj) / T
    return F_market, S_adj, K_adj, q_implied, PV_divs
