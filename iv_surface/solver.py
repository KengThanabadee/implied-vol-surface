import numpy as np
from scipy.stats import norm

def _compute_d1_d2(S, K, T, t, r, sigma):
        if np.any(S <= 0):
            raise ValueError(f"S must be > 0, got {S}")
        if np.any(sigma <= 0):
            raise ValueError(f"sigma must be > 0, got {sigma}")
        tau = np.maximum(T - t, 1e-12)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * (tau)) / (sigma * np.sqrt(tau))
        d2 = d1 - (sigma * np.sqrt(tau))
        return d1, d2

def bs_price(S, K, T, r, sigma, flag, t=0):
    if t >= T:                                                          
        if flag == "call" : 
            return np.maximum(S - K, 0)                                   
        if flag == "put":
            return np.maximum(K - S, 0)  
        
    d1, d2 = _compute_d1_d2(S, K, T, t, r, sigma)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    discount = np.exp(-r * (T - t))
    C = S * N_d1 - K * discount * N_d2
    P = C + K * discount - S # put-call parity

    if flag == "call" :
        return C
    if flag == "put":
        return P