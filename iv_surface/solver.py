import numpy as np
from scipy.stats import norm

def _compute_d1_d2(S, K, T, t, r, sigma):
        if np.any(S <= 0):
            raise ValueError(f"S must be > 0, got {S}")
        if np.any(K <= 0):
            raise ValueError(f"K must be > 0, got {K}")
        if np.any(sigma <= 0):
            raise ValueError(f"sigma must be > 0, got {sigma}")
        tau = np.maximum(T - t, 1e-12)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * (tau)) / (sigma * np.sqrt(tau))
        d2 = d1 - (sigma * np.sqrt(tau))
        return d1, d2

def bs_price(S, K, T, r, sigma, flag, t=0):
    if flag not in ("call", "put"):
            raise ValueError(f"flag must be 'call' or 'put', got '{flag}'")
    
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

def _vega_compute(S, sigma, T, K, r, t=0):
    if t >= T:
        return 0
    d1, _ = _compute_d1_d2(S, K, T, t, r, sigma)
    vega = S * norm.pdf(d1) * np.sqrt(T - t)
    return vega

def _newton_step(sigma, S, K, T, r, price, flag):
    f_sigma = bs_price(S, K, T, r, sigma, flag) - price
    f_diff_sigma = _vega_compute(S, sigma, T, K, r)
    if f_diff_sigma < 1e-8:
        return -1
    sigma_next = sigma - f_sigma / f_diff_sigma
    return sigma_next
    
def solve_iv(price, S, K, T, r, flag, sigma_low=1e-6, sigma_high=10.0):
    tolerance = 1e-4
    interval = sigma_high - sigma_low
    if np.isnan(price):
        raise ValueError(f"price must be a finite number, got {price}")
    bs_low = bs_price(S, K, T, r, sigma_low, flag)
    bs_high = bs_price(S, K, T, r, sigma_high, flag)

    if (bs_low - price) * (bs_high - price) > 0:
        raise ValueError(f"price {price} is outside no-arbitrage bounds")
    
    sigma_mid = (sigma_low + sigma_high) / 2
    while interval >= tolerance:
        sigma_nr = _newton_step(sigma_mid, S, K, T, r, price, flag)
        if sigma_nr != -1 and abs(sigma_nr - sigma_mid) < tolerance:
            return sigma_nr 
        
        if (sigma_nr >= sigma_low) and (sigma_nr <= sigma_high):
            sigma_mid = sigma_nr
        else:
            sigma_mid = (sigma_low + sigma_high) / 2

        bs_mid = bs_price(S, K, T, r, sigma_mid, flag)

        if abs(bs_mid - price) < tolerance:
            return sigma_mid
        
        if (bs_low - price) * (bs_mid - price) < 0:
            # root is in left half
            sigma_high = sigma_mid
            bs_high = bs_mid

        else:
            # root is in right half
            sigma_low = sigma_mid
            bs_low = bs_mid

        interval = sigma_high - sigma_low

    return (sigma_low + sigma_high) / 2

def build_surface(prices, S, strikes, expiries, r=0, flag="call"):
    # prices: 2D array shape (n_expiries, n_strikes)
    # returns: 2D array same shape, IV at each (expiry, strike)
    surface = np.full((len(expiries), len(strikes)), np.nan)
    for i, T in enumerate(expiries):
        for j, K in enumerate(strikes):
            try:
                surface[i, j] = solve_iv(prices[i, j], S, K, T, r, flag)
            except ValueError:
                pass
    return surface
