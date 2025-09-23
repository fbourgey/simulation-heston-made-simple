import numpy as np
from scipy.integrate import quad_vec

import numpy as np
from scipy import stats

# Module-level constants
IMPVOL_MIN = 1e-10
IMPVOL_MAX = 5.0


def black_price(K, T, F, vol, opttype: float | np.ndarray = 1.0):
    """
    Calculate the Black option price.

    Parameters
    ----------
    K : float
        Strike price of the option.
    T : float
        Time to maturity of the option.
    F : float
        Forward price of the underlying asset.
    vol : float
        Volatility of the underlying asset.
    opttype : float or np.ndarray, optional
        Option type: 1 for call options, -1 for put options. Default is 1.

    Returns
    -------
    float
        The Black price of the option.
    """
    s = vol * T**0.5
    d1 = np.log(F / K) / s + 0.5 * s
    d2 = d1 - s
    price = opttype * (
        F * stats.norm.cdf(opttype * d1) - K * stats.norm.cdf(opttype * d2)
    )
    return price


def black_impvol(
    K, T, F, value, opttype: int | np.ndarray = 1, TOL=1e-5, MAX_ITER=1000
):
    """
    Calculate the Black implied volatility using a bisection method.

    Parameters
    ----------
    K : ndarray or float
        Strike price(s) of the option(s).
    T : float
        Time to maturity of the option(s).
    F : float
        Forward price of the underlying asset.
    value : ndarray or float
        Observed market price(s) of the option(s).
    opttype : int or ndarray, optional
        Option type: 1 for call options, -1 for put options. Default is 1.
    TOL : float, optional
        Tolerance for convergence of the implied volatility. Default is 1e-6.
    MAX_ITER : int, optional
        Maximum number of iterations for the bisection method. Default is 1000.

    Returns
    -------
    ndarray or float
        Implied volatility(ies) corresponding to the input option prices. If the
        input arrays are multidimensional, the output will have the same shape.
        Returns NaN if the implied volatility does not converge or if invalid
        inputs are provided.

    Raises
    ------
    ValueError
        If `K` and `value` do not have the same shape.
        If `opttype` is not 1 or -1.
        If the implied volatility does not converge within `MAX_ITER` iterations.
    """
    K = np.atleast_1d(K)
    value = np.atleast_1d(value)
    opttype = np.full_like(K, opttype)

    if K.shape != value.shape:
        raise ValueError("K and value must have the same shape.")

    # Fix: check all opttype values
    if not np.all(np.abs(opttype) == 1):
        raise ValueError("opttype must be either 1 or -1.")

    F = float(F)
    T = float(T)

    if T <= 0 or F <= 0:
        return np.full_like(K, np.nan)

    low = IMPVOL_MIN * np.ones_like(K)
    high = IMPVOL_MAX * np.ones_like(K)
    mid = 0.5 * (low + high)
    for _ in range(MAX_ITER):
        price = black_price(K, T, F, mid, opttype)
        diff = (price - value) / value

        if np.all(np.abs(diff) < TOL):
            return mid

        mask = diff > 0
        high[mask] = mid[mask]
        low[~mask] = mid[~mask]
        mid = 0.5 * (low + high)

    # raise ValueError("Implied volatility did not converge.")
    print("Implied volatility did not converge for all log(K/F) values.")

    # Set mid to NaN where the tolerance is not met
    mid = np.where(np.abs(diff) < TOL, mid, np.nan)
    return mid


def lewis_formula_otm_price(phi, k, tau):
    """
    Compute the OTM (Out-of-The-Money) option price using the Lewis formula.

    The Lewis formula is used to price European options by applying Fourier transform
    methods. It calculates the price of OTM options given the characteristic function
    of the log price.

    Parameters
    ----------
    phi : callable
        The characteristic function of the log price process.
        Should take two arguments: complex number u and time to maturity tau.
    k : float or array_like
        Log strike price(s). k = log(K/S) where K is strike and S is spot price.
    tau : float or array_like
        Time(s) to maturity in years.

    Returns
    -------
    ndarray
        OTM option prices. For k < 0, returns put prices; for k >= 0, returns
        call prices.

    Notes
    -----
    The formula uses the following representation:
    For k >= 0 (calls): C(k,T) = 1/π ∫[0,∞] Re[e^(-iuk)φ(u-i/2,τ)/(u^2+1/4)]du
    For k < 0 (puts): P(k,T) = e^k - 1/π ∫[0,∞] Re[e^(-iuk)φ(u-i/2,τ)/(u^2+1/4)]du

    References
    ----------
    Lewis, A. L. (2000). Option Valuation under Stochastic Volatility.
    """
    k = np.atleast_1d(np.asarray(k))
    tau = np.atleast_1d(np.asarray(tau))

    def integrand(u):
        return np.real(np.exp(-1j * u * k) * phi(u - 1j / 2, tau) / (u**2 + 1 / 4))

    k_minus = k * (k < 0)

    integral, _ = quad_vec(integrand, 0, np.inf, epsrel=1e-10, limit=1000)
    result = np.exp(k_minus) - np.exp(k / 2) / np.pi * integral

    return result


def black_otm_impvol_mc(
    S: np.ndarray, k: float | np.ndarray, T: float, mc_error: bool = False
) -> dict | np.ndarray:
    """
    Calculate Black implied volatility using Monte Carlo simulated stock prices and
    out-of-the-money (OTM) prices.

    Parameters
    ----------
    S : ndarray
        Array of Monte Carlo simulated stock prices.
    k : float or ndarray
        Log-Forward Moneyness `k=log(K/F)` for which the implied volatility is
        calculated.
    T : float
        Time to maturity of the option.
    mc_error : bool, optional
        If True, computes the 95% confidence interval for the implied volatility.

    Returns
    -------
    dict or ndarray
        If `mc_error` is False, returns an ndarray of OTM implied volatilities.
        If `mc_error` is True, returns a dictionary with the following keys:
        - 'otm_impvol': ndarray of OTM implied volatilities.
        - 'otm_impvol_high': ndarray of upper bounds of the 95% confidence interval.
        - 'otm_impvol_low': ndarray of lower bounds of the 95% confidence interval.
        - 'error_95': ndarray of the 95% confidence interval errors for the option
                      prices.
        - 'otm_price': ndarray of the calculated OTM option prices.
    """
    k = np.atleast_1d(k)
    F = np.mean(S)
    K = F * np.exp(k)
    # opttype: 1 for call, -1 for put, depending on moneyness
    opttype = 2 * (K >= F) - 1  # 1 if K >= F (call), -1 if K < F (put)
    payoff = np.maximum(opttype[None, :] * (S[:, None] - K[None, :]), 0.0)
    otm_price = np.mean(payoff, axis=0)
    otm_impvol = black_impvol(K=K, T=T, F=F, value=otm_price, opttype=opttype)

    if mc_error:
        error_95 = 1.96 * np.std(payoff, axis=0) / S.shape[0] ** 0.5
        otm_impvol_high = black_impvol(
            K=K, T=T, F=F, value=otm_price + error_95, opttype=opttype
        )
        otm_impvol_low = black_impvol(
            K=K, T=T, F=F, value=otm_price - error_95, opttype=opttype
        )
        return {
            "otm_impvol": otm_impvol,
            "otm_impvol_high": otm_impvol_high,
            "otm_impvol_low": otm_impvol_low,
            "error_95": error_95,
            "otm_price": otm_price,
        }

    return otm_impvol
