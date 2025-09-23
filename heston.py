import numpy as np
from utils import lewis_formula_otm_price, black_impvol


def phi_heston(u, tau, params):
    """
    Compute the characteristic function of the Heston model E[exp(i * u * X_tau)]
    where X_tau = log(S_tau/S_0) at u for time tau.

    Parameters
    ----------
    u : float or array_like
        Points at which to evaluate the characteristic function
    tau : float
        Time to maturity
    params : dict
        Dictionary containing model parameters:
        - lbd: mean reversion rate
        - rho: correlation between asset and variance
        - nu: volatility of variance
        - vbar: long-term variance
        - v: initial variance

    Returns
    -------
    complex
        Value of the characteristic function
    """
    lbd = params["lbd"]
    rho = params["rho"]
    nu = params["nu"]
    vbar = params["vbar"]
    v = params["v"]

    al = -u * u / 2 - 1j * u / 2
    bet = lbd - rho * nu * 1j * u
    gam = nu**2 / 2
    d = np.sqrt(bet * bet - 4 * al * gam)
    rp = (bet + d) / (2 * gam)
    rm = (bet - d) / (2 * gam)
    g = rm / rp
    D = rm * (1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau))
    C = lbd * (rm * tau - 2 / nu**2 * np.log((1 - g * np.exp(-d * tau)) / (1 - g)))
    return np.exp(C * vbar + D * v)


def impvol_heston_charfunc(k, tau, params):
    """
    Calculate implied volatility in the Heston model using the characteristic function.

    Parameters
    ----------
    k : array_like
        Log strike k = log(K/F)
    tau : float
        Time to maturity
    params : dict
        Model parameters

    Returns
    -------
    array_like
        Black implied volatility
    """
    k = np.atleast_1d(np.asarray(k))
    otm_price = lewis_formula_otm_price(
        lambda u, tau: phi_heston(u=u, tau=tau, params=params),
        k=k,
        tau=tau,
    )
    opttype = 2 * (k > 0) - 1  # otm options
    impvol = black_impvol(K=np.exp(k), T=tau, F=1, value=otm_price, opttype=opttype)
    return impvol
