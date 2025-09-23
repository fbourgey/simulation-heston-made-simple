import numpy as np
from scipy import stats

from utils import black_otm_impvol_mc


def ivi_scheme(T, a, b, c, V0, n_disc, n_paths, seed=None):
    """
    Simulate the variance process of the Heston model using the integrated variance
    implicit (iVi) scheme of Abi Jaber (2025).

    Parameters
    ----------
    T : float
        Maturity time
    a : float
        Drift parameter (non-negative)
    b : float
        Mean reversion parameter
    c : float
        Volatility of volatility parameter
    V0 : float
        Initial variance (non-negative)
    n_disc : int
        Number of time discretization steps
    n_paths : int
        Number of simulation paths
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    V : ndarray, shape (n_disc + 1, n_paths)
        Variance process values
    U : ndarray, shape (n_disc, n_paths)
        Integrated variance increments
    Z : ndarray, shape (n_disc, n_paths)
        Standardized increments
    ts : ndarray, shape (n_disc + 1,)
        Time grid points
    """
    if seed is not None:
        np.random.seed(seed)

    if a < 0.0:
        raise ValueError("a must be non-negative.")

    if V0 < 0.0:
        raise ValueError("V0 must be non-negative.")

    np.random.seed(seed)
    ts = np.linspace(0, T, n_disc + 1)
    dts = ts[1:] - ts[:-1]

    exp_arg = dts if b == 0.0 else (np.exp(b * dts) - 1.0) / b
    alpha_0 = (a / b) * (exp_arg - dts)
    sigma = c * exp_arg

    V = np.zeros((n_disc + 1, n_paths))
    V[0, :] = V0  # initial variance

    U = np.zeros((n_disc, n_paths))
    Z = np.zeros((n_disc, n_paths))

    for i in range(n_disc):
        alpha_i = V[i, :] * exp_arg[i] + alpha_0[i]
        nu = alpha_i
        lam = (alpha_i / sigma[i]) ** 2
        U[i, :] = stats.invgauss.rvs(mu=nu / lam, loc=0, scale=lam, size=n_paths)
        Z[i, :] = (U[i, :] - alpha_i) / sigma[i]
        V[i + 1, :] = V[i, :] + a * dts[i] + b * U[i, :] + c * Z[i, :]

    return V, U, Z, ts


def impvol_heston_ivi_scheme(k, T, params, n_disc, n_paths, S0=1.0, seed=None):
    """
    Compute implied volatility in the Heston model using the iVi scheme.

    Parameters
    ----------
    k : array_like
        Log strike k = log(K/F)
    T : float
        Maturity time
    params : dict
        Model parameters with keys: 'v', 'lbd', 'vbar', 'nu', 'rho'
    n_disc : int
        Number of time discretization steps
    n_paths : int
        Number of simulation paths
    S0 : float, optional
        Initial stock price (default: 1.0)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    ndarray
        Black implied volatility for each strike
    """
    if seed is not None:
        np.random.seed(seed)

    k = np.atleast_1d(np.asarray(k))
    V0 = params["v"]
    b = -params["lbd"]
    a = params["vbar"] * params["lbd"]
    c = params["nu"]
    rho = params["rho"]
    rho_bar = np.sqrt(1.0 - rho**2)

    _, U, Z, _ = ivi_scheme(
        T=T, a=a, b=b, c=c, V0=V0, n_disc=n_disc, n_paths=n_paths, seed=seed
    )
    normal = np.random.normal(size=(n_disc, n_paths))
    log_S = np.log(S0) + (
        -0.5 * U.sum(axis=0)
        + rho * Z.sum(axis=0)
        + rho_bar * (np.sqrt(U) * normal).sum(axis=0)
    )
    return np.asarray(black_otm_impvol_mc(S=np.exp(log_S), k=k, T=T))
