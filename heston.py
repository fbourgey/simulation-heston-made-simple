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


def simulate_paths_qe_scheme(
    T, params, n_disc, n_paths, psi_c=1.5, seed=None, eps=1e-14
):
    """
    Simulate Heston model paths using Andersen's QE discretization scheme.

    Parameters
    ----------
    T : float
        Time to maturity.
    params : dict
        Model parameters: 'S0', 'v', 'lbd', 'vbar', 'nu', 'rho'.
    n_disc : int
        Number of discretization steps.
    n_paths : int
        Number of Monte Carlo paths.
    psi_c : float, default 1.5
        Critical value for quadratic/exponential scheme switching.
    seed : int, optional
        Random seed for reproducibility.
    eps : float, default 1e-14
        Floor to keep conditional mean/variance and v_t non-negative.

    Returns
    -------
    S : ndarray, shape (n_disc + 1, n_paths)
        Stock price paths.
    v : ndarray, shape (n_disc + 1, n_paths)
        Variance paths.
    """
    if seed is not None:
        np.random.seed(seed)

    nu = params["nu"]
    lbd = params["lbd"]
    vbar = params["vbar"]
    rho = params["rho"]

    logS_qe = np.zeros((n_disc + 1, n_paths), dtype=float)
    v_qe = np.zeros((n_disc + 1, n_paths), dtype=float)
    logS_qe[0, :] = np.log(params["S0"])
    v_qe[0, :] = params["v"]

    ts = np.linspace(0.0, T, n_disc + 1)
    dt = ts[1] - ts[0]
    edt = np.exp(-lbd * dt)

    for i in range(n_disc):
        # conditional mean and variance
        m = (v_qe[i, :] - vbar) * edt + vbar
        m = np.maximum(m, eps)  # ensure positivity
        s2 = (nu**2 / lbd) * (
            edt * (1 - edt) * (v_qe[i, :] - vbar) + vbar / 2 * (1 - edt**2)
        )
        # compute relative variance
        psi = s2 / m**2

        # Regime 1: Quadratic form
        mask_quad = psi <= psi_c
        if np.any(mask_quad):
            psi_quad = psi[mask_quad]
            Z_quad = np.random.normal(size=mask_quad.sum())
            b2 = (2.0 + 2.0 * np.sqrt(1.0 - psi_quad / 2.0) - psi_quad) / psi_quad
            a = m[mask_quad] / (1.0 + b2)
            v_qe[i + 1, mask_quad] = a * (b2**0.5 + Z_quad) ** 2

        # Regime 2: Exponential form
        mask_exp = ~mask_quad
        if np.any(mask_exp):
            psi_exp = psi[mask_exp]
            # clip p to [0, 1)
            p = np.clip((psi_exp - 1.0) / (psi_exp + 1.0), 0.0, 1.0 - 1e-15)
            beta = m[mask_exp] * (psi_exp + 1.0) / 2.0

            U_exp = np.random.uniform(0.0, 1.0, size=mask_exp.sum())
            alive = U_exp > p  # if False -> atom at zero

            # start at zero
            v_qe[i + 1, mask_exp] = 0.0
            if np.any(alive):
                U_exp_new = np.random.uniform(0.0, 1.0, size=alive.sum())
                U_exp_new = np.clip(U_exp_new, eps, 1.0)
                idx_exp = np.where(mask_exp)[0]
                alive_idx = idx_exp[alive]
                v_qe[i + 1, alive_idx] = -beta[alive] * np.log(U_exp_new)

        v_qe[i + 1, :] = np.maximum(v_qe[i + 1, :], eps)  # ensure positivity

        # trapezoidal rule for integrated variance
        int_v_trap_i = 0.5 * (v_qe[i, :] + v_qe[i + 1, :]) * dt
        int_v_trap_i = np.maximum(int_v_trap_i, 0.0)

        logS_qe[i + 1, :] = (
            logS_qe[i, :]
            - 0.5 * int_v_trap_i
            + rho
            * (v_qe[i + 1, :] - v_qe[i, :] - lbd * vbar * dt + lbd * int_v_trap_i)
            / nu
            + np.sqrt(np.maximum((1.0 - rho**2) * int_v_trap_i, 0.0))
            * np.random.normal(size=n_paths)
        )

    return np.exp(logS_qe), v_qe
