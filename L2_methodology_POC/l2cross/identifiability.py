"""Identifiability tooling for the Layer-2 cross-scale model.

Provides Fisher information, Cramer-Rao lower bounds (CRLB) and profile
likelihood for the latent phi = [log_r0, theta_cord, theta_dentate,
theta_nerve, t0] (dim = 5), given a CONCRETE patient and acquisition schedule.

Design notes
------------
- The forward model (`l2cross.model.forward`) is JAX-differentiable in `phi`
  with the patient and times treated as fixed concrete constants. We use
  `jax.jacobian` / `jax.grad` for sensitivities and convert to numpy at the
  scipy boundary.
- The acquisition schedule is passed as plain arrays `times (T,)` and
  `mask (T,3) bool` to avoid a hard dependency on the schedule generator.
- `noise_sigma` is a per-observable std-dev vector of shape (3,).
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from . import model

# Delta chi^2 / 2 threshold for a 95% CI on a single parameter (1 dof).
THRESHOLD_1DOF = 1.92


def fisher(phi, patient, times, mask, noise_sigma):
    """Gaussian Fisher information matrix F (5,5) for the schedule.

    Parameters
    ----------
    phi : (5,) latent parameter vector (the point at which info is evaluated).
    patient : concrete Patient (l1_bridge.Patient).
    times : (T,) concrete acquisition times.
    mask : (T,3) bool; True where observable o is measured at time t.
    noise_sigma : (3,) per-observable noise std-dev.

    Returns
    -------
    F : (5,5) symmetric numpy array,
        F = sum_{measured (t,o)} (1/sigma_o^2) * outer(J[t,o,:], J[t,o,:]).
    """
    phi = jnp.asarray(phi, dtype=jnp.float32)
    times = jnp.asarray(times, dtype=jnp.float32)

    # J : (T, 3, 5) sensitivity of each observable w.r.t. each phi component.
    J = jax.jacobian(lambda p: model.forward(p, patient, times))(phi)

    mask = jnp.asarray(mask, dtype=bool)  # (T,3)
    sigma = jnp.asarray(noise_sigma, dtype=jnp.float32)  # (3,)

    # Per-measurement inverse variance, zeroed where not measured.
    inv_var = jnp.where(mask, 1.0 / (sigma[None, :] ** 2), 0.0)  # (T,3)

    # F = sum_{t,o} w[t,o] * J[t,o,:] outer J[t,o,:]
    # einsum over time t and observable o.
    F = jnp.einsum("to,toi,toj->ij", inv_var, J, J)

    F = np.asarray(F, dtype=np.float64)
    # Enforce exact symmetry (guards against tiny float asymmetry).
    F = 0.5 * (F + F.T)
    return F


def crlb(F, ridge=1e-9):
    """Cramer-Rao lower bound variances: diag(inv(F + ridge*I)).

    Returns a (5,) numpy array of parameter variances.
    """
    F = np.asarray(F, dtype=np.float64)
    n = F.shape[0]
    Finv = np.linalg.inv(F + ridge * np.eye(n))
    return np.diag(Finv).copy()


def crlb_std(F, ridge=1e-9):
    """CRLB standard deviations: sqrt(crlb(F))."""
    return np.sqrt(crlb(F, ridge=ridge))


def _make_cost(patient, times, mask, Y_obs, noise_sigma, prior_mu=None, prior_sigma=None):
    """Build a JAX-traceable least-squares cost over the full 5-vector phi.

    cost(phi) = 0.5 * sum_{measured} ((Y_obs - forward)/sigma)^2
                + 0.5 * sum ((phi - prior_mu)/prior_sigma)^2   [if prior given]

    Unmeasured / NaN entries (per `mask`) are ignored.
    """
    times_j = jnp.asarray(times, dtype=jnp.float32)
    Y_obs_j = jnp.asarray(Y_obs, dtype=jnp.float32)
    sigma_j = jnp.asarray(noise_sigma, dtype=jnp.float32)  # (3,)
    mask_j = jnp.asarray(mask, dtype=bool)  # (T,3)
    # Treat NaN observations as unmeasured too.
    valid = mask_j & jnp.isfinite(Y_obs_j)
    Y_clean = jnp.where(valid, Y_obs_j, 0.0)

    use_prior = prior_mu is not None and prior_sigma is not None
    if use_prior:
        pmu = jnp.asarray(prior_mu, dtype=jnp.float32)
        psig = jnp.asarray(prior_sigma, dtype=jnp.float32)

    def cost(phi):
        Y = model.forward(phi, patient, times_j)  # (T,3)
        resid = jnp.where(valid, (Y_clean - Y) / sigma_j[None, :], 0.0)
        c = 0.5 * jnp.sum(resid ** 2)
        if use_prior:
            z = (phi - pmu) / psig
            c = c + 0.5 * jnp.sum(z * z)
        return c

    return cost


def profile_likelihood(
    patient,
    times,
    mask,
    Y_obs,
    noise_sigma,
    idx,
    grid,
    prior_mu=None,
    prior_sigma=None,
    phi_init=None,
    bounds=None,
):
    """Profile-likelihood curve for parameter `idx` over `grid`.

    Self-contained (no inference module). First finds the full 5-param MLE
    `theta_hat` minimizing the least-squares (+optional prior) cost. Then for
    each value v in `grid`, fixes phi[idx]=v and re-optimizes the remaining 4
    free parameters, recording the achieved cost.

    Parameters
    ----------
    bounds : optional (n,2) sequence of (lo, hi) box-constraints on the full
        phi. When the prior is excluded (likelihood-only profile) these keep the
        free-parameter re-optimisation inside the physical region; without them
        an unconstrained free `log_r0` can overflow `exp` and yield NaN costs.
        Bounds do not add curvature inside the feasible region, so genuinely
        flat (non-identifiable) directions stay flat.

    Returns
    -------
    (grid, costs) : grid is returned as a numpy array; costs is np.array of the
        minimized cost at each grid value. Compare `costs - cost(theta_hat)`
        against `THRESHOLD_1DOF` for a 95% (1 dof) interval.
    """
    grid = np.asarray(grid, dtype=np.float64)
    cost = _make_cost(
        patient, times, mask, Y_obs, noise_sigma, prior_mu=prior_mu, prior_sigma=prior_sigma
    )
    grad_cost = jax.grad(cost)
    bounds = None if bounds is None else [tuple(b) for b in bounds]

    def _full_obj(x):
        xj = jnp.asarray(x, dtype=jnp.float32)
        f = float(cost(xj))
        g = np.asarray(grad_cost(xj), dtype=np.float64)
        return f, g

    if phi_init is None:
        if prior_mu is not None:
            x0 = np.asarray(prior_mu, dtype=np.float64)
        else:
            from .priors import POP_MU

            x0 = np.asarray(POP_MU, dtype=np.float64)
    else:
        x0 = np.asarray(phi_init, dtype=np.float64)

    # --- Full 5-param fit -> theta_hat, cost0 ---
    res = minimize(_full_obj, x0, jac=True, method="L-BFGS-B", bounds=bounds)
    theta_hat = np.asarray(res.x, dtype=np.float64)
    cost0 = float(res.fun)

    n = theta_hat.shape[0]
    free_idx = [k for k in range(n) if k != idx]
    bounds_free = None if bounds is None else [bounds[k] for k in free_idx]

    def _profile_obj_factory(v):
        v32 = np.float64(v)

        def _obj(xfree):
            # Reassemble full phi: insert v at position idx.
            phi = np.empty(n, dtype=np.float64)
            phi[free_idx] = xfree
            phi[idx] = v32
            phij = jnp.asarray(phi, dtype=jnp.float32)
            f = float(cost(phij))
            gfull = np.asarray(grad_cost(phij), dtype=np.float64)
            return f, gfull[free_idx]

        return _obj

    costs = np.empty(grid.shape[0], dtype=np.float64)
    # Warm-start the free params from theta_hat.
    xfree0 = theta_hat[free_idx].copy()
    for i, v in enumerate(grid):
        obj = _profile_obj_factory(v)
        r = minimize(obj, xfree0, jac=True, method="L-BFGS-B", bounds=bounds_free)
        costs[i] = float(r.fun)

    return grid, np.asarray(costs, dtype=np.float64)
