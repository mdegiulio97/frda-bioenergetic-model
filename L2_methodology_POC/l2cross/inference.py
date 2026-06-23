"""MAP estimation + Laplace approximation for the Layer-2 cross-scale model.

The L2 likelihood is near-singular / weakly identified for a single patient
(Fisher condition number ~1e9; the three theta_* and log_r0 are only weakly
separated). Therefore inference is performed as **MAP with the population
prior** (`l2cross.priors.log_prior`), never plain MLE. The prior regularises
the problem and makes the Laplace Hessian well-conditioned; the prior term is
ALWAYS included.

Pipeline per patient:
  1. minimise the (negative) log-posterior over phi with L-BFGS-B, using a
     JAX gradient at the scipy boundary (float64 <-> jax conversion);
  2. form the Laplace covariance cov = inv(H + ridge*I), where H is the Hessian
     of the negative log-posterior at the MAP point.

The forward model (`l2cross.model.forward`) casts its inputs to float32
internally, so the JAX scalars here are kept compatible with that; the
scipy-side optimisation variables are float64.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from . import model
from .priors import POP_MU, POP_SIGMA, log_prior

# Ridge added to the Laplace Hessian before inversion (numerical guard).
_RIDGE = 1e-6


def neg_log_post(phi, patient, times, mask, Y_obs, noise_sigma, mu, sigma):
    """Negative log-posterior (up to additive constants), a JAX scalar.

    likelihood = 0.5 * sum over measured (t,o) of ((Y_obs - forward)/sigma_o)^2
    prior_term = -log_prior(phi, mu, sigma)
    returns likelihood + prior_term.

    `mask` selects measured (t,o); NaN entries in `Y_obs` are also treated as
    unmeasured. Selection is done with `jnp.where` (no boolean indexing) so the
    function stays differentiable w.r.t. phi.
    """
    times_j = jnp.asarray(times, dtype=jnp.float32)
    Y_j = jnp.asarray(Y_obs, dtype=jnp.float32)
    sigma_obs = jnp.asarray(noise_sigma, dtype=jnp.float32).reshape(-1)  # (3,)
    mask_j = jnp.asarray(mask, dtype=bool)  # (T,3)

    valid = mask_j & jnp.isfinite(Y_j)
    Y_clean = jnp.where(valid, Y_j, 0.0)

    Y_pred = model.forward(phi, patient, times_j)  # (T,3)
    resid = (Y_clean - Y_pred) / sigma_obs[None, :]
    resid2 = jnp.where(valid, resid ** 2, 0.0)
    likelihood = 0.5 * jnp.sum(resid2)

    prior_term = -log_prior(phi, mu, sigma)
    return likelihood + prior_term


def fit_map(
    patient,
    schedule,
    Y_obs,
    noise_sigma,
    mu=POP_MU,
    sigma=POP_SIGMA,
    phi_init=None,
):
    """MAP estimate of phi + Laplace covariance for one concrete patient.

    Returns
    -------
    (phi_hat, cov) : phi_hat is a (5,) numpy float64 array; cov is the (5,5)
        Laplace posterior covariance = inv(H + ridge*I), H the Hessian of the
        negative log-posterior at phi_hat.
    """
    times = np.asarray(schedule.times, dtype=float)
    mask = np.asarray(schedule.mask, dtype=bool)
    mu_j = jnp.asarray(mu, dtype=jnp.float32)
    sigma_j = jnp.asarray(sigma, dtype=jnp.float32)
    Y_obs_j = jnp.asarray(Y_obs, dtype=jnp.float32)
    noise_j = jnp.asarray(noise_sigma, dtype=jnp.float32)
    times_j = jnp.asarray(times, dtype=jnp.float32)
    mask_j = jnp.asarray(mask, dtype=bool)

    def _objective(phi):
        return neg_log_post(
            phi, patient, times_j, mask_j, Y_obs_j, noise_j, mu_j, sigma_j
        )

    grad_obj = jax.grad(_objective)

    def _scipy_fun_and_grad(x):
        phi = jnp.asarray(x, dtype=jnp.float32)
        f = float(_objective(phi))
        g = np.asarray(grad_obj(phi), dtype=np.float64)
        return f, g

    if phi_init is None:
        x0 = np.asarray(mu, dtype=np.float64)
    else:
        x0 = np.asarray(phi_init, dtype=np.float64)

    res = minimize(_scipy_fun_and_grad, x0, jac=True, method="L-BFGS-B")
    phi_hat = jnp.asarray(res.x, dtype=jnp.float32)

    H = jax.hessian(_objective)(phi_hat)  # (5,5)
    H = np.asarray(H, dtype=np.float64)
    H = 0.5 * (H + H.T)  # enforce exact symmetry
    n = H.shape[0]
    cov = np.linalg.inv(H + _RIDGE * np.eye(n))
    cov = 0.5 * (cov + cov.T)

    return np.asarray(res.x, dtype=np.float64), cov


def fit_cohort(cohort, schedule, noise_sigma, mu=POP_MU, sigma=POP_SIGMA):
    """Fit MAP + Laplace for every patient record in a cohort.

    `cohort` is a list of dicts with keys {"patient", "phi_true", "Y_obs"}
    (as produced by `generator.sample_cohort`). Returns a list of dicts with
    keys {"phi_true", "phi_hat", "cov"}. Patients are concrete, so this is a
    plain python loop.
    """
    out = []
    for rec in cohort:
        phi_hat, cov = fit_map(
            rec["patient"],
            schedule,
            rec["Y_obs"],
            noise_sigma,
            mu=mu,
            sigma=sigma,
        )
        out.append(
            {
                "phi_true": np.asarray(rec["phi_true"], dtype=np.float64),
                "phi_hat": phi_hat,
                "cov": cov,
            }
        )
    return out
