"""Population-level priors over the latent Layer-2 parameters phi.

phi = [log_r0, theta_cord, theta_dentate, theta_nerve, t0] (dim = 5).

Units / meaning:
  log_r0       : log of base damage rate r0 (1/year). r0 ~ exp(log_r0).
  theta_cord   : vulnerability threshold for spinal cord  (dimensionless, ~ (0,1))
  theta_dentate: vulnerability threshold for dentate nucleus (dimensionless)
  theta_nerve  : vulnerability threshold for peripheral nerve (dimensionless)
  t0           : clinical onset time (years)
"""

import jax
import jax.numpy as jnp

PHI_NAMES = ["log_r0", "theta_cord", "theta_dentate", "theta_nerve", "t0"]
TISSUES = ["cord", "dentate", "nerve"]
OBSERVABLES = ["cord_CSA", "QSM", "SNAP"]

# Prior means:
#   log_r0 ~ log(0.05) ≈ -3.0  (base rate ~0.05 /yr)
#   theta_* ~ 0.4 (in a plausible (0,1) band)
#   t0 ~ 10 years
POP_MU = jnp.array([-3.0, 0.4, 0.4, 0.4, 10.0])
# Prior std devs (same order)
POP_SIGMA = jnp.array([0.5, 0.15, 0.15, 0.15, 5.0])


def sample_prior(key, mu=POP_MU, sigma=POP_SIGMA):
    """Draw one phi ~ N(mu, diag(sigma^2))."""
    eps = jax.random.normal(key, shape=mu.shape)
    return mu + sigma * eps


def log_prior(phi, mu=POP_MU, sigma=POP_SIGMA):
    """Gaussian log-prior (additive constants dropped). Max at phi == mu."""
    z = (phi - mu) / sigma
    return -0.5 * jnp.sum(z * z)
