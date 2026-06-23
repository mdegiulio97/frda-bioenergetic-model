import jax
import jax.numpy as jnp

from l2cross.priors import (
    PHI_NAMES,
    POP_MU,
    POP_SIGMA,
    log_prior,
    sample_prior,
)


def test_shapes_dim5():
    assert len(PHI_NAMES) == 5
    assert POP_MU.shape == (5,)
    assert POP_SIGMA.shape == (5,)


def test_sample_prior_finite():
    key = jax.random.PRNGKey(0)
    phi = sample_prior(key)
    assert phi.shape == (5,)
    assert jnp.all(jnp.isfinite(phi))


def test_log_prior_max_at_mu():
    at_mu = log_prior(POP_MU)
    at_mu_plus_sigma = log_prior(POP_MU + POP_SIGMA)
    assert at_mu >= at_mu_plus_sigma
    # mu is the unique maximum (strictly greater than a perturbed point)
    assert at_mu > at_mu_plus_sigma
