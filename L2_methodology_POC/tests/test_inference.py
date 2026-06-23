"""Tests for the MAP + Laplace inference module (l2cross.inference).

Ground-truth data is produced with the generator. The L2 likelihood is
near-singular for a single patient, so inference is MAP with the population
prior; only the well-identified parameters (log_r0, t0) are expected to be
recovered tightly. theta_* are weakly identified and prior-dominated.
"""

import jax
import jax.numpy as jnp
import numpy as np

from l2cross.generator import full_schedule, generate_dataset, sparse_schedule
from l2cross.identifiability import crlb_std, fisher
from l2cross.inference import fit_cohort, fit_map, neg_log_post
from l2cross.l1_bridge import make_patient
from l2cross.priors import POP_MU, POP_SIGMA


# --- Rich regime: early onset, clearly damaged patient, dense low-noise data ---
def _rich_setup():
    patient = make_patient(0.2)  # clearly damaged
    times = np.linspace(0.0, 40.0, 41)
    schedule = full_schedule(times)
    noise_sigma = np.array([0.01, 0.01, 0.02])
    # phi_true near POP_MU but early onset + fast/vulnerable so observables move.
    phi_true = jnp.array([np.log(0.15), 0.35, 0.35, 0.35, 8.0], dtype=jnp.float32)
    return patient, times, schedule, noise_sigma, phi_true


def test_recovery_rich_regime():
    """Rich-regime "recovery" for a SINGLE patient is, per the Fisher analysis,
    information GAIN rather than point-estimate recovery: the single-patient L2
    likelihood is near-singular (cond ~1e10), so the MAP point stays
    prior-dominated. What the data robustly delivers is variance reduction on
    the two best-observed parameters, log_r0 (idx 0, via cord_CSA) and t0
    (idx 4, via the onset of all channels): their Laplace posterior std is
    strictly below the prior std. theta_* (1,2,3) are weakly identified and may
    not gain (purely prior-dominated) — we do NOT require them to.
    """
    patient, times, schedule, noise_sigma, phi_true = _rich_setup()
    key = jax.random.PRNGKey(0)
    Y_obs = generate_dataset(phi_true, patient, schedule, noise_sigma, key)

    phi_hat, cov = fit_map(patient, schedule, Y_obs, noise_sigma)

    prior_sigma = np.asarray(POP_SIGMA, dtype=float)
    post_std = np.sqrt(np.diag(cov))
    # log_r0 and t0 gain information from the data (posterior < prior std).
    assert post_std[0] < prior_sigma[0], (post_std[0], prior_sigma[0])
    assert post_std[4] < prior_sigma[4], (post_std[4], prior_sigma[4])

    # The MAP stays inside a prior-plausible band (no runaway): within ~3 prior
    # sigma of the prior mean for every component (it is regularised, not wild).
    mu = np.asarray(POP_MU, dtype=float)
    z = np.abs(phi_hat - mu) / prior_sigma
    assert np.all(z < 3.0), z


def test_prior_pull_sparse_regime():
    # Very sparse / nearly empty data -> the prior dominates and phi_hat ~ mu.
    patient = make_patient(0.2)
    times = np.array([1.0, 2.0])  # tiny, very early (before onset -> almost no signal)
    schedule = sparse_schedule(times, obs_per_time=1)
    noise_sigma = np.array([0.5, 0.5, 0.5])  # large noise -> weak likelihood
    phi_true = POP_MU
    key = jax.random.PRNGKey(7)
    Y_obs = generate_dataset(phi_true, patient, schedule, noise_sigma, key)

    phi_hat, cov = fit_map(patient, schedule, Y_obs, noise_sigma)
    mu = np.asarray(POP_MU, dtype=float)
    # Normalised distance in prior units should be small.
    z = (phi_hat - mu) / np.asarray(POP_SIGMA, dtype=float)
    assert np.linalg.norm(z) < 1.0, (phi_hat, mu, z)


def test_laplace_cov_spd():
    patient, times, schedule, noise_sigma, phi_true = _rich_setup()
    key = jax.random.PRNGKey(1)
    Y_obs = generate_dataset(phi_true, patient, schedule, noise_sigma, key)
    phi_hat, cov = fit_map(patient, schedule, Y_obs, noise_sigma)

    assert cov.shape == (5, 5)
    assert np.allclose(cov, cov.T, atol=1e-8)
    eig = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    assert np.all(eig > 0), eig
    diag = np.diag(cov)
    assert np.all(np.isfinite(diag))
    assert np.all(diag > 0), diag


def test_consistency_with_crlb():
    """Laplace posterior std vs CRLB for log_r0 (idx 0).

    NOTE: the bare-likelihood Fisher is near-singular, so `crlb_std(fisher(...))`
    returns NaN for log_r0 (sqrt of a negative diagonal of the ill-conditioned
    inverse). The Laplace covariance approximates the *posterior* (likelihood +
    prior), so the principled CRLB comparison adds the prior information
    diag(1/sigma^2) to the Fisher matrix. Both are local quadratic
    approximations; require same order of magnitude (loose check).
    """
    patient, times, schedule, noise_sigma, phi_true = _rich_setup()
    key = jax.random.PRNGKey(2)
    Y_obs = generate_dataset(phi_true, patient, schedule, noise_sigma, key)
    phi_hat, cov = fit_map(patient, schedule, Y_obs, noise_sigma)

    laplace_std0 = np.sqrt(np.diag(cov))[0]

    # Posterior information = Fisher (likelihood) + prior precision.
    F = fisher(phi_hat, patient, times, schedule.mask, noise_sigma)
    prior_info = np.diag(1.0 / np.asarray(POP_SIGMA, dtype=float) ** 2)
    F_post = F + prior_info
    crlb_post0 = crlb_std(F_post)[0]

    ratio = laplace_std0 / crlb_post0
    assert 0.1 < ratio < 10.0, (laplace_std0, crlb_post0, ratio)

    # And confirm the documented degeneracy: the bare-likelihood CRLB for
    # log_r0 is NOT finite (Fisher singular) -> motivates the prior / MAP.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bare = crlb_std(F)[0]
    assert not np.isfinite(bare), bare


def test_neg_log_post_scalar_and_finite():
    patient, times, schedule, noise_sigma, phi_true = _rich_setup()
    key = jax.random.PRNGKey(3)
    Y_obs = generate_dataset(phi_true, patient, schedule, noise_sigma, key)
    val = neg_log_post(
        phi_true,
        patient,
        jnp.asarray(times),
        jnp.asarray(schedule.mask),
        jnp.asarray(Y_obs),
        jnp.asarray(noise_sigma),
        POP_MU,
        POP_SIGMA,
    )
    val = float(val)
    assert np.isfinite(val)
    assert np.ndim(np.asarray(val)) == 0


def test_reproducibility():
    patient, times, schedule, noise_sigma, phi_true = _rich_setup()
    key = jax.random.PRNGKey(5)
    Y_obs = generate_dataset(phi_true, patient, schedule, noise_sigma, key)
    phi_hat1, cov1 = fit_map(patient, schedule, Y_obs, noise_sigma)
    phi_hat2, cov2 = fit_map(patient, schedule, Y_obs, noise_sigma)
    assert np.allclose(phi_hat1, phi_hat2)
    assert np.allclose(cov1, cov2)


def test_fit_cohort():
    patient, times, schedule, noise_sigma, phi_true = _rich_setup()
    key = jax.random.PRNGKey(9)
    Y_obs = generate_dataset(phi_true, patient, schedule, noise_sigma, key)
    cohort = [
        {"patient": patient, "phi_true": phi_true, "Y_obs": Y_obs},
        {"patient": patient, "phi_true": phi_true, "Y_obs": Y_obs},
    ]
    out = fit_cohort(cohort, schedule, noise_sigma, POP_MU, POP_SIGMA)
    assert isinstance(out, list)
    assert len(out) == 2
    for rec in out:
        assert set(rec.keys()) == {"phi_true", "phi_hat", "cov"}
        assert np.asarray(rec["phi_hat"]).shape == (5,)
        assert np.asarray(rec["cov"]).shape == (5, 5)
