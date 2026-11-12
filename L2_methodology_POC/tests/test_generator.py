import jax
import jax.numpy as jnp
import numpy as np

from l2cross import model
from l2cross.generator import (
    Schedule,
    full_schedule,
    generate_dataset,
    sample_cohort,
    sparse_schedule,
)
from l2cross.l1_bridge import Patient, make_patient


def _phi():
    # fast, vulnerable, early onset -> non-trivial trajectory
    return jnp.array([jnp.log(0.4), 0.1, 0.1, 0.1, 2.0])


def _patient():
    return make_patient(0.2)


def _times(T=20):
    return np.linspace(0.0, 40.0, T)


def test_full_schedule_shapes_and_mask():
    times = _times(7)
    sch = full_schedule(times)
    assert isinstance(sch, Schedule)
    assert sch.times.shape == (7,)
    assert sch.mask.shape == (7, 3)
    assert sch.mask.dtype == bool
    assert sch.mask.all()
    # ascending
    assert np.all(np.diff(sch.times) > 0)


def test_generate_dataset_shape_and_nan_mask():
    phi = _phi()
    patient = _patient()
    times = _times(15)
    # custom sparse mask: only observable 0 at every time
    mask = np.zeros((15, 3), dtype=bool)
    mask[:, 0] = True
    sch = Schedule(times=np.asarray(times, dtype=float), mask=mask)
    noise_sigma = np.array([0.01, 0.01, 0.01])
    key = jax.random.PRNGKey(0)
    Y_obs = generate_dataset(phi, patient, sch, noise_sigma, key)

    assert isinstance(Y_obs, np.ndarray)
    assert Y_obs.shape == (15, 3)
    # measured (mask True) -> finite; unmeasured -> NaN
    assert np.all(np.isfinite(Y_obs[mask]))
    assert np.all(np.isnan(Y_obs[~mask]))


def test_generate_dataset_zero_noise_matches_model():
    phi = _phi()
    patient = _patient()
    times = _times(25)
    sch = full_schedule(times)
    noise_sigma = np.zeros(3)
    key = jax.random.PRNGKey(1)
    Y_obs = generate_dataset(phi, patient, sch, noise_sigma, key)

    Y_true = np.asarray(model.observe(phi, patient, sch.times))
    assert np.allclose(Y_obs, Y_true, atol=1e-4)


def test_noise_level_sanity():
    phi = _phi()
    patient = _patient()
    T = 4000
    times = _times(T)
    sch = full_schedule(times)
    sigma = np.array([0.02, 0.03, 0.05])
    key = jax.random.PRNGKey(7)
    Y_obs = generate_dataset(phi, patient, sch, sigma, key)
    Y_true = np.asarray(model.observe(phi, patient, sch.times))
    resid = Y_obs - Y_true  # (T,3), all measured (full schedule)
    emp_std = resid.std(axis=0)
    for k in range(3):
        assert abs(emp_std[k] - sigma[k]) / sigma[k] < 0.30


def test_reproducibility():
    phi = _phi()
    patient = _patient()
    times = _times(30)
    sch = full_schedule(times)
    sigma = np.array([0.02, 0.02, 0.02])
    k1 = jax.random.PRNGKey(42)
    k2 = jax.random.PRNGKey(99)
    a = generate_dataset(phi, patient, sch, sigma, k1)
    a2 = generate_dataset(phi, patient, sch, sigma, k1)
    b = generate_dataset(phi, patient, sch, sigma, k2)
    assert np.array_equal(a, a2)
    assert not np.allclose(a, b)


def test_sample_cohort():
    times = _times(12)
    sch = full_schedule(times)
    sigma = np.array([0.02, 0.02, 0.02])
    key = jax.random.PRNGKey(123)
    cohort = sample_cohort(key, 5, sch, sigma, F_range=(0.1, 0.5))
    assert isinstance(cohort, list)
    assert len(cohort) == 5
    Fs = []
    for rec in cohort:
        assert set(rec.keys()) >= {"patient", "phi_true", "Y_obs"}
        assert isinstance(rec["patient"], Patient)
        assert np.asarray(rec["phi_true"]).shape == (5,)
        assert np.asarray(rec["Y_obs"]).shape == (12, 3)
        Fs.append(rec["patient"].F)
    # patients have varying F
    assert len(set(Fs)) > 1
    assert all(0.1 <= F <= 0.5 for F in Fs)


def test_sparse_schedule_helper():
    times = _times(10)
    sch = sparse_schedule(times, obs_per_time=1)
    assert sch.mask.shape == (10, 3)
    # exactly one observable measured per time
    assert np.all(sch.mask.sum(axis=1) == 1)
