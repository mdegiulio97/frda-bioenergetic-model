"""Synthetic data generator for the Layer-2 cross-scale FRDA model.

Builds noisy, possibly sparsely-sampled longitudinal observations Y_obs from the
JAX forward model (`l2cross.model.observe`) for a CONCRETE Patient (the L1 bridge
runs in scipy, so patients are host-side python objects and cannot be vmapped).

Conventions:
  - times       : (T,) ascending float array (years).
  - observables : 3, in `priors.OBSERVABLES` order = ["cord_CSA", "QSM", "SNAP"].
  - mask        : (T, 3) bool, True where an observable is measured at that time.
  - unmeasured entries are returned as np.nan.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from l2cross import model, priors
from l2cross.l1_bridge import make_patient


@dataclass
class Schedule:
    """Acquisition schedule.

    times : (T,) float, ascending.
    mask  : (T, 3) bool, which of the 3 observables is measured at each time.
    """

    times: np.ndarray
    mask: np.ndarray


def full_schedule(times) -> Schedule:
    """All 3 observables measured at every time."""
    times = np.asarray(times, dtype=float)
    T = times.shape[0]
    mask = np.ones((T, 3), dtype=bool)
    return Schedule(times=times, mask=mask)


def sparse_schedule(times, obs_per_time: int = 1) -> Schedule:
    """Measure `obs_per_time` observables per time, cycling through the 3 channels.

    Deterministic round-robin assignment so each time has exactly
    `obs_per_time` (in [1, 3]) measured observables.
    """
    times = np.asarray(times, dtype=float)
    T = times.shape[0]
    n_obs = len(priors.OBSERVABLES)
    obs_per_time = int(np.clip(obs_per_time, 1, n_obs))
    mask = np.zeros((T, n_obs), dtype=bool)
    for t in range(T):
        for j in range(obs_per_time):
            mask[t, (t + j) % n_obs] = True
    return Schedule(times=times, mask=mask)


def generate_dataset(phi, patient, schedule: Schedule, noise_sigma, key) -> np.ndarray:
    """Generate one noisy (T, 3) observation array.

    Y_true = model.observe(phi, patient, schedule.times); add Gaussian noise with
    per-observable std `noise_sigma` (shape (3,)); set unmeasured entries to NaN.
    """
    times = np.asarray(schedule.times, dtype=float)
    T = times.shape[0]
    noise_sigma = jnp.asarray(noise_sigma, dtype=jnp.float32).reshape(-1)

    Y = model.observe(phi, patient, jnp.asarray(times, dtype=jnp.float32))  # (T, 3)
    noise = jax.random.normal(key, (T, 3)) * noise_sigma[None, :]
    Y_noisy = np.asarray(Y + noise, dtype=float)

    mask = np.asarray(schedule.mask, dtype=bool)
    Y_obs = np.where(mask, Y_noisy, np.nan)
    return Y_obs


def sample_cohort(key, n: int, schedule: Schedule, noise_sigma, F_range=(0.1, 0.5)):
    """Sample a cohort of `n` synthetic FRDA patients.

    For each patient: split the key; draw F ~ U(F_range) (FRDA range); build the
    concrete Patient via the L1 bridge (scipy, looped in python); draw phi_true from
    the population prior; generate noisy observations.

    Returns a list of dicts: {"patient", "phi_true", "Y_obs"}.
    """
    lo, hi = F_range
    records = []
    for i in range(n):
        key, k_patient, k_phi, k_data = jax.random.split(key, 4)
        F = float(jax.random.uniform(k_patient, (), minval=lo, maxval=hi))
        patient = make_patient(F)
        phi_true = priors.sample_prior(k_phi)
        Y_obs = generate_dataset(phi_true, patient, schedule, noise_sigma, k_data)
        records.append({"patient": patient, "phi_true": phi_true, "Y_obs": Y_obs})
    return records
