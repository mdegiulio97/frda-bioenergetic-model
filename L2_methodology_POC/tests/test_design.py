"""Tests for the acquisition-design / value-of-information module.

These follow the same near-singular-Fisher discipline as the identifiability
tests: comparisons between designs are made via the D-optimality criterion
``logdet(F + ridge*I)`` under a COMMON ridge, which is the numerically robust
information score for this rank-deficient Layer-2 Fisher.
"""

import numpy as np

from l2cross.priors import POP_MU, OBSERVABLES
from l2cross.l1_bridge import make_patient
from l2cross.design import (
    build_mask,
    score_schedule,
    voi_rank,
    minimal_sufficient,
)


# --- Shared concrete setup (mirrors the spec) ---
def _setup():
    patient = make_patient(0.25)
    phi = np.asarray(POP_MU, dtype=np.float64)
    times = np.linspace(0.0, 40.0, 21)
    noise_sigma = np.array([0.03, 0.03, 0.05])
    return patient, phi, times, noise_sigma


# --------------------------------------------------------------------------- #
# build_mask
# --------------------------------------------------------------------------- #
def test_build_mask_shape_and_columns():
    _, _, times, _ = _setup()
    mask = build_mask(times, ("cord_CSA", "SNAP"))
    assert mask.shape == (len(times), 3)
    assert mask.dtype == bool
    # cord_CSA is column 0, QSM column 1, SNAP column 2.
    assert np.all(mask[:, 0])            # cord_CSA selected at every time
    assert not np.any(mask[:, 1])        # QSM not selected
    assert np.all(mask[:, 2])            # SNAP selected at every time


def test_build_mask_single_and_full():
    _, _, times, _ = _setup()
    m1 = build_mask(times, ("QSM",))
    assert np.all(m1[:, 1]) and not np.any(m1[:, 0]) and not np.any(m1[:, 2])

    mfull = build_mask(times, tuple(OBSERVABLES))
    assert np.all(mfull)


# --------------------------------------------------------------------------- #
# score_schedule
# --------------------------------------------------------------------------- #
def test_score_schedule_keys_and_count():
    patient, phi, times, noise_sigma = _setup()
    mask = build_mask(times, ("cord_CSA", "SNAP"))
    s = score_schedule(phi, patient, times, mask, noise_sigma)
    assert set(s.keys()) == {"logdet", "total_crlb", "n_measurements"}
    assert s["n_measurements"] == len(times) * 2
    assert np.isfinite(s["logdet"])
    assert np.isfinite(s["total_crlb"]) and s["total_crlb"] > 0.0


def test_information_monotonicity():
    """Adding a modality never reduces information (D-optimality)."""
    patient, phi, times, noise_sigma = _setup()
    ridge = 1e-3

    full = build_mask(times, ("cord_CSA", "QSM", "SNAP"))
    partial = build_mask(times, ("cord_CSA", "SNAP"))

    s_full = score_schedule(phi, patient, times, full, noise_sigma, ridge=ridge)
    s_partial = score_schedule(phi, patient, times, partial, noise_sigma, ridge=ridge)

    assert s_full["logdet"] >= s_partial["logdet"] - 1e-6


def test_qsm_unique_dentate_information():
    """Dropping QSM strictly reduces logdet by a non-trivial margin.

    QSM is the only observable that loads on the dentate-nucleus direction, so
    a schedule without it loses that dimension of information.
    """
    patient, phi, times, noise_sigma = _setup()
    ridge = 1e-3

    full = build_mask(times, ("cord_CSA", "QSM", "SNAP"))
    no_qsm = build_mask(times, ("cord_CSA", "SNAP"))

    s_full = score_schedule(phi, patient, times, full, noise_sigma, ridge=ridge)
    s_no_qsm = score_schedule(phi, patient, times, no_qsm, noise_sigma, ridge=ridge)

    # Strictly less, by more than a floating-point margin.
    assert s_no_qsm["logdet"] < s_full["logdet"] - 1.0


# --------------------------------------------------------------------------- #
# voi_rank
# --------------------------------------------------------------------------- #
def test_voi_rank_respects_budget_and_sorted():
    patient, phi, times, noise_sigma = _setup()
    dense = times
    sparse = np.linspace(0.0, 40.0, 6)
    cadences = [dense, sparse]
    subsets = [
        ("cord_CSA",),
        ("cord_CSA", "SNAP"),
        ("cord_CSA", "QSM"),
        ("cord_CSA", "QSM", "SNAP"),
    ]
    budget = 40  # excludes the dense full-3-obs design (21*3 = 63)

    ranking = voi_rank(phi, patient, subsets, cadences, noise_sigma, budget)

    assert len(ranking) > 0
    # Budget respected.
    for r in ranking:
        assert r["n_measurements"] <= budget
    # The dense full design (63 measurements) must be excluded.
    for r in ranking:
        assert not (r["cadence_size"] == len(dense) and len(r["observables"]) == 3)
    # Sorted by descending logdet.
    logdets = [r["logdet"] for r in ranking]
    assert logdets == sorted(logdets, reverse=True)
    # Entry schema.
    for r in ranking:
        assert set(r.keys()) == {
            "observables",
            "cadence_size",
            "n_measurements",
            "logdet",
            "total_crlb",
        }


def test_voi_rank_full_budget_includes_dense_full():
    patient, phi, times, noise_sigma = _setup()
    cadences = [times]
    subsets = [("cord_CSA", "QSM", "SNAP")]
    ranking = voi_rank(phi, patient, subsets, cadences, noise_sigma, budget=1000)
    assert len(ranking) == 1
    assert ranking[0]["n_measurements"] == len(times) * 3


# --------------------------------------------------------------------------- #
# minimal_sufficient
# --------------------------------------------------------------------------- #
def test_minimal_sufficient_cheaper_than_full_dense():
    patient, phi, times, noise_sigma = _setup()
    dense = times
    sparse = np.linspace(0.0, 40.0, 6)
    cadences = [dense, sparse]
    subsets = [
        ("cord_CSA",),
        ("cord_CSA", "SNAP"),
        ("cord_CSA", "QSM"),
        ("cord_CSA", "QSM", "SNAP"),
    ]
    ranking = voi_rank(phi, patient, subsets, cadences, noise_sigma, budget=1000)

    full_dense_n = len(dense) * 3
    rec = minimal_sufficient(ranking, logdet_tol=0.95)

    # It is a valid entry from the ranking.
    assert rec in ranking
    # And it does not exceed the full-dense cost (ideally cheaper).
    assert rec["n_measurements"] <= full_dense_n


def test_minimal_sufficient_returns_valid_entry():
    patient, phi, times, noise_sigma = _setup()
    cadences = [times, np.linspace(0.0, 40.0, 6)]
    subsets = [("cord_CSA", "QSM", "SNAP"), ("cord_CSA", "SNAP")]
    ranking = voi_rank(phi, patient, subsets, cadences, noise_sigma, budget=1000)
    rec = minimal_sufficient(ranking)
    assert rec in ranking
