"""Smoke test for experiment E3 (does not run the full ~15 min main())."""

import numpy as np


def test_e3_import_and_main_callable():
    from l2cross.experiments import e3_misspec

    assert callable(e3_misspec.main)
    # The misspecified forward must be importable and exposed.
    assert callable(e3_misspec.observe_drift)


def test_observe_drift_shape_and_finiteness():
    """observe_drift should return a finite (T,3) array like model.observe."""
    from l2cross.experiments import e3_misspec
    from l2cross.l1_bridge import make_patient
    from l2cross import priors

    patient = make_patient(0.3)
    phi = np.asarray(priors.POP_MU, dtype=float)
    times = np.linspace(0.0, 40.0, 15)
    Y = np.asarray(e3_misspec.observe_drift(phi, patient, times, drift=0.01))
    assert Y.shape == (15, 3)
    assert np.all(np.isfinite(Y))
