"""Smoke test for the E2 VoI experiment.

Imports the experiment module and checks that `main` is callable. Does NOT run
the full experiment (which fits Fisher matrices and writes figures).
"""


def test_e2_imports_and_main_callable():
    from l2cross.experiments import e2_voi

    assert hasattr(e2_voi, "main")
    assert callable(e2_voi.main)
