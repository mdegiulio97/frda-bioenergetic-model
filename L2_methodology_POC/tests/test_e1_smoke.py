"""Smoke test for experiment E1.

Only verifies that the module imports and exposes a callable ``main``. The full
``main()`` is intentionally NOT executed here: each fit_map is ~40 s, so a real
run is far too slow for the test suite.
"""


def test_e1_importable_and_main_callable():
    from l2cross.experiments import e1_recovery

    assert hasattr(e1_recovery, "main")
    assert callable(e1_recovery.main)
