from l2cross.l1_bridge import make_patient


def test_frda_signature_lower_F():
    """Lower F -> less reserve (rho), more iron (Fe), more ROS (R)."""
    healthy = make_patient(1.0)
    frda = make_patient(0.2)

    assert frda.rho < healthy.rho
    assert frda.Fe > healthy.Fe
    assert frda.R > healthy.R


def test_rho_in_unit_interval():
    for F in (1.0, 0.5, 0.2):
        p = make_patient(F)
        assert 0.0 < p.rho <= 1.0


def test_healthy_rho_is_one():
    healthy = make_patient(1.0)
    assert abs(healthy.rho - 1.0) < 1e-6
