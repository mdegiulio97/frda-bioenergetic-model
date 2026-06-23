import jax
import jax.numpy as jnp

from l2cross.l1_bridge import make_patient
from l2cross.model import forward, observe, simulate_damage


def _damaged_patient():
    return make_patient(0.2)


def _early_onset_phi():
    # log_r0 -> r0 ~ 0.4/yr (fast), low thresholds (vulnerable), early onset t0=2yr.
    return jnp.array([jnp.log(0.4), 0.1, 0.1, 0.1, 2.0])


def test_cord_csa_and_snap_monotone_decreasing():
    patient = _damaged_patient()
    phi = _early_onset_phi()
    times = jnp.linspace(0.0, 40.0, 80)
    Y = observe(phi, patient, times)
    cord_CSA = Y[:, 0]
    SNAP = Y[:, 2]
    tol = 1e-5
    assert jnp.all(jnp.diff(cord_CSA) <= tol)
    assert jnp.all(jnp.diff(SNAP) <= tol)


def test_qsm_non_monotone():
    patient = _damaged_patient()
    phi = _early_onset_phi()
    times = jnp.linspace(0.0, 60.0, 120)
    Y = observe(phi, patient, times)
    QSM = Y[:, 1]
    argmax = int(jnp.argmax(QSM))
    # Maximum is interior (iron paradox: rises then falls), not at the last point.
    assert 0 < argmax < len(QSM) - 1


def test_differentiable_finite_grad():
    patient = _damaged_patient()
    phi = _early_onset_phi()
    times = jnp.linspace(0.0, 40.0, 40)

    def loss(p):
        return jnp.sum(forward(p, patient, times))

    g = jax.grad(loss)(phi)
    assert g.shape == (5,)
    assert jnp.all(jnp.isfinite(g))


def test_damage_bounded():
    patient = _damaged_patient()
    phi = _early_onset_phi()
    times = jnp.linspace(0.0, 60.0, 120)
    D = simulate_damage(phi, patient, times)
    assert jnp.all(D >= 0.0)
    assert jnp.all(D <= 1.0 + 1e-6)
