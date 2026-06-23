"""Layer-2 cross-scale damage + observation model (fully JAX, differentiable in phi).

Latent phi = [log_r0, theta_cord, theta_dentate, theta_nerve, t0] (see priors.py).
A Patient (from l1_bridge) supplies the L1-derived constants rho, Fe, R, which
are treated as fixed numeric constants (NOT differentiated).

Tissues order : ["cord", "dentate", "nerve"]
Observables   : ["cord_CSA", "QSM", "SNAP"]
"""

import jax
import jax.numpy as jnp

# --- Structural link constants (phenomenological L2 -> observable maps) ---
LINK = {
    "a_c": 1.0,   # cord_CSA = a_c - b_c * D_cord
    "b_c": 0.6,
    "a_s": 1.0,   # SNAP = a_s * exp(-b_s * D_nerve)
    "b_s": 3.0,
    "a_q": 1.0,   # QSM = a_q * Fe * D/(1 + (D/Dstar)^2)
    "Dstar": 0.4,
}

GAMMA = 0.5      # ROS feedback gain on the damage rate
SIG_S = 0.05     # sigmoid sharpness for tissue vulnerability weighting

# Internal RK4 step (years) for the damage ODE.
_DT = 0.1


def _sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))


def simulate_damage(phi, patient, times):
    """Integrate the per-tissue damage ODE and sample at `times`.

    Returns D of shape (T, 3) in tissue order ["cord", "dentate", "nerve"],
    each entry in [0, 1].
    """
    times = jnp.asarray(times, dtype=jnp.float32)

    log_r0 = phi[0]
    theta = jnp.stack([phi[1], phi[2], phi[3]])  # (3,)
    t0 = phi[4]
    r0 = jnp.exp(log_r0)

    # L1-derived constants (plain floats captured into the trace).
    rho = float(patient.rho)
    R = float(patient.R)
    rho_stress = 1.0 - rho  # greater deficit -> more stress

    # Per-tissue vulnerability weight: lower theta_k -> more vulnerable.
    w = _sigmoid((rho_stress - theta) / SIG_S)  # (3,)

    # Constant part of the rate (shared across tissues except w).
    base = r0 * w * (1.0 - rho) * (1.0 + GAMMA * R)  # (3,)

    t_max = jnp.max(times)
    # Fixed number of internal steps; pad so we always cover t_max.
    n_steps = jnp.maximum(1, jnp.ceil(t_max / _DT).astype(jnp.int32))
    # Use a static upper bound for the scan length so it is traceable.
    # We rely on times being concrete at call time for n_steps; lax.scan needs a
    # static length, so derive it from the python-visible max of `times`.
    n_static = int(jnp.ceil(float(jnp.max(times)) / _DT)) + 1
    n_static = max(n_static, 1)

    grid = jnp.arange(n_static + 1, dtype=jnp.float32) * _DT  # (n_static+1,)

    def deriv(t, D):
        onset = _sigmoid((t - t0) / 0.5)
        return base * onset * (1.0 - D)

    def step(D, t):
        k1 = deriv(t, D)
        k2 = deriv(t + 0.5 * _DT, D + 0.5 * _DT * k1)
        k3 = deriv(t + 0.5 * _DT, D + 0.5 * _DT * k2)
        k4 = deriv(t + _DT, D + _DT * k3)
        D_next = D + (_DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        D_next = jnp.clip(D_next, 0.0, 1.0)
        return D_next, D_next

    D0 = jnp.zeros(3)
    _, D_traj = jax.lax.scan(step, D0, grid[:-1])  # (n_static, 3)
    # Prepend the initial state so trajectory aligns with `grid`.
    D_full = jnp.concatenate([D0[None, :], D_traj], axis=0)  # (n_static+1, 3)

    # Sample each tissue at requested times via linear interpolation.
    D_cord = jnp.interp(times, grid, D_full[:, 0])
    D_dent = jnp.interp(times, grid, D_full[:, 1])
    D_nerve = jnp.interp(times, grid, D_full[:, 2])
    D = jnp.stack([D_cord, D_dent, D_nerve], axis=1)  # (T, 3)
    return jnp.clip(D, 0.0, 1.0)


def observe(phi, patient, times):
    """Map latent phi -> observables Y of shape (T, 3).

    Observables order: ["cord_CSA", "QSM", "SNAP"].
    """
    D = simulate_damage(phi, patient, times)
    D_cord = D[:, 0]
    D_dent = D[:, 1]
    D_nerve = D[:, 2]

    Fe = float(patient.Fe)

    cord_CSA = LINK["a_c"] - LINK["b_c"] * D_cord
    SNAP = LINK["a_s"] * jnp.exp(-LINK["b_s"] * D_nerve)
    QSM = LINK["a_q"] * Fe * (D_dent / (1.0 + (D_dent / LINK["Dstar"]) ** 2))

    return jnp.stack([cord_CSA, QSM, SNAP], axis=1)  # (T, 3)


# Alias used by inference / Fisher machinery.
forward = observe
