"""Experiment E3 - robustness to model misspecification.

The inference model assumes a constant quasi-steady-state bioenergetic input per
patient. Data are generated twice from the same phi and patient: once with that
model (well-specified arm) and once with a forward model in which reserve and ROS
drift over time (misspecified arm, `observe_drift`, defined in this file). Both
arms are fitted with the same `fit_map`; the experiment reports bias, RMSE and
95% Laplace-CI coverage for log_r0 and t0 in the two arms.

Run:
    python -m l2cross.experiments.e3_misspec
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from l2cross import priors
from l2cross.generator import full_schedule, generate_dataset
from l2cross.inference import fit_map
from l2cross.l1_bridge import make_patient
from l2cross.model import GAMMA, LINK, SIG_S, _DT, _sigmoid


# ----------------------------------------------------------------------------
# Misspecified (drifting-L1) data generator -- lives HERE, not in model.py.
# ----------------------------------------------------------------------------
def _simulate_damage_drift(phi, patient, times, drift):
    """RK4 damage integration with TIME-VARYING reserve rho(t) and ROS R(t).

    Mirrors `model.simulate_damage` exactly, except that the L1 constants drift:

        rho(t) = rho0 * (1 - drift * t)   (clipped >= 0.05)
        R(t)   = R0   * (1 + drift * t)

    Both the base damage rate and the per-tissue vulnerability weight w depend
    on (rho, R), so we recompute them at every RK4 sub-step from rho(t), R(t).
    Returns D of shape (T, 3) in tissue order ["cord", "dentate", "nerve"].
    """
    times = jnp.asarray(times, dtype=jnp.float32)

    log_r0 = phi[0]
    theta = jnp.stack([phi[1], phi[2], phi[3]])  # (3,)
    t0 = phi[4]
    r0 = jnp.exp(log_r0)

    rho0 = float(patient.rho)
    R0 = float(patient.R)

    def rho_of_t(t):
        return jnp.clip(rho0 * (1.0 - drift * t), 0.05, 1.0)

    def R_of_t(t):
        return R0 * (1.0 + drift * t)

    def rate(t):
        """Per-tissue base*onset rate at time t with drifting rho, R."""
        rho_t = rho_of_t(t)
        R_t = R_of_t(t)
        rho_stress = 1.0 - rho_t
        w = _sigmoid((rho_stress - theta) / SIG_S)          # (3,)
        base = r0 * w * (1.0 - rho_t) * (1.0 + GAMMA * R_t)  # (3,)
        onset = _sigmoid((t - t0) / 0.5)
        return base * onset                                  # (3,)

    def deriv(t, D):
        return rate(t) * (1.0 - D)

    n_static = int(jnp.ceil(float(jnp.max(times)) / _DT)) + 1
    n_static = max(n_static, 1)
    grid = jnp.arange(n_static + 1, dtype=jnp.float32) * _DT

    def step(D, t):
        k1 = deriv(t, D)
        k2 = deriv(t + 0.5 * _DT, D + 0.5 * _DT * k1)
        k3 = deriv(t + 0.5 * _DT, D + 0.5 * _DT * k2)
        k4 = deriv(t + _DT, D + _DT * k3)
        D_next = D + (_DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        D_next = jnp.clip(D_next, 0.0, 1.0)
        return D_next, D_next

    D0 = jnp.zeros(3)
    _, D_traj = jax.lax.scan(step, D0, grid[:-1])
    D_full = jnp.concatenate([D0[None, :], D_traj], axis=0)

    D_cord = jnp.interp(times, grid, D_full[:, 0])
    D_dent = jnp.interp(times, grid, D_full[:, 1])
    D_nerve = jnp.interp(times, grid, D_full[:, 2])
    D = jnp.stack([D_cord, D_dent, D_nerve], axis=1)
    return jnp.clip(D, 0.0, 1.0)


def observe_drift(phi, patient, times, drift=0.01):
    """Misspecified forward: drifting-L1 damage, same LINK observation maps.

    Same observable order as `model.observe`: ["cord_CSA", "QSM", "SNAP"].
    """
    D = _simulate_damage_drift(phi, patient, times, drift)
    D_cord = D[:, 0]
    D_dent = D[:, 1]
    D_nerve = D[:, 2]

    Fe = float(patient.Fe)

    cord_CSA = LINK["a_c"] - LINK["b_c"] * D_cord
    SNAP = LINK["a_s"] * jnp.exp(-LINK["b_s"] * D_nerve)
    QSM = LINK["a_q"] * Fe * (D_dent / (1.0 + (D_dent / LINK["Dstar"]) ** 2))
    return jnp.stack([cord_CSA, QSM, SNAP], axis=1)  # (T, 3)


def _generate_dataset_drift(phi, patient, schedule, noise_sigma, key, drift):
    """Like generator.generate_dataset but with the drifting forward."""
    times = np.asarray(schedule.times, dtype=float)
    T = times.shape[0]
    noise_sigma = jnp.asarray(noise_sigma, dtype=jnp.float32).reshape(-1)
    Y = observe_drift(phi, patient, jnp.asarray(times, dtype=jnp.float32), drift)
    noise = jax.random.normal(key, (T, 3)) * noise_sigma[None, :]
    Y_noisy = np.asarray(Y + noise, dtype=float)
    mask = np.asarray(schedule.mask, dtype=bool)
    return np.where(mask, Y_noisy, np.nan)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def _metrics(phi_true, phi_hat, cov, idx):
    """bias, RMSE, 95% Laplace-CI coverage for parameter `idx` across cohort.

    phi_true, phi_hat : (N, 5); cov : (N, 5, 5).
    """
    z95 = 1.959963984540054  # 0.975 normal quantile
    err = phi_hat[:, idx] - phi_true[:, idx]
    bias = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    sd = np.sqrt(np.clip(cov[:, idx, idx], 0.0, np.inf))
    half = z95 * sd
    covered = np.abs(err) <= half
    coverage = float(np.mean(covered))
    return bias, rmse, coverage


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def _make_figure(stats, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = ["log_r0", "t0"]
    well_bias = [abs(stats["well"]["log_r0"][0]), abs(stats["well"]["t0"][0])]
    miss_bias = [abs(stats["miss"]["log_r0"][0]), abs(stats["miss"]["t0"][0])]
    well_cov = [stats["well"]["log_r0"][2], stats["well"]["t0"][2]]
    miss_cov = [stats["miss"]["log_r0"][2], stats["miss"]["t0"][2]]

    x = np.arange(len(params))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    ax1.bar(x - width / 2, well_bias, width, label="well-specified", color="#4C72B0")
    ax1.bar(x + width / 2, miss_bias, width, label="misspecified", color="#C44E52")
    ax1.set_xticks(x)
    ax1.set_xticklabels(params)
    ax1.set_ylabel("|bias|")
    ax1.set_title("Absolute bias")
    ax1.legend()

    ax2.bar(x - width / 2, well_cov, width, label="well-specified", color="#4C72B0")
    ax2.bar(x + width / 2, miss_cov, width, label="misspecified", color="#C44E52")
    ax2.axhline(0.95, ls="--", color="k", lw=1, label="nominal 0.95")
    ax2.set_xticks(x)
    ax2.set_xticklabels(params)
    ax2.set_ylabel("95% CI coverage")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Laplace CI coverage")
    ax2.legend()

    fig.suptitle(
        "E3: robustness to L1-drift misspecification (constant-QSS inference)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    N = 12
    times = np.linspace(0.0, 40.0, 15)
    sigma = np.array([0.02, 0.02, 0.04])
    drift = 0.01
    F_range = (0.1, 0.5)

    schedule = full_schedule(times)
    key = jax.random.PRNGKey(20260623)

    print("=" * 70)
    print("E3  robustness to misspecification (drifting-L1 vs constant-QSS)")
    print(f"N={N} patients, T={len(times)} times in [0,40] yr, "
          f"sigma={sigma.tolist()}, drift={drift}")
    print("=" * 70)

    # Pre-sample paired (patient, phi_true, data both arms) records.
    records = []
    for i in range(N):
        key, k_pat, k_phi, k_well, k_miss = jax.random.split(key, 5)
        F = float(jax.random.uniform(k_pat, (), minval=F_range[0], maxval=F_range[1]))
        patient = make_patient(F)
        phi_true = priors.sample_prior(k_phi)
        Y_well = generate_dataset(phi_true, patient, schedule, sigma, k_well)
        Y_miss = _generate_dataset_drift(
            phi_true, patient, schedule, sigma, k_miss, drift
        )
        records.append(
            dict(patient=patient, phi_true=np.asarray(phi_true, dtype=np.float64),
                 Y_well=Y_well, Y_miss=Y_miss, F=F)
        )

    phi_true_all = np.stack([r["phi_true"] for r in records])  # (N,5)

    well_hat = np.zeros((N, 5))
    well_cov = np.zeros((N, 5, 5))
    miss_hat = np.zeros((N, 5))
    miss_cov = np.zeros((N, 5, 5))

    t_start = time.time()
    n_fits = 2 * N
    done = 0
    for i, r in enumerate(records):
        ph, cv = fit_map(r["patient"], schedule, r["Y_well"], sigma)
        well_hat[i] = ph
        well_cov[i] = cv
        done += 1
        print(f"  [{done:2d}/{n_fits}] well-specified  patient {i+1:2d} "
              f"(F={r['F']:.3f})  elapsed {time.time()-t_start:6.1f}s", flush=True)

        ph, cv = fit_map(r["patient"], schedule, r["Y_miss"], sigma)
        miss_hat[i] = ph
        miss_cov[i] = cv
        done += 1
        print(f"  [{done:2d}/{n_fits}] misspecified    patient {i+1:2d} "
              f"(F={r['F']:.3f})  elapsed {time.time()-t_start:6.1f}s", flush=True)

    # Metrics for well-identified params: log_r0 (idx 0), t0 (idx 4).
    idx_map = {"log_r0": 0, "t0": 4}
    stats = {"well": {}, "miss": {}}
    for name, idx in idx_map.items():
        stats["well"][name] = _metrics(phi_true_all, well_hat, well_cov, idx)
        stats["miss"][name] = _metrics(phi_true_all, miss_hat, miss_cov, idx)

    # ---- Table ----
    print()
    print("-" * 70)
    print("RESULTS  (well-identified parameters; coverage nominal = 0.95)")
    print("-" * 70)
    header = f"{'param':10s} {'arm':14s} {'bias':>9s} {'RMSE':>9s} {'cover95':>9s}"
    print(header)
    print("-" * 70)
    for name in idx_map:
        for arm, label in (("well", "well-specified"), ("miss", "misspecified")):
            b, rm, cov = stats[arm][name]
            print(f"{name:10s} {label:14s} {b:9.4f} {rm:9.4f} {cov:9.3f}")
    print("-" * 70)

    # ---- Figure ----
    fig_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "figures"
    )
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, "e3_misspec.png")
    _make_figure(stats, out_path)
    print(f"\nFigure saved -> {out_path}")

    # ---- Summary ----
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name in idx_map:
        bw, rmw, cw = stats["well"][name]
        bm, rmm, cm = stats["miss"][name]
        d_bias = abs(bm) - abs(bw)
        d_cov = cm - cw
        rmse_ratio = (rmm / rmw) if rmw > 0 else float("nan")
        print(f"{name}: |bias| {abs(bw):.4f} -> {abs(bm):.4f} "
              f"(Delta {d_bias:+.4f}); RMSE x{rmse_ratio:.2f}; "
              f"coverage {cw:.2f} -> {cm:.2f} (Delta {d_cov:+.2f})")
    print()
    print(f"Tested drift rate: {drift}/yr.")
    print("=" * 70)

    return stats


if __name__ == "__main__":
    main()
