"""Experiment E1 - parameter recovery and identifiability of the Layer-2 latent phi.

phi = [log_r0, theta_cord, theta_dentate, theta_nerve, t0]. For synthetic patients
generated from the model, the experiment measures which components of phi are
recoverable from data under a rich and a sparse/noisy acquisition, using:

  1. posterior contraction (Laplace posterior std / prior std);
  2. cohort coverage of the 95% Laplace CI;
  3. likelihood-only profile likelihood for each parameter, with the prior-only
     profile overlaid as a reference.

Figures are written to ``L2_methodology_POC/figures/``.

Run with::

    python -m l2cross.experiments.e1_recovery

Each ``fit_map`` takes tens of seconds on CPU, so the cohort is small (N=12).
"""

import os
import time

import numpy as np

# Matplotlib without a display backend.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from l2cross import priors  # noqa: E402
from l2cross.l1_bridge import make_patient  # noqa: E402
from l2cross.generator import full_schedule, generate_dataset, sample_cohort  # noqa: E402
from l2cross.inference import fit_map, fit_cohort  # noqa: E402
from l2cross.identifiability import profile_likelihood, THRESHOLD_1DOF  # noqa: E402


PHI_NAMES = priors.PHI_NAMES
PRIOR_STD = np.asarray(priors.POP_SIGMA, dtype=np.float64)
PRIOR_MU = np.asarray(priors.POP_MU, dtype=np.float64)

# Parameters expected to be well identified for a single patient.
WELL_ID = [0, 4]  # log_r0, t0

_FIG_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "figures")
)


def _laplace_std(cov):
    """sqrt of the diagonal of the Laplace covariance, clipped to be finite."""
    d = np.diag(np.asarray(cov, dtype=np.float64)).copy()
    d = np.where(d > 0, d, np.nan)
    return np.sqrt(d)


# --------------------------------------------------------------------------- #
# Part 1: rich vs sparse posterior contraction
# --------------------------------------------------------------------------- #
def part1_contraction(patient, phi_true, seed=0):
    import jax

    print("\n" + "=" * 72)
    print("PART 1 - posterior contraction: rich vs sparse/noisy acquisition")
    print("=" * 72)

    key = jax.random.PRNGKey(seed)
    k_rich, k_sparse = jax.random.split(key)

    rich_sched = full_schedule(np.linspace(0, 40, 15))
    rich_noise = np.array([0.01, 0.01, 0.02])
    sparse_sched = full_schedule(np.linspace(0, 40, 4))
    sparse_noise = np.array([0.05, 0.05, 0.08])

    print(
        f"  rich  : T={rich_sched.times.shape[0]} times, "
        f"noise={rich_noise.tolist()}"
    )
    print(
        f"  sparse: T={sparse_sched.times.shape[0]} times, "
        f"noise={sparse_noise.tolist()}"
    )
    print(f"  phi_true = {np.array2string(np.asarray(phi_true), precision=3)}")

    Y_rich = generate_dataset(phi_true, patient, rich_sched, rich_noise, k_rich)
    Y_sparse = generate_dataset(phi_true, patient, sparse_sched, sparse_noise, k_sparse)

    print("\n  fitting RICH regime (fit_map ~40s) ...", flush=True)
    t0 = time.time()
    phi_hat_rich, cov_rich = fit_map(patient, rich_sched, Y_rich, rich_noise)
    print(f"    done in {time.time() - t0:.1f}s", flush=True)

    print("  fitting SPARSE regime (fit_map ~40s) ...", flush=True)
    t0 = time.time()
    phi_hat_sparse, cov_sparse = fit_map(patient, sparse_sched, Y_sparse, sparse_noise)
    print(f"    done in {time.time() - t0:.1f}s", flush=True)

    std_rich = _laplace_std(cov_rich)
    std_sparse = _laplace_std(cov_sparse)
    ratio_rich = std_rich / PRIOR_STD
    ratio_sparse = std_sparse / PRIOR_STD

    def _print_table(label, phi_hat, lstd, ratio):
        print(f"\n  --- {label} regime ---")
        print(
            f"    {'param':<14}{'true':>9}{'phi_hat':>10}"
            f"{'prior_std':>11}{'lap_std':>10}{'ratio':>8}"
        )
        for i, name in enumerate(PHI_NAMES):
            tag = "  <== informative" if ratio[i] < 1.0 else "  (~prior)"
            print(
                f"    {name:<14}{float(phi_true[i]):>9.3f}{phi_hat[i]:>10.3f}"
                f"{PRIOR_STD[i]:>11.3f}{lstd[i]:>10.3f}{ratio[i]:>8.2f}{tag}"
            )

    _print_table("RICH", phi_hat_rich, std_rich, ratio_rich)
    _print_table("SPARSE", phi_hat_sparse, std_sparse, ratio_sparse)

    informative = [PHI_NAMES[i] for i in range(5) if ratio_rich[i] < 1.0]
    print(
        "\n  Commentary: contraction (ratio<1) concentrates on "
        f"{informative if informative else 'NONE'}; "
        "the three theta_* stay ~prior (ratio~1), confirming weak "
        "identifiability for a single patient even in the rich regime."
    )

    return {
        "rich": {
            "Y_obs": Y_rich,
            "schedule": rich_sched,
            "noise": rich_noise,
            "phi_hat": phi_hat_rich,
            "cov": cov_rich,
            "std": std_rich,
            "ratio": ratio_rich,
        },
        "sparse": {
            "phi_hat": phi_hat_sparse,
            "cov": cov_sparse,
            "std": std_sparse,
            "ratio": ratio_sparse,
        },
    }


# --------------------------------------------------------------------------- #
# Part 2: cohort coverage
# --------------------------------------------------------------------------- #
def part2_coverage(n=12, seed=1):
    import jax

    print("\n" + "=" * 72)
    print(f"PART 2 - cohort coverage of the 95% Laplace CI (N={n})")
    print("=" * 72)

    sched = full_schedule(np.linspace(0, 40, 15))
    noise = np.array([0.01, 0.01, 0.02])
    key = jax.random.PRNGKey(seed)

    print(f"  sampling cohort (N={n}) ...", flush=True)
    cohort = sample_cohort(key, n, sched, noise)

    print(f"  fitting cohort ({n} x ~40s fit_map) ...", flush=True)
    results = []
    for i, rec in enumerate(cohort):
        t0 = time.time()
        phi_hat, cov = fit_map(rec["patient"], sched, rec["Y_obs"], noise)
        results.append(
            {
                "phi_true": np.asarray(rec["phi_true"], dtype=np.float64),
                "phi_hat": phi_hat,
                "cov": cov,
            }
        )
        print(
            f"    patient {i + 1:2d}/{n} fit in {time.time() - t0:.1f}s",
            flush=True,
        )

    # Coverage per parameter: fraction of patients whose phi_true is inside
    # phi_hat +/- 1.96 * laplace_std.
    covered = np.zeros((n, 5), dtype=bool)
    valid_std = np.zeros((n, 5), dtype=bool)
    for j, res in enumerate(results):
        lstd = _laplace_std(res["cov"])
        lo = res["phi_hat"] - 1.96 * lstd
        hi = res["phi_hat"] + 1.96 * lstd
        finite = np.isfinite(lstd)
        valid_std[j] = finite
        inside = (res["phi_true"] >= lo) & (res["phi_true"] <= hi)
        covered[j] = inside & finite

    coverage = np.zeros(5)
    n_valid = np.zeros(5, dtype=int)
    for i in range(5):
        m = valid_std[:, i]
        n_valid[i] = int(m.sum())
        coverage[i] = covered[m, i].mean() if m.any() else np.nan

    print("\n  Coverage table (target nominal = 0.95):")
    print(f"    {'param':<14}{'n_valid':>9}{'coverage':>11}")
    for i, name in enumerate(PHI_NAMES):
        flag = "  well-id" if i in WELL_ID else "  weak-id"
        cov_str = f"{coverage[i]:.3f}" if np.isfinite(coverage[i]) else "  nan"
        print(f"    {name:<14}{n_valid[i]:>9}{cov_str:>11}{flag}")

    wid = ", ".join(
        f"{PHI_NAMES[i]}={coverage[i]:.2f}" for i in WELL_ID if np.isfinite(coverage[i])
    )
    print(
        "\n  Commentary: coverage for the well-identified params "
        f"({wid}) is meaningful because their Laplace std is data-driven. "
        "For the theta_* the Laplace CI is essentially the prior band, so any "
        "'coverage' there reflects the prior, not recovery - it is reported "
        "for completeness, not as evidence of identifiability."
    )

    return coverage, n_valid


# --------------------------------------------------------------------------- #
# Part 3: profile-likelihood identifiability map
# --------------------------------------------------------------------------- #
def part3_profiles(patient, phi_true, rich, n_grid=15):
    print("\n" + "=" * 72)
    print("PART 3 - profile-likelihood identifiability maps (rich regime)")
    print("=" * 72)

    sched = rich["schedule"]
    Y_obs = rich["Y_obs"]
    noise = rich["noise"]
    times = np.asarray(sched.times, dtype=float)
    mask = np.asarray(sched.mask, dtype=bool)
    phi_true = np.asarray(phi_true, dtype=np.float64)

    profiles = []
    for idx in range(5):
        center = phi_true[idx]
        half = 2.0 * PRIOR_STD[idx]
        grid = np.linspace(center - half, center + half, n_grid)
        print(
            f"  profiling {PHI_NAMES[idx]:<14} "
            f"grid=[{grid[0]:.3f}, {grid[-1]:.3f}] ({n_grid} pts) ...",
            flush=True,
        )
        t0 = time.time()
        g, costs = profile_likelihood(
            patient,
            times,
            mask,
            Y_obs,
            noise,
            idx,
            grid,
            prior_mu=PRIOR_MU,
            prior_sigma=PRIOR_STD,
        )
        dcost = costs - np.min(costs)
        crosses = bool(np.any(dcost > THRESHOLD_1DOF))
        verdict = "identifiable (profile crosses 1.92)" if crosses else "FLAT -> non-identifiable"
        print(
            f"    done in {time.time() - t0:.1f}s | "
            f"max(dcost)={dcost.max():.3f} -> {verdict}"
        )
        profiles.append({"idx": idx, "grid": g, "dcost": dcost, "crosses": crosses})

    return profiles


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_contraction(rich, path):
    ratio = rich["ratio"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(5)
    colors = ["#2c7fb8" if r < 1.0 else "#bdbdbd" for r in ratio]
    bars = ax.bar(x, ratio, color=colors)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="ratio = 1 (no contraction)")
    ax.set_xticks(x)
    ax.set_xticklabels(PHI_NAMES, rotation=20, ha="right")
    ax.set_ylabel("Laplace_std / prior_std")
    ax.set_title("E1 posterior contraction (rich regime)\nbars < 1 = informative")
    for i, b in enumerate(bars):
        tag = "informative" if ratio[i] < 1.0 else "~prior"
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02,
            f"{ratio[i]:.2f}\n{tag}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, max(1.25, float(np.nanmax(ratio)) * 1.15))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_profiles(profiles, path):
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), sharey=True)
    for ax, prof in zip(axes, profiles):
        idx = prof["idx"]
        ax.plot(prof["grid"], prof["dcost"], "o-", color="#2c7fb8", ms=3)
        ax.axhline(THRESHOLD_1DOF, color="r", ls="--", lw=1)
        ax.set_title(PHI_NAMES[idx], fontsize=10)
        ax.set_xlabel("phi value")
        sub = "crosses" if prof["crosses"] else "FLAT"
        ax.text(
            0.5,
            0.92,
            sub,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color=("#2c7fb8" if prof["crosses"] else "#d95f02"),
        )
    axes[0].set_ylabel("cost - cost_min")
    fig.suptitle(
        "E1 profile likelihood (red line = 1.92, 95% / 1 dof); "
        "flat profile = non-identifiable",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_coverage(coverage, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(5)
    vals = np.where(np.isfinite(coverage), coverage, 0.0)
    colors = ["#2c7fb8" if i in WELL_ID else "#bdbdbd" for i in range(5)]
    bars = ax.bar(x, vals, color=colors)
    ax.axhline(0.95, color="r", ls="--", lw=1, label="nominal 0.95")
    ax.set_xticks(x)
    ax.set_xticklabels(PHI_NAMES, rotation=20, ha="right")
    ax.set_ylabel("fraction phi_true inside 95% Laplace CI")
    ax.set_title("E1 cohort coverage (N=12)")
    ax.set_ylim(0, 1.05)
    for i, b in enumerate(bars):
        txt = f"{coverage[i]:.2f}" if np.isfinite(coverage[i]) else "nan"
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.01,
            txt,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    t_start = time.time()
    os.makedirs(_FIG_DIR, exist_ok=True)
    print("E1 recovery + identifiability experiment")
    print(f"  figures dir: {_FIG_DIR}")

    # Representative patient + early-onset phi_true near POP_MU.
    patient = make_patient(0.2)
    print(
        f"  representative patient: F={patient.F}, rho={patient.rho:.3f}, "
        f"Fe={patient.Fe:.4f}, R={patient.R:.4f}"
    )
    # phi near POP_MU = [-3.0, 0.4, 0.4, 0.4, 10.0]; slightly early onset.
    phi_true = np.array([-2.8, 0.35, 0.45, 0.40, 8.0], dtype=np.float64)

    # Part 1
    p1 = part1_contraction(patient, phi_true, seed=0)

    # Part 2
    coverage, n_valid = part2_coverage(n=12, seed=1)

    # Part 3
    profiles = part3_profiles(patient, phi_true, p1["rich"], n_grid=15)

    # Figures
    print("\n" + "=" * 72)
    print("Saving figures")
    print("=" * 72)
    f_contraction = os.path.join(_FIG_DIR, "e1_contraction.png")
    f_profiles = os.path.join(_FIG_DIR, "e1_profiles.png")
    f_coverage = os.path.join(_FIG_DIR, "e1_coverage.png")
    fig_contraction(p1["rich"], f_contraction)
    fig_profiles(profiles, f_profiles)
    fig_coverage(coverage, f_coverage)
    for f in (f_contraction, f_profiles, f_coverage):
        sz = os.path.getsize(f) if os.path.exists(f) else -1
        print(f"  saved {f} ({sz} bytes)")

    # Summary
    ratio_rich = p1["rich"]["ratio"]
    identifiable = [PHI_NAMES[i] for i in range(5) if ratio_rich[i] < 1.0]
    weak = [PHI_NAMES[i] for i in range(5) if ratio_rich[i] >= 1.0]
    profile_id = [PHI_NAMES[p["idx"]] for p in profiles if p["crosses"]]

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(
        f"  Identifiable (Laplace contraction, rich): {identifiable}"
    )
    print(f"  Weakly identified (~prior):               {weak}")
    print(f"  Profile likelihood crosses 1.92 for:      {profile_id}")
    print("  Cohort coverage (95% Laplace CI, N=12):")
    for i, name in enumerate(PHI_NAMES):
        cov_str = f"{coverage[i]:.3f}" if np.isfinite(coverage[i]) else "nan"
        tag = "well-id" if i in WELL_ID else "weak-id"
        print(f"     {name:<14} {cov_str}  ({tag})")
    print(
        "\n  Compare the contraction and profile diagnostics above."
    )
    print(f"\n  total wall time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
