"""
Parameter uncertainty & practical identifiability for the calibrated FRDA twin.

Two standard analyses for ODE-model calibration rigor:

  (A) Monte-Carlo bootstrap over MEASUREMENT uncertainty — resample each calibration target from its
      reported confidence interval, refit, and collect the fitted-parameter distributions
      (mean ± SD, 95% CI, correlations). Uses the real reported CIs.

  (B) Profile likelihood — for each parameter, fix it across a range, re-optimize the rest, and trace
      the cost. A parabola crossing the 95% threshold on both sides = practically identifiable;
      a flat profile = non-identifiable ("sloppy"). This is the Raue et al. standard.

It reports parameter identifiability and uncertainty,
and makes explicit which parameters additional data (esp. the missing Seahorse OCR)
would most constrain. Run: python uncertainty.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from fit_real_data import FREE, TARGETS, fold_changes, REG

np.random.seed(0)  # reproducible (Date/random caveats are JS-only; numpy RNG is fine here)

KEYS = list(TARGETS)
NOMINAL_T = np.array([TARGETS[k][0] for k in KEYS])
WEIGHTS = np.array([TARGETS[k][1] for k in KEYS])
NOMINAL_P = np.array([FREE[n][0] for n in FREE])
LO = np.array([FREE[n][1] for n in FREE])
HI = np.array([FREE[n][2] for n in FREE])
PNAMES = list(FREE)

# 1-sigma measurement uncertainty from reported 95% CIs (half-width / 1.96):
#   iron  6.5 [5.8-7.2]   ROS 2.05 [1.95-2.15]   ATP 0.75 [0.70-0.80]   aconitase 0.59 [0.50-0.68]
SIGMA_T = np.array([
    (7.2 - 5.8) / 2 / 1.96,    # mito_iron
    (2.15 - 1.95) / 2 / 1.96,  # ROS
    (0.68 - 0.50) / 2 / 1.96,  # aconitase
    (0.80 - 0.70) / 2 / 1.96,  # ATP
])


def fit_targets(tv, fixed=None):
    """Fit free parameters to target vector tv (aligned with KEYS); optionally fix some by index."""
    fixed = fixed or {}
    free_idx = [i for i in range(len(FREE)) if i not in fixed]

    def expand(z):
        theta = NOMINAL_P.copy()
        for k, i in enumerate(free_idx):
            theta[i] = z[k]
        for i, v in fixed.items():
            theta[i] = v
        return theta

    def res(z):
        theta = expand(z)
        fc = fold_changes(theta)
        r = [WEIGHTS[ii] * (fc[KEYS[ii]] - tv[ii]) / tv[ii] for ii in range(len(KEYS))]
        r += list(REG * (z - NOMINAL_P[free_idx]) / NOMINAL_P[free_idx])
        return np.array(r)

    sol = least_squares(res, NOMINAL_P[free_idx], bounds=(LO[free_idx], HI[free_idx]),
                        method="trf", xtol=1e-12, ftol=1e-12)
    return expand(sol.x), float(sol.cost)


def bootstrap(n=300):
    theta_hat, _ = fit_targets(NOMINAL_T)
    samples = []
    for _ in range(n):
        tv = np.clip(np.random.normal(NOMINAL_T, SIGMA_T), 1e-3, None)
        theta, _ = fit_targets(tv)
        samples.append(theta)
    S = np.array(samples)
    mean, sd = S.mean(0), S.std(0)
    lo, hi = np.percentile(S, 2.5, 0), np.percentile(S, 97.5, 0)
    corr = np.corrcoef(S.T)
    return theta_hat, mean, sd, lo, hi, corr, S


def profile_likelihood(npts=21):
    theta_hat, cost0 = fit_targets(NOMINAL_T)
    thr = cost0 + 1.92  # 95% for 1 dof: Δ(0.5·χ²) = 3.84/2
    profiles = {}
    for j, name in enumerate(PNAMES):
        grid = np.linspace(max(theta_hat[j] * 0.4, LO[j]), min(theta_hat[j] * 2.0, HI[j]), npts)
        costs = []
        for val in grid:
            _, c = fit_targets(NOMINAL_T, fixed={j: val})
            costs.append(c)
        profiles[name] = (grid, np.array(costs))
    return theta_hat, cost0, thr, profiles


def figures(boot, prof):
    theta_hat, mean, sd, lo, hi, corr, S = boot
    _, cost0, thr, profiles = prof

    # Fig 5 — profile likelihood
    fig, axes = plt.subplots(1, len(PNAMES), figsize=(15, 3.0), sharey=True)
    for ax, name in zip(axes, PNAMES):
        g, c = profiles[name]
        ax.plot(g, c, "-o", ms=3, lw=1.6)
        ax.axhline(thr, color="tab:red", ls=":", lw=1.2)
        ax.axvline(theta_hat[PNAMES.index(name)], color="tab:green", ls="--", lw=1.0)
        ax.set_title(name, fontsize=10); ax.set_xlabel("value"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("cost (0.5·Σr²)")
    fig.suptitle("Fig 5 — profile likelihood (red = 95% threshold; identifiable = profile rises above it both sides)",
                 fontsize=10, y=1.05)
    fig.tight_layout(); fig.savefig("figures/fig5_profile_likelihood.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 6 — bootstrap parameter correlation heatmap
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(PNAMES))); ax.set_xticklabels(PNAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(PNAMES))); ax.set_yticklabels(PNAMES, fontsize=8)
    for i in range(len(PNAMES)):
        for k in range(len(PNAMES)):
            ax.text(k, i, f"{corr[i,k]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr[i,k]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.85, label="correlation")
    ax.set_title("Fig 6 — bootstrap parameter correlations")
    fig.tight_layout(); fig.savefig("figures/fig6_param_correlations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def report():
    boot = bootstrap()
    prof = profile_likelihood()
    theta_hat, mean, sd, lo, hi, corr, S = boot
    _, cost0, thr, profiles = prof

    print("Parameter uncertainty (bootstrap over measurement CIs, N=300) + identifiability (profile):\n")
    print(f"{'param':<8}{'fit':>8}{'boot mean':>11}{'± SD':>8}{'95% CI':>18}   identifiable?")
    print("-" * 72)
    for j, name in enumerate(PNAMES):
        g, c = profiles[name]
        rises_left = c[0] > thr
        rises_right = c[-1] > thr
        ident = "yes" if (rises_left and rises_right) else ("one-sided" if (rises_left or rises_right) else "NO (sloppy)")
        ci = f"[{lo[j]:.2f}, {hi[j]:.2f}]"
        print(f"{name:<8}{theta_hat[j]:>8.3f}{mean[j]:>11.3f}{sd[j]:>8.3f}{ci:>18}   {ident}")

    print("\nStrongest parameter correlations (|r|>0.6):")
    found = False
    for i in range(len(PNAMES)):
        for k in range(i + 1, len(PNAMES)):
            if abs(corr[i, k]) > 0.6:
                print(f"  {PNAMES[i]} ~ {PNAMES[k]}: r = {corr[i,k]:+.2f}")
                found = True
    if not found:
        print("  (none — parameters are largely independently constrained)")

    figures(boot, prof)
    print("\nFigures written: figures/fig5_profile_likelihood.png, figures/fig6_param_correlations.png")
    print("\nReading: identifiable params are pinned by the 4 readouts; 'sloppy' ones need additional data"
          "\n(esp. the missing FRDA-fibroblast Seahorse OCR) -> additional measurements would constrain them.")


if __name__ == "__main__":
    report()
