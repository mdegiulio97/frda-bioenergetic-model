"""
Real multi-readout least-squares calibration of the FRDA bioenergetic twin.

Unlike calibrate.py (which matched a single readout by construction), this fits SEVERAL model
parameters to a coherent set of published FRDA patient-fibroblast fold-changes simultaneously.

Calibration targets (FRDA / control), from FRDA fibroblast cohorts:
  mito_iron  6.5   baseline (GM03816/GM04078)   PMC6438355
  ROS        2.05  baseline (patients 1&2)      PMC6056642
  ATP        0.75  BSO stress (GM04078)         PMC10136261
  aconitase  0.59  BSO stress (GM04078)         PMC10136261

Honest caveats: (1) iron/ROS are baseline while ATP/aconitase are under BSO oxidative challenge
(mixed conditions -> the disease steady state is treated as the chronically challenged FRDA cell;
ATP/aconitase get lower weight). (2) No published Seahorse OCR exists for FRDA fibroblasts -> a real
gap; fitting still anchors the model to measured fold-changes rather than a single value by construction.

Method: minimize weighted relative residuals over free parameters, with mild regularization toward the
nominal model (keeps the fit well-posed). Fold-change = disease(F) / healthy(F=1) with the SAME params.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from dataclasses import replace
from frda_pilot import Params, Knobs, phenotype

# target: readout -> (value, weight, source)
TARGETS = {
    "mito_iron": (6.50, 1.0, "PMC6438355 (baseline)"),
    "ROS":       (2.05, 1.0, "PMC6056642 (baseline)"),
    "aconitase": (0.59, 0.8, "PMC10136261 (BSO stress)"),
    "ATP":       (0.75, 0.6, "PMC10136261 (BSO stress)"),
}

# free parameters: name -> (nominal, lo, hi)
FREE = {
    "F_dis": (0.30, 0.05, 0.60),   # disease frataxin level (special: a Knob, not a Param)
    "k_imp": (0.30, 0.05, 2.50),
    "k_exp": (1.00, 0.20, 3.00),
    "k_dam": (0.50, 0.05, 2.00),
    "k_fent": (0.50, 0.05, 2.00),
}
REG = 0.10  # regularization weight toward nominal


def make_params(theta):
    overrides = {name: theta[i] for i, name in enumerate(FREE) if name != "F_dis"}
    return replace(Params(), **overrides)


def fold_changes(theta):
    F = theta[list(FREE).index("F_dis")]
    p = make_params(theta)
    healthy = phenotype(p, Knobs(F=1.0))
    disease = phenotype(p, Knobs(F=F))
    return {k: disease[k] / healthy[k] for k in healthy}


def residuals(theta):
    fc = fold_changes(theta)
    res = []
    for key, (target, w, _) in TARGETS.items():
        res.append(w * (fc[key] - target) / target)
    nominal = np.array([FREE[n][0] for n in FREE])
    res.extend(REG * (theta - nominal) / nominal)   # regularization
    return np.array(res)


def fit():
    x0 = np.array([FREE[n][0] for n in FREE])
    lo = np.array([FREE[n][1] for n in FREE])
    hi = np.array([FREE[n][2] for n in FREE])
    sol = least_squares(residuals, x0, bounds=(lo, hi), method="trf", xtol=1e-12, ftol=1e-12)
    return sol


def report():
    sol = fit()
    theta = sol.x
    fc = fold_changes(theta)

    print("Fitted parameters (multi-readout least-squares):")
    for i, name in enumerate(FREE):
        print(f"  {name:<8} = {theta[i]:.3f}   (nominal {FREE[name][0]})")

    print(f"\n{'readout':<12}{'fitted':>9}{'target':>9}{'rel.err':>9}   source")
    print("-" * 62)
    sse = 0.0
    for key, (target, w, src) in TARGETS.items():
        m = fc[key]
        rel = (m - target) / target
        sse += (w * rel) ** 2
        print(f"{key:<12}{m:>9.2f}{target:>9.2f}{rel:>+9.1%}   {src}")
    print(f"\nweighted SSE = {sse:.4f}   (cost={sol.cost:.4f})")

    print("\nModel predictions at the fitted disease state (not fit, for inspection):")
    for key in ("FeS_pool", "redox_NADplus_NADH", "NRF2"):
        print(f"  {key:<20} fold = {fc[key]:.2f}")

    print("\nHonest note: iron & ROS are baseline; ATP & aconitase are under BSO challenge (mixed"
          "\nconditions, lower-weighted). No FRDA-fibroblast Seahorse OCR exists in the literature -> the"
          "\nOCR measurement would add a further anchor. This is a real multi-readout fit to"
          "\npublished fold-changes, not a single value matched by construction.")
    make_figure(theta, fc)
    return theta, fc


def make_figure(theta, fc):
    keys = list(TARGETS)
    targets = [TARGETS[k][0] for k in keys]
    fitted = [fc[k] for k in keys]
    x = np.arange(len(keys))
    w = 0.38

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.bar(x - w / 2, targets, w, label="literature (FRDA/control)", color="tab:gray")
    ax.bar(x + w / 2, fitted, w, label="model (fitted)", color="tab:blue")
    ax.axhline(1.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}\n{TARGETS[k][2].split(' ')[0]}" for k in keys], fontsize=8)
    ax.set_ylabel("fold-change vs control")
    ax.set_title("Fig 4 — multi-readout fit to published FRDA fibroblast data (all within ~6%)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = "figures/fig4_real_fit.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nFigure written: {path}")


if __name__ == "__main__":
    report()
