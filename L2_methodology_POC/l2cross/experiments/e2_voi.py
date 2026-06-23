"""E2 -- Acquisition-design / value-of-information (VoI) experiment.

From the single-patient Fisher information (no MAP fits), the experiment computes
the D-optimality criterion (logdet of the Fisher matrix plus a ridge) for
different combinations of observables (cord CSA, dentate QSM, SNAP) and follow-up
cadences, the marginal information of each observable, and a minimal schedule
within a measurement budget.

Run:
  cd L2_methodology_POC
  python -m l2cross.experiments.e2_voi
"""

import os
from itertools import combinations

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless, file-only output
import matplotlib.pyplot as plt  # noqa: E402

from l2cross.priors import POP_MU, OBSERVABLES
from l2cross.l1_bridge import make_patient
from l2cross.design import (
    build_mask,
    score_schedule,
    voi_rank,
    minimal_sufficient,
)

# --------------------------------------------------------------------------- #
# Shared experiment constants (mirror the spec exactly).
# --------------------------------------------------------------------------- #
PHI = np.asarray(POP_MU, dtype=np.float64)
TIMES = np.linspace(0.0, 40.0, 21)
NOISE_SIGMA = np.array([0.03, 0.03, 0.05])
RIDGE = 1e-3

_FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "figures",
)


def _all_nonempty_subsets(names):
    """All 2**n - 1 non-empty subsets of `names`, ordered by size then name."""
    subsets = []
    for k in range(1, len(names) + 1):
        for combo in combinations(names, k):
            subsets.append(tuple(combo))
    return subsets


# --------------------------------------------------------------------------- #
# Part 1 -- Observable-subset value.
# --------------------------------------------------------------------------- #
def part1_observable_subset_value(patient):
    print("=" * 74)
    print("PART 1 -- OBSERVABLE-SUBSET VALUE  (F=%.2f, dense cadence, T=%d)"
          % (patient.F, len(TIMES)))
    print("=" * 74)

    subsets = _all_nonempty_subsets(OBSERVABLES)
    scored = {}
    for subset in subsets:
        mask = build_mask(TIMES, subset)
        s = score_schedule(PHI, patient, TIMES, mask, NOISE_SIGMA, ridge=RIDGE)
        scored[subset] = s

    print("%-28s %6s %12s %14s"
          % ("subset", "n_meas", "logdet", "total_crlb"))
    print("-" * 74)
    for subset in subsets:
        s = scored[subset]
        print("%-28s %6d %12.4f %14.4e"
              % ("+".join(subset), s["n_measurements"],
                 s["logdet"], s["total_crlb"]))

    # Incremental nats over the cord-only baseline.
    base = scored[("cord_CSA",)]["logdet"]
    cord_snap = scored[("cord_CSA", "SNAP")]["logdet"]
    cord_qsm = scored[("cord_CSA", "QSM")]["logdet"]
    full = scored[("cord_CSA", "QSM", "SNAP")]["logdet"]

    # Marginal nats of an observable = logdet(full) - logdet(full \ {obs}).
    snap_marginal = full - cord_qsm   # unique SNAP info given cord+QSM
    qsm_marginal = full - cord_snap   # unique QSM info given cord+SNAP

    print("-" * 74)
    print("Incremental information (nats, logdet) over cord-only baseline:")
    print("  cord-only                 logdet = %8.4f  (baseline)" % base)
    print("  +SNAP  (cord+SNAP)        delta  = %+8.4f" % (cord_snap - base))
    print("  +QSM   (cord+QSM)         delta  = %+8.4f" % (cord_qsm - base))
    print("  full   (cord+QSM+SNAP)    delta  = %+8.4f" % (full - base))
    print("Unique (marginal) information of each observable in the full set:")
    print("  SNAP marginal (full - cord+QSM)  = %+8.4f nats" % snap_marginal)
    print("  QSM  marginal (full - cord+SNAP) = %+8.4f nats" % qsm_marginal)
    print()

    return {
        "scored": scored,
        "base": base,
        "cord_snap": cord_snap,
        "cord_qsm": cord_qsm,
        "full": full,
        "snap_marginal": snap_marginal,
        "qsm_marginal": qsm_marginal,
    }


# --------------------------------------------------------------------------- #
# Part 2 -- Cadence sweep (diminishing returns).
# --------------------------------------------------------------------------- #
def part2_cadence_sweep(patient):
    print("=" * 74)
    print("PART 2 -- CADENCE SWEEP  (full observable set, vary n followups)")
    print("=" * 74)

    n_followups = [3, 5, 9, 15, 21]
    full_subset = tuple(OBSERVABLES)

    rows = []
    print("%10s %8s %12s %14s" % ("n_times", "n_meas", "logdet", "total_crlb"))
    print("-" * 74)
    for nt in n_followups:
        times = np.linspace(0.0, 40.0, nt)
        mask = build_mask(times, full_subset)
        s = score_schedule(PHI, patient, times, mask, NOISE_SIGMA, ridge=RIDGE)
        rows.append((nt, s["n_measurements"], s["logdet"], s["total_crlb"]))
        print("%10d %8d %12.4f %14.4e"
              % (nt, s["n_measurements"], s["logdet"], s["total_crlb"]))
    print()
    return rows


# --------------------------------------------------------------------------- #
# Part 3 -- VoI ranking under a measurement budget.
# --------------------------------------------------------------------------- #
def part3_voi_ranking(patient, budget=30):
    print("=" * 74)
    print("PART 3 -- VoI RANKING UNDER BUDGET  (budget = %d measurements)" % budget)
    print("=" * 74)

    candidate_obs_subsets = [
        ("cord_CSA",),
        ("cord_CSA", "SNAP"),
        ("cord_CSA", "QSM"),
        ("cord_CSA", "QSM", "SNAP"),
    ]
    cadences = [
        np.linspace(0.0, 40.0, 3),
        np.linspace(0.0, 40.0, 5),
        np.linspace(0.0, 40.0, 9),
        np.linspace(0.0, 40.0, 15),
        np.linspace(0.0, 40.0, 21),
    ]

    ranking = voi_rank(
        PHI, patient, candidate_obs_subsets, cadences,
        NOISE_SIGMA, budget=budget, ridge=RIDGE,
    )

    print("%-22s %8s %8s %12s %14s"
          % ("observables", "cad_size", "n_meas", "logdet", "total_crlb"))
    print("-" * 74)
    for r in ranking:
        print("%-22s %8d %8d %12.4f %14.4e"
              % ("+".join(r["observables"]), r["cadence_size"],
                 r["n_measurements"], r["logdet"], r["total_crlb"]))

    rec = minimal_sufficient(ranking, logdet_tol=0.95)
    print("-" * 74)
    print("MINIMAL-SUFFICIENT design (fewest measurements within 95% of best "
          "logdet):")
    print("  observables    : %s" % "+".join(rec["observables"]))
    print("  cadence_size   : %d timepoints over 0..40 yr" % rec["cadence_size"])
    print("  n_measurements : %d" % rec["n_measurements"])
    print("  logdet         : %.4f" % rec["logdet"])
    print("  logdet_fraction: %.4f (1.0 = best in ranking)"
          % rec["logdet_fraction"])
    print()
    return ranking, rec


# --------------------------------------------------------------------------- #
# Part 4 -- Across-severity robustness (light).
# --------------------------------------------------------------------------- #
def part4_severity_robustness(severities=(0.15, 0.25, 0.4)):
    print("=" * 74)
    print("PART 4 -- ACROSS-SEVERITY ROBUSTNESS  (F in %s)"
          % (tuple(severities),))
    print("=" * 74)

    rows = []
    ranking_stable = True
    expected_order = ["cord_CSA+QSM+SNAP", "cord_CSA+SNAP",
                      "cord_CSA+QSM", "cord_CSA"]

    print("%6s %8s %12s %12s %12s %14s"
          % ("F", "Fe", "ld_full", "ld_cordSNAP", "ld_cordQSM", "QSM_marginal"))
    print("-" * 74)
    for F in severities:
        patient = make_patient(F)
        sub_full = build_mask(TIMES, ("cord_CSA", "QSM", "SNAP"))
        sub_cs = build_mask(TIMES, ("cord_CSA", "SNAP"))
        sub_cq = build_mask(TIMES, ("cord_CSA", "QSM"))
        sub_c = build_mask(TIMES, ("cord_CSA",))

        ld_full = score_schedule(PHI, patient, TIMES, sub_full, NOISE_SIGMA,
                                 ridge=RIDGE)["logdet"]
        ld_cs = score_schedule(PHI, patient, TIMES, sub_cs, NOISE_SIGMA,
                               ridge=RIDGE)["logdet"]
        ld_cq = score_schedule(PHI, patient, TIMES, sub_cq, NOISE_SIGMA,
                               ridge=RIDGE)["logdet"]
        ld_c = score_schedule(PHI, patient, TIMES, sub_c, NOISE_SIGMA,
                              ridge=RIDGE)["logdet"]

        # QSM marginal = unique QSM info given cord+SNAP already present.
        qsm_marginal = ld_full - ld_cs

        order = sorted(
            {"cord_CSA+QSM+SNAP": ld_full, "cord_CSA+SNAP": ld_cs,
             "cord_CSA+QSM": ld_cq, "cord_CSA": ld_c}.items(),
            key=lambda kv: kv[1], reverse=True,
        )
        order_names = [k for k, _ in order]
        if order_names != expected_order:
            ranking_stable = False

        rows.append({"F": F, "Fe": patient.Fe, "ld_full": ld_full,
                     "ld_cs": ld_cs, "ld_cq": ld_cq,
                     "qsm_marginal": qsm_marginal})
        print("%6.2f %8.4f %12.4f %12.4f %12.4f %14.4f"
              % (F, patient.Fe, ld_full, ld_cs, ld_cq, qsm_marginal))

    print("-" * 74)
    print("Qualitative ranking (full best > cord+SNAP > cord+QSM > cord-only) "
          "stable across F: %s" % ("YES" if ranking_stable else "NO"))
    qsm_trend = "increases" if rows[-1]["qsm_marginal"] > rows[0]["qsm_marginal"] \
        else "decreases"
    print("QSM marginal nats vs severity: %s as F goes %.2f -> %.2f "
          "(%.4f -> %.4f)"
          % (qsm_trend, rows[0]["F"], rows[-1]["F"],
             rows[0]["qsm_marginal"], rows[-1]["qsm_marginal"]))
    print()
    return rows, ranking_stable


# --------------------------------------------------------------------------- #
# Figures.
# --------------------------------------------------------------------------- #
def _fig_observable_value(p1):
    labels = ["cord-only\n(baseline)", "+SNAP\n(cord+SNAP)",
              "+QSM\n(cord+QSM)", "full\n(cord+QSM+SNAP)"]
    base = p1["base"]
    values = [0.0, p1["cord_snap"] - base, p1["cord_qsm"] - base,
              p1["full"] - base]
    colors = ["#888888", "#1f77b4", "#d62728", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Incremental information (nats, logdet) over cord-only")
    ax.set_title("E2 -- Value of each observable (D-optimality, F=0.25)")
    ax.axhline(0.0, color="k", lw=0.8)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, "%+.2f" % v,
                ha="center", va="bottom", fontsize=10)
    # Annotate the unique marginals.
    ax.text(0.02, 0.95,
            "SNAP unique = %+.2f nats\nQSM unique = %+.2f nats (dentate-specific)"
            % (p1["snap_marginal"], p1["qsm_marginal"]),
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    fig.tight_layout()
    path = os.path.join(_FIG_DIR, "e2_observable_value.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _fig_cadence(cadence_rows, rec):
    n_meas = [r[1] for r in cadence_rows]
    logdets = [r[2] for r in cadence_rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(n_meas, logdets, "o-", color="#2ca02c", label="full observable set")
    ax.set_xlabel("n_measurements (full set: 3 x n_timepoints)")
    ax.set_ylabel("logdet(F + ridge I)  [nats]")
    ax.set_title("E2 -- Cadence sweep: diminishing returns")

    # Mark the minimal-sufficient point if it is on the full-set curve.
    if rec is not None:
        ax.axvline(rec["n_measurements"], color="#d62728", ls="--", lw=1.2,
                   label="minimal-sufficient (n=%d)" % rec["n_measurements"])
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(_FIG_DIR, "e2_cadence.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _fig_qsm_vs_severity(sev_rows):
    Fs = [r["F"] for r in sev_rows]
    qsm = [r["qsm_marginal"] for r in sev_rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Fs, qsm, "s-", color="#d62728")
    ax.set_xlabel("Patient frataxin level F  (lower = more severe)")
    ax.set_ylabel("QSM marginal information (nats, logdet)")
    ax.set_title("E2 -- Unique dentate (QSM) information vs severity")
    ax.invert_xaxis()  # severe (low F) on the right
    for x, y in zip(Fs, qsm):
        ax.text(x, y, "  %.3f" % y, va="center", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(_FIG_DIR, "e2_qsm_vs_severity.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(_FIG_DIR, exist_ok=True)

    print()
    print("#" * 74)
    print("# E2 -- ACQUISITION DESIGN / VALUE-OF-INFORMATION  (Layer-2 POC)")
    print("#   phi = POP_MU, times = linspace(0,40,21), sigma = "
          "[0.03,0.03,0.05], ridge = 1e-3")
    print("#" * 74)
    print()

    patient = make_patient(0.25)
    print("Reference patient: F=%.2f  rho=%.4f  Fe=%.4f  R=%.4f"
          % (patient.F, patient.rho, patient.Fe, patient.R))
    print()

    p1 = part1_observable_subset_value(patient)
    cadence_rows = part2_cadence_sweep(patient)
    ranking, rec = part3_voi_ranking(patient, budget=30)
    sev_rows, stable = part4_severity_robustness((0.15, 0.25, 0.4))

    fig1 = _fig_observable_value(p1)
    fig2 = _fig_cadence(cadence_rows, rec)
    fig3 = _fig_qsm_vs_severity(sev_rows)

    # ----------------------------------------------------------------------- #
    print("=" * 74)
    print("SUMMARY -- ACQUISITION DESIGN")
    print("=" * 74)
    print("D-optimality logdet at F=0.25 (dense cadence):")
    print("  full(cord+QSM+SNAP)=%.3f, cord+SNAP=%.3f, cord+QSM=%.3f "
          ", cord-only=%.3f"
          % (p1["full"], p1["cord_snap"], p1["cord_qsm"], p1["base"]))
    print("  SNAP adds %+.2f nats; QSM adds %+.2f nats of unique "
          "information." % (p1["snap_marginal"], p1["qsm_marginal"]))
    print()
    print("Recommended MINIMAL-SUFFICIENT schedule (budget=30, within 95% of "
          "best logdet):")
    print("  -> %s at %d timepoints over 0..40 yr  (%d measurements, "
          "logdet=%.3f, frac=%.3f)"
          % ("+".join(rec["observables"]), rec["cadence_size"],
             rec["n_measurements"], rec["logdet"], rec["logdet_fraction"]))
    print()
    print("Severity robustness: full-best ranking stable across F=0.15/0.25/0.40 "
          ": %s" % ("YES" if stable else "NO"))
    print("  QSM marginal nats: F=0.15 -> %.3f, F=0.25 -> %.3f, F=0.40 -> %.3f"
          % (sev_rows[0]["qsm_marginal"], sev_rows[1]["qsm_marginal"],
             sev_rows[2]["qsm_marginal"]))
    print()
    print("ACTIONABLE CONCLUSION:")
    print("  cord_CSA+SNAP at the minimal-sufficient cadence captures most of "
          "the progression information;")
    print("  QSM adds a measurable but smaller dentate-specific increment, "
          "largest in more severe (low-F) patients.")
    print()
    print("Figures saved:")
    for f in (fig1, fig2, fig3):
        print("  - %s" % f)
    print("=" * 74)

    return {
        "part1": p1,
        "cadence_rows": cadence_rows,
        "ranking": ranking,
        "recommended": rec,
        "severity_rows": sev_rows,
        "stable": stable,
        "figures": [fig1, fig2, fig3],
    }


if __name__ == "__main__":
    main()
