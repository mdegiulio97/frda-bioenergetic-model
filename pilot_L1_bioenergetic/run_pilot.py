"""
Run the Layer-1 pilot (v2): produce the proof-of-concept figures + summary table.

  Fig 1 — frataxin dose-response: lowering F reproduces the FRDA signature.
  Fig 2 — "medium perturbation": antioxidant partial rescue vs medium-iron worsening.
  Fig 3 — defensible interventions: NRF2 inducer (omaveloxolone-like) and NAD+ precursor (NMN-like).

Outputs to ./figures/. Run: python run_pilot.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from frda_pilot import Params, Knobs, phenotype
from calibrate import calibrate_F_to_aconitase

p = Params()
OUT = "figures"
F_DIS = calibrate_F_to_aconitase(0.68)   # calibrated disease level (aconitase anchor)


def fig1_frataxin_dose_response():
    Fs = np.linspace(1.0, 0.05, 40)
    keys = ["ATP", "aconitase", "mito_iron", "ROS"]
    data = {k: [phenotype(p, Knobs(F=f))[k] for f in Fs] for k in keys}
    ref = phenotype(p, Knobs(F=1.0))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    styles = {"ATP": "-", "aconitase": "--", "mito_iron": "-.", "ROS": ":"}
    labels = {"ATP": "ATP production", "aconitase": "aconitase activity",
              "mito_iron": "mitochondrial iron", "ROS": "ROS"}
    for k in keys:
        ax.plot(Fs, np.array(data[k]) / ref[k], styles[k], lw=2.2, label=labels[k])
    ax.axhline(1.0, color="gray", lw=0.8, alpha=0.6)
    ax.set_xlabel("frataxin fraction  F   (1 = healthy  →  FRDA)")
    ax.set_ylabel("steady state, normalized to healthy")
    ax.set_title("Fig 1 — frataxin deficiency reproduces the FRDA bioenergetic signature")
    ax.invert_xaxis(); ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout()
    path = f"{OUT}/fig1_frataxin_dose_response.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig2_medium_perturbation():
    levels = np.linspace(1.0, 3.0, 30)
    aox_atp = [phenotype(p, Knobs(F=F_DIS, u_aox=u))["ATP"] for u in levels]
    aox_ros = [phenotype(p, Knobs(F=F_DIS, u_aox=u))["ROS"] for u in levels]
    fe_ros = [phenotype(p, Knobs(F=F_DIS, u_fe=u))["ROS"] for u in levels]
    fe_iron = [phenotype(p, Knobs(F=F_DIS, u_fe=u))["mito_iron"] for u in levels]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
    axL.plot(levels, aox_ros, "-", lw=2.2, color="tab:green", label="ROS")
    axL.plot(levels, aox_atp, "--", lw=2.2, color="tab:blue", label="ATP")
    axL.set_title(f"Antioxidant (F={F_DIS:.2f}): partial rescue")
    axL.set_xlabel("antioxidant capacity  u_aox"); axL.set_ylabel("steady-state level")
    axL.legend(frameon=False); axL.grid(alpha=0.25)
    axR.plot(levels, fe_ros, "-", lw=2.2, color="tab:red", label="ROS")
    axR.plot(levels, fe_iron, "-.", lw=2.2, color="tab:orange", label="mito iron")
    axR.set_title(f"Iron loading (F={F_DIS:.2f}): worsens (honest nuance)")
    axR.set_xlabel("medium iron  u_fe"); axR.set_ylabel("steady-state level")
    axR.legend(frameon=False); axR.grid(alpha=0.25)
    fig.suptitle('Fig 2 — "medium perturbation": the bioenergetic state is perturbable', y=1.02)
    fig.tight_layout()
    path = f"{OUT}/fig2_medium_perturbation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def fig3_defensible_interventions():
    base = phenotype(p, Knobs(F=F_DIS))
    # NRF2 inducer (omaveloxolone-like)
    us_n = np.linspace(1.0, 5.0, 30)
    nrf2_ros = [phenotype(p, Knobs(F=F_DIS, u_nrf2=u))["ROS"] / base["ROS"] for u in us_n]
    nrf2_atp = [phenotype(p, Knobs(F=F_DIS, u_nrf2=u))["ATP"] / base["ATP"] for u in us_n]
    # NAD+ precursor (NMN/NR-like)
    us_d = np.linspace(1.0, 3.0, 30)
    nad_atp = [phenotype(p, Knobs(F=F_DIS, u_nad=u))["ATP"] / base["ATP"] for u in us_d]
    nad_redox = [phenotype(p, Knobs(F=F_DIS, u_nad=u))["redox_NADplus_NADH"] / base["redox_NADplus_NADH"]
                 for u in us_d]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
    axL.plot(us_n, nrf2_ros, "-", lw=2.2, color="tab:green", label="ROS")
    axL.plot(us_n, nrf2_atp, "--", lw=2.2, color="tab:blue", label="ATP")
    axL.axhline(0.43, color="tab:red", lw=1.3, ls=":", label="omaveloxolone ROS target (~0.43)")
    axL.set_title(f"NRF2 inducer (F={F_DIS:.2f}) — omaveloxolone-like")
    axL.set_xlabel("NRF2 induction  u_nrf2"); axL.set_ylabel("fold vs untreated disease")
    axL.legend(frameon=False, fontsize=8); axL.grid(alpha=0.25)

    axR.plot(us_d, nad_atp, "-", lw=2.2, color="tab:blue", label="ATP (throughput)")
    axR.plot(us_d, nad_redox, "-.", lw=2.2, color="tab:purple", label="NAD+/NADH ratio")
    axR.set_title("NAD+ precursor — NMN/NR-like")
    axR.set_xlabel("NAD+ pool  u_nad"); axR.set_ylabel("fold vs untreated disease")
    axR.legend(frameon=False, fontsize=8); axR.grid(alpha=0.25)
    fig.suptitle("Fig 3 — defensible interventions: NRF2 (omaveloxolone) raises ATP & lowers ROS; "
                 "NAD+ raises ATP throughput", y=1.02, fontsize=10)
    fig.tight_layout()
    path = f"{OUT}/fig3_interventions_nad_nrf2.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def summary_table():
    rows = [("healthy", Knobs(F=1.0)),
            (f"FRDA (F={F_DIS:.2f})", Knobs(F=F_DIS)),
            ("FRDA + NRF2 inducer", Knobs(F=F_DIS, u_nrf2=3.0)),
            ("FRDA + NAD precursor", Knobs(F=F_DIS, u_nad=2.0)),
            ("FRDA + antioxidant", Knobs(F=F_DIS, u_aox=3.0)),
            ("FRDA + iron loading", Knobs(F=F_DIS, u_fe=2.0))]
    keys = ["ATP", "aconitase", "mito_iron", "ROS", "redox_NADplus_NADH", "NRF2"]
    print(f"\n{'condition':<22}" + "".join(f"{k.split('_')[0]:>11}" for k in keys))
    print("-" * (22 + 11 * len(keys)))
    for name, k in rows:
        ph = phenotype(p, k)
        print(f"{name:<22}" + "".join(f"{ph[key]:>11.3f}" for key in keys))


if __name__ == "__main__":
    f1 = fig1_frataxin_dose_response()
    f2 = fig2_medium_perturbation()
    f3 = fig3_defensible_interventions()
    summary_table()
    print(f"\nCalibrated disease level F = {F_DIS:.3f}")
    print(f"Figures written:\n  {f1}\n  {f2}\n  {f3}")
