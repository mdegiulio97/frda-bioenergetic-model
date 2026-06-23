"""
Illustrative calibration of the reduced model against REAL literature fold-changes.

This is NOT a formal parameter fit. It (1) picks the disease knob F so the model's aconitase
fold-change matches the best tissue-matched anchor (FRDA fibroblasts), then (2) reports the model's
other predicted fold-changes next to the reported literature ranges, honestly flagging where the toy
model agrees and where it does not (those gaps are exactly what the calibrated Aim-1 model fixes).

Literature targets (FRDA / control), with sources:
  aconitase   0.68   fibroblasts            PMID 27518705   <- primary anchor (tissue-matched)
  ROS         ~2.0   fibroblasts            PMID 29922309
  mito_iron   6.5    fibroblasts (labile)   PMID 30778428
  ATP         0.33   skeletal muscle in vivo PMID 10500204  (different tissue -> caveat)
  NAD+/NADH   strong decrease (heart 85x)   PMID 22394676   (qualitative for fibroblasts)
Intervention validation targets:
  omaveloxolone (NRF2): mito-ROS to ~0.43 of untreated (-57%)   PMID 29922309
  NAD+ precursor (NMN): restores cardiac efficiency (SIRT3/NAD+) PMID 28724806
"""

import numpy as np
from frda_pilot import Params, Knobs, phenotype

p = Params()

LIT = {
    "aconitase":          (0.68, "PMID 27518705 (fibroblasts)"),
    "ROS":                (2.0,  "PMID 29922309 (fibroblasts)"),
    "mito_iron":          (6.5,  "PMID 30778428 (fibroblasts, labile Fe)"),
    "ATP":                (0.33, "PMID 10500204 (muscle, in vivo)"),
    "redox_NADplus_NADH": (None, "PMID 22394676 (heart, strong decrease)"),
}


def fold_changes(F):
    healthy = phenotype(p, Knobs(F=1.0))
    disease = phenotype(p, Knobs(F=F))
    return {k: disease[k] / healthy[k] for k in healthy}, healthy, disease


def calibrate_F_to_aconitase(target=0.68):
    Fs = np.linspace(1.0, 0.05, 400)
    best_F, best_err = 1.0, 1e9
    for F in Fs:
        fc, _, _ = fold_changes(F)
        err = abs(fc["aconitase"] - target)
        if err < best_err:
            best_err, best_F = err, F
    return best_F


def report():
    F = calibrate_F_to_aconitase(LIT["aconitase"][0])
    fc, healthy, disease = fold_changes(F)

    print(f"Illustrative calibration: disease knob F = {F:.3f} "
          f"(chosen so aconitase matches the fibroblast anchor)\n")
    print(f"{'readout':<22}{'model':>9}{'literature':>13}   source / note")
    print("-" * 78)
    for key, (lit_val, src) in LIT.items():
        m = fc[key]
        lit_str = "decrease" if lit_val is None else f"{lit_val:g}"
        flag = ""
        if lit_val is not None:
            ratio = m / lit_val
            flag = " OK" if 0.5 <= ratio <= 2.0 else "  <-- toy model off (needs real fit)"
        else:
            flag = " OK (direction)" if m < 1.0 else "  <-- wrong direction"
        print(f"{key:<22}{m:>9.2f}{lit_str:>13}   {src}{flag}")

    print("\nIntervention validation (at calibrated disease F):")
    dis = phenotype(p, Knobs(F=F))
    # NRF2 inducer (omaveloxolone-like): target mito-ROS ~0.43 of untreated
    for u in (2.0, 3.0, 4.0):
        omav = phenotype(p, Knobs(F=F, u_nrf2=u))
        print(f"  NRF2 inducer u_nrf2={u:<3}: ROS -> {omav['ROS']/dis['ROS']:.2f} of untreated "
              f"(omaveloxolone target ~0.43, PMID 29922309)")
    # NAD precursor (NMN-like): improves redox
    for u in (1.5, 2.0, 3.0):
        nad = phenotype(p, Knobs(F=F, u_nad=u))
        print(f"  NAD precursor u_nad={u:<3}: redox -> {nad['redox_NADplus_NADH']/dis['redox_NADplus_NADH']:.2f}x, "
              f"ATP -> {nad['ATP']/dis['ATP']:.2f}x (NMN restores efficiency, PMID 28724806)")

    print("\nHonest note: aconitase is matched by construction; ROS/redox agree in direction and rough"
          "\nmagnitude; mito-iron and muscle-ATP fold-changes are stronger in vivo than this uncalibrated"
          "\ntoy reproduces. Closing those gaps quantitatively IS the Aim-1 deliverable (fit to Seahorse/"
          "\naconitase/frataxin/iron data + learned Fe-S kinetics).")
    return F


if __name__ == "__main__":
    report()
