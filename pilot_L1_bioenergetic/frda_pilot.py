"""
In-silico pilot (v2) — Reduced mechanistic ODE model of FRDA cellular bioenergetics,
extended with NAD+/NADH redox and NRF2 antioxidant induction.

Proof-of-concept mechanistic model of FRDA cellular bioenergetics.
This is a REDUCED, transparent model — NOT the full Cortassa & Aon system (PMID 22057575).

Core causal chain (v1) + two defensible mechanisms (v2):
    frataxin (F) -> Fe-S biogenesis (S) -> ETC/aconitase -> ATP
                                       \-> (IRP1-like, low S) -> mito iron (Fe) up -> ROS (R) up -> damages S
    NEW: substrate -> NADH (H); ETC oxidizes NADH (needs S) -> NAD+/NADH redox
    NEW: ROS induces NRF2 (N); NRF2 augments antioxidant capacity (omaveloxolone mechanism)

These two additions matter because they map onto interventions with REAL support:
    NRF2 induction  = omaveloxolone (only FDA-approved drug; T4; PMID 37097282 / PMC6056642)
    NAD+ restoration = NAD+ precursors NR/NMN (T5; Lancet Neurol 2026; PMID 28724806)

State (dimensionless; healthy steady state at F=1 is the reference):
  S = functional Fe-S cluster pool   Fe = mitochondrial labile iron   R = ROS
  H = NADH   N = NRF2 activity
Knobs: F frataxin (disease) · u_sub substrate · u_fe medium iron · u_aox antioxidant
       · u_nad NAD+ pool size (NAD precursor) · u_nrf2 NRF2 inducer (omaveloxolone-like)
"""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class Params:
    # Fe-S cluster biogenesis / loss
    Smax: float = 1.0
    k_syn: float = 2.0
    K_fe: float = 0.30
    d_S: float = 0.10
    k_dam: float = 0.50

    # Mitochondrial labile iron
    k_imp: float = 0.30
    K_irp: float = 0.30
    a_syn: float = 0.50
    k_exp: float = 1.00

    # ROS
    k_leak: float = 0.50
    k_fent: float = 0.50
    d_R: float = 0.20
    k_aox: float = 1.00
    alpha_N: float = 2.00   # NRF2 augmentation of antioxidant capacity
    beta_nad: float = 0.30  # NAD+/SIRT3-dependent antioxidant defense (PMID 22394676)

    # NAD+/NADH redox
    NADtot0: float = 1.00   # total NAD pool (NAD+ + NADH), scaled by u_nad
    k_dh: float = 1.50      # substrate-driven NADH production (TCA dehydrogenases)
    k_etc: float = 3.00     # ETC oxidation of NADH (Fe-S/complex-dependent)
    K_H: float = 0.50       # NADH half-saturation of ETC (so ATP saturates, not linear in NAD pool)

    # NRF2 induction
    k_non: float = 0.80     # ROS-driven NRF2 activation (drug-inducible)
    K_nrf: float = 0.30
    k_noff: float = 0.50

    # Output couplings
    k_atp: float = 1.00
    c_atp: float = 1.00
    c_ac: float = 1.00


@dataclass
class Knobs:
    F: float = 1.0
    u_sub: float = 1.0
    u_fe: float = 1.0
    u_aox: float = 1.0
    u_nad: float = 1.0      # NAD+ precursor (scales total NAD pool)
    u_nrf2: float = 1.0     # NRF2 inducer (omaveloxolone-like)


def rhs(t, y, p: Params, k: Knobs):
    S, Fe, R, H, N = [max(v, 0.0) for v in y]
    NAD_total = p.NADtot0 * k.u_nad
    NADplus = max(NAD_total - H, 0.0)

    biogenesis = p.k_syn * k.F * (Fe / (p.K_fe + Fe)) * (p.Smax - S)
    dS = biogenesis - (p.d_S + p.k_dam * R) * S

    import_Fe = p.k_imp * k.u_fe * (p.K_irp / (p.K_irp + S))
    export_Fe = p.k_exp * S * Fe
    dFe = import_Fe - p.a_syn * biogenesis - export_Fe

    v_dh = p.k_dh * k.u_sub * NADplus            # NAD+ -> NADH (substrate)
    v_etc = p.k_etc * S * H / (p.K_H + H)        # NADH -> NAD+ (ETC, needs Fe-S; saturating)
    dH = v_dh - v_etc

    ros_prod = p.k_leak * (p.Smax - S) + p.k_fent * Fe
    antiox = p.d_R + p.k_aox * (k.u_aox + p.alpha_N * N + p.beta_nad * NADplus)
    dR = ros_prod - antiox * R

    dN = p.k_non * (R / (p.K_nrf + R)) * k.u_nrf2 - p.k_noff * N

    return [dS, dFe, dR, dH, dN]


def steady_state(p: Params, k: Knobs, t_end: float = 600.0, y0=(0.5, 0.3, 0.2, 0.4, 0.1)):
    sol = solve_ivp(rhs, (0.0, t_end), list(y0), args=(p, k),
                    method="LSODA", rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    S, Fe, R, H, N = sol.y[:, -1]
    return {"S": S, "Fe": Fe, "R": R, "H": H, "N": N}


def outputs(state, p: Params, k: Knobs):
    S, Fe, R, H, N = (state[x] for x in ("S", "Fe", "R", "H", "N"))
    NAD_total = p.NADtot0 * k.u_nad
    NADplus = max(NAD_total - H, 0.0)
    v_etc = p.k_etc * S * H / (p.K_H + H)
    return {
        "ATP": p.k_atp * v_etc / (1.0 + p.c_atp * R),
        "aconitase": S / (1.0 + p.c_ac * R),
        "FeS_pool": S,
        "mito_iron": Fe,
        "ROS": R,
        "redox_NADplus_NADH": NADplus / (H + 1e-9),
        "NRF2": N,
    }


def phenotype(p: Params, k: Knobs):
    return outputs(steady_state(p, k), p, k)
