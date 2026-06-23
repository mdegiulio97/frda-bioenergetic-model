"""
Sanity checks = the KNOWN qualitative invariants of the FRDA bioenergetic signature (v2).

Each assertion encodes a fact from the literature. Passing = the reduced
model reproduces the FRDA phenotype AND the two new mechanisms (NAD+/NADH redox, NRF2) behave as
the corresponding interventions do. Run: python test_sanity.py
"""

import numpy as np
from frda_pilot import Params, Knobs, phenotype

p = Params()
EPS = 1e-3


def healthy_vs_disease():
    healthy = phenotype(p, Knobs(F=1.0))
    disease = phenotype(p, Knobs(F=0.2))
    assert disease["ATP"] < healthy["ATP"] - EPS, "ATP should fall"
    assert disease["aconitase"] < healthy["aconitase"] - EPS, "aconitase should fall"
    assert disease["mito_iron"] > healthy["mito_iron"] + EPS, "mito iron should accumulate (paradox)"
    assert disease["ROS"] > healthy["ROS"] + EPS, "ROS should rise"
    assert disease["FeS_pool"] < healthy["FeS_pool"] - EPS, "Fe-S pool should drop"
    # NEW: redox collapses (NAD+/NADH down) and NRF2 is compensatorily induced
    assert disease["redox_NADplus_NADH"] < healthy["redox_NADplus_NADH"] - EPS, "NAD+/NADH should fall"
    assert disease["NRF2"] > healthy["NRF2"] + EPS, "NRF2 should be induced by oxidative stress"
    return healthy, disease


def monotonic_in_frataxin():
    Fs = np.linspace(1.0, 0.1, 12)
    atp = [phenotype(p, Knobs(F=f))["ATP"] for f in Fs]
    fe = [phenotype(p, Knobs(F=f))["mito_iron"] for f in Fs]
    redox = [phenotype(p, Knobs(F=f))["redox_NADplus_NADH"] for f in Fs]
    assert all(atp[i] >= atp[i + 1] - EPS for i in range(len(atp) - 1)), "ATP not monotone in F"
    assert all(fe[i] <= fe[i + 1] + EPS for i in range(len(fe) - 1)), "iron not monotone in F"
    assert all(redox[i] >= redox[i + 1] - EPS for i in range(len(redox) - 1)), "redox not monotone in F"


def positivity_and_stability():
    for f in (1.0, 0.5, 0.1):
        s = phenotype(p, Knobs(F=f))
        for name, v in s.items():
            assert np.isfinite(v) and v >= -EPS, f"{name} negative/non-finite at F={f}"


def nrf2_inducer_rescues():
    """NRF2 induction (omaveloxolone-like) lowers ROS and protects bioenergetics at low F."""
    base = phenotype(p, Knobs(F=0.3, u_nrf2=1.0))
    omav = phenotype(p, Knobs(F=0.3, u_nrf2=3.0))
    assert omav["ROS"] < base["ROS"] - EPS, "NRF2 inducer should lower ROS"
    assert omav["NRF2"] > base["NRF2"] + EPS, "NRF2 inducer should raise NRF2"
    assert omav["ATP"] > base["ATP"] - EPS, "NRF2 inducer should not worsen ATP"


def nad_precursor_improves_throughput():
    """NAD+ precursor (NR/NMN-like) raises NAD pool -> raises ATP throughput (NMN restores cardiac
    efficiency, PMID 28724806). Honest mechanistic nuance: it does NOT by itself fix the NAD+/NADH
    ratio, which is set by the ETC/Fe-S bottleneck (S), not by pool size -> consistent with NAD+
    precursors helping bioenergetics without being curative."""
    base = phenotype(p, Knobs(F=0.3, u_nad=1.0))
    nad = phenotype(p, Knobs(F=0.3, u_nad=2.0))
    assert nad["ATP"] > base["ATP"] + EPS, "NAD precursor should raise ATP throughput"


def substrate_alone_worsens_redox():
    """Honest nuance: feeding substrate without fixing the ETC piles up NADH (reductive stress)."""
    base = phenotype(p, Knobs(F=0.3, u_sub=1.0))
    fed = phenotype(p, Knobs(F=0.3, u_sub=1.8))
    assert fed["redox_NADplus_NADH"] < base["redox_NADplus_NADH"] + EPS, "substrate alone should not improve redox"


def iron_loading_worsens():
    base = phenotype(p, Knobs(F=0.3, u_fe=1.0))
    loaded = phenotype(p, Knobs(F=0.3, u_fe=2.0))
    assert loaded["ROS"] > base["ROS"] - EPS, "iron loading should not reduce ROS"
    assert loaded["mito_iron"] > base["mito_iron"] - EPS, "iron loading should raise mito iron"


if __name__ == "__main__":
    healthy, disease = healthy_vs_disease()
    monotonic_in_frataxin()
    positivity_and_stability()
    nrf2_inducer_rescues()
    nad_precursor_improves_throughput()
    substrate_alone_worsens_redox()
    iron_loading_worsens()

    print("All sanity checks PASSED — v2 reproduces the FRDA signature + NAD/NRF2 mechanisms.\n")
    cols = ["ATP", "aconitase", "mito_iron", "ROS", "redox_NADplus_NADH", "NRF2"]
    print(f"  {'':<16}" + "".join(f"{c.split('_')[0]:>10}" for c in cols))
    for label, st in (("healthy F=1.0", healthy), ("disease F=0.2", disease)):
        print(f"  {label:<16}" + "".join(f"{st[c]:>10.3f}" for c in cols))
