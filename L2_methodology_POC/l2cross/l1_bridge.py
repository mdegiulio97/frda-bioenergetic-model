"""L1 -> L2 bridge.

Runs the plain-numpy/scipy Layer-1 bioenergetic pilot ONCE per patient at
quasi-steady-state and condenses it into a small set of numeric constants
(reserve rho, mitochondrial iron Fe, ROS R) that parameterise the Layer-2
damage model. This module is NOT differentiated; it is pure host-side python.
"""

import os
import sys
from dataclasses import dataclass

# Make the L1 pilot importable: it lives in ../pilot_L1_bioenergetic
_PILOT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "pilot_L1_bioenergetic")
)
if _PILOT_DIR not in sys.path:
    sys.path.insert(0, _PILOT_DIR)

import frda_pilot  # noqa: E402
from frda_pilot import Params, Knobs, phenotype  # noqa: E402

# Weight of ROS in the (ATP, ROS) reserve form rho_raw = ATP / (ATP + K_ROS*ROS).
K_ROS = 1.0


@dataclass
class Patient:
    """Condensed L1 phenotype for one patient.

    F   : frataxin level (0 < F <= 1; healthy = 1.0)
    rho : bioenergetic reserve, normalised so healthy (F=1) -> ~1.0, in (0, 1]
    Fe  : mitochondrial labile iron (L1 'mito_iron')
    R   : ROS (L1 'ROS')
    """

    F: float
    rho: float
    Fe: float
    R: float


def make_patient(F: float) -> Patient:
    """Build a Patient from the L1 pilot at frataxin level F."""
    ph = phenotype(Params(), Knobs(F=F))
    ph0 = phenotype(Params(), Knobs(F=1.0))

    def _rho_raw(p):
        return p["ATP"] / (p["ATP"] + K_ROS * p["ROS"])

    rho_raw = _rho_raw(ph)
    rho_raw_healthy = _rho_raw(ph0)
    rho = float(rho_raw / rho_raw_healthy)
    # Clip to (0, 1]
    rho = min(rho, 1.0)
    rho = max(rho, 1e-6)

    return Patient(
        F=float(F),
        rho=rho,
        Fe=float(ph["mito_iron"]),
        R=float(ph["ROS"]),
    )
