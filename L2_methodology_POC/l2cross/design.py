"""Acquisition design / value-of-information (VoI) for the Layer-2 model.

Given the latent phi = [log_r0, theta_cord, theta_dentate, theta_nerve, t0]
(dim = 5), a concrete patient, candidate observable subsets, and candidate
follow-up cadences, this module ranks which designs (observables x cadence)
actually add information about phi, subject to a measurement budget.

Design criterion
----------------
The single-patient Fisher information matrix is intrinsically near-singular
(condition number ~1e9): QSM is the only observable that loads on the
dentate-nucleus direction, so several latent directions are weakly or not
constrained. A plain CRLB with a tiny ridge is then dominated by the
regularization artifact in those unconstrained directions.

The PRIMARY, numerically-robust criterion used here is therefore
D-optimality:  ``logdet(F + ridge*I)`` evaluated with a COMMON ridge across
all candidate designs. This is monotone in the Loewner order (more data ->
larger logdet) and stable for rank-deficient F. The SECONDARY criterion is the
total CRLB ``sum(diag(inv(F + ridge*I)))`` under the same common ridge.

The acquisition schedule is passed as plain arrays (`times`, `mask`) so this
module does not depend on the schedule generator.
"""

import numpy as np

from .priors import OBSERVABLES
from . import identifiability

# Column index of each observable name in the (T, 3) observation matrix.
_OBS_INDEX = {name: i for i, name in enumerate(OBSERVABLES)}


def build_mask(times, observables_subset):
    """Boolean acquisition mask (T, 3).

    Parameters
    ----------
    times : (T,) array of acquisition times.
    observables_subset : iterable of observable names (from OBSERVABLES).
        The corresponding columns are True at *every* time; all other
        columns are False.

    Returns
    -------
    mask : (T, 3) bool array.
    """
    times = np.asarray(times)
    T = times.shape[0]
    mask = np.zeros((T, len(OBSERVABLES)), dtype=bool)
    for name in observables_subset:
        if name not in _OBS_INDEX:
            raise KeyError(
                f"unknown observable {name!r}; expected one of {OBSERVABLES}"
            )
        mask[:, _OBS_INDEX[name]] = True
    return mask


def score_schedule(phi, patient, times, mask, noise_sigma, ridge=1e-3):
    """Score one acquisition schedule with the D-optimality criterion.

    Parameters
    ----------
    phi : (5,) latent parameter vector at which information is evaluated.
    patient : concrete Patient (l1_bridge.Patient).
    times : (T,) acquisition times.
    mask : (T, 3) bool acquisition mask.
    noise_sigma : (3,) per-observable noise std-dev.
    ridge : common ridge added to F before logdet / inversion.

    Returns
    -------
    dict with keys:
      "logdet"         : float, logdet(F + ridge*I)   (primary criterion).
      "total_crlb"     : float, sum(diag(inv(F + ridge*I))) (secondary).
      "n_measurements" : int, number of measured (time, observable) entries.
    """
    F = identifiability.fisher(phi, patient, times, mask, noise_sigma)
    n = F.shape[0]
    _, logdet = np.linalg.slogdet(F + ridge * np.eye(n))
    total_crlb = float(np.sum(identifiability.crlb(F, ridge=ridge)))
    return {
        "logdet": float(logdet),
        "total_crlb": total_crlb,
        "n_measurements": int(np.asarray(mask).sum()),
    }


def voi_rank(
    phi,
    patient,
    candidate_obs_subsets,
    cadences,
    noise_sigma,
    budget,
    ridge=1e-3,
):
    """Rank candidate (observable-subset x cadence) designs by information.

    Parameters
    ----------
    phi : (5,) latent parameter vector.
    patient : concrete Patient.
    candidate_obs_subsets : list of tuples of observable names, e.g.
        [("cord_CSA",), ("cord_CSA", "SNAP"), ("cord_CSA", "QSM", "SNAP")].
    cadences : list of time grids (np arrays), e.g. a dense vs sparse
        follow-up schedule.
    noise_sigma : (3,) per-observable noise std-dev.
    budget : int. Designs whose n_measurements exceeds this are SKIPPED.
    ridge : common ridge for the D-optimality criterion.

    Returns
    -------
    list of dicts, sorted by descending logdet (best design first). Each entry:
      {"observables", "cadence_size", "n_measurements", "logdet", "total_crlb"}.
    """
    results = []
    for subset in candidate_obs_subsets:
        for times in cadences:
            times = np.asarray(times)
            n_meas = len(times) * len(subset)
            if n_meas > budget:
                continue
            mask = build_mask(times, subset)
            s = score_schedule(phi, patient, times, mask, noise_sigma, ridge=ridge)
            results.append(
                {
                    "observables": tuple(subset),
                    "cadence_size": len(times),
                    "n_measurements": s["n_measurements"],
                    "logdet": s["logdet"],
                    "total_crlb": s["total_crlb"],
                }
            )
    results.sort(key=lambda r: r["logdet"], reverse=True)
    return results


def minimal_sufficient(ranking, logdet_tol=0.95):
    """Recommend the cheapest design that retains most of the information.

    From a `voi_rank` ranking, find the design with the FEWEST measurements
    whose logdet is within `logdet_tol` of the best achievable logdet.

    Because logdet can be negative for a rank-deficient ridged Fisher, the
    "within tolerance" test is normalized by the maximum logdet so the
    comparison is sign-safe:

        accept iff (logdet - min_ld) / (max_ld - min_ld) >= logdet_tol

    where min_ld / max_ld are the worst / best logdet in the ranking. This maps
    the best design to 1.0 and the worst to 0.0, regardless of sign. When all
    logdets are (nearly) equal the cheapest design is returned.

    Parameters
    ----------
    ranking : list of dicts from `voi_rank` (non-empty).
    logdet_tol : float in (0, 1], fraction of the information span to retain.

    Returns
    -------
    dict : the recommended minimal-sufficient entry. This is the SAME dict
        object taken from `ranking` (so ``result in ranking`` holds), with two
        in-place annotations added: ``"recommended" = True`` and
        ``"logdet_fraction"`` (its normalized information score in [0, 1]).
    """
    if not ranking:
        raise ValueError("ranking is empty; nothing to recommend")

    logdets = np.array([r["logdet"] for r in ranking], dtype=np.float64)
    max_ld = float(logdets.max())
    min_ld = float(logdets.min())
    span = max_ld - min_ld

    def _fraction(ld):
        if span <= 1e-12:
            return 1.0  # all designs equally informative
        return (ld - min_ld) / span

    # Candidates whose normalized information meets the tolerance.
    qualifying = [
        (r, _fraction(r["logdet"]))
        for r in ranking
        if _fraction(r["logdet"]) >= logdet_tol
    ]

    if qualifying:
        # Fewest measurements; ties broken by higher logdet.
        best, frac = min(
            qualifying, key=lambda rf: (rf[0]["n_measurements"], -rf[0]["logdet"])
        )
    else:
        # No design qualifies -> fall back to the most informative (rank[0]).
        best = ranking[0]
        frac = _fraction(best["logdet"])

    # Annotate the original entry in place so ``best in ranking`` still holds.
    best["recommended"] = True
    best["logdet_fraction"] = float(frac)
    return best
