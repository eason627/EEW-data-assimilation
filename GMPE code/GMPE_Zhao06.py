"""
Wrapper around openquake.hazardlib's Zhao et al. (2006) crustal GMPE.

Reference
---------
Zhao, J.X. et al. (2006), "Attenuation Relations of Strong Ground Motion
in Japan Using Site Classification Based on Predominant Period",
Bull. Seismol. Soc. Am., 96(3), 898-913.

One of the most widely cited Japan-calibrated GMPEs internationally;
this module only wraps the crustal ("Asc") branch (Zhao06 also has
SInter/SSlab branches for subduction events, not covered here).

Why a wrapper instead of a hand-typed implementation
------------------------------------------------------
The model has a large
period-dependent coefficient table, so this calls
openquake.hazardlib's `ZhaoEtAl2006Asc` (GEM Foundation, tested
against the paper's verification tables) rather than retyping
coefficients from memory.

Requires: `pip install openquake.engine` (installs openquake.hazardlib).

Rake convention
----------------
`rake` (degrees) selects the faulting-style term: reverse for
rake in [45, 135], normal for rake in [-135, -45], strike-slip
otherwise. Defaults to 0.0 (pure strike-slip) here, matching this
project's default use case (e.g. Kumamoto).
"""

import numpy as np
from openquake.hazardlib.gsim.zhao_2006 import ZhaoEtAl2006Asc
from openquake.hazardlib.imt import PGA, SA

_GSIM = ZhaoEtAl2006Asc()


def _resolve_imt(imt, period):
    imt = imt.upper()
    if imt == "SA":
        if period is None:
            raise ValueError("period (s) is required when imt='SA'")
        return SA(period)
    if imt == "PGA":
        return PGA()
    raise ValueError(f"Unsupported imt {imt!r}; use 'PGA' or 'SA'")


def Zhao06(M, Rrup, hypo_depth, Vs30, rake=0.0, imt="PGA", period=None):
    """
    Ground motion prediction using Zhao et al. (2006), crustal ("Asc") model.

    Parameters
    ----------
    M          : magnitude, scalar or array (nr,)
    Rrup       : closest distance to rupture plane, km, array (nr,)
    hypo_depth : hypocentral depth, km, scalar or array broadcastable to Rrup
    Vs30       : site Vs30, m/s, scalar or array broadcastable to Rrup
    rake       : rake angle, degrees (default 0.0 = strike-slip; see module
                 docstring for the reverse/normal/strike-slip cutoffs)
    imt        : "PGA" (default) or "SA"
    period     : spectral period in s, required if imt == "SA"

    Returns
    -------
    median : predicted median PGA/SA in gal (cm/s^2).
    sigma  : total standard deviation, natural-log units (array, same shape).
    """
    imt_obj = _resolve_imt(imt, period)

    Rrup = np.atleast_1d(np.asarray(Rrup, dtype=float))
    n = Rrup.shape[0]

    dtype = [("mag", float), ("hypo_depth", float), ("rrup", float),
             ("vs30", float), ("rake", float), ("sids", int)]
    ctx = np.recarray(n, dtype=dtype)
    ctx.mag = M
    ctx.hypo_depth = hypo_depth
    ctx.rrup = Rrup
    ctx.vs30 = Vs30
    ctx.rake = rake
    ctx.sids = np.arange(n)

    mean = np.zeros((1, n))
    sig = np.zeros((1, n))
    tau = np.zeros((1, n))
    phi = np.zeros((1, n))

    _GSIM.compute(ctx, [imt_obj], mean, sig, tau, phi)

    median = np.exp(mean[0]) * 980.665  # g -> gal
    return median, sig[0]


if __name__ == "__main__":
    # Smoke test: M7.1 crustal event, hypo_depth=10km, Vs30=400 m/s
    pga, sigma = Zhao06(
        M=7.1, Rrup=np.array([5, 20, 50, 100, 200]), hypo_depth=10.0, Vs30=400.0
    )
    for r, p, s in zip([5, 20, 50, 100, 200], pga, sigma):
        print(f"Rrup={r:>4} km  PGA={p:8.2f} gal  sigma(ln)={s:.3f}")
