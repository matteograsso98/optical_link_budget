"""
lut_interp.py
=============
Shared 1-D linear interpolation helper for elevation-indexed LUT lookups.

Used by both Channel (RF, ITU-R P.618) and DynamicLinkBudget (FSO, ITU-R P.1622).
"""

import numpy as np


def lookup_lut_dB(
    att_lut_dB: np.ndarray,
    att_lut_elev_deg: np.ndarray,
    user_indices,
    elevation_deg,
) -> np.ndarray:
    """
    Per-user 1-D linear interpolation into an (U, G) attenuation LUT.

    Parameters
    ----------
    att_lut_dB      : (U, G) float64 — attenuation table [dB].
    att_lut_elev_deg: (G,)   float64 — elevation grid [deg], uniform spacing.
    user_indices    : (K,) int       — row indices into att_lut_dB.
    elevation_deg   : (K,) float     — query elevations [deg].

    Returns
    -------
    att_dB : (K,) float64 — interpolated attenuation per user [dB].
    """
    grid   = att_lut_elev_deg
    elev   = np.clip(np.asarray(elevation_deg, dtype=np.float64), grid[0], grid[-1])
    step   = grid[1] - grid[0]
    idx_f  = (elev - grid[0]) / step
    idx_lo = np.floor(idx_f).astype(np.int64)
    idx_hi = np.minimum(idx_lo + 1, grid.size - 1)
    frac   = idx_f - idx_lo
    ui     = np.asarray(user_indices, dtype=np.int64)
    return (1.0 - frac) * att_lut_dB[ui, idx_lo] + frac * att_lut_dB[ui, idx_hi]
