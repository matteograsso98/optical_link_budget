"""
base.py
=======
Shared code for the channel models (RF and optical).

Both channels follow the same lifecycle —

    precompute_lut(...)  once   →   compute(...)  per timestep

— and the same post-processing: read a per-user atmospheric loss [dB] from an
elevation-indexed LUT, convert it to a channel-amplitude scaling, build the
realistic channel, and zero out non-visible (slant ≤ 0) links.

The physics that *differs* between the two lives in the subclasses:
  - RFChannel      : ITU-R P.618, antenna-array radiation pattern, carrier phase.
  - OpticalChannel : ITU-R P.1622, single aperture, real (phase-less) channel.

Only the mechanics common to both live here.
"""

import numpy as np

from channel_model.lut_interp import lookup_lut_dB


class BaseChannel:
    def __init__(self) -> None:
        # Per-user atmospheric-attenuation LUT, populated by the subclass'
        # precompute step. Shape (U, G) [dB] over an elevation grid (G,) [deg].
        self._att_lut_dB = None
        self._att_lut_elev_deg = None

    def _lookup_attenuation_dB(self, user_indices, elevation_deg):
        """Per-user atmospheric loss [dB] via 1-D linear interp in elevation."""
        return lookup_lut_dB(
            self._att_lut_dB, self._att_lut_elev_deg, user_indices, elevation_deg
        )

    @staticmethod
    def _amplitude_scaling(loss_dB):
        """Convert a power loss [dB] to a channel-amplitude scaling factor."""
        return 1.0 / np.sqrt(10 ** (0.1 * np.asarray(loss_dB)))

    @staticmethod
    def _visibility_mask(slant_range):
        """True where a (sat, user) link is geometrically valid (finite, slant > 0).

        A non-visible link (slant ≤ 0 or non-finite) would divide-by-zero in the
        ideal-channel construction; callers mask those rows to |H| = 0 so the
        downstream rate is 0.
        """
        slant_range = np.asarray(slant_range)
        return np.isfinite(slant_range) & (slant_range > 0)
