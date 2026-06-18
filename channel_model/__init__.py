"""
channel_model — channel models for the link budget.

Two channels share a common lifecycle (precompute LUT → compute per timestep)
and base machinery (BaseChannel):

  - RFChannel      : RF cellular channel (ITU-R P.618, antenna arrays, carrier
                     phase). Kept as a reference; depends on the original RRM
                     project's large-scale-fading tables, not shipped here.
  - OpticalChannel : FSO optical channel (ITU-R P.1622, single narrow-beam
                     aperture, real phase-less channel). Liang et al.,
                     arXiv:2204.13177v1.
"""

from channel_model.base import BaseChannel
from channel_model.rf_channel import RFChannel
from channel_model.optical_channel import OpticalChannel

__all__ = ["BaseChannel", "RFChannel", "OpticalChannel"]
