"""
snr.py
======
SNR and Shannon rate for an optical FSO (IM/DD) link.

Optical convention:
    H is the physical channel amplitude gain (field gain): the received
    *signal power* is P_r = P_tx * |H|^2 and the SNR is that signal power
    referred to the receiver noise floor N0:

        SNR = P_tx * |H|^2 / N0          (power ratio, dimensionless)

    Note the correct form is  P_tx * |H|^2  (received signal power),
    NOT |H * P_tx|^2  (= P_tx^2 * |H|^2, which carries a spurious extra
    factor of P_tx and inflates the result).
"""

import numpy as np


def _snr_linear(H_sys, P_tx, noise_W):
    # Received signal power P_r = P_tx * ||h_i||^2, referred to the noise floor.
    return P_tx * np.sum(np.abs(H_sys) ** 2, axis=1) / noise_W


def compute_snr_dB(H_sys, P_tx, noise_W):
    """
    SNR in dB per user:  SNR_i = P_tx * ||h_i||^2 / N0.

    Parameters
    ----------
    H_sys   : real array (K, 1) — physical channel amplitude gain (K users)
    P_tx    : float             — transmit power [W]
    noise_W : float             — receiver noise floor N0 [W]

    Returns
    -------
    SNR_dB : array (K,) in dB
    """
    return 10 * np.log10(np.maximum(_snr_linear(H_sys, P_tx, noise_W), 1e-30))


def compute_shannon_rate(H_sys, bandwidth_hz, P_tx, noise_W):
    """
    Shannon rate:  R_i = B * log2(1 + SNR_i).

    Parameters
    ----------
    H_sys        : real array (K, 1) — physical channel amplitude gain
    bandwidth_hz : float             — bandwidth [Hz]
    P_tx         : float             — transmit power [W]
    noise_W      : float             — receiver noise floor N0 [W]

    Returns
    -------
    rate : array (K,) in bps
    """
    return bandwidth_hz * np.log2(1 + _snr_linear(H_sys, P_tx, noise_W))
