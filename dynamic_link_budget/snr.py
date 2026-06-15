"""
snr.py
======
Calcul du SNR et du débit Shannon à partir des matrices de canal H.
Ajout par rapport à channel.py — non présent dans le code original.
"""

import numpy as np


def _snr_linear(h, P_tx):
    return P_tx * np.sum(np.abs(h) ** 2, axis=1)


def compute_snr_dB(h, P_tx):
    """
    Calcule le SNR en dB à partir du vecteur de canal h.

    SNR = ||h||²  (ξ dans h absorbe déjà G_R / noise_W)

    Paramètres
    ----------
    h : array complexe (U, N_ant)

    Retourne
    --------
    SNR_dB : array (U,) en dB
    """
    return 10 * np.log10(np.maximum(_snr_linear(h, P_tx), 1e-30))


def compute_shannon_rate(h, bandwidth_hz, P_tx):
    """
    Calcule le débit Shannon : R = B · log2(1 + SNR).

    Paramètres
    ----------
    h            : array complexe (U, N_ant)
    bandwidth_hz : float — largeur de bande en Hz
    P_tx         : float — puissance d'émission

    Retourne
    --------
    rate : array (U,) en bps
    """
    return bandwidth_hz * np.log2(1 + _snr_linear(h, P_tx))