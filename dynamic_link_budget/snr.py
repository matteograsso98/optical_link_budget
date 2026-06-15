"""
snr.py
======
Calcul du SNR et du débit Shannon à partir de la matrice de canal H_sys.
"""

import numpy as np


def _snr_linear(H_sys, P_tx):
    # SNR_i = P_tx · ||h_i||²  (somme incohérente — beamforming MRT optimal)
    return P_tx * np.sum(np.abs(H_sys) ** 2, axis=1)


def compute_snr_dB(H_sys, P_tx):
    """
    Calcule le SNR en dB pour chaque utilisateur.

    SNR_i = P_tx · ||h_i||²

    Paramètres
    ----------
    H_sys : array complexe (K, N) — matrice canal système (K users, N antennes)
    P_tx  : float                 — puissance d'émission (W)

    Retourne
    --------
    SNR_dB : array (K,) en dB
    """
    return 10 * np.log10(np.maximum(_snr_linear(H_sys, P_tx), 1e-30))


def compute_shannon_rate(H_sys, bandwidth_hz, P_tx):
    """
    Calcule le débit Shannon : R_i = B · log2(1 + SNR_i).

    Paramètres
    ----------
    H_sys        : array complexe (K, N) — matrice canal système
    bandwidth_hz : float                 — largeur de bande (Hz)
    P_tx         : float                 — puissance d'émission (W)

    Retourne
    --------
    rate : array (K,) en bps
    """
    return bandwidth_hz * np.log2(1 + _snr_linear(H_sys, P_tx))
