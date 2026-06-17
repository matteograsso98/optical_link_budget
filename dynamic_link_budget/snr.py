"""
snr.py
======
Calcul du SNR et du débit Shannon pour un lien FSO optique (IM/DD).

Convention optique :
    H_sys encodes l'amplitude du gain de canal (racine du rapport de puissances).
    SNR = P_tx · ||h_i||²   où P_tx est une puissance [W].
    C'est un rapport de puissances optiques : SNR = P_r / N₀.
"""

import numpy as np


def _snr_linear(H_sys, P_tx):
    # SNR_i = P_tx · ||h_i||²  (P_tx = puissance [W], H_sys = gain d'amplitude)
    return P_tx * np.sum(np.abs(H_sys) ** 2, axis=1)


def compute_snr_dB(H_sys, P_tx):
    """
    Calcule le SNR en dB pour chaque utilisateur.

    SNR_i = P_tx · ||h_i||²

    Paramètres
    ----------
    H_sys : array réel (K, 1) — gain d'amplitude du canal FSO (K users)
    P_tx  : float             — puissance effective d'émission [W]

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
