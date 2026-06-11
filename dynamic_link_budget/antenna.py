"""
antenna.py
==========
Steering vector pour un réseau d'antennes planaire uniforme (UPA).
Remplace tx.compute_tx_radiation_pattern() de channel.py.
"""

import numpy as np


def upa_steering_vector(az_deg, el_deg, N_x, N_y, d_x=0.5, d_y=0.5):
    """
    Calcule le steering vector d'un UPA (N_x × N_y éléments).

    Paramètres
    ----------
    az_deg : array (U,) — azimut vers chaque utilisateur [deg]
    el_deg : array (U,) — élévation vers chaque utilisateur [deg]
    N_x    : int — nombre d'éléments en azimut
    N_y    : int — nombre d'éléments en élévation
    d_x    : float — espacement en x (en longueurs d'onde, défaut 0.5)
    d_y    : float — espacement en y (en longueurs d'onde, défaut 0.5)

    Retourne
    --------
    a : array complexe (U, N_x * N_y)
        Chaque ligne = steering vector normalisé pour un utilisateur.

    Notes
    -----
    Formule UPA standard :
        a(u) = a_x(u) ⊗ a_y(u)
    avec :
        a_x[n] = exp(j·2π·d_x·cos(el)·sin(az)·n)
        a_y[m] = exp(j·2π·d_y·sin(el)·m)
    normalisé par 1/sqrt(N_x·N_y).
    """
    az = np.radians(np.asarray(az_deg, dtype=np.float64))
    el = np.radians(np.asarray(el_deg, dtype=np.float64))
    U  = az.size

    nx = np.arange(N_x)[None, :]   # (1, N_x)
    ny = np.arange(N_y)[None, :]   # (1, N_y)

    # Phases inter-éléments
    phi_x = 2*np.pi * d_x * np.cos(el)[:,None] * np.sin(az)[:,None] * nx  # (U, N_x)
    phi_y = 2*np.pi * d_y * np.sin(el)[:,None] * ny                        # (U, N_y)

    a_x = np.exp(1j * phi_x)   # (U, N_x)
    a_y = np.exp(1j * phi_y)   # (U, N_y)

    # Produit de Kronecker ligne par ligne → (U, N_x*N_y)
    a = np.einsum('ui,uj->uij', a_x, a_y).reshape(U, N_x * N_y)
    return a / np.sqrt(N_x * N_y)