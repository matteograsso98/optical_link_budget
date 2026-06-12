"""
antenna.py
==========
Steering vector complet pour un réseau d'antennes planaire uniforme (UPA) :

    a(ϑ, φ) = gE(ϑ, φ) · Fsub(ϑ, φ) · aupa(ϑ, φ)

  - gE    : diagramme de l'élément individuel (modèle cosinus)
  - Fsub  : facteur de sous-réseau (groupement d'éléments en subarrays)
  - aupa  : déphasages géométriques du réseau planaire
"""

import numpy as np


def upa_steering_vector(
    az_deg, el_deg,
    N_x, N_y,
    d_x=0.5, d_y=0.5,
    N_sub_x=1, N_sub_y=1,
    d_sub_x=0.5, d_sub_y=0.5,
    ge_alpha=1.0,
):
    """
    Steering vector complet d'un UPA avec diagramme d'élément et sous-réseaux.

    Paramètres
    ----------
    az_deg   : array (U,) — azimut vers chaque utilisateur [deg]
    el_deg   : array (U,) — élévation vers chaque utilisateur [deg]
    N_x      : int — nombre de subarrays (ou éléments) en azimut
    N_y      : int — nombre de subarrays (ou éléments) en élévation
    d_x      : float — espacement inter-subarrays en x [longueurs d'onde]
    d_y      : float — espacement inter-subarrays en y [longueurs d'onde]
    N_sub_x  : int — éléments par subarray en x (1 = pas de subarray)
    N_sub_y  : int — éléments par subarray en y (1 = pas de subarray)
    d_sub_x  : float — espacement intra-subarray en x [longueurs d'onde]
    d_sub_y  : float — espacement intra-subarray en y [longueurs d'onde]
    ge_alpha : float — exposant du diagramme d'élément gE = sin^ge_alpha(el)
                       (0 = isotrope, 1 = cosinus, 2 = cosinus²)

    Retourne
    --------
    a : array complexe (U, N_x * N_y)
        Steering vector complet normalisé pour chaque utilisateur.
    """
    az  = np.radians(np.asarray(az_deg, dtype=np.float64))
    el  = np.radians(np.asarray(el_deg, dtype=np.float64))
    U   = az.size

    # ── gE : diagramme de l'élément individuel ────────────────────────────────
    # Boresight = nadir → angle depuis le boresight = 90° − el
    # gE(el) = cos(90°−el)^ge_alpha = sin^ge_alpha(el)
    # Valeurs ∈ [0, 1], max à el=90° (zénith).
    gE = np.sin(el) ** ge_alpha   # (U,)

    # ── Fsub : facteur de sous-réseau ─────────────────────────────────────────
    # Somme cohérente des N_sub_x × N_sub_y éléments du subarray.
    # Normalisé par N_sub_x * N_sub_y → amplitude ∈ [0, 1].
    # Avec N_sub_x = N_sub_y = 1 : Fsub = 1 (pas de subarray).
    if N_sub_x == 1 and N_sub_y == 1:
        Fsub = np.ones(U, dtype=np.complex128)
    else:
        ns = np.arange(N_sub_x)[None, :]   # (1, N_sub_x)
        ms = np.arange(N_sub_y)[None, :]   # (1, N_sub_y)
        phi_sx = 2*np.pi * d_sub_x * np.cos(el)[:,None] * np.sin(az)[:,None] * ns
        phi_sy = 2*np.pi * d_sub_y * np.sin(el)[:,None] * ms
        sum_x  = np.sum(np.exp(1j * phi_sx), axis=1)   # (U,)
        sum_y  = np.sum(np.exp(1j * phi_sy), axis=1)   # (U,)
        Fsub   = (sum_x * sum_y) / (N_sub_x * N_sub_y) # (U,) complexe, |.| ≤ 1

    # ── aupa : réponse géométrique du réseau de subarrays ────────────────────
    nx = np.arange(N_x)[None, :]   # (1, N_x)
    ny = np.arange(N_y)[None, :]   # (1, N_y)

    phi_x = 2*np.pi * d_x * np.cos(el)[:,None] * np.sin(az)[:,None] * nx  # (U, N_x)
    phi_y = 2*np.pi * d_y * np.sin(el)[:,None] * ny                        # (U, N_y)

    a_x = np.exp(1j * phi_x)   # (U, N_x)
    a_y = np.exp(1j * phi_y)   # (U, N_y)

    aupa = np.einsum('ui,uj->uij', a_x, a_y).reshape(U, N_x * N_y)
    aupa = aupa / np.sqrt(N_x * N_y)   # normalisé : ||aupa||² = 1

    # ── Steering vector complet ───────────────────────────────────────────────
    # gE et Fsub sont des scalaires par utilisateur → broadcast sur les antennes
    scale = (gE * Fsub)[:, None]   # (U, 1)
    return scale * aupa            # (U, N_x * N_y)
