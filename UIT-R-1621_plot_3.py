"""
UIT-R P.1621-2 — Propagation atmosphérique (20–375 THz)
========================================================
Module de calcul des effets de propagation pour les liaisons Terre-espace
dans la gamme optique/infrarouge.

Sources des formules :
  - Rec. UIT-R P.1621-2 (07/2015) — ci-après "P.1621"
  - Liang et al., arXiv:2204.13177v1 — ci-après "Liang"

Formules implémentées (toutes vérifiées dans les documents sources) :
  - Diffusion de Mie : extinction ρ et atténuation Im   (Liang éqs. 6–7)
  - Diffusion géométrique (brouillard/nuages) : Ig       (Liang éqs. 8–10)
  - Atténuation atmosphérique totale LA                  (Liang éq. 11)
  - Indice de réfraction neff (std et T,P)               (P.1621 éqs. 2–3)
  - Angle d'élévation apparent                           (P.1621 éq. 4)
  - Turbulences Huffnagel-Valley 5/7 : Cn², r0, θ0, τ0  (P.1621 § 5)

FORMULES MANQUANTES (non fournies par les deux documents) :
  - Atténuation Rayleigh en dB/km : la Fig. 4 de P.1621 montre la courbe
    graphiquement mais aucune équation analytique n'est donnée dans le
    document. À rechercher dans d'autres références (ex. : modèles de
    diffusion moléculaire standard).

Auteur  : généré depuis Rec. UIT-R P.1621-2 et arXiv:2204.13177v1
"""

from typing import Union, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Constantes physiques
# ---------------------------------------------------------------------------
C_LIGHT = 3e8          # vitesse de la lumière (m/s)
Z_TURB  = 20_000       # hauteur effective de la turbulence (m) — P.1621 § 5.1.1

# ---------------------------------------------------------------------------
# Utilitaires de conversion
# ---------------------------------------------------------------------------

def freq_THz_to_wavelength_um(freq_THz: Union[float, np.ndarray]) -> np.ndarray:
    """Convertit une fréquence en THz en longueur d'onde en µm."""
    freq_THz = np.asarray(freq_THz, dtype=float)
    return C_LIGHT / (freq_THz * 1e12) * 1e6   # µm


def freq_THz_to_wavelength_m(freq_THz: Union[float, np.ndarray]) -> np.ndarray:
    """Convertit une fréquence en THz en longueur d'onde en mètres."""
    freq_THz = np.asarray(freq_THz, dtype=float)
    return C_LIGHT / (freq_THz * 1e12)           # m


# ===========================================================================
# § 3.1  Diffusion de Rayleigh
# ===========================================================================
#
# FORMULE MANQUANTE
# -----------------
# La Rec. UIT-R P.1621-2 indique au § 3.1 que la diffusion de Rayleigh
# dépend de λ⁻⁴ et qu'elle est négligeable en dessous de 375 THz.
# La Fig. 4 montre les courbes Rayleigh et Mie graphiquement pour une
# atmosphère de référence au niveau de la mer, MAIS aucune équation
# analytique en dB/km n'est fournie dans le document.
#
# Aucune formule du papier Liang et al. ne couvre non plus ce terme.
# La fonction ci-dessous est donc supprimée du code.
# Pour implémenter ce terme, chercher par exemple :
#   - Zuev V.E., "Laser Beams in the Atmosphere" (1982)
#   - Formules de section efficace de Rayleigh standard (σ ∝ λ⁻⁴)
#   - Ou lire numériquement la Fig. 4 de P.1621-2


# ===========================================================================
# § 3.2  Diffusion de Mie — Liang éqs. (6) et (7)
# ===========================================================================

def mie_extinction_ratio(lam_um: Union[float, np.ndarray],
                         hE_km: float) -> np.ndarray:
    """
    Taux d'extinction ρ par diffusion de Mie.
    Source : Liang et al. éq. (6), citant ITU-R P.1622 [14].

    Valide pour des stations au sol entre 0 et 5 km au-dessus du niveau
    de la mer, et pour des longueurs d'onde λ ∈ [0,8 ; 2] µm
    (soit f ∈ [150 ; 375] THz). En dehors de cette plage les polynômes
    extrapolent et peuvent donner ρ < 0 (Im > 1), physiquement impossible.

    Paramètres
    ----------
    lam_um : longueur d'onde dans le vide (µm)
    hE_km  : altitude de la station au sol au-dessus du niveau de la mer (km)

    Retour
    ------
    ρ (sans unité), clampé à 0 si négatif (hors domaine de validité)
    """
    lam = np.asarray(lam_um, dtype=float)
    hE  = float(hE_km)   # hE en km, conformément à l'équation source

    # Avertissement si hors domaine de validité
    if np.any(lam < 0.8) or np.any(lam > 2.0):
        import warnings
        warnings.warn(
            f"mie_extinction_ratio: λ={lam} µm hors domaine [0.8, 2.0] µm "
            f"(soit f hors [150, 375] THz). Les polynômes extrapolent.",
            RuntimeWarning, stacklevel=2
        )

    a = -0.000545 * lam**2 + 0.002  * lam - 0.0038
    b =  0.00628  * lam**2 - 0.0232 * lam + 0.00439
    c = -0.028    * lam**2 + 0.101  * lam - 0.18
    d = -0.228    * lam**3 + 0.922  * lam**2 - 1.26 * lam + 0.719

    rho = a * hE**3 + b * hE**2 + c * hE + d
    # Clamp : ρ < 0 signifie hors domaine → on force à 0 (pas d'extinction)
    return np.maximum(rho, 0.0)


def mie_attenuation(lam_um: Union[float, np.ndarray],
                    hE_km: float,
                    theta_E_deg: float) -> np.ndarray:
    """
    Atténuation atmosphérique par diffusion de Mie Im (sans unité, < 1).
    Source : Liang et al. éq. (7).

    Im = exp(−ρ / sin(θE))

    Paramètres
    ----------
    lam_um      : longueur d'onde dans le vide (µm)
    hE_km       : altitude de la station au sol (km), entre 0 et 5 km
    theta_E_deg : angle d'élévation (°)

    Retour
    ------
    Im (facteur d'atténuation linéaire, entre 0 et 1)
    """
    rho   = mie_extinction_ratio(lam_um, hE_km)
    theta = np.radians(theta_E_deg)
    return np.exp(-rho / np.sin(theta))

def mie_attenuation_lineique(lam_um: Union[float, np.ndarray],
                             hE_km: float,
                             theta_E_deg: float) -> np.ndarray:
    """
    Atténuation atmosphérique par diffusion de Mie Im (sans unité, < 1).
    Source : Liang et al. éq. (7).

    Im = exp(−ρ / sin(θE))

    Paramètres
    ----------
    lam_um      : longueur d'onde dans le vide (µm)
    hE_km       : altitude de la station au sol (km), entre 0 et 5 km
    theta_E_deg : angle d'élévation (°)

    Retour
    ------
    Im (facteur d'atténuation linéaire, entre 0 et 1)
    """
    d = 20 #km , épaisseur de la couche de Mie (troposphère) 
    Im  = mie_attenuation(lam_um, hE_km, theta_E_deg)
    return Im/d

# ===========================================================================
# Diffusion géométrique (brouillard/nuages denses) — Liang éqs. (8)–(10)
# ===========================================================================

def geometric_visibility_km(LW: float, N: float) -> float:
    """
    Visibilité V en km selon le modèle nuageux.
    Source : Liang et al. éq. (8).

    V = 1.002 / (LW * N)^0.6473

    Paramètres
    ----------
    LW : contenu en eau liquide (g/m⁻³)
    N  : concentration de gouttelettes nuageuses (cm⁻³)

    Retour
    ------
    V (km)
    """
    return 1.002 / (LW * N) ** 0.6473


def geometric_attenuation_coeff(lam_um: float,
                                V_km: float,
                                phi: float) -> float:
    """
    Coefficient d'atténuation géométrique θA (km⁻¹) selon le modèle de Kim.
    Source : Liang et al. éq. (9).

    θA = (3.91 / V) * (λ / 550)^(−φ)

    Paramètres
    ----------
    lam_um : longueur d'onde (µm) — la formule utilise λ en nm / 550 nm
    V_km   : visibilité (km)
    phi    : coefficient de taille de particule (modèle de Kim)

    Retour
    ------
    θA (km⁻¹)
    """
    lam_nm = lam_um * 1e3   # µm → nm
    return (3.91 / V_km) * (lam_nm / 550.0) ** (-phi)


def geometric_attenuation(theta_A: float, dA_km: float) -> float:
    """
    Atténuation par diffusion géométrique Ig (sans unité, < 1).
    Loi de Beer-Lambert — Source : Liang et al. éq. (10).

    Ig = exp(−θA * dA)

    Paramètres
    ----------
    theta_A : coefficient d'atténuation (km⁻¹)
    dA_km   : distance du faisceau dans la troposphère (km)

    Retour
    ------
    Ig (facteur d'atténuation linéaire)
    """
    return np.exp(-theta_A * dA_km)


def troposphere_path_length_km(hA_km: float,
                               hE_km: float,
                               theta_E_deg: float) -> float:
    """
    Distance du faisceau optique dans la couche troposphérique dA.
    Source : Liang et al. § III-B-2.

    dA = (hA − hE) / sin(θE)   [= (hA − hE) * csc(θE)]

    Paramètres
    ----------
    hA_km       : hauteur de la troposphère (km), typiquement 20 km
    hE_km       : altitude de la station au sol (km)
    theta_E_deg : angle d'élévation (°)

    Retour
    ------
    dA (km)
    """
    return (hA_km - hE_km) / np.sin(np.radians(theta_E_deg))


def atmospheric_attenuation_total(Im: float, Ig: float) -> float:
    """
    Atténuation atmosphérique totale LA (Mie + géométrique).
    Source : Liang et al. éq. (11).

    LA = Im * Ig

    Retour
    ------
    LA (facteur d'atténuation linéaire)
    """
    return Im * Ig


# ===========================================================================
# § 4.1  Indice effectif de réfraction atmosphérique — P.1621 éqs. (2) et (3)
# ===========================================================================

def indice_refraction_std(lam_vac_um: Union[float, np.ndarray]) -> np.ndarray:
    """
    Indice effectif de réfraction atmosphérique neff à T = 15 °C, P = 1013,25 hPa.
    Source : Rec. UIT-R P.1621-2 éq. (2).

    Valide pour f > 150 THz (λ < 2 µm).

    Paramètres
    ----------
    lam_vac_um : longueur d'onde dans le vide (µm)

    Retour
    ------
    neff (sans unité)
    """
    lam = np.asarray(lam_vac_um, dtype=float)
    neff = 1.0 + 1e-8 * (
        6432.8
        + 2_949_810.0 / (146.0 - lam ** -2)
        + 25_540.0   / (41.0  - lam ** -2)
    )
    return neff


def indice_refraction_TP(lam_vac_um: Union[float, np.ndarray],
                         T_C: float = 15.0,
                         P_hPa: float = 1013.25) -> np.ndarray:
    """
    Indice effectif de réfraction pour une température T et pression P données.
    Source : Rec. UIT-R P.1621-2 éq. (3).

    Paramètres
    ----------
    lam_vac_um : longueur d'onde dans le vide (µm)
    T_C        : température (°C)
    P_hPa      : pression atmosphérique (hPa)

    Retour
    ------
    neff (sans unité)
    """
    neff_std = indice_refraction_std(lam_vac_um)
    neff = 1.0 + (neff_std - 1.0) * (
        1.162 * P_hPa * (1.0 + P_hPa * (0.7868 - 0.0113 * T_C) * 1e-6)
        / (760.4696 * (1.0 + 0.0366 * T_C))
    )
    return neff


# ===========================================================================
# § 4.2  Angle d'élévation apparent — P.1621 éq. (4)
# ===========================================================================

def elevation_apparente_deg(theta_reel_deg: Union[float, np.ndarray],
                            lam_vac_um: float,
                            T_C: float = 15.0,
                            P_hPa: float = 1013.25) -> np.ndarray:
    """
    Angle d'élévation apparent d'un engin spatial vu depuis la Terre.
    Source : Rec. UIT-R P.1621-2 éq. (4).

    θ_obs = arccos( cos(θt) / neff(T,P) )

    Paramètres
    ----------
    theta_reel_deg : angle d'élévation réel (°)
    lam_vac_um     : longueur d'onde dans le vide (µm)
    T_C            : température (°C)
    P_hPa          : pression (hPa)

    Retour
    ------
    Angle d'élévation observé (°)
    """
    theta_reel = np.asarray(theta_reel_deg, dtype=float)
    neff = float(indice_refraction_TP(lam_vac_um, T_C, P_hPa))
    theta_obs_rad = np.arccos(np.cos(np.radians(theta_reel)) / neff)
    return np.degrees(theta_obs_rad)


# ===========================================================================
# § 5.1.1  Profil de turbulence Cn² — Modèle Huffnagel-Valley 5/7
#          P.1621 éqs. (5)–(7)
# ===========================================================================

def wind_rms(vg: float = 2.3) -> float:
    """
    Vitesse quadratique moyenne du vent le long du trajet vertical vrms.
    Source : Rec. UIT-R P.1621-2 éq. (5) (modèle de Bufton simplifié).

    vrms = sqrt(vg² + 30,69·vg + 348,91)   [m/s]

    Paramètre
    ---------
    vg : vitesse du vent au sol (m/s).
         Si inconnue, prendre vg = 2,3 m/s → vrms ≈ 21 m/s.

    Retour
    ------
    vrms (m/s)
    """
    return np.sqrt(vg ** 2 + 30.69 * vg + 348.91)


def Cn2_profile(h: Union[float, np.ndarray],
                vg: float = 2.3,
                C0: float = 1.7e-14) -> np.ndarray:
    """
    Paramètre de structure de la turbulence Cn² en fonction de l'altitude.
    Source : Rec. UIT-R P.1621-2 éq. (6) — modèle Huffnagel-Valley 5/7.

    Cn²(h) = 8,148×10⁻⁵⁶ · vrms² · h¹⁰ · exp(−h/1000)
           + 2,7×10⁻¹⁶ · exp(−h/1500)
           + C0 · exp(−h/100)

    Paramètres
    ----------
    h   : hauteur au-dessus du sol (m), scalaire ou tableau
    vg  : vitesse du vent au sol (m/s)
    C0  : valeur nominale de Cn² au sol (m⁻²/³), défaut 1,7×10⁻¹⁴

    Retour
    ------
    Cn² (m⁻²/³)
    """
    h    = np.asarray(h, dtype=float)
    vrms = wind_rms(vg)
    return (
        8.148e-56 * vrms ** 2 * h ** 10 * np.exp(-h / 1000.0)
        + 2.7e-16 * np.exp(-h / 1500.0)
        + C0      * np.exp(-h / 100.0)
    )


def layer_heights() -> np.ndarray:
    """
    Grille de hauteurs à pas exponentiellement croissant.
    Source : Rec. UIT-R P.1621-2 éq. (7).

    hi = exp((i−1) * ln(Z_TURB) / (N−1))   pour i = 1 à 139

    139 couches, de 1 m à Z_TURB = 20 000 m.

    Correction : le facteur original /20 ne montait qu'à ~1000 m.
    Le facteur correct est ln(20000)/138 ≈ 0.07176 pour couvrir
    l'intégralité de la couche turbulente jusqu'à Z_TURB.
    """
    i = np.arange(1, 140)
    factor = np.log(float(Z_TURB)) / 138.0   # ln(20000)/138 ≈ 0.07176
    return np.exp((i - 1) * factor)           # m, de 1 m à Z_TURB


# ===========================================================================
# § 5.1.2  Longueur de cohérence r0 — P.1621 éqs. (9)–(13)
# ===========================================================================

def coherence_length_r0(lam_um: float,
                        theta_deg: float,
                        h0_m: float = 0.0,
                        vg: float = 2.3,
                        C0: float = 1.7e-14) -> float:
    """
    Longueur de cohérence atmosphérique r0 (paramètre de Fried).
    Source : Rec. UIT-R P.1621-2 éqs. (9)–(13).

    Approximation valide pour h0 ∈ [0, 5] km et θ > 45°.

    Paramètres
    ----------
    lam_um    : longueur d'onde (µm)
    theta_deg : angle d'élévation (°)
    h0_m      : altitude de la station terrienne (m)
    vg        : vitesse du vent au sol (m/s)
    C0        : Cn² au sol (m⁻²/³)

    Retour
    ------
    r0 (m)
    """
    # Note : la constante 1.1654e-8 de l'éq. (13) est calibrée pour lam en µm.
    # Ne pas convertir en mètres ici.
    vrms  = wind_rms(vg)
    theta = np.radians(theta_deg)

    # Éq. (9) — terme vent
    C_wind = (8.148e-17 * vrms ** 2) * (
        0.0026 * (1.0 - np.exp((0.001 * h0_m ** 1.055) - 5.0)) + 3.587369
    )

    # Éq. (10) — terme altitude
    C_height = -6.5594e-19 + 4.05e-13 * np.exp(-h0_m / 1500.0)

    # Éq. (11) — terme turbulence de surface
    C_turb = -C0 * (1.383899e-85 - 100.0 * np.exp(-h0_m / 100.0))

    # Éq. (12) — intégrale approchée
    integral = C_wind + C_height + C_turb

    # Éq. (13) — lam_um directement (constante calibrée en µm)
    r0 = (1.1654e-8 * lam_um ** 1.2 * np.sin(theta) ** 0.6
          / max(integral, 1e-30) ** 0.6)
    return float(r0)


# ===========================================================================
# § 5.1.3  Angle isoplanétique θ0 — P.1621 éqs. (15)–(18)
# ===========================================================================

def isoplanatic_angle_theta0(lam_um: float,
                             theta_deg: float,
                             h0_m: float = 0.0,
                             vg: float = 2.3,
                             C0: float = 1.7e-14) -> float:
    """
    Angle isoplanétique θ0 (rad).
    Source : Rec. UIT-R P.1621-2 éqs. (15)–(18).

    Approximation valide pour h0 ∈ [0, 5] km et θ > 45°.

    Paramètres
    ----------
    lam_um    : longueur d'onde (µm)
    theta_deg : angle d'élévation (°)
    h0_m      : altitude de la station terrienne (m)
    vg        : vitesse du vent au sol (m/s)
    C0        : Cn² au sol (m⁻²/³)

    Retour
    ------
    θ0 (rad)
    """
    # Note : la constante 3.663e-9 est calibrée pour lam en µm.
    vrms  = wind_rms(vg)
    theta = np.radians(theta_deg)

    # Éq. (15)
    C_wind = (8.148e-10 * vrms ** 2) * (
        0.002 * (1.0 - np.exp((0.0018 * h0_m ** 1.014) - 9.0)) + 2.0043
    )

    # Éq. (16)
    C_height = (
        -7.0236e-23 * h0_m ** 4
        + 1.5015e-18 * h0_m ** 3
        - 8.9834e-15 * h0_m ** 2
        + 2.3855e-12 * h0_m
        + 9.6181e-8
    )

    # Éq. (17)
    C_turb = 3.3e5 * C0 * np.exp(-0.000222 * h0_m ** 1.45)

    # Éq. (18) — lam_um directement (constante calibrée en µm)
    integral = C_wind + C_height + C_turb
    theta0 = (3.663e-9 * lam_um ** 1.2 * np.sin(theta) ** 1.6
              / max(integral, 1e-30) ** 0.6)
    return float(theta0)


# ===========================================================================
# § 5.1.4  Constante de temps critique τ0 — P.1621 éqs. (19)–(21)
# ===========================================================================

def wind_profile(h: Union[float, np.ndarray], vg: float = 2.3) -> np.ndarray:
    """
    Profil horizontal de la vitesse du vent en fonction de l'altitude.
    Source : Rec. UIT-R P.1621-2 éq. (19).

    v(h) = vg + 30 · exp(−((h − 9400) / 4800)²)   [m/s]

    Paramètres
    ----------
    h  : hauteur au-dessus du sol (m)
    vg : vitesse du vent au sol (m/s).
         Défaut 2,3 m/s — unifié avec wind_rms() et les autres fonctions.
         (valeur type P.1621 : 2,8 m/s mais incohérente avec le reste du module)

    Retour
    ------
    v(h) (m/s)
    """
    h = np.asarray(h, dtype=float)
    return vg + 30.0 * np.exp(-((h - 9400.0) / 4800.0) ** 2)


def turbulence_time_constant_tau0(lam_um: float,
                                  theta_deg: float,
                                  h0_m: float = 0.0,
                                  vg: float = 2.8,
                                  C0: float = 1.7e-14) -> float:
    """
    Constante de temps critique de l'atmosphère τ0.
    Source : Rec. UIT-R P.1621-2 éqs. (19)–(21).

    Méthode : intégration numérique de v_{5/3} (éq. 20) sur la grille
    exponentielle de couches (éq. 7), puis application de l'éq. (21).

    Valide pour θ > 45°.

    Paramètres
    ----------
    lam_um    : longueur d'onde (µm)
    theta_deg : angle d'élévation (°)
    h0_m      : altitude de la station terrienne (m)
    vg        : vitesse du vent au sol (m/s)
    C0        : Cn² au sol (m⁻²/³)

    Retour
    ------
    τ0 (s)
    """
    # Note : la constante 2.729e-8 est calibrée pour lam en µm.
    theta = np.radians(theta_deg)

    heights = layer_heights()          # grille éq. (7), de 1 m à Z_TURB
    heights = heights[heights >= h0_m]

    cn2 = Cn2_profile(heights, vg, C0)    # éq. (6)
    v   = wind_profile(heights, vg)       # éq. (19)

    # Éq. (20) : v_{5/3} = ∫ Cn²(h) · (v(h))^(5/3) dh
    dh   = np.diff(heights, prepend=h0_m)
    v53  = np.sum(cn2 * v ** (5.0 / 3.0) * dh)

    # Éq. (21) — lam_um directement (constante calibrée en µm)
    tau0 = (2.729e-8 * lam_um ** 1.2 * np.sin(theta) ** 0.6
            / max(v53, 1e-30) ** 0.6)
    return float(tau0)


# ===========================================================================
# Distance de slant (liaison sol–satellite) — Liang éq. (4)
# ===========================================================================

def slant_distance_km(hS_km: float,
                      hE_km: float,
                      theta_E_deg: float,
                      RE_km: float = 6378.1) -> float:
    """
    Distance de slant dGS entre la station au sol et le satellite.
    Source : Liang et al. éq. (4).

    dGS = R · (sqrt(((R+H)/R)² − cos²(θE)) − sin(θE))
    avec R = RE + hE  et  H = hS − hE

    Paramètres
    ----------
    hS_km       : altitude du satellite (km)
    hE_km       : altitude de la station au sol (km)
    theta_E_deg : angle d'élévation (°)
    RE_km       : rayon terrestre (km), défaut 6378,1 km

    Retour
    ------
    dGS (km)
    """
    R     = RE_km + hE_km
    H     = hS_km - hE_km
    theta = np.radians(theta_E_deg)
    return R * (np.sqrt(((R + H) / R) ** 2 - np.cos(theta) ** 2) - np.sin(theta))


# ===========================================================================
# Grilles de calcul pour visualisation 2D/3D
# ===========================================================================

def compute_grid_attenuation(freq_THz_min: float,
                              freq_THz_max: float,
                              n_freq: int,
                              hE_km_min: float,
                              hE_km_max: float,
                              n_altitude: int,
                              theta_E_deg: float = 40.0,
                              hA_km: float = 20.0,
                              LW: float = 0.5,
                              N: float = 3.128e-4,
                              phi: float = 1.6) -> dict:
    """
    Calcule l'atténuation atmosphérique totale sur une grille 2D :
    fréquence × altitude de station.

    Retourne des données structurées pour visualisation 2D ou 3D.

    Paramètres
    ----------
    freq_THz_min, freq_THz_max : plage de fréquences (THz)
    n_freq : nombre de points de fréquence
    hE_km_min, hE_km_max : plage d'altitudes (km)
    n_altitude : nombre de points d'altitude
    theta_E_deg : angle d'élévation (°)
    hA_km : hauteur troposphère (km)
    LW, N, phi : paramètres diffusion géométrique

    Retour
    ------
    dict avec clés :
      'freq_THz' : array de fréquences (THz)
      'altitude_km' : array d'altitudes (km)
      'wavelength_um' : array de longueurs d'onde (µm)
      'attenuation_linear' : matrice 2D (n_altitude × n_freq)
      'attenuation_dB' : matrice 2D en dB
      'mie_linear' : matrice Mie
      'geometric_linear' : matrice géométrique
    """
    freq = np.linspace(freq_THz_min, freq_THz_max, n_freq)
    alt = np.linspace(hE_km_min, hE_km_max, n_altitude)

    wavelength = freq_THz_to_wavelength_um(freq)

    attenuation_lin = np.zeros((n_altitude, n_freq))
    mie_lin = np.zeros((n_altitude, n_freq))
    geom_lin = np.zeros((n_altitude, n_freq))
    mie_lin_km = np.zeros((n_altitude, n_freq))

    for i, h in enumerate(alt):
        for j, lam in enumerate(wavelength):
            rho = mie_extinction_ratio(lam, h)
            Im = mie_attenuation(lam, h, theta_E_deg)
            Im_km = mie_attenuation_lineique(lam, h, theta_E_deg) 
            mie_lin[i, j] = Im
            mie_lin_km[i, j] = Im_km


            dA = troposphere_path_length_km(hA_km, h, theta_E_deg)
            V = geometric_visibility_km(LW, N)
            tA = geometric_attenuation_coeff(lam, V, phi)
            Ig = geometric_attenuation(tA, dA)
            geom_lin[i, j] = Ig

            LA = atmospheric_attenuation_total(Im, Ig)
            attenuation_lin[i, j] = LA

    return {
        'freq_THz': freq,
        'altitude_km': alt,
        'wavelength_um': wavelength,
        'attenuation_linear': attenuation_lin,
        'attenuation_dB': 10 * np.log10(attenuation_lin),
        'mie_linear': mie_lin,
        'mie_dB': 10 * np.log10(mie_lin),
        'mie_km_dB': 10*np.log10(mie_lin_km),
        'geometric_linear': geom_lin,
        'geometric_dB': 10 * np.log10(geom_lin),
    }


def compute_grid_turbulence(freq_THz_min: float,
                             freq_THz_max: float,
                             n_freq: int,
                             hE_km_min: float,
                             hE_km_max: float,
                             n_altitude: int,
                             theta_E_deg: float = 40.0,
                             vg: float = 2.3,
                             C0: float = 1.7e-14) -> dict:
    """
    Calcule les paramètres de turbulence sur une grille 2D :
    fréquence × altitude de station.

    Paramètres turbulence : r0, θ0, τ0

    Retour
    ------
    dict avec clés :
      'freq_THz', 'altitude_km', 'wavelength_um' : axes
      'r0_cm', 'theta0_urad', 'tau0_ms' : matrices 2D
    """
    freq = np.linspace(freq_THz_min, freq_THz_max, n_freq)
    alt = np.linspace(hE_km_min, hE_km_max, n_altitude)

    wavelength = freq_THz_to_wavelength_um(freq)

    r0_array = np.zeros((n_altitude, n_freq))
    theta0_array = np.zeros((n_altitude, n_freq))
    tau0_array = np.zeros((n_altitude, n_freq))

    for i, h_km in enumerate(alt):
        h_m = h_km * 1000.0
        for j, lam in enumerate(wavelength):
            r0_array[i, j] = coherence_length_r0(lam, theta_E_deg, h_m, vg, C0) * 100.0  # cm
            theta0_array[i, j] = isoplanatic_angle_theta0(lam, theta_E_deg, h_m, vg, C0) * 1e6  # µrad
            tau0_array[i, j] = turbulence_time_constant_tau0(lam, theta_E_deg, h_m, vg, C0) * 1e3  # ms

    return {
        'freq_THz': freq,
        'altitude_km': alt,
        'wavelength_um': wavelength,
        'r0_cm': r0_array,
        'theta0_urad': theta0_array,
        'tau0_ms': tau0_array,
    }


# ===========================================================================
# POINT D'ENTRÉE — démo
# ===========================================================================

if __name__ == '__main__':

    print("=" * 65)
    print("UIT-R P.1621-2 + Liang et al. — Propagation optique/IR Terre-espace")
    print("=" * 65)

    # Paramètres de démonstration (Table 3 de Liang et al.)
    lam_um      = 1.55    # µm  (1550 nm)
    hE_km       = 1.0     # km  (altitude station)
    theta_E_deg = 90.0    # °   (angle d'élévation)
    hA_km       = 20.0    # km  (hauteur troposphère)
    hS_km       = 550.0   # km  (altitude satellite)
    LW          = 0.5     # g/m⁻³
    N           = 3.128e-4 # cm⁻³
    phi         = 1.6

    print(f"\n[Paramètres] λ = {lam_um} µm | hE = {hE_km} km | θE = {theta_E_deg}°")

    # Diffusion de Mie
    rho = mie_extinction_ratio(lam_um, hE_km)
    Im  = mie_attenuation(lam_um, hE_km, theta_E_deg)
    print(f"\n  Taux d'extinction Mie ρ  : {rho:.4f}")
    print(f"  Atténuation Mie Im       : {Im:.4f}  ({10*np.log10(Im):.2f} dB)")

    # Diffusion géométrique
    dA = troposphere_path_length_km(hA_km, hE_km, theta_E_deg)
    V  = geometric_visibility_km(LW, N)
    tA = geometric_attenuation_coeff(lam_um, V, phi)
    Ig = geometric_attenuation(tA, dA)
    print(f"\n  Visibilité V             : {V:.2f} km")
    print(f"  Chemin troposphérique dA : {dA:.2f} km")
    print(f"  Atténuation géom. Ig     : {Ig:.4f}  ({10*np.log10(Ig):.2f} dB)")

    # Atténuation totale
    LA = atmospheric_attenuation_total(Im, Ig)
    print(f"\n  Atténuation totale LA    : {LA:.4f}  ({10*np.log10(LA):.2f} dB)")

    # Distance de slant
    dGS = slant_distance_km(hS_km, hE_km, theta_E_deg)
    print(f"\n  Distance de slant dGS   : {dGS:.1f} km")

    # Indice de réfraction
    neff = indice_refraction_std(lam_um)
    print(f"\n  Indice de réfraction neff (std) : {neff:.8f}")

    # Angle d'élévation apparent
    theta_app = elevation_apparente_deg(theta_E_deg, lam_um)
    print(f"  Angle apparent           : {theta_app:.6f}°  "
          f"(Δ = {theta_app - theta_E_deg:.2e}°)")

    # Turbulences
    r0   = coherence_length_r0(lam_um, theta_E_deg, hE_km * 1000)
    t0   = isoplanatic_angle_theta0(lam_um, theta_E_deg, hE_km * 1000)
    tau0 = turbulence_time_constant_tau0(lam_um, theta_E_deg, hE_km * 1000)
    print(f"\n  Longueur de cohérence r0 : {r0*100:.2f} cm")
    print(f"  Angle isoplanétique  θ0 : {t0*1e6:.2f} µrad")
    print(f"  Constante de temps   τ0 : {tau0*1e3:.2f} ms")

    print("\n" + "=" * 65)
    print("NOTE : L'atténuation Rayleigh en dB/km n'est pas implémentée.")
    print("Aucune formule analytique n'est fournie dans les deux documents.")
    print("Cf. commentaire en tête de § 3.1 dans ce fichier.")
    print("=" * 65)
