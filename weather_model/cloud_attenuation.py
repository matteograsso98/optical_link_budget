"""
Cloud attenuation time-series synthesizer (Ka-band to optical frequencies).

Implementation of:
  N. K. Lyras, C. I. Kourogiorgas, A. D. Panagopoulos, "Cloud Attenuation
  Statistics Prediction From Ka-Band to Optical Frequencies: Integrated
  Liquid Water Content Field Synthesizer", IEEE Trans. Antennas Propag.,
  vol. 65, no. 1, pp. 319-328, Jan. 2017.

Step-by-step algorithm (paper, p. 323):
  1)  m, sigma, P_CLW from ITU-R P.840 (here: SG3 validation workbook, 45N/0E)
  2)  spatial correlation rho_ij, eq. (7)  [Luini & Capsoni MODIS model]
  3)  matrices B, C, S (eqs. 3, 8, 9; S via Cholesky of 2*beta*C)
  4)  n-D Wiener process
  5)  ILWC time series L_i(t), eq. (12)
  6)  c1, c2 parameters, eq. (16)
  7)  cloud vertical extent dh from LWC profile, eqs. (15), (17), Table I
  8)  PSD parameters a, b, gamma (Table II)
  9)  g(h, t), eq. (20)
 10)  sigma_ext(r, lambda)  [Mie theory]
 11)  A(t, lambda), eq. (21)

Key implementation notes
------------------------
* The SDE (1)-(2) is an n-D Ornstein-Uhlenbeck process; its exact
  discretization is the one-pole recursion of ITU-R P.1853-1 sec. 4
  (X_k = rho X_{k-1} + sqrt(1-rho^2) * w) with spatially correlated noise
  w = chol(C) @ z.  The same white-noise vector feeds both the fast and the
  slow component, as in P.1853-1 / eq. (6) of the paper.
* X1, X2 are initialized from their exact joint stationary distribution
  (instead of zero + 500k discarded warm-up samples). The stationary
  cross-covariance of the shared-noise pair is c12 = a1*a2/(1-rho1*rho2),
  a_k = sqrt(1-rho_k^2), so the stacked covariance is
  [[1, c12], [c12, 1]] (x) C and is sampled with a Kronecker Cholesky.
* For a fixed cloud type, the modified-gamma PSD (18) has fixed (a, b, gamma)
  and g is linear in LWC (eq. 20).  Hence the Mie integral collapses to a
  per-(wavelength, cloud-type) mass-extinction constant
      K [dB/km per g/m^3],  beta_ext = K * w.
* With beta_ext linear in w, the slant-path integral (21) through each grid
  column has a closed form: the vertical LWC profile (15) is L times a gamma
  PDF in (h - h0), so the column contribution is
      A_col = K * L / sin(el) * [GammaCDF(h_out - h0) - GammaCDF(h_in - h0)]
  evaluated with the regularized incomplete gamma function (no numerical
  path discretization needed).
* h0 (cloud base) is drawn from the CloudSat GEV (17) once per cloud event
  per grid column and held until the event ends.
* dh(L) and hence the cloud type are deterministic functions of L; both are
  precomputed on a lookup table.
"""

import numpy as np
from scipy import signal, special
from scipy.optimize import brentq
from scipy.stats import norm
import miepython

# ----------------------------------------------------------------------
# ITU-R P.1853-1 sec. 4.2 cloud dynamic parameters
BETA1 = 7.17e-4    # s^-1 (fast component)
BETA2 = 2.01e-5    # s^-1 (slow component)
GAMMA1 = 0.349
GAMMA2 = 0.830

# CloudSat GEV cloud-base-height PDF, paper eq. (17)
GEV_XI, GEV_S, GEV_MU = 0.484, 0.582, 0.987     # km a.m.s.l.
H0_MAX = 12.0                                   # km, resample heavy tail above

# Table I / II: cloud types ordered by vertical extent
CLOUD_TYPES = ("stratus", "nimbostratus", "cumulus", "cumulonimbus")
VERT_EXTENT = {"cumulonimbus": 2.5, "cumulus": 2.5,
               "nimbostratus": 1.4, "stratus": 0.7}      # km (Cb is "> 2.5")
PSD_ABG = {                                              # a, b, gamma (Table II)
    "cumulonimbus": (3.0, 0.5,   1.0),
    "cumulus":      (3.0, 0.5,   1.0),
    "nimbostratus": (2.0, 0.425, 1.0),
    "stratus":      (2.0, 0.6,   1.0),
}
# classification thresholds on dh (midpoints between Table I values)
DH_EDGES = (1.05, 1.95, 2.5)   # St | Nb | Cu | Cb

RHO_W = 1.0  # g/cm^3, liquid water density


# ======================================================================
#  Dielectric / refractive properties of liquid water
# ======================================================================
def water_permittivity(f_ghz, T=273.15):
    """Double-Debye complex permittivity of liquid water (ITU-R P.840,
    Liebe model). Returns (eps_prime, eps_double_prime)."""
    th = 300.0 / T
    eps0 = 77.66 + 103.3 * (th - 1.0)
    eps1 = 0.0671 * eps0
    eps2 = 3.52
    fp = 20.20 - 146.0 * (th - 1.0) + 316.0 * (th - 1.0) ** 2   # GHz
    fs = 39.8 * fp
    epsp = (eps0 - eps1) / (1 + (f_ghz / fp) ** 2) \
         + (eps1 - eps2) / (1 + (f_ghz / fs) ** 2) + eps2
    epspp = f_ghz * (eps0 - eps1) / (fp * (1 + (f_ghz / fp) ** 2)) \
          + f_ghz * (eps1 - eps2) / (fs * (1 + (f_ghz / fs) ** 2))
    return epsp, epspp


def water_refractive_index(wavelength_m, T=273.15):
    """Complex refractive index m = n - j*k of liquid water.

    RF/mm-wave (lambda > 100 um): from the double-Debye permittivity.
    Optical/NIR: tabulated values (Hale & Querry 1973) around common
    laser-com wavelengths.
    """
    if wavelength_m > 100e-6:
        f_ghz = 299792458.0 / wavelength_m * 1e-9
        epsp, epspp = water_permittivity(f_ghz, T)
        m = np.sqrt(complex(epsp, -epspp))
        return complex(m.real, -abs(m.imag))
    table = {  # wavelength_um: (n, k)
        0.532:  (1.3336, 1.4e-9),
        0.690:  (1.3308, 4.2e-8),
        0.850:  (1.3287, 2.93e-7),
        1.064:  (1.3261, 1.4e-6),
        1.550:  (1.3180, 1.0e-4),
        10.6:   (1.1790, 6.21e-2),
    }
    wl_um = wavelength_m * 1e6
    keys = np.array(sorted(table))
    key = keys[np.argmin(np.abs(np.log(keys / wl_um)))]
    n, k = table[key]
    return complex(n, -k)


def rayleigh_kl(f_ghz, T=273.15):
    """ITU-R P.840-6 Rayleigh specific attenuation coefficient
    Kl [(dB/km)/(g/m^3)]  ==  [dB/(kg/m^2)] for the columnar content."""
    epsp, epspp = water_permittivity(f_ghz, T)
    eta = (2.0 + epsp) / epspp
    return 0.819 * f_ghz / (epspp * (1.0 + eta * eta))


# ======================================================================
#  Mie mass-extinction constants  K(lambda, cloud type)
# ======================================================================
def mass_extinction_constant(wavelength_m, cloud_type, T=273.15, nr=4000):
    """K [dB/km per g/m^3]: specific extinction of a cloud with the
    modified-gamma PSD of `cloud_type` (Table II), per unit LWC.

    beta_ext = 4.343e3 * Int sigma_ext(r) n(r) dr   (paper eq. 14)
    n(r) = g r^a exp(-b r^gamma)                    (paper eq. 18)
    w    = (4/3) pi rho_w Int r^3 n(r) dr           (paper eq. 19)
    K    = beta_ext / w   (independent of g; eq. 20 makes g linear in w)

    Internally everything is reduced to ratios of PSD moments weighted by
    the Mie extinction efficiency Q_ext:
        K = 3257.25 / rho_w * <Q_ext r^2> / <r^3>      (r in um)
    """
    a, b, gam = PSD_ABG[cloud_type]
    m = water_refractive_index(wavelength_m, T)
    # radius grid (um): resolve both the PSD and the Mie oscillations
    r = np.linspace(1e-3, 150.0, nr)
    x = 2 * np.pi * r / (wavelength_m * 1e6)         # size parameter
    qext, _, _, _ = miepython.efficiencies_mx(m, x)
    psd = r ** a * np.exp(-b * r ** gam)             # unnormalized n(r)
    num = np.trapezoid(qext * r ** 2 * psd, r)
    den = np.trapezoid(r ** 3 * psd, r)
    # 4.343e-3 * pi * Int Qext r^2 n dr  /  [(4/3) pi rho_w 1e-6 Int r^3 n dr]
    return 4.343e-3 * 1e6 * 3.0 / (4.0 * RHO_W) * num / den


# ======================================================================
#  Vertical structure: c1, c2, dh(L), cloud type(L)   (eqs. 15-17)
# ======================================================================
def c1_c2(L):
    """Shape/scale parameters of the LWC vertical profile, eq. (16)."""
    L = np.asarray(L, dtype=float)
    c1 = 4.27 * np.exp(-4.93 * (L + 0.06)) \
       + 54.12 * np.exp(-61.25 * (L + 0.06)) + 1.71
    c2 = 3.17 * c1 ** (-3.04) + 0.074
    return c1, c2


def _gamma_pdf(xx, c1, c2):
    return np.exp((c1 - 1) * np.log(xx) - xx / c2
                  - c1 * np.log(c2) - special.gammaln(c1))


def _dh_single(L):
    """Vertical extent dh = h_th - h0 for a single ILWC value.

    h_th is the upper height where w(h,t) = 0.06 * L(t) (SMOC remark used in
    the paper); since w = L * gammapdf(h - h0; c1, c2), the condition is
    gammapdf(x) = 0.06 [1/km], independent of L itself except through c1, c2.
    """
    c1, c2 = c1_c2(L)
    mode = max((c1 - 1.0) * c2, 1e-9)
    f = lambda xx: _gamma_pdf(xx, c1, c2) - 0.06
    if f(mode) <= 0.0:                # whole profile below threshold
        return 0.0
    hi = mode + c2
    while f(hi) > 0.0:
        hi += c2
    return brentq(f, mode, hi)


class VerticalStructure:
    """Lookup tables dh(L) and cloud-type(L) on a fine ILWC grid."""

    def __init__(self, l_max=20.0, n=3000):
        self.lgrid = np.linspace(1e-4, l_max, n)
        self.dh = np.array([_dh_single(l) for l in self.lgrid])
        self.type_idx = np.searchsorted(DH_EDGES, self.dh, side="right")

    def vertical_extent(self, L):
        return np.interp(L, self.lgrid, self.dh)

    def cloud_type_index(self, L):
        """0=stratus, 1=nimbostratus, 2=cumulus, 3=cumulonimbus."""
        dh = self.vertical_extent(L)
        return np.searchsorted(DH_EDGES, dh, side="right")


# ======================================================================
#  ILWC field synthesizer (paper sec. II / ITU-R P.1853-1 sec. 4)
# ======================================================================
def spatial_correlation(d_km):
    """MODIS spatial correlation of ILWC, eq. (7) (valid d < 250 km)."""
    d = np.asarray(d_km, dtype=float)
    return 0.35 * np.exp(-d / 7.8) + 0.65 * np.exp(-d / 225.3)


class IlwcFieldSynthesizer:
    """Time series of spatially correlated ILWC on a 1-D row of n grid
    columns (1 km spacing) under the slant path.

    Parameters
    ----------
    m, sigma : conditional log-normal parameters of ILWC [ln kg/m^2]
    p_clw    : probability of cloud (fraction, 0..1)
    n_cols   : number of 1x1 km grid columns
    ts       : sampling period (s)
    """

    def __init__(self, m, sigma, p_clw, n_cols, ts=1.0, seed=None):
        self.m, self.sigma, self.p_clw = m, sigma, p_clw
        self.n, self.ts = n_cols, ts
        self.alpha_th = norm.isf(p_clw)                       # eq. 12
        self.rng = np.random.default_rng(seed)

        d = np.abs(np.subtract.outer(np.arange(n_cols), np.arange(n_cols)))
        self.C = spatial_correlation(d.astype(float))         # eq. 8
        self.Lc = np.linalg.cholesky(self.C)                  # S = chol(2 b C)/...

        self.rho1 = np.exp(-BETA1 * ts)
        self.rho2 = np.exp(-BETA2 * ts)
        self.a1 = np.sqrt(1.0 - self.rho1 ** 2)
        self.a2 = np.sqrt(1.0 - self.rho2 ** 2)

        # exact stationary init of (X1, X2) with shared driving noise
        c12 = self.a1 * self.a2 / (1.0 - self.rho1 * self.rho2)
        L2 = np.linalg.cholesky(np.array([[1.0, c12], [c12, 1.0]]))
        z = self.rng.standard_normal(2 * n_cols)
        X = np.kron(L2, self.Lc) @ z
        self.X1, self.X2 = X[:n_cols].copy(), X[n_cols:].copy()

    def synthesize_gaussian(self, n_samples):
        """Advance the OU pair and return G (n_cols x n_samples), eq. (11)."""
        z = self.rng.standard_normal((self.n, n_samples))
        w = self.Lc @ z                                       # spatial mixing
        X1, zf1 = signal.lfilter([self.a1], [1.0, -self.rho1], w, axis=1,
                                 zi=(self.rho1 * self.X1)[:, None])
        X2, zf2 = signal.lfilter([self.a2], [1.0, -self.rho2], w, axis=1,
                                 zi=(self.rho2 * self.X2)[:, None])
        self.X1, self.X2 = X1[:, -1].copy(), X2[:, -1].copy()
        return GAMMA1 * X1 + GAMMA2 * X2

    def gaussian_to_ilwc(self, G):
        """Memoryless non-linearity, eq. (12).  Returns L [kg/m^2]."""
        L = np.zeros_like(G)
        mask = G > self.alpha_th
        q = norm.sf(G[mask]) / self.p_clw                     # in (0, 1)
        L[mask] = np.exp(norm.isf(q) * self.sigma + self.m)
        return L

    def synthesize_ilwc(self, n_samples):
        return self.gaussian_to_ilwc(self.synthesize_gaussian(n_samples))


# ======================================================================
#  Cloud-base sampler: one h0 per cloud event per column
# ======================================================================
class CloudBaseSampler:
    """Draws h0 from the GEV (17) at each 0 -> cloud transition of every
    grid column and holds it constant for the event duration; keeps state
    across chunks."""

    def __init__(self, n_cols, rng):
        self.rng = rng
        self.in_cloud = np.zeros(n_cols, dtype=bool)
        self.h0_cur = np.full(n_cols, np.nan)

    def _draw(self, size):
        # scipy genextreme uses c = -xi
        out = np.empty(size)
        need = np.ones(size, dtype=bool)
        while need.any():
            u = self.rng.random(need.sum())
            # inverse CDF of GEV (xi != 0)
            x = GEV_MU + GEV_S / GEV_XI * ((-np.log(u)) ** (-GEV_XI) - 1.0)
            out[need] = x
            need[need] = (x < 0.0) | (x > H0_MAX)
        return out

    def sample(self, cloudy):
        """cloudy: bool array (n_cols x nt). Returns h0 (n_cols x nt),
        NaN where no cloud."""
        n, nt = cloudy.shape
        h0 = np.full((n, nt), np.nan)
        for i in range(n):
            ci = cloudy[i]
            prev = self.in_cloud[i]
            starts = np.flatnonzero(ci & ~np.concatenate(([prev], ci[:-1])))
            # segment boundaries: each start begins a new event
            bounds = np.concatenate((starts, [nt]))
            if prev and ci[0]:
                # event continuing from previous chunk
                stop = bounds[0] if len(starts) else nt
                h0[i, :stop][ci[:stop]] = self.h0_cur[i]
            for s, e in zip(bounds[:-1], bounds[1:]):
                hb = self._draw(1)[0]
                seg = slice(s, e)
                h0[i, seg][ci[seg]] = hb
                self.h0_cur[i] = hb
            self.in_cloud[i] = ci[-1]
        return h0


# ======================================================================
#  Slant-path cloud attenuation synthesizer (paper sec. III)
# ======================================================================
class CloudAttenuationSynthesizer:
    """End-to-end synthesizer: ILWC field -> LWC vertical profiles ->
    slant-path attenuation time series at one or more wavelengths.

    Parameters
    ----------
    m, sigma, p_clw : ILWC log-normal parameters (P.840)
    elevation_deg   : link elevation angle
    station_alt_km  : ground-station altitude a.m.s.l.
    wavelengths_m   : iterable of wavelengths (m)
    ts              : sampling period (s)
    """

    def __init__(self, m, sigma, p_clw, elevation_deg, station_alt_km,
                 wavelengths_m, ts=1.0, seed=None, max_cloud_top_km=16.0):
        self.el = np.deg2rad(elevation_deg)
        self.hg = station_alt_km
        self.wavelengths = list(wavelengths_m)
        self.ts = ts

        # grid columns crossed by the path up to the max cloud top
        n_cols = int(np.ceil((max_cloud_top_km - station_alt_km)
                             / np.tan(self.el))) + 1
        self.n_cols = n_cols

        self.field = IlwcFieldSynthesizer(m, sigma, p_clw, n_cols, ts, seed)
        self.bases = CloudBaseSampler(n_cols, self.field.rng)
        self.vs = VerticalStructure()

        # Mie mass-extinction constants K[wavelength][cloud type]
        self.K = np.array([[mass_extinction_constant(wl, ct)
                            for ct in CLOUD_TYPES] for wl in self.wavelengths])

        # heights at which the path enters/leaves each column
        j = np.arange(n_cols, dtype=float)
        self.h_in = station_alt_km + j * np.tan(self.el)
        self.h_out = station_alt_km + (j + 1) * np.tan(self.el)

    # ------------------------------------------------------------------
    def synthesize(self, n_samples, chunk=200_000, return_ilwc=False,
                   progress=False):
        """Synthesize attenuation time series.

        Returns
        -------
        A : array (n_wavelengths x n_samples)  [dB]
        L0 : ILWC at the station column (if return_ilwc)
        """
        nw = len(self.wavelengths)
        A = np.empty((nw, n_samples))
        L0 = np.empty(n_samples) if return_ilwc else None
        done = 0
        while done < n_samples:
            nt = min(chunk, n_samples - done)
            Ablk, Lblk = self._chunk(nt)
            A[:, done:done + nt] = Ablk
            if return_ilwc:
                L0[done:done + nt] = Lblk
            done += nt
            if progress:
                print(f"  {done/n_samples:6.1%}", flush=True)
        return (A, L0) if return_ilwc else A

    # ------------------------------------------------------------------
    def _chunk(self, nt):
        L = self.field.synthesize_ilwc(nt)            # (n_cols, nt) kg/m^2
        cloudy = L > 0.0
        h0 = self.bases.sample(cloudy)                # (n_cols, nt)

        c1, c2 = c1_c2(L)
        dh = self.vs.vertical_extent(L)
        tidx = np.searchsorted(DH_EDGES, dh, side="right")    # cloud type
        hth = h0 + dh

        # clip path segment of each column to the cloud layer
        hi = np.maximum(self.h_in[:, None], h0)
        ho = np.minimum(self.h_out[:, None], hth)
        valid = cloudy & (ho > hi)

        # column attenuation factor: L/sin(el) * [P(c1,(ho-h0)/c2)-P(c1,(hi-h0)/c2)]
        frac = np.zeros_like(L)
        if valid.any():
            c1v, c2v = c1[valid], c2[valid]
            x_hi = (ho[valid] - h0[valid]) / c2v
            x_lo = (hi[valid] - h0[valid]) / c2v
            frac[valid] = special.gammainc(c1v, x_hi) - special.gammainc(c1v, x_lo)
        col_water = L * frac / np.sin(self.el)        # kg/m^2 along the path

        # sum over columns with per-type Mie constants per wavelength
        A = np.zeros((len(self.wavelengths), nt))
        for iw in range(len(self.wavelengths)):
            Kmap = self.K[iw][tidx]                   # (n_cols, nt)
            A[iw] = np.sum(Kmap * col_water, axis=0)
        return A, L[0]


# ======================================================================
#  Analytic references for benchmarking
# ======================================================================
def ilwc_ccdf_analytic(x, m, sigma, p_clw):
    """P(L > x) for the conditional log-normal ILWC model."""
    return p_clw * norm.sf((np.log(x) - m) / sigma)


def itu_p840_attenuation(p_exceed, kl, m, sigma, p_clw, elevation_deg):
    """ITU-R P.840 slant-path cloud attenuation exceeded at probability p:
    A(p) = Kl * L(p) / sin(el),  L(p) = exp(m + sigma Q^-1(p / P_CLW))."""
    p = np.asarray(p_exceed, dtype=float)
    L = np.where(p < p_clw,
                 np.exp(m + sigma * norm.isf(np.minimum(p / p_clw, 1.0))),
                 0.0)
    return kl * L / np.sin(np.deg2rad(elevation_deg))
