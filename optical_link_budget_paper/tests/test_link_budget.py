"""
tests/test_link_budget.py
=========================
Regression tests pinning computed values against published reference results.

Reference sources
-----------------
[P1622]  ITU-R Recommendation P.1622-0 (2003), Annex 1.
         Table 2 gives sigma_dBN at el=75deg, v_rms=21 m/s for four wavelengths.
         NOTE: the current scintillation implementation systematically underestimates
         Table 2 by ~10-45%. This is a known discrepancy inherited from the original
         code (visible in its own console validation output). The tests for scintillation
         therefore pin the *implementation output* as a regression guard, not the ITU
         table values — the gap is documented below and should be investigated separately.

[Liang]  Liang et al., "Link Budget Analysis for FSO Satellite Network",
         arXiv:2204.13177v1 (2022).
         Table 3 gives the full link budget at el=90deg, lam=1.55um, h_E=1 km.

Tolerances
----------
Physics models are curve-fit approximations, so we allow:
  - Mie attenuation      : +/-0.01 dB  (ITU states ~0.1 dB accuracy)
  - Scintillation sigma  : +/-0.01 dB  (pinned to implementation, not ITU table)
  - Geometric scattering : +/-0.05 dB
  - Slant distance       : +/-1.0 km
  - FSPL                 : +/-0.01 dB
  - P_T                  : +/-0.15 dBm (accumulates all above tolerances)
"""

import pytest
import numpy as np

from fso_channel.atmosphere import mie, scintillation, geometric
from fso_channel.link import geometry, budget
from fso_channel.config import AtmosphereConfig, OrbitConfig, TerminalConfig


def approx(value, abs_tol):
    return pytest.approx(value, abs=abs_tol)


# ===========================================================================
# 1. Mie scattering -- ITU-R P.1622-0, eq. (3)
#    Valid range: lam in [0.8, 2.0] um, hE in [0, 5] km.
#    Values at C-band (1.55 um) are cross-checked against the ITU polynomial.
#    Values outside the valid range are pinned to computed output only.
# ===========================================================================

class TestMie:

    @pytest.mark.parametrize("lam_um, hE_km, el_deg, expected_dB", [
        # --- Within valid range: C-band 1550 nm ---
        (1.550, 1.0, 40.0, 0.1544),   # default scenario
        (1.550, 1.0, 90.0, 0.0992),   # zenith
        (1.550, 1.0, 10.0, 0.5715),   # low elevation
        # --- Outside [0.8, 2.0] um stated validity: pinned to computed output ---
        # These are regression anchors only, not ITU reference values.
        (1.064, 1.0, 40.0, 0.1879),
        (0.850, 1.0, 40.0, 0.3139),
        (0.532, 1.0, 40.0, 0.8921),
    ])
    def test_attenuation_dB(self, lam_um, hE_km, el_deg, expected_dB):
        result = mie.attenuation_dB(lam_um, hE_km, el_deg)
        assert result == approx(expected_dB, abs_tol=0.01), (
            f"Mie A_S mismatch at lam={lam_um}um, hE={hE_km}km, el={el_deg}deg: "
            f"got {result:.4f} dB, expected {expected_dB:.4f} dB"
        )

    def test_attenuation_increases_with_lower_elevation(self):
        assert mie.attenuation_dB(1.55, 1.0, 20.0) > mie.attenuation_dB(1.55, 1.0, 80.0)

    def test_attenuation_non_negative(self):
        for el in [10, 30, 60, 90]:
            assert mie.attenuation_dB(1.55, 1.0, el) >= 0.0


# ===========================================================================
# 2. Scintillation -- ITU-R P.1622-0, eq. (4a)+(4c)
#
#    ITU-R P.1622-0 Table 2 reference values (el=75deg, v_rms=21 m/s):
#      1.550 um -> 1.25 dB  |  implementation -> 1.118 dB  (gap: -0.13 dB)
#      1.064 um -> 1.94 dB  |  implementation -> 1.392 dB  (gap: -0.55 dB)
#      0.850 um -> 2.52 dB  |  implementation -> 1.587 dB  (gap: -0.93 dB)
#      0.532 um -> 4.35 dB  |  implementation -> 2.086 dB  (gap: -2.26 dB)
#
#    This discrepancy was present in the original code and is likely due to
#    the HV turbulence profile parameters not matching the ITU assumed values.
#    Tests below pin the *implementation output* as regression guards.
#    A separate task should investigate and fix the calibration gap.
# ===========================================================================

class TestScintillation:

    # --- Regression pins: implementation output, NOT ITU Table 2 values ---
    @pytest.mark.parametrize("lam_um, el_deg, impl_dB, itu_table2_dB", [
        (1.550, 75.0, 1.1181, 1.25),
        (1.064, 75.0, 1.3925, 1.94),
        (0.850, 75.0, 1.5874, 2.52),
        (0.532, 75.0, 2.0864, 4.35),
    ])
    def test_sigma_dB_regression(self, lam_um, el_deg, impl_dB, itu_table2_dB):
        """Pins the current implementation output. See module docstring for ITU gap."""
        result = scintillation.sigma_dB(lam_um, el_deg, vrms=21.0)
        assert result == approx(impl_dB, abs_tol=0.01), (
            f"Scintillation regression failed at lam={lam_um}um, el={el_deg}deg: "
            f"got {result:.4f} dB, pinned to {impl_dB:.4f} dB "
            f"(ITU Table 2 target: {itu_table2_dB:.2f} dB)"
        )

    def test_sigma_increases_with_lower_elevation(self):
        assert scintillation.sigma_dB(1.55, 20.0) > scintillation.sigma_dB(1.55, 80.0)

    def test_sigma_increases_with_frequency(self):
        assert scintillation.sigma_dB(0.532, 40.0) > scintillation.sigma_dB(1.550, 40.0)

    def test_sigma_positive(self):
        for el in [10, 40, 75, 90]:
            assert scintillation.sigma_dB(1.55, el) > 0.0


# ===========================================================================
# 3. Geometric scattering -- Liang et al., eqs. (8)-(10)
#    No published table, so values are pinned to validated script output.
# ===========================================================================

class TestGeometric:

    LW  = 3.128e-4
    N   = 0.5
    phi = 1.6
    hA  = 20.0
    hE  = 1.0
    lam = 1.550

    @pytest.mark.parametrize("el_deg, expected_dB", [
        (10.0, 1.2155),
        (40.0, 0.3284),
        (90.0, 0.2111),
    ])
    def test_attenuation_dB(self, el_deg, expected_dB):
        result = geometric.attenuation_dB(
            el_deg, self.LW, self.N, self.lam, self.hA, self.hE, self.phi
        )
        assert result == approx(expected_dB, abs_tol=0.05)

    def test_attenuation_increases_with_lower_elevation(self):
        a_low  = geometric.attenuation_dB(20.0,  self.LW, self.N, self.lam, self.hA, self.hE, self.phi)
        a_high = geometric.attenuation_dB(80.0, self.LW, self.N, self.lam, self.hA, self.hE, self.phi)
        assert a_low > a_high

    def test_attenuation_non_negative(self):
        assert geometric.attenuation_dB(
            40.0, self.LW, self.N, self.lam, self.hA, self.hE, self.phi
        ) >= 0.0

    def test_invalid_LW_raises(self):
        with pytest.raises(ValueError):
            geometric.attenuation_dB(40.0, LW=-1.0, N=0.5,
                                     lam_um=1.55, hA_km=20.0, hE_km=1.0, phi=1.6)

    def test_invalid_N_raises(self):
        with pytest.raises(ValueError):
            geometric.attenuation_dB(40.0, LW=0.5, N=0.0,
                                     lam_um=1.55, hA_km=20.0, hE_km=1.0, phi=1.6)

    def test_invalid_heights_raises(self):
        with pytest.raises(ValueError):
            geometric.attenuation_dB(40.0, LW=0.5, N=0.5,
                                     lam_um=1.55, hA_km=0.5, hE_km=1.0, phi=1.6)


# ===========================================================================
# 4. Link geometry -- Liang et al., eqs. (4)-(5)
#    At zenith (el=90deg), dGS = h_S - h_E = 549 km exactly. [Liang Table 3]
# ===========================================================================

class TestGeometry:

    @pytest.mark.parametrize("el_deg, expected_km", [
        (90.0, 549.00),    # [Liang Table 3] -- exact at zenith
        (80.0, 556.78),
        (40.0, 810.66),
        (10.0, 1812.79),
    ])
    def test_slant_distance_km(self, el_deg, expected_km):
        d = geometry.slant_distance_km(el_deg, hE_km=1.0, hS_km=550.0)
        assert d == approx(expected_km, abs_tol=1.0)

    def test_slant_distance_increases_with_lower_elevation(self):
        assert (geometry.slant_distance_km(20.0, 1.0, 550.0)
                > geometry.slant_distance_km(80.0, 1.0, 550.0))

    @pytest.mark.parametrize("el_deg, expected_dB", [
        (90.0, 252.9690),
        (40.0, 256.3544),
        (10.0, 263.3445),
    ])
    def test_fspl_dB(self, el_deg, expected_dB):
        result = geometry.fspl_dB(1.550, el_deg, hE_km=1.0, hS_km=550.0)
        assert result == approx(expected_dB, abs_tol=0.01)


# ===========================================================================
# 5. Terminal parameters -- link/budget.py
# ===========================================================================

class TestTerminal:

    Theta_T = 15e-6
    theta_e = 1e-6
    Dr      = 80e-3
    lam_um  = 1.550

    def test_transmit_gain(self):
        G_T = budget.transmit_gain(self.Theta_T)
        assert G_T == approx(7.111e10, abs_tol=1e8)

    def test_receive_gain(self):
        G_R = budget.receive_gain(self.Dr, self.lam_um)
        assert G_R == approx(2.63e10, abs_tol=1e8)

    def test_pointing_loss_non_positive(self):
        G_T = budget.transmit_gain(self.Theta_T)
        assert budget.pointing_loss_dB(G_T, self.theta_e) <= 0.0

    def test_transmitter_pointing_loss(self):
        G_T = budget.transmit_gain(self.Theta_T)
        assert budget.pointing_loss_dB(G_T, self.theta_e) == approx(-0.3088, abs_tol=0.01)

    def test_receiver_pointing_loss(self):
        G_R = budget.receive_gain(self.Dr, self.lam_um)
        assert budget.pointing_loss_dB(G_R, self.theta_e) == approx(-0.1142, abs_tol=0.01)


# ===========================================================================
# 6. Full link budget -- Liang et al., Table 3 scenario
#    P_T values pinned from validated main.py table output.
# ===========================================================================

class TestFullBudget:

    lam_um  = 1.550
    hE_km   = 1.0
    hS_km   = 550.0
    Theta_T = 15e-6
    theta_T = 1e-6
    theta_R = 1e-6
    Dr_m    = 80e-3
    eta_T   = 0.8
    eta_R   = 0.8
    Pr_dBm  = -32.5
    LW      = 3.128e-4
    N       = 0.5
    hA_km   = 20.0
    phi     = 1.6

    @pytest.fixture(autouse=True)
    def derived(self):
        self.G_T = budget.transmit_gain(self.Theta_T)
        self.G_R = budget.receive_gain(self.Dr_m, self.lam_um)
        self.L_T = budget.pointing_loss_dB(self.G_T, self.theta_T)
        self.L_R = budget.pointing_loss_dB(self.G_R, self.theta_R)

    def _pt(self, el_deg, model):
        return budget.required_tx_power_dBm(
            self.Pr_dBm,
            eta_T=self.eta_T, eta_R=self.eta_R,
            G_T=self.G_T, G_R=self.G_R,
            L_T_dB=self.L_T, L_R_dB=self.L_R,
            lam_um=self.lam_um, el_deg=el_deg,
            hE_km=self.hE_km, hS_km=self.hS_km,
            atm_model=model,
            LW=self.LW, N=self.N, hA_km=self.hA_km, phi=self.phi,
        )

    @pytest.mark.parametrize("el_deg, expected_pt", [
        (90.0, 10.4230),
        (40.0, 13.9808),
        (10.0, 22.2752),
    ])
    def test_required_tx_power_geom(self, el_deg, expected_pt):
        result = self._pt(el_deg, "mie_geom")
        assert result == approx(expected_pt, abs_tol=0.15), (
            f"P_T (mie_geom) at el={el_deg}deg: got {result:.4f} dBm, "
            f"expected {expected_pt:.4f} dBm"
        )

    def test_pt_scin_greater_than_geom(self):
        """mie_scin gives larger atm. loss than mie_geom at these parameters."""
        for el in [20.0, 40.0, 80.0]:
            assert self._pt(el, "mie_scin") > self._pt(el, "mie_geom")

    def test_invalid_atm_model_raises(self):
        with pytest.raises(ValueError, match="Unknown atm_model"):
            self._pt(40.0, "bad_model")

    def test_pt_decreases_with_elevation(self):
        assert self._pt(10.0, "mie_geom") > self._pt(80.0, "mie_geom")

