"""Tests for the python interface to the RefractiveIndex database."""

import re
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

import refractiveindex as ri


class TestTabulatedData(unittest.TestCase):
    """Tests for tabulated n, k, and nk data types."""

    def test_tabulated_nk(self):
        """Tabulated nk material (SiO/Hass) — backward compat."""
        SiO = ri.RefractiveIndexMaterial(shelf='main', book='SiO', page='Hass')
        wl = 600

        self.assertAlmostEqual(SiO.get_refractive_index(wl), 1.96553846)
        self.assertAlmostEqual(SiO.get_extinction_coefficient(wl), 0.001)
        self.assertAlmostEqual(
            SiO.get_epsilon(wl),
            3.8633404437869827 + 0.003931076923076923j,
        )

    def test_tabulated_nk_silver(self):
        """Tabulated nk material (Ag/Johnson)."""
        ag = ri.RefractiveIndexMaterial(shelf='main', book='Ag', page='Johnson')
        self.assertAlmostEqual(ag.get_refractive_index(500), 0.05, places=1)
        self.assertAlmostEqual(ag.get_extinction_coefficient(500), 3.13, places=1)

    def test_tabulated_k_only(self):
        """Tabulated k-only material has no refractive index."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='BaF2', page='Bosomworth-5K')
        # k should work in valid range (far-IR, ~55–1000 μm)
        k = m.get_extinction_coefficient(104170)  # 104.17 μm
        self.assertAlmostEqual(k, 0.000609, places=4)
        # n should raise
        with self.assertRaises(Exception):
            m.get_refractive_index(104170)


class TestFormulas(unittest.TestCase):
    """Tests for all 9 dispersion formula types."""

    def test_formula_1_sellmeier(self):
        """Formula 1 (Sellmeier): MgAl2O4/Tropf."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='MgAl2O4', page='Tropf')
        self.assertAlmostEqual(m.get_refractive_index(500), 1.72299, places=4)

    def test_formula_2_sellmeier2(self):
        """Formula 2 (Sellmeier-2): Ar/Borzsonyi."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='Ar', page='Borzsonyi')
        self.assertAlmostEqual(m.get_refractive_index(500), 1.00028, places=5)

    def test_formula_3_polynomial(self):
        """Formula 3 (Polynomial): BeAl6O10/Pestryakov-alpha."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='BeAl6O10', page='Pestryakov-α')
        self.assertAlmostEqual(m.get_refractive_index(500), 1.74817, places=4)

    def test_formula_4_refractiveindex_info(self):
        """Formula 4 (RefractiveIndex.INFO): BeAl2O4/Walling-alpha."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='BeAl2O4', page='Walling-α')
        self.assertAlmostEqual(m.get_refractive_index(500), 1.74856, places=4)

    def test_formula_5_cauchy(self):
        """Formula 5 (Cauchy): SiC/Shaffer."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='SiC', page='Shaffer')
        self.assertAlmostEqual(m.get_refractive_index(500), 2.6906, places=3)

    def test_formula_6_gases(self):
        """Formula 6 (Gases): Ar/Bideau-Mehu."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='Ar', page='Bideau-Mehu')
        self.assertAlmostEqual(m.get_refractive_index(500), 1.000283, places=5)

    def test_formula_7_herzberger(self):
        """Formula 7 (Herzberger): Si/Edwards."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='Si', page='Edwards')
        self.assertAlmostEqual(m.get_refractive_index(2000), 3.45229, places=4)

    def test_formula_8_retro(self):
        """Formula 8 (Retro): AgBr/Schroter."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='AgBr', page='Schröter')
        self.assertAlmostEqual(m.get_refractive_index(500), 2.30945, places=4)

    def test_formula_9_exotic(self):
        """Formula 9 (Exotic): urea/Rosker-e."""
        m = ri.RefractiveIndexMaterial(shelf='organic', book='urea', page='Rosker-e')
        self.assertAlmostEqual(m.get_refractive_index(500), 1.61670, places=4)


class TestInputHandling(unittest.TestCase):
    """Tests for scalar and array input."""

    def test_scalar_input(self):
        """Scalar wavelength returns a scalar-like value."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='SiO', page='Hass')
        n = m.get_refractive_index(600)
        self.assertIsInstance(float(n), float)

    def test_array_input(self):
        """Array wavelength returns array with correct shape."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='SiO', page='Hass')
        wavelengths = np.array([400, 500, 600, 700])
        n = m.get_refractive_index(wavelengths)
        self.assertEqual(n.shape, (4,))
        self.assertTrue(np.all(n > 1.0))
        self.assertTrue(np.all(n < 3.0))

    def test_single_element_array(self):
        """Single-element array input returns single-element array."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='SiO', page='Hass')
        n = m.get_refractive_index(np.array([600]))
        self.assertEqual(n.shape, (1,))
        self.assertAlmostEqual(float(n[0]), 1.96553846)


class TestErrorHandling(unittest.TestCase):
    """Tests for error conditions."""

    def test_no_extinction_coefficient(self):
        """NoExtinctionCoefficient raised for formula-only material."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='BaF2', page='Malitson')
        with self.assertRaises(ri.NoExtinctionCoefficient):
            m.get_extinction_coefficient(500)

    def test_material_not_found(self):
        """KeyError raised for nonexistent material."""
        with self.assertRaises(KeyError):
            ri.RefractiveIndexMaterial(shelf='main', book='Unobtainium', page='Fake')

    def test_unknown_keyword_argument(self):
        """TypeError for a misspelled keyword argument, instead of ignoring it."""
        with self.assertRaises(TypeError):
            ri.RefractiveIndexMaterial(shelf='main', book='SiO', page='Hass', dbpath='/nonexistent')


class TestUnits(unittest.TestCase):
    """Tests for per-call unit= parameter on all public methods."""

    def setUp(self):
        # Formula material (n only) — BaF2 Malitson
        self.baf2 = ri.RefractiveIndexMaterial(shelf='main', book='BaF2', page='Malitson')
        # Tabulated nk material — Ag Johnson
        self.ag = ri.RefractiveIndexMaterial(shelf='main', book='Ag', page='Johnson')

    # --- unit equivalence ---

    def test_default_is_nm(self):
        """No unit= kwarg behaves identically to unit='nm'."""
        self.assertAlmostEqual(
            self.baf2.get_refractive_index(500),
            self.baf2.get_refractive_index(500, unit='nm'),
        )

    def test_nm_um_equivalent_n(self):
        """500 nm and 0.5 µm give the same refractive index."""
        self.assertAlmostEqual(
            self.baf2.get_refractive_index(500, unit='nm'),
            self.baf2.get_refractive_index(0.5, unit='um'),
        )

    def test_nm_cm1_equivalent_n(self):
        """500 nm = 20000 cm⁻¹ give the same refractive index."""
        self.assertAlmostEqual(
            self.baf2.get_refractive_index(500, unit='nm'),
            self.baf2.get_refractive_index(20000, unit='cm-1'),
        )

    def test_nm_A_equivalent_n(self):
        """500 nm = 5000 Å give the same refractive index."""
        self.assertAlmostEqual(
            self.baf2.get_refractive_index(500, unit='nm'),
            self.baf2.get_refractive_index(5000, unit='A'),
        )

    def test_nm_m_equivalent_n(self):
        """500 nm = 5e-7 m give the same refractive index."""
        self.assertAlmostEqual(
            self.baf2.get_refractive_index(500, unit='nm'),
            self.baf2.get_refractive_index(5e-7, unit='m'),
        )

    def test_nm_eV_equivalent_n(self):
        """500 nm = 1.23984193/0.5 eV give the same refractive index."""
        eV = 1.23984193 / 0.5  # λ(µm) = 0.5 → E(eV) = hc/λ
        self.assertAlmostEqual(
            self.baf2.get_refractive_index(500, unit='nm'),
            self.baf2.get_refractive_index(eV, unit='eV'),
        )

    def test_nm_THz_equivalent_n(self):
        """500 nm = 299.792458/0.5 THz give the same refractive index."""
        THz = 299.792458 / 0.5  # λ(µm) = 0.5 → ν(THz) = c/λ
        self.assertAlmostEqual(
            self.baf2.get_refractive_index(500, unit='nm'),
            self.baf2.get_refractive_index(THz, unit='THz'),
        )

    def test_nm_um_equivalent_k(self):
        """500 nm and 0.5 µm give the same extinction coefficient."""
        self.assertAlmostEqual(
            self.ag.get_extinction_coefficient(500, unit='nm'),
            self.ag.get_extinction_coefficient(0.5, unit='um'),
        )

    def test_nm_cm1_equivalent_k(self):
        """500 nm = 20000 cm⁻¹ give the same extinction coefficient."""
        self.assertAlmostEqual(
            self.ag.get_extinction_coefficient(500, unit='nm'),
            self.ag.get_extinction_coefficient(20000, unit='cm-1'),
        )

    def test_nm_um_equivalent_epsilon(self):
        """unit= is forwarded correctly through get_epsilon."""
        self.assertAlmostEqual(
            self.ag.get_epsilon(500, unit='nm'),
            self.ag.get_epsilon(0.5, unit='um'),
        )

    def test_nm_cm1_equivalent_epsilon(self):
        """unit='cm-1' forwarded correctly through get_epsilon."""
        self.assertAlmostEqual(
            self.ag.get_epsilon(500, unit='nm'),
            self.ag.get_epsilon(20000, unit='cm-1'),
        )

    def test_array_units_equivalent(self):
        """Array input gives identical results across all three units."""
        wl_nm = np.array([400.0, 500.0, 600.0])
        wl_um = wl_nm / 1000.0
        wl_cm1 = 1e4 / wl_um  # 25000, 20000, 16667 cm⁻¹

        n_nm = self.baf2.get_refractive_index(wl_nm, unit='nm')
        n_um = self.baf2.get_refractive_index(wl_um, unit='um')
        n_cm1 = self.baf2.get_refractive_index(wl_cm1, unit='cm-1')

        np.testing.assert_allclose(n_nm, n_um)
        np.testing.assert_allclose(n_nm, n_cm1)

    # --- get_wl_range ---

    def test_wl_range_default_is_nm(self):
        """get_wl_range() with no argument returns nm."""
        lo, hi = self.baf2.get_wl_range()
        self.assertAlmostEqual(lo, self.baf2.get_wl_range(unit='nm')[0])
        self.assertAlmostEqual(hi, self.baf2.get_wl_range(unit='nm')[1])

    def test_wl_range_nm_sorted(self):
        """get_wl_range in nm returns (min, max)."""
        lo, hi = self.baf2.get_wl_range(unit='nm')
        self.assertLess(lo, hi)

    def test_wl_range_um_consistent_with_nm(self):
        """get_wl_range in µm is 1000× smaller than in nm."""
        lo_nm, hi_nm = self.baf2.get_wl_range(unit='nm')
        lo_um, hi_um = self.baf2.get_wl_range(unit='um')
        self.assertAlmostEqual(lo_nm / 1000, lo_um)
        self.assertAlmostEqual(hi_nm / 1000, hi_um)

    def test_wl_range_cm1_sorted(self):
        """get_wl_range in cm⁻¹ is still returned as (min, max) despite reciprocal ordering."""
        lo_cm1, hi_cm1 = self.baf2.get_wl_range(unit='cm-1')
        self.assertLess(lo_cm1, hi_cm1)

    def test_wl_range_cm1_consistent_with_nm(self):
        """get_wl_range in cm⁻¹ values match 1e4/λ(µm) conversion."""
        lo_nm, hi_nm = self.baf2.get_wl_range(unit='nm')
        lo_cm1, hi_cm1 = self.baf2.get_wl_range(unit='cm-1')
        # long λ (hi_nm) → low wavenumber (lo_cm1)
        self.assertAlmostEqual(1e7 / hi_nm, lo_cm1)
        # short λ (lo_nm) → high wavenumber (hi_cm1)
        self.assertAlmostEqual(1e7 / lo_nm, hi_cm1)

    def test_wl_range_A_consistent_with_nm(self):
        """get_wl_range in Å is 10× larger than in nm."""
        lo_nm, hi_nm = self.baf2.get_wl_range(unit='nm')
        lo_A, hi_A = self.baf2.get_wl_range(unit='A')
        self.assertAlmostEqual(lo_nm * 10, lo_A)
        self.assertAlmostEqual(hi_nm * 10, hi_A)

    def test_wl_range_m_consistent_with_nm(self):
        """get_wl_range in m is 1e-9 × nm values."""
        lo_nm, hi_nm = self.baf2.get_wl_range(unit='nm')
        lo_m, hi_m = self.baf2.get_wl_range(unit='m')
        self.assertAlmostEqual(lo_nm * 1e-9, lo_m)
        self.assertAlmostEqual(hi_nm * 1e-9, hi_m)

    def test_wl_range_eV_sorted(self):
        """get_wl_range in eV is sorted (min, max) despite reciprocal ordering."""
        lo_eV, hi_eV = self.baf2.get_wl_range(unit='eV')
        self.assertLess(lo_eV, hi_eV)

    def test_wl_range_THz_sorted(self):
        """get_wl_range in THz is sorted (min, max) despite reciprocal ordering."""
        lo_THz, hi_THz = self.baf2.get_wl_range(unit='THz')
        self.assertLess(lo_THz, hi_THz)

    # --- invalid unit ---

    def test_invalid_unit_n(self):
        """ValueError for unrecognised unit in get_refractive_index."""
        with self.assertRaises(ValueError):
            self.baf2.get_refractive_index(500, unit='angstrom')

    def test_invalid_unit_k(self):
        """ValueError for unrecognised unit in get_extinction_coefficient."""
        with self.assertRaises(ValueError):
            self.ag.get_extinction_coefficient(500, unit='furlong')

    def test_invalid_unit_wl_range(self):
        """ValueError for unrecognised unit in get_wl_range."""
        with self.assertRaises(ValueError):
            self.baf2.get_wl_range(unit='lightyear')


class TestWavelengthRange(unittest.TestCase):
    """Tests for the quantity= argument of get_wl_range."""

    def setUp(self):
        # Formula n (0.4–12 µm) and tabulated k (9.49308–11.9739 µm)
        self.laf3 = ri.RefractiveIndexMaterial(shelf='main', book='LaF3', page='Amotchkina')

    def test_n_and_k_ranges(self):
        """quantity='n' and quantity='k' give the range of each data block."""
        np.testing.assert_allclose(self.laf3.get_wl_range(quantity='n'), (400, 12000))
        np.testing.assert_allclose(self.laf3.get_wl_range(quantity='k'), (9493.08, 11973.9))

    def test_default_is_epsilon(self):
        """The default is quantity='epsilon', the overlap of n and k."""
        self.assertEqual(self.laf3.get_wl_range(), self.laf3.get_wl_range(quantity='epsilon'))
        np.testing.assert_allclose(self.laf3.get_wl_range(), (9493.08, 11973.9))

    def test_nk_same_as_epsilon(self):
        """quantity='nk' is another name for quantity='epsilon'."""
        self.assertEqual(self.laf3.get_wl_range('um', 'nk'), self.laf3.get_wl_range('um', 'epsilon'))

    def test_epsilon_finite_inside_range(self):
        """get_epsilon is not nan inside get_wl_range() when k covers less than n."""
        lo, hi = self.laf3.get_wl_range()
        eps = self.laf3.get_epsilon(np.linspace(lo, hi, 7)[1:-1])
        self.assertTrue(np.all(np.isfinite(eps)))

    def test_epsilon_overlap_n_ends_first(self):
        """The overlap ends where the k data ends, also for tabulated n and k."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='MoS2', page='Yim-3nm')
        np.testing.assert_allclose(m.get_wl_range(quantity='n'), (382.417, 888.102))
        np.testing.assert_allclose(m.get_wl_range(quantity='k'), (382.127, 887.428))
        np.testing.assert_allclose(m.get_wl_range(), (382.417, 887.428))

    def test_epsilon_without_k_is_n_range(self):
        """Without k data, the epsilon range is the n range, as get_epsilon takes k as 0."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='BaF2', page='Malitson')
        self.assertEqual(m.get_wl_range(), m.get_wl_range(quantity='n'))
        with self.assertRaises(ri.NoExtinctionCoefficient):
            m.get_wl_range(quantity='k')

    def test_without_n(self):
        """Without n data, only quantity='k' gives a range."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='BaF2', page='Bosomworth-5K')
        np.testing.assert_allclose(m.get_wl_range(quantity='k'), (54945, 1000000))
        for quantity in ('n', 'nk', 'epsilon'):
            with self.assertRaisesRegex(Exception, 'No refractive index'):
                m.get_wl_range(quantity=quantity)

    def test_no_overlap(self):
        """ValueError with both ranges when the n and k data do not overlap."""
        m = ri.RefractiveIndexMaterial(shelf='specs', book='HIKARI-optical', page='J-SF03HS')
        for quantity in ('nk', 'epsilon'):
            with self.assertRaisesRegex(ValueError, r'388\.865–2058\.09 nm.*280–290 nm.*do not overlap'):
                m.get_wl_range(quantity=quantity)
        np.testing.assert_allclose(m.get_wl_range(quantity='n'), (388.865, 2058.09))
        np.testing.assert_allclose(m.get_wl_range(quantity='k'), (280, 290))

    def test_single_point_overlap(self):
        """n at one wavelength inside the k range gives a range of one point."""
        m = ri.RefractiveIndexMaterial(shelf='specs', book='SCHOTT-misc', page='DURAN')
        np.testing.assert_allclose(m.get_wl_range(), (587.6, 587.6))

    def test_table_not_in_wavelength_order(self):
        """The range of a table is its smallest and largest wavelength, not its first and last row."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='CaMg_CO3_2', page='Querry-o')
        np.testing.assert_allclose(m.get_wl_range(quantity='n'), (2500, 40000))

    def test_finite_at_range_ends(self):
        """The ends of get_wl_range give values, not nan, after unit conversion in both directions."""
        # 3.3 µm -> 3300.0 nm -> 3.3000000000000003 µm, out of the table before the fix
        au = ri.RefractiveIndexMaterial(shelf='main', book='Au', page='Yakubovsky-4nm')
        # 1e4 µm -> 0.0223171 eV -> 1e4 µm plus some ulp
        c = ri.RefractiveIndexMaterial(shelf='main', book='C', page='Querry-DixonKS2')
        for m in (au, c, self.laf3):
            for unit in ('nm', 'um', 'eV', 'cm-1'):
                for quantity, getter in (('epsilon', m.get_epsilon),
                                         ('n', m.get_refractive_index),
                                         ('k', m.get_extinction_coefficient)):
                    with self.subTest(m=m, unit=unit, quantity=quantity):
                        ends = np.array(m.get_wl_range(unit, quantity))
                        self.assertTrue(np.all(np.isfinite(getter(ends, unit=unit))))

    def test_nan_beyond_range_ends(self):
        """Tabulated data still gives nan just outside the range."""
        au = ri.RefractiveIndexMaterial(shelf='main', book='Au', page='Yakubovsky-4nm')
        lo, hi = au.get_wl_range()
        self.assertTrue(np.all(np.isnan(au.get_epsilon([lo * (1 - 1e-6), hi * (1 + 1e-6)]))))

    def test_invalid_quantity(self):
        """ValueError for an unknown quantity."""
        with self.assertRaisesRegex(ValueError, 'quantity must be one of'):
            self.laf3.get_wl_range(quantity='eps')


class TestEpsilon(unittest.TestCase):
    """Tests for dielectric permittivity calculation."""

    def test_epsilon_default_convention(self):
        """Default exp(-iwt): epsilon = (n + ik)^2."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='Ag', page='Johnson')
        wl = 500
        n = m.get_refractive_index(wl)
        k = m.get_extinction_coefficient(wl)
        eps = m.get_epsilon(wl)
        expected = (n + 1j * k) ** 2
        self.assertAlmostEqual(eps, expected)

    def test_epsilon_opposite_convention(self):
        """Opposite exp(+iwt): epsilon = (n - ik)^2."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='Ag', page='Johnson')
        wl = 500
        n = m.get_refractive_index(wl)
        k = m.get_extinction_coefficient(wl)
        eps = m.get_epsilon(wl, exp_type='exp_plus_i_omega_t')
        expected = (n - 1j * k) ** 2
        self.assertAlmostEqual(eps, expected)

    def test_epsilon_no_extinction_data(self):
        """Materials without k data: epsilon = n^2 (k treated as 0)."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='SiO2', page='Malitson')
        wl = 800
        n = m.get_refractive_index(wl)
        eps = m.get_epsilon(wl)
        self.assertAlmostEqual(eps, n ** 2)
        self.assertEqual(eps.imag, 0)

    def test_epsilon_n_ik_consistency(self):
        """Verify (n + ik)^2 == epsilon identity for both conventions."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='Ag', page='Johnson')
        wl = 600
        eps_minus = m.get_epsilon(wl, exp_type='exp_minus_i_omega_t')
        eps_plus = m.get_epsilon(wl, exp_type='exp_plus_i_omega_t')
        # Real parts should be equal, imaginary parts opposite sign
        self.assertAlmostEqual(eps_minus.real, eps_plus.real)
        self.assertAlmostEqual(eps_minus.imag, -eps_plus.imag)

    def test_epsilon_invalid_exp_type(self):
        """ValueError for an unknown exp_type, so a typo cannot flip the sign of Im(epsilon)."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='Ag', page='Johnson')
        # Only exact spellings are accepted.
        for exp_type in ('exp_minus_iwt', 'exp(-iwt )', 'EXP(-iwt)'):
            with self.subTest(exp_type=exp_type):
                with self.assertRaises(ValueError):
                    m.get_epsilon(500, exp_type=exp_type)

    def test_epsilon_all_exp_types(self):
        """Every accepted exp_type gives (n + ik)^2 or (n - ik)^2, as selected by its sign."""
        from refractiveindex.refractiveindex import _EXP_TYPES
        minus = ['exp(-iwt)', 'exp(-iωt)', 'exp(-jwt)', 'exp(-jωt)', 'exp_minus_i_omega_t']
        plus = ['exp(+iwt)', 'exp(iwt)', 'exp(+iωt)', 'exp(iωt)',
                'exp(+jwt)', 'exp(jwt)', 'exp(+jωt)', 'exp(jωt)', 'exp_plus_i_omega_t']
        self.assertEqual(set(_EXP_TYPES), set(minus + plus))

        m = ri.RefractiveIndexMaterial(shelf='main', book='Ag', page='Johnson')
        wl = 500
        n = m.get_refractive_index(wl)
        k = m.get_extinction_coefficient(wl)
        for exp_type in minus:
            with self.subTest(exp_type=exp_type):
                self.assertAlmostEqual(m.get_epsilon(wl, exp_type=exp_type), (n + 1j * k) ** 2)
        for exp_type in plus:
            with self.subTest(exp_type=exp_type):
                self.assertAlmostEqual(m.get_epsilon(wl, exp_type=exp_type), (n - 1j * k) ** 2)

    def test_epsilon_default_exp_type(self):
        """The default exp_type gives the same result as 'exp(-iwt)' and 'exp_minus_i_omega_t'."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='Ag', page='Johnson')
        eps = m.get_epsilon(500)
        self.assertEqual(eps, m.get_epsilon(500, exp_type='exp(-iwt)'))
        self.assertEqual(eps, m.get_epsilon(500, exp_type='exp_minus_i_omega_t'))


class TestCatalogFixes(unittest.TestCase):
    """Tests for temporary fixes of upstream catalog errors (_CATALOG_FIXES)."""

    def test_synowicki_own_data(self):
        """MgO and ZrO2 Synowicki give their own data, not As2S3's (issue #10)."""
        n = {
            book: ri.RefractiveIndexMaterial(shelf='main', book=book, page='Synowicki')
            .get_refractive_index(633)
            for book in ('As2S3', 'MgO', 'ZrO2')
        }
        self.assertAlmostEqual(n['As2S3'], 2.6008, places=3)
        self.assertAlmostEqual(n['MgO'], 1.7257, places=3)
        self.assertAlmostEqual(n['ZrO2'], 2.1406, places=3)

    def test_fixes_still_needed(self):
        """Fails once upstream fixes a catalog error, as a reminder to remove the fix."""
        from refractiveindex.refractiveindex import _CATALOG_FIXES, _DEFAULT_DB_PATH, _read_catalog
        catalog = _read_catalog(_DEFAULT_DB_PATH)
        for key, (bad_path, _) in _CATALOG_FIXES.items():
            self.assertEqual(
                catalog[key].path if key in catalog else None, bad_path,
                f"Upstream catalog no longer has {bad_path!r} for {key}. "
                f"Remove this entry from _CATALOG_FIXES in refractiveindex.py.",
            )


class TestBundledDatabase(unittest.TestCase):
    """The database included in the package is the default, and no network is used."""

    def test_default_is_bundled(self):
        """The default database is refractiveindex/database/ inside the package."""
        from refractiveindex.refractiveindex import _DEFAULT_DB_PATH
        self.assertEqual(_DEFAULT_DB_PATH, Path(ri.__file__).parent / 'database')
        self.assertTrue((_DEFAULT_DB_PATH / 'catalog-nk.yml').is_file())
        self.assertTrue((_DEFAULT_DB_PATH / 'LICENSE').is_file())

    def test_no_network(self):
        """Loading a material makes no network request."""
        with mock.patch('urllib.request.urlopen', side_effect=AssertionError('network used')):
            m = ri.RefractiveIndexMaterial(shelf='main', book='SiO2', page='Malitson')
        self.assertAlmostEqual(m.get_refractive_index(633), 1.4570, places=4)

    def test_version_constants(self):
        """DATABASE_VERSION is an upstream release tag and DATABASE_COMMIT a full commit SHA."""
        self.assertRegex(ri.DATABASE_VERSION, r'^v\d{4}-\d{2}-\d{2}$')
        self.assertRegex(ri.DATABASE_COMMIT, r'^[0-9a-f]{40}$')

    def test_db_path(self):
        """db_path (str or Path) selects a database folder."""
        from refractiveindex.refractiveindex import _DEFAULT_DB_PATH
        default = ri.RefractiveIndexMaterial(shelf='main', book='SiO', page='Hass')
        for db_path in (_DEFAULT_DB_PATH, str(_DEFAULT_DB_PATH)):
            with self.subTest(db_path=db_path):
                m = ri.RefractiveIndexMaterial(shelf='main', book='SiO', page='Hass', db_path=db_path)
                self.assertEqual(m.get_refractive_index(600), default.get_refractive_index(600))

    def test_db_path_without_catalog(self):
        """A db_path without catalog-nk.yml raises FileNotFoundError that names the file."""
        with tempfile.TemporaryDirectory() as tempdir:
            with self.assertRaisesRegex(FileNotFoundError, re.escape('catalog-nk.yml')):
                ri.RefractiveIndexMaterial(shelf='main', book='SiO', page='Hass', db_path=tempdir)


class TestDeprecatedArguments(unittest.TestCase):
    """Arguments of the download code warn and have no effect (removed in 2.0.0)."""

    def test_no_warning_by_default(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ri.RefractiveIndexMaterial(shelf='main', book='SiO', page='Hass')
        self.assertEqual([w for w in caught if issubclass(w.category, DeprecationWarning)], [])

    def test_download_arguments_warn(self):
        """Each argument warns for any value, at the line of the caller."""
        for kwargs in ({'auto_download': True}, {'auto_download': False},
                       {'update_database': True}, {'ssl_certificate_location': None},
                       {'ssl_certificate_location': ''}):
            with self.subTest(**kwargs):
                with self.assertWarns(DeprecationWarning) as cm:
                    m = ri.RefractiveIndexMaterial(shelf='main', book='SiO', page='Hass', **kwargs)
                self.assertIn(next(iter(kwargs)), str(cm.warning))
                self.assertEqual(cm.filename, __file__)
                self.assertAlmostEqual(m.get_refractive_index(600), 1.96553846)

    def test_databasePath_alias(self):
        """databasePath warns and still selects the database folder."""
        with tempfile.TemporaryDirectory() as tempdir:
            with self.assertWarns(DeprecationWarning):
                with self.assertRaises(FileNotFoundError):
                    ri.RefractiveIndexMaterial(shelf='main', book='SiO', page='Hass', databasePath=tempdir)


class TestFromUrl(unittest.TestCase):
    """RefractiveIndexMaterial.from_url reads shelf, book and page from a refractiveindex.info URL."""

    def test_same_as_constructor(self):
        """A URL gives the same data as the constructor with its shelf, book and page."""
        m = ri.RefractiveIndexMaterial.from_url('https://refractiveindex.info/?shelf=main&book=SiO&page=Hass')
        ref = ri.RefractiveIndexMaterial(shelf='main', book='SiO', page='Hass')
        self.assertEqual(m.get_refractive_index(600), ref.get_refractive_index(600))
        self.assertEqual(m.get_extinction_coefficient(600), ref.get_extinction_coefficient(600))

    def test_percent_encoded(self):
        """Percent-encoded page names are decoded (Querry-%CE%B2 is Querry-β)."""
        m = ri.RefractiveIndexMaterial.from_url('https://refractiveindex.info/?shelf=main&book=CaSO4&page=Querry-%CE%B2')
        ref = ri.RefractiveIndexMaterial(shelf='main', book='CaSO4', page='Querry-β')
        self.assertEqual(m.get_refractive_index(10000), ref.get_refractive_index(10000))

    def test_other_url_forms(self):
        """Scheme, host, parameter order and other parameters do not matter."""
        ref = ri.RefractiveIndexMaterial(shelf='main', book='SiO2', page='Malitson')
        for url in ('http://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson',
                    'refractiveindex.info/?shelf=main&book=SiO2&page=Malitson',
                    'https://www.refractiveindex.info/?page=Malitson&shelf=main&book=SiO2',
                    'https://example.org/?shelf=main&book=SiO2&page=Malitson&wavelength=0.5#chart'):
            with self.subTest(url=url):
                m = ri.RefractiveIndexMaterial.from_url(url)
                self.assertEqual(m.get_refractive_index(633), ref.get_refractive_index(633))

    def test_missing_parameter(self):
        """ValueError that names the missing parameter."""
        for url, missing in (('https://refractiveindex.info/?shelf=main&book=SiO2', 'page'),
                             ('https://refractiveindex.info/?shelf=main&book=SiO2&page=', 'page'),
                             ('https://refractiveindex.info/?book=SiO2&page=Malitson', 'shelf')):
            with self.subTest(url=url):
                with self.assertRaisesRegex(ValueError, f'no {missing} parameter'):
                    ri.RefractiveIndexMaterial.from_url(url)

    def test_data_file_link(self):
        """Links to data files are not accepted, and the message tells to use the address bar."""
        for url in ('https://refractiveindex.info/database/data/main/SiO2/nk/Malitson.yml',
                    'https://refractiveindex.info/data_csv.php?datafile=database/data/main/SiO2/nk/Malitson.yml'):
            with self.subTest(url=url):
                with self.assertRaisesRegex(ValueError, 'address bar'):
                    ri.RefractiveIndexMaterial.from_url(url)

    def test_material_not_found(self):
        """KeyError that tells to use a newer database with db_path."""
        with self.assertRaisesRegex(KeyError, 'db_path'):
            ri.RefractiveIndexMaterial.from_url('https://refractiveindex.info/?shelf=main&book=Unobtainium&page=Fake')

    def test_db_path(self):
        """db_path is given to the constructor."""
        from refractiveindex.refractiveindex import _DEFAULT_DB_PATH
        url = 'https://refractiveindex.info/?shelf=main&book=SiO&page=Hass'
        m = ri.RefractiveIndexMaterial.from_url(url, db_path=str(_DEFAULT_DB_PATH))
        self.assertAlmostEqual(m.get_refractive_index(600), 1.96553846)
        with tempfile.TemporaryDirectory() as tempdir:
            with self.assertRaises(FileNotFoundError):
                ri.RefractiveIndexMaterial.from_url(url, db_path=tempdir)


# A small database: HTML in names, a page listed twice under two dividers, and a missing data file
_TEST_CATALOG = """\
- DIVIDER: "Research data"
- SHELF: main
  name: "MAIN - test"
  content:
    - BOOK: X2
      name: "X<sub>2</sub> (Test material)"
      content:
        - DIVIDER: "First"
        - PAGE: Good
          name: "Good 2020: n,k 0.4–0.8 µm"
          data: main/X2/Good.yml
        - PAGE: Broken
          name: "Broken 2021: n 0.4–0.8 µm"
          data: main/X2/Missing.yml
        - DIVIDER: "Second"
        - PAGE: Good
          name: "Good 2020: n,k 0.4–0.8 µm"
          data: main/X2/Good.yml
"""

_TEST_DATA = """\
DATA:
  - type: tabulated nk
    data: |
        0.4 1.5 0.1
        0.8 1.4 0.2
"""


class TestSearch(unittest.TestCase):
    """ri.search finds pages by text, shelf, book, quantity and wavelength."""

    def test_exact_book_first(self):
        """The pages of the book whose id is the query come first, in catalog order."""
        hits = ri.search('SiO2')
        book = ri.search(shelf='main', book='SiO2')
        self.assertEqual((hits[0].shelf, hits[0].book, hits[0].page), ('main', 'SiO2', 'Malitson'))
        self.assertEqual(list(hits[:len(book)]), list(book))

    def test_every_word_matches(self):
        """Each query word is in the book or page id, the book or page name, or the divider."""
        for query in ('SiO2', 'silica', 'SiO2 Malitson', 'Ciesielski Ag'):
            hits = ri.search(query)
            with self.subTest(query=query):
                self.assertTrue(hits)
                for m in hits:
                    text = ' '.join((m.book, m.page, m.book_name, m.name, m.divider or '')).lower()
                    for word in query.lower().split():
                        self.assertIn(word, text)

    def test_several_words(self):
        """Several words must all match."""
        hits = ri.search('SiO2 Malitson')
        self.assertIn(('main', 'SiO2', 'Malitson'), [(m.shelf, m.book, m.page) for m in hits])
        self.assertNotIn('Nyakuchena', [m.page for m in hits])

    def test_ignores_case(self):
        """The query ignores case."""
        self.assertEqual(ri.search('sio2'), ri.search('SiO2'))

    def test_ranking(self):
        """Exact book id, then a word of the book name, then part of the book id, then the rest."""
        from refractiveindex.refractiveindex import _DEFAULT_DB_PATH, _load_catalog, _search_tier
        catalog = _load_catalog(_DEFAULT_DB_PATH)
        for query in ('SiO2', 'silica', 'Si', 'gold', 'BK7'):
            words = query.lower().split()
            tiers = [_search_tier(words, m.book, m.page, catalog[m.shelf, m.book, m.page])
                     for m in ri.search(query)]
            with self.subTest(query=query):
                self.assertEqual(tiers, sorted(tiers))
        # Silica is a word of the book name of SiO2. Bi12SiO20 has SiO2 in its book id only.
        self.assertEqual(ri.search('silica')[0].book, 'SiO2')
        books = [m.book for m in ri.search('SiO2')]
        self.assertLess(books.index('Bi12SiO20'), books.index('Ag'))

    def test_shelf_and_book(self):
        """shelf= and book= are exact ids, alone or in a list."""
        self.assertTrue(all(m.shelf == 'glass' for m in ri.search(shelf='glass')))
        self.assertEqual({m.book for m in ri.search(shelf='main', book=['SiO2', 'SiO'])}, {'SiO2', 'SiO'})
        self.assertEqual({m.shelf for m in ri.search('Malitson', shelf=('main', 'glass'))}, {'main', 'glass'})

    def test_unknown_shelf_or_book(self):
        """ValueError that lists the shelves, or suggests a book id."""
        with self.assertRaisesRegex(ValueError, r"No shelf 'mian'.*'main'"):
            ri.search(shelf='mian')
        with self.assertRaisesRegex(ValueError, r"No book 'sio2'.*Did you mean 'SiO2'"):
            ri.search(book='sio2')
        with self.assertRaisesRegex(ValueError, r"No book 'SiO3'"):
            ri.search(book=['SiO2', 'SiO3'])

    def test_quantity(self):
        """quantity= keeps the pages for which get_wl_range(quantity=...) gives a range."""
        pages = lambda **kwargs: {m.page for m in ri.search(shelf='main', book='BaF2', **kwargs)}
        self.assertIn('Bosomworth-5K', pages(quantity='k'))
        self.assertNotIn('Malitson', pages(quantity='k'))
        self.assertIn('Malitson', pages(quantity='n'))
        self.assertNotIn('Bosomworth-5K', pages(quantity='n'))
        self.assertEqual(pages(quantity='epsilon'), pages(quantity='nk'))
        # The n and k data of J-SF03HS do not overlap
        self.assertEqual(ri.search('J-SF03HS', shelf='specs', quantity='epsilon'), [])
        self.assertEqual(len(ri.search('J-SF03HS', shelf='specs', quantity='k')), 1)

    def test_wavelength(self):
        """wavelength= keeps the pages whose range contains it, ends included."""
        for wavelength in (1550, 400, (400, 800), (800, 400)):
            for quantity in (None, 'n', 'k'):
                with self.subTest(wavelength=wavelength, quantity=quantity):
                    hits = ri.search('SiO2', wavelength=wavelength, quantity=quantity)
                    for m in hits:
                        lo, hi = m.material().get_wl_range(quantity=quantity or 'epsilon')
                        self.assertTrue(lo * (1 - 1e-9) <= np.min(wavelength))
                        self.assertTrue(np.max(wavelength) <= hi * (1 + 1e-9))
        pages = lambda wavelength: [m.page for m in ri.search('SiO2', shelf='main', wavelength=wavelength)]
        # Malitson: 0.21–6.7 µm
        self.assertIn('Malitson', pages(6700))
        self.assertIn('Malitson', pages(210))
        self.assertNotIn('Malitson', pages(6701))
        self.assertNotIn('Malitson', pages((200, 300)))
        self.assertEqual(pages((800, 400)), pages((400, 800)))

    def test_wavelength_default_quantity_is_epsilon(self):
        """Without quantity=, wavelength= uses the range of get_wl_range(), the overlap of n and k."""
        # LaF3/Amotchkina: n 400–12000 nm, k 9493.08–11973.9 nm
        self.assertEqual(ri.search('Amotchkina', book='LaF3', wavelength=1000), [])
        self.assertEqual(len(ri.search('Amotchkina', book='LaF3', wavelength=1000, quantity='n')), 1)
        self.assertEqual(len(ri.search('Amotchkina', book='LaF3', wavelength=10000)), 1)

    def test_wavelength_unit(self):
        """unit= gives the unit of wavelength, and reciprocal units can be in either order."""
        from refractiveindex.refractiveindex import _hc_eV_um
        nm = ri.search('SiO2', wavelength=(_hc_eV_um / 3 * 1000, _hc_eV_um / 1 * 1000))
        self.assertTrue(nm)
        self.assertEqual(ri.search('SiO2', wavelength=(1, 3), unit='eV'), nm)
        self.assertEqual(ri.search('SiO2', wavelength=(3, 1), unit='eV'), nm)
        self.assertEqual(ri.search('SiO2', wavelength=1.55, unit='um'), ri.search('SiO2', wavelength=1550))

    def test_invalid_arguments(self):
        """ValueError for an unknown quantity or unit, also without wavelength, and for a bad wavelength."""
        with self.assertRaisesRegex(ValueError, 'quantity must be one of'):
            ri.search('SiO2', quantity='eps')
        with self.assertRaisesRegex(ValueError, 'unit must be one of'):
            ri.search('SiO2', unit='nanometer')
        with self.assertRaisesRegex(ValueError, 'number or a'):
            ri.search('SiO2', wavelength=(400, 600, 800))

    def test_no_query(self):
        """Without a query, every page passes; without any argument, search gives all pages."""
        from refractiveindex.refractiveindex import _DEFAULT_DB_PATH, _load_catalog
        self.assertEqual(len(ri.search()), len(_load_catalog(_DEFAULT_DB_PATH)))
        self.assertEqual(len(ri.search(shelf='main', book='SiO2')), len(ri.search('SiO2', book='SiO2')))

    def test_no_matches(self):
        """No matches gives an empty list that prints as No matches."""
        hits = ri.search('Unobtainium')
        self.assertEqual(hits, [])
        self.assertEqual(repr(hits), 'No matches')

    def test_duplicates_shown(self):
        """Pages with the same data file on other shelves are all shown."""
        keys = [(m.shelf, m.book) for m in ri.search('Malitson 1965', book=['SiO2', 'fused_silica'])]
        self.assertIn(('main', 'SiO2'), keys)
        self.assertIn(('glass', 'fused_silica'), keys)

    def test_no_runtime_warning(self):
        """Filters do not give the RuntimeWarning of formulas that the constructor samples."""
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            self.assertTrue(ri.search(shelf='main', book=['GaSe', 'CS2'], quantity='n'))

    def test_match(self):
        """A match prints as Match and loads its page with material()."""
        m = ri.search('SiO2')[0]
        self.assertTrue(repr(m).startswith("Match(shelf='main', book='SiO2', page='Malitson', "))
        self.assertEqual(m.book_name, 'SiO2 (Silicon dioxide, Silica, Quartz)')
        self.assertEqual(m.divider, 'Fused silica')
        ref = ri.RefractiveIndexMaterial(shelf='main', book='SiO2', page='Malitson')
        self.assertEqual(m.material().get_refractive_index(633), ref.get_refractive_index(633))
        self.assertEqual(hash(m), hash(ri.search('SiO2')[0]))


class TestSearchPrint(unittest.TestCase):
    """The list from ri.search prints one line per match: index, shelf/book/page, name [divider]."""

    def test_lines(self):
        """Line i is i, shelf/book/page of hits[i], and its name with the divider in brackets."""
        hits = ri.search('SiO2')
        lines = repr(hits).splitlines()
        self.assertEqual(len(lines), len(hits))
        path_width = max(len(f'{m.shelf}/{m.book}/{m.page}') for m in hits)
        for i, (line, m) in enumerate(zip(lines, hits)):
            with self.subTest(line=line):
                number, rest = line.split('  ', 1)
                self.assertEqual(int(number), i)
                # The names line up after the longest path
                self.assertEqual(rest[:path_width].rstrip(), f'{m.shelf}/{m.book}/{m.page}')
                description = m.name if m.divider is None else f'{m.name} [{m.divider}]'
                self.assertEqual(rest[path_width:], '  ' + description)
        self.assertEqual(lines[0].split(), ['0', 'main/SiO2/Malitson', 'Malitson', '1965:', 'n',
                                            '0.21–6.7', 'µm', '[Fused', 'silica]'])

    def test_path_gives_ids(self):
        """The path of a line splits at / into the shelf, book and page of the constructor."""
        hits = ri.search('polychlorotrifluoroethylene')
        path = repr(hits).splitlines()[0].split('  ')[1].rstrip()
        m = ri.RefractiveIndexMaterial(*path.split('/'))
        self.assertEqual(repr(m), repr(hits[0].material()))

    def test_prints_all(self):
        """All matches are printed, also many; a slice prints fewer."""
        hits = ri.search('Si')
        self.assertGreater(len(hits), 100)
        self.assertEqual(len(repr(hits).splitlines()), len(hits))
        self.assertEqual(len(repr(hits[:10]).splitlines()), 10)
        self.assertEqual(str(hits), repr(hits))

    def test_slice_and_add(self):
        """Slices and + keep the print, numbered from 0."""
        hits = ri.search('SiO2')
        part = hits[5:8]
        self.assertEqual(list(part), list(hits)[5:8])
        self.assertEqual([line.split()[0] for line in repr(part).splitlines()], ['0', '1', '2'])
        both = hits[:2] + ri.search('Au')[:2]
        self.assertEqual(len(repr(both).splitlines()), 4)
        self.assertIsInstance(hits[0], type(hits[1]))


class TestSearchTestDatabase(unittest.TestCase):
    """ri.search and the repr of materials with a small database given by db_path."""

    @classmethod
    def setUpClass(cls):
        cls._tempdir = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls._tempdir.name)
        (cls.db_path / 'catalog-nk.yml').write_text(_TEST_CATALOG, encoding='utf-8')
        (cls.db_path / 'data' / 'main' / 'X2').mkdir(parents=True)
        (cls.db_path / 'data' / 'main' / 'X2' / 'Good.yml').write_text(_TEST_DATA, encoding='utf-8')

    @classmethod
    def tearDownClass(cls):
        cls._tempdir.cleanup()

    def test_names_without_html(self):
        """HTML tags are removed from names, so 'X<sub>2</sub>' is found as 'X2'."""
        hits = ri.search('test material', db_path=self.db_path)
        self.assertEqual(hits[0].book_name, 'X2 (Test material)')

    def test_page_listed_twice(self):
        """A page that the catalog lists twice is found once, with its first divider."""
        hits = ri.search('Good', db_path=self.db_path)
        self.assertEqual([(m.page, m.divider) for m in hits], [('Good', 'First')])

    def test_broken_page(self):
        """Filters skip a page whose data cannot be loaded, with one warning that names it."""
        self.assertEqual([m.page for m in ri.search(db_path=self.db_path)], ['Good', 'Broken'])
        with self.assertWarnsRegex(UserWarning, r'Skipped 1 page .*main/X2/Broken \(FileNotFoundError') as cm:
            hits = ri.search(wavelength=500, db_path=self.db_path)
        self.assertEqual(cm.filename, __file__)
        self.assertEqual([m.page for m in hits], ['Good'])

    def test_db_path_in_code(self):
        """Matches load from db_path, and the material shows db_path= if it is not the included database."""
        hits = ri.search('Good', db_path=str(self.db_path))
        self.assertAlmostEqual(float(hits[0].material().get_refractive_index(400)), 1.5)
        code = f"RefractiveIndexMaterial(shelf='main', book='X2', page='Good', db_path={str(self.db_path)!r})"
        self.assertEqual(repr(hits[0].material()), code)


class TestMaterialRepr(unittest.TestCase):
    """RefractiveIndexMaterial prints as the code that loads it."""

    def test_repr(self):
        from refractiveindex.refractiveindex import _DEFAULT_DB_PATH
        code = "RefractiveIndexMaterial(shelf='main', book='SiO2', page='Malitson')"
        for db_path in (None, _DEFAULT_DB_PATH, str(_DEFAULT_DB_PATH)):
            with self.subTest(db_path=db_path):
                m = ri.RefractiveIndexMaterial(shelf='main', book='SiO2', page='Malitson', db_path=db_path)
                self.assertEqual(repr(m), code)
        m = ri.RefractiveIndexMaterial.from_url('https://refractiveindex.info/?shelf=main&book=CaSO4&page=Querry-%CE%B2')
        self.assertEqual(repr(m), "RefractiveIndexMaterial(shelf='main', book='CaSO4', page='Querry-β')")


class TestSpacesInIds(unittest.TestCase):
    """A space and an underscore in shelf, book and page ids count as the same."""

    def test_space_for_underscore(self):
        """page='Rodriguez-de Marcos' loads Rodriguez-de_Marcos."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='SiO2', page='Rodriguez-de Marcos')
        ref = ri.RefractiveIndexMaterial(shelf='main', book='SiO2', page='Rodriguez-de_Marcos')
        self.assertEqual(m.get_refractive_index(500), ref.get_refractive_index(500))
        self.assertEqual(repr(m), "RefractiveIndexMaterial(shelf='main', book='SiO2', page='Rodriguez-de_Marcos')")

    def test_ids_with_spaces(self):
        """Ids that have spaces in the catalog load with spaces or with underscores."""
        book = 'C2ClF3_n - polychlorotrifluoroethylene'
        for written in (book, book.replace(' ', '_')):
            with self.subTest(book=written):
                m = ri.RefractiveIndexMaterial(shelf='organic', book=written, page='Smith')
                self.assertEqual(repr(m), f"RefractiveIndexMaterial(shelf='organic', book={book!r}, page='Smith')")

    def test_from_url_plus(self):
        """A + in a URL is a space, so it loads the page with an underscore."""
        m = ri.RefractiveIndexMaterial.from_url('https://refractiveindex.info/?shelf=main&book=SiO2&page=Rodriguez-de+Marcos')
        self.assertEqual(m._key, ('main', 'SiO2', 'Rodriguez-de_Marcos'))

    def test_not_found(self):
        """Other differences still raise KeyError."""
        with self.assertRaises(KeyError):
            ri.RefractiveIndexMaterial(shelf='main', book='SiO2', page='Rodriguez de Marcos')

    def test_search_shelf_and_book(self):
        """search(book=) and search(shelf=) accept spaces for underscores too."""
        self.assertEqual(ri.search(shelf='glass', book='fused silica'), ri.search(shelf='glass', book='fused_silica'))
        self.assertTrue(ri.search(shelf='popular glass'))
        self.assertTrue(ri.search(book='C2ClF3_n_-_polychlorotrifluoroethylene'))
