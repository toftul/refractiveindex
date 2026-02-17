"""Tests for the python interface to the RefractiveIndex database."""

import unittest

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

    def test_epsilon_n_ik_consistency(self):
        """Verify (n + ik)^2 == epsilon identity for both conventions."""
        m = ri.RefractiveIndexMaterial(shelf='main', book='Ag', page='Johnson')
        wl = 600
        eps_minus = m.get_epsilon(wl, exp_type='exp_minus_i_omega_t')
        eps_plus = m.get_epsilon(wl, exp_type='exp_plus_i_omega_t')
        # Real parts should be equal, imaginary parts opposite sign
        self.assertAlmostEqual(eps_minus.real, eps_plus.real)
        self.assertAlmostEqual(eps_minus.imag, -eps_plus.imag)
