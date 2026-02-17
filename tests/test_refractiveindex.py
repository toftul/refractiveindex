"""Tests for the python interface to the RefractiveIndex database."""

import unittest

import numpy as np

import refractiveindex as ri


class RefractiveIndexTest(unittest.TestCase):
    def test_basic_usage(self):
        """Test tabulated nk material (SiO/Hass) — backward compat."""
        SiO = ri.RefractiveIndexMaterial(shelf='main', book='SiO', page='Hass')
        wavelength_nm = 600

        epsilon = SiO.get_epsilon(wavelength_nm)
        expected_epsilon = 3.8633404437869827 + 0.003931076923076923j
        self.assertAlmostEqual(epsilon, expected_epsilon)

        refractive_index = SiO.get_refractive_index(wavelength_nm)
        expected_refractive_index = 1.96553846
        self.assertAlmostEqual(refractive_index, expected_refractive_index)

        extinction_coefficient = SiO.get_extinction_coefficient(wavelength_nm)
        expected_extinction_coefficient = 0.001
        self.assertAlmostEqual(extinction_coefficient, expected_extinction_coefficient)

    def test_formula_material(self):
        """Test a formula-based material (BK7 glass, Sellmeier formula 1)."""
        bk7 = ri.RefractiveIndexMaterial(shelf='specs', book='SCHOTT-optical', page='N-BK7')
        n = bk7.get_refractive_index(589.3)  # sodium D line
        # N-BK7 at 589.3 nm should be ~1.5168
        self.assertAlmostEqual(n, 1.5168, places=3)

    def test_tabulated_nk_material(self):
        """Test a tabulated nk material (Ag/Johnson)."""
        ag = ri.RefractiveIndexMaterial(shelf='main', book='Ag', page='Johnson')
        n = ag.get_refractive_index(500)
        k = ag.get_extinction_coefficient(500)
        # Silver at 500 nm: n ~ 0.05, k ~ 3.13 (Johnson & Christy)
        self.assertAlmostEqual(n, 0.05, places=1)
        self.assertAlmostEqual(k, 3.13, places=1)

    def test_array_input(self):
        """Test that array wavelength input returns arrays."""
        SiO = ri.RefractiveIndexMaterial(shelf='main', book='SiO', page='Hass')
        wavelengths = np.array([400, 500, 600, 700])
        n = SiO.get_refractive_index(wavelengths)
        self.assertEqual(n.shape, (4,))
        # All values should be reasonable refractive indices
        self.assertTrue(np.all(n > 1.0))
        self.assertTrue(np.all(n < 3.0))

    def test_no_extinction_coefficient(self):
        """Test that NoExtinctionCoefficient is raised for materials without k."""
        baf2 = ri.RefractiveIndexMaterial(shelf='main', book='BaF2', page='Malitson')
        with self.assertRaises(ri.NoExtinctionCoefficient):
            baf2.get_extinction_coefficient(500)
