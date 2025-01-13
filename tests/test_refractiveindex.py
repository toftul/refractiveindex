"""Tests for the python interface to the RefractiveIndex databse."""

import unittest

import refractiveindex as ri


class RefractiveIndexTest(unittest.TestCase):
    def test_basic_usage(self):
        # Test the basic usage exercised in the project readme.
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
