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
                catalog.get(key), bad_path,
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
