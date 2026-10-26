import urllib.parse
import warnings
from pathlib import Path

import numpy as np
import scipy.interpolate
import yaml

try:
    from yaml import CBaseLoader as BaseLoader
except ImportError:
    from yaml import BaseLoader


# The database included in the package. scripts/update_database.py replaces it with an
# upstream release, and _database_version.py records which release it is.
_DEFAULT_DB_PATH = Path(__file__).parent / "database"

# Module-level cache: db_path -> {(shelf, book, page): filepath}
_catalog_cache = {}

# Temporary fixes for errors in the upstream catalog: (shelf, book, page) -> (bad_path, good_path).
# A fix is applied only while the catalog still has bad_path, so it becomes a no-op once
# upstream is fixed. TestCatalogFixes.test_fixes_still_needed then fails as a reminder
# to remove the entry.
_CATALOG_FIXES = {
    # https://github.com/toftul/refractiveindex/issues/10
    # Fixed upstream on develop (polyanskiy/refractiveindex.info-database@eb734f7a), not yet in a release.
    ("main", "MgO", "Synowicki"): ("main/As2S3/nk/Synowicki.yml", "main/MgO/nk/Synowicki.yml"),
    ("main", "ZrO2", "Synowicki"): ("main/As2S3/nk/Synowicki.yml", "main/ZrO2/nk/Synowicki.yml"),
}


class _Deprecated:
    """Default value of deprecated arguments, so that passing any value can be detected."""

    def __repr__(self):
        return "<deprecated>"


_DEPRECATED = _Deprecated()


def _read_catalog(db_path):
    """Parse catalog-nk.yml as is into {(shelf, book, page): data path relative to data/}."""
    catalog_file = db_path / "catalog-nk.yml"
    if not catalog_file.is_file():
        raise FileNotFoundError(
            f"No catalog-nk.yml in {db_path}. db_path must point at the database/ folder "
            "of refractiveindex.info-database, the folder with catalog-nk.yml and data/."
        )
    with open(catalog_file, "rt", encoding="utf-8") as f:
        catalog = yaml.load(f, Loader=BaseLoader)

    entries = {}
    for shelf in catalog:
        if "DIVIDER" in shelf:
            continue
        shelf_name = shelf["SHELF"]
        for book_entry in shelf.get("content", []):
            if "DIVIDER" in book_entry:
                continue
            book_name = book_entry["BOOK"]
            for page_entry in book_entry.get("content", []):
                if "DIVIDER" in page_entry:
                    continue
                entries[(shelf_name, book_name, page_entry["PAGE"])] = page_entry["data"]
    return entries


def _load_catalog(db_path):
    key = str(db_path)
    if key in _catalog_cache:
        return _catalog_cache[key]

    index = {}
    for material_key, data_rel in _read_catalog(db_path).items():
        fix = _CATALOG_FIXES.get(material_key)
        if fix is not None and data_rel == fix[0]:
            data_rel = fix[1]
        index[material_key] = db_path / "data" / Path(data_rel)

    _catalog_cache[key] = index
    return index


def _parse_tabulated(data_str):
    rows = data_str.strip().split("\n")
    wavelengths = []
    col1 = []
    col2 = []
    for row in rows:
        parts = row.split()
        if not parts:
            continue
        wavelengths.append(float(parts[0]))
        col1.append(float(parts[1]))
        if len(parts) > 2:
            col2.append(float(parts[2]))
    wl = np.array(wavelengths)
    c1 = np.array(col1)
    c2 = np.array(col2) if col2 else None
    return wl, c1, c2


def _compute_formula(formula_id, coefficients, wl_um):
    """Compute refractive index n from dispersion formula.

    Args:
        formula_id: Integer 1-9.
        coefficients: List of formula coefficients.
        wl_um: Wavelength(s) in micrometers (scalar or array).

    Returns:
        Refractive index n (same shape as wl_um).
    """
    wl = np.asarray(wl_um, dtype=float)
    C = coefficients
    # Zero-pad to avoid index errors
    Cp = list(C) + [0.0] * 20

    if formula_id == 1:  # Sellmeier
        nsq = 1 + Cp[0]
        for i in range(1, len(C), 2):
            nsq = nsq + C[i] * wl**2 / (wl**2 - C[i + 1] ** 2)
        return np.sqrt(nsq)

    elif formula_id == 2:  # Sellmeier-2
        nsq = 1 + Cp[0]
        for i in range(1, len(C), 2):
            nsq = nsq + C[i] * wl**2 / (wl**2 - C[i + 1])
        return np.sqrt(nsq)

    elif formula_id == 3:  # Polynomial
        nsq = Cp[0]
        for i in range(1, len(C), 2):
            nsq = nsq + C[i] * wl ** C[i + 1]
        return np.sqrt(nsq)

    elif formula_id == 4:  # RefractiveIndex.INFO
        nsq = Cp[0]
        for i in range(1, min(8, len(C)), 4):
            nsq = nsq + C[i] * wl ** C[i + 1] / (wl**2 - C[i + 2] ** C[i + 3])
        if len(C) > 9:
            for i in range(9, len(C), 2):
                nsq = nsq + C[i] * wl ** C[i + 1]
        return np.sqrt(nsq)

    elif formula_id == 5:  # Cauchy
        n = Cp[0]
        for i in range(1, len(C), 2):
            n = n + C[i] * wl ** C[i + 1]
        return n

    elif formula_id == 6:  # Gases
        n = 1 + Cp[0]
        for i in range(1, len(C), 2):
            n = n + C[i] / (C[i + 1] - wl ** (-2))
        return n

    elif formula_id == 7:  # Herzberger
        n = Cp[0]
        n = n + Cp[1] / (wl**2 - 0.028)
        n = n + Cp[2] / (wl**2 - 0.028) ** 2
        for i in range(3, len(C)):
            n = n + C[i] * wl ** (2 * (i - 2))
        return n

    elif formula_id == 8:  # Retro
        tmp = Cp[0] + Cp[1] * wl**2 / (wl**2 - Cp[2]) + Cp[3] * wl**2
        return np.sqrt((2 * tmp + 1) / (1 - tmp))

    elif formula_id == 9:  # Exotic
        return np.sqrt(
            Cp[0]
            + Cp[1] / (wl**2 - Cp[2])
            + Cp[3] * (wl - Cp[4]) / ((wl - Cp[4]) ** 2 + Cp[5])
        )

    else:
        raise ValueError(f"Unknown formula type: {formula_id}")


# Unit registry: maps unit name -> (to_um, from_um) callable pair.
# to_um:   converts a wavelength/wavenumber/frequency/energy value to micrometers.
# from_um: converts micrometers back to that unit.
# Add new units here; no other code needs to change.
# Note: energy/frequency units (eV, THz, cm-1) are self-inverse because λ ∝ 1/x.
h_const = 6.62607015e-34   # Planck constant [J·s]
c_const = 299792458        # Speed of light [m/s]
e_const = 1.602176634e-19  # Elementary charge [C]

# Derived: hc in eV·µm  (h [J·s] * c [m/s] / e [J/eV] * 1e6 [µm/m])
_hc_eV_um = h_const * c_const / e_const * 1e6

# Derived: c in µm·THz  (c [m/s] * 1e-6 [µm/m] / 1e12 [THz/Hz] * 1e12 simplifies to c * 1e-6)
_c_um_THz = c_const * 1e-6

_UNITS = {
    # Length units
    "m":       (lambda wl: wl * 1e6,              lambda wl_um: wl_um * 1e-6),
    "mm":      (lambda wl: wl * 1e3,              lambda wl_um: wl_um * 1e-3),
    "um":      (lambda wl: wl,                    lambda wl_um: wl_um),
    "nm":      (lambda wl: wl * 1e-3,             lambda wl_um: wl_um * 1e3),
    "A":       (lambda wl: wl * 1e-4,             lambda wl_um: wl_um * 1e4),
    # Reciprocal / energy units  (to_um == from_um)
    "cm-1":    (lambda wl: 1e4 / wl,              lambda wl_um: 1e4 / wl_um),
    "THz":     (lambda wl: _c_um_THz / wl,        lambda wl_um: _c_um_THz / wl_um),
    "eV":      (lambda wl: _hc_eV_um / wl,        lambda wl_um: _hc_eV_um / wl_um),
}


def _to_um(wl, unit):
    try:
        to_um, _ = _UNITS[unit]
    except KeyError:
        raise ValueError(f"unit must be one of {list(_UNITS)}, got {unit!r}")
    return to_um(np.asarray(wl, dtype=float))


def _from_um(wl_um, unit):
    try:
        _, from_um = _UNITS[unit]
    except KeyError:
        raise ValueError(f"unit must be one of {list(_UNITS)}, got {unit!r}")
    return from_um(wl_um)


# Time convention for get_epsilon: exp_type -> sign s of the time dependence exp(s*i*omega*t).
# j is the imaginary unit too, so only the sign in the string selects the convention.
_EXP_TYPES = {
    # exp(-iωt): epsilon = (n + ik)^2
    "exp(-iwt)": -1,
    "exp(-iωt)": -1,
    "exp(-jwt)": -1,
    "exp(-jωt)": -1,
    "exp_minus_i_omega_t": -1,
    # exp(+iωt): epsilon = (n - ik)^2
    "exp(+iwt)": +1,
    "exp(iwt)": +1,
    "exp(+iωt)": +1,
    "exp(iωt)": +1,
    "exp(+jwt)": +1,
    "exp(jwt)": +1,
    "exp(+jωt)": +1,
    "exp(jωt)": +1,
    "exp_plus_i_omega_t": +1,
}


# Values of quantity in get_wl_range. 'nk' is another name for 'epsilon': the overlap of n and k.
_WL_RANGE_QUANTITIES = ("epsilon", "nk", "n", "k")


def _make_interpolator(wavelengths, values):
    if len(wavelengths) == 1:
        val = values[0]
        return lambda wl: np.full_like(np.asarray(wl, dtype=float), val)
    interp = scipy.interpolate.interp1d(wavelengths, values, bounds_error=False)

    # Unit conversion can move a range end a few ulp out of the table, so that the
    # wavelengths from get_wl_range would give nan. Move such wavelengths onto the end.
    rtol = 1e-9
    lo, hi = np.min(wavelengths), np.max(wavelengths)

    def interpolate(wl):
        wl = np.asarray(wl, dtype=float)
        in_range = (wl >= lo * (1 - rtol)) & (wl <= hi * (1 + rtol))
        return interp(np.where(in_range, np.clip(wl, lo, hi), wl))

    return interpolate


class NoExtinctionCoefficient(Exception):
    pass


class RefractiveIndexMaterial:
    """Public API for looking up optical material properties.

    Wavelengths default to nanometers. Pass unit='um' to use micrometers.

    The data comes from the database included in the package (see DATABASE_VERSION).
    To use another copy, pass db_path= with the path of its database/ folder, the folder
    with catalog-nk.yml and data/.
    """

    def __init__(self, shelf, book, page, *, db_path=None,
                 # Deprecated since 1.2.0, to be removed in 2.0.0
                 auto_download=_DEPRECATED, update_database=_DEPRECATED,
                 ssl_certificate_location=_DEPRECATED, databasePath=_DEPRECATED):
        for name, value in (("auto_download", auto_download),
                            ("update_database", update_database),
                            ("ssl_certificate_location", ssl_certificate_location)):
            if value is not _DEPRECATED:
                warnings.warn(
                    f"{name} has no effect since refractiveindex 1.2.0 and will be removed in 2.0.0. "
                    "The database is part of the package. To use another copy, pass db_path=.",
                    DeprecationWarning, stacklevel=2,
                )
        if databasePath is not _DEPRECATED:
            warnings.warn(
                "databasePath will be removed in refractiveindex 2.0.0. Use db_path= instead.",
                DeprecationWarning, stacklevel=2,
            )
            if db_path is None:
                db_path = databasePath

        db_path = _DEFAULT_DB_PATH if db_path is None else Path(db_path)
        catalog = _load_catalog(db_path)

        key = (shelf, book, page)
        if key not in catalog:
            raise KeyError(f"Material not found: shelf={shelf!r}, book={book!r}, page={page!r}")
        filepath = catalog[key]

        with open(filepath, "rt", encoding="utf-8") as f:
            material = yaml.load(f, Loader=BaseLoader)

        self._n_func = None
        self._k_func = None
        # Wavelength ranges in µm of the n and k data, set together with _n_func and _k_func
        self._n_range = None
        self._k_range = None
        self._original_data = None

        for data in material["DATA"]:
            dtype = data["type"].split()
            category = dtype[0]
            subtype = dtype[1] if len(dtype) > 1 else None

            if category == "tabulated":
                wl, col1, col2 = _parse_tabulated(data["data"])
                # Some tables are not in wavelength order, so not (wl[0], wl[-1])
                wl_range = (wl.min(), wl.max())

                if subtype == "n":
                    self._n_func = _make_interpolator(wl, col1)
                    self._n_range = wl_range
                    self._original_data = {"wavelength (um)": wl, "n": col1}

                elif subtype == "k":
                    self._k_func = _make_interpolator(wl, col1)
                    self._k_range = wl_range
                    if self._original_data is None:
                        self._original_data = {"wavelength (um)": wl, "n": 1j * col1}

                elif subtype == "nk":
                    self._n_func = _make_interpolator(wl, col1)
                    self._k_func = _make_interpolator(wl, col2)
                    self._n_range = wl_range
                    self._k_range = wl_range
                    self._original_data = {"wavelength (um)": wl, "n": col1 + 1j * col2}

            elif category == "formula":
                formula_id = int(subtype)
                coefficients = [float(s) for s in data["coefficients"].split()]

                # Support both 'range' and 'wavelength_range' keys
                for range_key in ("range", "wavelength_range"):
                    if range_key in data:
                        break
                range_parts = data[range_key].split()
                range_min = float(range_parts[0])
                range_max = float(range_parts[1])

                self._n_range = (range_min, range_max)
                self._n_func =lambda wl_um, fid=formula_id, co=coefficients: _compute_formula(fid, co, wl_um)

                wl_sample = np.linspace(range_min, range_max, 1000)
                self._original_data = {
                    "wavelength (um)": wl_sample,
                    "n": _compute_formula(formula_id, coefficients, wl_sample),
                }

    @classmethod
    def from_url(cls, url, *, db_path=None):
        """Load the material of a refractiveindex.info page from its URL.

        Copy the URL from the address bar of the browser, for example
        https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson
        The data comes from the database selected by db_path, not from the website.
        """
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        missing = [name for name in ("shelf", "book", "page") if name not in query]
        if missing:
            raise ValueError(
                f"URL has no {', '.join(missing)} parameter{'s' if len(missing) > 1 else ''}: {url!r}. "
                "Copy the URL from the "
                "address bar of a material page on refractiveindex.info, for example "
                "https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson"
            )
        try:
            return cls(query["shelf"][0], query["book"][0], query["page"][0], db_path=db_path)
        except KeyError as e:
            raise KeyError(
                f"{e.args[0]}. The website can have materials that are not in the database in use. "
                "To use a newer copy, clone https://github.com/polyanskiy/refractiveindex.info-database "
                "and pass its database/ folder as db_path="
            ) from None

    def get_refractive_index(self, wavelength, unit="nm"):
        """Return refractive index n at the given wavelength(s).

        Args:
            wavelength: Wavelength(s) in the specified unit.
            unit: Input unit. Supported: 'm', 'mm', 'um', 'nm' (default),
                  'A' (Angstrom), 'cm-1', 'THz', 'eV'.
        """
        if self._n_func is None:
            raise Exception("No refractive index specified for this material")
        return self._n_func(_to_um(wavelength, unit))

    def get_extinction_coefficient(self, wavelength, unit="nm"):
        """Return extinction coefficient k at the given wavelength(s).

        Args:
            wavelength: Wavelength(s) in the specified unit.
            unit: Input unit. Supported: 'm', 'mm', 'um', 'nm' (default),
                  'A' (Angstrom), 'cm-1', 'THz', 'eV'.
        """
        if self._k_func is None:
            raise NoExtinctionCoefficient("No extinction coefficient specified for this material")
        return self._k_func(_to_um(wavelength, unit))

    def get_epsilon(self, wavelength, unit="nm", exp_type="exp(-iwt)"):
        """Return complex dielectric permittivity at the given wavelength(s).

        Args:
            wavelength: Wavelength(s) in the specified unit.
            unit: Input unit. Supported: 'm', 'mm', 'um', 'nm' (default),
                  'A' (Angstrom), 'cm-1', 'THz', 'eV'.
            exp_type: Time convention. j is the imaginary unit too, so only the sign selects it.
                  exp(-iωt), result (n + ik)**2: 'exp(-iwt)' (default), 'exp(-iωt)',
                  'exp(-jwt)', 'exp(-jωt)', 'exp_minus_i_omega_t'.
                  exp(+iωt), result (n - ik)**2: 'exp(+iwt)', 'exp(iwt)', 'exp(+iωt)',
                  'exp(iωt)', 'exp(+jwt)', 'exp(jwt)', 'exp(+jωt)', 'exp(jωt)',
                  'exp_plus_i_omega_t'.

        Note:
            If the material has no extinction coefficient data, k is taken as 0.
        """
        if exp_type not in _EXP_TYPES:
            minus = [name for name, sign in _EXP_TYPES.items() if sign < 0]
            plus = [name for name, sign in _EXP_TYPES.items() if sign > 0]
            raise ValueError(
                f"exp_type must be one of {minus} for exp(-iωt), "
                f"or one of {plus} for exp(+iωt), got {exp_type!r}"
            )
        n = self.get_refractive_index(wavelength, unit=unit)
        if self._k_func is None:
            k = 0.0
        else:
            k = self.get_extinction_coefficient(wavelength, unit=unit)
        if _EXP_TYPES[exp_type] < 0:
            return (n + 1j * k) ** 2
        return (n - 1j * k) ** 2

    def get_wl_range(self, unit="nm", quantity="epsilon"):
        """Return the wavelength range of the data as (min, max) in the requested unit.

        Args:
            unit: Output unit. Supported: 'm', 'mm', 'um', 'nm' (default),
                  'A' (Angstrom), 'cm-1', 'THz', 'eV'.
            quantity: The data that the range is for.
                  'epsilon' (default) or 'nk': the overlap of the n and k ranges, where
                  get_epsilon is valid. Without k data, the n range, as get_epsilon then
                  takes k as 0.
                  'n': the range of the n data, where get_refractive_index is valid.
                  'k': the range of the k data, where get_extinction_coefficient is valid.

        Raises:
            Exception: quantity is 'n', 'nk' or 'epsilon' and the material has no n data,
                  as from get_refractive_index.
            NoExtinctionCoefficient: quantity is 'k' and the material has no k data.
            ValueError: The n and k ranges do not overlap, or quantity or unit is unknown.

        Note:
            The result is always sorted (min, max). For reciprocal units
            ('cm-1', 'THz', 'eV') this reverses the wavelength ordering.
        """
        if quantity not in _WL_RANGE_QUANTITIES:
            raise ValueError(f"quantity must be one of {list(_WL_RANGE_QUANTITIES)}, got {quantity!r}")

        def convert(wl_range):
            a = _from_um(wl_range[0], unit)
            b = _from_um(wl_range[1], unit)
            return (min(a, b), max(a, b))

        if quantity == "k":
            if self._k_range is None:
                raise NoExtinctionCoefficient("No extinction coefficient specified for this material")
            return convert(self._k_range)
        if self._n_range is None:
            raise Exception("No refractive index specified for this material")
        if quantity == "n" or self._k_range is None:
            return convert(self._n_range)

        lo = max(self._n_range[0], self._k_range[0])
        hi = min(self._n_range[1], self._k_range[1])
        if lo > hi:
            n_lo, n_hi = convert(self._n_range)
            k_lo, k_hi = convert(self._k_range)
            raise ValueError(
                f"The n data ({n_lo:g}–{n_hi:g} {unit}) and the k data ({k_lo:g}–{k_hi:g} {unit}) "
                "do not overlap. Use quantity='n' or quantity='k'."
            )
        return convert((lo, hi))
