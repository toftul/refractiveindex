import sys
from pathlib import Path

import numpy as np
import scipy.interpolate
import yaml

try:
    from yaml import CBaseLoader as BaseLoader
except ImportError:
    from yaml import BaseLoader


# Latest commit as of 2026-02-17
# https://github.com/polyanskiy/refractiveindex.info-database/commits/master/
_DATABASE_SHA = "a66ef8805cdb200973fc7ae9181587e1d89d14eb"

_DEFAULT_DB_PATH = Path.home() / ".refractiveindex.info-database"

# Module-level cache: db_path -> {(shelf, book, page): filepath}
_catalog_cache = {}


def _download_database(db_path, ssl_certificate_location=None):
    import shutil
    import ssl
    import tempfile
    import urllib.request
    import zipfile

    url = f"https://github.com/polyanskiy/refractiveindex.info-database/archive/{_DATABASE_SHA}.zip"

    if ssl_certificate_location is not None:
        if ssl_certificate_location == "":
            ssl._create_default_https_context = ssl._create_unverified_context
        else:
            if not ssl_certificate_location.endswith(".pem") or not Path(ssl_certificate_location).is_file():
                raise ValueError(
                    f"Does not appear to be an existing .pem certificate file: {ssl_certificate_location}"
                )
            ssl._create_default_https_context = ssl.create_default_context(cafile=ssl_certificate_location)

    with tempfile.TemporaryDirectory() as tempdir:
        zip_path = Path(tempdir) / "db.zip"
        print("downloading refractiveindex.info database...", file=sys.stderr)
        urllib.request.urlretrieve(url, zip_path)
        print("extracting...", file=sys.stderr)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tempdir)

        if db_path.is_dir():
            print("removing old database...", file=sys.stderr)
            shutil.rmtree(db_path)

        extracted = Path(tempdir) / f"refractiveindex.info-database-{_DATABASE_SHA}" / "database"
        shutil.move(str(extracted), str(db_path))
        print("done", file=sys.stderr)


def _ensure_database(db_path, auto_download, update_database, ssl_certificate_location):
    if not db_path.exists() and auto_download or update_database:
        _download_database(db_path, ssl_certificate_location)
    return db_path


def _load_catalog(db_path):
    key = str(db_path)
    if key in _catalog_cache:
        return _catalog_cache[key]

    catalog_file = db_path / "catalog-nk.yml"
    with open(catalog_file, "rt", encoding="utf-8") as f:
        catalog = yaml.load(f, Loader=BaseLoader)

    index = {}
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
                page_name = page_entry["PAGE"]
                data_rel = page_entry["data"]
                filepath = db_path / "data" / Path(data_rel)
                index[(shelf_name, book_name, page_name)] = filepath

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


def _make_interpolator(wavelengths, values):
    if len(wavelengths) == 1:
        val = values[0]
        return lambda wl: np.full_like(np.asarray(wl, dtype=float), val)
    return scipy.interpolate.interp1d(wavelengths, values, bounds_error=False)


class NoExtinctionCoefficient(Exception):
    pass


class RefractiveIndexMaterial:
    """Public API for looking up optical material properties.

    All wavelengths are in nanometers.
    """

    def __init__(self, shelf, book, page, *,
                 db_path=None, auto_download=True,
                 update_database=False, ssl_certificate_location=None,
                 # Legacy kwargs
                 databasePath=None, **_ignored):
        # Support legacy kwarg
        if db_path is None and databasePath is not None:
            db_path = Path(databasePath)
        if db_path is None:
            db_path = _DEFAULT_DB_PATH
        else:
            db_path = Path(db_path)

        _ensure_database(db_path, auto_download, update_database, ssl_certificate_location)
        catalog = _load_catalog(db_path)

        key = (shelf, book, page)
        if key not in catalog:
            raise KeyError(f"Material not found: shelf={shelf!r}, book={book!r}, page={page!r}")
        filepath = catalog[key]

        with open(filepath, "rt", encoding="utf-8") as f:
            material = yaml.load(f, Loader=BaseLoader)

        self._n_func = None
        self._k_func = None
        self._wl_range = None
        self._original_data = None

        for data in material["DATA"]:
            dtype = data["type"].split()
            category = dtype[0]
            subtype = dtype[1] if len(dtype) > 1 else None

            if category == "tabulated":
                wl, col1, col2 = _parse_tabulated(data["data"])

                if subtype == "n":
                    self._n_func = _make_interpolator(wl, col1)
                    self._wl_range = (wl[0], wl[-1])
                    self._original_data = {"wavelength (um)": wl, "n": col1}

                elif subtype == "k":
                    self._k_func = _make_interpolator(wl, col1)
                    if self._wl_range is None:
                        self._wl_range = (wl[0], wl[-1])
                    if self._original_data is None:
                        self._original_data = {"wavelength (um)": wl, "n": 1j * col1}

                elif subtype == "nk":
                    self._n_func = _make_interpolator(wl, col1)
                    self._k_func = _make_interpolator(wl, col2)
                    self._wl_range = (wl[0], wl[-1])
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

                self._wl_range = (range_min, range_max)
                self._n_func = lambda wl_um, fid=formula_id, co=coefficients: _compute_formula(fid, co, wl_um)

                wl_sample = np.linspace(range_min, range_max, 1000)
                self._original_data = {
                    "wavelength (um)": wl_sample,
                    "n": _compute_formula(formula_id, coefficients, wl_sample),
                }

    def get_refractive_index(self, wavelength_nm):
        """Return refractive index n at the given wavelength(s) in nm."""
        if self._n_func is None:
            raise Exception("No refractive index specified for this material")
        wl_um = np.asarray(wavelength_nm, dtype=float) / 1000.0
        return self._n_func(wl_um)

    def get_extinction_coefficient(self, wavelength_nm):
        """Return extinction coefficient k at the given wavelength(s) in nm."""
        if self._k_func is None:
            raise NoExtinctionCoefficient("No extinction coefficient specified for this material")
        wl_um = np.asarray(wavelength_nm, dtype=float) / 1000.0
        return self._k_func(wl_um)

    def get_epsilon(self, wavelength_nm, exp_type="exp_minus_i_omega_t"):
        """Return complex dielectric permittivity at the given wavelength(s) in nm."""
        n = self.get_refractive_index(wavelength_nm)
        k = self.get_extinction_coefficient(wavelength_nm)
        if exp_type == "exp_minus_i_omega_t":
            return (n + 1j * k) ** 2
        else:
            return (n - 1j * k) ** 2
        
    def get_wl_range(self):
        """Return the valid wavelength range as (min, max) in nanometers, or None if unknown."""
        if self._wl_range is None:
            return None
        return (self._wl_range[0] * 1000.0, self._wl_range[1] * 1000.0)
