import difflib
import re
import urllib.parse
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

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

# Module-level cache: db_path -> {(shelf, book, page): _CatalogEntry with the full filepath}
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


class _CatalogEntry(NamedTuple):
    """A page of the catalog. Names are without HTML tags, so 'SiO<sub>2</sub>' is 'SiO2'."""

    path: str | Path  # data path relative to data/ from _read_catalog, full filepath from _load_catalog
    book_name: str
    name: str
    divider: str | None  # the divider above the page in its book


def _strip_html(text):
    return re.sub(r"<[^>]+>", "", text)


def _read_catalog(db_path):
    """Parse catalog-nk.yml as is into {(shelf, book, page): _CatalogEntry}.

    A page that the catalog lists twice in a book keeps its first entry.
    """
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
            book_title = _strip_html(book_entry.get("name", ""))
            divider = None
            for page_entry in book_entry.get("content", []):
                if "DIVIDER" in page_entry:
                    divider = _strip_html(page_entry["DIVIDER"])
                    continue
                entries.setdefault(
                    (shelf_name, book_name, page_entry["PAGE"]),
                    _CatalogEntry(page_entry["data"], book_title,
                                  _strip_html(page_entry.get("name", "")), divider),
                )
    return entries


def _load_catalog(db_path):
    key = str(db_path)
    if key in _catalog_cache:
        return _catalog_cache[key]

    index = {}
    for material_key, entry in _read_catalog(db_path).items():
        data_rel = entry.path
        fix = _CATALOG_FIXES.get(material_key)
        if fix is not None and data_rel == fix[0]:
            data_rel = fix[1]
        index[material_key] = entry._replace(path=db_path / "data" / Path(data_rel))

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

# Relative distance from a range end that counts as on the end, for rounding in unit conversion
_RANGE_RTOL = 1e-9


def _make_interpolator(wavelengths, values):
    if len(wavelengths) == 1:
        val = values[0]
        return lambda wl: np.full_like(np.asarray(wl, dtype=float), val)
    interp = scipy.interpolate.interp1d(wavelengths, values, bounds_error=False)

    # Unit conversion can move a range end a few ulp out of the table, so that the
    # wavelengths from get_wl_range would give nan. Move such wavelengths onto the end.
    rtol = _RANGE_RTOL
    lo, hi = np.min(wavelengths), np.max(wavelengths)

    def interpolate(wl):
        wl = np.asarray(wl, dtype=float)
        in_range = (wl >= lo * (1 - rtol)) & (wl <= hi * (1 + rtol))
        return interp(np.where(in_range, np.clip(wl, lo, hi), wl))

    return interpolate


def _material_code(shelf, book, page, db_path):
    """The code that loads a page, with db_path= only if it is not the included database."""
    args = f"shelf={shelf!r}, book={book!r}, page={page!r}"
    if db_path != _DEFAULT_DB_PATH:
        args += f", db_path={str(db_path)!r}"
    return f"RefractiveIndexMaterial({args})"


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
        filepath = catalog[key].path
        self._key = key
        self._db_path = db_path

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

    def __repr__(self):
        return _material_code(*self._key, self._db_path)

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


@dataclass(frozen=True)
class Match:
    """A page found by search(). material() loads it."""

    shelf: str
    book: str
    page: str
    book_name: str
    name: str
    divider: str | None
    db_path: Path = field(repr=False, compare=False)

    def material(self):
        """Load the page as a RefractiveIndexMaterial, from the database that search() used."""
        return RefractiveIndexMaterial(self.shelf, self.book, self.page, db_path=self.db_path)


class _SearchResults(list):
    """The list of Match from search(). Prints one line per match: index, shelf/book/page, name."""

    def __getitem__(self, index):
        result = super().__getitem__(index)
        return _SearchResults(result) if isinstance(index, slice) else result

    def __add__(self, other):
        result = super().__add__(other)
        return result if result is NotImplemented else _SearchResults(result)

    def __repr__(self):
        if not self:
            return "No matches"
        paths = [f"{m.shelf}/{m.book}/{m.page}" for m in self]
        number_width = len(str(len(self) - 1))
        path_width = max(len(path) for path in paths)
        lines = []
        for i, (match, path) in enumerate(zip(self, paths)):
            description = match.name if match.divider is None else f"{match.name} [{match.divider}]"
            lines.append(f"{i:>{number_width}}  {path:<{path_width}}  {description}")
        return "\n".join(lines)


def _search_ids(value, name, catalog_ids):
    """The ids given as shelf= or book= of search(), as a set. None if not given."""
    if value is None:
        return None
    ids = [value] if isinstance(value, str) else list(value)
    valid = dict.fromkeys(catalog_ids)
    for id_ in ids:
        if id_ in valid:
            continue
        if name == "shelf":
            hint = f" Shelves: {', '.join(map(repr, valid))}."
        else:
            close = ([v for v in valid if v.lower() == str(id_).lower()]
                     or difflib.get_close_matches(str(id_), valid, n=3))
            hint = f" Did you mean {' or '.join(map(repr, close))}?" if close else ""
        raise ValueError(f"No {name} {id_!r} in the catalog.{hint}")
    return set(ids)


def _search_tier(words, book, page, entry):
    """Rank of a page for the lower-case query words, 0 best, or None if a word is not found.

    0: a word is the book id. 1: a word is a whole word of the book name.
    2: a word is part of the book id. 3: a word is somewhere else.
    """
    book_lower = book.lower()
    book_name = entry.book_name.lower()
    text = " ".join((book_lower, page.lower(), book_name, entry.name.lower(), (entry.divider or "").lower()))
    best = 3
    for word in words:
        if word not in text:
            return None
        if word == book_lower:
            tier = 0
        elif re.search(rf"(?<!\w){re.escape(word)}(?!\w)", book_name):
            tier = 1
        elif word in book_lower:
            tier = 2
        else:
            tier = 3
        best = min(best, tier)
    return best


def search(query=None, *, shelf=None, book=None, quantity=None, wavelength=None, unit="nm", db_path=None):
    """Find pages of the database by text, shelf, book and the range of their data.

    Args:
        query: Words to find, for example 'SiO2', 'silica' or 'SiO2 Malitson'. Each word must
              appear, ignoring case, in the book or page id, the book or page name, or the
              divider above the page in its book. None (default) finds every page.
        shelf: Shelf id or a list of them, for example 'main'. Exact and case-sensitive.
        book: Book id or a list of them, for example 'SiO2'. Exact and case-sensitive.
        quantity: Keep only the pages that have this data, as in get_wl_range: 'n', 'k',
              or 'epsilon' or 'nk', which need n data only. A page is kept if
              get_wl_range(quantity=quantity) gives a range.
        wavelength: Keep only the pages whose range for quantity (default 'epsilon') contains
              this wavelength, or all of a (min, max) interval. The ends count as inside.
        unit: Unit of wavelength. Supported: 'm', 'mm', 'um', 'nm' (default),
              'A' (Angstrom), 'cm-1', 'THz', 'eV'.
        db_path: The database/ folder to search, as for RefractiveIndexMaterial.
              None (default) is the database included in the package.

    Returns:
        A list of Match, best first: pages whose book id is a query word, then pages with a
        query word in the book name, then pages whose book id contains a query word, then
        the others, each group in catalog order. It prints one line per match: its index,
        shelf/book/page, and the page name with the divider in brackets. Slice it to print
        fewer. Match.material() loads a page.

    Raises:
        ValueError: shelf or book is not in the catalog, quantity or unit is unknown,
              or wavelength is not a number or a pair of numbers.

    Warns:
        UserWarning: With quantity or wavelength, pages whose data cannot be loaded are skipped.
    """
    db_path = _DEFAULT_DB_PATH if db_path is None else Path(db_path)
    catalog = _load_catalog(db_path)
    shelves = _search_ids(shelf, "shelf", (key[0] for key in catalog))
    books = _search_ids(book, "book", (key[1] for key in catalog))
    if quantity is not None and quantity not in _WL_RANGE_QUANTITIES:
        raise ValueError(f"quantity must be one of {list(_WL_RANGE_QUANTITIES)}, got {quantity!r}")
    if unit not in _UNITS:
        raise ValueError(f"unit must be one of {list(_UNITS)}, got {unit!r}")
    wl_range = None
    if wavelength is not None:
        wl_um = _to_um(wavelength, unit)
        if wl_um.shape not in ((), (2,)):
            raise ValueError(f"wavelength must be a number or a (min, max) pair, got {wavelength!r}")
        # Sorted after conversion, as reciprocal units reverse the order
        wl_range = (wl_um.min(), wl_um.max())

    words = [] if query is None else query.lower().split()
    ranked = []
    for order, ((shelf_id, book_id, page_id), entry) in enumerate(catalog.items()):
        if shelves is not None and shelf_id not in shelves:
            continue
        if books is not None and book_id not in books:
            continue
        tier = _search_tier(words, book_id, page_id, entry)
        if tier is not None:
            match = Match(shelf_id, book_id, page_id, entry.book_name, entry.name, entry.divider, db_path)
            ranked.append((tier, order, match))
    ranked.sort(key=lambda item: item[:2])
    matches = [match for _, _, match in ranked]

    if quantity is not None or wl_range is not None:
        kept, failed = [], []
        for match in matches:
            try:
                # Some formulas give nan inside their own range when the constructor samples them
                with np.errstate(all="ignore"):
                    material = match.material()
            except Exception as e:
                failed.append(f"{match.shelf}/{match.book}/{match.page} ({type(e).__name__}: {e})")
                continue
            try:
                lo, hi = material.get_wl_range(unit="um", quantity=quantity or "epsilon")
            except Exception:
                continue  # No data for quantity, or the n and k data do not overlap
            if wl_range is None or (lo * (1 - _RANGE_RTOL) <= wl_range[0]
                                    and wl_range[1] <= hi * (1 + _RANGE_RTOL)):
                kept.append(match)
        if failed:
            warnings.warn(
                f"Skipped {len(failed)} page{'s' if len(failed) > 1 else ''} that could not be loaded: "
                + ", ".join(failed),
                UserWarning, stacklevel=2,
            )
        matches = kept
    return _SearchResults(matches)
