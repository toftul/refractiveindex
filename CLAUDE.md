# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python interface to the [refractiveindex.info](https://refractiveindex.info) database. Provides lookup of refractive index (n), extinction coefficient (k), and dielectric permittivity (epsilon) for optical materials. The database is included in the package at `refractiveindex/database/`: an unchanged copy of an upstream release (plus its `LICENSE`), whose tag and commit are in `refractiveindex/_database_version.py`. Nothing is downloaded at run time.

## Build & Test Commands

- **Install**: `pip install .` (uses Poetry backend via pyproject.toml)
- **Install dev (editable)**: `pip install -e .`
- **Run tests**: `pytest` or `python -m pytest tests/test_refractiveindex.py`
- **Run single test**: `pytest tests/test_refractiveindex.py::RefractiveIndexTest::test_basic_usage`
- **Local venv**: `.venv/` in project root

Tests need no network access.

- **Update the included database**: `python scripts/update_database.py [TAG] [--changelog]` (default: latest upstream release). It replaces `refractiveindex/database/` and rewrites `_database_version.py`; a second run for the same tag gives no git diff. Never edit files in `refractiveindex/database/` by hand. `.github/workflows/update-database.yml` runs the script weekly and opens a PR.

## Architecture

Single-module library (`refractiveindex/refractiveindex.py`) with 1 public class, `search` + helper functions:

- **`RefractiveIndexMaterial`** — Public API. Takes `shelf`, `book`, `page` identifiers to locate a material. All wavelengths are in **nanometers**. Stores `_n_func` and `_k_func` callables internally. `db_path=` selects another `database/` folder (default `_DEFAULT_DB_PATH`, the included one). `__repr__` is the constructor call (`_material_code`), with `db_path=` only when it is not the default.
- **`RefractiveIndexMaterial.from_url(url, db_path=None)`** — Classmethod. Reads `shelf`, `book`, `page` from the query string of a refractiveindex.info address-bar URL and calls the constructor. Other URLs, such as data-file links, raise `ValueError`.
- **`get_wl_range(unit="nm", quantity="epsilon")`** — `_n_range` and `_k_range` (µm) are stored separately, each from the smallest and largest wavelength of the block that sets `_n_func` or `_k_func`. `quantity` is `"n"`, `"k"`, or `"epsilon"`/`"nk"` (the overlap; the n range without k data). Raises like the matching getter when the data is missing, `ValueError` when n and k do not overlap.
- **`_DEPRECATED`** — Default of `auto_download`, `update_database`, `ssl_certificate_location` and `databasePath`. Passing any of them gives a `DeprecationWarning`; the first three have no effect, `databasePath` is an alias for `db_path`. Remove all four in 2.0.0.
- **`search(query=None, *, shelf, book, quantity, wavelength, unit="nm", db_path)`** — Module-level, exported. Each whitespace-separated query word must be a case-insensitive substring of book id, page id, book name, page name or divider. Ranked by `_search_tier` (0: word == book id, 1: whole word of book name, 2: part of book id, 3: elsewhere; best tier over words), catalog order within a tier. `shelf`/`book` are exact ids (str or list), unknown ones raise `ValueError`. `quantity`/`wavelength` load each match and keep it if `get_wl_range(quantity=quantity or "epsilon")` contains the wavelength or the whole `(min, max)` interval (sorted after unit conversion, `_RANGE_RTOL` at the ends). Pages that fail to load are skipped with one `UserWarning`. Duplicates (same data file on other shelves) are kept.
- **`Match`** — Frozen dataclass returned by `search` (not exported): `shelf, book, page, book_name, name, divider`, plus `db_path` hidden from repr and comparison. `.material()` loads it.
- **`_SearchResults`** — `list` subclass returned by `search`. `__repr__` prints one line per item, `index  shelf/book/page  name [divider]` with the names aligned after the longest path, or `No matches`; no truncation, slice to print fewer. Slices and `+` keep the type, numbered from 0 again.
- **`_read_catalog(db_path)`** — Parses `catalog-nk.yml` as is into a flat `(shelf, book, page) -> _CatalogEntry(path, book_name, name, divider)` dict: data path relative to `data/`, names without HTML tags, divider above the page. A key listed twice keeps its first entry.
- **`_load_catalog(db_path)`** — Applies `_CATALOG_FIXES` to `_read_catalog` output and replaces `path` with the full filepath, cached at module level in `_catalog_cache`.
- **`_CATALOG_FIXES`** — Temporary fixes for upstream catalog errors, applied at run time so the included database stays an exact copy of upstream. Each applies only while the catalog still has the bad path. `TestCatalogFixes.test_fixes_still_needed` fails once the included release has the fix (for example in the weekly update PR); then remove the entry.
- **`_compute_formula(formula_id, coefficients, wl_um)`** — All 9 dispersion formulas (Sellmeier, Cauchy, Herzberger, Retro, Exotic, etc.) in one function.
- **`_parse_tabulated(data_str)`** — Parses tabulated `n`, `k`, `nk` data from YAML.
- **`_make_interpolator(wavelengths, values)`** — Wraps `scipy.interpolate.interp1d`. Wavelengths within a relative `_RANGE_RTOL` (1e-9) of a table end are moved onto the end, because unit conversion rounding would otherwise give `nan` at the ends from `get_wl_range`. Farther out, the result is still `nan`.

Internally, wavelengths are converted from nm to μm (divided by 1000) at the public API boundary (`get_refractive_index`, `get_extinction_coefficient`, `get_epsilon`).

## Key Conventions

- A space and an underscore in ids count as the same (`_underscored`): the constructor tries the exact key first, then compares with spaces as underscores, because a few book ids really contain spaces (e.g. `organic/C2ClF3_n - polychlorotrifluoroethylene`). `search(shelf=, book=)` compares the same way. No id contains `/`.
- Time dependence convention: e^{-iωt}, so Im(ε) > 0 means lossy.
- All `tabulated n/k/nk` and `formula 1-9` data types are supported.
- Public exports: `RefractiveIndexMaterial`, `NoExtinctionCoefficient`, `search`, `DATABASE_VERSION` and `DATABASE_COMMIT` (see `__init__.py`).
- Database material paths use `data/` subdirectory (not the old `data-nk/`).
- The catalog YAML has top-level `DIVIDER` entries that must be skipped during parsing.
