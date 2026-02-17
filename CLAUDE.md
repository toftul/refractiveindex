# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python interface to the [refractiveindex.info](https://refractiveindex.info) database. Provides lookup of refractive index (n), extinction coefficient (k), and dielectric permittivity (epsilon) for optical materials. The database YAML files are auto-downloaded from GitHub to `~/.refractiveindex.info-database` on first use.

## Build & Test Commands

- **Install**: `pip install .` (uses Poetry backend via pyproject.toml)
- **Install dev (editable)**: `pip install -e .`
- **Run tests**: `pytest` or `python -m pytest tests/test_refractiveindex.py`
- **Run single test**: `pytest tests/test_refractiveindex.py::RefractiveIndexTest::test_basic_usage`
- **Local venv**: `.venv/` in project root

Tests require network access on first run (downloads the database).

## Architecture

Single-module library (`refractiveindex/refractiveindex.py`) with 1 public class + helper functions:

- **`RefractiveIndexMaterial`** — Public API. Takes `shelf`, `book`, `page` identifiers to locate a material. All wavelengths are in **nanometers**. Stores `_n_func` and `_k_func` callables internally.
- **`_load_catalog(db_path)`** — Parses `catalog-nk.yml` into a flat `(shelf, book, page) -> filepath` dict, cached at module level in `_catalog_cache`.
- **`_ensure_database()` / `_download_database()`** — Downloads the database if missing. Pinned to a specific commit via `_DATABASE_SHA`.
- **`_compute_formula(formula_id, coefficients, wl_um)`** — All 9 dispersion formulas (Sellmeier, Cauchy, Herzberger, Retro, Exotic, etc.) in one function.
- **`_parse_tabulated(data_str)`** — Parses tabulated `n`, `k`, `nk` data from YAML.
- **`_make_interpolator(wavelengths, values)`** — Wraps `scipy.interpolate.interp1d`.

Internally, wavelengths are converted from nm to μm (divided by 1000) at the public API boundary (`get_refractive_index`, `get_extinction_coefficient`, `get_epsilon`).

## Key Conventions

- Spaces in material page names must be replaced with underscores (e.g., `Rodriguez-de_Marcos` not `Rodriguez-de Marcos`).
- Time dependence convention: e^{-iωt}, so Im(ε) > 0 means lossy.
- All `tabulated n/k/nk` and `formula 1-9` data types are supported.
- Public exports: `RefractiveIndexMaterial` and `NoExtinctionCoefficient` (see `__init__.py`).
- Database material paths use `data/` subdirectory (not the old `data-nk/`).
- The catalog YAML has top-level `DIVIDER` entries that must be skipped during parsing.
