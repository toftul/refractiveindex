# Changelog

This file lists the changes in each release of `refractiveindex`.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `RefractiveIndexMaterial.from_url(url)` loads a material from the URL of its page on refractiveindex.info,
  for example `https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson`.
  The data comes from the included database (or `db_path=`), not from the website.

## [1.2.0] - 2026-10-26

### Added

- The package now includes the refractiveindex.info database, release v2026-05-24.
  The library does not download the database any more and needs no network access.
  The package download is about 15 MB.
- `refractiveindex.DATABASE_VERSION` and `refractiveindex.DATABASE_COMMIT` give the release and the upstream commit of the included database.
- A weekly GitHub workflow opens a PR when upstream publishes a new database release.
- The publish workflow stops if the wheel does not contain all database files.

### Changed

- All users of a package version now get the same data.
  Before this change, each user got the database version that the library downloaded on first use.
- `db_path` must point at a folder with `catalog-nk.yml`, for example the `database/` folder of the upstream repository.
  Before this change, the library downloaded the database into a missing folder. Now it raises `FileNotFoundError`.
- The weekly CI test run is removed, because the data no longer changes between runs.

### Deprecated

- `auto_download`, `update_database` and `ssl_certificate_location` have no effect and give a `DeprecationWarning`.
  Version 2.0.0 will remove them.
- `databasePath` gives a `DeprecationWarning`. Use `db_path` instead. Version 2.0.0 will remove it.

### Removed

- The download code. The library no longer reads or writes `~/.refractiveindex.info-database`, so you can delete this folder.

## [1.1.0] - 2026-10-26

### Added

- `get_epsilon` accepts `exp_type="exp(-iwt)"` and `exp_type="exp(+iwt)"`.
  It also accepts `"exp(iwt)"`, `j` instead of `i`, and `ω` instead of `w`, for example `"exp(-jωt)"`.
  `j` is the imaginary unit, so only the sign in the string selects the convention.
  The old names `"exp_minus_i_omega_t"` and `"exp_plus_i_omega_t"` continue to work.

### Changed

- The default value of `exp_type` is now `"exp(-iwt)"`.
  It has the same meaning as the old default `"exp_minus_i_omega_t"`, so the result does not change.

### Fixed

- `ssl_certificate_location` with a path to a `.pem` file now works.
  Before this fix, every HTTPS request in the process failed with `TypeError: 'SSLContext' object is not callable`.
- `ssl_certificate_location` now applies only to the database download.
  Before this fix, it changed the default for all HTTPS requests in the process.
  With `ssl_certificate_location=""`, this turned off certificate checks for all of them.
- The request for the latest database commit now uses `ssl_certificate_location` too.
  Before this fix, this request ignored it, failed, and the library downloaded an older known commit.
- `get_epsilon` now raises `ValueError` for an unknown `exp_type`.
  Before this fix, any value other than `'exp_minus_i_omega_t'` gave the `'exp_plus_i_omega_t'` result.
  A typo thus changed the sign of Im(ε) without an error.
- `RefractiveIndexMaterial` now raises `TypeError` for an unknown keyword argument.
  Before this fix, it ignored the argument. For example, `dbpath=...` used the default database without an error.
  Releases before 1.0.0 also raised `TypeError`.

## [1.0.5] - 2026-10-26

### Fixed

- `main/MgO/Synowicki` and `main/ZrO2/Synowicki` now return MgO and ZrO2 data.
  Before this fix, both pages returned the As2S3 data.
  The cause is an error in the upstream `catalog-nk.yml`.
  ([#10](https://github.com/toftul/refractiveindex/issues/10))

### Added

- A table of temporary fixes for errors in the upstream catalog (`_CATALOG_FIXES`).
  The library applies a fix only while the upstream catalog has the error.
- A test that fails when the upstream catalog no longer has the error.
  This failure tells the maintainers to remove the fix.
- A weekly CI run against the latest upstream database.

## [1.0.4] - 2026-07-14

### Fixed

- `get_epsilon` now works for materials without extinction coefficient data.
  It uses k = 0 and returns n². Before this fix, it raised `NoExtinctionCoefficient`.

### Changed

- The PyPI workflow starts when you publish a GitHub release, not when you push a tag.
  ([#9](https://github.com/toftul/refractiveindex/pull/9))

## [1.0.3] - 2026-06-19

### Added

- The library gets the latest commit of the upstream database from the GitHub API and downloads it.
  If the API request fails, the library uses a known commit.
  ([#8](https://github.com/toftul/refractiveindex/pull/8))
- The `.version` file in the database directory stores the commit of the downloaded database.
  With `update_database=True`, the library compares this commit with the latest one and updates the database.
  ([#8](https://github.com/toftul/refractiveindex/pull/8))
- A GitHub workflow that publishes the package to PyPI.
- PyPI metadata: license, homepage, repository, keywords, and classifiers.
- The README shows the conversion formula for each unit.

### Changed

- The library downloads the database from the upstream `main` branch.
  Upstream renamed `master` to `main` on 2026-01-07.
  ([#8](https://github.com/toftul/refractiveindex/pull/8))

## [1.0.2] - 2026-02-24

### Changed

- This release changes only the version number. The code is the same as in 1.0.1.

## [1.0.1] - 2026-02-24

### Added

- `get_wl_range()` returns the wavelength range of the data as `(min, max)`.
  ([#7](https://github.com/toftul/refractiveindex/pull/7))
- A `unit=` argument for `get_refractive_index`, `get_extinction_coefficient`, `get_epsilon`, and `get_wl_range`.
  The supported units are `m`, `mm`, `um`, `nm`, `A`, `cm-1`, `THz`, and `eV`. The default is `nm`.

## [1.0.0] - 2026-02-17

### Changed

- The library is a full rewrite. It is one module with one public class, `RefractiveIndexMaterial`,
  and one public exception, `NoExtinctionCoefficient`.
- The library supports all tabulated data types (`n`, `k`, `nk`) and all 9 dispersion formulas.
- The library uses the new upstream database layout, with `data/` in place of `data-nk/`.
- A wavelength outside the data range no longer raises an exception.
  Tabulated data returns `nan`. Formula data extrapolates.
- The `db_path` argument sets the database location. The old `databasePath` argument still works.
- The test suite has 20 tests.

### Removed

- The classes `RefractiveIndex`, `Material`, and the other internal classes.
- The public `originalData` attribute.

## [0.0.7] - 2025-05-29

### Added

- The `ssl_certificate_location` argument. Use it to give a `.pem` certificate behind a proxy.
  Give an empty string to skip SSL verification.
  ([#5](https://github.com/toftul/refractiveindex/pull/5))
- The `update_database` argument. Use it to download the database again.
  ([#5](https://github.com/toftul/refractiveindex/pull/5))
- `RefractiveIndexMaterial` sends its keyword arguments to `RefractiveIndex`.
  ([#5](https://github.com/toftul/refractiveindex/pull/5))

### Fixed

- `update_database` now removes the old database before it downloads the new one.
  ([#5](https://github.com/toftul/refractiveindex/pull/5))

## [0.0.6] - 2025-01-14

### Added

- The library downloads a fixed commit of the upstream database.
  ([#4](https://github.com/toftul/refractiveindex/pull/4))
- Tests for basic use.
  ([#4](https://github.com/toftul/refractiveindex/pull/4))

## [0.0.5] - 2024-06-20

### Added

- The `originalData` attribute of `Material` holds the data from the database file.
  The wavelengths in it are NumPy arrays.
  ([#2](https://github.com/toftul/refractiveindex/pull/2))

## [0.0.4] - 2024-02-12

### Fixed

- The library uses the new upstream file names `catalog-nk.yml` and `data-nk/`.
  ([#1](https://github.com/toftul/refractiveindex/pull/1))

## [0.0.3] - 2023-08-29

### Changed

- The package requires PyYAML 6.

## [0.0.2] - 2023-04-17

### Fixed

- The argument name `wavelenght_nm` is now `wavelength_nm`.

### Added

- The README tells you to replace spaces in page names with underscores.

## [0.0.1] - 2023-04-11

- First release.

[Unreleased]: https://github.com/toftul/refractiveindex/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/toftul/refractiveindex/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/toftul/refractiveindex/compare/v1.0.5...v1.1.0
[1.0.5]: https://github.com/toftul/refractiveindex/compare/v1.0.4...v1.0.5
[1.0.4]: https://github.com/toftul/refractiveindex/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/toftul/refractiveindex/releases/tag/v1.0.3
[1.0.2]: https://pypi.org/project/refractiveindex/1.0.2/
[1.0.1]: https://pypi.org/project/refractiveindex/1.0.1/
[1.0.0]: https://pypi.org/project/refractiveindex/1.0.0/
[0.0.7]: https://pypi.org/project/refractiveindex/0.0.7/
[0.0.6]: https://pypi.org/project/refractiveindex/0.0.6/
[0.0.5]: https://pypi.org/project/refractiveindex/0.0.5/
[0.0.4]: https://pypi.org/project/refractiveindex/0.0.4/
[0.0.3]: https://pypi.org/project/refractiveindex/0.0.3/
[0.0.2]: https://pypi.org/project/refractiveindex/0.0.2/
[0.0.1]: https://pypi.org/project/refractiveindex/0.0.1/
