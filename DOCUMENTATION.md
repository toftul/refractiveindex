# Documentation

Fine points that the [README](README.md) leaves out.

## Units

`get_refractive_index`, `get_extinction_coefficient`, `get_epsilon` and `get_wl_range` take `unit=` (default `'nm'`):

| Unit | `unit=` | Conversion from µm |
|---|---|---|
| nanometers | `'nm'` | $\lambda{[\text{nm}]} = \lambda{[\mu\text{m}]} \cdot 10^{3}$ |
| micrometers | `'um'` | $\lambda{[\mu\text{m}]}$ |
| millimeters | `'mm'` | $\lambda{[\text{mm}]} = \lambda{[\mu\text{m}]} \cdot 10^{-3}$ |
| meters | `'m'` | $\lambda{[\text{m}]} = \lambda{[\mu\text{m}]} \cdot 10^{-6}$ |
| Angstroms | `'A'` | $\lambda{[\text{Å}]} = \lambda{[\mu\text{m}]} \cdot 10^{4}$ |
| wavenumbers | `'cm-1'` | $\tilde{\nu}{[\text{cm}^{-1}]} = 10^4 / \lambda{[\mu\text{m}]}$ |
| terahertz | `'THz'` | $\nu{[\text{THz}]} = \frac{c[\text{m}/\text{s}] \cdot 10^{-6}}{\lambda{[\mu\text{m}]}}$ |
| electron volts | `'eV'` | $E [\text{eV}] = \frac{h[\text{J}\cdot \text{s}]\cdot c[\text{m}/\text{s}] \cdot 10^{6}}{e[\text{C}] \cdot \lambda [\mu \text{m}]}$ |

Here $c$ is the speed of light, $h$ is the Planck constant, $e$ is the elementary charge. For the reciprocal units (`'cm-1'`, `'THz'`, `'eV'`), `get_wl_range` still returns `(min, max)`.

## Wavelength range

The n and k data of a material can cover different ranges. `get_wl_range(quantity=...)` selects one:

| `quantity=` | Range |
|---|---|
| `'epsilon'` (default) or `'nk'` | Overlap of n and k, where `get_epsilon` is valid. Without k data, the n range. |
| `'n'` | n data |
| `'k'` | k data |

```python
laf3 = ri.RefractiveIndexMaterial(shelf='main', book='LaF3', page='Amotchkina')
laf3.get_wl_range(quantity='n')  # (400.0, 12000.0)
laf3.get_wl_range(quantity='k')  # (9493.08, 11973.9)
laf3.get_wl_range()              # (9493.08, 11973.9), the overlap
```

If the n and k ranges do not overlap, the default `quantity` raises `ValueError`.

## Outside the range

No exception is raised, but the result is not reliable:

| Data type | Result outside the range |
|---|---|
| Tabulated (`n`, `k`, `nk`) | `nan` |
| Formula (1–9) | Extrapolation, which can be unphysical, or `nan` with a `RuntimeWarning` if the formula takes the square root of a negative number |

Check the range first if you are not sure:

```python
lo, hi = mat.get_wl_range()
if not lo <= wavelength <= hi:
    raise ValueError(f"{wavelength} nm is outside [{lo}, {hi}] nm")
```

## Missing k data

If a material has no k data, `get_extinction_coefficient` and `get_wl_range(quantity='k')` raise `ri.NoExtinctionCoefficient`. `get_epsilon` then takes k = 0.

## Time convention

The default time dependence is $\mathrm{e}^{-\mathrm{i} \omega t}$, so $\varepsilon = (n + \mathrm{i}k)^2$ and $\mathrm{Im}(\varepsilon) > 0$ means loss. `exp_type="exp(+iwt)"` gives $(n - \mathrm{i}k)^2$. You can also write `"exp(iwt)"`, `j` for `i`, and `ω` for `w`, for example `"exp(-jωt)"`.

## Material names

- Spaces in names become underscores: `page='Rodriguez-de_Marcos'`, not `page='Rodriguez-de Marcos'`.
- The address-bar URL `https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson` gives `shelf='main'`, `book='SiO2'`, `page='Malitson'`. `from_url` reads these three names and nothing else; other URLs, such as links to data files, raise `ValueError`.
- You can also find the names in the folders of the [upstream repository](https://github.com/polyanskiy/refractiveindex.info-database).

## Supported data

Tabulated `n`, `k` and `nk` data, and all 9 [dispersion formulas](https://github.com/polyanskiy/refractiveindex.info-database/blob/master/database/doc/Dispersion%20formulas.pdf): Sellmeier (1), Sellmeier-2 (2), Polynomial (3), RefractiveIndex.INFO (4), Cauchy (5), Gases (6), Herzberger (7), Retro (8), and Exotic (9).

## Database version

The data comes from the database release in the package, not from the website, so a version of the package always gives the same data. A new upstream release comes with a new package version (`pip install --upgrade refractiveindex`).

To use newer data, or a material that the website has and the included release does not, clone the upstream repository and give its `database/` folder as `db_path` (also accepted by `from_url`):

```shell
git clone https://github.com/polyanskiy/refractiveindex.info-database
```

```python
SiO2 = ri.RefractiveIndexMaterial(shelf='main', book='SiO2', page='Malitson',
                                  db_path='refractiveindex.info-database/database')
```

For the version statement in a paper, write for example "refractiveindex.info database v2026-05-24, used through the refractiveindex Python package 1.2.0". The database licence is in `refractiveindex/database/LICENSE`.

## Upgrading from versions before 1.2.0

- The old versions downloaded the database to `~/.refractiveindex.info-database`. Version 1.2.0 and later do not use this folder, so you can delete it.
- `auto_download`, `update_database` and `ssl_certificate_location` have no effect and give a `DeprecationWarning`. `databasePath` is a deprecated alias for `db_path`. All four will be removed in 2.0.0.
