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
- You can also find the names with `ri.search` (see below), or in the folders of the [upstream repository](https://github.com/polyanskiy/refractiveindex.info-database).

## Search

`ri.search(query)` finds pages in the catalog of the database. It prints one line per page: the index, `shelf/book/page`, and the page name with its group in the book in brackets:

```python
>>> ri.search('SiO2')
 0  main/SiO2/Malitson                       Malitson 1965: n 0.21–6.7 µm [Fused silica]
 1  main/SiO2/Nyakuchena                     Nyakuchena et al. 2023: n 1.10–1.65 µm [Fused silica]
...
16  main/Bi12SiO20/Gospodinov                Gospodinov et al. 1988: n 0.48–0.70 µm
17  main/Ag/Ciesielski                       Ciesielski et al. 2017: Ag/SiO2; n,k 0.191–20.9 µm [Experimental data: thin film]
...
```

The result is a list. The number of a line is its index, so `ri.search('SiO2')[0].material()` loads the first page. The material prints as the code that loads it, to copy into a script:

```python
>>> ri.search('SiO2')[0].material()
RefractiveIndexMaterial(shelf='main', book='SiO2', page='Malitson')
```

No id contains `/`, so `shelf/book/page` always splits into the three ids. A few book ids contain spaces, so the columns are separated by two spaces. To print fewer lines, slice the list: `ri.search('Si')[:10]` prints the first 10, numbered from 0. An item of the list is a `Match` with the fields `shelf`, `book`, `page`, `book_name`, `name` (the page name) and `divider` (the group, or `None`).

Each word of the query must appear, ignoring case, in the book or page id, the book or page name, or the group. The book names include common names, so `'silica'` and `'quartz'` find SiO2, and the page names include authors, so `'SiO2 Malitson'` finds one page. The best matches come first:

1. The book id is a query word (`'SiO2'` finds `main/SiO2`).
2. A query word is a whole word of the book name (`'silica'` in "SiO2 (Silicon dioxide, Silica, Quartz)").
3. The book id contains a query word (`'SiO2'` finds `Bi12SiO20`).
4. A query word is somewhere else, for example in a page name (`'SiO2'` finds "Ag/SiO2").

Pages with the same rank keep the order of the catalog, which is the order on the website.

Keyword arguments select pages:

| Argument | Keeps the pages |
|---|---|
| `shelf='main'` | on this shelf; also a list of shelves |
| `book='SiO2'` | of this book; also a list of books |
| `quantity='n'`, `'k'`, `'epsilon'` or `'nk'` | for which `get_wl_range(quantity=...)` gives a range: with n data, with k data, or with n data for `'epsilon'` and `'nk'` (see [Wavelength range](#wavelength-range)) |
| `wavelength=1550` or `wavelength=(400, 800)` | whose range for `quantity` (default `'epsilon'`) contains the wavelength, or all of the interval. The ends count as inside. `unit=` gives the unit, as for the other functions. |

```python
ri.search('SiO2', shelf='main', wavelength=(400, 800))  # data for all of 400–800 nm
ri.search('gold', quantity='k', wavelength=1.55, unit='um')
ri.search(shelf='glass')                                 # without a query, all pages that the filters keep
```

`shelf` and `book` are exact ids, so an unknown id raises `ValueError`. No matches give an empty list. Some data is on more than one shelf, for example `glass/fused_silica/Malitson` and `main/SiO2/Malitson`. Search shows each of these pages.

For `quantity` and `wavelength`, search loads each matching page. A page whose data cannot be loaded is left out, with a `UserWarning` that names it.

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
