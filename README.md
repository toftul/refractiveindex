# Easy Python interface to RefractiveIndex database

![Tests](https://github.com/toftul/refractiveindex/actions/workflows/build-ci.yml/badge.svg)

The original database<br>
https://github.com/polyanskiy/refractiveindex.info-database

## Installation

```
pip install refractiveindex
```

## Usage

```python
import numpy as np
from refractiveindex import RefractiveIndexMaterial

# Tabulated nk material
SiO = RefractiveIndexMaterial(shelf='main', book='SiO', page='Hass')

wavelength_nm = 600  # [nm]

SiO.get_refractive_index(wavelength_nm)
# 1.96553846

SiO.get_extinction_coefficient(wavelength_nm)
# 0.001

SiO.get_epsilon(wavelength_nm)
# (3.8633404437869827+0.003931076923076923j)

# Formula-based material
bk7 = RefractiveIndexMaterial(shelf='specs', book='SCHOTT-optical', page='N-BK7')
bk7.get_refractive_index(589.3)
# 1.5168...

# Array input
wavelengths = np.array([400, 500, 600, 700])
SiO.get_refractive_index(wavelengths)
# array([2.15092857, 1.999, 1.96553846, 1.942])
```

Notes:
- Here the time dependence is assumed to be $\mathrm{e}^{-\mathrm{i} \omega t}$, so $\mathrm{Im}(\varepsilon) > 0$ is responsible for the losses.
- If there is a space in the name, write underscore instead, i.e. not `page='Rodriguez-de Marcos'` but `page='Rodriguez-de_Marcos'`.
- All 9 [dispersion formula types](https://github.com/polyanskiy/refractiveindex.info-database/blob/master/database/doc/Dispersion%20formulas.pdf) used by the refractiveindex.info database are supported: Sellmeier (1), Sellmeier-2 (2), Polynomial (3), RefractiveIndex.INFO (4), Cauchy (5), Gases (6), Herzberger (7), Retro (8), and Exotic (9).
- The database is automatically downloaded on first use to `~/.refractiveindex.info-database`.

## How to get material page names

You can find the proper "page" name by hovering your cursor on the link in the Data section

![How to get page name](./fig/link.png)

Or you can look up folders in this repository<br>
https://github.com/polyanskiy/refractiveindex.info-database

## Similar projects

- [PyTMM](https://github.com/kitchenknif/PyTMM) by [Pavel Dmitriev](https://github.com/kitchenknif) — the original database parsing code this project was based on
- [RefractiveIndex.jl](https://github.com/stillyslalom/RefractiveIndex.jl) — Julia interface to refractiveindex.info database

## Possible problems

If the upstream database has changed its structure, upgrade the package
```shell
pip install --upgrade refractiveindex
```
and remove the database folder at `~/.refractiveindex.info-database`.
