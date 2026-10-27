# Python interface to RefractiveIndex database

![Tests](https://github.com/toftul/refractiveindex/actions/workflows/build-ci.yml/badge.svg)

Python interface to the [refractiveindex.info database](https://github.com/polyanskiy/refractiveindex.info-database). The database (~15 MB) is bundled. Install with

```
pip install refractiveindex
```

## Usage

```python
import refractiveindex as ri

# load a material by shelf, book and page (replace spaces with underscores)
SiO = ri.RefractiveIndexMaterial(shelf='main', book='SiO', page='Hass')
# or from the address-bar URL of a refractiveindex.info page
SiO2 = ri.RefractiveIndexMaterial.from_url(
    'https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson'
)

# wavelength in nm by default
SiO.get_refractive_index(600)        # 1.9655

# unit= accepts 'nm', 'um', 'mm', 'm', 'A', 'cm-1', 'THz', 'eV'; arrays work too
SiO.get_extinction_coefficient([0.6, 0.7], unit='um')  # array([0.001, 0.00075])

# permittivity for exp(-iωt) (default) or exp(+iωt) time dependence
SiO.get_epsilon(600)                        # (3.8633+0.0039j)
SiO.get_epsilon(600, exp_type='exp(+iwt)')  # (3.8633-0.0039j)

# valid wavelength range
SiO.get_wl_range(unit='nm')          # (240.0, 14000.0)
```

## Documentation

See [DOCUMENTATION.md](https://github.com/toftul/refractiveindex/blob/main/DOCUMENTATION.md) for unit conversions, separate n and k ranges, values outside the range, and other copies of the database.

## Citation


M. N. Polyanskiy, "Refractiveindex.info database of optical constants," *Scientific Data* **11**, 94 (2024). https://doi.org/10.1038/s41597-023-02898-2



Please also state the database version for reproducibility. The database is [CC0](https://creativecommons.org/publicdomain/zero/1.0/).
```python
print(ri.DATABASE_VERSION)  
print(ri.DATABASE_COMMIT)
```

## See also
- [RefractiveIndex.info database](https://github.com/polyanskiy/refractiveindex.info-database) — upstream data.
- [PyTMM](https://github.com/kitchenknif/PyTMM) — original database parsing code.
- [RefractiveIndex.jl](https://github.com/stillyslalom/RefractiveIndex.jl) — Julia interface.
