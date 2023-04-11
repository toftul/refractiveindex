# Easy Python interface to RefractiveIndex database

Database files parsing was made with a modified version of `refractiveIndex.py` from [PyTMM project](https://github.com/kitchenknif/PyTMM) by [Pavel Dmitriev](https://github.com/kitchenknif).

## Installation

```
pip install refractiveindex
```

## Usage


```python
from refractiveindex import RefractiveIndexMaterial

SiO = RefractiveIndexMaterial(shelf='main', book='SiO', page='Hass')

wavelength_nm = 600  # [nm]

SiO.get_epsilon(wavelength_nm)
# returns
# (3.8633404437869827+0.003931076923076923j)
```

Note: here the time dependence is assumed to be $\mathrm{e}^{-\mathrm{i} \omega t}$, so $\operatorname{Im}\varepsilon > 0$ is responsible for the losses.
