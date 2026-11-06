# steqpy

Stochastic declustering of earthquake catalogues: a faster Python/Numba port of Jiancang
Zhuang's space-time ETAS code `etas8p`. For each event it estimates the probability that the
event is a background event rather than triggered by an earlier one. It also fits the ETAS
parameters and the background rate.

With preset `original`, the port reproduced the Fortran outputs bit for bit on the JMA example.

## Install

Requires Python 3.10 or newer.

```bash
pip install git+https://github.com/earthequationseiga/stochasticEarthquakesPython.git
```

From a local copy: `pip install .`

## Use

From Python:

```python
from steqpy import StochasticDeclustering

sd = StochasticDeclustering.from_config("configs/jma.toml", threads=8)
result = sd.fit()

result.params             # fitted ETAS parameters
result.background_prob    # P(event i is a background event), per event
lon, lat, rates = result.rate_grid()
result.write_outputs("results/jma")
```

From the command line:

```bash
python -m steqpy configs/jma.toml --out results/jma --threads 8
```

`configs/jma.toml` documents every option. The JMA example catalogue is available from
Zhuang's site below; the config expects it in `data/jma/`.

## Credit

The method and the original Fortran are the work of Jiancang Zhuang and colleagues. Original
code and data: https://bemlar.ism.ac.jp/zhuang/software.html

- Zhuang, J., Ogata, Y., & Vere-Jones, D. (2002). Stochastic declustering of space-time
  earthquake occurrences. *Journal of the American Statistical Association*, 97(458), 369-380.
  https://doi.org/10.1198/016214502760046925
- Zhuang, J., Ogata, Y., & Vere-Jones, D. (2004). Analyzing earthquake clustering features by
  using stochastic reconstruction. *Journal of Geophysical Research*, 109, B05301.
  https://doi.org/10.1029/2003JB002879
- Zhuang, J. (2006). Second-order residual analysis of spatiotemporal point processes and
  applications in model evaluation. *Journal of the Royal Statistical Society: Series B*,
  68(4), 635-653. https://doi.org/10.1111/j.1467-9868.2006.00559.x

## License

MIT, see `LICENSE`. Files in `original/` are Zhuang's and are not covered by it.
