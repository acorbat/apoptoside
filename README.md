# apoptoside

Analysis for apoptotic cascade experiments.

Simulations use the [SimBio](https://github.com/dyscolab/simbio) models
of [caspase_model](https://github.com/acorbat/caspase_model):

```python
from caspase_model.simbio_model import ARM

from apoptoside.model import Model

model = Model(ARM)
model.load_sensors("sensors.json")
model.add_parameter_sweep("caspases.XIAP", 1e3, 1e4)
model.set_duplicate_stimuli()  # simulate extrinsic and intrinsic stimuli
df = model.simulate_experiment(n_exps=10)
```

Parameters are referred to by their name in the SimBio model,
listed in `model.initial_conditions` and `model.rates`.

## Development

This project is managed by [pixi](https://pixi.sh).
It expects a checkout of [caspase_model](https://github.com/acorbat/caspase_model)
next to it:

```sh
git clone https://github.com/acorbat/caspase_model
git clone https://github.com/acorbat/apoptoside
cd apoptoside
```

Run the tests with:

```sh
pixi run test
```

Code style is enforced through [prek](https://prek.j178.dev) hooks.
Install them with:

```sh
pixi run prek install
```

or run them on all files with:

```sh
pixi run lint
```
