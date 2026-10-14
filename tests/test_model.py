from pathlib import Path

import numpy as np
import pytest
from caspase_model.simbio_model import ARM, Albeck11e

from apoptoside import experimental_tests as et
from apoptoside.model import INTRINSIC_STIMULI, LIGAND, Model

SENSORS = Path(__file__).parent / "data" / "sensors.json"


@pytest.fixture
def model():
    m = Model(ARM)
    m.time = np.arange(0, 40_000, 100)
    return m


def test_parameters(model):
    assert model.parameters[LIGAND] == 1e3
    assert model.parameters[INTRINSIC_STIMULI] == 0
    assert model.parameters["caspases.C3_pro"] == 1e4
    assert "caspases.C3_pro" in model.initial_conditions
    assert "rec_to_bid.bid_cleavage_kf" in model.rates


def test_kwargs_set_initial_conditions():
    assert Model(ARM, L=0).parameters[LIGAND] == 0


def test_unknown_parameter(model):
    with pytest.raises(ValueError):
        model.add_parameter_sweep("unknown", 0, 1)
    with pytest.raises(ValueError):
        model.add_parameter_set({"unknown": 0})


def test_params_matrix(model):
    model.add_parameter_sweep("caspases.XIAP", 1e3, 1e4)
    model.add_parameter_set({"caspases.C3_pro": 5e3})
    params = model._build_params_matrix(n_exps=3)
    assert len(params) == 3 + 2
    assert not params.isna().any(axis=None)
    assert params["caspases.XIAP"].between(1e3, 1e4).all()
    assert (params["caspases.C3_pro"].iloc[:-1] == 1e4).all()
    assert params["caspases.C3_pro"].iloc[-1] == 5e3


def test_duplicate_stimuli(model):
    model.set_duplicate_stimuli()
    params = model._build_params_matrix(n_exps=None)
    assert params[[LIGAND, INTRINSIC_STIMULI]].values.tolist() == [[1e3, 0], [0, 1e2]]
    assert params["param_set_id"].tolist() == [0, 0]


def test_duplicate_stimuli_requires_intrinsic_stimuli():
    with pytest.raises(ValueError):
        Model(Albeck11e).set_duplicate_stimuli()


def test_simulate_western_blot(model):
    df = model.simulate_western_blot()
    assert len(df) == 1
    assert df["Cas3_active"].iloc[0][-1] > 1


def test_simulate_experiment(model):
    model.load_sensors(SENSORS)
    df = model.simulate_experiment()
    for sensor in model.sensors.sensors:
        anisotropy = df[f"{sensor.name}_anisotropy"].iloc[0]
        assert anisotropy[0] == pytest.approx(sensor.anisotropy_dimer)
        assert anisotropy[-1] == pytest.approx(sensor.anisotropy_monomer)
        assert df[f"{sensor.enzyme}_cleaved"].iloc[0]


def test_caspase_activation():
    assert et.test_caspase_activation(ARM)


def test_caspase8_knockout():
    assert et.test_caspase8_knockout(ARM)
