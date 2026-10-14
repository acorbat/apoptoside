import numpy as np
import pandas as pd
from caspase_model.simbio_model.observables import (
    biosensor_observables,
    caspase_observables,
)
from poincare.solvers import LSODA
from scipy.stats import qmc
from simbio import Simulator, System

from . import anisotropy_functions as af
from .sensors import Sensors

# Names of the stimuli species in the SimBio models
LIGAND = "L"
INTRINSIC_STIMULI = "IntrinsicStimuli"


def get_parameters(model: System | type[System]) -> dict[str, float]:
    """Initial conditions and rates of a SimBio model, by name."""
    sim = Simulator(model)
    problem = sim.create_problem()
    names = map(str, [*sim.compiled.variables, *sim.compiled.parameters])
    values = [*problem.y, *problem.p]
    return dict(zip(names, map(float, values)))


class Model:
    """Model class contains all the parameters to be used in a simulation of
    an experiment. It knows which SimBio model to use for the apoptotic network
    (see caspase_model.simbio_model). Keyword arguments set initial conditions
    or rates of the model, for instance, to choose the stimuli. A biosensor
    DataFrame should be given in order to estimate anisotropy curves. Time
    vector can be changed if necessary.

    Parameters are referred to by their name in the SimBio model,
    such as "caspases.C3_pro" or "rec_to_bid.bid_cleavage_kf".
    """

    def __init__(self, model: System | type[System], biosensors=True, **kwargs):
        if kwargs:
            model = model(**kwargs)
        self.model = model

        observables = caspase_observables(model)
        if biosensors:
            observables |= biosensor_observables(model)
        self.simulator = (
            Simulator(model)
            .with_solver(LSODA(atol=1e-6, rtol=1e-6))
            .with_transform(observables)
        )
        self._components = {
            str(c): c
            for c in [
                *self.simulator.compiled.variables,
                *self.simulator.compiled.parameters,
            ]
        }

        self.parameters = get_parameters(model)
        self.initial_conditions = list(map(str, self.simulator.compiled.variables))
        self.rates = list(map(str, self.simulator.compiled.parameters))
        self.time = np.arange(0, 40000, 1)
        self.sensors = None
        self.paramsweep = pd.DataFrame(
            columns=["parameter", "min_value", "max_value", "correlation"]
        )
        self.duplicate_stimuli = False
        self.param_set = []

    def load_sensors(self, path):
        """Load sensors from file."""
        self.sensors = Sensors.parse_file(path)

    def get_initial_conditions(self):
        """Names of the initial conditions."""
        return self.initial_conditions

    def get_rates(self):
        """Names of the rates."""
        return self.rates

    def get_simultion_results(self, params=None) -> dict[str, np.ndarray]:
        """Simulates model with given parameters and returns observable curves."""
        sim = self.simulator
        if params:
            sim = sim.with_values(
                {self._components[name]: value for name, value in params.items()}
            )
        result = sim.solve(save_at=self.time)
        return {name: result[name].values for name in result.data_vars}

    def get_anisotropy_curves(self, params=None):
        """Simulates model and calculates the anisotropy curves from the
        monomer curves and parameters of biosensors."""
        simres = self.get_simultion_results(params=params)
        anis_curves = {}
        anis_states = {}
        for sensor in self.sensors.sensors:
            fluo = sensor.name
            cas = sensor.enzyme
            b = 1 + sensor.delta_b
            monomer_curve = simres["s" + cas + "_monomer"]
            anis_curves[fluo], anis_states[fluo] = estimate_anisotropy(
                monomer_curve, sensor.anisotropy_monomer, sensor.anisotropy_dimer, b
            )

        return anis_curves, anis_states

    def add_parameter_sweep(
        self, parameter_name, min_value, max_value, correlation="uncorrelated"
    ):
        """Adds a parameter to the list of parameter to sweep.

        Parameters
        ----------
        parameter_name : basestring
            Name of the parameter in the model to sweep.
        min_value : float
            minimal value of the parameter.
        max_value : float
            Maximum value of the parameter to sweep.
        correlation : tuple, (Default='uncorrelated')
            Tuple of parameter to which it's correlated and it's value."""
        if parameter_name not in self.parameters:
            raise ValueError(f"Parameter {parameter_name} not in model.")
        if parameter_name in self.paramsweep.parameter.values:
            raise ValueError(f"Parameter {parameter_name} already in list.")

        new_param_line = {
            "parameter": [parameter_name],
            "min_value": [min_value],
            "max_value": [max_value],
            "correlation": [correlation],
        }
        new_param_line = pd.DataFrame(new_param_line)
        self.paramsweep = pd.concat(
            [self.paramsweep, new_param_line], ignore_index=True
        )

    def add_parameter_set(self, param_dict):
        """Add a set of parameters to be used in simulation.

        Parameters
        ----------
        param_dict: dictionary
            A dictionary containing parameter name as keys and values as
            parameter values to be used.
        """
        unknown = param_dict.keys() - self.parameters.keys()
        if unknown:
            raise ValueError(f"Parameters {unknown} not in model.")
        self.param_set.append(param_dict)

    def set_duplicate_stimuli(self):
        """Duplicate parameter sweep but with both stimuli separately."""
        for name in (LIGAND, INTRINSIC_STIMULI):
            if name not in self.parameters:
                raise ValueError(f"Model must have a {name} species.")
        self.duplicate_stimuli = True

    def _iter_params(self, n_exps=None, params=None):
        """Yield the parameter set id and the parameters of each simulation."""
        if params is None:
            params = self._build_params_matrix(n_exps)

        for _, row in params.iterrows():
            row = row.to_dict()
            param_set_id = row.pop("param_set_id", None)
            yield param_set_id, row

    def _simulate_experiment(self, n_exps=None, params=None):
        """Simulate experiments varying parameters with the values given at
        paramsweep.

        Parameters
        ----------
        n_exps : integer
            Number of experiments to simulate. Default is 1 if there is no
            paramsweep values, else it's 1000.
        params : numpy.array
            Matrix with parameters to be used in simulation.

        Returns
        -------
        simulated_dataframe : pandas.DataFrame
            DataFrame containing the anisotropy curves for the simulations and
            columns with the values of the parameters changed.
        """
        for param_set_id, this_param_dict in self._iter_params(n_exps, params):
            anis_curves, anis_state = self.get_anisotropy_curves(params=this_param_dict)
            for sensor in self.sensors.sensors:
                fluo = sensor.name
                cas = sensor.enzyme

                anis_curves[fluo + "_anisotropy"] = anis_curves.pop(fluo)
                anis_state[cas + "_cleaved"] = anis_state.pop(fluo)

            this_param_dict.update(anis_curves)
            this_param_dict.update(anis_state)
            this_res = pd.DataFrame([this_param_dict])

            if param_set_id is not None:
                this_res["param_set_id"] = param_set_id

            yield this_res

    def _simulate_western_blot(self, n_exps=None):
        """Simulate western blot experiment varying parameters with the values
        given at paramset.

        Parameters
        ----------
        n_exps : integer
            Number of experiments to simulate. Default is 1 if there is no
            paramsweep values, else it's 1000.

        Returns
        -------
        simulated_dataframe : pandas.DataFrame
            DataFrame containing the observable curves for the simulations and
            columns with the values of the parameters changed.
        """
        for param_set_id, this_param_dict in self._iter_params(n_exps):
            sim_res = self.get_simultion_results(params=this_param_dict)
            this_param_dict.update(sim_res)
            this_res = pd.DataFrame([this_param_dict])

            if param_set_id is not None:
                this_res["param_set_id"] = param_set_id

            yield this_res

    def _build_params_matrix(self, n_exps):
        if not n_exps and len(self.paramsweep) == 0:
            n_exps = 1
        if not n_exps:
            n_exps = 1000
        params = self._generate_parameter_sweep_matrix(n_exps)
        params = self._add_parameter_set(params)
        if self.duplicate_stimuli:
            if params.empty:
                params = pd.DataFrame([{LIGAND: self.parameters[LIGAND]}])
            if LIGAND not in params:
                params[LIGAND] = self.parameters[LIGAND]
            params = self._duplicate_stimuli_in_matrix(params)
        return params

    def _duplicate_stimuli_in_matrix(self, params):
        params[INTRINSIC_STIMULI] = 0
        params["param_set_id"] = np.arange(len(params))
        params_int = params.copy()
        params_int[LIGAND] = 0
        params_int[INTRINSIC_STIMULI] = 1e2
        params = pd.concat([params, params_int], ignore_index=True)
        return params

    def _generate_parameter_sweep_matrix(self, n_exps):
        not_correlated_params = self.paramsweep.query('correlation == "uncorrelated"')
        matrix = qmc.LatinHypercube(d=len(not_correlated_params)).random(n_exps)
        mins = not_correlated_params.min_value.values
        maxs = not_correlated_params.max_value.values
        matrix = mins + (maxs - mins) * matrix
        params = pd.DataFrame(matrix, columns=not_correlated_params.parameter.values)
        for _, correlated_param in self.paramsweep.query(
            'correlation != "uncorrelated"'
        ).iterrows():
            correlated_to, correlation_val = correlated_param.correlation
            param = correlated_param.parameter
            params[param] = params[correlated_to].values * correlation_val

        return params

    def _add_parameter_set(self, params):
        cols = set(params.columns)

        for parameter_set in self.param_set:
            cols.update(parameter_set.keys())

        original_param_dict = {col: self.parameters[col] for col in cols}

        if params.empty:
            params = pd.DataFrame([original_param_dict])
        else:
            params = pd.concat([params, pd.DataFrame([original_param_dict])])

        for parameter_set in self.param_set:
            this_parameter_dict = original_param_dict | parameter_set
            params = pd.concat([params, pd.DataFrame([this_parameter_dict])])

        # Parameters not swept keep their value in the model
        return params.fillna(original_param_dict).reset_index(drop=True)

    def simulate_experiment(self, n_exps=None, params=None):
        dfs = list(self._simulate_experiment(n_exps=n_exps, params=params))
        return pd.concat(dfs, ignore_index=True)

    def simulate_western_blot(self, n_exps=None):
        dfs = list(self._simulate_western_blot(n_exps=n_exps))
        return pd.concat(dfs, ignore_index=True)


def estimate_anisotropy(monomer_curve, anisotropy_monomer, anisotropy_dimer, b):
    if monomer_curve.max() == 0:
        anisotropy = np.array([anisotropy_dimer] * len(monomer_curve))
    else:
        m_curve = monomer_curve / monomer_curve.max()

        anisotropy = af.anisotropy_from_monomer(
            m_curve, anisotropy_monomer, anisotropy_dimer, b
        )

    # Check whether all dimer has transformed into monomer
    anis_state = True
    if np.abs(anisotropy[-1] - anisotropy_monomer) > 1e-8:
        print("Not all dimer was cleaved!")
        anis_state = False

    return anisotropy, anis_state


def keep_cols_from_model(df, model: Model):
    """Keep the columns of df that are parameters of the model."""
    return df[[col for col in df.columns if col in model.parameters]]
