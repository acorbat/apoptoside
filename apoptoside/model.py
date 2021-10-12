from itertools import islice
import numpy as np
import pandas as pd
from pyDOE import lhs
import pysb
from pysb.simulator import ScipyOdeSimulator

from caspase_model.shared import observe_biosensors
from . import anisotropy_functions as af
from .sensors import Sensors


class Model(object):
    """Model class contains all the parameters to be used in a simulation of
    an experiment. It knows which model to use for the apoptotic network, the
    stimuli should be specified as intrinsic or extrinsic will be used as
    default. A biosensor DataFrame should be given in order to estimate
    anisotropy curves. Time vector can be changed if necessary.
    """
    def __init__(self, model, **kwargs):
        try:
            self.model = model(**kwargs)
        except TypeError:
            self.model = pysb.Model(base=model)
        self.parameters = self.model.parameters
        self.initial_conditions = self.get_initial_conditions()
        self.rates = self.get_rates()
        self.time = np.arange(0, 40000, 1)
        self.sensors = None
        self.paramsweep = pd.DataFrame(columns=['parameter', 'min_value',
                                       'max_value', 'correlation'])
        self.simulator = None
        self.sim_batch_size = 1
        try:
            observe_biosensors()
        except pysb.ComponentDuplicateNameError:
            pass

    def load_sensors(self, path):
        """Load sensors from file."""
        self.sensors = Sensors.parse_file(path)

    def get_initial_conditions(self):
        """Generates a list of initial conditions (actually parameters in model
        ending with _0)."""
        initial_conditions = []
        for param in self.parameters:
            if param.name.endswith('_0'):
                initial_conditions.append(param.name)
        return initial_conditions

    def get_rates(self):
        """Generates a list of rates (actually parameters in model ending with
        _kf, _kr or kc)."""
        rates = []
        for param in self.parameters:
            if param.name.endswith('_kf') or param.name.endswith('_kr') or \
                    param.name.endswith('_kc'):
                rates.append(param.name)
        return rates

    def get_monomer_curves(self, params=None):
        """Simulates model with given parameters and returns monomer curves."""
        if self.simulator is None:
            self.simulator = ScipyOdeSimulator(self.model, tspan=self.time)
        simres = self.simulator.run(param_values=params, num_processors=self.sim_batch_size)
        return simres.all

    def get_anisotropy_curves(self, params=None):
        """Simulates model and calculates the anisotropy curves from the
        monomer curves and parameters of biosensors."""
        simres = self.get_monomer_curves(params=params)
        anis_curves = {}
        anis_states = {}
        for sensor in self.sensors.sensors:
            fluo = sensor.name
            cas = sensor.enzyme
            anisotropy_monomer = sensor.anisotropy_monomer
            anisotropy_dimer = sensor.anisotropy_dimer
            b = 1 + sensor.delta_b

            try:
                dimer_curve = simres['s' + cas + '_dimer']
                monomer_curve = simres['s' + cas + '_monomer']

                anis_curves[fluo], anis_states[fluo] = estimate_anisotropy(monomer_curve,
                                                                           anisotropy_monomer,
                                                                           anisotropy_dimer,
                                                                           b)
            except TypeError:
                anis_curves[fluo] = []
                anis_states[fluo] = []
                for this_simres in simres:
                    dimer_curve = this_simres['s' + cas + '_dimer']
                    monomer_curve = this_simres['s' + cas + '_monomer']

                    anis_curve, anis_state = estimate_anisotropy(monomer_curve,
                                                                 anisotropy_monomer,
                                                                 anisotropy_dimer,
                                                                 b)

                    anis_curves[fluo].append(anis_curve)
                    anis_states[fluo].append(anis_state)

        return anis_curves, anis_states

    def add_parameter_sweep(self, parameter_name, min_value, max_value,
                            correlation='uncorrelated'):
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
        if parameter_name not in self.initial_conditions and \
                parameter_name not in self.rates:
            raise ValueError('Parameter %s not in model.' % parameter_name)
        if parameter_name in self.paramsweep.parameter.values:
            raise ValueError('Parameter %s already in list.' % parameter_name)

        new_param_line = {'parameter': [parameter_name],
                          'min_value': [min_value],
                          'max_value': [max_value],
                          'correlation': [correlation]}
        new_param_line = pd.DataFrame(new_param_line)
        self.paramsweep = self.paramsweep.append(new_param_line,
                                                 ignore_index=True)

    def _simulate_experiment(self, n_exps=None):
        """Simulate experiments varying parameters with the values given at
        paramsweep.

        Parameters
        ----------
        n_exps : integer
            Number of experiments to simulate. Default is 1 if there is no
            paramsweep values, else it's 1000.

        Returns
        -------
        simulated_dataframe : pandas.DataFrame
            DataFrame containing the anisotropy curves for the simulations and
            columns with the values of the parameters changed.
        """
        if not n_exps and len(self.paramsweep) == 0:
            n_exps = 1
        if not n_exps:
            n_exps = 1000

        not_correlated_params = self.paramsweep.query('correlation == "uncorrelated"')
        matrix = lhs(len(not_correlated_params), n_exps)

        mins = not_correlated_params.min_value.values
        maxs = not_correlated_params.max_value.values

        matrix = mins + (maxs - mins) * matrix

        params = pd.DataFrame(matrix,
                              columns=not_correlated_params.parameter.values)
        for row, correlated_param in self.paramsweep.query('correlation != "uncorrelated"').iterrows():
            correlated_to, correlation_val = correlated_param.correlation
            param = correlated_param.parameter
            params[param] = params[correlated_to].values * correlation_val

        for rows in grouper(range(len(params)), self.sim_batch_size):
            this_params = params.loc[rows]

            this_param_dict = {}
            for col in this_params.columns:
                if self.sim_batch_size == 1:
                    this_param_dict[col] = this_params[col].values[0]
                else:
                    this_param_dict[col] = this_params[col].values
            if this_params.empty:
                this_param_dict = None

            anis_curves, anis_state = self.get_anisotropy_curves(params=this_param_dict)
            for sensor in self.sensors.sensors:
                fluo = sensor.name
                cas = sensor.enzyme

                anis_curves[fluo + '_anisotropy'] = anis_curves.pop(fluo)
                anis_state[cas + '_cleaved'] = anis_state.pop(fluo)

            if this_param_dict is None:
                this_param_dict = {}
            this_param_dict.update(anis_curves)
            this_param_dict.update(anis_state)
            this_res = pd.DataFrame([this_param_dict])
            yield this_res

    def simulate_experiment(self, n_exps=None):
        dfs = [this for this in self._simulate_experiment(n_exps=n_exps)]
        return pd.concat(dfs)


def grouper(iterable, n):
    iterable = iter(iterable)
    while True:
        tup = list(islice(iterable, 0, n))
        if tup:
            yield tup
        else:
            break


def estimate_anisotropy(monomer_curve, anisotropy_monomer, anisotropy_dimer, b):
    if monomer_curve.max() == 0:
        anisotropy = np.array([anisotropy_dimer] * len(monomer_curve))
    else:
        m_curve = monomer_curve / monomer_curve.max()

        anisotropy = af.anisotropy_from_monomer(m_curve,
                                                anisotropy_monomer,
                                                anisotropy_dimer,
                                                b)

    # Check whether all dimer has transformed into monomer
    anis_state = True
    if np.abs(anisotropy[-1] - anisotropy_monomer) > 1e-8:
        print('Not all dimer was cleaved!')
        anis_state = False

    return anisotropy, anis_state
