from itertools import islice
import numpy as np
import pandas as pd
from pyDOE import lhs
import pysb
from pysb.simulator import ScipyOdeSimulator

from caspase_model.shared import observe_biosensors, intrinsic_stimuli, observe_caspases
from . import anisotropy_functions as af
from .sensors import Sensors


class Model(object):
    """Model class contains all the parameters to be used in a simulation of
    an experiment. It knows which model to use for the apoptotic network, the
    stimuli should be specified as intrinsic or extrinsic will be used as
    default. A biosensor DataFrame should be given in order to estimate
    anisotropy curves. Time vector can be changed if necessary.
    """
    def __init__(self, model, biosensors=True, **kwargs):
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
        self.duplicate_stimuli = False
        self.simulator = None
        self.sim_batch_size = 1
        self.param_set = []
        if biosensors:
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

    def get_simultion_results(self, params=None):
        """Simulates model with given parameters and returns monomer curves."""
        if self.simulator is None:
            self.simulator = ScipyOdeSimulator(self.model, tspan=self.time)
        simres = self.simulator.run(param_values=params, num_processors=self.sim_batch_size)
        return simres.all

    def get_anisotropy_curves(self, params=None):
        """Simulates model and calculates the anisotropy curves from the
        monomer curves and parameters of biosensors."""
        simres = self.get_simultion_results(params=params)
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

    def add_parameter_set(self, param_dict):
        """Add a set of parameters to be used in simulation.

        Parameters
        ----------
        param_dict: dictionary
            A dictionary containing parameter name as keys and values as
            parameter values to be used.
        """
        self.param_set.append(param_dict)

    def set_duplicate_stimuli(self):
        """Duplicate parameter sweep but with both stimuli separately.
        WARNING: must have L_0 and IntrinsicStimuli_0 initials."""
        self.duplicate_stimuli = True

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
        if params is None:
            params = self._build_params_matrix(n_exps)

        for rows in grouper(range(len(params)), self.sim_batch_size):
            param_set_id = None
            this_params = params.loc[rows]

            this_param_dict = {}
            for col in this_params.columns:
                if col == 'param_set_id':
                    param_set_id = this_params[col].values[0]
                    continue

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

            if param_set_id is not None:
                this_res['param_set_id'] = param_set_id

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
            DataFrame containing the anisotropy curves for the simulations and
            columns with the values of the parameters changed.
        """
        params = self._build_params_matrix(n_exps)

        for rows in grouper(range(len(params)), self.sim_batch_size):
            param_set_id = None
            this_params = params.loc[rows]

            this_param_dict = {}
            for col in this_params.columns:
                if col == 'param_set_id':
                    param_set_id = this_params[col].values[0]
                    continue

                if self.sim_batch_size == 1:
                    this_param_dict[col] = this_params[col].values[0]
                else:
                    this_param_dict[col] = this_params[col].values
            if this_params.empty:
                this_param_dict = None

            try:
                observe_caspases()
            except pysb.ComponentDuplicateNameError:
                pass

            sim_res = self.get_simultion_results(params=this_param_dict)

            if this_param_dict is None:
                this_param_dict = {}
            for col in sim_res.dtype.names:
                if col.startswith('__'):
                    continue
                this_param_dict.update({col: sim_res[col]})

            this_res = pd.DataFrame([this_param_dict])

            if param_set_id is not None:
                this_res['param_set_id'] = param_set_id

            yield this_res

    def _build_params_matrix(self, n_exps):
        if not n_exps and len(self.paramsweep) == 0:
            n_exps = 1
        if not n_exps:
            n_exps = 1000
        params = self._generate_parameter_sweep_matrix(n_exps)
        params = self._add_parameter_set(params)
        if self.duplicate_stimuli:
            try:
                self.model.parameters['IntrinsicStimuli_0']
            except KeyError:
                if params.empty:
                    params = pd.DataFrame([{'L_0': self.model.parameters['L_0'].value}])
                try:
                    params['L_0']
                except KeyError:
                    params['L_0'] = self.model.parameters['L_0'].value
                intrinsic_stimuli(self.model)
                self.model.parameters['IntrinsicStimuli_0'].value = 0
            params = self._duplicate_stimuli_in_matrix(params)
        return params

    def _duplicate_stimuli_in_matrix(self, params):
        params['IntrinsicStimuli_0'] = 0
        params['param_set_id'] = np.arange(len(params))
        params_int = params.copy()
        params_int['L_0'] = 0
        params_int['IntrinsicStimuli_0'] = 1e2
        params['IntrinsicStimuli_0'] = 0
        params = pd.concat([params, params_int], ignore_index=True)
        return params

    def _generate_parameter_sweep_matrix(self, n_exps):
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

        return params

    def _add_parameter_set(self, params):
        cols = set(params.columns)

        for parameter_set in self.param_set:
            cols.update(parameter_set.keys())

        original_param_dict = {col: self.model.parameters[col].value for col in cols}

        if params.empty:
            params = pd.DataFrame([original_param_dict])
        else:
            params = params.append(pd.Series(original_param_dict), ignore_index=True)

        for parameter_set in self.param_set:
            this_parameter_dict = original_param_dict.copy()
            this_parameter_dict.update(parameter_set)
            params = params.append(pd.Series(this_parameter_dict), ignore_index=True)

        return params

    def simulate_experiment(self, n_exps=None, params=None):
        dfs = [this for this in self._simulate_experiment(n_exps=n_exps,
                                                          params=params)]
        return pd.concat(dfs, ignore_index=True)

    def simulate_western_blot(self, n_exps=None):
        dfs = [this for this in self._simulate_western_blot(n_exps=n_exps)]
        return pd.concat(dfs, ignore_index=True)


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


def keep_cols_from_model(df, model):
    keep = []
    parameter_list = [param.name for param in model.parameters]
    for col in df.columns:
        if col in parameter_list:
            keep.append(col)
    return df[keep]
