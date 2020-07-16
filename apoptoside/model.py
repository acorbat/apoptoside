import numpy as np
import pandas as pd
from pyDOE import lhs
from pysb.simulator import ScipyOdeSimulator
from tqdm import tqdm

from caspase_model.shared import observe_biosensors
from . import anisotropy_functions as af


class Model(object):
    """Model class contains all the parameters to be used in a simulation of
    an experiment. It know which model to use for the apoptotic network, the
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
        self.sensors = pd.DataFrame(columns=['fluorophore',
                                             'anisotropy_monomer',
                                             'anisotropy_monomer',
                                             'b', 'color', 'caspase'])
        self.paramsweep = pd.DataFrame(columns=['parameter', 'min_value',
                                       'max_value', 'correlation'])
        observe_biosensors()  # This way observables are only defined at init

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

    def get_monomer_curves(self):
        """Simulates model with given parameters and returns monomer curves."""
        simres = ScipyOdeSimulator(self.model, tspan=self.time).run()
        return simres.all

    def get_anisotropy_curves(self):
        """Simulates model and calculates the anisotropy curves from the
        monomer curves and parameters of biosensors."""
        simres = self.get_monomer_curves()
        anis_curves = {}
        for row, this_sensor in self.sensors.iterrows():
            fluo = this_sensor.fluorophore
            anisotropy_monomer = this_sensor.anisotropy_monomer
            anisotropy_dimer = this_sensor.anisotropy_dimer
            b = 1 + this_sensor.delta_b

            dimer_curve = simres[fluo + '_dimer']
            monomer_curve = simres[fluo + '_monomer']

            # Check whether all dimer has transformed into monomer
            if (2 * dimer_curve.max() - monomer_curve.max() ) / monomer_curve.max()> 1e-4:
                print('Not all dimer was cleaved!')

            m_curve = monomer_curve / monomer_curve.max()

            anisotropy = af.anisotropy_from_monomer(m_curve,
                                                    anisotropy_monomer,
                                                    anisotropy_dimer,
                                                    b)

            anis_curves[fluo] = anisotropy

        return anis_curves

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

    def simulate_experiment(self, n_exps=None):
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

        for fluo in self.sensors.fluorophore.values:
            params[fluo + '_anisotropy'] = [[np.nan], ] * len(params)

        for row, param_set in tqdm(params.iterrows(), total=(len(params))):
            for param_name, param_val in param_set.items():
                if 'anisotropy' in param_name:
                    continue
                self.model.parameters[param_name].value = param_val
            anis_curves = self.get_anisotropy_curves()
            for fluo in self.sensors.fluorophore.values:
                params.at[row, fluo + '_anisotropy'] = anis_curves[fluo]

        return params
