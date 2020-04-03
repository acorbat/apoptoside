import numpy as np
import pandas as pd
from pysb.simulator import ScipyOdeSimulator

from caspase_model.models import arm
from caspase_model.shared import observe_biosensors
from . import anisotropy_functions as af

class Model(object):
    def __init__(self, stimuli='extrinsic'):
        self.model = arm(stimuli)
        self.parameters = self.model.parameters
        self.initial_conditions = self.get_initial_conditions()
        self.rates = self.get_rates()
        self.time = np.arange(0, 40000, 1)
        self.sensors = pd.DataFrame(columns=['fluorophore',
                                             'anisotropy_monomer',
                                             'anisotropy_monomer',
                                             'b', 'color'])
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
            b = this_sensor.b

            dimer_curve = simres[fluo + '_dimer']
            monomer_curve = simres[fluo + '_monomer']

            # Check whether all dimer has transformed into monomer
            if 2 * dimer_curve.max() - monomer_curve.max() > 1e-3:
                print('Not all dimer was cleaved!')

            m_curve = monomer_curve / monomer_curve.max()

            anisotropy = af.anisotropy_from_monomer(m_curve,
                                                    anisotropy_monomer,
                                                    anisotropy_dimer,
                                                    b)

            anis_curves[fluo] = anisotropy

        return anis_curves
