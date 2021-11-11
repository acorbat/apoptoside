import pandas as pd
import pysb
from typing import Dict

from .model import Model as apop_model


def test_dynamics(df: pd.Series) -> pd.Series:
    """Receives a pandas series and corroborates whether timing between
    caspases and activity widths are among experimental values found.

    Parameters
    ----------
    df: pandas.Series
        Pandas Series containing the dynamic descriptors as columns.

    Returns
    -------
    test_results: pandas.Series
        Pandas Series determining whether each descriptor is inside
        experimental range
    """
    test_dict = {'extrinsic': {'Cas9_to_Cas3': (5.5, 15),
                               'Cas8_to_Cas3': (2, 9),
                               'Cas3_act_width': (19.0, 57.5),
                               'Cas8_act_width': (19.0, 30.0),
                               'Cas9_act_width': (30.5, 125.0)},
                 'intrinsic': {'Cas9_to_Cas3': (2, 7),
                               'Cas8_to_Cas3': (5, 14),
                               'Cas3_act_width': (12.0, 16.0),
                               'Cas8_act_width': (14.0, 34.0),
                               'Cas9_act_width': (12.0, 28.0)}}

    this_stimuli = df.stimuli
    this_condition_dict = test_dict[this_stimuli]

    test_results = {}
    for this_cond, (min_val, max_val) in this_condition_dict.items():
        test_results[this_cond + '_test'] = min_val < df[this_cond] < max_val

    return pd.Series(test_results)


def test_caspase_activation(model: pysb.Model) -> bool:
    """Simple test to corroborate that every caspase is active when using
    extrinsic and intrinsic stimuli"""
    extrinsic_active_caspases = {'Cas3': True, 'Cas6': True,
                                 'Cas8': True, 'Apop': True}
    intrinsic_active_caspases = {'Cas3': True, 'Cas6': True,
                                 'Cas8': True, 'Apop': True}

    check_caspases_western_blots(model,
                                 extrinsic_active_caspases=extrinsic_active_caspases,
                                 intrinsic_active_caspases=intrinsic_active_caspases)


def check_caspases_western_blots(model: pysb.Model, conditions: Dict = None,
                                 extrinsic_active_caspases: Dict = {},
                                 intrinsic_active_caspases: Dict = {},
                                 extrinsic_inactive_caspases: Dict = {},
                                 intrinsic_inactive_caspases: Dict = {}) -> bool:
    """Tests whether a model under the given conditions has active or inactive
    caspases as evaluated through western blot while using extrinsic or
    intrinsic stimuli.

    Parameters
    ----------
    model: pysb.Model
        PySB model to test
    conditions: dictionary
        Dictionary with specific conditions to add to the model. For example,
        caspase knockout experiments by taking procaspase concentration to 0.
    extrinsic_active_caspases: dictionary
        Dictionary containing which active caspase (key) should be found or not
        (boolean value) in an extrinsic stimulated case
    intrinsic_active_caspases dictionary
        Dictionary containing which active caspase (key) should be found or not
        (boolean value) in an intrinsic stimulated case
    extrinsic_inactive_caspases dictionary
        Dictionary containing which inactive caspase (key) should be found or
        not (boolean value) in an extrinsic stimulated case
    intrinsic_inactive_caspases dictionary
        Dictionary containing which inactive caspase (key) should be found or
        not (boolean value) in an intrinsic stimulated case

    Returns
    -------
    bool
        True if every caspase is found in the expected state
    """
    sim_data = simulate_for_WB(model, conditions)

    for col in sim_data.columns:
        if not col.endswith('active'):
            continue
        sim_data[col + '_WB'] = sim_data.apply(lambda x: check_presence(x, col),
                                               axis=1)

    if conditions is not None:
        for item in conditions.items():
            sim_data = sim_data.query('%s == %s' % item)

    ext_sim_data = sim_data.query('IntrinsicStimuli_0 == 0')
    int_sim_data = sim_data.query('L_0 == 0')

    check_dicts = [extrinsic_active_caspases, intrinsic_active_caspases,
                   extrinsic_inactive_caspases, intrinsic_inactive_caspases]
    datas = [ext_sim_data, int_sim_data, ext_sim_data, int_sim_data]
    states = ['active', 'active', 'inactive', 'inactive']

    for check_dict, data, state in zip(check_dicts, datas, states):
        for casp, presence in check_dict.items():
            if all(data['_'.join([casp, state, 'WB'])] != presence):
                return False
    return True

def simulate_for_WB(model: pysb.Model, conditions: dict = None) -> pd.DataFrame:
    """Simulate model with specific conditions as perturbations.

    Parameters
    ----------
    model: pysb.Model
        Model to be evaluated and perturbed
    conditions: dictionary
        Dictionary containing parameters and values to be modified in the model

    Returns
    -------
    sim_data: pandas.DataFrame
        Pandas DataFrame containing data from the simulation and columns for
        each caspase curve in active and inactive state
    """
    model = apop_model(model, biosensors=False)
    model.set_duplicate_stimuli()
    if conditions is not None:
        model.add_parameter_set(conditions)
    sim_data = model.simulate_western_blot()
    return sim_data


def check_presence(sim_data: pd.Series, protein_name: str,
                   timepoint: int = -1) -> bool:
    """Checks whether there is presence of some protein at some timepoint."""
    return sim_data[protein_name][timepoint] > 1