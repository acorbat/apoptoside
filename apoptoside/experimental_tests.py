import numpy as np
import pandas as pd
import pysb
from typing import Dict
from caspase_model.shared import intrinsic_stimuli

from .model import Model as apop_model
import apoptoside.apoptoside as apop


def test_dynamics(df: pd.Series) -> pd.Series:
    """Receives a pandas series and corroborates whether timing between
    caspases and activity widths are among experimental values found [1]_[2]_.

    Parameters
    ----------
    df: pandas.Series
        Pandas Series containing the dynamic descriptors as columns.

    Returns
    -------
    test_results: pandas.Series
        Pandas Series determining whether each descriptor is inside
        experimental range

    References
    ----------
    .. [1] Corbat, AA; Schuermann, KC; Liguzinski, P; Radon, Y; Bastiaens, PIH;
        Verveer, PJ; Grecco, HE. "Co-imaging extrinsic, intrinsic and effector
        caspase activity by fluorescence anisotropy microscopy." Redox
        Biology 19 (2018): 210-217. :DOI:`10.1016/j.redox.2018.07.023`
    .. [2] Corbat, AA; Silberberg, M; Grecco, HE. "An Integrative Apoptotic
        Reaction Model for extrinsic and intrinsic stimuli." bioRxiv (2021):
        DOI:`10.1101/2021.05.21.444824`
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
    extrinsic and intrinsic stimuli [1]_ [2]_.

     References
    ----------
    .. [1] McComb, S; Chan, PK; Guinot, A; Hartmannsdottir, H; Jenni,
    S; Dobay, MP; Bourquin, J-P; Bornhauser, BC. "Efficient apoptosis
    requires feedback amplification of upstream apoptotic signals by effector
    caspase-3 or -7." Science Advances 5 (2019): eaau9433.
    :DOI:`10.1126/sciadv.aau9433`
    .. [2] Inoue, S; Browne, G; Melino, G; Cohen, GM. "Ordering of caspases in
    cells undergoing apoptosis by the intrinsic pathway." Cell Death and
    Differentiation 16 (2009): 1053-1061. :DOI:`10.1038/cdd.2009.29`
    """
    extrinsic_active_caspases = {'Cas3': True, 'Cas6': True,
                                 'Cas8': True, 'Apop': True}
    intrinsic_active_caspases = {'Cas3': True, 'Cas6': True,
                                 'Cas8': True, 'Apop': True}

    return check_caspases_western_blots(model,
                                        extrinsic_active_caspases=extrinsic_active_caspases,
                                        intrinsic_active_caspases=intrinsic_active_caspases)


def test_caspase8_knockout(model: pysb.Model) -> bool:
    """Test to corroborate adequate behaviour of the model when knocking out
    caspase 8. No caspase should be active while using extrinsic stimuli.
    Intrinsic and effector caspases should be active while using intrinsic
    stimuli[1]_.

     References
    ----------
    .. [1] McComb, S; Chan, PK; Guinot, A; Hartmannsdottir, H; Jenni,
    S; Dobay, MP; Bourquin, J-P; Bornhauser, BC. "Efficient apoptosis
    requires feedback amplification of upstream apoptotic signals by effector
    caspase-3 or -7." Science Advances 5 (2019): eaau9433.
    :DOI:`10.1126/sciadv.aau9433`"""
    extrinsic_active_caspases = {'Cas3': False, 'Cas6': False,
                                 'Cas8': False, 'Apop': False}
    extrinsic_inactive_caspases = {'Cas3': True, 'Cas6': True,
                                   'Cas8': False}
    intrinsic_active_caspases = {'Cas3': True, 'Cas6': True,
                                 'Cas8': False, 'Apop': True}
    intrinsic_inactive_caspases = {'Cas8': False}

    return check_caspases_western_blots(model, conditions={'C8_0': 0},
                                        extrinsic_active_caspases=extrinsic_active_caspases,
                                        intrinsic_active_caspases=intrinsic_active_caspases,
                                        extrinsic_inactive_caspases=extrinsic_inactive_caspases,
                                        intrinsic_inactive_caspases=intrinsic_inactive_caspases)


def test_caspase9_knockout(model: pysb.Model) -> bool:
    """Test to corroborate adequate behaviour of the model when knocking out
    caspase 9. Extrinsic and effector caspases should be active while using
    intrinsic stimuli. No caspase should be active while using intrinsic
    stimuli[1]_.

     References
    ----------
    .. [1] McComb, S; Chan, PK; Guinot, A; Hartmannsdottir, H; Jenni,
    S; Dobay, MP; Bourquin, J-P; Bornhauser, BC. "Efficient apoptosis
    requires feedback amplification of upstream apoptotic signals by effector
    caspase-3 or -7." Science Advances 5 (2019): eaau9433.
    :DOI:`10.1126/sciadv.aau9433`"""
    extrinsic_active_caspases = {'Cas3': True, 'Cas6': True,
                                 'Cas8': True, 'Apop': False}
    extrinsic_inactive_caspases = {'Apaf': False}
    intrinsic_active_caspases = {'Cas3': False, 'Cas6': False,
                                 'Cas8': False, 'Apop': False}
    intrinsic_inactive_caspases = {'Cas3': True, 'Cas6': True,
                                   'Cas8': True, 'Apaf': False}

    return check_caspases_western_blots(model, conditions={'Apaf_0': 0},
                                        extrinsic_active_caspases=extrinsic_active_caspases,
                                        intrinsic_active_caspases=intrinsic_active_caspases,
                                        extrinsic_inactive_caspases=extrinsic_inactive_caspases,
                                        intrinsic_inactive_caspases=intrinsic_inactive_caspases)


def test_caspase6_knockout(model: pysb.Model) -> bool:
    """Test to corroborate adequate behaviour of the model when knocking out
    caspase 6. Intrinsic and effector caspases should be active while using
    intrinsic stimuli, but extrinsic caspase should not be active. Every
    caspase should be active while using extrinsic stimuli[1]_.

     References
    ----------
    .. [1] McComb, S; Chan, PK; Guinot, A; Hartmannsdottir, H; Jenni,
    S; Dobay, MP; Bourquin, J-P; Bornhauser, BC. "Efficient apoptosis
    requires feedback amplification of upstream apoptotic signals by effector
    caspase-3 or -7." Science Advances 5 (2019): eaau9433.
    :DOI:`10.1126/sciadv.aau9433`"""
    extrinsic_active_caspases = {'Cas3': True, 'Cas6': False,
                                 'Cas8': True, 'Apop': True}
    extrinsic_inactive_caspases = {'Cas6': False}
    intrinsic_active_caspases = {'Cas3': True, 'Cas6': False,
                                 'Cas8': False, 'Apop': True}
    intrinsic_inactive_caspases = {'Cas6': False}

    return check_caspases_western_blots(model, conditions={'C6_0': 0},
                                        extrinsic_active_caspases=extrinsic_active_caspases,
                                        intrinsic_active_caspases=intrinsic_active_caspases,
                                        extrinsic_inactive_caspases=extrinsic_inactive_caspases,
                                        intrinsic_inactive_caspases=intrinsic_inactive_caspases)


def test_caspase3or7_knockout(model: pysb.Model) -> bool:
    """Test to corroborate adequate behaviour of the model when knocking out
    caspase 3 or 7. As these caspases are redundant, the test consists in
    halving caspase 3 initial concentration. All caspases should be active
    when using either stimuli[1]_.

     References
    ----------
    .. [1] McComb, S; Chan, PK; Guinot, A; Hartmannsdottir, H; Jenni,
    S; Dobay, MP; Bourquin, J-P; Bornhauser, BC. "Efficient apoptosis
    requires feedback amplification of upstream apoptotic signals by effector
    caspase-3 or -7." Science Advances 5 (2019): eaau9433.
    :DOI:`10.1126/sciadv.aau9433`"""
    extrinsic_active_caspases = {'Cas3': True, 'Cas6': True,
                                 'Cas8': True, 'Apop': True}
    extrinsic_inactive_caspases = {}
    intrinsic_active_caspases = {'Cas3': True, 'Cas6': True,
                                 'Cas8': True, 'Apop': True}
    intrinsic_inactive_caspases = {}

    return check_caspases_western_blots(model, conditions={'C3_0':
                                                               model.parameters['C3_0'].value // 2},
                                        extrinsic_active_caspases=extrinsic_active_caspases,
                                        intrinsic_active_caspases=intrinsic_active_caspases,
                                        extrinsic_inactive_caspases=extrinsic_inactive_caspases,
                                        intrinsic_inactive_caspases=intrinsic_inactive_caspases)


def test_caspase3_and_7_knockout(model: pysb.Model) -> bool:
    """Test to corroborate adequate behaviour of the model when knocking out
    caspase 3 and 7. When using extrinsic stimuli, only caspase 8 is active.
    When using intrinsic stimuli, only a slow form of caspase 9 should be
    active[1]_.

     References
    ----------
    .. [1] McComb, S; Chan, PK; Guinot, A; Hartmannsdottir, H; Jenni,
    S; Dobay, MP; Bourquin, J-P; Bornhauser, BC. "Efficient apoptosis
    requires feedback amplification of upstream apoptotic signals by effector
    caspase-3 or -7." Science Advances 5 (2019): eaau9433.
    :DOI:`10.1126/sciadv.aau9433`"""
    extrinsic_active_caspases = {'Cas3': False, 'Cas6': False,
                                 'Cas8': True, 'Apop': False}
    extrinsic_inactive_caspases = {'Cas3': False}
    intrinsic_active_caspases = {'Cas3': False, 'Cas6': False,
                                 'Cas8': False, 'Apop': False}
    intrinsic_inactive_caspases = {'Cas3': False}

    return check_caspases_western_blots(model, conditions={'C3_0': 0},
                                        extrinsic_active_caspases=extrinsic_active_caspases,
                                        intrinsic_active_caspases=intrinsic_active_caspases,
                                        extrinsic_inactive_caspases=extrinsic_inactive_caspases,
                                        intrinsic_inactive_caspases=intrinsic_inactive_caspases)


def test_caspase3_and_6_knockout(model: pysb.Model) -> bool:
    """Test to corroborate adequate behaviour of the model when knocking out
    caspase 3 and 6. Intrinsic and effector caspases should be active while
    using intrinsic stimuli, but extrinsic caspase should not be active. Every
    caspase should be active while using extrinsic stimuli[1]_.

     References
    ----------
    .. [1] McComb, S; Chan, PK; Guinot, A; Hartmannsdottir, H; Jenni,
    S; Dobay, MP; Bourquin, J-P; Bornhauser, BC. "Efficient apoptosis
    requires feedback amplification of upstream apoptotic signals by effector
    caspase-3 or -7." Science Advances 5 (2019): eaau9433.
    :DOI:`10.1126/sciadv.aau9433`"""
    extrinsic_active_caspases = {'Cas3': True, 'Cas6': False,
                                 'Cas8': True, 'Apop': True}
    extrinsic_inactive_caspases = {'Cas6': False}
    intrinsic_active_caspases = {'Cas3': True, 'Cas6': False,
                                 'Cas8': False, 'Apop': True}
    intrinsic_inactive_caspases = {'Cas6': False}

    return check_caspases_western_blots(model, conditions={'C3_0':
                                                               model.parameters['C3_0'].value // 2,
                                                           'C6_0': 0},
                                        extrinsic_active_caspases=extrinsic_active_caspases,
                                        intrinsic_active_caspases=intrinsic_active_caspases,
                                        extrinsic_inactive_caspases=extrinsic_inactive_caspases,
                                        intrinsic_inactive_caspases=intrinsic_inactive_caspases)


def test_caspase3_7_and_6_knockout(model: pysb.Model) -> bool:
    """Test to corroborate adequate behaviour of the model when knocking out
    caspase 3, 7 and 6. When using extrinsic stimuli, only caspase 8 is active.
    When using intrinsic stimuli, only a slow form of caspase 9 should be
    active[1]_.

     References
    ----------
    .. [1] McComb, S; Chan, PK; Guinot, A; Hartmannsdottir, H; Jenni,
    S; Dobay, MP; Bourquin, J-P; Bornhauser, BC. "Efficient apoptosis
    requires feedback amplification of upstream apoptotic signals by effector
    caspase-3 or -7." Science Advances 5 (2019): eaau9433.
    :DOI:`10.1126/sciadv.aau9433`"""
    extrinsic_active_caspases = {'Cas3': False, 'Cas6': False,
                                 'Cas8': True, 'Apop': False}
    extrinsic_inactive_caspases = {'Cas3': False, 'Cas6': False}
    intrinsic_active_caspases = {'Cas3': False, 'Cas6': False,
                                 'Cas8': False, 'Apop': False}
    intrinsic_inactive_caspases = {'Cas3': False, 'Cas6': False}

    return check_caspases_western_blots(model, conditions={'C3_0': 0,
                                                           'C6_0': 0},
                                        extrinsic_active_caspases=extrinsic_active_caspases,
                                        intrinsic_active_caspases=intrinsic_active_caspases,
                                        extrinsic_inactive_caspases=extrinsic_inactive_caspases,
                                        intrinsic_inactive_caspases=intrinsic_inactive_caspases)


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


def test_bid_perturbation(model: pysb.Model, sensors_path) -> bool:
    """Tests the effect of reducing bid concentration on the apoptosis onset
    time when using extrinsic stimuli [1]_[2]_. Reducing the initial
    concentration of this proapoptotic protein should increase onset time.

    References
    ----------
    .. [1] Gaudet, S; Spencer, SL; Chen, WW; Sorger, PK. "Exploring the
    contextual sensitivity of factors that determine cell-to-cell variability in
     receptor-mediated apoptosis." PLoS Computational Biology 8 (2012).
     :DOI:`10.1371/journal.pcbi.1002482`
     .. [2] Spencer, SL,; Gaudet, S; Albeck, JG; Burke, JM; Sorger,
     PK. "Non-genetic origins of cell-to-cell variability in TRAIL-induced
     apoptosis." Nature 459 (2009): 428-432. :DOI:`10.1038/nature08012`
    """
    model = apop_model(model)
    model.load_sensors(sensors_path)
    model.add_parameter_set({'Bid_0': model.parameters['Bid_0'].value // 2})
    sim_data = model.simulate_experiment()

    analyzer = apop.Apop(sim_data=sim_data, sensors_path=sensors_path)
    analyzer.df = analyzer.df.sort_values('Bid_0', ascending=True)
    if analyzer.df.Cas3_max_time.values[0] > \
            analyzer.df.Cas3_max_time.values[1]:
        return True
    else:
        return False


def test_bcl_2_perturbation(model: pysb.Model, sensors_path) -> bool:
    """Tests the effect of reducing bcl-2 concentration on the apoptosis onset
    time when using extrinsic stimuli [1]_. Reducing concentration of this
    anti-apoptotic protein should reduce onset time.

    References
    ----------
    .. [1] Gaudet, S; Spencer, SL; Chen, WW; Sorger, PK. "Exploring the
    contextual sensitivity of factors that determine cell-to-cell variability in
     receptor-mediated apoptosis." PLoS Computational Biology 8 (2012).
     :DOI:`10.1371/journal.pcbi.1002482`
    """
    model = apop_model(model)
    model.load_sensors(sensors_path)
    model.add_parameter_set({'Bcl2_0': model.parameters['Bcl2_0'].value // 2})
    sim_data = model.simulate_experiment()

    analyzer = apop.Apop(sim_data=sim_data, sensors_path=sensors_path)
    analyzer.df = analyzer.df.sort_values('Bcl2_0', ascending=True)
    if analyzer.df.Cas3_max_time.values[0] < \
            analyzer.df.Cas3_max_time.values[1]:
        return True
    else:
        return False


def test_flip_perturbation(model: pysb.Model, sensors_path) -> bool:
    """Tests the effect of reducing FLIP concentration on the apoptosis onset
    time when using extrinsic stimuli [1]_. Reducing the initial
    concentration of this anti-apoptotic protein should reduce onset time.

    References
    ----------
    .. [1] Spencer, SL,; Gaudet, S; Albeck, JG; Burke, JM; Sorger,
     PK. "Non-genetic origins of cell-to-cell variability in TRAIL-induced
     apoptosis." Nature 459 (2009): 428-432. :DOI:`10.1038/nature08012`
    """
    model = apop_model(model)
    model.load_sensors(sensors_path)
    model.add_parameter_set({'flip_0': model.parameters['flip_0'].value // 2})
    sim_data = model.simulate_experiment()

    analyzer = apop.Apop(sim_data=sim_data, sensors_path=sensors_path)
    analyzer.df = analyzer.df.sort_values('flip_0', ascending=True)
    if analyzer.df.Cas3_max_time.values[0] < \
            analyzer.df.Cas3_max_time.values[1]:
        return True
    else:
        return False


def test_intrinsic_xiap_perturbation(model: pysb.Model, sensors_path) -> bool:
    """Tests the effect of reducing XIAP concentration on the apoptosis onset
    time when using intrinsic stimuli [1]_. Reducing the initial
    concentration of this anti-apoptotic protein should reduce onset time.

    References
    ----------
    .. [1] Rehm, M; Huber, HJ; Dussmann, H; Prehn, JHM. "Systems analysis of
    effector caspase activation and its control by X-linked inhibitor of
    apoptosis protein." EMBO Journal 25 (2006): 4338-4349.
    :DOI:`10.1038/sj.emboj.7601295`
    """
    try:
        intrinsic_stimuli(model)
    except pysb.ComponentDuplicateNameError:
        pass
    model = apop_model(model)
    model.load_sensors(sensors_path)
    model.add_parameter_set({'XIAP_0': model.parameters['XIAP_0'].value // 2})
    sim_data = model.simulate_experiment()

    analyzer = apop.Apop(sim_data=sim_data, sensors_path=sensors_path)
    analyzer.df = analyzer.df.sort_values('XIAP_0', ascending=True)
    if analyzer.df.Cas3_max_time.values[0] < \
            analyzer.df.Cas3_max_time.values[1]:
        return True
    else:
        return False


def test_intrinsic_caspase3_perturbation(model: pysb.Model, sensors_path) -> \
        bool:
    """Tests the effect of reducing caspase 3 concentration on the apoptosis
    onset time when using intrinsic stimuli [1]_. Reducing the initial
    concentration of this caspase should increase onset time.

    References
    ----------
    .. [1] Rehm, M; Huber, HJ; Dussmann, H; Prehn, JHM. "Systems analysis of
    effector caspase activation and its control by X-linked inhibitor of
    apoptosis protein." EMBO Journal 25 (2006): 4338-4349.
    :DOI:`10.1038/sj.emboj.7601295`
    """
    try:
        intrinsic_stimuli(model)
    except pysb.ComponentDuplicateNameError:
        pass
    model = apop_model(model)
    model.load_sensors(sensors_path)
    model.add_parameter_set({'C3_0': model.parameters['C3_0'].value // 2})
    sim_data = model.simulate_experiment()

    analyzer = apop.Apop(sim_data=sim_data, sensors_path=sensors_path)
    analyzer.df = analyzer.df.sort_values('C3_0', ascending=True)
    if analyzer.df.Cas3_max_time.values[0] > \
            analyzer.df.Cas3_max_time.values[1]:
        return True
    else:
        return False


def test_intrinsic_smac_perturbation(model: pysb.Model, sensors_path) -> \
        bool:
    """Tests the effect of reducing Smac concentration on the apoptosis
    onset time when using intrinsic stimuli [1]_. Reducing the initial
    concentration of Smac should not affect considerably onset time (less
    than 15 min effect).

    References
    ----------
    .. [1] Rehm, M; Huber, HJ; Dussmann, H; Prehn, JHM. "Systems analysis of
    effector caspase activation and its control by X-linked inhibitor of
    apoptosis protein." EMBO Journal 25 (2006): 4338-4349.
    :DOI:`10.1038/sj.emboj.7601295`
    """
    try:
        intrinsic_stimuli(model)
    except pysb.ComponentDuplicateNameError:
        pass
    model = apop_model(model)
    model.load_sensors(sensors_path)
    model.add_parameter_set({'Smac_0': model.parameters['Smac_0'].value // 2})
    sim_data = model.simulate_experiment()

    analyzer = apop.Apop(sim_data=sim_data, sensors_path=sensors_path)
    analyzer.df = analyzer.df.sort_values('Smac_0', ascending=True)
    if np.abs(np.diff(analyzer.df.Cas3_max_time.values)[0]) < 15:
        return True
    else:
        return False


def run_all_tests(model: pysb.Model, sensors_path) -> dict:
    """Runs all tests on the model and print the results."""
    test_results = {}
    success = True

    print('\nTesting Caspase Dynamics\n'
          '------------------------\n')
    my_model = pysb.Model(base=model)
    a_model = apop_model(my_model)
    a_model.load_sensors(sensors_path)
    a_model.set_duplicate_stimuli()
    sim_data = a_model.simulate_experiment()
    analyzer = apop.Apop(sim_data=sim_data, sensors_path=sensors_path)
    analyzer.add_test_dynamics()

    test_cols = [col for col in analyzer.df.columns if col.endswith('test')]
    test_dynamics_results = {}
    for stim in ['extrinsic', 'intrinsic']:
        this_test_results = {stim + ' ' + col: analyzer.df.query('stimuli == '
                                                                 '"%s"' %
                                                                 stim)[
            col].values[0]
                             for col in test_cols}
        test_dynamics_results.update(this_test_results)

    overall_dynamics = True
    for key, val in test_dynamics_results.items():
        result = 'passed' if val else 'failed'
        print(key + ': ' + result)

        if not val:
            overall_dynamics = False
            success = False

    test_results['dynamics'] = overall_dynamics

    print('\nTesting Western Blot Experiments\n'
          '--------------------------------\n')

    KO_tests = {'caspase activation': test_caspase_activation,
                'caspase 8 KO': test_caspase8_knockout,
                'caspase 9 KO': test_caspase9_knockout,
                'caspase 3 or 7 KO': test_caspase3or7_knockout,
                'caspase 6 KO': test_caspase6_knockout,
                'caspase 3 and 6 KO': test_caspase3_and_6_knockout,
                'caspase 3 and 7 KO': test_caspase3_and_7_knockout,
                'caspase 3, 7 and 6 KO': test_caspase3_7_and_6_knockout}

    for name, test in KO_tests.items():
        my_model = pysb.Model(base=model)
        test_result = test(my_model)
        result = 'passed' if test_result else 'failed'
        print(name + ': ' + result)

        test_results[name] = test_result
        if not test_result:
            success = False

    print('\nTesting Perturbation Experiments\n'
          '--------------------------------\n')

    perturbation_tests = {'extrinsic Bid perturbation':
                              test_bid_perturbation,
                          'extrinsic Bcl-2 perturbation':
                              test_bcl_2_perturbation,
                          'extrinsic FLIP perturbation': test_flip_perturbation,
                          'intrinsic XIAP perturbation':
                              test_intrinsic_xiap_perturbation,
                          'intrinsic caspase 3 perturbation':
                              test_intrinsic_caspase3_perturbation,
                          'intrinsic SMAC perturbation':
                              test_intrinsic_smac_perturbation}

    for name, test in perturbation_tests.items():
        my_model = pysb.Model(base=model)
        test_result = test(my_model, sensors_path)
        result = 'passed' if test_result else 'failed'
        print(name + ': ' + result)

        test_results[name] = test_result
        if not test_result:
            success = False

    print('\nOverall\n'
          '-------\n')
    print('Passed' if success else 'Failed')
    test_results['success'] = success
    return test_results
