import pandas as pd


def test_dynamics(df: pd.Series) -> dict:
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
