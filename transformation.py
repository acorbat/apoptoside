import numpy as np


def get_region(region, curve, mask, length=5):
    """
    Return values of curve around the beginning ('pre' region) or ending ('pos'
     region) of mask with specified length.
    Parameters
    ----------
    region : str, 'pre' or 'pos'
        Region to be returned
    curve : list, np.array
        Curve of interest
    mask : list, np.array
        Mask of the region of interest of the curve
    length : int, optional (default=5)
        length of the region of interest to be returned. Should be odd.

    Returns
    -------
    array of curve around the region of interest
    """
    if not any(mask):
        return [np.nan] * length

    inds = np.where(mask)[0]

    if (np.diff(inds) != 1).any():
        return [np.nan] * length

    if region not in ['pre', 'pos']:
        raise ValueError('Only pre or pos region.')
    center = inds[0] if region == 'pre' else inds[-1]

    start_ini = np.clip(center - length // 2, 0, len(curve))
    end_ini = np.clip(center + length // 2, 0, len(curve))

    return curve[start_ini:end_ini + 1]


def estimate_delta_b(fluo_pre, fluo_pos):
    return fluo_pre/fluo_pos - 1
