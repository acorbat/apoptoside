import numpy as np
from scipy.misc import derivative
from scipy.interpolate import splrep, splev


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
    """Estimates delta b parameter from the pre and pos cleavage fluorescence
    """
    return fluo_pre/fluo_pos - 1


def calculate_activity(time, anisotropy, delta_b, order=5):
    """Calculates activity of enzyme according to anisotropy curve."""
    time_steps = np.diff(time)
    time_step = np.nanmin(time_steps)
    if not all(time_steps == time_step):
        time, anisotropy = fill_in_curves(time, anisotropy, smooth=True)

    def this_vect(t):
        inds = np.searchsorted(time, t)
        inds = np.clip(inds, 0, len(anisotropy) - 1)
        return anisotropy[inds]

    der = derivative(this_vect, time, dx=time_step, order=order)

    anisotropy_normalized = normalize(anisotropy)
    activity = der / ((1 + delta_b * anisotropy_normalized) ** 2)

    return np.asarray(time), np.asarray(activity)


def fill_in_curves(time, curve, smooth=False):
    """Fill in the missing values with interpolation to get equispaced curve"""
    start = np.nanmin(time)
    end = np.nanmax(time)
    time_step = np.nanmin(np.diff(time))
    complete_time = np.arange(start, end, time_step)

    if not smooth:
        complete_curve = np.interp(complete_time, time, curve,
                                   left=curve[0], right=curve[0])
    else:
        complete_curve = interpolate(complete_time, time, curve)

    return complete_time, complete_curve


def normalize(vect):
    """
    Returns a normalization of vect from 0 to 1
    """
    this_vect = vect[:]
    this_vect -= np.nanmin(this_vect)
    this_vect /= np.nanmax(this_vect)
    return this_vect


def interpolate(new_time, time, curve):
    """Interpolate curve using new_time as xdata"""
    f = splrep(time, curve, k=3)
    return splev(new_time, f, der=0)
