import numpy as np

from scipy.optimize import curve_fit


def sigmoid(x, base, amplitude, rate, x0):
    """
    Sigmoid function.
    """
    y = base + amplitude / (1 + np.exp(-rate * (x - x0)))
    return y


def sigmoid_der1(x, base, amplitude, rate, x0):
    """
    First derivative of sigmoid function.
    """
    return (amplitude * rate * np.exp(-rate * (x - x0))) / ((1 + np.exp(-rate * (x - x0))) ** 2)


def nanfit(func, ydata, xdata=None, timepoints=10, returnfit=False, p0=None):
    """
    Fits function func to ydata ignoring nans.

    Parameters
    ----------
    func : function
        Function used to fit.
    ydata : Array-like, list
        Curve to be fitted
    xdata : Array-like, list
        x curve for the y curve. Defaults to None. If None, timepoints is used to generate
        an equispaced vector.
    timepoints : float
        Time span of the y curve. Defaults to 10. Ignored if xdata is given.
    returnfit : boolean, optional
        If True, returns fitline and resline
    p0 : list of floats, optional
        Initial parameters to be used in non linear fit.

    Returns
    -------
    popt : Array-like
        Parameters returned from non linear fit.
    pcov : Array-like
        Covariance matrix from fit.
    fitline : tuple of curves, optional
        If returnfit, returns a tuple of xdata and fit result.
    resline : tuple of curves, optional
        If returnfit, returns a tuple of xdata and fit result residues.
    """
    if xdata is None:
        xdata = np.arange(ydata.shape[0])
        xdata = xdata * timepoints

    mask = np.where(np.isfinite(ydata))

    if mask == np.array([]):
        popt = None
        pcov = None
    else:
        if p0 is not None:
            popt, pcov = curve_fit(func, xdata[mask], ydata[mask], p0=p0)
        else:
            popt, pcov = curve_fit(func, xdata[mask], ydata[mask])

    if returnfit and popt is not None:
        fitline = (xdata, func(xdata, *popt))
        resline = (xdata, ydata - fitline[1])
    else:
        fitline = None
        resline = None

    return popt, pcov, fitline, resline


def window_fit(func, y, x=None, timepoints=10, windowsize=30, windowstep=10):
    """
    Performs succesive fits with func in the windowed curve y, with window size windowsize
    and window step windowstep. Returns a list with the popts for each window.

    Take into consideration that windowsize will set a maximum for the rates that can
    be fitted adequately. windowstep should be smaller thatn windowsize in order to avoid
    the possibility that half a sigmoid is in one window and the other half in another.

    Parameters
    ----------
    func : function
        Function used to fit.
    y : Array-like, list
        Curve to be fitted.
    x : Array-like, list, optional
        x curve for y data. Default is None. Makes timepoints unnecessary if given.
    timepoints : float, optional
        Time span of the y curve. Default is 10.
    windowsize : int, optional
        Amount of points to be taken into the window for fitting. Default is 30
    windowstep : int, optional
        Step for the traslation of each window generated.

    Returns
    -------
    popts : list of list of floats
        List containing the popts returned for each windowed fit.
    """
    if x is None:
        x = np.arange(y.shape[0])
        x = x * timepoints

    popts = []
    start_ind = 0
    end_ind = 0
    end_of_series = y.shape[0]
    while end_ind < end_of_series-1:
        end_ind += windowsize
        end_ind = np.clip(end_ind, 0, end_of_series-1)
        windowed_y = y[start_ind:end_ind]
        windowed_x = x[start_ind:end_ind]
        half_point = (end_ind - start_ind) // 2 + start_ind

        chi2 = float('+inf')
        if np.sum(np.isfinite(windowed_y)) >= windowsize // 2:
            for rate in (.1, .17, .1, .7, 1):
                try:
                    _popt, _, _, (_, res) = nanfit(func, windowed_y, xdata=windowed_x, timepoints=timepoints,
                                                   p0=[0.2, 0.2, rate, x[half_point]], returnfit=True)
                    _chi2 = np.nansum(res * res)
                    if _chi2 < chi2:
                        popt = np.copy(_popt)
                        chi2 = np.copy(_chi2)
                except RuntimeError:
                    if 'popt' not in locals():
                        popt = [np.nan] * 4

            popts.append((popt))

        else:
            popts.append([np.nan] * 4)

        start_ind += windowstep

    return popts
