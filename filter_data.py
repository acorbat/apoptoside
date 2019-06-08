import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
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
        half_point = (end_ind + start_ind) // 2

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


def is_apoptotic(base, amplitude, rate, x0,
                    base_lim=(-0.1, 0.5),
                    amplitude_lim=(0.005, 0.5),
                    rate_lim=(0, np.inf),
                    x0_lim=(0, np.inf)):
    """
    Quick checks that parameters of sigmoid are within possible range.

    -0.1 <= base <= 0.5
    0.005 <= amplitude <= 0.5
    0 <= rate <= np.inf
    0 <= x0 <= np.inf
    """

    if base_lim[0] <= base <= base_lim[1] \
            and amplitude_lim[0] <= amplitude <= amplitude_lim[1] \
            and rate_lim[0] <= rate <= rate_lim[1] \
            and x0_lim[0] <= x0 <= x0_lim[1]:
        return True

    else:
        return False


def is_apoptotic_region(mask, popts, windowsize=50, windowstep=30):
    start_ind = 0
    end_ind = 0
    end_of_series = len(mask)
    apop_reg_ind = 0
    while end_ind < end_of_series:
        end_ind += windowsize
        end_ind = np.clip(end_ind, 0, end_of_series)

        mask[start_ind:end_ind] = [is_apoptotic(*popts[apop_reg_ind])] * (end_ind - start_ind)

        start_ind += windowstep
        apop_reg_ind += 1

    return mask


def manual_region_selection(time, anis):

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    t = np.arange(0.0, 1.0, 0.001)
    s = np.sin(2 * np.pi * freqs[0] * t)
    l, = plt.plot(t, s, lw=2)

    class Index(object):
        ind = 0

        def next(self, event):
            self.ind += 1
            i = self.ind % len(freqs)
            ydata = np.sin(2 * np.pi * freqs[i] * t)
            l.set_ydata(ydata)
            plt.draw()

        def prev(self, event):
            self.ind -= 1
            i = self.ind % len(freqs)
            ydata = np.sin(2 * np.pi * freqs[i] * t)
            l.set_ydata(ydata)
            plt.draw()

    callback = Index()
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    plt.show()

    def onselect(eclick, erelease):
        "eclick and erelease are matplotlib events at press and release."
        xs = []
        ys = []
        xs.append(eclick.xdata)
        ys.append(eclick.ydata)
        xs.append(erelease.xdata)
        ys.append(erelease.ydata)

        return [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]

    fig, axs = plt.subplots()

    axs.plot(time, anis)
    rect = RectangleSelector(axs, onselect)

    selections = []
    def on_click(selections=selections):
        plt.close()

        selections.append(rect.extents)

    button = Button(axs, 'Finish')
    button.on_clicked(on_click)

    plt.show()

    return rect.extents

