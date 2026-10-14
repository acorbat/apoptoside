import numpy as np
import pytest

from apoptoside import filter_data as fd

POPT = (0.2, 0.1, 0.5, 40.0)


def test_sigmoid_der1():
    x = np.linspace(0, 80, 801)
    np.testing.assert_allclose(
        fd.sigmoid_der1(x, *POPT)[1:-1],
        np.gradient(fd.sigmoid(x, *POPT), x)[1:-1],
        rtol=1e-3,
        atol=1e-6,
    )


def test_nanfit():
    x = np.arange(0, 80.0)
    y = fd.sigmoid(x, *POPT)
    y[[3, 10, 50]] = np.nan
    popt, _, fitline, resline = fd.nanfit(
        fd.sigmoid, y, xdata=x, p0=[0.2, 0.2, 0.3, 30], returnfit=True
    )
    np.testing.assert_allclose(popt, POPT, rtol=1e-6)
    np.testing.assert_array_equal(fitline[0], x)
    assert np.nanmax(np.abs(resline[1])) < 1e-8


def test_window_fit():
    x = np.arange(0, 200.0)
    y = fd.sigmoid(x, 0.2, 0.1, 0.5, 100.0)
    popts = fd.window_fit(fd.sigmoid, y, x=x, windowsize=50, windowstep=30)
    assert len(popts) == 6
    # the window centered on the sigmoid recovers its parameters
    np.testing.assert_allclose(popts[2], (0.2, 0.1, 0.5, 100.0), rtol=1e-3)


def test_is_apoptotic():
    assert fd.is_apoptotic(*POPT)
    assert not fd.is_apoptotic(0.2, 0.001, 0.5, 40.0)  # small amplitude
    assert not fd.is_apoptotic(0.2, 0.1, -0.5, 40.0)  # negative rate


@pytest.mark.parametrize("apoptotic", [True, False])
def test_is_apoptotic_region(apoptotic):
    popt = POPT if apoptotic else (0.2, 0.001, 0.5, 40.0)
    mask = fd.is_apoptotic_region(
        np.zeros(100, dtype=bool), [popt, popt], windowsize=50, windowstep=30
    )
    assert mask[:80].all() == apoptotic
    assert not mask[80:].any()
