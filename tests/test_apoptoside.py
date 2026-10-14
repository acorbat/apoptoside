import numpy as np
import pytest

from apoptoside.apoptoside import Apop


@pytest.mark.parametrize("offset", [0, 1e-9])
def test_get_activity_width(offset):
    time = np.arange(0, 600) / 60
    sigma = 0.5
    activity = np.exp(-((time - 5) ** 2) / (2 * sigma**2))
    max_time = time[np.argmax(activity)] + offset  # from a resampled time vector
    width = Apop._get_activity_width(None, time, activity, max_time)
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    assert width == pytest.approx(fwhm, rel=1e-2)
