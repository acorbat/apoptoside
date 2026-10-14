import numpy as np
import pytest

from apoptoside import viewer


@pytest.mark.parametrize(
    "percentage, expected",
    # Computed with the previous implementation, based on seaborn's KDE.
    [
        (0.34, 0.7083944236087397),
        (0.68, 0.3877122387386788),
        (0.95, 0.07206563927259543),
    ],
)
def test_get_level_for_kde(percentage, expected):
    rng = np.random.default_rng(1)
    xs, ys = rng.normal(size=(2, 300))
    ys = ys + xs
    assert viewer.get_level_for_kde(xs, ys, percentage) == pytest.approx(expected)
