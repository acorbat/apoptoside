import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

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


@pytest.mark.parametrize("cmap", ["Blues_d", "Blues"])
def test_plot_kdes(cmap):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(200, 2)), columns=["a", "b"])
    viewer.plot_kdes(df, "a", "b", cmaps=cmap, labels="x", gridsizes=30)
    plt.close("all")
