import numpy as np
import pytest

from apoptoside import anisotropy_functions as af
from apoptoside import transformation as tf

AM, AD, DELTA_B = 0.3, 0.2, 0.5


def sigmoid_anisotropy(time):
    monomer = 1 / (1 + np.exp(-(time - 50) / 5))
    return af.anisotropy_from_monomer(monomer, AM, AD, 1 + DELTA_B)


def test_calculate_monomer_curve():
    monomer = np.linspace(0, 1, 11)
    anisotropy = af.anisotropy_from_monomer(monomer, AM, AD, 1 + DELTA_B)
    np.testing.assert_allclose(tf.calculate_monomer_curve(anisotropy, DELTA_B), monomer)


def test_estimate_delta_b():
    assert tf.estimate_delta_b(3.0, 2.0) == pytest.approx(0.5)


@pytest.mark.parametrize("method", ["findif", "savgol"])
def test_calculate_activity_linear(method):
    time = np.arange(0, 100.0)
    anisotropy = 0.2 + 1e-3 * time
    t, activity = tf.calculate_activity(time, anisotropy, delta_b=0, method=method)
    np.testing.assert_array_equal(t, time)
    # away from the edges
    np.testing.assert_allclose(activity[10:-10], 1e-3)


# savgol smooths the curve, flattening its peak
@pytest.mark.parametrize("method, atol", [("findif", 1e-4), ("savgol", 1e-3)])
def test_calculate_activity_sigmoid(method, atol):
    time = np.arange(0, 100.0)
    anisotropy = sigmoid_anisotropy(time)
    normalized = (anisotropy - AD) / (AM - AD)
    expected = np.gradient(anisotropy, time) / (1 + DELTA_B * normalized) ** 2
    _, activity = tf.calculate_activity(time, anisotropy, DELTA_B, method=method)
    np.testing.assert_allclose(activity[5:-5], expected[5:-5], atol=atol)
    assert abs(np.argmax(activity) - np.argmax(expected)) <= 1


def test_calculate_activity_does_not_modify_input():
    time = np.arange(0, 100.0)
    anisotropy = sigmoid_anisotropy(time)
    original = anisotropy.copy()
    tf.calculate_activity(time, anisotropy, DELTA_B)
    np.testing.assert_array_equal(anisotropy, original)


def test_normalize():
    vect = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(tf.normalize(vect), [0, 0.5, 1])
    np.testing.assert_array_equal(vect, [1.0, 2.0, 3.0])


def test_fill_in_curves():
    time = np.array([0.0, 1.0, 2.0, 4.0, 5.0])
    complete_time, curve = tf.fill_in_curves(time, 2 * time)
    np.testing.assert_array_equal(complete_time, np.arange(0, 5.0))
    np.testing.assert_allclose(curve, 2 * complete_time)


def test_single_apoptotic():
    assert tf.single_apoptotic([False, True, True, False])
    assert not tf.single_apoptotic([False, False])
    assert not tf.single_apoptotic([True, False, True])


def test_get_region():
    curve = np.arange(10.0)
    mask = np.zeros(10, dtype=bool)
    mask[4:7] = True
    np.testing.assert_array_equal(
        tf.get_region("pre", curve, mask, length=3), [3, 4, 5]
    )
    np.testing.assert_array_equal(
        tf.get_region("pos", curve, mask, length=3), [5, 6, 7]
    )
    with pytest.raises(ValueError):
        tf.get_region("other", curve, mask)
    assert np.isnan(tf.get_region("pre", curve, np.zeros(10, dtype=bool))).all()
