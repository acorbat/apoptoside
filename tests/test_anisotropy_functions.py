import numpy as np
import pytest

from apoptoside import anisotropy_functions as af

# Monomer and dimer anisotropy, and brightness relation
AM, AD, B = 0.3, 0.2, 1.5
M = np.linspace(0, 1, 11)


def test_anisotropy_from_monomer_limits():
    assert af.anisotropy_from_monomer(0, AM, AD, B) == pytest.approx(AD)
    assert af.anisotropy_from_monomer(1, AM, AD, B) == pytest.approx(AM)


def test_intensities_from_anisotropy():
    anisotropy, fluorescence = 0.25, 100.0
    i_par = af.intensity_parallel_from_anisotropy(anisotropy, fluorescence)
    i_per = af.intensity_perpendicular_from_anisotropy(anisotropy, fluorescence)
    assert af.anisotropy_from_intensity(i_par, i_per) == pytest.approx(anisotropy)
    assert af.total_fluorescence_from_intensity(i_par, i_per) == pytest.approx(
        fluorescence
    )


def test_intensities_from_monomer():
    c_b_1 = 2.0
    i_par = af.intensity_parallel_from_monomer(M, AM, AD, B, c_b_1)
    i_per = af.intensity_perpendicular_from_monomer(M, AM, AD, B, c_b_1)
    np.testing.assert_allclose(
        af.anisotropy_from_intensity(i_par, i_per),
        af.anisotropy_from_monomer(M, AM, AD, B),
    )
    np.testing.assert_allclose(
        af.total_fluorescence_from_intensity(i_par, i_per),
        af.total_fluorescence_from_monomer(M, B, c_b_1),
    )


def test_normalized_intensities_from_monomer():
    i_par = af.intensity_parallel_from_monomer_r(M, AM, AD, B)
    i_per = af.intensity_perpendicular_from_monomer_r(M, AM, AD, B)
    np.testing.assert_allclose(i_par + 2 * i_per, 1)
    np.testing.assert_allclose(
        af.anisotropy_from_intensity(i_par, i_per),
        af.anisotropy_from_monomer(M, AM, AD, B),
    )
