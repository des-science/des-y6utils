from ..mdet import _compute_asinh_dered_mag, _compute_asinh_flux, _compute_asinh_mags
import math


def test_compute_asinh_flux():
    # this test ensures the function computes magnitudes and fluxes correctly.
    flux_input = 10000
    mag_g = _compute_asinh_mags(flux_input, 0)
    flux_g = _compute_asinh_flux(mag_g, 0)

    assert math.isclose(flux_input, flux_g, rel_tol=1e-9)


def test_compute_asinh_dered_mag():
    flux_input = 10000
    mag_g = _compute_asinh_mags(flux_input, 0)
    dered_mag_g = _compute_asinh_dered_mag(flux_input, 0, 1 / 3.186)

    assert math.isclose(mag_g - 1.0, dered_mag_g, rel_tol=1e-9)
