from ..mdet import _compute_dered_flux, _compute_asinh_flux, _compute_asinh_mags
import math


def test_compute_asinh_flux():
    # this test ensures the function computes magnitudes and fluxes correctly.
    flux_input = 10000
    mag_g = _compute_asinh_mags(flux_input, 0)
    flux_g = _compute_asinh_flux(mag_g, 0)

    assert math.isclose(flux_input, flux_g, rel_tol=1e-9)


def test_compute_dered_flux():
    flux_input = 10000
    dered_flux = _compute_dered_flux(flux_input, 0, 2.5 / 3.186)

    assert math.isclose(flux_input*10, dered_flux, rel_tol=1e-9)

    mag_g = _compute_asinh_mags(flux_input, 0)
    dered_mag_g = _compute_asinh_mags(dered_flux, 0)

    assert math.isclose(mag_g - 2.5, dered_mag_g, rel_tol=1e-8)
