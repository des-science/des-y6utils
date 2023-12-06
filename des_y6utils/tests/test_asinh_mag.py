from ..mdet import _compute_asinh_dered_mag, _compute_asinh_flux, _compute_asinh_mags


def test_compute_asinh_flux():
    # this test ensures the function computes magnitudes and fluxes correctly.
    flux_input = 10000
    mag_g = _compute_asinh_mags(flux_input, 0)
    flux_g = _compute_asinh_flux(mag_g, 0)

    assert flux_input == flux_g

def test_compute_asinh_dered_mag():
    flux_input = 10000
    mag_g = _compute_asinh_mags(flux_input, 0)
    dered_mag_g = _compute_asinh_dered_mag(flux_input, 0, 1/3.186)

    assert mag_g == dered_mag_g