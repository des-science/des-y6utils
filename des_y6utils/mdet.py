import os
import subprocess
from functools import lru_cache

import numpy as np

import healsparse


def make_mdet_cuts(data, version, verbose=False):
    """A function to make the standard metadetection cuts.

    Note that this function will download a ~100 MB healsparse mask to
    the environment variable `MEDS_DIR`. If no such environment variable
    is found, a directory called `MEDS_DIR` will be made in the users
    HOME area.

    Parameters
    ----------
    data : np.ndarray
        The structured array of data to be cut.
    version : str or int
        The version of the cuts. The versions are whole integers.
    verbose : bool, optional
        If True, print info as cuts are being made.
        Default is False.

    Returns
    -------
    msk : np.ndarray of bool
        A boolean array with the cuts. To cut the data, use `data[msk]`.
    """
    if str(version) == "1":
        return _make_mdet_cuts_v1(data, verbose=verbose)
    elif str(version) == "2":
        return _make_mdet_cuts_v2(data, verbose=verbose)
    else:
        raise ValueError("the mdet cut version '%r' is not recognized!" % version)


def _make_mdet_cuts_raw_v12(d, verbose=False):
    """The raw v1 cuts come from extensive analysis over summer 2022. They
    reflect a first-pass at a consensus set of cuts.

    components/comments

      - We use wmom for shears and pgauss for fluxes. The weighted moments
        appear to have less noise in terms of precision on the mean shear.
        The pgauss fluxes remove the effects of the PSF on the fluxes to first
        order and match the aperture for colors across bands.
      - We use a cut in the pgauss T-Terr plane. This cut removes junk detections
        near the wings of stars. For this cut we require pgauss_T_flags == 0 as well.
      - We use a signal to noise cut of 10 in wmom.
      - We use an error dependent cut in the size ratio. This ensures that as the noise
        increases we move the size cut higher to eliminate stellar contamination.
      - We use "gold-inspired" cuts for crazy colors.
      - We cut especially faint objects in each band.
    """

    mag_g = _compute_asinh_mags(d["pgauss_band_flux_g"], 0)
    mag_r = _compute_asinh_mags(d["pgauss_band_flux_r"], 1)
    mag_i = _compute_asinh_mags(d["pgauss_band_flux_i"], 2)
    mag_z = _compute_asinh_mags(d["pgauss_band_flux_z"], 3)
    gmr = mag_g - mag_r
    rmi = mag_r - mag_i
    imz = mag_i - mag_z

    msk = np.ones(d.shape[0]).astype(bool)

    potential_flag_columns = [
        "psfrec_flags",
        "wmom_flags",
        "pgauss_T_flags",
        "pgauss_band_flux_flags_g",
        "pgauss_band_flux_flags_r",
        "pgauss_band_flux_flags_i",
        "pgauss_band_flux_flags_z",
        "mask_flags",
    ]
    for col in potential_flag_columns:
        if col in d.dtype.names:
            msk &= (d[col] == 0)
            if verbose:
                print("did cut " + col, np.sum(msk))

    if "shear_bands" in d.dtype.names:
        msk &= (d["shear_bands"] == "123")
        if verbose:
            print("did cut shear_bands", np.sum(msk))

    if "pgauss_s2n" in d.dtype.names:
        msk &= (d["pgauss_s2n"] > 5)
        if verbose:
            print("did cut pgauss_s2n", np.sum(msk))

    # now do the rest
    msk &= (
        (d["wmom_s2n"] > 10)
        & (d["mfrac"] < 0.1)
        & (np.abs(gmr) < 5)
        & (np.abs(rmi) < 5)
        & (np.abs(imz) < 5)
        & np.isfinite(mag_g)
        & np.isfinite(mag_r)
        & np.isfinite(mag_i)
        & np.isfinite(mag_z)
        & (mag_g < 26.5)
        & (mag_r < 26.5)
        & (mag_i < 26.2)
        & (mag_z < 25.6)
        & (d["pgauss_T"] < (1.9 - 2.8*d["pgauss_T_err"]))
        & (
            d["wmom_T_ratio"] >= np.maximum(
                1.2,
                (1.0 + 3.0*d["wmom_T_err"]/d["wmom_psf_T"])
            )
        )
    )
    if verbose:
        print("did mdet cuts", np.sum(msk))

    return msk


def _make_mdet_cuts_v1(d, verbose=False):

    msk = _make_mdet_cuts_raw_v12(d, verbose=verbose)

    # apply the mask
    hmap = _read_hsp_mask("y6-combined-hleda-gaiafull-hsmap16384-nomdet.fits")
    in_footprint = hmap.get_values_pos(d["ra"], d["dec"], valid_mask=True)
    msk &= in_footprint
    if verbose:
        print("did mask cuts", np.sum(msk))

    return msk


def _make_mdet_cuts_v2(d, verbose=False):

    msk = _make_mdet_cuts_raw_v12(d, verbose=verbose)

    # apply the mask
    hmap = _read_hsp_mask(
        "y6-combined-hleda-gaiafull-des-stars-hsmap16384-nomdet-v2.fits"
    )
    in_footprint = hmap.get_values_pos(d["ra"], d["dec"], valid_mask=True)
    msk &= in_footprint
    if verbose:
        print("did mask cuts", np.sum(msk))

    return msk


def _compute_asinh_mags(flux, i):
    """This function and coefficients are from from Eli. Ask him.

    Parameters
    ----------
    flux : float or np.ndarray
        The flux.
    i : int
        The index of the band in griz (i.e., 0 for g, 1 for r, 2 for i, 3 for z).

    Returns
    -------
    mag : float or np.ndarray
        The asinh magnitude for the flux.
    """
    zp = 30.0
    # array is griz
    b_array = np.array([3.27e-12, 4.83e-12, 6.0e-12, 9.0e-12])
    bscale = np.array(b_array) * 10.**(zp / 2.5)
    mag = (
        2.5 * np.log10(1.0 / b_array[i])
        - np.arcsinh(0.5 * flux / bscale[i]) / (0.4 * np.log(10.0))
    )
    # mag_err = (
    #     2.5 * fluxerr / (
    #         2.0 * bscale[i] * np.log(10.0)
    #         * np.sqrt(1.0 + (0.5 * flux / bscale[i])**2.)
    #     )
    # )
    return mag


@lru_cache
def _read_hsp_mask(fname):
    mpth = _get_mask_path(fname)
    return healsparse.HealSparseMap.read(mpth)


def _get_mask_path(fname):
    # get or make meds dir
    meds_dir = os.environ.get("MEDS_DIR", None)
    if meds_dir is None:
        meds_dir = os.path.expandvars(
            os.path.expanduser("~/MEDS_DIR")
        )
        os.makedirs(meds_dir, exist_ok=True)

    # download if needed
    fpth = os.path.join(meds_dir, fname)
    if not os.path.exists(fpth):
        _download_fname_from_bnl(fpth)

    return fpth


def _download_fname_from_bnl(fpth):
    fdir, fname = os.path.split(fpth)

    wget_res = subprocess.run("which wget", shell=True, capture_output=True)
    curl_res = subprocess.run("which curl", shell=True, capture_output=True)

    bnl = "https://www.cosmo.bnl.gov/www/esheldon/data/y6-healsparse"
    if wget_res.returncode == 0:
        subprocess.run(
            "cd %s && wget %s/%s" % (
                fdir, bnl, fname,
            ),
            shell=True,
            check=True,
            capture_output=True,
        )
    elif curl_res.returncode == 0:
        subprocess.run(
            "cd %s && curl -L %s/%s --output %s" % (
                fdir, bnl, fname, fname,
            ),
            shell=True,
            check=True,
            capture_output=True,
        )
    else:
        raise RuntimeError(
            "Could not download mask '%s' from BNL due "
            "to wget or curl missing!" % fname,
        )
