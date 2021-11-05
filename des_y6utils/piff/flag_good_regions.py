import ngmix
from ngmix.admom import AdmomFitter
import galsim
import copy
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

ALL_BAD = 2**0
BAD_BOX = 2**1


def _make_a_good_box(bad_msk, verbose=False):
    flag = 0

    b = dict(
        xmax=bad_msk.shape[1]-1,
        xmin=0,
        ymax=bad_msk.shape[0]-1,
        ymin=0,
    )
    if not np.any(bad_msk[b["ymin"]:b["ymax"]+1, b["xmin"]:b["xmax"]+1]):
        return b, flag

    if np.all(bad_msk[b["ymin"]:b["ymax"]+1, b["xmin"]:b["xmax"]+1]):
        flag |= ALL_BAD
        return b, flag

    def _is_ok_range(b):
        return (
            (b["xmin"] <= b["xmax"])
            and (b["ymin"] <= b["ymax"])
        )

    def _condition(b):
        return (
            (b["xmin"] <= b["xmax"])
            and (b["ymin"] <= b["ymax"])
            and np.any(bad_msk[b["ymin"]:b["ymax"]+1, b["xmin"]:b["xmax"]+1])
        )

    itr = 0
    while _condition(b) and itr < 10_000:
        new_b = copy.deepcopy(b)
        curr_frac = np.sum(bad_msk[b["ymin"]:b["ymax"]+1, b["xmin"]:b["xmax"]+1])
        new_fracs = {}
        for key, shift in zip(["ymax", "ymin", "xmax", "xmin"], [-1, 1, -1, 1]):
            _new_b = copy.deepcopy(b)
            _new_b[key] = b[key] + shift
            if _is_ok_range(_new_b):
                new_fracs[key] = np.sum(bad_msk[
                    _new_b["ymin"]:_new_b["ymax"]+1,
                    _new_b["xmin"]:_new_b["xmax"]+1]
                )
            else:
                new_fracs[key] = np.nan

        if np.all(np.isnan(list(new_fracs.values()))):
            flag |= BAD_BOX
            break

        mval = np.nanmin(list(new_fracs.values()))
        for key, shift in zip(["ymax", "ymin", "xmax", "xmin"], [-1, 1, -1, 1]):
            _new_b = copy.deepcopy(b)
            _new_b[key] = b[key] + shift
            if new_fracs[key] == mval and _is_ok_range(_new_b):
                new_b[key] = b[key] + shift
                break

        if verbose:
            print("itr:", itr)
            print("    curr # bad:", curr_frac)
            print(
                "    new # bad:",
                " ".join("%s: %0.3f" % (k, v) for k, v in new_fracs.items()),
            )
            print("    new min val:", mval)
            print(
                "    edge|old|new: ",
                " ".join("%s|%s|%s" % (k, b[k], new_b[k]) for k in b),
            )
        if new_b == b:
            flag |= BAD_BOX
            break

        b = new_b
        itr += 1

    if (
        b["xmin"] > b["xmax"]
        or b["ymin"] > b["ymax"]
        or np.any(bad_msk[b["ymin"]:b["ymax"]+1, b["xmin"]:b["xmax"]+1])
    ):
        flag |= BAD_BOX

    return b, flag


def nanmad(x, axis=None):
    """
    median absolute deviation - scaled like a standard deviation

        mad = 1.4826*median(|x-median(x)|)

    Parameters
    ----------
    x: array-like
        array to take MAD of
    axis : {int, sequence of int, None}, optional
        `axis` keyword for

    Returns
    -------
    mad: float
        MAD of array x
    """
    return 1.4826*np.nanmedian(np.abs(x - np.nanmedian(x, axis=axis)), axis=axis)


def _rescale_b(b, grid_size):
    b["xmin"] = b["xmin"] * grid_size
    b["xmax"] = (b["xmax"] + 1) * grid_size
    b["ymin"] = b["ymin"] * grid_size
    b["ymax"] = (b["ymax"] + 1) * grid_size
    return b


def measure_t_grid_for_piff_model(
    piff_mod, piff_kwargs, grid_size=128, seed=None
):
    """Make a bounding box of good regions for a given Piff model.

    Parameters
    ----------
    piff_mod : piff Model
        The Piff model read via `piff.read`.
    piff_kwargs : dict, optional
        Any extra keyword arguments to pass to `piff_mod.draw`. Typically these
        might include things like `{"GI_COLOR": 0.61}`.
    grid_size : int, optional
        The grid size to construct the map. Must divide 2048 and 4096 evenly.
        Default is 128.
    seed : int, optional
        The seed to use for the RNG fed to ngmix. Default of None will seed the
        RNG on-the-fly.

    Returns
    -------
    t_arr : array
        The map of T over the model with one cell per `grid_size`.
    """
    piff_kwargs = piff_kwargs or {}
    rng = np.random.RandomState(seed=seed)

    if (
        ((4096//grid_size) * grid_size != 4096)
        or ((2048//grid_size) * grid_size != 2048)
    ):
        raise RuntimeError("The grid size must evenly divide 4096 and 2048.")

    delta_to_pix = np.floor((grid_size-1)/2 + 0.5)
    y = np.arange(4096//grid_size)*grid_size + delta_to_pix
    x = np.arange(2048//grid_size)*grid_size + delta_to_pix

    t_arr = np.zeros((y.shape[0], x.shape[0])) + np.nan
    for i, yv in enumerate(y):
        for j, xv in enumerate(x):
            img = piff_mod.draw(
                x=xv+1, y=yv+1, chipnum=list(piff_mod.wcs.keys())[0],
                **piff_kwargs,
            )
            obs = ngmix.Observation(
                image=img.array,
                jacobian=ngmix.Jacobian(
                    y=img.center.y - img.bounds.ymin,
                    x=img.center.x - img.bounds.xmin,
                    wcs=img.wcs.local(
                        image_pos=galsim.PositionD(x=xv+1, y=yv+1)
                    ).jacobian(),
                )
            )
            try:
                res = AdmomFitter(
                    rng=rng
                ).go(obs, ngmix.moments.fwhm_to_T(1))
                if res["flags"] == 0:
                    t_arr[i, j] = res["T"]
            except Exception:
                pass

    return t_arr


def _get_star_stamp_pos(s, img, wgt):
    xint = int(np.floor(s.x - 1 + 0.5))
    yint = int(np.floor(s.y - 1 + 0.5))
    bbox = 17
    bbox_2 = (bbox - 1)//2

    return dict(
        img=img[yint-bbox_2: yint+bbox_2+1, xint-bbox_2: xint+bbox_2+1],
        wgt=wgt[yint-bbox_2: yint+bbox_2+1, xint-bbox_2: xint+bbox_2+1],
        xstart=xint-bbox_2,
        ystart=yint-bbox_2,
        dim=bbox,
        x=s.x - 1,
        y=s.y - 1,
    )


def _get_star_piff_obs(piff_mod, s, img, wgt, piff_prop):

    if piff_prop:
        kwargs = {
            piff_prop: s.data.properties[piff_prop]
        }
    else:
        kwargs = {}
    sres = _get_star_stamp_pos(s, img, wgt)

    xv = sres["x"]+1
    yv = sres["y"]+1
    wcs = list(piff_mod.wcs.values())[0].local(
        image_pos=galsim.PositionD(x=xv, y=yv)
    ).jacobian()
    img = galsim.ImageD(sres["dim"], sres["dim"], wcs=wcs)
    cen = (
        sres["x"] - sres["xstart"] + 1,
        sres["y"] - sres["ystart"] + 1,
    )
    img = piff_mod.draw(
        x=xv, y=yv, chipnum=list(piff_mod.wcs.keys())[0],
        image=img, center=cen, **kwargs,
    )
    model_obs = ngmix.Observation(
        image=img.array,
        jacobian=ngmix.Jacobian(
            y=cen[1]-1,
            x=cen[0]-1,
            wcs=wcs,
        )
    )
    star_obs = ngmix.Observation(
        image=sres["img"],
        weight=sres["wgt"],
        jacobian=ngmix.Jacobian(
            y=cen[1]-1,
            x=cen[0]-1,
            wcs=wcs,
        )
    )
    return model_obs, star_obs, sres


def measure_star_t_for_piff_model(piff_mod, img, wgt, piff_prop=None, seed=None):
    """Make a bounding box of good regions for a given Piff model.

    Parameters
    ----------
    piff_mod : piff Model
        The Piff model read via `piff.read`.
    img : array
        The image.
    wgt : array
        The weight
    piff_prop : str, optional
        The name of the piff property to use. Should be one of "GI_COLOR" or "IZ_COLOR"
        for Y6.
    seed : int, optional
        The seed to use for the RNG fed to ngmix. Default of None will seed the
        RNG on-the-fly.

    Returns
    -------
    data : array
        An array with columns x, y, t for the stars.
    """
    piff_prop = piff_prop or {}
    rng = np.random.RandomState(seed=seed)

    nd = len(piff_mod.stars)
    data = np.zeros(nd, dtype=[("x", "f8"), ("y", "f8"), ("t", "f8")])
    for col in data.dtype.names:
        data[col] += np.nan

    for i, s in enumerate(piff_mod.stars):

        mobs, sobs, sres = _get_star_piff_obs(piff_mod, s, img, wgt, piff_prop)

        try:
            res = ngmix.admom.AdmomFitter(
                rng=rng
            ).go(mobs, ngmix.moments.fwhm_to_T(1))
            if res["flags"] == 0:
                data["x"][i] = sres["x"]
                data["y"][i] = sres["y"]
                data["t"][i] = res["T"]
        except Exception:
            pass

    return data


def map_star_t_to_grid(star_data, grid_size=128, degree=2):
    """Fit star T data to a polynomial and evaluate on a grid.

    Parameters
    ----------
    star_data : array
        The array pf star data from `measure_star_t_for_piff_model`.
    grid_size : int, optional
        The grid size to construct the map. Must divide 2048 and 4096 evenly.
        Default is 128.
    degree : int, optional
        The degree of the polynomial.

    Returns
    -------
    tg : array
        The array of T values from evaluating the git on a grid.
    """
    # see this blog post
    # https://towardsdatascience.com/polynomial-regression-with-scikit-learn-what-you-should-know-bed9d3296f2
    polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polyreg.fit(
        np.array([star_data["x"], star_data["y"]]).T,
        np.array(star_data["t"]),
    )

    if (
        ((4096//grid_size) * grid_size != 4096)
        or ((2048//grid_size) * grid_size != 2048)
    ):
        raise RuntimeError("The grid size must evenly divide 4096 and 2048.")

    delta_to_pix = np.floor((grid_size-1)/2 + 0.5)
    y, x = np.mgrid[0:4096:grid_size, 0:2048:grid_size] + delta_to_pix
    tg = polyreg.predict(np.array([x.ravel(), y.ravel()]).T)
    tg = tg.reshape(x.shape)

    return tg


def make_good_regions_for_piff_model_gal_grid(
    piff_mod, piff_kwargs=None, grid_size=128, seed=None, verbose=False,
    any_bad_thresh=25, flag_bad_thresh=15,
):
    """Make a bounding box of good regions for a given Piff model.

    Parameters
    ----------
    piff_mod : piff Model
        The Piff model read via `piff.read`.
    piff_kwargs : dict, optional
        Any extra keyword arguments to pass to `piff_mod.draw`. Typically these
        might include things like `{"GI_COLOR": 0.61}`.
    grid_size : int, optional
        The grid size to construct the map. Must divide 2048 and 4096 evenly.
        Default is 128.
    seed : int, optional
        The seed to use for the RNG fed to ngmix. Default of None will seed the
        RNG on-the-fly.
    verbose : bool, optional
        If True, print some stats as the code is running. Default of False makes
        the code silent.
    any_bad_thresh : float, optional
        The threshold used to figure out if any region of the CCD is bad. Models
        with any regions where |t - t_mn| > any_bad_thresh * t_std are considered
        bad. The default of 25 appears to account for PSF size variation while also
        flagging weird models.
    flag_bad_thresh : float, optional
        The threshold used to mark good versus bad regions. Any region of the model
        where |t - t_mn| > any_bad_thresh * t_std is marked as bad if any
        region also exceeds the `any_bad_thresh`. Default of 15 accounts for
        model size variation while also avoiding the very bad regions.

    Returns
    -------
    data : dict
        A dictionary with the following keys:

            flags : int
                The flags value. If non-zero, the entire model is flagged.
            t_mn : float
                The median T over the whole model.
            t_std : float
                The median absolute deviation over the whole model.
            t_arr : array
                The map of T over the model with one cell per `grid_size`.
            bbox : dict
                A dictionary with keys xmin, xmax, ymin, and ymax. Any model
                whose center falls within this box is considered OK.
    """
    t_arr = measure_t_grid_for_piff_model(
        piff_mod, piff_kwargs, grid_size=grid_size, seed=seed
    )
    flags = 0

    b = dict(
        xmax=t_arr.shape[1]-1,
        xmin=0,
        ymax=t_arr.shape[0]-1,
        ymin=0,
    )

    if not np.any(np.isfinite(t_arr)):
        flags |= ALL_BAD
        t_mn = np.nan
        t_std = np.nan

    if flags == 0:
        t_mn = np.nanmedian(t_arr)
        t_std = nanmad(t_arr)

        any_very_bad = (
            (~np.isfinite(t_arr))
            | (np.abs(t_arr - t_mn) > any_bad_thresh * t_std)
        )
        if np.any(any_very_bad):
            some_bad = (
                (~np.isfinite(t_arr))
                | (np.abs(t_arr - t_mn) > flag_bad_thresh * t_std)
            )
            b, flag = _make_a_good_box(some_bad, verbose=verbose)
            flags |= flag

    return dict(
        flags=flags,
        t_mn=t_mn,
        t_std=t_std,
        t_arr=t_arr,
        bbox=_rescale_b(b, grid_size),
    )


def make_good_regions_for_piff_model_star_and_gal_grid(
    piff_mod, img, wgt, piff_kwargs=None, grid_size=128, seed=None, verbose=False,
    any_bad_thresh=5, flag_bad_thresh=5, degree=2,
):
    """Make a bounding box of good regions for a given Piff model.

    Parameters
    ----------
    piff_mod : piff Model
        The Piff model read via `piff.read`.
    img : array
        The image.
    wgt : array
        The weight.
    piff_kwargs : dict, optional
        Any extra keyword arguments to pass to `piff_mod.draw`. Typically these
        might include things like `{"GI_COLOR": 0.61}`.
    grid_size : int, optional
        The grid size to construct the map. Must divide 2048 and 4096 evenly.
        Default is 128.
    seed : int, optional
        The seed to use for the RNG fed to ngmix. Default of None will seed the
        RNG on-the-fly.
    verbose : bool, optional
        If True, print some stats as the code is running. Default of False makes
        the code silent.
    any_bad_thresh : float, optional
        The threshold used to figure out if any region of the CCD is bad. Models
        with any regions where |t - t_mn| > any_bad_thresh * t_std are considered
        bad. The default of 25 appears to account for PSF size variation while also
        flagging weird models.
    flag_bad_thresh : float, optional
        The threshold used to mark good versus bad regions. Any region of the model
        where |t - t_mn| > any_bad_thresh * t_std is marked as bad if any
        region also exceeds the `any_bad_thresh`. Default of 15 accounts for
        model size variation while also avoiding the very bad regions.
    degree : int, optional
        The degree of the polynomial.

    Returns
    -------
    data : dict
        A dictionary with the following keys:

            flags : int
                The flags value. If non-zero, the entire model is flagged.
            t_star : float
                The star T map.
            t_gal : float
                The gal T map.
            bbox : dict
                A dictionary with keys xmin, xmax, ymin, and ymax. Any model
                whose center falls within this box is considered OK.
    """
    flags = 0

    tg_arr = measure_t_grid_for_piff_model(
        piff_mod, piff_kwargs, grid_size=grid_size, seed=seed
    )
    data = measure_star_t_for_piff_model(
        piff_mod, img, wgt, piff_prop=list(piff_kwargs.keys())[0], seed=seed,
    )

    if np.all(np.isnan(data["t"])):
        flags |= ALL_BAD

    if flags == 0:
        msk = np.isfinite(data["t"])

        ts_arr = map_star_t_to_grid(
            data[msk], grid_size=grid_size, degree=degree,
        )
    else:
        ts_arr = None

    b = dict(
        xmax=tg_arr.shape[1]-1,
        xmin=0,
        ymax=tg_arr.shape[0]-1,
        ymin=0,
    )

    if flags == 0 and (not np.any(np.isfinite(tg_arr))) or np.any(np.isnan(ts_arr)):
        flags |= ALL_BAD

    if flags == 0:
        tdiff = tg_arr - ts_arr
        t_mn = np.nanmedian(tdiff)
        t_std = nanmad(tdiff)

        any_very_bad = (
            (~np.isfinite(tdiff))
            | (np.abs(tdiff - t_mn) > any_bad_thresh * t_std)
        )
        if np.any(any_very_bad):
            some_bad = (
                (~np.isfinite(tg_arr))
                | (np.abs(tdiff - t_mn) > flag_bad_thresh * t_std)
            )
            b, flag = _make_a_good_box(some_bad, verbose=verbose)
            flags |= flag

    return dict(
        flags=flags,
        t_star=ts_arr,
        t_gal=tg_arr,
        bbox=_rescale_b(b, grid_size),
    )
