import ngmix
from ngmix.admom import AdmomFitter
import galsim
import copy
import numpy as np

ALL_BAD = 2**0
BAD_BOX = 2**1
EMPTY_BOX = 2**2


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

    def _condition(b):
        return (
            (b["xmin"] != b["xmax"])
            and (b["xmin"] != b["xmax"])
            and np.any(bad_msk[b["ymin"]:b["ymax"]+1, b["xmin"]:b["xmax"]+1])
        )

    itr = 0
    while _condition(b):
        new_b = copy.deepcopy(b)
        curr_frac = np.mean(bad_msk[b["ymin"]:b["ymax"]+1, b["xmin"]:b["xmax"]+1])
        new_fracs = dict(
            xmin=np.mean(bad_msk[b["ymin"]:b["ymax"]+1, b["xmin"]+1:b["xmax"]+1]),
            xmax=np.mean(bad_msk[b["ymin"]:b["ymax"]+1, b["xmin"]:b["xmax"]+1-1]),
            ymin=np.mean(bad_msk[b["ymin"]+1:b["ymax"]+1, b["xmin"]:b["xmax"]+1]),
            ymax=np.mean(bad_msk[b["ymin"]:b["ymax"]+1-1, b["xmin"]:b["xmax"]+1]),
        )
        mval = np.nanmin(list(new_fracs.values()))
        for key, shift in zip(["ymax", "ymin", "xmax", "xmin"], [-1, 1, -1, 1]):
            if new_fracs[key] == mval:
                new_b[key] = b[key] + shift
                break

        if verbose:
            print("itr:", itr)
            print("    curr frac bad:", curr_frac)
            print(
                "    new fracs:",
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

    if b["xmin"] == b["xmax"] or b["ymin"] == b["ymax"]:
        flag |= EMPTY_BOX

    return b, flag


def _nanmad(x, axis=None):
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


def make_good_regions_for_piff_model(
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
    piff_kwargs = piff_kwargs or {}
    rng = np.random.RandomState(seed=seed)
    flags = 0

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
                x=xv, y=yv, chipnum=list(piff_mod.wcs.keys())[0],
                **piff_kwargs,
            )
            obs = ngmix.Observation(
                image=img.array,
                jacobian=ngmix.Jacobian(
                    y=img.center.y - img.bounds.ymin,
                    x=img.center.x - img.bounds.xmin,
                    wcs=img.wcs.local(
                        image_pos=galsim.PositionD(x=xv, y=yv)
                    ).jacobian(),
                )
            )
            try:
                res = AdmomFitter(
                    rng=rng
                ).go(obs, ngmix.moments.fwhm_to_T(1))
                t_arr[i, j] = res["T"]
            except Exception:
                pass

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
        t_std = _nanmad(t_arr)

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
