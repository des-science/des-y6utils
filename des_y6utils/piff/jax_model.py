try:
    # this may not work
    from jax.config import config
    config.update("jax_enable_x64", True)
except Exception:
    pass

import scipy.optimize
import jax.numpy as jnp
import itertools
import numpy as np
import galsim
import ngmix
import jax


def interp2d(
    x: jnp.ndarray,
    y: jnp.ndarray,
    xp: jnp.ndarray,
    yp: jnp.ndarray,
    zp: jnp.ndarray,
    fill_value: jnp.ndarray = None,
) -> jnp.ndarray:
    """
    WARNING: x and y appear swapped here!

    Bilinear interpolation on a grid.

    Args:
        x, y: 1D arrays of point at which to interpolate. Any out-of-bounds
            coordinates will be clamped to lie in-bounds.
        xp, yp: 1D arrays of points specifying grid points where function values
            are provided.
        zp: 2D array of function values. For a function `f(x, y)` this must
            satisfy `zp[i, j] = f(xp[i], yp[j])`

    Returns:
        1D array `z` satisfying `z[i] = f(x[i], y[i])`.

    The code in this function is used under MIT from
    https://github.com/adam-coogan/jaxinterp2d

    MIT License

    Copyright (c) [2021] [Adam Coogan]

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    if xp.ndim != 1 or yp.ndim != 1:
        raise ValueError("xp and yp must be 1D arrays")
    if zp.shape != (xp.shape + yp.shape):
        raise ValueError("zp must be a 2D array with shape xp.shape + yp.shape")

    ix = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, len(xp) - 1)
    iy = jnp.clip(jnp.searchsorted(yp, y, side="right"), 1, len(yp) - 1)

    # Using Wikipedia's notation (https://en.wikipedia.org/wiki/Bilinear_interpolation)
    z_11 = zp[ix - 1, iy - 1]
    z_21 = zp[ix, iy - 1]
    z_12 = zp[ix - 1, iy]
    z_22 = zp[ix, iy]

    z_xy1 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_11 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_21
    z_xy2 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_12 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_22

    z = (yp[iy] - y) / (yp[iy] - yp[iy - 1]) * z_xy1 + (y - yp[iy - 1]) / (
        yp[iy] - yp[iy - 1]
    ) * z_xy2

    if fill_value is not None:
        oob = (x < xp[0]) | (x > xp[-1]) | (y < yp[0]) | (y > yp[-1])
        z = jnp.where(oob, fill_value, z)

    return z


def _get_star_stamp_pos(s, img, wgt, bbox):
    # xint, yint is the pixel that contains the center
    # the -1 here is because piff works in one-indexed coordinates
    xint = int(np.floor(s.x - 1 + 0.5))
    yint = int(np.floor(s.y - 1 + 0.5))

    # the lower left corner of the stamp will be this many pixels away
    bbox_2 = (bbox - 1)//2

    # this is the offset to the center of the PSF from the stamp center
    # the -1 here is because piff works in one-indexed coordinates
    dx = s.x - 1 - xint
    dy = s.y - 1 - yint

    return dict(
        img=img[yint-bbox_2: yint+bbox_2+1, xint-bbox_2: xint+bbox_2+1].copy(),
        wgt=wgt[yint-bbox_2: yint+bbox_2+1, xint-bbox_2: xint+bbox_2+1].copy(),
        # the start of the stamp
        xstart=xint-bbox_2,
        ystart=yint-bbox_2,
        dim=bbox,
        # the -1 here is because piff works in one-indexed coordinates
        x=s.x - 1,
        y=s.y - 1,
        dx=dx,
        dy=dy,
        # these are the location of the PSf center in the stamp
        xcen=bbox_2 + dx,
        ycen=bbox_2 + dy,
    )


def get_star_obs(piff_mod, s, img, wgt, piff_prop, star_dim):
    if piff_prop:
        kwargs = {
            piff_prop: s.data.properties[piff_prop]
        }
    else:
        kwargs = {}
    sres = _get_star_stamp_pos(s, img, wgt, star_dim)
    sres.update(**kwargs)

    # the WCS is 1-indexed
    xv = sres["x"]+1
    yv = sres["y"]+1
    wcs = list(piff_mod.wcs.values())[0].local(
        image_pos=galsim.PositionD(x=xv, y=yv)
    ).jacobian()
    img = galsim.ImageD(sres["dim"], sres["dim"], wcs=wcs)
    _img = sres.pop("img")
    _wgt = sres.pop("wgt")
    nrm = _img.sum()
    nrm2 = nrm*nrm
    _img /= nrm
    _wgt *= nrm2

    star_obs = ngmix.Observation(
        image=_img,
        weight=_wgt,
        jacobian=ngmix.Jacobian(
            y=sres["ycen"],
            x=sres["xcen"],
            wcs=wcs,
        )
    )
    sres["local_wcs"] = wcs
    star_obs.update_meta_data(sres)

    # if a star is too high s/n, we make it lower
    if star_obs.get_s2n() > 100:
        nse = np.sqrt(np.sum(star_obs.image**2)) / 100
        star_obs.weight = np.ones_like(star_obs.weight) / nse**2

    return star_obs


def _build_star_xycol_vectors(stars):
    # these are all shape (n_stars,)
    x = jnp.array([star.meta["x"] for star in stars])
    y = jnp.array([star.meta["y"] for star in stars])
    col = jnp.array([star.meta["col"] for star in stars])
    return dict(x=x, y=y, col=col)


def _build_star_img_wgt_jac_arrays(stars):
    return dict(
        # these are shape (dim*dim, n_stars)
        img=jnp.array([star.image.ravel() for star in stars]).T,
        wgt=jnp.array([np.sqrt(star.weight.ravel()) for star in stars]).T,
        # these are all shape (n_stars,)
        dudx=jnp.array([star.meta["local_wcs"].dudx for star in stars]).T,
        dudy=jnp.array([star.meta["local_wcs"].dudy for star in stars]),
        dvdx=jnp.array([star.meta["local_wcs"].dvdx for star in stars]),
        dvdy=jnp.array([star.meta["local_wcs"].dvdy for star in stars]),
        xcen=jnp.array([star.meta["xcen"] for star in stars]),
        ycen=jnp.array([star.meta["ycen"] for star in stars]),
    )


def _build_star_poly_basis(
    *, x, y, col, colcen, colscale,
    xorder, colorder,
    xcen=1024, ycen=2048, xyscale=2048
):
    # the coordinates are scaled to a range that is roughly [-1, 1]
    # this means we can use a single regularization coeff later.
    xs = (x-xcen)/xyscale
    ys = (y-ycen)/xyscale
    cols = (col-colcen)/colscale

    # these terms are for baseline spatial variation
    arrs = [xs, ys]
    # nms = "xy"
    xy_arrs = [jnp.ones_like(xs)]
    for ln in range(1, xorder+1):
        for cmb in itertools.combinations_with_replacement([0, 1], ln):
            # print("".join(nms[i] for i in cmb))
            _r = 1
            for i in cmb:
                _r = _r * arrs[i]
            xy_arrs.append(_r)

    # these terms are for the variation of the dilation and shear
    # which has color dependence
    arrs = [cols, xs, ys]
    # nms = "cxy"
    colxy_arrs = []
    for ln in range(1, colorder+1):
        for cmb in itertools.combinations_with_replacement([0, 1, 2], ln):
            # the color is first and so if it is not in the combination,
            # we do not use it
            # these means all terms have at least one factor of the color in them
            if 0 not in cmb:
                continue
            # print("".join(nms[i] for i in cmb))
            _r = 1
            for i in cmb:
                _r = _r * arrs[i]
            colxy_arrs.append(_r)

    return dict(xy=jnp.vstack(xy_arrs), colxy=jnp.vstack(colxy_arrs))


def _build_fit_data(
    *, star_dim, model_dim, model_scale,
    stars, colname, colcen, colscale, init_scale,
    xy_order, colxy_order,
):
    # we are assembling a lot of metadata here
    fit_data = dict(
        star_dim=star_dim,
        model_dim=model_dim,
        stars=stars,
        colcen=colcen,
        colscale=colscale,
        model_scale=model_scale,
        n_stars=len(stars),
        colname=colname,
        init_scale=init_scale,
        verbose=False,
    )
    for star in stars:
        star.meta["col"] = star.meta[colname]
    fit_data["stars"] = stars
    fit_data.update(_build_star_xycol_vectors(stars))
    fit_data.update(_build_star_img_wgt_jac_arrays(stars))
    fit_data["n_stars"] = len(stars)

    # we build the polynomial basis here so we can extract the dimensions
    res = _build_star_poly_basis(
        x=fit_data["x"],
        y=fit_data["y"],
        col=fit_data["col"],
        colcen=fit_data["colcen"],
        colscale=fit_data["colscale"],
        xorder=xy_order,
        colorder=colxy_order,
    )

    # jax likes to complain unless you use .to_py() here
    # it tracks all data internally and this basically turns that off so it
    # ends up thinking these are unrelated variables
    fit_data["dim_xy"] = res["xy"].to_py().shape[0]
    fit_data["dim_colxy"] = res["colxy"].to_py().shape[0]

    n_params = (
        fit_data["model_dim"]*fit_data["model_dim"]*fit_data["dim_xy"]
        + 3*fit_data["dim_colxy"]
        + 2*len(stars)
    )
    fit_data["dof"] = len(stars) * star_dim**2 - n_params
    fit_data["n_params"] = n_params

    # done in numpy since a constant
    # we map every image to the uv plane and then do a baseline polynomial fit
    # these parameters are then used to seed the fit
    images = np.array([
        galsim.InterpolatedImage(
            galsim.ImageD(s.image),
            wcs=s.meta["local_wcs"],
            offset=(s.meta["dx"], s.meta["dy"]),
            flux=1,
            x_interpolant="linear",
        ).drawImage(
            nx=model_dim, ny=model_dim, scale=model_scale, method="no_pixel"
        ).array.ravel()
        for s in stars
    ])

    coeffs = res["xy"].T
    A = np.linalg.lstsq(coeffs, images, rcond=None)[0]
    fit_data["poly_guess"] = A.T.ravel()

    return fit_data


def _init_model_params(
    *, model_dim, rng, stars, model_scale, init_scale,
    poly_guess, dim_xy, dim_colxy, n_stars
):
    # the order here matches the order the parameters are stored in the array
    tot = model_dim*model_dim*dim_xy + 3*dim_colxy + 2*n_stars
    nse = rng.normal(size=tot, scale=init_scale)
    nn = model_dim**2
    # here we add in the polynomial guess computed in _build_fit_data
    nse[0:nn*dim_xy] += poly_guess
    # for the star centers we use 0.1 pixels always
    nse[-2*n_stars:] = rng.normal(size=2*n_stars, scale=0.1)
    return nse


def _extract_model_params(*, params, model_dim, dim_xy, dim_colxy):
    nxy = model_dim**2 * dim_xy
    xy_params = params[0:nxy]
    dilation_params = params[nxy:nxy+dim_colxy]
    g1_params = params[nxy+dim_colxy:nxy+dim_colxy*2]
    g2_params = params[nxy+dim_colxy*2:nxy+dim_colxy*3]

    return nxy, xy_params, dilation_params, g1_params, g2_params


# this one was complicated to figure out and tool some guess and check
# basically it lets jax do a bunch of array interpolation over more than one
# image all at once
_multi_interp2d = jax.vmap(interp2d, in_axes=(1, 1, None, None, 2), out_axes=-1)


def _get_inv_shear_mat(eta1, eta2):
    # this algorithm is lifted from galsim and recoded in jax
    # see https://github.com/GalSim-developers/GalSim/blob/
    #            7d6cefb142553d9c8413dd7386f080b0015799f2/galsim/shear.py#L339
    abseta2 = eta1**2 + eta2**2
    abseta = jnp.sqrt(abseta2)
    eta2g = jax.lax.select(
        abseta > 1.e-4,
        # true
        jnp.tanh(0.5*abseta)/abseta,
        # false
        0.5 + abseta2*((-1./24.) + abseta2*(1./240.)),
    )

    g1 = eta1 * eta2g
    g2 = eta2 * eta2g
    g2 = g1*g1 + g2*g2
    denom = jnp.sqrt(1-g2)
    return (
        (1+g1)/denom,
        g2/denom,
        g2/denom,
        (1-g1)/denom,
    )


def _predict_stars_lowlev(
    *, xy_params, dilation_params, g1_params, g2_params,
    star_xy, star_colxy, xcen, ycen, dudx, dudy, dvdx, dvdy,
    star_dim, model_dim, model_scale, dim_xy
):
    # this reshaping + matrix multiplication gets us an array that is
    # (model_dim, model_dim, n_stars)
    # this array is the uv image for each star computed from the polynomial basis
    uv_img = jnp.dot(xy_params.reshape(model_dim, model_dim, dim_xy), star_xy)

    # now we map star pixels back to uv coordinates using the WCS
    y, x = jnp.mgrid[0:star_dim, 0:star_dim]
    y = jnp.reshape(jnp.ravel(y), (star_dim*star_dim, 1))
    x = jnp.reshape(jnp.ravel(x), (star_dim*star_dim, 1))
    dy = y - ycen
    dx = x - xcen
    us = (dudx * dx + dudy * dy)
    vs = (dvdx * dx + dvdy * dy)

    # then we shear those coords
    # really we should be using the inverse shear here but that comes out in
    # wash anyways
    eta1 = jnp.dot(g1_params, star_colxy)
    eta2 = jnp.dot(g2_params, star_colxy)

    dudus, dudvs, dvdus, dvdvs = _get_inv_shear_mat(eta1, eta2)

    us = dudus * us + dudvs * vs
    vs = dvdus * us + dvdvs * vs

    # an we dilate them
    # note the inverse since we define the dilation as that of the model
    # the use of an exp here makes sure the dilation is alwasy > 0
    dilation = jnp.exp(jnp.dot(dilation_params, star_colxy))
    us = us/dilation
    vs = vs/dilation

    # these are the pixel index locations of the uv images above
    ui = jnp.arange(model_dim) - (model_dim-1)/2
    vi = jnp.arange(model_dim) - (model_dim-1)/2

    # we remove the model pixel scale from us, vs here to get indices
    # now we pass v,u instead of u,v since the interp function has indices
    # swapped relative to standard numpy conventions
    iimg = _multi_interp2d(vs/model_scale, us/model_scale, vi, ui, uv_img)
    return jnp.reshape(iimg, (star_dim*star_dim, -1))


def _predict_stars_impl(
    *,
    x,
    y,
    col,
    colcen,
    colscale,
    dudx,
    dudy,
    dvdx,
    dvdy,
    xcen,
    ycen,
    params,
    model_dim,
    dim_xy,
    dim_colxy,
    model_scale,
    star_dim,
    n_stars,
    use_cen,
    xy_order,
    colxy_order,
):
    nxy, xy_params, dilation_params, g1_params, g2_params = _extract_model_params(
        params=params,
        model_dim=model_dim,
        dim_xy=dim_xy,
        dim_colxy=dim_colxy,
    )
    sdx = params[nxy:nxy+n_stars]
    sdy = params[nxy+n_stars:nxy+2*n_stars]
    res = _build_star_poly_basis(
        x=x,
        y=y,
        col=col,
        colcen=colcen,
        colscale=colscale,
        xorder=xy_order,
        colorder=colxy_order,
    )

    pimg = _predict_stars_lowlev(
        xy_params=xy_params,
        dilation_params=dilation_params,
        g1_params=g1_params,
        g2_params=g2_params,
        star_xy=res["xy"],
        star_colxy=res["colxy"],
        xcen=jax.lax.select(use_cen == 1, xcen+sdx, xcen),
        ycen=jax.lax.select(use_cen == 0, ycen+sdy, ycen),
        dudx=dudx,
        dudy=dudy,
        dvdx=dvdx,
        dvdy=dvdy,
        star_dim=star_dim,
        model_dim=model_dim,
        model_scale=model_scale,
        dim_xy=dim_xy,
    )

    # this final image is of shape (star_dim*star_dim, n_stars)
    # we alwasy normalize the star images to unity
    return pimg/jnp.sum(pimg, axis=0)


# again more jax magic
# the jit call here compiles this code down to fast machine code
# the static argnums embeds those args as constants in the code
# this means if you make different models at different orders, jax will compile
# again
_predict_stars = jax.jit(
    _predict_stars_impl,
    static_argnames=(
        "model_dim", "star_dim", "dim_xy",
        "dim_colxy", "n_stars", "xy_order", "colxy_order",
    )
)


def _compute_chi2_per_dof_impl(
    params,
    x,
    y,
    col,
    dudx,
    dudy,
    dvdx,
    dvdy,
    xcen,
    ycen,
    img,
    wgt,
    colcen,  # 12th index
    colscale,
    model_dim,
    dim_xy,
    dim_colxy,
    model_scale,
    star_dim,
    dof,
    n_stars,
    inv_sigma_poly,
    xy_order,
    colxy_order,
):
    pimg = _predict_stars(
        x=x,
        y=y,
        col=col,
        colcen=colcen,
        colscale=colscale,
        dudx=dudx,
        dudy=dudy,
        dvdx=dvdx,
        dvdy=dvdy,
        xcen=xcen,
        ycen=ycen,
        params=params,
        model_dim=model_dim,
        dim_xy=dim_xy,
        dim_colxy=dim_colxy,
        model_scale=model_scale,
        star_dim=star_dim,
        n_stars=n_stars,
        use_cen=1,
        xy_order=xy_order,
        colxy_order=colxy_order,
    )
    resids = (pimg - img) * wgt
    return (
        jnp.sum(resids**2)
        # penalty on poly coeffs
        + jnp.sum(params[:-2*n_stars:]**2) * inv_sigma_poly**2
        # penalty on centers - the factor of 0.25 is 0.5**2 and means a Gaussian
        # of width half a pixel
        + jnp.sum(params[-2*n_stars:]**2)/0.25
    )/dof


# more jit with more static argnums
_compute_chi2_per_dof = jax.jit(
    _compute_chi2_per_dof_impl,
    static_argnums=tuple(range(12, 24)),
)

# this jax magic gets us a function that computes a value and gradient
_compute_chi2_per_dof_val_and_grad = jax.value_and_grad(_compute_chi2_per_dof)


class MattsPSFColorModel(object):
    """A very silly PSF model coded in jax.

    This model is a baseline level of polynomial variation plus a color and position
    dependent dilation and shear applied to that baseline model.

    The model is coded in jax and then autograd + L-BFGS-B is used to find the
    optimal location.

    Parameters
    ----------
    piff_model : piff model
        A piff model that is used to get the set of stars to fit and the WCS.
    image : array
        The image.
    weight : array
        The weight map.
    colname : str
        The name of the color in the piff star properties (e.g., "GI_COLOR").
    colcen : float
        The central/mean/median/default color for stars.
    colscale : float
        The approximate range of colors of the input stars.
    init_scale : float, optional
        The initial scale for seeding parameters with an RNG. Default it 1e-3.
    star_dim : int, optional
        The dimension of the star images used for the fit. Make sure it is odd.
        Default is 19.
    model_dim : int, optional
        The dimension of the star model in the uv-plane. Make sure it is odd.
        Default is 17.
    model_scale : float
        The pixel scale of the uv=plane model. Default is 0.3 arcseconds.
    seed : int or None, optional
        The RNG seed.
    sigma_poly : float or None, optional
        The optional amount of regularization of the polynomial coeffs. This
        value corresponds to add sum(polycoeffs**2)/sigma_poly**2 to the chi2.
        The default of None is equivalent to sigma_poly = np.inf.
    xy_order : int, optional
        The default order of the x-y polynomial basis for the uv-model. Default is 2.
    colxy_order : int, optional
        The default order of the color-x-y polynomial basis for the color-dependent
        parts of the model. Default is 2.
    """
    def __init__(
        self,
        *,
        piff_model, image, weight,
        colname, colcen, colscale,
        init_scale=0.001,
        star_dim=19, model_dim=17,
        model_scale=0.3,
        seed=None,
        sigma_poly=None,
        xy_order=2,
        colxy_order=2,
    ):
        self.image = image
        self.weight = weight
        self.piff = piff_model
        stars = [
            get_star_obs(self.piff, s, self.image, self.weight, colname, star_dim)
            for s in self.piff.stars
        ]

        self.fit_data = _build_fit_data(
            star_dim=star_dim,
            model_dim=model_dim,
            stars=stars,
            colname=colname,
            colcen=colcen,
            colscale=colscale,
            model_scale=model_scale,
            init_scale=init_scale,
            xy_order=xy_order,
            colxy_order=colxy_order,
        )
        self.stars = self.fit_data["stars"]

        if sigma_poly is None:
            sigma_poly = np.inf
        self.fit_data["inv_sigma_poly"] = 1.0 / sigma_poly
        self.fit_data["xy_order"] = xy_order
        self.fit_data["colxy_order"] = colxy_order

        self.rng = np.random.RandomState(seed=seed)
        self.wcs = {0: None}
        self.dof = self.fit_data["dof"]

    def fit(self, **kwargs):
        """Fit the PSF model using L-BFGS-B.

        This function sets the `fit_res` attribute which has the parameters
        as `fit_res.x`.

        Parameters
        ----------
        verbose : bool, optional
            If True, print the model iterations. Default is False.
        **kwargs : optional
            Any extra keyword arguments are passed as the `options` keyword
            to `scipy.optimize.minimize`.
        """
        if "verbose" in kwargs:
            kwargs.pop("verbose")
            iprint = 98
        else:
            iprint = -1
        args = tuple([
            self.fit_data[key]
            for key in [
                    "x",
                    "y",
                    "col",
                    "dudx",
                    "dudy",
                    "dvdx",
                    "dvdy",
                    "xcen",
                    "ycen",
                    "img",
                    "wgt",
                    "colcen",
                    "colscale",
                    "model_dim",
                    "dim_xy",
                    "dim_colxy",
                    "model_scale",
                    "star_dim",
                    "dof",
                    "n_stars",
                    "inv_sigma_poly",
                    "xy_order",
                    "colxy_order",
            ]
        ])

        # the float64 thing is weird here but basically needed to interface
        # with some fortran code
        x0 = self.get_init_params().astype(np.float64)

        def _fun(params, *args):
            v, g = _compute_chi2_per_dof_val_and_grad(params, *args)
            return v.to_py(), g.to_py().astype(np.float64)

        opts = {"disp": None, "iprint": iprint}
        if kwargs:
            opts.update(kwargs)
        else:
            opts["ftol"] = 1e-5

        self.fit_res = scipy.optimize.minimize(
            _fun,
            x0,
            args=args,
            method="L-BFGS-B",
            jac=True,
            options=opts,
        )

        self.chi2 = (
            self.fit_res.fun * self.dof
            - (
                np.sum(self.fit_res.x[:-2*self.fit_data["n_stars"]]**2)
                * self.fit_data["inv_sigma_poly"]**2
            )
            - np.sum(self.fit_res.x[-2*self.fit_data["n_stars"]:]**2)/0.25
        )

    def get_init_params(self, init_scale=None):
        """Get a set of randomly drawn initial parameters.

        Parameters
        ----------
        init_scale : float, optional
            If not None, this scale is used to seed the initial parameters via
            drawing Gaussian random variables with this standard deviation. Default
            of None indicates to use the value provide when the model class was
            instantiated.

        Returns
        -------
        params : float
            The parameters.
        """
        return _init_model_params(
                model_dim=self.fit_data["model_dim"],
                rng=self.rng,
                stars=self.fit_data["stars"],
                model_scale=self.fit_data["model_scale"],
                init_scale=init_scale or self.fit_data["init_scale"],
                poly_guess=self.fit_data["poly_guess"],
                dim_xy=self.fit_data["dim_xy"],
                dim_colxy=self.fit_data["dim_colxy"],
                n_stars=self.fit_data["n_stars"],
            )

    def get_dilation_eta1eta2(self, x, y, params=None, **kwargs):
        """Get the dilation, eta1 and eta2 associated with a point.

        Parameters
        ----------
        x : float
            The x location on CCD in zero-indexed pixel centered coordinates.
        y : float
            The y location on CCD in zero-indexed pixel centered coordinates.
        params : array
            The parameters to use. If None, the parameters from the fit are used.
            An error is raise if the fit has not yet been done.
        **kwargs : ignored
            Ignored but here to enable compatibility with piff.

        Returns
        -------
        dilation : float
            The dilation scale.
        eta1, eta2 : float
            The two shear components in conformal space.
        """
        if params is None:
            if not hasattr(self, "fit_res"):
                raise RuntimeError("call fit!")
            params = self.fit_res.x

        if self.fit_data["colname"] in kwargs:
            col = kwargs[self.fit_data["colname"]]
        else:
            col = self.fit_data["colcen"]

        res = _build_star_poly_basis(
            x=np.atleast_1d(x),
            y=np.atleast_1d(y),
            col=np.atleast_1d(col),
            colcen=self.fit_data["colcen"],
            colscale=self.fit_data["colscale"]
        )

        _, _, dilation_params, g1_params, g2_params = _extract_model_params(
            params=params,
            model_dim=self.fit_data["model_dim"],
            dim_xy=self.fit_data["dim_xy"],
            dim_colxy=self.fit_data["dim_colxy"],
        )

        dilation = np.exp(np.dot(dilation_params, res["colxy"]))[0]
        eta1 = np.dot(g1_params, res["colxy"])[0]
        eta2 = np.dot(g2_params, res["colxy"])[0]
        return dilation, eta1, eta2

    def draw(self, x, y, offset=None, params=None, **kwargs):
        """Draw the model.

        Parameters
        ----------
        x : float
            The x location on CCD in 1-indexed pixel centered coordinates.
        y : float
            The y location on CCD in 1-indexed pixel centered coordinates.
        offset : tuple of floats
            If given, draw the model at this offset relative to the center of the
            stamp. If None, the model is drawn at the exact offset implied by the
            request (x,y) location.
        params : array
            The parameters to use. If None, the parameters from the fit are used.
            An error is raise if the fit has not yet been done.
        **kwargs : ignored
            Ignored but here to enable compatibility with piff.

        Returns
        -------
        psf : array
            The PSF model as a galsim ImageD.
        """
        if self.fit_data["colname"] in kwargs:
            col = kwargs[self.fit_data["colname"]]
        else:
            col = self.fit_data["colcen"]

        if params is None:
            if not hasattr(self, "fit_res"):
                raise RuntimeError("call fit!")
            params = self.fit_res.x

        x -= 1
        y -= 1
        wcs = list(self.piff.wcs.values())[0].local(
            image_pos=galsim.PositionD(x=x+1, y=y+1)
        ).jacobian()

        scen = (self.fit_data["star_dim"]-1)/2

        if offset is None:
            offset = [
                x - np.floor(x+0.5),
                y - np.floor(y+0.5),
            ]

        _img = _predict_stars(
                x=jnp.array([x]),
                y=jnp.array([y]),
                col=jnp.array([col]),
                colcen=self.fit_data["colcen"],
                colscale=self.fit_data["colscale"],
                dudx=jnp.array([wcs.dudx]),
                dudy=jnp.array([wcs.dudy]),
                dvdx=jnp.array([wcs.dvdx]),
                dvdy=jnp.array([wcs.dvdy]),
                xcen=jnp.array([scen+offset[0]]),
                ycen=jnp.array([scen+offset[1]]),
                params=params,
                model_dim=self.fit_data["model_dim"],
                dim_xy=self.fit_data["dim_xy"],
                dim_colxy=self.fit_data["dim_colxy"],
                model_scale=self.fit_data["model_scale"],
                star_dim=self.fit_data["star_dim"],
                n_stars=1,
                use_cen=0,
                xy_order=self.fit_data["xy_order"],
                colxy_order=self.fit_data["colxy_order"],
            )[:, 0].to_py().reshape(
                (self.fit_data["star_dim"], self.fit_data["star_dim"])
            )
        return galsim.ImageD(
            _img/np.sum(_img),
            wcs=wcs,
        )
