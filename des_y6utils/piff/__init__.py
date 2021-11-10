# flake8: noqa
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("des_y6utils")
except PackageNotFoundError:
    # package is not installed
    pass

from .flag_good_regions import (
    make_good_regions_for_piff_model_gal_grid,
    make_good_regions_for_piff_model_star_and_gal_grid,
    nanmad,
    measure_t_grid_for_piff_model,
    measure_star_t_for_piff_model,
    map_star_t_to_grid,
)
