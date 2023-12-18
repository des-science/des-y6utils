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

from . import mdet
from . import viz
