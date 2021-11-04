import numpy as np

from ..flag_good_regions import _make_a_good_box


def test_flag_good_regions_make_a_good_box_smoke():
    bad_msk = np.zeros((12, 15)).astype(bool)
    b, flag = _make_a_good_box(bad_msk)

    assert b == {
        "xmin": 0,
        "xmax": 14,
        "ymin": 0,
        "ymax": 11,
    }
