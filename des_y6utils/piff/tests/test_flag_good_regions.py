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


def test_flag_good_regions_make_a_good_box():
    bad_msk = np.zeros((12, 15)).astype(bool)
    bad_msk[:, 0] = True
    b, flag = _make_a_good_box(bad_msk)

    assert b == {
        "xmin": 1,
        "xmax": 14,
        "ymin": 0,
        "ymax": 11,
    }


def test_flag_good_regions_make_a_good_box_random():
    rng = np.random.RandomState(seed=10)
    some_ok = False
    for _ in range(100):
        bad_msk = rng.uniform(size=(12, 15)) > 0.8
        b, flag = _make_a_good_box(bad_msk)
        if flag == 0:
            assert not np.any(bad_msk[b["ymin"]:b["ymax"]+1, b["xmin"]:b["xmax"]+1])
            some_ok = True

    assert some_ok


def test_flag_good_regions_make_a_good_box_all_bad():
    bad_msk = np.ones((12, 15)).astype(bool)
    b, flag = _make_a_good_box(bad_msk)

    assert flag != 0
    assert b == {
        "xmin": 0,
        "xmax": 14,
        "ymin": 0,
        "ymax": 11,
    }
