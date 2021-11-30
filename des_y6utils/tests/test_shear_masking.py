from ..shear_masking import generate_shear_masking_factor


def test_generate_shear_masking_factor():
    # this test ensures the code does not change so when we remove it, it works
    assert generate_shear_masking_factor("blah") == 1.033477298
