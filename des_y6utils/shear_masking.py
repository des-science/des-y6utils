import hashlib


def generate_shear_masking_factor(passphrase):
    """Generate a masking factor by hashing a passphrase.

    Code from Joe Zuntz w/ modifications for python 3 by Matt B.

    Parameters
    ----------
    passphrase : str
        A string.

    Returns
    -------
    factor : float
        The masking factor as a float in the range 0.9 to 1.1.
    """
    # make hex
    m = hashlib.md5(passphrase.encode("utf-8")).hexdigest()
    # convert to decimal
    s = int(m, 16)
    # get last 8 digits
    f = s % 100_000_000
    # turn 8 digit number into value between 0 and 1
    g = f / 1e8
    # scale value between 0.9 and 1.1
    return 0.9 + 0.2*g
