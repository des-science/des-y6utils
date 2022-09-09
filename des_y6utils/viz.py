import functools
import os
import subprocess
import contextlib

import ngmix.medsreaders

DEFAULT_PZ_TAG = "Y6A2_PIZZACUTTER_V3"


@functools.lru_cache(maxsize=128)
def _get_meds_files_mapping(pz_tag):
    import easyaccess as ea
    conn = ea.connect(section='desoper')

    query = """\
select
    concat(fai.filename, fai.compression) as filename,
    fai.path as path,
    m.band as band,
    m.tilename
from
    desfile d1,
    proctag t,
    miscfile m,
    file_archive_info fai
where
    d1.pfw_attempt_id = t.pfw_attempt_id
    and t.tag = '%s'
    and d1.filename = m.filename
    and d1.id = fai.desfile_id
    and fai.archive_name = 'desar2home'
    and d1.filetype = 'coadd_pizza_cutter'
""" % (pz_tag)

    curs = conn.cursor()
    curs.execute(query)

    meds_files = {}
    c = curs.fetchall()
    for filename, path, band, tilename in c:
        fname = os.path.join(path, filename)
        if tilename not in meds_files:
            meds_files[tilename] = {}
        meds_files[tilename][band] = fname

    conn.close()

    return meds_files


def get_meds_files_for_tile(tilename, pz_tag=None, bands=None):
    """Download cell-based coadd MEDS files for a tile.

    Parameters
    ----------
    tilename : str
        The name of the tile.
    pz_tag : str or None, optional
        The tag in the DESDM database for the files. If None, then use the default
        value in `des_y6utils.viz.DEFAULT_PZ_TAG`.
    bands : list of str or None, optional
        If not None, only get tiles for the specific bands in the list. If None, use the
        default of griz.

    Returns
    -------
    meds_files : dict of str
        Dictionary mapping band to the local MEDS file.
    """
    if pz_tag is None:
        pz_tag = DEFAULT_PZ_TAG

    if bands is None:
        bands = ["g", "r", "i", "z"]

    os.makedirs("./data", exist_ok=True)
    all_meds_files = _get_meds_files_mapping(pz_tag).get(tilename, {})
    meds_files = {}
    for band, fname in all_meds_files.items():
        if band not in bands:
            continue
        if not os.path.exists("./data/%s" % os.path.basename(fname)):
            cmd = """\
rsync \
-av \
--password-file $DES_RSYNC_PASSFILE \
${DESREMOTE_RSYNC_USER}@${DESREMOTE_RSYNC}/%s \
./data/%s
""" % (fname, os.path.basename(fname))
            subprocess.run(cmd, shell=True, check=True)
        meds_files[band] = "./data/%s" % os.path.basename(fname)

    return meds_files


@contextlib.contextmanager
def get_ngmix_meds_reader(tilename, pz_tag=None, bands=None):
    """Get an MultiBandNGMixMEDS read for a list of tiles.

    This function has to be used as a context manager.

    Parameters
    ----------
    tilename : str
        The name of the tile.
    pz_tag : str or None, optional
        The tag in the DESDM database for the files. If None, then use the default
        value in `des_y6utils.viz.DEFAULT_PZ_TAG`.
    bands : list of str or None, optional
        If not None, only get tiles for the specific bands in the list. If None, use the
        default of griz.

    Returns
    -------
    mbmeds : ngmix.medsreaders.MultiBandNGMixMEDS(mlist)
        A MEDS reader for tile.
    """
    try:
        if bands is None:
            bands = ["g", "r", "i", "z"]
        meds_files = get_meds_files_for_tile(tilename, pz_tag=pz_tag, bands=bands)
        mlist = [
            ngmix.medsreaders.NGMixMEDS(meds_files[band])
            for band in bands
        ]
        yield ngmix.medsreaders.MultiBandNGMixMEDS(mlist)
    finally:
        for m in mlist:
            m.close()
