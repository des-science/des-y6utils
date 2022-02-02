import glob
import io
import sys
import os
import contextlib
import click
import subprocess
import tempfile
import hdf5plugin
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from esutil.pbar import PBar
import fitsio
import h5py

from des_y6utils.shear_masking import generate_shear_masking_factor
from ngmix.shape import (
    e1e2_to_g1g2, g1g2_to_e1e2, g1g2_to_eta1eta2, eta1eta2_to_g1g2
)


COLUMNS_TO_KEEP = [
    "ra",
    "dec",
    "mdet_g_1",
    "mdet_g_2",
    "mdet_step",
    "hpix_16384",
    "mdet_s2n",
    "mdet_T",
    "mdet_T_ratio",
    "mdet_T_err",
    "mfrac",
    "psfrec_flags",
    "psfrec_g_1",
    "psfrec_g_2",
    "psfrec_T",
    "mdet_flux_flags",
    'mdet_g_flux',
    'mdet_r_flux',
    'mdet_i_flux',
    'mdet_z_flux',
    'mdet_g_flux_err',
    'mdet_r_flux_err',
    'mdet_i_flux_err',
    'mdet_z_flux_err',
]


def _create_array_hdf5(pth, arr, fp):
    if "U" in arr.dtype.kind:
        arr = np.char.encode(arr, "ascii")

    fp.create_dataset(
        pth,
        data=arr,
        chunks=(1_000_000,),
        maxshape=(len(arr),),
        shape=(len(arr),),
        **hdf5plugin.Blosc(cname="snappy")
    )


def _make_cuts(arr):
    msk = (
        (arr["mdet_s2n"] > 7)
        & (arr["mdet_flags"] == 0)
        & (arr["mask_flags"] == 0)
        & (arr["mdet_flux_flags"] == 0)
        & (arr["mdet_T_ratio"] > 0.5)
        & (arr["mfrac"] < 0.10)
        & (arr["mdet_T"] < 1.2)
    )
    return arr[msk]


def _mask_shear_arr(d, passphrase_file, fname):
    failed = False
    buff = io.StringIO()
    with contextlib.redirect_stderr(sys.stdout):
        with contextlib.redirect_stdout(buff):
            try:
                with open(passphrase_file, "r") as fp:
                    passphrase = fp.read().strip()

                fac = generate_shear_masking_factor(passphrase)
                msk = d["mdet_step"] == "noshear"
                e1o, e2o = d["mdet_g_1"][msk].copy(), d["mdet_g_2"][msk].copy()
                g1, g2 = e1e2_to_g1g2(e1o, e2o)
                eta1, eta2 = g1g2_to_eta1eta2(g1, g2)
                eta1 *= fac
                eta2 *= fac
                g1, g2 = eta1eta2_to_g1g2(eta1, eta2)
                e1, e2 = g1g2_to_e1e2(g1, g2)
                d["mdet_g_1"][msk] = e1
                d["mdet_g_2"][msk] = e2

                assert not np.array_equal(d["mdet_g_1"][msk], e1o)
                assert not np.array_equal(d["mdet_g_2"][msk], e2o)

            except Exception:
                failed = True

    if failed:
        raise RuntimeError("failed to mask shear in file %s" % fname)
    else:
        return d


def _process_file(passphrase_file, fname):
    try:
        subprocess.run(
            "python -c 'import fitsio; fitsio.read(\"%s\")'" % fname,
            shell=True,
            check=True,
        )
    except Exception:
        return None

    arr = fitsio.read(fname)
    arr = _make_cuts(arr)
    arr = _mask_shear_arr(arr, passphrase_file, fname)

    return arr


def _build_file(arrs, chunk, output_path, columns_to_keep, tmpdir):
    opth = output_path + "_%05d.h5" % chunk
    with tempfile.TemporaryDirectory(dir=tmpdir) as tt:
        topth = os.path.join(tt, os.path.basename(opth))
        with h5py.File(topth, 'w') as fp:
            print("\n", end="", flush=True)
            for cname in PBar(
                columns_to_keep, desc="writing cols for chunk %d" % chunk
            ):
                arr = np.concatenate(
                    [arr[cname] for arr in arrs if arr is not None]
                )
                pth = os.path.join("catalogs", "mdet", cname)
                _create_array_hdf5(pth, arr, fp)

        os.system("cp %s %s" % (topth, opth))


def make_hdf5_file(
    input_tile_glob,
    output_path,
    passphrase_file,
    tmpdir,
    columns_to_keep=None,
):
    columns_to_keep = columns_to_keep or COLUMNS_TO_KEEP

    input_fnames = sorted(glob.glob(input_tile_glob))
    n_files_per_chunk = 100
    n_chunks = len(input_fnames) // n_files_per_chunk
    if n_chunks * n_files_per_chunk < len(input_fnames):
        n_chunks += 1

    for chunk in PBar(range(n_chunks), desc="writing chunks"):
        start = chunk * n_files_per_chunk
        end = min(start + n_files_per_chunk, len(input_fnames))
        fnames = input_fnames[start:end]

        with ProcessPoolExecutor(max_workers=6) as exec:
            futs = {}
            for fname in fnames:
                futs[
                    exec.submit(
                        _process_file, passphrase_file, fname
                    )
                ] = fname

            print("\n", end="", flush=True)
            arrs = []
            for fut in PBar(futs, desc="processing files"):
                try:
                    res = fut.result()
                except Exception as e:
                    print(e, flush=True)
                    res = None

                if res is None:
                    print("\nfile %s failed" % futs[fut], flush=True)
                else:
                    arrs.append(res)

        _build_file(arrs, chunk, output_path, columns_to_keep, tmpdir)


@click.command()
@click.option(
    "--input-glob", type=str, required=True,
    help="glob expression to specify all input files"
)
@click.option(
    "--output", type=str, required=True,
    help="/path/to/output.hdf5"
)
@click.option(
    "--passphrase-file", type=str, required=True,
    help="path to passphrase file for masking"
)
@click.option(
    "--tmpdir", type=str, required=True,
    help="temp. dir for writing"
)
def cli_hdf5(input_glob, output, passphrase_file, tmpdir):
    """Combine mdet tile files into an HDF5 output."""

    make_hdf5_file(
        input_glob,
        output,
        passphrase_file,
        tmpdir,
        columns_to_keep=None,
    )
