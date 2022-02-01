import glob
import io
import sys
import os
import contextlib
import click
import joblib

import numpy as np
from esutil.pbar import PBar
import fitsio
import h5py
import hdf5plugin

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
    fp.create_dataset(
        pth,
        data=arr,
        chunksize=(1_000_000,),
        maxshape=(len(arr),),
        shape=(len(arr),),
        **hdf5plugin.Blosc(cname="snappy"),
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


def _process_file(passphrase_file, fname, cols_to_keep):
    arr = fitsio.read(fname)
    arr = _make_cuts(arr)

    if passphrase_file is not None:
        arr = _mask_shear_arr(arr, passphrase_file, fname)

    return {c: arr[c].copy() for c in cols_to_keep}


def make_hdf5_file(
    input_tile_glob,
    output_path,
    passphrase_file,
    columns_per_io_pass=5,
    columns_to_keep=None,
):
    columns_to_keep = columns_to_keep or COLUMNS_TO_KEEP

    n_column_chunks = len(columns_to_keep) // columns_per_io_pass
    if n_column_chunks * columns_per_io_pass < len(columns_to_keep):
        n_column_chunks += 1

    input_fnames = sorted(glob.glob(input_tile_glob))
    arrlen = None
    with h5py.File(output_path, 'w') as fp:
        for col_chunk in PBar(range(n_column_chunks), desc="column chunk"):
            start_col = col_chunk * columns_per_io_pass
            end_col = min(start_col + columns_per_io_pass, len(columns_to_keep))
            col_data = {
                cname: []
                for cname in columns_to_keep[start_col:end_col]
            }

            jobs = [
                joblib.delayed(_process_file)(
                    passphrase_file,
                    fname,
                    columns_to_keep[start_col:end_col],
                )
                for fname in input_fnames
            ]
            with joblib.Parallel(n_jobs=8, verbose=100) as par:
                arrs = par(jobs)
            for arr in arrs:
                for col_ind in range(start_col, end_col):
                    cname = columns_to_keep[col_ind]
                    col_data[cname].append(arr[cname])

            # with ProcessPoolExecutor(max_workers=16) as exec:
            #     futs = [
            #         exec.submit(_process_file, passphrase_file, fname)
            #         for fname in input_fnames
            #     ]
            #     print("\n", end="", flush=True)
            #     for fut in PBar(futs, total=len(futs), desc="processing data"):
            #         try:
            #             arr = fut.result()
            #             for col_ind in range(start_col, end_col):
            #                 cname = columns_to_keep[col_ind]
            #                 col_data[cname].append(arr[cname].copy())
            #         except Exception as e:
            #             print(e)

            # for fname in PBar(input_fnames, desc="files"):
            #     arr = _process_file(passphrase_file, fname)
            #     for col_ind in range(start_col, end_col):
            #         cname = columns_to_keep[col_ind]
            #         col_data[cname].append(arr[cname].copy())

            for cname in list(col_data):
                arr = np.concatenate(col_data[cname])
                del col_data[cname]
                if arrlen is None:
                    arrlen = len(arr)

                assert len(arr) == arrlen

                pth = os.path.join("catalogs", "mdet", cname)
                _create_array_hdf5(pth, arr, fp)


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
    "--cols-per-io-pass", type=int, default=5,
    help="# of columns to read per I/O pass over the catalog"
)
def cli_hdf5(input_glob, output, passphrase_file, cols_per_io_pass):
    """Combine mdet tile files into an HDF5 output."""

    make_hdf5_file(
        input_glob,
        output,
        None if passphrase_file == "null" else passphrase_file,
        columns_per_io_pass=5,
        columns_to_keep=None,
    )
