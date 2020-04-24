import os
import time
import logging

import zarr
from numcodecs import Zlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# TODO use daisy for this?


def snapshot(array, filename, dataset='volumes/supervoxels'):
    """Take a snapshot of a 3D volume and save next to raw data

    Args:
        array (3D np.array): the snapshot
        filename (str): path to exisiting zarr file
        dataset (str): name of snapshot
    """
    # try to open the existing raw zarr file
    raw = zarr.open(os.path.join(filename, 'volumes/raw'), 'r')
    # get needed attributes from raw file
    resolution = raw.attrs['resolution']
    offset = raw.attrs['offset']

    assert raw.shape == array.shape, \
        f'Raw shape {raw.shape} and snapshot shape {array.shape} do not match'

    snapshot = zarr.open(
        os.path.join(filename, dataset),
        mode='w',
        shape=raw.shape,
        chunks=raw.chunks,
        dtype='uint64',
        compressor=Zlib(level=5)
    )
    snapshot.attrs['resolution'] = resolution
    snapshot.attrs['offset'] = offset

    # write the new dataset
    snapshot[:] = array.astype('uint64')
