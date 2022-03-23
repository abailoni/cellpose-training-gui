import os

import h5py
import numpy as np
try:
    import vigra
except ImportError:
    vigra = None
from scipy.ndimage import zoom


def readHDF5(path,
             inner_path,
             crop_slice=None,
             dtype=None,
             ds_factor=None,
             ds_order=3,
             run_connected_components=False,
             ):
    if isinstance(crop_slice, str):
        crop_slice = parse_data_slice(crop_slice)
    elif crop_slice is not None:
        assert isinstance(crop_slice, tuple), "Crop slice not recognized"
        assert all([isinstance(sl, slice) for sl in crop_slice]), "Crop slice not recognized"
    else:
        crop_slice = slice(None)
    with h5py.File(path, 'r') as f:
        output = f[inner_path][crop_slice]

    if run_connected_components:
        assert vigra is not None, "Vigra module is needed to compute connected components"
        assert output.dtype in [np.dtype("uint32")]
        assert output.ndim == 3 or output.ndim == 2
        output = vigra.analysis.labelVolumeWithBackground(output.astype('uint32'))
    if dtype is not None:
        output = output.astype(dtype)

    if ds_factor is not None:
        assert isinstance(ds_factor, (list, tuple))
        assert output.ndim == len(ds_factor)
        output = zoom(output, tuple(1./fct for fct in ds_factor), order=ds_order)

    return output


def writeHDF5(data, path, inner_path, compression='gzip'):
    if os.path.exists(path):
        write_mode = 'r+'
    else:
        write_mode = 'w'
    with h5py.File(path, write_mode) as f:
        if inner_path in f:
            del f[inner_path]
        f.create_dataset(inner_path, data=data, compression=compression)


def parse_data_slice(data_slice):
    """Parse a dataslice as a list of slice objects."""
    if data_slice is None:
        return data_slice
    elif isinstance(data_slice, (list, tuple)) and \
            all([isinstance(_slice, slice) for _slice in data_slice]):
        return list(data_slice)
    else:
        assert isinstance(data_slice, str)
    # Get rid of whitespace
    data_slice = data_slice.replace(' ', '')
    # Split by commas
    dim_slices = data_slice.split(',')
    # Build slice objects
    slices = []
    for dim_slice in dim_slices:
        indices = dim_slice.split(':')
        if len(indices) == 2:
            start, stop, step = indices[0], indices[1], None
        elif len(indices) == 3:
            start, stop, step = indices
        else:
            raise RuntimeError
        # Convert to ints
        start = int(start) if start != '' else None
        stop = int(stop) if stop != '' else None
        step = int(step) if step is not None and step != '' else None
        # Build slices
        slices.append(slice(start, stop, step))
    # Done.
    return tuple(slices)
