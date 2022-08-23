import numpy as np
import cv2
import imageio
import os.path
import tifffile
import zarr
from PIL import Image

from traincellpose.io.ome_zarr_utils import get_channel_list_in_ome_zarr

try:
    from ome_zarr.conversions import rgba_to_int
except ImportError:
    rgba_to_int = None

try:
    from paquo.colors import QuPathColor
except ImportError:
    QuPathColor = None

colormaps = {"gray": [255, 255, 255, 255],
             "red": [255, 0, 0, 255],
             "yellow": [255, 255, 0, 255],
             "cyan": [0, 255, 255, 255]}


def write_segm_to_file(path, array):
    filename, extension = os.path.splitext(os.path.split(path)[1])
    if extension == ".png":
        imageio.imwrite(path, array.astype(np.uint16))
    elif extension == ".tif" or extension == ".tiff":
        cv2.imwrite(path, array.astype(np.uint16))
    else:
        raise ValueError("Only png and tif extensions supported")


def write_ome_tiff(path, data, axes='TCZYX',
                   channel_names=None, channel_colors=None):
    assert path.endswith(".ome.tif")
    assert rgba_to_int is not None, "ome-zarr module needs to be installed"

    metadata = {'axes': axes}

    if channel_names is not None or channel_colors is not None:
        metadata["Channel"] = {}

    if channel_names is not None:
        assert isinstance(channel_names, (tuple, list))
        metadata["Channel"]["Name"] = channel_names

    if channel_colors is not None:
        assert isinstance(channel_colors, (tuple, list))
        assert all([color in colormaps for color in channel_colors]), "Color not recognised"
        metadata["Channel"]["Color"] = [rgba_to_int(*colormaps[color]) for color in channel_colors]

    tifffile.imwrite(
        path, data, metadata=metadata
        # , ome=True
    )
    # with TiffWriter('temp.ome.tif') as tif:
    #     tif.save(data0, compress=6, photometric='rgb')
    #     tif.save(data1, photometric='minisblack',
    #         metadata = {'axes': 'ZYX', 'SignificantBits': 10,
    #                                 'Plane': {'PositionZ': [0.0, 1.0, 2.0, 3.0]}})


def write_image_to_file(path, array):
    # TODO: to be improved
    filename, extension = os.path.splitext(os.path.split(path)[1])
    if extension == ".png":
        imageio.imwrite(path, array)
    elif extension == ".tif" or extension == ".tiff":
        cv2.imwrite(path, array)
    else:
        raise ValueError("Only png and tif extensions supported")


def read_uint8_img(img_path, add_all_channels_if_needed=True):
    # TODO: rename and move to io module together with function exporting segmentation file
    assert os.path.isfile(img_path), "Image {} not found".format(img_path)

    # TODO: to be improved
    extension = os.path.splitext(img_path)[1]
    if extension == ".tif" or extension == ".tiff":
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img.dtype == 'uint16':
            img = cv2.convertScaleAbs(img, alpha=(255.0 / 65535.0))
        assert img.dtype == 'uint8'
    elif extension == ".png":
        img = imageio.v3.imread(img_path)
    else:
        raise ValueError("Extension {} not supported".format(extension))
    if len(img.shape) == 2 and add_all_channels_if_needed:
        # Add channel dimension:
        img = np.stack([img for _ in range(3)])
        img = np.rollaxis(img, axis=0, start=3)
    # assert len(img.shape) == 3 and img.shape[2] == 3, img.shape

    return img


def deduce_image_type(image_path, raise_if_not_recognized=False):
    image_type = None
    if os.path.isdir(image_path):
        # Try to open to see if it is a zarr directory
        try:
            zarr.convenience.open(image_path, mode="r")
            image_type = "zarr"
        except BaseException:
            pass
    elif os.path.isfile(image_path):
        extension = os.path.splitext(image_path)[1]
        if extension in [".tif", "tiff", ".png"]:
            image_type = extension

    if image_type is None and raise_if_not_recognized:
        raise ValueError(f"Image not recognized. Only zarr, tiff, and png formats are supported: {image_path}")

    return image_type


def get_image_info_dict(img_path: str,
                        channel_name: str = None):
    out_dict = {"path": img_path}

    img_type = deduce_image_type(img_path)
    if img_type == "zarr":
        out_dict["inner_channels"] = get_channel_list_in_ome_zarr(img_path)
    elif img_type is not None:
        # Check size image without reading it:
        im = Image.open(img_path)
        nb_img_channels = len(im.getbands())
        if nb_img_channels != 1:
            out_dict["inner_channels"] = [str(i) for i in range(nb_img_channels)]

    out_dict["type"] = img_type

    if channel_name is not None:
        assert isinstance(channel_name, str)
        out_dict["channel_name"] = channel_name

    return out_dict
