from napari_ome_zarr import napari_get_reader
import numpy as np
from ome_zarr.types import PathLike
from typing import Any, Callable, Dict, Iterator, List, Optional


def load_ome_zarr_channels(ome_zarr_path: PathLike,
                           channels_to_select: List[str]):
    zarr_group_elements = napari_get_reader(ome_zarr_path)()

    # Loop over images in the zarr file:
    collected_images = {}
    full_channel_list = get_channel_list_in_ome_zarr(ome_zarr_path)

    for i, zarr_element in enumerate(zarr_group_elements):
        # Load metadata:
        metadata_dict = zarr_element[1]
        channel_axis = metadata_dict.get("channel_axis", None)
        channel_names = metadata_dict.get("name", None)
        channel_names = [channel_names] if isinstance(channel_names, str) else channel_names

        # Loop over channels in the zarr element:
        ch_image, channel_slice = None, None

        for ch_name in channels_to_select:
            assert ch_name in full_channel_list, f"Channel not found in ome-zarr file: {ch_name}. " \
                                                 f"Available channels are {full_channel_list}"

            if channel_names is not None:
                if ch_name not in channel_names:
                    continue
                # Load image:
                image = zarr_element[0][0].compute()
                # Find channel index:
                ch_idx = channel_names.index(ch_name)
                if channel_axis is not None:
                    image = image.take(ch_idx, channel_axis)
                else:
                    assert len(channel_names) == 1
                ch_image = np.squeeze(image)
            else:
                img_idx, ch_idx = ch_name.split("_ch_")
                if int(img_idx) == i:
                    # Load image:
                    assert channel_axis is not None, "Cannot deduce number of channels without channel axis info"
                    image = zarr_element[0][0].compute()
                    # Get channel:
                    ch_image = np.squeeze(image.take(ch_idx, channel_axis))

            if ch_image is not None:
                assert ch_image.ndim == 2, "Channels images should be 2D"
                assert ch_name not in collected_images, "Channel name found in multiple elements of the ome-zarr file"
                collected_images[ch_name] = ch_image

    return [collected_images[ch_name] for ch_name in channels_to_select]


def get_channel_list_in_ome_zarr(ome_zarr_path: PathLike):
    zarr_group_elements = napari_get_reader(ome_zarr_path)()

    # Loop over images in the zarr file:
    collected_channels = []

    for i, zarr_element in enumerate(zarr_group_elements):
        # Load metadata:
        metadata_dict = zarr_element[1]
        names = metadata_dict.get("name", None)
        channel_axis = metadata_dict.get("channel_axis", None)
        if names is not None:
            names = [names] if isinstance(names, str) else names
            collected_channels += names
        elif channel_axis is not None:
            image_shape = zarr_element[0][0].shape
            collected_channels += [f"{i}_ch_{ch}" for ch in range(image_shape[channel_axis])]

    return collected_channels
