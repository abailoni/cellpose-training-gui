import math
import os
import shutil
import traceback
from copy import deepcopy
from shutil import copyfile
import logging
from typing import Dict, Union, List

import numpy as np
import pandas
import tifffile
import yaml
from pathlib import Path

import zarr

from speedrun import BaseExperiment, locate
from speedrun.yaml_utils import recursive_update
from .cellpose_training.start_training import start_cellpose_training

from .gui_widgets.main_gui import StartingGUI
from .io.images import read_uint8_img, write_image_to_file, write_ome_tiff, deduce_image_type, get_image_info_dict
from .io.hdf5 import readHDF5, writeHDF5
from .io.ome_zarr_utils import get_channel_list_in_ome_zarr, load_ome_zarr_channels
from .preprocessing.utils import apply_preprocessing_to_image
from .qupath import update_qupath_proj as qupath_utils
from .qupath.save_labels import export_labels_from_qupath
from .io.various import yaml2dict, get_path_components, open_path
from .qupath.update_qupath_proj import add_image_to_project


class BaseAnnotationExperiment(BaseExperiment):
    def __init__(self, experiment_directory):
        self._main_window = None
        assert isinstance(experiment_directory, str)
        super(BaseAnnotationExperiment, self).__init__(experiment_directory)

        # Simulate sys.argv, so that configuration is loaded from the experiment directory:
        self._simulated_sys_argv = ["script.py", experiment_directory]
        # Check if this is a new project or if we should load the previous config:
        config_path = os.path.join(experiment_directory, "Configurations/main_config.yml")
        load_prev_experiment = os.path.exists(config_path)
        if load_prev_experiment:
            old_config_path = os.path.join(experiment_directory, "Configurations/main_config_BAK.yml")
            copyfile(config_path, old_config_path)
            self._simulated_sys_argv += ["--inherit", old_config_path]

        # Load config and setup:
        self.auto_setup(update_git_revision=False)

        # Set default values:
        if not load_prev_experiment:
            self.set("max_nb_extra_channels", 2)
            self.set("extra_channels_names", ["Extra ch. 1", "Extra ch. 2"])
            self.set("labeling_tool", "QuPath")
            self.set_default_training_args()
            self.set_default_preprocessing_config()
            self.apply_preprocessing = False

        # Initialize or load dataframes:
        self._rois_df = None
        self.input_images = InputImageDict(self, "input_images")
        # TODO: Add ROIs to config file insetad of csv
        self._rois_dict = InputImageDict(self, "rois")
        self._input_images_df = None
        self._init_rois()
        self._init_input_images_df()

        self.dump_configuration()

    def run(self):
        self.show_start_page()

    def show_start_page(self):
        self.main_window.show()
        self.dump_configuration()

    @property
    def main_window(self):
        if self._main_window is None:
            # self._main_window = widgets.Container(widgets=[StartWindow(self)])
            self._main_window = StartingGUI(self)
            # self._main_window.max_width = 30
            self._main_window.show(run=True)
        return self._main_window

    # --------------------------------------------
    # ROIs:
    # --------------------------------------------

    def update_rois_image(self, image_id, new_napari_rois):
        if isinstance(new_napari_rois, list):
            new_napari_rois = np.array(new_napari_rois)

        # Get IDs of previous ROIs:
        prev_roi_ids = self._get_roi_ids_by_image_id(image_id)
        current_max_roi_id = self._napari_rois.shape[0]
        prev_napari_rois = self._napari_rois[prev_roi_ids]

        # Check if no new napari rois were passed:
        if new_napari_rois.size == 0:
            # Delete any previous ROIs:
            self._delete_training_images(prev_roi_ids)
            self._delete_roi_ids(prev_roi_ids)
        else:
            assert new_napari_rois.ndim == 3
            assert new_napari_rois.shape[1] == 4 and new_napari_rois.shape[
                2] == 2, "ROI array does not have the correct shape"

            # Check what is there:
            check_rois = np.array([[np.allclose(new_roi, old_roi) for old_roi in prev_napari_rois]
                                   for new_roi in new_napari_rois])
            # Add new ROIs:
            rois_not_already_in_project = ~ np.any(check_rois, axis=1)
            self._napari_rois = np.concatenate([self._napari_rois, new_napari_rois[rois_not_already_in_project]])
            for i in range(current_max_roi_id, current_max_roi_id + rois_not_already_in_project.sum()):
                self._rois_df.loc[i] = [i, image_id]
                self._create_training_images([i])

            # Remove ROIs that are not present anymore:
            old_rois_to_be_deleted = ~ np.any(check_rois, axis=0)
            old_rois_to_be_deleted = list(np.array(prev_roi_ids)[old_rois_to_be_deleted])
            self._delete_training_images(old_rois_to_be_deleted)
            self._delete_roi_ids(old_rois_to_be_deleted)

        # Update saved files:
        self.dump_rois()

    def get_list_rois_per_image(self):
        """
        Return a list of tuples, such that:
            output_list[index_input_image] = (path_main_image, nb_rois)
        """
        out_list = []
        for id_image in range(self.nb_input_images):
            selected_rows = self._input_images_df.loc[self._input_images_df["image_id"] == id_image]
            assert len(selected_rows) == 1
            nb_rois = len(self._get_roi_ids_by_image_id(id_image))
            out_list.append((selected_rows["main_path"].item(), nb_rois))
        return out_list

    def get_napari_roi_by_image_id(self, image_id):
        rois_ids = self._get_roi_ids_by_image_id(image_id)
        # Check if there are ROIs at all:
        if len(rois_ids):
            return [roi for roi in self._napari_rois[rois_ids]]
        else:
            return None

    def get_image_id_from_roi_id(self, roi_id):
        df = self._rois_df
        image_id = df.loc[df["roi_id"] == roi_id, "image_id"].tolist()
        assert len(image_id) == 1
        return image_id[0]

    def _get_roi_ids_by_image_id(self, image_id):
        df = self._rois_df
        rois_ids = df.loc[df["image_id"] == image_id, "roi_id"].tolist()

        return rois_ids

    def _delete_roi_ids(self, roi_ids):
        # TODO: Currently, ROIs are actually not deleted from the hdf5 file,
        #  but only from the dataframe (to avoid reordering)
        #  When done, uncomment assert to check consistency csv/hdf5
        df = self._rois_df
        self._rois_df = df[~df['roi_id'].isin(roi_ids)]

        # TODO Delete also files!

    def _init_rois(self):
        if self._rois_df is None:
            rois_csv_path = os.path.join(self.experiment_directory, "ROIs/rois.csv")
            rois_hdf5_path = os.path.join(self.experiment_directory, "ROIs/rois.hdf5")
            if os.path.exists(rois_csv_path):
                self._rois_df = pandas.read_csv(rois_csv_path)
                assert os.path.exists(rois_hdf5_path), "ROIs hdf5 file not found!"
                self._napari_rois = readHDF5(rois_hdf5_path, "data")
                rois_shape = self._napari_rois.shape
                assert rois_shape[1] == 4 and rois_shape[2] == 2
                # assert len(self._rois_df) == rois_shape[0], "ROIs csv and hdf5 files do not match!"
            else:
                # Create empty a dataframe and array:
                self._rois_df = pandas.DataFrame(columns=["roi_id", "image_id"])
                self._napari_rois = np.empty((0, 4, 2), dtype="float64")

    def dump_rois(self):
        # Get paths:
        proj_dir = self.experiment_directory
        rois_dir_path = os.path.join(proj_dir, "ROIs")
        roi_csv_path = os.path.join(rois_dir_path, "rois.csv")
        rois_hdf5_path = os.path.join(rois_dir_path, "rois.hdf5")

        # Write data to file:
        writeHDF5(self._napari_rois, rois_hdf5_path, "data")
        self._rois_df.to_csv(roi_csv_path, index=False)

        # Dump general configuration:
        self.dump_configuration()

    # --------------------------------------------
    # Input images:
    # --------------------------------------------
    def set_extra_channels_names(self, channels_names):
        # TODO: deprecate
        if not isinstance(channels_names, list):
            assert isinstance(channels_names, str)
            channels_names = [channels_names]
        assert len(channels_names) <= self.get("max_nb_extra_channels")
        new_names = self.get("extra_channels_names")
        for i, ch_name in enumerate(channels_names):
            new_names[i] = ch_name
        self.set("extra_channels_names", new_names)

    def get_input_image_id_from_path(self, main_image_path):
        df = self._input_images_df

        # If image is in proj dir, then get relative path:
        if os.path.isabs(main_image_path):
            is_in_exp_dir, main_image_path = self.is_path_in_exp_dir(main_image_path)

        image_id = df.loc[df["main_path"] == main_image_path, "image_id"].tolist()
        assert len(image_id) == 1
        return image_id[0]

    def is_path_in_exp_dir(self, path):
        if path is not None:
            path = path if isinstance(path, Path) else Path(path)
            is_in_exp_dir = Path(self.experiment_directory) in path.parents
            path = os.path.relpath(path, self.experiment_directory) if is_in_exp_dir else path
            return is_in_exp_dir, str(path)
        else:
            return False, None

    def get_image_info(self,
                       image_id: int,
                       channels_to_load: Union[int, List[int], str] = None):
        """
        :param channels_to_load: "all", int, or list of ints
        """
        if channels_to_load == "all":
            channels_to_load = [i for i in range(len(self.channel_names))]

        channels_to_load = channels_to_load if isinstance(channels_to_load, list) else list(channels_to_load)
        assert all(isinstance(ch, int) for ch in channels_to_load)

        ch_images_dicts = []

        all_image_ids = self.input_images.get_all_image_ids()
        assert image_id in all_image_ids, f"Image ID not found: {image_id}"
        img_info_dict = self.input_images[image_id]

        for ch_idx, ch_name in enumerate(self.channel_names):
            if ch_idx in img_info_dict:
                ch_img_dict = self.input_images[image_id, int(ch_idx)]
                ch_img_dict["channel_name"] = ch_name
                if int(ch_idx) in channels_to_load:
                    ch_img_dict["image"] = self.load_channel_img(img_info_dict=ch_img_dict)
                ch_images_dicts.append(ch_img_dict)

        return ch_images_dicts

    def load_channel_img(self,
                         img_info_dict: Dict = None,
                         img_path: str = None,
                         inner_channel_to_select: str = None,
                         raise_if_could_not_load: bool = False,
                         return_error_message: bool = False):
        """
        TODO: better format for path_dict could be (then update also the one for creating a new image!!)
            1. utility that return such dict for a given image path
            2. this method will simply add the loaded images to the dict (or return a list) or maybe just takes the dict of a channel
            3. get image info will return the full list of dicts (maybe including the arrays)
            4. use create_widget() to shorten things...?

        [{"image_path": ...,
        "channel_name": ...,
        "image_type": ...,
        "selected_inner_channel": ...,
        "inner_channels": ...,
        "image_array": ...}
        ]
        """
        assert not(img_path is not None and img_info_dict is not None), "Either path or info dictionary should be given, " \
                                                                   "not both"
        if img_path is not None:
            if isinstance(img_path, str) and img_path != "":
                # If image is in the proj dir, then construct the absolute path:
                if not os.path.isabs(img_path):
                    img_path = os.path.join(self.experiment_directory, img_path)
            img_info_dict = get_image_info_dict(img_path)

            assert inner_channel_to_select is not None
            img_info_dict["inner_channel_to_select"] = inner_channel_to_select
        elif img_info_dict is None:
            raise ValueError("At least path or info dictionary should be given")

        image = None
        img_path = img_info_dict["path"]
        img_type = img_info_dict.get("type", None)
        inner_channel_to_select = img_info_dict.get("inner_channel_to_select", None)
        all_inner_channels = img_info_dict.get("all_inner_channels", None)
        error_msg = None
        if img_type is None:
            error_msg = f"The given path is not a supported image " \
                        f"(ome-zarr, png, tif): {img_path}"
        else:
            if ((img_type == "zarr") or ((img_type != "zarr") and (all_inner_channels is not None))) and \
                    (inner_channel_to_select is None):
                error_msg = f"Inner channel to select not given"
            else:
                if img_type == "zarr":
                    try:
                        image = load_ome_zarr_channels(img_path,
                                           channels_to_select=[inner_channel_to_select])[0]
                    except Exception as e:
                        error_msg = traceback.format_exc()
                else:
                    try:
                        image = read_uint8_img(img_path, add_all_channels_if_needed=False)
                        if all_inner_channels is not None:
                            if inner_channel_to_select not in all_inner_channels:
                                error_msg = f"Image has channels {all_inner_channels} and channel " \
                                            f"{inner_channel_to_select} was not found"
                            else:
                                # Deduce channel axis:
                                assert image.ndim == 3
                                ch_axis = image.shape.argmin()
                                image = image.take(indices=int(inner_channel_to_select), axis=ch_axis)
                    except Exception as e:
                        error_msg = traceback.format_exc()

        if error_msg is not None:
            if raise_if_could_not_load:
                raise ValueError(error_msg)
            else:
                print(f'Warning: {error_msg}')

        if return_error_message:
            return image, error_msg
        else:
            return image

    def add_input_image(self,
                        img_info_dict,
                        id_input_image_to_rewrite=None
                        ):
        """
        # TODO: add option to remove input image? In that case, I need to update self.nb_input_images
        """
        # Paths should have been validated by the ROI selection widget, so here we just add the dict to
        # the proj config file:

        # Delete loaded images, if part of the dict:
        dict_to_dump = deepcopy(img_info_dict)
        for img_idx in dict_to_dump:
            dict_to_dump[img_idx].pop('image', None)

        # def validate_ch_paths(ch_path, ch_name=None):
        #     ch_path = None if ch_path == "" else ch_path
        #     ch_name = None if ch_name == "" else ch_name
        #     ch_type = None
        #     if ch_path is not None:
        #         ch_type = deduce_image_type(ch_path, raise_if_not_recognized=True)
        #
        #         if ch_type == "zarr":
        #             zarr_channels = get_channel_list_in_ome_zarr(ch_path)
        #             if ch_name is not None: assert ch_name in zarr_channels
        #         elif ch_name is None:
        #             # For an image, this should an integer
        #             ch_name = int(ch_name)
        #
        #     # Convert to relative, if in proj_directory:
        #     _, ch_path = self.is_path_in_exp_dir(ch_path)
        #
        #     return ch_path, ch_name, ch_type

        # main_ch_path, main_ch_name, main_ch_type = validate_ch_paths(main_ch_path, main_ch_name)
        # dapi_ch_path, dapi_ch_name, dapi_ch_type = validate_ch_paths(dapi_ch_path, dapi_ch_name)
        # ch_2_path, ch_2_name, _ = validate_ch_paths(ch_2_path, ch_2_name)
        # ch_3_path, ch_3_name, _ = validate_ch_paths(ch_3_path, ch_3_name)
        #
        # if main_ch_type == "zarr" and dapi_ch_path is None:
        #     # For compatibility with SpaceM zarr files, automatically assign DAPI image to the same zarr file:
        #     dapi_ch_path = main_ch_path
        #     dapi_ch_type = "zarr"

        # # If present, then set up the training to use it (cellpose can still train fine if some of the images do
        # # not have DAPI channel):
        # if dapi_ch_path is not None and (dapi_ch_name is not None or dapi_ch_type != "zarr"):
        #     self.use_dapi_channel_for_segmentation = True

        all_image_ids = self.input_images.get_all_image_ids()

        if id_input_image_to_rewrite is not None:
            assert id_input_image_to_rewrite in all_image_ids
        else:
            id_input_image_to_rewrite = 0 if len(all_image_ids) == 0 else np.array(all_image_ids).max() + 1
        self.input_images[id_input_image_to_rewrite] = dict_to_dump

        # # Check if main image has already been added:
        # matching_images = self._input_images_df.index[self._input_images_df["ch_0_path"] == main_ch_path].tolist()
        # assert len(matching_images) <= 1
        # if len(matching_images) == 1:
        #     print("The added image was already present in the project. Updating paths.")
        #     id_input_image_to_rewrite = matching_images[0]
        #
        # added_image_id = nb_input_images if id_input_image_to_rewrite is None else id_input_image_to_rewrite
        # self._input_images_df.loc[added_image_id] = [added_image_id] + image_info
        # self.dump_input_images_info()

        # Refresh all the ROIs, if there were any:
        self._create_training_images(self._get_roi_ids_by_image_id(id_input_image_to_rewrite))

        return id_input_image_to_rewrite

    def dump_input_images_info(self):
        # Write data to file:
        proj_dir = self.experiment_directory
        rois_dir_path = os.path.join(proj_dir, "ROIs")
        input_images_csv_path = os.path.join(rois_dir_path, "input_images.csv")
        self._input_images_df.to_csv(input_images_csv_path, index=False)

        # Dump general configuration:
        self.dump_configuration()

    @property
    def nb_input_images(self):
        return len(self.input_images)
        # assert self._input_images_df is not None
        # nb_input_images = self._input_images_df["image_id"].max()
        # return 0 if math.isnan(nb_input_images) else nb_input_images + 1

    def _init_input_images_df(self):
        if self._input_images_df is None:
            input_images_csv_path = os.path.join(self.experiment_directory, "ROIs/input_images.csv")
            columns_names = ["image_id"]
            columns_names += [f"ch_{i + 2}_path" for i in range(self.get("max_nb_extra_channels") + 2)]
            columns_names += [f"ch_{i + 2}_name" for i in range(self.get("max_nb_extra_channels") + 2)]
            if os.path.exists(input_images_csv_path):
                self._input_images_df = pandas.read_csv(input_images_csv_path, index_col=None)
                # TODO: remove image_id...?
                self._input_images_df.sort_values("image_id")
                self._input_images_df.reset_index(drop=True)
                # Make sure that index and image ID are the same, otherwise adding images will not work properly:
                assert all([idx == row["image_id"] for idx, row in self._input_images_df.iterrows()])
            else:
                self._input_images_df = pandas.DataFrame(columns=columns_names)

    def show_cellpose_input_folder(self):
        open_path(os.path.join(self.experiment_directory, "ROIs/cellpose_input"))

    def compress_qupath_proj_dir(self):
        shutil.make_archive(self.qupath_directory, 'zip', self.qupath_directory)
        open_path(self.experiment_directory)

    # --------------------------------------------
    # Image crops defined from ROIs and used for training:
    # --------------------------------------------
    def _create_training_images(self, list_roi_ids,
                                update_single_channels=True,
                                update_composite_images=True,
                                update_cellpose_inputs=True
                                ):
        """
        Create the actual cropped images that will be used for training and for annotation.
        """
        if not isinstance(list_roi_ids, (list, tuple)):
            list_roi_ids = [list_roi_ids]

        for roi_id in list_roi_ids:
            img_id = self.get_image_id_from_roi_id(roi_id)
            # FIXME: update these outputs
            _, image_paths, _, ch_names, img_channels = self.get_image_info(img_id, channels_to_load="all")
            crop_slice = self.get_crop_slice_from_roi_id(roi_id)
            roi_paths = self.get_training_image_paths(roi_id)

            # Apply crops:
            img_channels = [img[crop_slice] for img in img_channels if img is not None]

            # ----------------------------
            # Cellpose training image:
            # ----------------------------
            if update_cellpose_inputs:
                # Set green channel as main channel:
                cellpose_image = np.zeros_like(img_channels[0])
                cellpose_image[..., 1] = img_channels[0][..., 0]

                # Set red channel as DAPI:
                if self.use_dapi_channel_for_segmentation and img_channels[1] is not None:
                    cellpose_image[..., 2] = img_channels[1][..., 0]

                # Check if I should apply any preprocessing:
                preproc_kwargs = self.get("preprocessing")
                if preproc_kwargs is not None and self.apply_preprocessing:
                    print("INFO: Preprocessing image...")
                    cellpose_image[..., 1] = apply_preprocessing_to_image(cellpose_image[..., 1], "main_segm_ch",
                                                                          preproc_kwargs)
                    if self.use_dapi_channel_for_segmentation:
                        cellpose_image[..., 2] = apply_preprocessing_to_image(cellpose_image[..., 2], "DAPI",
                                                                              preproc_kwargs)

                # Write image:
                # tifffile.imwrite(roi_paths["cellpose_training_input_image"], cellpose_image)
                write_image_to_file(roi_paths["cellpose_training_input_image"], cellpose_image)
                # write_ome_tiff(roi_paths["cellpose_training_input_image"], cellpose_image, axes="YX")

            # ----------------------------
            # Write composite and single-channel cropped images:
            # ----------------------------
            # image_shape = img_channels[0][..., [0]]
            # Get channel names and colors:
            # TODO: make general variable
            channel_colormaps = ["gray", "red", "yellow", "cyan"]

            if update_single_channels:
                for i, ch_image in enumerate(img_channels):
                    if ch_image is not None:
                        write_image_to_file(roi_paths["single_channels"][ch_names[i]], ch_image)

            if update_composite_images:
                composite_image = np.stack([ch_image[..., 0] for ch_image in img_channels if ch_image is not None],
                                           axis=0)
                write_ome_tiff(roi_paths["composite_image"], composite_image, axes="CYX",
                               channel_names=[ch_name for ch_name, ch in zip(ch_names, img_channels) if ch is not None],
                               channel_colors=[ch_color for ch_color, ch in zip(channel_colormaps, img_channels) if
                                               ch is not None],
                               )

                # Finally, add the image to the QuPath project:
                qupath_utils.add_image_to_project(self.qupath_directory,
                                                  roi_paths["composite_image"])

    def refresh_all_training_images(self, **kwargs):
        all_rois = self._rois_df["roi_id"].tolist()
        self._create_training_images(all_rois, **kwargs)

    def _delete_training_images(self, list_roi_ids):
        """
        Delete cropped images (apart from labels)
        """
        if not isinstance(list_roi_ids, (list, tuple)):
            list_roi_ids = [list_roi_ids]

        for roi_id in list_roi_ids:
            roi_paths = self.get_training_image_paths(roi_id)

            # Remove single channels and cellpose inputs:
            os.remove(roi_paths["cellpose_training_input_image"])
            for ch_name in roi_paths["single_channels"]:
                if roi_paths["single_channels"][ch_name] is not None:
                    os.remove(roi_paths["single_channels"][ch_name])

            # Delete image in QuPath:
            qupath_utils.delete_image_from_project(self.qupath_directory, int(roi_id))
            os.remove(roi_paths["composite_image"])

    def get_crop_slice_from_roi_id(self, roi_id):
        self.assert_roi_id(roi_id)
        roi = self._napari_rois[roi_id]
        x_crop = slice(int(roi[:, 0].min()), int(roi[:, 0].max()))
        y_crop = slice(int(roi[:, 1].min()), int(roi[:, 1].max()))
        return (x_crop, y_crop)

    def assert_roi_id(self, roi_id):
        assert np.array(self._rois_df['roi_id'].isin([roi_id])).sum() == 1, "ROI id not found: {}".format(roi_id)

    def get_roi_list(self):
        roi_list = []
        for image_id in range(self.nb_input_images):
            rois_image = self._get_roi_ids_by_image_id(image_id)
            for i_roi, roi_id in enumerate(rois_image):
                out_roi_info = {}
                out_roi_info['roi_id'] = roi_id
                out_roi_info['image_id'] = image_id
                roi_info = self.get_training_image_paths(roi_id)
                out_roi_info['has_label'] = roi_info["has_labels"]
                out_roi_info['roi_index_per_image'] = i_roi
                roi_list.append(out_roi_info)
        return roi_list

    def get_training_image_paths(self, roi_id):
        """
        For a given ROI id, the function returns paths to the training image used by cellpose,
        the label file with created annotations, and cropped images (both in single-channel and composite versions)
        that are usually used for annotation.
        """
        self.assert_roi_id(roi_id)
        filename_roi_id = "{:04d}".format(roi_id)

        base_ROI_dir = os.path.join(self.experiment_directory, "ROIs")
        label_image_path = self.get_napari_label_file_path(roi_id)
        # Add main paths to crop images:
        # TODO: fix cellpose input mess (ome vs tif, channels colors cellpose input)
        out_dict = {
            "cellpose_training_input_image": os.path.join(base_ROI_dir,
                                                          "cellpose_input/{}.tif".format(filename_roi_id)),
            "composite_image": os.path.join(self.qupath_directory, "input_images/{}.ome.tif".format(filename_roi_id)),
            "label_image": label_image_path,
            "has_labels": os.path.exists(label_image_path),
            "single_channels": {}
        }

        # Add paths to single-channel crop images:
        image_id = self.get_image_id_from_roi_id(roi_id)
        ch_names = ["Main channel", "DAPI"] + self.get("extra_channels_names")
        for i in range(2 + self.get("max_nb_extra_channels")):
            path = self._input_images_df.iloc[image_id, i + 1]
            # Check if channel is present, then add:
            if isinstance(path, str):
                out_dict["single_channels"][ch_names[i]] = \
                    os.path.join(base_ROI_dir, "napari_data/roi_images/{}_ch{}.tif".format(filename_roi_id, i))
            else:
                out_dict["single_channels"][ch_names[i]] = None

        return out_dict

    def update_roi_labels(self, roi_id, roi_labels):
        roi_info = self.get_training_image_paths(roi_id)
        write_image_to_file(roi_info["label_image"], roi_labels)

    def get_napari_label_file_path(self, roi_id):
        return os.path.join(self.experiment_directory, "ROIs/napari_data/napari_annotations",
                            "{:04d}_masks.tif".format(roi_id))

    # --------------------------------------------
    # Cellpose training:
    # --------------------------------------------

    def setup_cellpose_training_data(self, model_name, show_training_folder=False):
        training_folder = os.path.join(self.experiment_directory, "CellposeTraining", model_name)
        training_images_dir = os.path.join(training_folder, "training_images")

        # Create dirs, if not already present:
        os.makedirs(training_folder, exist_ok=True)
        os.makedirs(training_images_dir, exist_ok=True)

        # Write training config to file
        training_config = deepcopy(self.get("training_config"))
        training_config.pop("custom_model_path_GUI")
        training_config.pop("pretrained_model_GUI")
        training_config.pop("model_name")
        # Specify relative training path:
        training_config["train_folder"] = os.path.join("CellposeTraining", model_name, "training_images")
        training_config_path = os.path.join(training_folder, "train_config.yml")
        existing_training_config = yaml2dict(training_config_path) if os.path.exists(training_config_path) else {}
        existing_training_config = recursive_update(existing_training_config, training_config)
        with open(training_config_path, 'w') as f:
            yaml.dump(existing_training_config, f)

        # Delete and recopy training images:
        shutil.rmtree(training_images_dir)
        cellpose_input_images_dir = os.path.join(self.experiment_directory, "ROIs/cellpose_input")
        shutil.copytree(cellpose_input_images_dir, training_images_dir)

        # Collect labels from QuPath or Napari:
        if self.get("labeling_tool") == "QuPath":
            export_labels_from_qupath(self.qupath_directory, training_images_dir, filename_postfix="masks")
        elif self.get("labeling_tool") == "Napari":
            # TODO: Only copy actual existing ROIs
            # TODO: assert that all labels are present
            shutil.copytree(os.path.join(self.experiment_directory, "ROIs/napari_data/napari_annotations"),
                            training_images_dir, dirs_exist_ok=True)
        else:
            raise ValueError("Labeling tool not recognized")

        # Zip files and open the folder:
        shutil.make_archive(training_folder, 'zip', training_folder)
        if show_training_folder:
            open_path(os.path.join(self.experiment_directory, "CellposeTraining"))

    def run_cellpose_training(self, model_name):
        try:
            import cellpose
        except ImportError:
            return False, "cellpose module is required to train a custom model"

        # Assert that training data is present:
        training_folder = os.path.join(self.experiment_directory, "CellposeTraining", model_name)
        training_images_dir = os.path.join(training_folder, "training_images")
        training_config_path = os.path.join(training_folder, "train_config.yml")
        if not os.path.exists(training_folder) or not os.path.exists(training_images_dir) or not os.path.exists(
                training_config_path):
            self.setup_cellpose_training_data(model_name)

        # Load config:
        training_config = yaml2dict(training_config_path)

        # Temporary check:
        train_folder = os.path.join(self.experiment_directory, training_config.pop("train_folder"))
        assert train_folder == training_images_dir

        training_was_successful, error_message = \
            start_cellpose_training(train_folder,
                                    *training_config.get("cellpose_args", []),
                                    # out_models_folder=os.path.split(train_folder)[0],
                                    **training_config.get("cellpose_kwargs", {}))

        return training_was_successful, error_message

    def update_main_training_config(self,
                                    model_name,
                                    **GUI_training_kwargs
                                    ):
        """
        Function called from magicgui widget to update training parameters set via the GUI
        """
        # Prepare training config:
        training_config = {}
        training_config["model_name"] = model_name
        training_config["cellpose_kwargs"] = cellpose_kwargs = {}

        # Validate pretrained model kwargs:
        if "pretrained_model" not in GUI_training_kwargs:
            return False, "No pretrained model was specified or recognized"
        if "custom_model_path" not in GUI_training_kwargs:
            return False, "No custom model path was specified"
        training_config["pretrained_model_GUI"] = GUI_training_kwargs["pretrained_model"]
        training_config["custom_model_path_GUI"] = GUI_training_kwargs["custom_model_path"]

        if GUI_training_kwargs["pretrained_model"] == "None":
            # TODO: Check if this works in config
            cellpose_kwargs["pretrained_model"] = None
        elif GUI_training_kwargs["pretrained_model"] == "Custom model":
            custom_model_path = GUI_training_kwargs["custom_model_path"]
            if not os.path.isfile(custom_model_path):
                return False, "The path of the custom model was not found"
            cellpose_kwargs["pretrained_model"] = custom_model_path
        else:
            # Set model to one of the default cellpose models:
            cellpose_kwargs["pretrained_model"] = GUI_training_kwargs["pretrained_model"]

        # Validate other kwargs from GUI:
        for kwarg in ["batch_size", "n_epochs", "learning_rate"]:
            if kwarg in GUI_training_kwargs:
                cellpose_kwargs[kwarg] = GUI_training_kwargs[kwarg]

        # Write to main config:
        old_training_config = self.get("training_config", ensure_exists=True)
        old_training_config.update(training_config)
        self.set("training_config", old_training_config)
        self.dump_configuration()

        return True, None

    def set_default_training_args(self):
        training_config = {"model_name": "my_trained_model",
                           "cellpose_args": ["no_npy",
                                             # "save_each", # Save models at different epochs, based on "save_every"
                                             # "dir_above", # Only useful when saving images, not trained models
                                             "verbose"],
                           "pretrained_model_GUI": "cyto2",
                           "custom_model_path_GUI": ""}

        training_config["cellpose_kwargs"] = cellpose_kwargs = {}

        cellpose_kwargs["pretrained_model"] = "cyto2"
        # cellpose_kwargs["save_every"] = 10
        cellpose_kwargs["learning_rate"] = 0.2
        cellpose_kwargs["chan"] = 2
        cellpose_kwargs["chan2"] = 1
        cellpose_kwargs["n_epochs"] = 500
        cellpose_kwargs["batch_size"] = 8
        cellpose_kwargs["mask_filter"] = "_masks"

        self.set("training_config", training_config)
        self.dump_configuration()

    def get_training_parameters_GUI(self):
        return self.get("training_config")

    # --------------------------------------------
    # Internal methods:
    # --------------------------------------------
    @property
    def use_dapi_channel_for_segmentation(self):
        if self.get("training/use_dapi_channel_for_segmentation") is None:
            self.use_dapi_channel_for_segmentation = True
        return self.get("training/use_dapi_channel_for_segmentation")

    @use_dapi_channel_for_segmentation.setter
    def use_dapi_channel_for_segmentation(self, use_dapi_channel_for_segmentation):
        assert isinstance(use_dapi_channel_for_segmentation, bool)
        self.set("training/use_dapi_channel_for_segmentation", use_dapi_channel_for_segmentation)
        self.dump_configuration()

    @property
    def channel_names(self):
        return ["Main channel", "DAPI"] + self.get("extra_channels_names")

    @property
    def apply_preprocessing(self):
        if self.get("apply_preprocessing") is None:
            self.apply_preprocessing = False
        return self.get("apply_preprocessing")

    @apply_preprocessing.setter
    def apply_preprocessing(self, apply_preprocessing):
        assert isinstance(apply_preprocessing, bool)
        self.set("apply_preprocessing", apply_preprocessing)
        self.dump_configuration()

    def record_args(self):
        # Simulate sys.argv, so that configuration is loaded from the experiment directory:
        self._argv = self._simulated_sys_argv
        return self

    @property
    def experiment_directory(self):
        """Directory for the experiment."""
        return self._experiment_directory

    def set_default_preprocessing_config(self):
        preproc_config = {"main_segm_ch":
                              [{"function_kwargs": {},
                                "function_name": "traincellpose.preprocessing.normalize_image"
                                }],
                          "DAPI": []
                          }

        self.set("preprocessing", preproc_config)
        self.dump_configuration()

    @property
    def qupath_directory(self):
        return os.path.join(self.experiment_directory, 'QuPathProject')

    @experiment_directory.setter
    def experiment_directory(self, value):
        if value is not None:
            # Make directories
            os.makedirs(os.path.join(value, 'Configurations'), exist_ok=True)
            os.makedirs(os.path.join(value, 'local_input_images'), exist_ok=True)
            os.makedirs(os.path.join(value, 'ROIs'), exist_ok=True)
            os.makedirs(os.path.join(value, 'ROIs/napari_data/roi_images'), exist_ok=True)
            os.makedirs(os.path.join(value, 'ROIs/cellpose_input'), exist_ok=True)
            os.makedirs(os.path.join(value, 'ROIs/napari_data/napari_annotations'), exist_ok=True)
            os.makedirs(os.path.join(value, "CellposeTraining"), exist_ok=True)

            # Create QuPathProject:
            os.makedirs(os.path.join(value, 'QuPathProject'), exist_ok=True)
            add_image_to_project(os.path.join(value, 'QuPathProject'))
            os.makedirs(os.path.join(value, 'QuPathProject/input_images'), exist_ok=True)

            self._experiment_directory = value

    @property
    def logger(self):
        return logging.getLogger(__name__)

class InputImageDict():
    def __init__(self,
                 base_exp: BaseAnnotationExperiment,
                 dict_key = "input_images"):
        self.base_exp = base_exp
        self.dict_key = dict_key

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.base_exp.get(f"{self.dict_key}/{idx}", ensure_exists=True)
        elif isinstance(idx, tuple):
            assert len(idx) == 2
            return self.base_exp.get(f"{self.dict_key}/{idx[0]}/{idx[1]}", ensure_exists=True)
        else:
            raise ValueError(idx)

    def __setitem__(self, idx, value):
        assert isinstance(value, dict)
        if isinstance(idx, int):
            self.base_exp.set(f"{self.dict_key}/{idx}", value)
        elif isinstance(idx, tuple):
            assert len(idx) == 2
            self.base_exp.set(f"{self.dict_key}/{idx[0]}/{idx[1]}", value)
        else:
            raise ValueError(idx)
        self.base_exp.dump_configuration()

    def __len__(self):
        return len(self.base_exp.get(f"{self.dict_key}", {}))

    def get_all_image_ids(self):
        return [int(key) for key in self.base_exp.get(f"{self.dict_key}", {})]


