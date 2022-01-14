import math

import os
import numpy as np

from speedrun import BaseExperiment
from shutil import copyfile

import pandas
from segmfriends.utils import readHDF5, writeHDF5, check_dir_and_create

# TODO: get list of images and rois per image
#   check if number of channels are consistent...?
#


class BaseAnnotationExperiment(BaseExperiment):
    def __init__(self, experiment_directory):
        super(BaseAnnotationExperiment, self).__init__(experiment_directory)

        # TODO: leave possibility to load initial configs...?
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

        self.create_directories()

        # Set default values:
        if not load_prev_experiment:
            self.set("max_nb_extra_channels", 2)
            self.set("extra_channels_names", ["Extra channel 1", "Extra channel 2"])

        # Initialize or load dataframes:
        self._rois_df = None
        self._input_images_df = None
        self._init_rois()
        self._init_input_images_df()


    # --------------------------------------------
    # ROIs:
    # --------------------------------------------

    def update_rois_image(self, image_id, new_napari_rois):
        assert new_napari_rois.ndim == 3
        assert new_napari_rois.shape[1] == 4 and new_napari_rois.shape[2] == 2, "ROI array does not have the correct shape"
        nb_added_rois = new_napari_rois.shape[0]

        # Get IDs of previous ROIs:
        prev_roi_ids = self._get_roi_ids_by_image_id(image_id)
        current_max_roi_id = self._napari_rois.shape[0]

        # FIXME: by deleting everytime I update the ROIs, I could lose annotations!!
        #    check if a ROI is already present (same exact points) and in that case leave it there
        # Possibly, delete previous ROIs:
        # Get previous napari rois:
        prev_napari_rois = self._napari_rois[prev_roi_ids]
        check_rois = np.array([[np.allclose(new_roi, old_roi) for old_roi in prev_napari_rois]
                               for new_roi in new_napari_rois])
        # Add new ROIs:
        rois_not_already_in_project = ~ np.any(check_rois, axis=1)
        self._napari_rois = np.concatenate([self._napari_rois, new_napari_rois[rois_not_already_in_project]])
        for i in range(current_max_roi_id, current_max_roi_id + rois_not_already_in_project.sum()):
            self._rois_df.loc[i] = [i, image_id]

        # Remove ROIs that are not present anymore:
        old_rois_to_be_deleted = ~ np.any(check_rois, axis=0)
        self._delete_roi_ids(list(np.array(prev_roi_ids)[old_rois_to_be_deleted]))

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
            return self._napari_rois[rois_ids]
        else:
            return None


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

    def _init_rois(self):
        if self._rois_df is None:
            rois_csv_path = os.path.join(self.experiment_directory, "rois/rois.csv")
            rois_hdf5_path = os.path.join(self.experiment_directory, "rois/rois.hdf5")
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
        rois_dir_path = os.path.join(proj_dir, "rois")
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
        if not isinstance(channels_names, list):
            assert isinstance(channels_names, str)
            channels_names = [channels_names]
        assert len(channels_names) <= self.get("max_nb_extra_channels")
        new_names = self.get("extra_channels_names")
        for i, ch_name in enumerate(channels_names):
            new_names[i] = ch_name
        self.set("extra_channels_names", new_names)

    def get_input_image_id_from_path(self, main_image_path):
        raise NotImplementedError

    def get_image_paths(self, image_id):
        """
        Return a dictionary with the paths for each channel. The key of the dictionary is the channel name.
        """
        if isinstance(image_id, str):
            image_id = self.get_input_image_id_from_path(image_id)
        assert image_id < self.nb_input_images, "Image ID not present in project"
        image_data = self._input_images_df.loc[self._input_images_df["image_id"] == image_id]
        ch_names = ["Main channel", "DAPI"] + self.get("extra_channels_names")
        out_dict = {}
        for i in range(2 + self.get("max_nb_extra_channels")):
            path = image_data.iloc[0, i+1]
            if isinstance(path, str):
                out_dict[ch_names[i]] = path
        return out_dict

    def add_input_image(self,
                        main_image_path,
                        main_image_filter=None,
                        dapi_path=None,
                        dapi_filter=None,
                        extra_ch_1_path=None,
                        extra_ch_1_filter=None,
                        extra_ch_2_path=None,
                        extra_ch_2_filter=None,
                        id_input_image_to_rewrite=None,
                        **extra_channels_kwargs
                        ):
        """
        # TODO: add option to remove input image? In that case, I need to update self.nb_input_images
        """
        # TODO: generalize to multiple extra channels
        assert len(extra_channels_kwargs) == 0, "Extra channels are not supported yet"

        # Validate main image path:
        assert os.path.isfile(main_image_path), "'{}' is not a file!"

        def validate_ch_paths(ch_path, name_filter):
            ch_path = None if ch_path == "" else ch_path
            name_filter = None if name_filter == "" else name_filter
            if ch_path is not None:
                assert os.path.isfile(ch_path), "'{}' is not a file!"
            else:
                if name_filter is not None:
                    assert isinstance(main_image_filter,
                                      str) and main_image_filter != "", "Please insert a proper filter string for main image"
                    assert isinstance(name_filter,
                                      str) and name_filter != "", "Wrong format for filter '{}'".format(name_filter)
                    ch_path = main_image_path.replace(main_image_filter, name_filter)
                    assert os.path.isfile(ch_path), "'{}' is not a file!"
            return ch_path

        # Validate DAPI image:
        dapi_image_path = validate_ch_paths(dapi_path, dapi_filter)
        if self.get("training/use_dapi_channel_for_segmentation"):
            assert dapi_image_path is not None, "Missing path of DAPI image. If not available, set `use_dapi_channel_for_segmentation`" \
                                                "to False."

        # Validate extra channels:
        extra_ch_1_path = validate_ch_paths(extra_ch_1_path, extra_ch_1_filter)
        extra_ch_2_path = validate_ch_paths(extra_ch_2_path, extra_ch_2_filter)

        # Add new image:
        image_info = [main_image_path, dapi_image_path, extra_ch_1_path, extra_ch_2_path]
        nb_input_images = self.nb_input_images

        # Check if main image has already been added:
        matching_images = self._input_images_df.index[self._input_images_df["main_path"] == main_image_path].tolist()
        assert len(matching_images) <= 1
        if len(matching_images) == 1:
            print("The added image was already present in the project. Updating paths.")
            id_input_image_to_rewrite = matching_images[0]

        if id_input_image_to_rewrite is not None:
            assert id_input_image_to_rewrite < nb_input_images
        added_image_id = nb_input_images if id_input_image_to_rewrite is None else id_input_image_to_rewrite
        self._input_images_df.loc[added_image_id] = [added_image_id] + image_info

        self.dump_input_images_info()

        return added_image_id


    def dump_input_images_info(self):
        # Write data to file:
        proj_dir = self.experiment_directory
        rois_dir_path = os.path.join(proj_dir, "rois")
        input_images_csv_path = os.path.join(rois_dir_path, "input_images.csv")
        self._input_images_df.to_csv(input_images_csv_path, index=False)

        # Dump general configuration:
        self.dump_configuration()

    @property
    def nb_input_images(self):
        assert self._input_images_df is not None
        nb_input_images = self._input_images_df["image_id"].max()
        return 0 if math.isnan(nb_input_images) else nb_input_images + 1


    def _init_input_images_df(self):
        if self._input_images_df is None:
            input_images_csv_path = os.path.join(self.experiment_directory, "rois/input_images.csv")
            if os.path.exists(input_images_csv_path):
                self._input_images_df = pandas.read_csv(input_images_csv_path, index_col=None)
                # TODO: remove image_id...?
                self._input_images_df.sort_values("image_id")
                self._input_images_df.reset_index(drop=True)
                # Make sure that index and image ID are the same, otherwise adding images will not work properly:
                assert all([idx == row["image_id"] for idx, row in self._input_images_df.iterrows()])
            else:
                columns_names = ["image_id",
                                 "main_path",
                                 "DAPI_path"]
                columns_names += ["extra_ch_{}_path".format(i) for i in range(self.get("max_nb_extra_channels"))]
                self._input_images_df = pandas.DataFrame(columns=columns_names)

    # --------------------------------------------
    # Internal methods:
    # --------------------------------------------
    @property
    def use_dapi_channel_for_segmentation(self):
        return self.get("training/use_dapi_channel_for_segmentation")


    @use_dapi_channel_for_segmentation.setter
    def use_dapi_channel_for_segmentation(self, use_dapi_channel_for_segmentation):
        assert isinstance(use_dapi_channel_for_segmentation, bool)
        self.set("training/use_dapi_channel_for_segmentation", use_dapi_channel_for_segmentation)

    def create_directories(self):
        check_dir_and_create(os.path.join(self.experiment_directory, "rois"))

    def record_args(self):
        # Simulate sys.argv, so that configuration is loaded from the experiment directory:
        self._argv = self._simulated_sys_argv
        return self
