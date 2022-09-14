import pathlib
from copy import deepcopy
from pathlib import Path
import os
from typing import List, Union

import logging

import numpy as np

logger = logging.getLogger(__name__)

from magicgui.types import FileDialogMode
import magicgui.widgets as widgets
from magicgui.widgets import (
    Container,
    PushButton,
)
from psygnal import Signal

from .gui_utils import show_napari_error_message
from ..io.images import read_uint8_img, deduce_image_type, get_image_info_dict


def check_posix_path_is_empty(path: pathlib.PosixPath):
    path = str(path)
    return path == "." or path == ""


def return_choices():
    return ()


class RoiSelectionWidget(Container):
    def __init__(self,
                 main_gui: "traincellpose.gui_widgets.start_window.StartingGUI"):
        super(RoiSelectionWidget, self).__init__()
        self.main_gui = main_gui
        nb_images_in_proj = main_gui.project.nb_input_images
        self.image_id = None if nb_images_in_proj == 0 else 0
        self.edit_image_paths_mode_active = nb_images_in_proj == 0
        self.shape_layer = None
        self._path_widgets = None
        self._inner_ch_specs_widgets = None
        self._paths_dict = {}

        self._setup_gui()
        self.logger.setLevel("DEBUG")

    def _setup_gui(self, info_message=None):
        """
        TODO: instead of recreating all elemetns every time I update, I should probably just show/hide/update them...?
        """
        self.clear()
        self.reset_button_widgets()

        # ----------------------------
        # Create combo box to decide which image will be loaded:
        # ----------------------------
        self.logger.debug(f"Setting up ROI gui with image ID '{self.image_id}'")
        nb_images_in_proj = self.main_gui.project.nb_input_images

        current_choice = "Add new image to project" if self.image_id is None else "Image {}".format(self.image_id + 1)
        self.selected_image = widgets.ComboBox(
            # name="choose_image",
            label="Shown Image:",
            choices=["Image {}".format(i + 1) for i in range(nb_images_in_proj)
                     ] + ["Add new image to project"],
            value=current_choice
            # description=None
        )

        @self.selected_image.changed.connect
        def update_selected_image_id(choice):
            # First update ROIs, if user forgot:
            if self.image_id is not None:
                self.update_rois()

            # Find the ID of the selected image:
            self.image_id = None if choice == "Add new image to project" else int(choice.split(" ")[1]) - 1
            self.edit_image_paths_mode_active = choice == "Add new image to project"
            self._setup_gui()

        # selected_image_container = widgets.Container(widgets=[selected_image])
        self.append(self.selected_image)
        # self.append(widgets.Container())

        # Load info about current image:
        self.cur_img_info = {} if self.image_id is None else self.main_gui.project.input_images[self.image_id]

        # Create path-widgets and load images:
        err_messages = self.init_path_widgets()
        err_messages = [None for _ in range(self.nb_channels)] if err_messages is None else err_messages

        if self.edit_image_paths_mode_active:
            # ----------------------------
            # Create interface to enter channels files:
            # ----------------------------
            for i, (path_widg, inner_path_widg, ch_name) in enumerate(self.paths_widgets):
                # Only show widgets for channels that are already present in a project image
                # (QuPath does not support a change of inner channels at the moment):
                if (self.image_id is not None) and (self.cur_img_info.get(str(i), {}).get("image", None) is None) and \
                        (err_messages[i] is None):
                    path_widg.visible = False
                    inner_path_widg.visible = False
                else:
                    container = widgets.Container(
                        # layout="horizontal",
                        layout="vertical",
                        widgets=[path_widg, inner_path_widg],
                        labels=False,
                        label=ch_name)
                    self.extend([container])
                # self.extend([path_widg, inner_path_widg])

            # ----------------------------
            # Create buttons to save edited paths:
            # ----------------------------
            # self.select_roi_button = PushButton(name="select_roi", text="Select ROIs",
            #                                     visible=self.image_id is not None)
            #
            # @self.select_roi_button.changed.connect
            # def select_roi_callback():
            #     # In case the user did forget to update the paths:
            #     self.image_id = self.main_gui.project.add_input_image(self.cur_img_info,
            #                                                           id_input_image_to_rewrite=self.image_id)
            #
            #     # Reload GUI in ROI-edit mode:
            #     self.edit_image_paths_mode_active = False
            #     self._setup_gui()

            self.save_image_paths_button = PushButton(
                name="save_image_paths",
                text="Add selected images to annotation project" if self.image_id is None else
                "Update image paths in annotation project")

            @self.save_image_paths_button.changed.connect
            def save_image_paths_callback():
                # Check that some images were actually loaded in napari:
                dict_to_save_to_proj = self.check_images_loaded_in_img_dict()

                if dict_to_save_to_proj is not None:
                    self.image_id = self.main_gui.project.add_input_image(dict_to_save_to_proj,
                                                                          id_input_image_to_rewrite=self.image_id)
                    # if self.image_id is not None:
                    #     self.select_roi_button.visible = True
                    # self.image_id = None

                    # Reload widgets in edit-path mode for a new image:
                    self.edit_image_paths_mode_active = False
                    self._setup_gui()
                else:
                    show_napari_error_message("First, you should at least specify a valid image as main segmentation channel")

            self.extend([self.save_image_paths_button])

        else:
            assert self.image_id is not None

            # Hide path widgets:
            for path_widg, inner_path_widg, ch_name in self.paths_widgets:
                path_widg.visible = False
                inner_path_widg.visible = False

            # ---------------------------------------
            # Add buttons to go back to edit mode:
            # ---------------------------------------
            self.update_image_paths_button = PushButton(name="update_image_paths", text="Update channel paths for "
                                                                                        "selected Image")

            @self.update_image_paths_button.changed.connect
            def update_image_paths():
                # First update ROIs, if user forgot:
                self.update_rois()

                # Reload widgets in edit-path mode:
                self.edit_image_paths_mode_active = True
                self._setup_gui()

            self.add_new_image_button = PushButton(name="add_new_image", text="Add new image to project")

            @self.add_new_image_button.changed.connect
            def add_new_image():
                # First update ROIs, if user forgot:
                self.update_rois()

                # Reload widgets in edit-path mode for a new image:
                self.image_id = None
                self.edit_image_paths_mode_active = True
                self._setup_gui()

            self.extend([self.update_image_paths_button, self.add_new_image_button])

            # ----------------------------
            # Button to update the ROIs:
            # ----------------------------
            self.update_rois_button = PushButton(name="update_rois", text="Save Regions of Interest")

            @self.update_rois_button.changed.connect
            def update_rois():
                self.logger.debug("Clicked on update ROI button")
                self.update_rois()

            self.append(self.update_rois_button)

        # ----------------------------
        # Button to go back to the main GUI
        # ----------------------------
        self.close_button = PushButton(name="close_and_go_back", text="Go Back to Starting Window")

        @self.close_button.changed.connect
        def close_viewer_and_go_back():
            self.logger.debug("Close viewer")
            # First update ROIs, if user forgot:
            if self.image_id is not None:
                self.update_rois()

            self.main_gui.roi_select_viewer.close()
            self.main_gui.show()
            self.main_gui.show_starting_gui()

        self.extend([self.close_button])

        # Display message:
        if info_message is not None:
            assert isinstance(info_message, str)
            show_napari_error_message(info_message)
            # self.append(widgets.Label(value=info_message))

        self.logger.debug("Done creating napari gui")

    def reset_path_dict(self, image_id=None):
        if image_id is None:
            self._paths_dict = {}
        else:
            # TODO: get from project
            #   load layers in napari; set values for all path widgets
            pass

    def check_images_loaded_in_img_dict(self):
        img_info = deepcopy(self.cur_img_info)
        for ch_idx in self.cur_img_info:
            # Check if an image is currently loaded, otherwise delete the entry:
            image_data = img_info[ch_idx].get("image", None)
            if not isinstance(image_data, np.ndarray):
                img_info.pop(ch_idx)

        # Check if the main channel has been correctly loaded:
        return img_info if "0" in img_info else None

    def update_rois(self):
        assert self.image_id is not None
        # TODO: create method to get annotation layer
        # TODO: create class attribute
        shape_layer_name = self.roi_layer_name

        loaded_layer_names = [lay.name for lay in self.main_gui.roi_select_viewer.layers]
        self.logger.debug(f"Updating ROIs: {loaded_layer_names}")
        if shape_layer_name in loaded_layer_names:
            self.logger.debug(f"Updating ROIs - getting data: {loaded_layer_names}")
            shape_layer = self.main_gui.roi_select_viewer.layers[shape_layer_name]
            # idx = self.main_gui.roi_select_viewer.layers.index(shape_layer)
            # shapes = self.main_gui.roi_select_viewer.layers[idx].data
            shapes = shape_layer.data
            self.main_gui.project.update_rois_image(self.image_id, shapes)

    def get_path_file(self, real_path):
        assert isinstance(real_path, pathlib.Path)
        if os.path.isfile(real_path):
            return real_path.resolve().as_posix()
        else:
            return None

    def init_path_widgets(self):
        self._path_widgets = []
        self._inner_ch_specs_widgets = []
        img_info_dict = self.cur_img_info
        assert img_info_dict is not None

        total_nb_channels = len(self.channel_names)

        tooltips_path_widg = [
            "Indicate the image path of the main image to be segmented",
            "Indicate the image path of the DAPI channel",
            "Indicate the path of an extra channel to load",
            "Indicate the path a second extra channel to load"
        ]
        tooltips_inner_ch_specs_widg = [
            f"The selected image for `{ch}` is multichannel. Choose one channel to load." for ch in
            self.channel_names
        ]
        # labels_inner_ch_specs_widg = [
        #     f"`{ch}` is multichannel. Choose which channel to load in napari " for ch in self.channel_names
        # ]

        for i in range(total_nb_channels):
            # Construct path widget:
            ch_info_dict = img_info_dict.get(f"{i}", {})
            self._path_widgets.append(widgets.FileEdit(
                name=f"path_ch_{i}", tooltip=tooltips_path_widg[i],
                # mode=None,
                # mode='r',
                mode=FileDialogMode.EXISTING_FILE,
                label=self.channel_names[i],
                value=ch_info_dict.get("path", ""),
                visible=True
            ))

            # Construct comboBox showing available inner channels:
            # TODO: move to separate function?
            img_type = ch_info_dict.get("type", None)
            inner_channels = ch_info_dict.get("inner_channels", None)
            inner_channel_to_select = ch_info_dict.get("inner_channel_to_select", None)
            combo_box_options = {
                "name": f"inner-channel-specs_ch_{i}",
                "tooltip": tooltips_inner_ch_specs_widg[i],
                "label": " ",
            }

            def return_choices_combobox(combobox_widg):
                if len(combobox_widg.choices) == 0:
                    return ("",)
                else:
                    return combobox_widg.choices
                # if combobox_widg.visible or visible:
                #     return inner_channels
                # else:
                #     return ()

            visible = (img_type is not None) and (inner_channels is not None)
            combo_box_options["visible"] = visible
            # combo_box_options["choices"] = inner_channels if visible else ()
            combo_box_options["choices"] = return_choices_combobox
            inner_widg = widgets.ComboBox(**combo_box_options)
            if visible:
                for ch in inner_channels:
                    inner_widg.set_choice(ch)
                if inner_channel_to_select is not None: inner_widg.value = inner_channel_to_select
            self._inner_ch_specs_widgets.append(inner_widg)

        # Callback function when any widget is updated:
        def widget_update_callback(value: Union[pathlib.PosixPath, str]):
            wid_type, _, ch_idx = Signal.sender().name.split("_")
            ch_idx = int(ch_idx)
            self.logger.warning(f"Start updating path widgets: {wid_type}, {ch_idx}, {value}")

            image_data_has_been_updated = False
            error_msg = None
            img_info_dict = {}

            if wid_type == "path":
                update_image_data = True
                self.logger.debug(f"Updating path-widg {ch_idx}...")

                # FIXME: this will be fixed when we load info from Napari layers:
                if str(value).endswith(".zgroup") or str(value).endswith(".zarray"):
                    zarr_dir, _ = os.path.split(str(value))
                    value = pathlib.PosixPath(zarr_dir)

                # Check if a valid image-data was already loaded:
                old_img_info_dict = self.cur_img_info.get(str(ch_idx), {})
                old_image_data = old_img_info_dict.get("image", None)
                if old_image_data is not None and isinstance(old_image_data, np.ndarray):
                    image_data_has_been_updated = True
                    img_info_dict = {"path": str(value)}

                # Run further checks only if the new path exists at all:
                path_wid = self._path_widgets[ch_idx]
                path_wid.visible = True
                inner_specs_wid = self._inner_ch_specs_widgets[ch_idx]
                # Update img dict, since from now on I could trigger the inner-path-widget value:
                self.cur_img_info[str(ch_idx)] = img_info_dict
                if value.exists():
                    # Get the related widgets:

                    # Gather info about the image:
                    img_info_dict = get_image_info_dict(str(value))
                    # Update img dict, since from now on I could trigger the inner-path-widget value:
                    self.cur_img_info[str(ch_idx)] = img_info_dict
                    # Check if the path is a valid image that can be loaded:
                    if img_info_dict.get("type", None) is not None:
                        # error_msg = "Image not recognized / supported (only ome-zarr, tiff, or png)"
                        # else:
                        # anything_has_changed = True
                        available_inner_channels = img_info_dict.get("inner_channels", None)
                        # selected_inner_channel = inner_specs_wid.value
                        # inner_channel_is_set = isinstance(inner_channel, str) and inner_channel != ""

                        default_inner_channels_choices = {
                            0: "Trans",
                            1: "Dapi"
                        }

                        # ---------------
                        # Reset the ComboBox widget:
                        # ---------------
                        # Before to reset we update the list of inner channels:
                        # FIXME:  will it change it...?
                        self.logger.warning("Reset choices")
                        inner_specs_wid.visible = False
                        inner_specs_wid.is_being_updated = True
                        updated_choices_combo_box_widget(inner_specs_wid,
                                                         choices_to_keep=[""])
                        # inner_specs_wid.reset_choices()
                        if available_inner_channels is not None:
                            self.logger.warning("Add choice '':")
                            # inner_specs_wid.set_choice("")
                            for choice in available_inner_channels:
                                self.logger.warning(f"Add choice '{choice}':")
                                inner_specs_wid.set_choice(choice)

                            self.logger.warning("Make visible:")
                            inner_specs_wid.visible = True

                            # Possibly set the default selected inner-channel:
                            inner_specs_wid.is_being_updated = False
                            if ch_idx in default_inner_channels_choices and img_info_dict["type"] == "zarr":
                                # img_info_dict["inner_channel_to_select"] = default_inner_channels_choices[ch_idx]
                                # self.cur_img_info[str(ch_idx)] = img_info_dict
                                self.logger.warning(f"Set value to '{default_inner_channels_choices[ch_idx]}':")
                                inner_specs_wid.value = default_inner_channels_choices[ch_idx]
                                image_data_has_been_updated = True
                            else:
                                self.logger.warning(f"Set value to '':")
                                inner_specs_wid.value = ""

                        else:
                            inner_specs_wid.visible = False

                            # Try to load the image anyway:
                            img_info_dict["image"], error_msg = self.main_gui.project.load_channel_img(img_info_dict,
                                                                                                       return_error_message=True)
                            if error_msg is None:
                                image_data_has_been_updated = True

                        if error_msg is not None and not check_posix_path_is_empty(path_wid.value):
                            # Reset widgets values and show error message:
                            path_wid.value = ''
                            updated_choices_combo_box_widget(inner_specs_wid,
                                                             choices_to_keep=[""])
                            # inner_specs_wid.reset_choices()
                            inner_specs_wid.visible = False
                            show_napari_error_message(error_msg)
                        elif image_data_has_been_updated and ch_idx == 0 and img_info_dict.get("type", None) == "zarr":
                            # Try to set DAPI channel automatically if we have zarr:
                            dapi_path_wid = self._path_widgets[1]
                            if check_posix_path_is_empty(dapi_path_wid.value):
                                dapi_path_wid.value = value
                        inner_specs_wid.is_being_updated = False
                    else:
                        updated_choices_combo_box_widget(inner_specs_wid,
                                                         choices_to_keep=[""])
                        inner_specs_wid.visible = False
                else:
                    updated_choices_combo_box_widget(inner_specs_wid,
                                                     choices_to_keep=[""])
                    inner_specs_wid.visible = False
                self.logger.debug(f"DONE Updating path-widg {ch_idx}...")
            elif wid_type == "inner-channel-specs":
                update_image_data = True
                self.logger.debug(f"Updating inner-chanhel {ch_idx}...")
                inner_specs_wid = self._inner_ch_specs_widgets[ch_idx]

                # In any case, we first remove previous data:
                img_info_dict = self.cur_img_info.get(str(ch_idx), {})
                img_info_dict.pop("image", None)
                img_info_dict.pop("inner_channel_to_select", None)

                # # Check if widget is empty, although this should not happen
                # # because I can only update the widg value if there is at least one value:
                # if len(inner_specs_wid.choices) == 0:
                #     img_info_dict.pop("image", None)
                #     img_info_dict.pop("inner_channel_to_select", None)
                #     update_image_data = True

                # From now we rely on the value of `inner_channels` in the dict:
                #  - If it is not present, it means that probably we are in the process of removing the widget options
                #         (during which widg value could be updated several times) so we should not load any image
                #  - If it is there, instead we will try to load the image with the current value
                inner_channels = img_info_dict.get("inner_channels", [])

                # If no channels are given, we ignore the current value of the widget and
                # just delete previous image data. Otherwise, we try to load the image:
                if len(inner_channels) != 0:
                    # Save widget value in the dictionary:
                    img_info_dict["inner_channel_to_select"] = value
                    if value != "":
                        if value in inner_channels:
                            img_info_dict["image"], error_msg = self.main_gui.project.load_channel_img(img_info_dict,
                                                                                                       return_error_message=True)
                            if error_msg is not None:
                                show_napari_error_message(error_msg)
                                inner_specs_wid.value = ""
                            # else:
                            #     image_data_has_been_updated = True
                        # else:
                        #     # It looks like the image is no longer multichannel, so widget should be hidden:
                        #     inner_specs_wid.visible = False

                    # # Check if something went wrong loading the image:
                    # if not image_data_has_been_updated:
                    #     # Then we set the widget to the default value:
                    #     img_info_dict["inner_channel_to_select"] = None
                    #     # self.cur_img_info[str(ch_idx)] = img_info_dict
                    #     # inner_specs_wid.value = ""

                self.logger.debug(f"DONE Updating inner-channel {ch_idx}...")
            else:
                raise ValueError(wid_type)

            # if update_image_data:
            self.cur_img_info[str(ch_idx)] = img_info_dict

            # if image_data_has_been_updated:
            # Finally, reload napari layer:
            error_msg = self.load_napari_layers(ch_indices=ch_idx,
                                                try_to_load_image=False,
                                                update_roi_layer=False)
            if error_msg is not None:
                show_napari_error_message(error_msg[ch_idx])

        # Link widgets to callback function:
        for wid in self._path_widgets + self._inner_ch_specs_widgets:
            wid.changed.connect(widget_update_callback)

        # Load all given napari layers:
        err_messages = self.load_napari_layers(ch_indices=[i for i in range(self.nb_channels)],
                                               try_to_load_image=True,
                                               update_roi_layer=True)
        if err_messages is not None:
            for idx, err in enumerate(err_messages):
                if err is not None:
                    # show_napari_error_message(err)
                    show_napari_error_message(f"Channel {self.channel_names[idx]} not found! Check and "
                                        f"edit the image paths: {err}")
        return err_messages

    def load_napari_layers(self,
                           ch_indices: Union[int, List[int]] = None,
                           try_to_load_image: bool = False,
                           update_roi_layer: bool = False):
        # Assert inputs:
        ch_indices = [] if ch_indices is None else ch_indices
        ch_indices = [ch_indices] if isinstance(ch_indices, int) else ch_indices
        assert isinstance(ch_indices, (list, tuple))
        assert all(isinstance(ch, int) for ch in ch_indices)

        # Get napari viewer:
        viewer = self.main_gui.roi_select_viewer
        layers = viewer.layers
        loaded_layer_names = [lay.name for lay in layers]
        channel_colormaps = ["gray", "red", "yellow", "cyan"]

        img_info_dict = self.cur_img_info
        error_msg_collected = [None for _ in range(self.nb_channels)]
        for ch_idx, layer_name in enumerate(self.channel_names):
            if ch_idx in ch_indices:
                remove_layer = True
                if str(ch_idx) in img_info_dict:
                    ch_info_dict = img_info_dict[str(ch_idx)]
                    if try_to_load_image:
                        path = ch_info_dict.get("path", None)
                        inner_channel_to_select = ch_info_dict.get("inner_channel_to_select", None)
                        if (path is not None and path != ""):
                            # and \
                            # (inner_channel_to_select is not None and inner_channel_to_select != ""):
                            ch_info_dict["image"], error_msg = \
                                self.main_gui.project.load_channel_img(ch_info_dict,
                                                                       return_error_message=True)
                            error_msg_collected[ch_idx] = error_msg
                    image_data = ch_info_dict.get("image", None)
                    if image_data is not None and isinstance(image_data, np.ndarray):
                        remove_layer = False
                        if layer_name in loaded_layer_names:
                            # layer_idx = layers.index(layer_name)
                            layers[layer_name].data = image_data
                        else:
                            viewer.add_image(image_data,
                                             name=layer_name,
                                             colormap=channel_colormaps[ch_idx],
                                             blending='additive'
                                             )
                        layers[layer_name].contrast_limits = (image_data.min(), image_data.max())
                        # new_layer.contrast_limits = (image_data.min(), image_data.max())
                        # new_layer.contrast_limits_range = (image_data.min(), image_data.max())

                # Check if I should remove a layer:
                if remove_layer and (layer_name in loaded_layer_names):
                    viewer.layers.remove(layer_name)

        # Now load the ROIs:
        shape_layer_name = "Regions of interest"
        shape_layer_is_visible = shape_layer_name in loaded_layer_names
        if self.edit_image_paths_mode_active and shape_layer_is_visible:
            layers.remove(shape_layer_name)
        elif update_roi_layer and not self.edit_image_paths_mode_active:
            if shape_layer_is_visible:
                layers.remove(shape_layer_name)
            assert self.image_id is not None
            napari_rois = self.main_gui.project.get_napari_roi_by_image_id(self.image_id)
            viewer.add_shapes(data=napari_rois, name=shape_layer_name,
                              shape_type="rectangle", opacity=0.60, edge_color='#00007fff',
                              face_color='#0055ffff')
            # shape_type="rectangle", opacity=0.15, edge_color='#fff01dff',
            # face_color='#f9ffbeff')

        if any(err is not None for err in error_msg_collected):
            return error_msg_collected

    def reset_button_widgets(self):
        self.selected_image = None
        self.save_image_paths_button = None
        self.update_rois_button = None
        self.close_button = None
        self.update_image_paths_button = None
        self.add_new_image_button = None

    @property
    def paths_widgets(self):
        assert self._path_widgets is not None

        # if zip_output:
        return [i for i in zip(self._path_widgets, self._inner_ch_specs_widgets, self.channel_names)]
        # else:
        #     return self._path_widgets, self._inner_ch_specs_widgets, self.channel_names

    @property
    def channel_names(self):
        return self.main_gui.project.channel_names

    @property
    def nb_channels(self):
        return len(self.channel_names)

    @property
    def roi_layer_name(self):
        return "Regions of interest"

    @property
    def logger(self):
        return logging.getLogger(__name__)


# class CustomComboBox(widgets.ComboBox):
#     is_being_updated: bool = False


def updated_choices_combo_box_widget(widget: widgets.ComboBox,
                                     choices_to_keep: List = None,
                                     choices_to_add: List = None):
    choices_to_keep = choices_to_keep if isinstance(choices_to_keep, list) else []
    assert isinstance(choices_to_keep, list)
    choices_to_add = choices_to_add if isinstance(choices_to_add, list) else []
    assert isinstance(choices_to_add, list)
    assert isinstance(widget, widgets.ComboBox)
    for current_choice in widget.choices:
        if current_choice not in choices_to_keep:
            widget.del_choice(current_choice)

    for new_choice in choices_to_add:
        widget.set_choice(new_choice)

    return widget
