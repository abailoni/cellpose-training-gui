import pathlib
from pathlib import Path
import os

import logging
logger = logging.getLogger(__name__)


from magicgui.types import FileDialogMode
import magicgui.widgets as widgets
from magicgui.widgets import (
    Container,
    PushButton,
)

from ..io.images import read_uint8_img


class RoiSelectionWidget(Container):
    def __init__(self,
                 main_gui: "traincellpose.gui_widgets.start_window.StartingGUI"):
        super(RoiSelectionWidget, self).__init__()
        self.main_gui = main_gui
        nb_images_in_proj = main_gui.project.nb_input_images
        self.image_id = None if nb_images_in_proj == 0 else 0
        self.shape_layer = None
        self._setup_gui()

    def _setup_gui(self, info_message=None):
        self.clear()

        # ----------------------------
        # Create combo box to decide which image will be loaded:
        # ----------------------------
        self.logger.debug(f"Setting up ROI gui with image ID '{self.image_id}'")
        nb_images_in_proj = self.main_gui.project.nb_input_images

        current_choice = "Add new image" if self.image_id is None else "Image {}".format(self.image_id+1)
        self.selected_image = widgets.ComboBox(
            # name="choose_image",
            label="Shown Image:",
            choices=["Image {}".format(i+1) for i in range(nb_images_in_proj)
                     ] + ["Add new image"],
            value=current_choice
            # description=None
        )

        @self.selected_image.changed.connect
        def update_selected_image_id():
            self.logger.debug("Updated image choice")
            # First update ROIs, if user forgot:
            if self.image_id is not None:
                self.update_rois()

            # Find the ID of the selected image:
            choice = self.selected_image.value
            self.image_id = None if choice == "Add new image" else int(choice.split(" ")[1]) - 1
            self._setup_gui()

        # selected_image_container = widgets.Container(widgets=[selected_image])
        self.append(self.selected_image)
        # self.append(widgets.Container())

        # ----------------------------
        # Create interface to enter channels file paths:
        # ----------------------------
        self.main_ch = widgets.FileEdit(
            name="main_channel", tooltip="Select the main channel that should be segmented",
            mode=FileDialogMode.EXISTING_FILE,
            label=self.channel_names[0], value=""
        )
        self.dapi_ch = widgets.FileEdit(
            name="dapi_channel", tooltip="Select the DAPI channel", mode=FileDialogMode.EXISTING_FILE,
            label=self.channel_names[1], value=""
        )
        # widgets.LineEdit(label="Extra ch1 name")
        self.extra_ch_1 = widgets.FileEdit(
            name="extra_ch_1", tooltip="Select an extra channel to load", mode=FileDialogMode.EXISTING_FILE,
            label=self.channel_names[2], value=""
        )
        self.extra_ch_2 = widgets.FileEdit(
            name="extra_ch_2", tooltip="Select a second extra channel to load", mode=FileDialogMode.EXISTING_FILE,
            label=self.channel_names[3], value=""
        )

        # Add button to load the channels and save them in the project:
        load_images_button = PushButton(name="load_images",
                                        text="Update Image Channels" if self.image_id is not None else
                                        "Load Image and Add it to the Project")
        add_new_image_button = PushButton(name="add_image", text="Add new image to project")

        @load_images_button.changed.connect
        def update_image_paths():
            self.logger.debug("Updated image path button")
            # Retrieve and validate paths:
            main_ch = self.get_path_file(self.main_ch.value)
            dapi_ch = self.get_path_file(self.dapi_ch.value)
            extra_ch_1 = self.get_path_file(self.extra_ch_1.value)
            extra_ch_2 = self.get_path_file(self.extra_ch_2.value)

            if main_ch is not None:
                if os.path.isfile(main_ch):
                    # Save the given paths in the project:
                    self.image_id = self.main_gui.project.add_input_image(
                        main_ch,
                        dapi_path=dapi_ch,
                        extra_ch_1_path=extra_ch_1,
                        extra_ch_2_path=extra_ch_2,
                        id_input_image_to_rewrite=self.image_id,
                    )

                    # Now we reload the interface of the widget:
                    self._setup_gui()
                    # TODO: after updating, select shape layer and the right tool!
                    return

            # Otherwise, show info message:
            self._setup_gui(info_message="Path to main channel not existing!")

        @add_new_image_button.changed.connect
        def add_new_image():
            self.logger.debug("Clicked on add new image")
            # First update ROIs, if user forgot:
            if self.image_id is not None:
                self.update_rois()

            # Then show interface to add new image:
            self.image_id = None
            self._setup_gui()

        # ----------------------------
        # If an image in the project was selected, load data in the viewer:
        # ----------------------------
        path_widgets_to_show = []
        if self.image_id is not None:
            image_paths = self.main_gui.project.get_image_paths(image_id=self.image_id)

            for ch_name, wid in zip(self.channel_names, self.get_list_path_widgets()):
                if ch_name in image_paths:
                    path = Path(image_paths[ch_name])
                    wid.value = path
                    path_widgets_to_show.append(wid)
                    # print(path.resolve().as_posix(), wid.value)
        else:
            path_widgets_to_show = self.get_list_path_widgets()

        self.extend(path_widgets_to_show)
        self.append(load_images_button)

        # Now load images in the viewer:
        self.load_images_in_viewer()

        # ----------------------------
        # Add additional buttons:
        # ----------------------------
        # Button to update the ROIs:
        update_rois_button = PushButton(name="update_rois", text="Save Regions of Interest")

        @update_rois_button.changed.connect
        def update_rois():
            self.logger.debug("Clicked on update ROI button")
            self.update_rois()

        if self.image_id is not None:
            self.extend([update_rois_button,
                         widgets.Label(label=""),
                         add_new_image_button])
        else:
            self.append(widgets.Label(label=""))

        # Button to go back to the main
        close_button = PushButton(name="close_and_go_back", text="Go Back to Starting Window")
        @close_button.changed.connect
        def close_viewer_and_go_back():
            self.logger.debug("Close viewer")
            # First update ROIs, if user forgot:
            if self.image_id is not None:
                self.update_rois()

            self.main_gui.roi_select_viewer.close()
            self.main_gui.show()
            self.main_gui.show_starting_gui()
        self.extend([close_button])

        # Display message:
        if info_message is not None:
            assert isinstance(info_message, str)
            self.append(widgets.Label(value=info_message))

        self.logger.debug("Done creating napari gui")


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

    def load_images_in_viewer(self):
        viewer = self.main_gui.roi_select_viewer
        layers = viewer.layers
        loaded_layer_names = [lay.name for lay in layers]
        channel_colormaps = ["gray", "red", "yellow", "cyan"]
        for i, wid in enumerate(self.get_list_path_widgets()):
            layer_name = self.channel_names[i]
            if os.path.isfile(wid.value):
                image_data = read_uint8_img(self.get_path_file(wid.value))[..., 0]

                if layer_name in loaded_layer_names:
                    layer_idx = layers.index(layer_name)
                    layers[layer_idx].data = image_data
                else:
                    viewer.add_image(image_data, name=layer_name,
                                     colormap=channel_colormaps[i],
                                     blending='additive')
            else:
                # Check if I should remove a layer:
                if layer_name in loaded_layer_names:
                    layer = viewer.layers[layer_name]
                    viewer.layers.remove(layer)


        # Now load the ROIs:
        layers = viewer.layers
        loaded_layer_names = [lay.name for lay in layers]
        shape_layer_is_visible = self.roi_layer_name in loaded_layer_names
        self.logger.debug(f"Loading layers: {loaded_layer_names}")
        if shape_layer_is_visible:
            self.logger.debug(f"Removing ROI layer")
            shape_layer = viewer.layers[self.roi_layer_name]
            layers.remove(shape_layer)
        if self.image_id is not None:
            self.logger.debug(f"Creating ROI layer")
            napari_rois = self.main_gui.project.get_napari_roi_by_image_id(self.image_id)
            viewer.add_shapes(data=napari_rois, name=self.roi_layer_name,
                                                 shape_type="rectangle", opacity=0.15, edge_color='#fff01dff',
                                                 face_color='#f9ffbeff')


    @property
    def channel_names(self):
        return ["Main channel", "DAPI", "Extra ch. 1", "Extra ch. 2"]

    def get_list_path_widgets(self):
        return [self.main_ch, self.dapi_ch, self.extra_ch_1, self.extra_ch_2]

    @property
    def roi_layer_name(self):
        return "Regions of interest"

    @property
    def logger(self):
        return logging.getLogger(__name__)
