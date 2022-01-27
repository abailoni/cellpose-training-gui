import pathlib
from pathlib import PosixPath
import os

import napari
from magicgui.types import FileDialogMode
import magicgui.widgets as widgets
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FloatSlider,
    PushButton,
    SpinBox,
    show_file_dialog,
)

from segmfriends.io.images import read_uint8_img


class RoiSelectionWidget(Container):
    def __init__(self,
                 main_gui: "annotationtools.gui_widgets.start_window.StartWindow",
                 image_id=None):
        super(RoiSelectionWidget, self).__init__()
        self.main_gui = main_gui
        self.image_id = image_id
        self.shape_layer = None
        self._setup_gui()

    def _setup_gui(self):
        self.clear()

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
        load_images_button = PushButton(name="load_images", text="Load/update image channels")

        @load_images_button.changed.connect
        def update_image_paths():
            # Retrieve and validate paths:
            main_ch = self.get_path_file(self.main_ch.value)
            dapi_ch = self.get_path_file(self.dapi_ch.value)
            extra_ch_1 = self.get_path_file(self.extra_ch_1.value)
            extra_ch_2 = self.get_path_file(self.extra_ch_2.value)

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

        self.extend(self.get_list_path_widgets())
        self.append(load_images_button)

        # ----------------------------
        # If an image in the project was selected, load data in the viewer:
        # ----------------------------
        if self.image_id is not None:
            image_paths = self.main_gui.project.get_image_paths(image_id=self.image_id)
            print(image_paths)

            for ch_name, wid in zip(self.channel_names, self.get_list_path_widgets()):
                if ch_name in image_paths:
                    path = PosixPath(image_paths[ch_name])
                    wid.value = path
                    # print(path.resolve().as_posix(), wid.value)

            # Now load images in the viewer:
            self.load_images_in_viewer()

        # ----------------------------
        # Add additional buttons:
        # ----------------------------
        # Button to update the ROIs:
        update_rois_button = PushButton(name="update_rois", text="Update regions of interest")

        @update_rois_button.changed.connect
        def update_rois():
            assert self.image_id is not None
            shapes = self.shape_layer.data
            if len(shapes):
                self.main_gui.project.update_rois_image(self.image_id, shapes)

        print(self.image_id)
        if self.image_id is not None:
            self.append(update_rois_button)

        # Button to go back to the main
        close_button = PushButton(name="close_and_go_back", text="Close viewer and go back")
        @close_button.changed.connect
        def close_viewer_and_go_back():
            self.main_gui.roi_select_viewer.close()
            self.main_gui.show()
            self.main_gui.show_starting_gui()
        self.append(close_button)

    def get_path_file(self, real_path):
        assert isinstance(real_path, pathlib.PosixPath)
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
                    viewer.layers.remove(layer_name)


        # Now load the ROIs:
        if self.image_id is not None:
            napari_rois = self.main_gui.project.get_napari_roi_by_image_id(self.image_id)

            if self.shape_layer is None:
                self.shape_layer = viewer.add_shapes(data=napari_rois, name="Regions of interest",
                                                     shape_type="rectangle", opacity=0.15, edge_color='#fff01dff',
                                                     face_color='#f9ffbeff')
            else:
                if napari_rois is not None:
                    self.shape_layer.data = napari_rois

    @property
    def channel_names(self):
        return ["Main channel", "DAPI", "Extra ch. 1", "Extra ch. 2"]

    def get_list_path_widgets(self):
        return [self.main_ch, self.dapi_ch, self.extra_ch_1, self.extra_ch_2]
