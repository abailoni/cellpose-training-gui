import imageio
import numpy as np

import magicgui.widgets as widgets
from magicgui.widgets import (
    Container,
    PushButton,
)

from ..io.images import read_uint8_img


class RoiLabeling(Container):
    def __init__(self,
                 main_gui: "spacem_annotator.gui_widgets.start_window.StartingGUI"):
        super(RoiLabeling, self).__init__()
        self.roi_id = 0
        self.main_gui = main_gui
        self.annotation_layer_name = "Annotations"
        self._setup_gui()

    def _setup_gui(self):
        self.clear()

        # ----------------------------
        # Create ComboBox to select the ROI to label:
        # ----------------------------
        rois_list = self.main_gui.project.get_roi_list()

        self.roi_to_annotate = widgets.ComboBox(
            label="Shown Region of Interest:",
            choices=[roi_idx for roi_idx in range(len(rois_list))],
            value=self.roi_id
            # description=None
        )

        @self.roi_to_annotate.changed.connect
        def update_roi_id():
            # Save labels, in case the user forgot to click:
            self.update_labels()

            # Now display another ROI:
            rois_indx = self.roi_to_annotate.value
            self.roi_id = rois_list[rois_indx]['roi_id']
            self.load_images_in_viewer()

        self.append(self.roi_to_annotate)

        # ----------------------------
        # Add button to save labels:
        # ----------------------------
        save_labels = PushButton(name="save_labels", text="Save Annotations")

        @save_labels.changed.connect
        def update_labels():
            self.update_labels()

        close_button = PushButton(name="close_and_go_back", text="Go Back to Starting Window")

        @close_button.changed.connect
        def close_viewer_and_go_back():
            # Save labels, in case the user forgot:
            self.update_labels()

            # Now close and go back:
            self.main_gui.roi_select_viewer.close()
            self.main_gui.show()
            self.main_gui.show_starting_gui()

        self.extend([save_labels, close_button])

        self.load_images_in_viewer()

    def update_labels(self):
        viewer = self.main_gui.roi_select_viewer
        annotation_layer = viewer.layers[viewer.layers.index(self.annotation_layer_name)]
        self.main_gui.project.update_roi_labels(self.roi_id,
                                                annotation_layer.data.astype('uint16'))

    def load_images_in_viewer(self):
        viewer = self.main_gui.roi_select_viewer
        layers = viewer.layers
        loaded_layer_names = [lay.name for lay in layers]
        channel_colormaps = ["gray", "red", "yellow", "cyan"]

        roi_paths = self.main_gui.project.get_training_image_paths(self.roi_id)

        # composite_image = read_uint8_img(roi_paths["composite_image"])
        # composite_image.shape

        # TODO: update layers instead of replacing them?
        # Remove previous layers:
        for name in roi_paths["single_channels"]:
            if name in loaded_layer_names:
                layers.remove(name)

        # Now add new layers:
        for i, name in enumerate(roi_paths["single_channels"]):
            if roi_paths["single_channels"][name] is not None:
                image = read_uint8_img(roi_paths["single_channels"][name])
                viewer.add_image(image[..., 0],
                                 name=name,
                                 colormap=channel_colormaps[i],
                                 blending="additive")

        annotations = imageio.imread(roi_paths["label_image"]) if roi_paths["has_labels"] else np.zeros(
            shape=image.shape[:2], dtype='uint16')

        # Load images in napari:
        if self.annotation_layer_name in loaded_layer_names:
            layers.remove(self.annotation_layer_name)
        viewer.add_labels(annotations, name=self.annotation_layer_name)
