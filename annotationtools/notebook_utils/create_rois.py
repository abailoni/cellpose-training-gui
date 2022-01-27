import ipywidgets as widgets
from IPython.core.display import display, Markdown
from ipywidgets import Button
from magicgui import magicgui

from segmfriends.io.images import read_uint8_img
from .widgets_utils import SelectImagePath
from ..core import BaseAnnotationExperiment

import napari
from napari.layers import Shapes


# from typing import TYPE_CHECKING
#
# if TYPE_CHECKING:
#   import napari


class CreateROIs():
    def __init__(self, project):
        self.project = project

        self.image_choice = None

    def display_widgets(self):
        input_images_with_nb_rois = self.project.get_list_rois_per_image()

        self.image_choice = widgets.Dropdown(
            options=[(img_data[0], i) for i, img_data in enumerate(input_images_with_nb_rois)],
            value=0,
            disabled=False,
            layout=widgets.Layout(width='90%')
        )

        # def on_dropdown_value_change(change):
        #     self.id_selected_image = change['new']
        #
        # dropdown_widget.observe(on_dropdown_value_change, names='value')

        submit_button = Button(description="Select ROIs",
                               layout=widgets.Layout(margin='10px 0px 0px 25px', width='250px'))
        submit_button.on_click(self._visualize_in_napari)

        image_choice_widgets = widgets.VBox([widgets.HBox([widgets.Label(value="Select an image in the project: "), self.image_choice]),
                                             submit_button])
        display(image_choice_widgets)

    def _visualize_in_napari(self, b):
        assert self.image_choice is not None

        image_paths = self.project.get_image_paths(image_id=self.image_choice.value)

        # create the viewer and display the image
        self.viewer = viewer = napari.Viewer()

        channel_colormaps = ["gray", "red", "yellow", "cyan"]
        for i, channel in enumerate(image_paths):
            viewer.add_image(read_uint8_img(image_paths[channel])[..., 0], name=channel, colormap=channel_colormaps[i],
                             blending='additive')

        napari_rois = self.project.get_napari_roi_by_image_id(self.image_choice.value)

        # Load images in napari:
        s_layer = viewer.add_shapes(data=napari_rois, name="ROIs image {}".format(self.image_choice.value),
                                    shape_type="rectangle", opacity=0.15, edge_color='#fff01dff',
                                    face_color='#f9ffbeff')

        @magicgui(
            call_button="Save regions of interest",
            shapes={'label': 'ROIs layer:', "visible": False},
        )
        def save_ROIs(shapes: napari.layers.Shapes):
            id_image = int(shapes.name.split(" ")[2])
            # print(id_image)
            if len(shapes.data):
                self.project.update_rois_image(id_image, shapes.data)
                self.viewer.close()

        viewer.window.add_dock_widget(save_ROIs, area='left')  # Add our gui instance to napari viewer


        # TODO: update the layer dropdown menu when the layer list changes
        # viewer.layers.events.changed.connect(gaussian_blur.reset_choices)
