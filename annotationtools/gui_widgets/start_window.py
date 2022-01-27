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

from ..napari_gui.roi_selection import RoiSelectionWidget


class StartWindow(Container):
    def __init__(self, annotation_project: "annotationtools.BaseAnnotationExperiment"):
        super(StartWindow, self).__init__()
        self.project = annotation_project
        # self.max_width = 200
        self.roi_select_viewer = None
        self.show_starting_gui()

    def show_starting_gui(self):
        self.clear()

        select_rois_button = PushButton(name="select_rois", text="Select regions of interest")
        select_rois_button.changed.connect(self.show_roi_selection_gui)
        label_rois_button = PushButton(name="label_rois", text="Label regions of interest")
        label_rois_button.changed.connect(self.show_roi_labeling_gui)
        train_button = PushButton(name="start_training", text="Start training")
        train_button.changed.connect(self.show_training_gui)

        button_container = widgets.Container(layout="horizontal",
                                             widgets=[select_rois_button,
                                                      label_rois_button,
                                                      train_button],
                                             label="Select one option:"
                                             )

        self.extend([
            button_container])

    def show_roi_selection_gui(self):
        self.clear()

        buttons_to_show = []

        add_new_image_button = PushButton(name="add_new_image", text="Add new image to project")
        buttons_to_show.append(add_new_image_button)

        @add_new_image_button.changed.connect
        def add_new_image():
            self.setup_roi_selected_gui_napari()

        images_in_project = self.project.get_list_rois_per_image()
        if len(images_in_project) > 0:
            image_to_be_updated = widgets.ComboBox(
                # name="choose_image",
                label="Images in project:",
                choices=[image_info[0] for image_info in images_in_project
                         ],
                # description=None
            )

            update_rois_button = PushButton(name="update_rois_in_image",
                                            text="Update regions of interest for selected image")

            @update_rois_button.changed.connect
            def update_rois_for_image():
                # Find the ID of the selected image:
                image_path = image_to_be_updated.value
                image_id = self.project.get_input_image_id_from_path(image_path)
                self.setup_roi_selected_gui_napari(image_id)

            buttons_to_show.append(update_rois_button)
            self.append(image_to_be_updated)


        go_back_button = PushButton(name="go_back", text="Go back")
        go_back_button.changed.connect(self.show_starting_gui)
        buttons_to_show.append(go_back_button)
        button_container = widgets.Container(layout="horizontal",
                                             widgets=buttons_to_show,
                                             label="Select one option:"
                                             )

        self.append(button_container)

    def show_roi_labeling_gui(self):
        pass

    def show_training_gui(self):
        pass

    def setup_roi_selected_gui_napari(self, image_id=None):
        # Clear and hide the current GUI interface:
        self.clear()
        self.hide()

        # Create a napari viewer with an additional widget:
        self.roi_select_viewer = viewer = napari.Viewer()

        roi_selection_widget = RoiSelectionWidget(self, image_id)
        viewer.window.add_dock_widget(roi_selection_widget, area='right',
                                      name="Image paths")  # Add our gui instance to napari viewer
