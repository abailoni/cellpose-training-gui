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
from ..napari_gui.roi_labeling import RoiLabeling


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

        button_container = widgets.Container(layout="vertical",
                                             widgets=[select_rois_button,
                                                      label_rois_button,
                                                      train_button],
                                             # label="Select one option:"
                                             )

        self.extend([
            button_container])

    def show_roi_selection_gui(self):
        # Clear and hide the current GUI interface:
        self.clear()
        self.hide()

        # Create a napari viewer with an additional widget:
        self.roi_select_viewer = viewer = napari.Viewer()

        roi_selection_widget = RoiSelectionWidget(self)
        viewer.window.add_dock_widget(roi_selection_widget, area='right',
                                      name="ROI selection")  # Add our gui instance to napari viewer



    def show_roi_labeling_gui(self):
        # FIXME: check if ROIS are more than zero!!
        rois_list = self.project.get_roi_list()
        if len(rois_list) == 0:
            self.show_starting_gui()

        # Clear and hide the current GUI interface:
        self.clear()
        self.hide()

        # Create a napari viewer with an additional widget:
        self.roi_select_viewer = viewer = napari.Viewer()

        roi_labeling_widget = RoiLabeling(self)
        viewer.window.add_dock_widget(roi_labeling_widget, area='right',
                                      name="ROI labeling")

    def show_training_gui(self):
        pass
