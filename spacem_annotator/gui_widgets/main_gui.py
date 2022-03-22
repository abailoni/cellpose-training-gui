import os
import sys

import napari
import magicgui.widgets as widgets
from magicgui.types import FileDialogMode
from magicgui.widgets import (
    Container,
    PushButton,
)

from ..napari_gui.roi_selection import RoiSelectionWidget
from ..napari_gui.roi_labeling import RoiLabeling


def show_message_pop_up(info_message: str):
    assert isinstance(info_message, str)
    info_window = widgets.Container()
    close_window_button = PushButton(name="close_window", text="Ok")
    info_window.extend([
        widgets.Label(label=info_message),
        close_window_button
    ])

    info_window.show()

    @close_window_button.changed.connect
    def close_info_window():
        info_window.clear()
        info_window.hide()


class StartingGUI(Container):
    def __init__(self, annotation_project: "spacem_annotator.BaseAnnotationExperiment"):
        super(StartingGUI, self).__init__()
        self.project = annotation_project
        # self.max_width = 300
        self.roi_select_viewer = None
        self.show_starting_gui()

    def show_starting_gui(self):
        self.clear()

        select_rois_button = PushButton(name="select_rois", text="Select Regions of Interest")
        select_rois_button.changed.connect(self.show_roi_selection_gui)

        # TODO: add attributes to init
        self.labeling_tool_combobox = widgets.ComboBox(
            label="using",
            choices=["QuPath", "Napari"],
            value=self.project.get("labeling_tool")
        )

        @self.labeling_tool_combobox.changed.connect
        def update_labeling_tool(new_value: str):
            self.project.set("labeling_tool", new_value)
            self.project.dump_configuration()

        label_rois_button = PushButton(name="label_rois", text="Manually Annotate")
        label_rois_button.changed.connect(self.show_roi_labeling_gui)
        # get_qupath_labels_button = PushButton(name="get_qupath_labels", text="Manually Annotate in QuPath")
        # get_qupath_labels_button.changed.connect(self.start_qupath)
        train_button = PushButton(name="start_training", text="Train Cellpose Model")
        train_button.changed.connect(self.show_training_gui)

        labels = widgets.Container(widgets=[widgets.Label(value="Step 1:"),
                                            widgets.Label(value="Step 2:"),
                                            widgets.Label(value="Step 3:")])
        column_1 = widgets.Container(widgets=[select_rois_button,
                                              label_rois_button,
                                              train_button])

        widgest_to_diplay = widgets.Container(widgets=[widgets.Label(value=""),
                                                       self.labeling_tool_combobox,
                                                       widgets.Label(value=""),
                                                       ])

        button_container = widgets.Container(layout="horizontal",
                                             widgets=[labels, column_1, widgest_to_diplay],
                                             # label="Select one option:"
                                             )

        # button_container = widgets.Container(layout="vertical",
        #                                      widgets=widgest_to_diplay,
        #                                      # label="Select one option:"
        #                                      )

        self.extend([
            button_container])

    def show_roi_selection_gui(self):
        # Clear and hide the current GUI interface:
        # self.clear()
        # self.hide()

        # Create a napari viewer with an additional widget:
        self.roi_select_viewer = viewer = napari.Viewer()

        roi_selection_widget = RoiSelectionWidget(self)
        viewer.window.add_dock_widget(roi_selection_widget, area='right',
                                      name="ROI selection")  # Add our gui instance to napari viewer

    def show_roi_labeling_gui(self):
        rois_list = self.project.get_roi_list()
        if len(rois_list) == 0:
            self.show_starting_gui()
            show_message_pop_up("First select some regions of interest!")
        else:
            labeling_tool = self.labeling_tool_combobox.value
            if labeling_tool == "QuPath":
                # Launch QuPath:
                python_interpreter = sys.executable
                open_qupath_command = "{} -m paquo {} open {}".format(
                    python_interpreter,
                    "--" if "ipython" in python_interpreter else "",
                    self.project.qupath_directory
                )
                print(open_qupath_command)
                os.system(open_qupath_command)
            elif labeling_tool == "Napari":
                # Create a napari viewer with an additional widget:
                # FIXME: update name of the attribute: simply napari-viewer?
                self.roi_select_viewer = viewer = napari.Viewer()

                roi_labeling_widget = RoiLabeling(self)
                viewer.window.add_dock_widget(roi_labeling_widget, area='right',
                                              name="ROI labeling")
            else:
                raise ValueError("Labeling tool not recognized")

    # ------------------------
    # Training GUI:
    # ------------------------

    def show_training_gui(self):
        rois_list = self.project.get_roi_list()
        if len(rois_list) == 0:
            self.show_starting_gui()
            show_message_pop_up("First select some regions of interest!")
        else:
            # Clear and hide the current GUI interface:
            self.clear()

            training_params = self.project.get_training_parameters_GUI()
            cellopose_training_kwargs = training_params["cellpose_kwargs"]

            # Show button to go back:
            close_button = PushButton(name="close_and_go_back", text="Go Back to Starting Window")

            @close_button.changed.connect
            def close_viewer_and_go_back():
                if self.is_valid_training_config():
                    # TODO: update main train config in main project

                    self.clear()
                    # Hide and show to reset the size of the window:
                    self.hide()
                    self.show()
                    self.show_starting_gui()

            self.model_name = widgets.LineEdit(
                name="model_name",
                label="Model name (no spaces)",
                tooltip="Pick a name for the model that will be trained (do not use spaces)",
                value=training_params["model_name"]
            )

            # Choose model:
            self.pretrained_model = widgets.ComboBox(
                label="Pretrained model",
                tooltip="Choose the pretrained model from which to start training",
                choices=["None", "cyto2", "Custom model"],
                value=training_params["pretrained_model_GUI"]
            )

            # Custom model path:
            self.custom_model_path = widgets.FileEdit(
                name="main_channel",
                tooltip="If using a custom model, select its path",
                mode=FileDialogMode.EXISTING_FILE,
                value=training_params["custom_model_path_GUI"],
                label="Custom model path"
            )

            self.n_epochs = widgets.SpinBox(
                name="n_epochs",
                label="Number of epochs",
                tooltip="Select for how many epochs the model will be trained",
                min=1,
                step=20,
                max=5000,
                value=cellopose_training_kwargs["n_epochs"],
            )

            self.batch_size = widgets.SpinBox(
                name="batch_size",
                label="Batch size",
                tooltip="Select the size of the training batch",
                min=1,
                step=1,
                max=50,
                value=cellopose_training_kwargs["batch_size"]
            )

            self.learning_rate = widgets.LineEdit(
                name="learning_rate",
                label="Learning rate",
                tooltip="Select the learning rate used for training",
                value=cellopose_training_kwargs["learning_rate"],
            )

            setup_training_data_button = PushButton(name="setup_training_data",
                                                    text="Setup training data and save training config")
            start_training_button = PushButton(name="start_training", text="Start training")

            @setup_training_data_button.changed.connect
            def setup_training_data():
                if self.is_valid_training_config():
                    self.project.setup_cellpose_training_data(self.model_name.value)

            @start_training_button.changed.connect
            def start_training():
                self.is_valid_training_config()
                pass

            self.extend([
                self.model_name,
                self.pretrained_model,
                self.custom_model_path,
                self.n_epochs,
                self.batch_size,
                self.learning_rate,
                setup_training_data_button,
                start_training_button,
                close_button,

            ])

    def is_valid_training_config(self):
        if " " in self.model_name.value:
            show_message_pop_up("The model name should not contain spaces")
            return False
        try:
            float(self.learning_rate.value)
        except ValueError:
            show_message_pop_up("The learning rate should be a float number")
            return False
        self.update_main_training_config()
        return True

    def update_main_training_config(self):
        training_kwargs = {
            "pretrained_model": self.pretrained_model.value,
            "custom_model_path": str(self.custom_model_path.value),
            "n_epochs": self.n_epochs.value,
            "batch_size": self.batch_size.value,
            "learning_rate": eval(self.learning_rate.value)
        }
        was_updated, error_message = self.project.update_main_training_config(self.model_name.value, **training_kwargs)
        if not was_updated:
            show_message_pop_up(error_message)

    # self.model_name.value
