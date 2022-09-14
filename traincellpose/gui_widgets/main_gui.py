import subprocess
import sys

import napari
import magicgui.widgets as widgets
from magicgui.types import FileDialogMode
from magicgui.widgets import (
    Container,
    PushButton,
)

from ..napari_gui.gui_utils import show_message_pop_up_in_separate_window
from ..napari_gui.roi_selection import RoiSelectionWidget
from ..napari_gui.roi_labeling import RoiLabeling
from ..qupath.update_qupath_proj import update_paths_images_in_project




class StartingGUI(Container):
    def __init__(self, annotation_project: "traincellpose.BaseAnnotationExperiment"):
        super(StartingGUI, self).__init__()
        self.project = annotation_project
        # self.max_width = 300
        self.roi_select_viewer = None
        self.labeling_tool_combobox = None
        self.show_starting_gui()

    def show_starting_gui(self):
        self.clear()

        select_rois_button = PushButton(name="select_rois", text="Select Regions of Interest (ROIs) in Napari")
        select_rois_button.changed.connect(self.show_roi_selection_gui)

        self.labeling_tool_combobox = widgets.ComboBox(
            label="using:",
            choices=["QuPath", "Napari"],
            value=self.project.get("labeling_tool")
        )

        @self.labeling_tool_combobox.changed.connect
        def update_labeling_tool(new_value: str):
            self.project.set("labeling_tool", new_value)
            self.project.dump_configuration()

        label_rois_button = PushButton(name="label_rois", text="Annotate ROIs")
        label_rois_button.changed.connect(self.show_roi_labeling_gui)

        qupath_zip_button = PushButton(name="qupath_zip", text="Export QuPath Project (.zip)")
        see_cellpose_input_button = PushButton(name="see_cellpose_input", text="Show Cellpose Input Images")
        # get_qupath_labels_button = PushButton(name="get_qupath_labels", text="Manually Annotate in QuPath")
        # get_qupath_labels_button.changed.connect(self.start_qupath)
        train_button = PushButton(name="start_training", text="Configure Training of Custom Cellpose Model")
        train_button.changed.connect(self.show_training_gui)

        @see_cellpose_input_button.changed.connect
        def see_cellpose_input():
            rois_list = self.project.get_roi_list()
            if len(rois_list) == 0:
                show_message_pop_up_in_separate_window("No region of interest has been selected yet")
            else:
                self.project.show_cellpose_input_folder()

        @qupath_zip_button.changed.connect
        def qupath_zip():
            rois_list = self.project.get_roi_list()
            if len(rois_list) == 0:
                show_message_pop_up_in_separate_window("No region of interest has been selected yet")
            else:
                self.project.compress_qupath_proj_dir()

        row_1 = widgets.Container(
            label="Step 1:",
            layout="horizontal",
            widgets=[select_rois_button]
        )
        row_1b = widgets.Container(
            label="",
            layout="horizontal",
            widgets=[qupath_zip_button, see_cellpose_input_button]
        )
        row_2 = widgets.Container(
            label="Step 2:",
            layout="horizontal",
            widgets=[label_rois_button, self.labeling_tool_combobox]
        )
        row_3 = widgets.Container(
            label="Step 3:",
            layout="horizontal",
            widgets=[train_button]
        )

        button_container = widgets.Container(widgets=[row_1, row_1b, row_2, row_3])

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
            show_message_pop_up_in_separate_window("First select some regions of interest!")
        else:
            labeling_tool = self.labeling_tool_combobox.value
            if labeling_tool == "QuPath":
                # Check if ROIs paths in QuPath should be updated:
                update_paths_images_in_project(self.project.qupath_directory,
                                               self.project.experiment_directory,
                                               ("QuPathProject", "input_images"))

                # Start QuPath:
                python_interpreter = sys.executable
                open_qupath_command = '"{}" -m paquo {} open "{}"'.format(
                    python_interpreter,
                    "--" if "ipython" in python_interpreter else "",
                    self.project.qupath_directory
                )
                print(open_qupath_command)
                try:
                    subprocess.run(open_qupath_command, shell=True, check=True)
                    # result = subprocess.run(open_qupath_command, shell=True, check=True, capture_output=True, text=True)
                    # print(result.stdout)
                    # print(result.stderr)
                except subprocess.CalledProcessError:
                    self.show_starting_gui()
                    show_message_pop_up_in_separate_window("QuPath could not be started. Make sure that it is installed.")
                    pass
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
            show_message_pop_up_in_separate_window("First select some regions of interest!")
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
                tooltip="Choose the pretrained model from which to start training (see cellpose docs for details)",
                choices=["None",
                         "cyto2",
                         "Custom model",
                         "livecell",
                         "tissuenet",
                         "cyto",
                         "nuclei",
                         "CP",
                         "CPx",
                         "TN1",
                         "TN2",
                         "TN3",
                         "LC1",
                         "LC2",
                         "LC3",
                         "LC4",
                         "LC4",
                         ],
                value=training_params["pretrained_model_GUI"]
            )

            # Use DAPI for training:
            self.use_dapi_for_training = widgets.CheckBox(
                name="use_dapi",
                tooltip="Use DAPI channel during training, otherwise it will be ignored",
                text="Use DAPI channel during training",
                value=self.project.use_dapi_channel_for_segmentation,
            )

            @self.use_dapi_for_training.changed.connect
            def use_dapi_for_training():
                print(self.use_dapi_for_training.value)
                self.project.use_dapi_channel_for_segmentation = self.use_dapi_for_training.value

                # Recompute cellpose input images:
                self.project.refresh_all_training_images(
                    update_single_channels=False,
                    update_composite_images=False,
                    update_cellpose_inputs=True)

            # Use DAPI for training:
            self.apply_preprocessing = widgets.CheckBox(
                name="apply_preprocessing",
                tooltip="Auto-adjust image saturation or apply custom preprocessing from config file",
                text="Preprocess images before training",
                value=self.project.apply_preprocessing,
            )

            @self.apply_preprocessing.changed.connect
            def apply_preprocessing():
                self.project.apply_preprocessing = self.apply_preprocessing.value

                # Recompute cellpose input images:
                self.project.refresh_all_training_images(
                    update_single_channels=False,
                    update_composite_images=False,
                    update_cellpose_inputs=True)

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
                                                    text="Prepare and Export Cellpose Training Data (.zip)")
            start_training_button = PushButton(name="start_training", text="Start Training (requires local GPU)")

            @setup_training_data_button.changed.connect
            def setup_training_data():
                if self.is_valid_training_config():
                    out_message = self.project.setup_cellpose_training_data(self.model_name.value, show_training_folder=True)
                    if out_message is not None:
                        show_message_pop_up_in_separate_window(out_message)

            @start_training_button.changed.connect
            def start_training():
                if self.is_valid_training_config():
                    training_was_successful, error_message = self.project.run_cellpose_training(self.model_name.value)
                    message = "Model training completed!" if training_was_successful else error_message
                    show_message_pop_up_in_separate_window(message)

            self.extend([
                self.model_name,
                self.pretrained_model,
                self.custom_model_path,
                self.n_epochs,
                # widgets.Container(layout="horizontal", widgets=[self.n_epochs, self.use_dapi_for_training],
                #                   label="Number of epochs"),
                self.batch_size,
                self.learning_rate,
                # widgets.Container(widgets=[self.use_dapi_for_training], label=""),
                # widgets.Container(widgets=[self.use_dapi_for_training]),
                self.use_dapi_for_training,
                # self.apply_preprocessing,
                widgets.Label(label=""),
                setup_training_data_button,
                start_training_button,
                close_button,
            ])

    def is_valid_training_config(self):
        if " " in self.model_name.value:
            show_message_pop_up_in_separate_window("The model name should not contain spaces")
            return False
        try:
            float(self.learning_rate.value)
        except ValueError:
            show_message_pop_up_in_separate_window("The learning rate should be a float number")
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
            show_message_pop_up_in_separate_window(error_message)

    # self.model_name.value
