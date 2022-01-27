from ipywidgets import Button
import ipywidgets as widgets
from IPython.display import display

from ..io import file_dialog, dir_dialog


class SelectImagePath():
    def __init__(self, path_string_widget, dialog_type="file"):
        self.path_string_widget = path_string_widget

        self.dialog_type = dialog_type

        # Create browse button:
        self.button = Button(description="Browse...")
        self.button.on_click(self.on_button_click)

    def on_button_click(self, b):
        if self.dialog_type == "file":
            file_path = file_dialog.gui_fname()
        elif self.dialog_type == "dir":
            file_path = dir_dialog.gui_fname()
        else:
            raise ValueError("Dialog type not recognized: ", self.dialog_type)
        self.path_string_widget.value = file_path


def display_proj_dir_browse_button():
    proj_dir_path = widgets.Text(placeholder='Path to project directory (to be copied below)', layout=widgets.Layout(width='60%'))
    select_proj_dir = SelectImagePath(proj_dir_path, dialog_type="dir")
    print("\n")
    display(widgets.HBox([proj_dir_path, select_proj_dir.button]))
    print("\n")

