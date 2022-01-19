from ipywidgets import Button
import ipywidgets as widgets
from IPython.display import clear_output, display
from IPython.display import Markdown, Latex


from ..io import file_dialog


class SelectImagePath():
    def __init__(self, path_string_widget, ):
        self.path_string_widget = path_string_widget

        # Create browse button:
        self.button = Button(description="Browse...")
        self.button.on_click(self.on_button_click)

    def on_button_click(self, b):
        file_path = file_dialog.gui_fname()
        self.path_string_widget.value = file_path


def display_new_image_widgets():
    # Create widgets to add a new image to the project:

    main_image_path = widgets.Text(placeholder='Main channel path', layout=widgets.Layout(width='35%'))
    dapi_image_path = widgets.Text(placeholder='DAPI channel path', layout=widgets.Layout(width='35%'))
    ch1_image_path = widgets.Text(placeholder='Extra channel 1 path', layout=widgets.Layout(width='35%'))
    ch2_image_path = widgets.Text(placeholder='Extra channel 2 path', layout=widgets.Layout(width='35%'))

    ch1_name = widgets.Text(placeholder='Name extra ch 1', layout=widgets.Layout(width='15%'))
    ch2_name = widgets.Text(placeholder='Name extra ch 2', layout=widgets.Layout(width='15%'))

    main_image_filter = widgets.Text(placeholder="_ch0", layout=widgets.Layout(width='15%'))
    dapi_filter = widgets.Text(placeholder='_ch1', layout=widgets.Layout(width='15%'))
    ch1_filter = widgets.Text(placeholder='_ch2', layout=widgets.Layout(width='15%'))
    ch2_filter = widgets.Text(placeholder='_ch3', layout=widgets.Layout(width='15%'))

    # Checkboxes:
    use_dapi = widgets.Checkbox(value=False, layout=widgets.Layout(width='20px'), description='', indent=False)
    use_ch1 = widgets.Checkbox(value=False, layout=widgets.Layout(width='20px'), description='', indent=False)
    use_ch2 = widgets.Checkbox(value=False, layout=widgets.Layout(width='20px'), description='', indent=False)

    select_main = SelectImagePath(main_image_path)
    select_dapi = SelectImagePath(dapi_image_path)
    select_ch1 = SelectImagePath(ch1_image_path)
    select_ch2 = SelectImagePath(ch2_image_path)

    # print("\n")
    display(Markdown("""
Use the fields below to add an additional image to the project:
- **Main channel**: select the path of the main channel that should be segmented (usually the bright-field channel)
- **DAPI channel**: If you have it, please provide it because it can be very useful for getting a better segmentation
- **Additional channels**: If you have additional channels that may helpful for the manual annotation process, you can load two extra ones. 
- **How to specify image paths:**
    - *Option 1*: Specify all channel paths manually
    - *Option 2*: If all channel images are in the same directory and have similar names, e.g. *img_ch0.tif*, *img_ch1.tif*, *img_ch2.tif*, 
you only need to provide the filename filters *_ch0*, *_ch1*, *_ch2*. The full paths of the additional image channels
will be automatically deduced from the main-channel image.
"""
                     ))
    print("\n")
    display(widgets.VBox([
        widgets.HBox([widgets.Label(value="", layout=widgets.Layout(width='20px')),
                      widgets.Label(value="Channel name:", layout=widgets.Layout(width='15%')),
                      widgets.Label(value="Filename filter:", layout=widgets.Layout(width='15%')),
                      widgets.Label(value="Image path:", layout=widgets.Layout(width='35%')),
                      widgets.Label(value="")]),
        widgets.HBox([widgets.Label(value="", layout=widgets.Layout(width='20px')),
                      widgets.Label(value="Main", layout=widgets.Layout(width='15%')),
                      main_image_filter,
                      main_image_path,
                      select_main.button]),
        widgets.HBox([use_dapi,
                      widgets.Label(value="DAPI", layout=widgets.Layout(width='15%')),
                      dapi_filter,
                      dapi_image_path,
                      select_dapi.button]),
        widgets.HBox([use_ch1,
                      ch1_name,
                      ch1_filter,
                      ch1_image_path,
                      select_ch1.button]),
        widgets.HBox([use_ch2,
                      ch2_name,
                      ch2_filter,
                      ch2_image_path,
                      select_ch2.button]),
    ]))
