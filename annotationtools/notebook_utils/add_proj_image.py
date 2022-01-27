import ipywidgets as widgets
from IPython.core.display import display, Markdown
from ipywidgets import Button

from .widgets_utils import SelectImagePath
from ..core import BaseAnnotationExperiment


class AddNewImageToProject():
    def __init__(self, project):
        self.project = project
        self.id_selected_image = None

        # Initialize widgets that will be used later:
        # TODO: create variables for width values
        self.main_image_path = widgets.Text(placeholder='Main channel path', layout=widgets.Layout(width='35%'))
        self.dapi_image_path = widgets.Text(placeholder='DAPI channel path', layout=widgets.Layout(width='35%'))
        self.ch1_image_path = widgets.Text(placeholder='Extra channel 1 path', layout=widgets.Layout(width='35%'))
        self.ch2_image_path = widgets.Text(placeholder='Extra channel 2 path', layout=widgets.Layout(width='35%'))

        self.ch1_name = widgets.Text(placeholder='Name extra ch 1', layout=widgets.Layout(width='15%'))
        self.ch2_name = widgets.Text(placeholder='Name extra ch 2', layout=widgets.Layout(width='15%'))

        self.main_image_filter = widgets.Text(placeholder="_ch0", layout=widgets.Layout(width='15%'))
        self.dapi_filter = widgets.Text(placeholder='_ch1', layout=widgets.Layout(width='15%'))
        self.ch1_filter = widgets.Text(placeholder='_ch2', layout=widgets.Layout(width='15%'))
        self.ch2_filter = widgets.Text(placeholder='_ch3', layout=widgets.Layout(width='15%'))

        # Checkboxes:
        self.use_dapi = widgets.Checkbox(value=False, layout=widgets.Layout(width='20px'), description='', indent=False)
        self.use_ch1 = widgets.Checkbox(value=False, layout=widgets.Layout(width='20px'), description='', indent=False)
        self.use_ch2 = widgets.Checkbox(value=False, layout=widgets.Layout(width='20px'), description='', indent=False)

    def _add_new_image(self, b):
        use_dapi = self.use_dapi.value
        use_ch1 = self.use_ch1.value
        use_ch2 = self.use_ch2.value

        self.project.use_dapi_channel_for_segmentation = self.use_dapi.value
        self.project.set_extra_channels_names([self.ch1_name.value, self.ch2_name.value])
        self.id_selected_image = self.project.add_input_image(self.main_image_path.value,
                                                              self.main_image_filter.value,
                                                              self.dapi_image_path.value if use_dapi else None,
                                                              self.dapi_filter.value if use_dapi else None,
                                                              self.ch1_image_path.value if use_ch1 else None,
                                                              self.ch1_filter.value if use_ch1 else None,
                                                              self.ch2_image_path.value if use_ch2 else None,
                                                              self.ch2_filter.value if use_ch2 else None)

    def display_notebook_widgets(self):
        # Create widgets to add a new image to the project:
        select_main = SelectImagePath(self.main_image_path)
        select_dapi = SelectImagePath(self.dapi_image_path)
        select_ch1 = SelectImagePath(self.ch1_image_path)
        select_ch2 = SelectImagePath(self.ch2_image_path)

        submit_button = Button(description="Add image to project", layout=widgets.Layout(margin='10px 0px 0px 25px', width='250px'))
        submit_button.on_click(self._add_new_image)

        print("\n")
        display(Markdown("""
Use the fields below to add an additional image to the project:
- **Main channel**: select the path of the main channel that should be segmented (usually the bright-field channel)
- **DAPI channel**: If you have it, please provide it because it can be very useful for getting a better segmentation
- **Additional channels**: If you have additional channels that may helpful for the manual annotation process, you can load two extra ones. 
- **How to specify image paths:**
    - *Option 1*: Specify all channel paths manually
    - *Option 2*: If all channel images are in the same directory and have similar names, e.g. *img_ch0.tif*, *img_ch1.tif*, *img_ch2.tif*, 
you only need to provide the filename filters (*_ch0*, *_ch1*, *_ch2*, ...) and the paths of the additional channels
will be automatically deduced from the main-channel image.
- When you have specified all the paths, press on the submit button *"Add image to project"*. 
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
                          self.main_image_filter,
                          self.main_image_path,
                          select_main.button]),
            widgets.HBox([self.use_dapi,
                          widgets.Label(value="DAPI", layout=widgets.Layout(width='15%')),
                          self.dapi_filter,
                          self.dapi_image_path,
                          select_dapi.button]),
            widgets.HBox([self.use_ch1,
                          self.ch1_name,
                          self.ch1_filter,
                          self.ch1_image_path,
                          select_ch1.button]),
            widgets.HBox([self.use_ch2,
                          self.ch2_name,
                          self.ch2_filter,
                          self.ch2_image_path,
                          select_ch2.button]),
            submit_button
        ]))
