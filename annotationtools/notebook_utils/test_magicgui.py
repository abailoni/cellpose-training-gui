# import datetime
# from enum import Enum
# from pathlib import Path
#
# from magicgui import magicgui
#
#
# class Medium(Enum):
#     """Using Enums is a great way to make a dropdown menu."""
#     Glass = 1.520
#     Oil = 1.515
#     Water = 1.333
#     Air = 1.0003
#
# @magicgui(
#     call_button="Calculate",
#     layout="vertical",
#     result_widget=True,
#     # numbers default to spinbox widgets, but we can make
#     # them sliders using the `widget_type` option
#     slider_float={"widget_type": "FloatSlider", "max": 100},
#     slider_int={"widget_type": "Slider", "readout": False},
#     radio_option={
#         "widget_type": "RadioButtons",
#         "orientation": "horizontal",
#         "choices": [("first option", 1), ("second option", 2)],
#     },
#     filename={"label": "Pick a file:"},  # custom label
#     label_test={}
# )
# def widget_demo(
#         label_test: ,
#     boolean=True,
#     integer=1,
#     spin_float=3.14159,
#     slider_float=43.5,
#     slider_int=550,
#     string="Text goes here",
#     dropdown=Medium.Glass,
#     radio_option=2,
#     date=datetime.date(1999, 12, 31),
#     time=datetime.time(1, 30, 20),
#     datetime=datetime.datetime.now(),
#     filename=Path.home(),  # path objects are provided a file picker
# ):
#     """Run some computation."""
#     return locals().values()
#
# widget_demo.show(run=True) # if running locally, use `show(run=True)`
# widget_demo.close()
#


from magicgui import widgets
import datetime

from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FloatSlider,
    PushButton,
    SpinBox,
    show_file_dialog,
    MainWindow
)

class ToyProjectClass:
    def get_list_rois_per_image(self):
        return [(i,i) for i in range(10)]


class AddImage(Container):
    def __init__(self, annotation_project):
        super(AddImage, self).__init__()
        self.project = annotation_project
        self.setup_gui()

    def setup_gui(self):
        self._choose_image = ComboBox(
            name="choose_image", label="Image"
        )
        self._choose_image.choices = ["Test {}".format(img[0]) for img in self.project.get_list_rois_per_image()]
        clear_button = PushButton(name="clear_button", text="Clear data")
        clear_button.changed.connect(self.test_reset)

        self.extend([
            widgets.Label(value="""Test tst *test*
sdfsd
sdfsdf            
"""),
            widgets.LineEdit(value="line edit value", label="LineEdit:"),
            widgets.TextEdit(value="text edit value...", label="TextEdit:"),
            widgets.FileEdit(value="/home", label="FileEdit:"),
            widgets.RangeEdit(value=range(0, 10, 2), label="RangeEdit:"),
            widgets.SliceEdit(value=slice(0, 10, 2), label="SliceEdit:"),
            widgets.DateTimeEdit(
              value=datetime.datetime(1999, 12, 31, 11, 30), label="DateTimeEdit:"
            ),
            widgets.DateEdit(value=datetime.date(81, 2, 18), label="DateEdit:"),
            widgets.TimeEdit(value=datetime.time(12, 20), label="TimeEdit:"),
            self._choose_image,
            clear_button
                     ])

    def setup_gui_2(self):
        self._choose_image = ComboBox(
            name="choose_image", label="Image", choices=[img[0] for img in self.project.get_list_rois_per_image()]
        )
        clear_button = PushButton(name="clear_button", text="Clear data")
        clear_button.changed.connect(self.test_reset)

        self.extend([
            widgets.Label(value="test label test label test label test label test label test label test label test label test label test label", ),
            widgets.Container(layout="horizontal", widgets=[widgets.CheckBox(), widgets.FileEdit()]),
            widgets.LineEdit(value="line edit value", label="LineEdit:"),
            widgets.TextEdit(value="text edit value...", label="TextEdit:"),
            widgets.FileEdit(value="/home", label="FileEdit:"),
            widgets.RangeEdit(value=range(0, 10, 2), label="RangeEdit:"),
            widgets.SliceEdit(value=slice(0, 10, 2), label="SliceEdit:"),
            widgets.DateTimeEdit(
                value=datetime.datetime(1999, 12, 31, 11, 30), label="DateTimeEdit:"
            ),
            widgets.DateEdit(value=datetime.date(81, 2, 18), label="DateEdit3wssada:"),
            widgets.TimeEdit(value=datetime.time(12, 20), label="TimeEdit:"),
            self._choose_image,
            clear_button
        ])

    def test_reset(self):
        self.clear()
        self.setup_gui_2()

proj_instance = ToyProjectClass()


wdg_list = [
    # widgets.Label(value="Test tst \n *test*"),
    # widgets.LineEdit(value="line edit value", label="LineEdit:"),
    # widgets.TextEdit(value="text edit value...", label="TextEdit:"),
    # widgets.FileEdit(value="/home", label="FileEdit:"),
    # widgets.RangeEdit(value=range(0, 10, 2), label="RangeEdit:"),
    # widgets.SliceEdit(value=slice(0, 10, 2), label="SliceEdit:"),
    # widgets.DateTimeEdit(
    #   value=datetime.datetime(1999, 12, 31, 11, 30), label="DateTimeEdit:"
    # ),
    # widgets.DateEdit(value=datetime.date(81, 2, 18), label="DateEdit:"),
    # widgets.TimeEdit(value=datetime.time(12, 20), label="TimeEdit:"),
    AddImage(proj_instance)
]
# container = widgets.Container(widgets=wdg_list)
container = AddImage(proj_instance)
container.max_height = 350
container.show(run=True)
