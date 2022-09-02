from magicgui import widgets as widgets
from magicgui.widgets import PushButton
from napari.utils.notifications import show_error

def show_message_pop_up(info_message: str):
    show_error(info_message)
    # assert isinstance(info_message, str)
    # info_window = widgets.Container()
    # close_window_button = PushButton(name="close_window", text="Ok")
    # info_window.extend([
    #     widgets.Label(label=info_message),
    #     close_window_button
    # ])
    #
    # info_window.show(run=True)
    #
    # @close_window_button.changed.connect
    # def close_info_window():
    #     info_window.clear()
    #     info_window.hide()

