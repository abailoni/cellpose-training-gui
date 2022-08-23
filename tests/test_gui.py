import logging
import os
import pathlib

import pytest
import shutil
from pathlib import Path

from napari_console._tests.test_qt_console import make_test_viewer

# TODO: Make sure that `module` makes sense...
@pytest.fixture()
def roi_napari_viewer(make_napari_viewer):
    # make_napari_viewer takes any keyword arguments that napari.Viewer() takes
    viewer = make_napari_viewer()
    yield viewer

@pytest.fixture()
def annotation_project(tmpdir_factory):
    from traincellpose.core import BaseAnnotationExperiment
    proj_dir = str(tmpdir_factory.mktemp('annotation_proj'))
    # TODO: copy some test images in the proj_dir?
    experiment = BaseAnnotationExperiment(proj_dir)
    yield experiment

@pytest.fixture()
def starting_gui_widget(annotation_project, roi_napari_viewer):
    from traincellpose.gui_widgets.main_gui import StartingGUI
    starting_gui_widget = StartingGUI(annotation_project)
    starting_gui_widget.roi_select_viewer = roi_napari_viewer
    yield starting_gui_widget

@pytest.fixture()
def roi_widget(starting_gui_widget):
    from traincellpose.napari_gui.roi_selection import RoiSelectionWidget
    roi_selection_widget = RoiSelectionWidget(starting_gui_widget)
    yield roi_selection_widget

def test_roi_selection(roi_widget):
    # logger = logging.getLogger("traincellpose")
    # logger.setLevel("DEBUG")
    logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(name)s] [%(levelname)s] %(message)s',
                        datefmt='%H:%M:%S',
                        level="DEBUG")
    logging.basicConfig()
    roi_widget.logger.setLevel("DEBUG")


    roi_viewer = roi_widget.main_gui.roi_select_viewer
    ann_project = roi_widget.main_gui.project

    assert roi_widget.edit_image_paths_mode_active
    assert roi_widget.image_id is None

    # assert GUI elements
    assert roi_widget.selected_image is not None
    assert roi_widget.close_button is not None
    assert roi_widget.add_new_image_button is None
    assert roi_widget.update_image_paths_button is None
    assert roi_widget.save_image_paths_button is not None

    # Add a single tif image as main channel:
    main_path_widg, main_inner_path_widg, _ = roi_widget.paths_widgets[0]
    # assert main_path_widg.visible
    main_path_widg.value = "/Users/alberto-mac/EMBL_ATeam/cellpose_training_pipeline/test_images/img1/fused_tp_0_ch_3.tif"
    assert len(roi_viewer.layers) == 1
    assert roi_viewer.layers[0].name == roi_widget.channel_names[0]
    assert not main_inner_path_widg.visible

    # Add DAPI:
    dapi_path_widg, dapi_inner_path_widg, _ = roi_widget.paths_widgets[1]
    dapi_path_widg.value = "/Users/alberto-mac/EMBL_ATeam/cellpose_training_pipeline/test_images/img1/fused_tp_0_ch_2.tif"
    assert len(roi_viewer.layers) == 2
    assert roi_viewer.layers[1].name == roi_widget.channel_names[1]
    assert not dapi_inner_path_widg.visible
    # TODO: assert color channel?

    # Add image to project:
    roi_widget.save_image_paths_button.clicked(True)
    assert ann_project.nb_input_images == 1
    assert len(ann_project.input_images[0]) == 2

    # Now assert that the interface has changed:
    assert roi_widget.selected_image is not None
    assert roi_widget.close_button is not None
    assert roi_widget.add_new_image_button is not None
    assert roi_widget.update_image_paths_button is not None
    assert roi_widget.save_image_paths_button is None

    # Go back to input-mode:
    roi_widget.add_new_image_button.clicked(True)
    select_choices = roi_widget.selected_image.choices
    # Make sure we are in "add new image" mode:
    assert roi_widget.selected_image.value == select_choices[-1]
    # Go back to first image using the image-select tool:
    roi_widget.selected_image.value = select_choices[0]

    # Go back to adding new image:
    roi_widget.selected_image.value = select_choices[1]
    # Now also add another zarr image:
    main_path_widg, main_inner_path_widg, _ = roi_widget.paths_widgets[0]
    main_path_widg.value = pathlib.PosixPath("/Users/alberto-mac/EMBL_ATeam/cellpose_training_pipeline/test_images/sample_zarr_data/pre_maldi")
    # main_path_widg.value = "/Users/alberto-mac/EMBL_ATeam/cellpose_training_pipeline/test_images/img1/fused_tp_0_ch_3.tif"
    # Check if DAPI channel was also loaded automatically:
    assert len(roi_viewer.layers) == 2
    # assert main_inner_path_widg.visible
    # assert roi_widget.paths_widgets[0][1].visible
    assert roi_viewer.layers[0].name == roi_widget.channel_names[0]
    assert roi_viewer.layers[1].name == roi_widget.channel_names[1]

    # Try to set an invalid path:
    main_path_widg.value = pathlib.PosixPath(
        "/Users/alberto-mac/EMBL_ATeam/cellpose_training_pipeline/test_images/sample_zarr_data/pre_")
    assert len(roi_viewer.layers) == 1
    # TODO: Draw a sample ROI:
    #  test a png image with multiple channels
    pass
    print("Test")

    # roi_widget.main_gui.project.add_input_image(roi_widget.cur_img_info,
    #                                       id_input_image_to_rewrite=roi_widget.image_id)


