from annotationtools.base_experiment import BaseAnnotationExperiment


project_directory = "/Users/alberto-mac/EMBL_ATeam/cellpose_training_pipeline/test_project"

annotation_exp = BaseAnnotationExperiment(project_directory)








# !!! Do not forget to re-run this cell and the following one every time you change any path !!!

# --------------------  TO BE FILLED BY USER - START  -----------------------------

# Main image to be segmented:
main_image_path = "/Users/alberto-mac/EMBL_ATeam/cellpose_training_pipeline/test_images/img1/fused_tp_0_ch_4.tif"

use_dapi_channel_for_segmentation = True

# For additional (optional) channels, you can either specify the image paths...
dapi_image_path = ""
extra_ch_1_image_path = ""
extra_ch_2_image_path = ""

# ...or you can define some name filters, so that additional channels are automatically found in the same
# directory of the main image:
filter_main_image = "_ch_4"
filter_dapi_image = "_ch_2"
filter_extra_ch_1 = "_ch_1"
filter_extra_ch_2 = ""

# If you want, you can also define the extra channel names, for clarity:
extra_ch_1_name = "GFP"
extra_ch_2_name = "Extra channel 2"

# ---------------------  TO BE FILLED BY USER - END  ------------------------------

annotation_exp.use_dapi_channel_for_segmentation = use_dapi_channel_for_segmentation
annotation_exp.set_extra_channels_names([extra_ch_1_name, extra_ch_2_name])
id_new_image = annotation_exp.add_input_image(main_image_path,
                               filter_main_image,
                               dapi_image_path,
                               filter_dapi_image,
                               extra_ch_1_image_path,
                               filter_extra_ch_1,
                               extra_ch_2_image_path,
                               filter_extra_ch_2)

napari_rois = annotation_exp.get_napari_roi_by_image_id(id_new_image)

annotation_exp.get_image_paths(0)

input_images_with_nb_rois = annotation_exp.get_list_rois_per_image()

import numpy as np
fake_rois = np.random.uniform(size=(4,4,2))
annotation_exp.update_rois_image(id_new_image, fake_rois)

# Now update them:
new_fake_rois = np.concatenate([fake_rois[1:], np.random.uniform(size=(1,4,2))], axis=0)
annotation_exp.update_rois_image(id_new_image, new_fake_rois)
print(input_images_with_nb_rois)





