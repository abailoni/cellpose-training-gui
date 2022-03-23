# Custom cellpose training and CLI commands 
On top of the cellpose training parameters that can be modified via graphical interface, it is also possible to set all the training parameters supported by cellpose by modifying the training config file:
- First, generate a new training configuration via graphical interface or by executing the following command:
  - `python -m cellpose-training-gui setup_training --model_name my_model`
- After doing that, all the training data can be found in the `PROJ_DIR/CellposeTraining/my_model` directory:
  - In this directory you find a config file `train_config.yml` containing the cellpose parameters that will be used for training, so if needed it can be updated with custom parameters
  - Training images are placed in the `PROJ_DIR/CellposeTraining/my_model/training_images` directory, so if needed you can also add external training images to the folder. In that case, keep in mind that green channel should represent the main segmentation channel, and red channel should represent the DAPI channel (if present, otherwise you can leave it empty).
- After you are done making your modifications, you can start the training via graphical interface or by executing the following command:
  - `python -m cellpose-training-gui train --model_name my_model`

