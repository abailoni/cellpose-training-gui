# Custom cellpose training parameters and CLI commands 
On top of the cellpose training parameters that can be modified via graphical interface, it is also possible to set all the training parameters supported by cellpose by modifying the training config file:
1. First, generate a new training configuration via graphical interface or by executing the following command:
    ```shell 
    python -m traincellpose setup_training --model_name my_model
    ```
2. After doing that, all the training data can be found in the `PROJ_DIR/CellposeTraining/my_model` directory, including:
    - A config file `train_config.yml` with cellpose parameters that will be used for training (can be updated with custom parameters if needed)
    - Training images placed in the `training_images` subdirectory: if needed you can add external training images. In that case, keep in mind that green channel should represent the main segmentation channel, and red channel should represent the DAPI channel (if present, otherwise you can leave it empty).
3. After you are done making your modifications, you can start the training via graphical interface or by executing the following command:
    ```shell 
    python -m traincellpose train --model_name my_model
    ```


