# Train cellpose
At the moment it is not possible to directly train a new cellpose model from the graphical interface.

For the moment, you can either send your annotations to me (Alberto) or, if you are familiar with the cellpose training script, you can run it yourself.

## Manually training a new cellpose model

#### Input data
- You can find all the necessary images to start the training in the `project_dir/Cellpose` folder. Input images and training labels are already in the format expected by the cellpose script.
- In the input training images, the green channel represents the main segmentation channel and the red channel is the DAPI channel (if present, otherwise it will be black).

#### Tips for training
Coming soon
