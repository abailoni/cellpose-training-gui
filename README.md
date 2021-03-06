# Tool for Image Annotation and Training Cellpose
Graphical tool for quickly creating image segmentation annotations and training custom [Cellpose](https://www.cellpose.org) models.

Main features:
1. Select regions of interests using Napari (from one or multiple images)
2. Create cell annotations using QuPath or Napari 
3. Train a new Cellpose model (with the option to train your model via the browser using a remote server with GPU and CUDA support) or export the training data and train a new model using the `cellpose>=2.0` interface.

To install it, see the [documentation](https://abailoni.github.io/cellpose-training-gui). 

![](docs/cellpose-training-gui/images/starting-gui-v0.2.jpg)

## Features
### Select regions of interests in Napari
Load one or more images (with one or multiple channels) and select the areas that you want to annotate and use for training.

![](docs/cellpose-training-gui/images/selecting_rois.jpg)

### Create annotations in QuPath or Napari
Get a QuPath project that you can directly use to annotate the regions you selected, or annotate them directly using Napari. Otherwise, you can also export the regions you selected in the format expected by the cellpose software and label them in cellpose 

![](docs/cellpose-training-gui/images/qupath/qupath-starting-interface.jpg)

### Train custom cellpose models
Train a new model directly from the tool, or export the training data and train a new model using the `cellpose>=2.0` interface.

![](docs/cellpose-training-gui/images/new_training_GUI.jpg)

#### Train custom model on a remote server
If your machine does not have a GPU and CUDA support, you can install the `TrainCellposeServer` tool on a remote server with a GPU and train your models remotely via the browser (see [Cellpose training server tool](https://github.com/abailoni/cellpose-training-server)). 

![](docs/cellpose-training-gui/images/training-remotely.gif)
