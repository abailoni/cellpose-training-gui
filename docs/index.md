# {{proj_name}}

Graphical tool for quickly creating image segmentation annotations and training custom [Cellpose](https://www.cellpose.org) models.

Main features:
1. Select regions of interests using Napari (from one or multiple images)
2. Create cell annotations using QuPath or Napari 
3. Train a new Cellpose model (with the option to train your model via the browser using a remote server with GPU and CUDA support) or export the training data and train a new model using the `cellpose>=2.0` interface.


```{toctree}
---
maxdepth: 2
caption: |
    Documentation:
---
cellpose-training-gui/install
cellpose-training-gui/select_rois
cellpose-training-gui/label_rois/label_rois
cellpose-training-gui/cellpose_training/training
```
