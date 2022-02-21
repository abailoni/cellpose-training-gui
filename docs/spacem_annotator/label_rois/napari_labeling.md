:::{image} ../images/starting-gui-label.jpg
:alt: Button to start labeling in Napari 
:width: 45%
:align: right
:::

# Labeling in Napari


You can start labeling in Napari by clicking on the "_Label Regions of Interest_" button in the main window of the annotation tool. A new napari viewer will then open, displaying one region of interest at the time.

**Selecting a region of interest**:
To start annotating, you first select a region of interest by using the drop-down menu in the top-right corner (see image below). Initially, the first region will be displayed.


:::{image} ../images/adjust-contrast.jpg
:alt: Adjust layer contrast 
:width: 100%
:align: center
:::

::::{important}
Remember that you should label all cells inside a region of interest!
::::

::::{tip}
For improving the annotation process, it may be helpful to **adjust the contrast** of some image layers. To do so, first select an image layer  (for example the main channel) in the layers' list on the left and then adjust the "_Contrast limits_" slider on the top-left (see image above).
::::

::::{note}
Annotations are automatically saved when you switch to another region of interest or go back to the starting window. However, you can manually save your annotations at any moment by clicking on the "_Save Annotations_" button on the right.
::::





## Using Napari annotation tools
Before to start annotating, make sure that the "_Annotations_" layer is selected in layers' list on the left. 

A **complete tutorial** explaining all the annotations tools in Napari can be found [here](https://napari.org/tutorials/fundamentals/labels.html).
However, in the following you will find a recap of the most important functions and tools.




### Napari annotation toolbar
- 
::::{tip}
When you have some of the annotation tools selected, you can **zoom or move around in the image** by keeping the \<space-bar> key pressed.
::::
::::{tip}
You can **adjust the opacity** of the annotation layer to see more of the original image in the painted regions. 
::::


### Useful keyboard shortcuts
- _M_: Select a new label color (when you want to start labeling a new cell) 
- _Ctrl + Z_: undo the last change



