:::{image} ../images/starting-gui-label.jpg
:alt: Button to start labeling in Napari 
:width: 45%
:align: right
:::

# Annotating in Napari


You can start labeling in Napari by clicking on the `Label Regions of Interest` button in the main window of the annotation tool. A new napari viewer will then open, displaying one region of interest at the time.

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
For making the annotation process easier, it may be helpful to **adjust the contrast** of some image channel. To do so, first select an image channel  (for example the main channel) in the layers list on the left and then adjust the `Contrast limits` slider on the top-left (see image above).
::::

::::{note}
Annotations are automatically saved when you switch to another region of interest or go back to the starting window. However, you can manually save your annotations at any moment by clicking on the `Save Annotations` button on the right.
::::





## Using Napari annotation tools
A **complete tutorial** explaining all the annotations tools in Napari can be found [here](https://napari.org/tutorials/fundamentals/labels.html).
However, in the following you will find a recap of the most important functions and tools.

### Napari annotation toolbar

:::{image} ../images/annotations-tools.jpg
:alt: Annotation toolbar
:width: 30%
:align: right
:::

Before to start annotating, make sure that the `Annotations` layer is selected in the layers list on the left:

- **Move around and zoom**: To move the image or zoom in/out, select the "Lens" tool (button 5 in the image) in the top-left corner toolbar in Napari. You can zoom by scrolling with your mouse/trackpad. Keep in mind that when you have some of the other annotation tools selected, you can zoom or move around in the image by keeping the `SpaceBar` key pressed.
- **Painting tools**: Use the _paint_ tool (button 2) to start drawing. You can adjust the brush size by using the slider on the left. If you want to erase or undo something, use the _erase_ tool (button 1). You can also press `Ctrl+Z` to undo your last changes. Finally, with the _fill_ tool (button 3) you can pour areas of paint on to the image that expand until it finds a border it cannot flow over. 
- **Annotating a new cell**: If you want to start labeling a new cell, you should pick a color that was not used before. You can do that by pressing the `M` key. 
- **Updating annotation of an existing cell**: If you want to keep labeling an existing cell or update its annotation, you can select its color by using the _picker_ tool (button 4). 

::::{tip}
Keep in mind that you can **decrease the opacity** of the annotation layer if you want to see more of the original image (and increase the transparency of the manual annotations). 
::::



