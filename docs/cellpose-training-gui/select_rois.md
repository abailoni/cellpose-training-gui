# Select Regions of Interest

After clinking on `Select Regions of Interest` in the starting window of the annotation tool, a napari viewer will open and let you add images to the project and select regions of interest. 

:::{image} ./images/selecting_rois.jpg
:alt: Napari interface to select ROIs
:width: 100%
:align: center
:::


## What are regions of interest? 
After adding an image to the project, you will need to select one or more region of interests (rectangular boxes) that will indicate the parts of the image that you want to manually annotate and that will be used in the following to train the segmentation algorithm. 

### How to select good regions of interest
When you select a region of interest, keep in mind that **you will need to manually annotate all cells inside that region of interest**. Each region of interest should contain **at least several dozens of cells, ideally ~100**. 

These region of interests will be used to train the segmentation algorithm, so you should **select the most representative parts of your data**. If there are some particularly challenging parts of your data the you would like the segmentation algorithm to properly segment, then you should also select some regions of interest from these difficult parts. 

### Some example cases
Here there are some possible scenarios describing how you should proceed while choosing the regions of interest:

- **Scenario 1**: Your images/datasets present several artifacts close to the borders, but you only care to have an accurate segmentation in the central part of the image, where image quality is higher and there are no artifacts. In this case, you can select regions of interest only from the central area of your image.
- **Scenario 2**: Your images/datasets present several artifacts in all parts of the image, and you would like cells to be properly segmented even in parts of the image where these artifacts are present. Then, you should also include regions of interest where artifacts are present (but you should also include regions that are easier to segment)
- **Scenario 3**: You acquired several images that look somehow different (different acquisition method, brightness, etc). Then, you should select regions of interest across all your images/datasets.

## Adding images to the annotation project
### Adding the first image
:::{image} ./images/add-new-image.jpg
:alt: Add new image options
:width: 250px
:align: right
:::
On the right side of the napari viewer, select the paths of an image you want to add to the project (either insert the path manually or click on the _Select file_ buttons).

For each image, you can add up to four channels:

- **Main channel**: This is the main channel that will be used for segmentation (usually it is the bright-field channel)
- **DAPI channel**: If available, add a DAPI channel because it often improves segmentation results
- **Extra channels**: If you have additional channels that may be helpful for the manual annotation process, you can load them. They won't be used by the segmentation algorithm though.

At the moment you can load images in `.tif`, `.png`, and `zarr` formats (currently working on adding compatibility with all image formats supported by Napari). If the loaded image (or zarr group) has multiple channels or sub-images in it, an additional choice-box will appear, letting you choose the channel you want to use. 

After you select the files with the image channels, click on `Add selected images to annotation project`. This will add the image to the project.


::::{note}
**ZARR images**: Currently, if you click on `Select File` you will only be able to select files, but not directories. Thus, if you want to load a zarr image, you can either paste the path of the zarr folder directly into the path field, or you can click on `Select file` button and then select the `.zarray` or `.zgroup` files included in the zarr folder of the image (for this, you may need to set your OS to display hidden files).
:::{image} ./images/add-zarr-image.jpg
:alt: Add a zarr image
:width: 60%
:align: center
:::
::::


::::{tip}
If you copy the images that you want to add to the project into the `traincellpose` project directory itself (for example in the `input_images` subdirectory), then only relative paths will be stored and the input images will be found even if you later decide to move your annotation project directory to another location or another machine.
::::




### Browsing images in the project and adding new ones
After adding the first image, you can add another image by selecting the _"Add new image"_ option from the dropdown menu _"Shown image"_ in the top-right corner. Alternatively, using the same menu, you can switch between the images in the project.

:::{image} ./images/adding-new-image.jpg
:alt: Modifying added images
:width: 100%
:align: center
:::


### Updating image channels
At any moment, you can update the channel paths of an image added to the project. Select the image you want to update, click on `Update channel paths for selected image`, update the paths, and then click on the `Update image paths in annotation project` button.

## Selecting regions of interest in Napari
Once you selected an image in your project, you will see that a new shapes layer named "_Regions of interest_" has been loaded in Napari.


After selecting the shape layer, you can perform the following three actions:

:::{image} ./images/select_roi_tools.jpg
:alt: Tools for creating ROIs
:width: 50%
:align: right
:::

1. **Move around and zoom**: To move the image or zoom in/out, make sure that you select the "Lens" tool in the top-left corner toolbar in Napari. You can zoom by scrolling with your mouse/trackpad.
2. **Create ROIs**: To create a new region of interest, select the Napari "Rectangle" tool (see button 2 in the image on the right) and then draw one or more boxes. Then you can click on the `Save Regions of Interest` button on the right to immediately save the selected ROIs.
3. **Delete or modify ROIs**: In this case, select the "Select" tool in Napari (see button 3 in the image on the right). Then you can click on a box and press \<backspace> to delete the rectangle. Otherwise, you can drag the rectangle around or adjust the corners.  


::::{tip}
While you have the "rectangle" or "select" tools selected, you can **zoom or move around in the image**  by keeping the `SpaceBar` key pressed.
::::

::::{note}
ROIs are automatically updated and saved when you switch to another image or go back to the starting window. Due to the current implementation, updating ROIs may take some time (depending on the size of the ROI) and the Napari interface may freeze for few seconds.
::::
