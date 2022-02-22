---
substitutions:
  move_tool: |
    ```{image} ../images/qupath/move-tool.png
    :alt: Move tool
    :width: 30px
    ```
  brush_tool: |
    ```{image} ../images/qupath/brush-tool.jpg
    :alt: Brush tool
    :width: 30px
    ```
  wand_tool: |
    ```{image} ../images/qupath/wand-tool.jpg
    :alt: Wand tool
    :width: 30px
    ```

---

:::{image} ../images/qupath/qupath-logo.png
:alt: QuPath logo 
:width: 15%
:align: right
:::

# Labeling in QuPath
::::{warning}
Support for QuPath is still experimental and under development
::::

## Getting started

:::{image} ../images/qupath/qupath-starting-interface.jpg
:alt: QuPath starting interface 
:width: 100%
:align: center
:::

### Starting QuPath
1. Once you have selected the regions of interest with the SpaceM Annotator tool, you should close the tool and launch QuPath from the installed applications.
2. In the SpaceM Annotator project folder, you can find a folder named `QuPathProject`. To load the regions of interest in QuPath, you should drag this folder into QuPath. Alternatively, you can click on the Menu item _File/Projects.../Open Project_ and open the `project.qpproj` file in the `QuPathProject` folder.
3. After loading the project, you should see the list of regions of interest on the left. You can double-click on one image to open it in the viewer.


<br/><br/>
:::{image} ../images/qupath/channel-contrast.jpg
:alt: Adjust channels contrast
:width: 35%
:align: right
:::
### Adjusting image channels contrast

For making the annotation process easier, it may be helpful to adjust the contrast of the image channels. To do so, first load one image and then select the _View/Brightness-Contrast_ menu item (or use the \<Shift-C> keyboard shortcut). 

From this window, you can select image channels and see the pixel-intensity distribution, adjust the _Min display_ and _Max display_ sliders to adjust the contrast, or changing the colormap of the single image channels.

<br/><br/><br/><br/><br/><br/>

:::{image} ../images/qupath/channel-viewer.jpg
:alt: Channel viewer
:width: 35%
:align: right
:::



### Viewing image channels one by one
If you open the _Channel Viewer_ from the Menu "_View/Mini viewers.../Show channel viewer_", you will be able to see each channel of the image singularly. 

This can be very handy when manually annotating. 

<br/><br/><br/>

***
## Annotating cells in QuPath
When you start annotating cells, you create Annotation objects. The list of the annotated cells can be seen by clicking on the _Annotations_ tab (see image below). You can easily select/delete/merge cell annotations by selecting items in the Annotations list.    

::::{important}
Remember that you should label all cells inside a region of interest! And when you are done annotating, do not forget to save your results (\<Ctrl+S> keyboard shortcut).
::::


:::{image} ../images/qupath/annotations-list.jpg
:alt: QuPath starting interface 
:width: 100%
:align: center
:::


### Annotation tools
On the toolbar at the top, you have all the drawing tools you will need. For annotating cells, you will only need the following tools: Move, Brush, and Wand. 

#### {{ move_tool }}  Move tool 

#### {{ brush_tool }}  Brush tool

#### {{ wand_tool }}  Wand tool


::::{note}
For a complete overview of QuPath annotation tools, check out the [QuPath documentation](https://qupath.readthedocs.io/en/stable/docs/starting/annotating.html#brush-brush-tool).
::::
