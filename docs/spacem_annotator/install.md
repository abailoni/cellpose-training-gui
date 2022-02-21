# Getting started

## Annotating on the Surface Go tablet
The tool has been installed on the Surface Go tablet (you can find it in office 120). 

To start the tool, follow these steps:  
- Start "_Anaconda Powershell Prompt_" from the applications
- Then, type the following two commands:
    ```shell
    conda activate spacem_annotator_env
    python -m spacem_annotator
    ````


## Manual installation
If you prefer to install it on your machine, you will need [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [anaconda](https://docs.anaconda.com/anaconda/install/index.html). The tool can be installed on Windows, Mac, or Linux. 

Once you have `conda`, open a terminal and run the following command:
:::{code-block} shell
conda create --name spacem_annotator_env -c abailoni -c conda-forge -c sdvillal spacem_annotator
:::

After being installed, you can start the tool by typing the following two commands in a command line:

```shell
conda activate spacem_annotator_env
python -m spacem_annotator
````

## Selecting a project directory
After starting SpaceM Annotator, you will be asked to select the project directory.

**What is the project directory?** It is the folder containing all the generated data and images. If you are starting a new annotation project, then create and select an empty folder. Otherwise, select the directory of an existing annotation project.

::::{tip}
You can also pass the path of the project directory as a command line argument when you run SpaceM Annotator:
:::{code-block} shell
python -m spacem_annotator --proj_dir /path/to/the/project/directory
:::
::::

### The Starting Window

:::{image} images/starting-gui.jpg
:alt: Starting GUI Window
:width: 250px
:align: right
:::
After selecting a project directory, you have two choices:
1. [](select_rois): First, select this option to add images to the project and select some regions of interest.
2. [](label_rois): After selecting some regions of interest, click on this option to start annotating your regions of interest.

