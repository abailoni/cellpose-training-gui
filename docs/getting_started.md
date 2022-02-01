# Getting started

## Installation
_Coming soon..._

## Running SpaceM annotator
After being installed, you can start a new project by typing the following two commands in a shell:

```shell
conda activate spacem_annotator_env
python -m spacem_annotator
````

## Selecting a project directory
After running the program, you will be asked to select the project directory

**What is the project directory?** It is the folder containing all the generated data and images. If you are starting a new annotation project, then create and select an empty folder. Otherwise, select the directory of an existing annotation project.

Tip: You can also directly pass the path of the project directory as a command line argument:
```shell
python -m spacem_annotator --proj_dir /path/to/the/project/directory
```

## The Starting Window

```{image} starting-gui-2.jpg
:alt: Starting GUI
:width: 250px
:align: center
```
After selecting a project directory, the main window will present two choices:
1. [](select_rois)
2. [](label_rois)



