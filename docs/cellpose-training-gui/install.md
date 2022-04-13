# Getting started

[comment]: <> (## Annotating on the Surface Go tablet)

[comment]: <> (The tool has been installed on the Surface Go tablet &#40;you can find it in office 120&#41;. )

[comment]: <> (To start the tool:  )

[comment]: <> (- Start "_Anaconda Powershell Prompt_" from the applications)

[comment]: <> (- Then, type the following two commands:)

[comment]: <> (    ```shell)

[comment]: <> (    conda activate traincellpose_env)

[comment]: <> (    python -m traincellpose)

[comment]: <> (    ````)

[comment]: <> (- Next you will need to {ref}`select a project directory<proj_dir>`)



## How to install
Training a custom cellpose model requires a pytorch installation with CUDA support and a local GPU (only supported on Linux and Windows). 

However, a basic version of the `traincellpose` tool is available for all OSs (Linux, Mac, Windows) and does not require a local GPU. In that case, you have the option to install the `traincellpose-server` tool on a remote server with CUDA support and conveniently train your model remotely.

### Basic installation (no local training of Cellpose models)
This basic installation is available for all OSs (Linux, Mac, Windows). To install it, you will need [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [anaconda](https://docs.anaconda.com/anaconda/install/index.html).

To install it, simply run the following conda command:
:::{code-block} shell
conda create --name traincellpose_env -c abailoni -c conda-forge traincellpose
:::


#### Installing QuPath
To use the tool, you need to install QuPath. If you don't have it already installed on your machine, the easiest may be to install it directly via conda:
- Activate the conda env you create previously with `conda activate traincellpose_env`
- Then install QuPath: `conda install qupath -c sdvillal`

Otherwise, if you want to install the latest version of QuPath manually or you already have it installed on your machine, please follow the instructions [here](https://paquo.readthedocs.io/en/latest/installation.html#install-qupath) to make sure that the tool will find your QuPath installation. Many times the QuPath installation found automatically, otherwise you will need to specify the path in a config file as described [here](https://paquo.readthedocs.io/en/latest/configuration.html#the-paquo-toml-file).

[comment]: <> (`python -m paquo config -l -o /home/bailoni/miniconda3/envs/trCell1/lib/python3.9`)

### Full installation with CUDA support (requires a local GPU)
This type of installation only works on Linux and Windows. The installation process is similar to the one for the basic installation, but here you should also install pytorch with CUDA support and then install `cellpose` via `pip`. The specific installation commands depend on your OS and the CUDA version you want to use (see [PyTorch website](https://pytorch.org/get-started/locally/) for more details), but they may look something like:
- Install `traincellpose` and `pytorch`: `conda create --name traincellpose_env -c abailoni -c pytorch -c conda-forge pytorch torchvision torchaudio cudatoolkit=11.3 traincellpose`
- Activate the environment: `conda activate traincellpose_env` 
- Install cellpose: `pip install cellpose`
- Finally, make sure that your QuPath installation is found (see Basic installation for details)

### Installing the traincellpose-server-daemon 
Train your models remotely on a server with GPU. More details coming soon.

## How to run
After having it installed, simply run:
```shell
conda activate traincellpose_env
traincellpose
````

(proj_dir)=
### Selecting a project directory
After starting the tool, you will be asked to select the project directory.

**What is the project directory?** It is the folder containing all the generated data and images. If you are starting a new annotation project, then create and select an empty folder. Otherwise, select the directory of an existing annotation project.

::::{tip}
You can also pass the path of the project directory as a command line argument when you run SpaceM Annotator:
:::{code-block} shell
traincellpose -d /path/to/the/project/directory
:::
::::

[comment]: <> (### The Starting Window)

[comment]: <> (:::{image} images/starting-gui.jpg)

[comment]: <> (:alt: Starting GUI Window)

[comment]: <> (:width: 250px)

[comment]: <> (:align: right)

[comment]: <> (:::)

[comment]: <> (After selecting a project directory, you have two choices:)

[comment]: <> (1. []&#40;select_rois&#41;: First, select this option to add images to the project and select some regions of interest.)

[comment]: <> (2. []&#40;label_rois/label_rois&#41;: After selecting some regions of interest, click on this option to start annotating your regions of interest in Napari.)

