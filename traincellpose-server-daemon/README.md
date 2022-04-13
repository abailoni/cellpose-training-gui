# Installation
- Create a conda environment with pyTorch and CUDA support
  - This could look like `conda create --name trCellServer pytorch torchvision torchaudio cudatoolkit=11.3  -c pytorch`
- Activate the conda environment you created in the previous step: `conda activate trCellServer` 
- `pip install -e "vcs+protocol://github.com/abailoni/cellpose-training-gui#egg=traincellposeserver&subdirectory=traincellpose-server-daemon"` 

# How to run
- Activate env
- `python -m traincellposeserver`
- Optionally, you can specify a directory where to store the temporary training data:
- `python -m traincellposeserver -d \PATH-TO-TEMP-DATA-DIR`
