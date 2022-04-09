import os
import shutil
import subprocess
import sys

try:
    import cellpose
except ImportError:
    cellpose = None


def start_cellpose_training(train_folder,
                            *cellpose_args,
                            test_folder=None,
                            out_models_folder=None,
                            **cellpose_kwargs
                            ):
    """
    :param cellpose_args: List of strings that should be passed to cellpose (those arguments that do not require a specific value)
    """
    if cellpose is None:
        return False, "cellpose module is required to train a custom model"

    # Compose the command to be run:
    # TODO: move fast_mode to config?
    python_interpreter = sys.executable
    CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else "0"
    command = "{} {} -m cellpose {} --train" \
              " --use_gpu --fast_mode --dir {} {} ".format(
        "CUDA_VISIBLE_DEVICES=" + CUDA_VISIBLE_DEVICES,
        python_interpreter,
        "--" if "ipython" in python_interpreter else "",
        train_folder,
        "" if test_folder is None else "--test_dir {}".format(test_folder),
    )
    # Add the args:
    for arg in cellpose_args:
        assert isinstance(arg, str), "Arguments should be strings"
        command += "--{} ".format(arg)

    # Add the kwargs:
    for kwarg in cellpose_kwargs:
        command += "--{} {} ".format(kwarg, cellpose_kwargs[kwarg])

    print(command)
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError:
        return False, "Some errors occurred during training"

    if out_models_folder is not None:
        os.makedirs(out_models_folder, exist_ok=True)
        basedir, dirname = os.path.split(out_models_folder)
        if dirname != "models":
            out_models_folder = os.path.join(out_models_folder, "models")

        # Copy new trained models to target folder:
        cellpose_out_model_dir = os.path.join(train_folder, "models")
        shutil.copytree(cellpose_out_model_dir, out_models_folder, dirs_exist_ok=True)

        # Now delete the original folder:
        shutil.rmtree(cellpose_out_model_dir)

    return True, None
