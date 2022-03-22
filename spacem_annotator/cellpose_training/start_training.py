import os
import sys

try:
    import cellpose
except ImportError:
    cellpose = None


def start_cellpose_training(train_folder,
                            test_folder=None,
                            *cellpose_args,
                            **cellpose_kwargs
                            ):
    """
    :param cellpose_args: List of strings that should be passed to cellpose (those arguments that do not require a specific value)
    """
    assert cellpose is not None, "Cellpose module is needed to training a new segmentation model"
    # Compose the command to be run:
    # TODO: move fast_mode to config?
    python_interpreter = sys.executable
    command = "{} -m cellpose {} --train" \
              " --use_gpu --fast_mode --dir {} {} ".format(
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

    os.system(command)

    # TODO: copy models when done?


def update_training_config():
    pass
