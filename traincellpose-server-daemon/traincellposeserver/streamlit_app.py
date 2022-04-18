import argparse
import os
import pathlib
import time
import zipfile
import random
import streamlit as st

import os
import shutil
import subprocess
import sys

import yaml

try:
    import cellpose
except ImportError:
    cellpose = None


def start_cellpose_training(training_folder):
    try:
        import cellpose
    except ImportError:
        return False, "cellpose module is required to train a custom model"

    # Assert that training data is present:
    training_images_dir = os.path.join(training_folder, "training_images")
    training_config_path = os.path.join(training_folder, "train_config.yml")
    if not os.path.exists(training_folder):
        return False, "Temp training data folder not found, something went wrong: {}".format(training_folder)
    if not os.path.exists(training_images_dir):
        return False, "Folder named `training_images` not found in zip file"
    if not os.path.exists(training_config_path):
        return False, "Training config `train_config.yml` not found in zip file"


    # Load config:
    with open(training_config_path, 'r') as f:
        training_config = yaml.load(f, Loader=yaml.FullLoader)

    # # FIXME: TEMP check
    # shutil.copytree(training_images_dir, os.path.join(training_images_dir, "models"))
    # training_was_successful, error_message = True, None

    training_was_successful, message = \
        _run_training(training_images_dir,
                      *training_config.get("cellpose_args", []),
                      # out_models_folder=os.path.split(train_folder)[0],
                      **training_config.get("cellpose_kwargs", {}))

    return training_was_successful, message


def _run_training(train_folder,
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
    # TODO: convert command to list of elements and disable the option shell=True
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

    # print(command)
    output = subprocess.run(command, shell=True, capture_output=True, text=True)
    if output.returncode != 0:
        return False, output.stderr

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

    return True, output.stdout


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--temp_data_dir', required=False,
                        default=os.path.join(os.getcwd(), "_training_data"), type=str,
                        help='Directory where to store temporary training files')

    args = parser.parse_args()
    data_dir = args.temp_data_dir
    os.makedirs(data_dir, exist_ok=True)

    uploaded_file = st.file_uploader("Upload cellpose training data", type="zip")
    if uploaded_file is not None:
        # Display bar:
        # my_bar = st.progress(0)


        model_name, _ = os.path.splitext(uploaded_file.name)
        training_id = random.randint(0, 10000)
        training_dir = os.path.join(data_dir, "training_data_{}_{}".format(model_name, training_id))
        training_dir = os.path.abspath(training_dir)

        # TODO: when should I delete this data...?

        # Unzip data:
        os.makedirs(training_dir, exist_ok=True)
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(training_dir)

        # Display some messages:
        st.info("Training data is being processed")
        # my_bar.progress(10)

        # Start training:
        training_dir = os.path.join(training_dir, model_name)
        tick = time.time()
        with st.spinner('Training has started...'):
            training_was_successful, output_message = start_cellpose_training(training_dir)
        training_runtime = time.time() - tick

        if not training_was_successful:
            # st.error("Training could not be completed")
            st.error("Training could not be completed. See error message below:")
            st.write(output_message)
        else:
            # my_bar.progress(90)
            models_dir = os.path.join(training_dir, "training_images", "models")
            out_zip_file = os.path.join(training_dir, "{}_trained_models.zip".format(model_name))

            with zipfile.ZipFile(out_zip_file, mode="w") as archive:
                for file_path in pathlib.Path(models_dir).iterdir():
                    archive.write(file_path, arcname=file_path.name)

            st.success("Training was completed successfully in {} s. You can now download the trained cellpose model:".format(training_runtime))
            with open(out_zip_file, "rb") as fp:
                btn = st.download_button(
                    label="Download the trained models (zip)",
                    data=fp,
                    file_name=os.path.split(out_zip_file)[1],
                    mime="application/zip"
                )

            # my_bar.progress(100)
            st.download_button('Download training log', output_message)

        # TODO: cleanup?

        # # To read file as bytes:
        # bytes_data = uploaded_file.getvalue()
        # #st.write(bytes_data)
        # st.write(f"The file has {len(bytes_data)} bytes")
        # # process the file
        # processed = bytes([0] * len(bytes_data))
        # # send to webapp to make downloadable link
        # outname = "processed.bin"
        # b64 = base64.b64encode(bytes_data).decode()
        # href = f'<a href="data:file/zip;base64,{b64}" download=\'{outname}\'>\
        #        Download processed file\
        #    </a>'
        # st.sidebar.markdown(href, unsafe_allow_html=True)
        # st.sidebar
