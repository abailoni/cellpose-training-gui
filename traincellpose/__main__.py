import argparse

from magicgui.types import FileDialogMode
from magicgui.widgets import show_file_dialog

from traincellpose.core import BaseAnnotationExperiment


def main():
    parser = argparse.ArgumentParser(description='cellpose parameters')

    # settings for CPU vs GPU
    parser.add_argument('--proj_dir', required=False, default=None, type=str, help='Project directory to load')

    args = parser.parse_args()

    proj_dir_path = args.proj_dir

    if proj_dir_path is None:
        proj_dir_path = show_file_dialog(
            mode=FileDialogMode.EXISTING_DIRECTORY,
            caption="Select the project directory",
            # filter="*.json",
            # parent=self.native,
        )

    if proj_dir_path is not None:
        exp = BaseAnnotationExperiment(proj_dir_path)
        exp.run()


if __name__ == '__main__':
    main()
