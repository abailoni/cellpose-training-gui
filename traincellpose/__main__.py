import argparse

from traincellpose.core import BaseAnnotationExperiment


def run_GUI(args):
    from magicgui.types import FileDialogMode
    from magicgui.widgets import show_file_dialog

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


def start_training_func(args):
    assert args.proj_dir is not None, "Specify project directory with argument --proj_dir"
    proj_dir_path = args.proj_dir
    model_name = args.model_name
    exp = BaseAnnotationExperiment(proj_dir_path)
    # print("test")
    training_was_successful, error_message = exp.run_cellpose_training(model_name)
    if not training_was_successful:
        raise Exception(error_message)


def setup_training_func(args):
    assert args.proj_dir is not None, "Specify project directory with argument --proj_dir"
    proj_dir_path = args.proj_dir
    model_name = args.model_name
    exp = BaseAnnotationExperiment(proj_dir_path)
    exp.setup_cellpose_training_data(model_name)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--proj_dir', required=False, default=None, type=str, help='Project directory to load')
    parser.set_defaults(func=run_GUI)

    subparsers = parser.add_subparsers(dest='subparser')

    # run_GUI_parser = subparsers.add_parser('run_GUI')
    # run_GUI_parser.set_defaults(func=run_GUI)

    start_training = subparsers.add_parser('train')
    start_training.set_defaults(func=start_training_func)
    start_training.add_argument(
        '-m', '--model_name', required=True, dest='model_name', help='Name of the model to be trained')

    setup_training = subparsers.add_parser('setup_training')
    setup_training.set_defaults(func=setup_training_func)
    setup_training.add_argument(
        '-m', '--model_name', required=True, dest='model_name', help='Name of the model to be trained')

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
