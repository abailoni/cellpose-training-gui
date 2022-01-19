from sys import executable, argv
from subprocess import check_output
from PyQt5.QtWidgets import QFileDialog, QApplication


def gui_fname(directory='./'):
    """Open a file dialog, starting in the given directory, and return
    the chosen filename"""
    # run this exact file in a separate process, and grab the result
    dir = check_output([executable, __file__, directory])
    return dir.strip().decode("utf-8")


if __name__ == "__main__":
    directory = argv[1]
    app = QApplication([directory])
    dirname = QFileDialog.getExistingDirectory(None, "Select a file...",
                                        directory)
    print(dirname)
