import os
import platform
import subprocess
import yaml


def yaml2dict(path):
    if isinstance(path, dict):
        # Forgivable mistake that path is a dict already
        return path
    with open(path, 'r') as f:
        readict = yaml.load(f, Loader=yaml.FullLoader)
    return readict


def get_path_components(path):
    folders = []
    while 1:
        path, folder = os.path.split(path)
        if folder != "":
            folders.append(folder)
        elif path != "":
            folders.append(path)
            break
    folders.reverse()
    return folders


def open_path(path):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        try:
            subprocess.Popen(["xdg-open", path])
        except FileNotFoundError:
            # Try with XFCE file manager:
            subprocess.Popen(["thunar", path])

def recursive_dict_update(source, target, zero_depth=True):
    # if zero_depth:
    #     pass
        # target = deepcopy(target)
    for key, value in source.items():
        if isinstance(value, dict):
            sub_target = target[key] if key in target else {}
            target[key] = recursive_dict_update(source[key], sub_target, zero_depth=False)
        else:
            target[key] = source[key]
    return target
