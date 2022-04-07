import os

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
