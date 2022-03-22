import yaml


def yaml2dict(path):
    if isinstance(path, dict):
        # Forgivable mistake that path is a dict already
        return path
    with open(path, 'r') as f:
        readict = yaml.load(f, Loader=yaml.FullLoader)
    return readict
