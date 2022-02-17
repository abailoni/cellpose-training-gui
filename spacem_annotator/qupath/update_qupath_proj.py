import numpy as np

try:
    import paquo
    from paquo.projects import QuPathProject
    from paquo.images import QuPathImageType
    from paquo.classes import QuPathPathClass
    from shapely.geometry import Polygon
except ImportError:
    paquo = None

from pathlib import Path


def add_image_to_project(qupath_proj_dir, image_path):
    assert paquo is not None, "Paquo library is required to interact with QuPath project"
    if not isinstance(image_path, Path):
        assert isinstance(image_path, str)
        image_path = Path(image_path)

    with QuPathProject(qupath_proj_dir, mode="a") as qp:
        # TODO: In the end I should be able to set allow-duplicates to false
        qp.add_image(image_path, image_type=QuPathImageType.OTHER, allow_duplicates=True)


def delete_image_from_project(qupath_proj_dir, image_id):
    assert paquo is not None, "Paquo library is required to interact with QuPath project"

    if not isinstance(image_id, (int, slice)):
        assert isinstance(image_id, np.ndarray)
        image_id = image_id.item()

    with QuPathProject(qupath_proj_dir, mode="r+") as qp:
        qp.remove_image(image_id)

