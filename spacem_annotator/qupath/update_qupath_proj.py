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
        # Check if image is already in QuPath project:
        add_new_image = True
        img_id = qp._image_provider.id(qp._image_provider.uri(image_path))
        for entry in qp.images:
            uri = qp._image_provider.id(entry.uri)
            if img_id == uri:
                add_new_image = False
                break

        # Now add it to the project:
        if add_new_image:
            qp.add_image(image_path, image_type=QuPathImageType.OTHER, allow_duplicates=False)


def delete_image_from_project(qupath_proj_dir, image_id):
    assert paquo is not None, "Paquo library is required to interact with QuPath project"

    print("Deleting ROI {} from QuPath project".format(image_id))
    if not isinstance(image_id, (int, slice)):
        assert isinstance(image_id, np.ndarray)
        image_id = image_id.item()

    with QuPathProject(qupath_proj_dir, mode="r+") as qp:
        image_entry_id = '{}'.format(image_id+1)
        qp.remove_image(image_entry_id)
