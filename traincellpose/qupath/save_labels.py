import os

try:
    import paquo
    from paquo.projects import QuPathProject
    from paquo.images import QuPathImageType
    from paquo.classes import QuPathPathClass
    from shapely.geometry import Polygon
    from paquo.jpype_backend import JClass

    LabeledImageServer = JClass("qupath.lib.images.servers.LabeledImageServer")
    ColorTools = JClass("qupath.lib.common.ColorTools")
    JavaString = JClass('java.lang.String')
    QP = JClass("qupath.lib.scripting.QP")
except ImportError:
    paquo = None


def export_labels_from_qupath(qupath_proj_dir, out_folder, filename_postfix=None):
    assert paquo is not None, "Paquo library is required to interact with QuPath project"

    with QuPathProject(qupath_proj_dir, mode="r+") as qp:
        for image in qp.images:
            # Export labels:
            image_name = os.path.split(image.uri)[1]
            assert "ome.tif" in image_name
            image_name = image_name.replace(".ome.tif", ".tif")
            if filename_postfix is not None:
                assert isinstance(filename_postfix, str)
                image_name = image_name.replace(".tif", "_{}.tif".format(filename_postfix))
            image_data = image.java_object.readImageData()
            out_labels = LabeledImageServer.Builder(image_data).backgroundLabel(0, ColorTools.BLACK).downsample(
                1).useInstanceLabels().multichannelOutput(False).build()

            # Write out ome.tif file:
            QP.writeImage(out_labels, JavaString(os.path.join(out_folder, image_name)))


