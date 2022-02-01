import unittest

class TestImports(unittest.TestCase):
    def test_spacem_annotator_imports(self):
        import spacem_annotator
        from spacem_annotator.core import BaseAnnotationExperiment
        from spacem_annotator.napari_gui.roi_selection import RoiSelectionWidget


if __name__ == '__main__':
    unittest.main()
