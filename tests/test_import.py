import unittest

class TestImports(unittest.TestCase):
    def test_spacem_annotator_imports(self):
        import traincellpose
        from traincellpose.core import BaseAnnotationExperiment
        from traincellpose.napari_gui.roi_selection import RoiSelectionWidget


if __name__ == '__main__':
    unittest.main()
