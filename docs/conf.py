import os
import sys

sys.path.insert(0, os.path.abspath("."))

project = "SpaceM Annotator"
copyright = "2022, Alberto Bailoni <alberto.bailoni@embl.de>"
author = "Alberto Bailoni <alberto.bailoni@embl.de>"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".venv"]
html_static_path = ["_static"]
templates_path = ["_templates"]
myst_enable_extensions = [
    "colon_fence",
]

html_theme = 'sphinx_rtd_theme'
