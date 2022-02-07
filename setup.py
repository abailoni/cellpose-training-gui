import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spacem_annotator",
    author="Alberto Bailoni",
    author_email="alberto.bailoni@embl.de",
    description="Library for annotating singe-cell microscopy images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.1.1",
    packages=setuptools.find_packages(),
)

