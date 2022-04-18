import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

install_deps = ['cellpose', 'streamlit', 'pyyaml', 'imagecodecs']

setuptools.setup(
    name="traincellposeserver",
    author="Alberto Bailoni",
    author_email="alberto.bailoni@embl.de",
    description="Daemon tool to train cellpose models on a server with CUDA support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.1.0",
    packages=setuptools.find_packages(),
    install_requires=install_deps,
    entry_points={
        'console_scripts': [
            'traincellposeserver = traincellposeserver.__main__:main']
    }
)

