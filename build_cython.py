"""Compile the Cython code for the encoding module in place to support editable installs."""

import glob
import os
import shutil
from distutils.command.build_ext import build_ext
from distutils.core import Distribution

import numpy
from Cython.Build import cythonize
from setuptools import Extension

# Define the extension
extensions = [
    Extension(
        name="bio_datasets.structure.pdbx.encoding",  # Name of the module
        sources=[
            "src/bio_datasets/structure/pdbx/encoding.pyx"
        ],  # Path to your Cython file
        include_dirs=[numpy.get_include()],  # Include NumPy headers if needed
    )
]

cythonized_extensions = cythonize(
    extensions, compiler_directives={"language_level": 3, "boundscheck": False}
)

# Create a distribution object
dist = Distribution({"ext_modules": cythonized_extensions})
dist.script_name = "setup.py"
dist.script_args = ["build_ext", "--inplace", "--verbose"]

# Run the build_ext command
cmd = build_ext(dist)
cmd.ensure_finalized()
cmd.run()

# Define the source pattern and target path
source_pattern = os.path.join(
    "build", "lib.*", "bio_datasets", "structure", "pdbx", "encoding*.so"
)
target_dir = os.path.join("src", "bio_datasets", "structure", "pdbx")

# Find the .so file with the potential suffix
so_files = glob.glob(source_pattern)

# Ensure that exactly one .so file is found
if len(so_files) == 1:
    source_path = so_files[0]
    target_path = os.path.join(target_dir, os.path.basename(source_path))
    # Copy the .so file from the build directory to the target directory
    shutil.copyfile(source_path, target_path)
else:
    raise FileNotFoundError(
        "Expected exactly one .so file, found: {}".format(len(so_files))
    )
