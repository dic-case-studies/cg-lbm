"""
pip install -e . # For installing locally
pip install "cglbm[develop]" -e . # For development purposes
"""

from setuptools import find_packages
from setuptools import setup

setup(
    name="cglbm",
    version="0.0.1",
    description=("A color gradient lattice boltzmann simulator"),
    author="CG LBM Authors",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    license="Apache 2.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "chex",
        "etils",
        # Installation of jax should not be done from here
        "jax",
        "jaxlib",
        "numpy",
        "orbax-checkpoint"
    ],
    extras_require={
        "develop": [
            "absl-py",
            "einops",
            "h5py",
            "ipykernel",
            "meshio",
            "pandas",
            "pyarrow",
            "pytest"
        ],
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
    ],
    keywords="JAX Color Gradient Lattice Bolzman"
)
