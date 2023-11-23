"""
pip install -e .
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
        "absl-py",
        "chex",
        "jax",
        "jaxlib",
        "numpy",
        "eutils"
        # TODO: Add meshio
    ],
    extras_require={
        "develop": ["pytest"],
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

