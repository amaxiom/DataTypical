import re

from setuptools import setup, find_packages


def _version():
    """datatypical.__version__, read without importing the module."""
    with open("datatypical.py", "r", encoding="utf-8") as fh:
        found = re.search(r'^__version__ = "([^"]+)"', fh.read(), re.MULTILINE)
    if found is None:
        raise RuntimeError("datatypical.py does not define __version__")
    return found.group(1)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="datatypical",
    version=_version(),
    author="Amanda S. Barnard",
    description="Explainable instance significance discovery for scientific datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amaxiom/DataTypical",
    py_modules=["datatypical", "datatypical_viz"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
        ],
    },
)