from setuptools import setup
import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    """
    Reading the package version dynamically.
    https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
    """
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name='thinkst-zippy',
    version=get_version("zippy/__init__.py"),
    packages=['zippy'],
    package_data={"": ["*.txt"]},
    entry_points={
        'console_scripts': [
            'zippy=zippy.zippy:main',
        ]
    },
    install_requires=[
        'numpy',
        'brotli'
    ]
)
