from setuptools import setup

setup(
    name='ZipPy setup file',
    version='0.1.2',
    packages=['zippy'],
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