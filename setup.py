import setuptools
from distutils.core import setup, Extension
import numpy
# from Cython.Build import cythonize

with open("README.md", 'r') as readme:
    long_description = readme.read()

setup(
    name='stochastic-arrow',
    version='0.0.10',
    packages=['arrow'],
    author='Ryan Spangler',
    author_email='spanglry@stanford.edu',
    url='https://github.com/CovertLab/arrow',
    license='MIT',
    ext_modules=[
        Extension(
            "arrow",
            ["arrow/arrow.pyx"],
            include_dirs=[numpy.get_include()])],
    long_description=long_description,
    long_description_content_type='text/markdown')
