import numpy as np

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


ext = Extension(
    "py_ur_kin",
    ["py_ur_kin.pyx", "ur_kin.cpp"],
    language="c++",
    include_dirs=[np.get_include()],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)

setup(
    ext_modules = cythonize(ext, language_level='3'),
)
