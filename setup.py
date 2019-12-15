import numpy as np

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


ext = Extension(
    "py_ur_kin",
    ["py_ur_kin.pyx", "ur_kin.cpp"],
    language="c++",
    include_dirs=[np.get_include()],
    compiler_directives={'language_level': '3'},
    # extra_compile_args=['--openmp'],
    # extra_link_args=['--openmp'],
)

setup(
    ext_modules = cythonize(ext),
)
