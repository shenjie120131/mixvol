from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="mixvol.binomial",
        sources=["mixvol/binomial.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
    ),
]

setup(
    name="mixvol",
    version="0.1.0",
    packages=["mixvol"],
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
        }
    ),
    install_requires=["numpy", "scipy", "Cython"],
    zip_safe=False,
)
