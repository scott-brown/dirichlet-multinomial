# -*- coding: utf-8 -*-
from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import cython_gsl
import numpy

setup(
    name = "dirmult",
    author = "Scott Brown",
    author_email = "sbrown103@gmail.com",
    version = "1.0",
    packages = ["dirmult","dirmult.tests"],
    include_dirs = [cython_gsl.get_include(), numpy.get_include()],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("dirmult.metropolis_hastings",
                             ["dirmult/metropolis_hastings.pyx"],
                             libraries=cython_gsl.get_libraries(),
                             library_dirs=[cython_gsl.get_library_dir()],
                             include_dirs=[cython_gsl.get_cython_include_dir()]),
                   Extension("dirmult.augmented_gibbs",
                             ["dirmult/augmented_gibbs.pyx"],
                             libraries=cython_gsl.get_libraries(),
                             library_dirs=[cython_gsl.get_library_dir()],
                             include_dirs=[cython_gsl.get_cython_include_dir()])]
    )