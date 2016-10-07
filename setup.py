from setuptools import setup, Extension
import numpy
import sys

if sys.platform == 'linux'
    extra_compile_args = ['-fopenmp'
    extra_link_args = ['-fopenmp']
else:
    extra_compile_args = ['-/openmp']
    extra_link_args = ['-/openmp']



ext_modules=[
    Extension(
        "heat_diffusion_experiment.cython_versions",
        ["heat_diffusion_experiment/cython_versions.pyx"], 
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
  name = 'heat_diffusion_experiment',
  ext_modules = ext_modules,
  include_dirs=[numpy.get_include()],
  packages=['heat_diffusion_experiment'],
)
