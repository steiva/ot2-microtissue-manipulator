# from setuptools import setup, Extension
# import pybind11

# ext = Extension(
#     "frst",
#     ["frst.cpp"],
#     include_dirs=[pybind11.get_include()],
#     language="c++",
#     extra_compile_args=["/std:c++17"],
# )

# setup(
#     name="frst",
#     version="0.0.1",
#     ext_modules=[ext],
# )

from setuptools import setup, Extension
import pybind11

setup(
    name="frst",
    ext_modules=[
        Extension(
            "frst",
            ["frst.cpp"],
            include_dirs=[pybind11.get_include()],
            extra_compile_args=["/O2", "/std:c++17"],
        )
    ],
)