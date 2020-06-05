from distutils.core import setup,Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
# from distutils.extension import 
import numpy
# import cv2

setup(
    name = 'post_process', #DL_libs
    version = '0.1.0    ',
    author = 'Dayta',
    ext_modules = [
        Extension("post_process",
            sources=["post_process.pyx", "argmax.cpp"], # Note, you can link against a c++ library instead of including the source
            include_dirs=[".","source" , "/opt/local/include/opencv", "/opt/local/include", numpy.get_include()],
            language="c++",
            library_dirs=['/opt/local/lib', 'source',numpy.get_include()],
            libraries=['opencv_core']),
    ],
    cmdclass = {'build_ext': build_ext},
)


# setup(name='post_process',
#       ext_modules=cythonize([Extension("post_process", ["post_process.pyx"], 
#                                     include_dirs=[numpy.get_include()]
#                                     )]
#                             )
# )

# print("========= Done =========")
# # python3 setup.py build_ext -inplace


# from distutils.core import setup, Extension
# from Cython.Build import cythonize
# import numpy

# setup(
#     ext_modules=[
#         Extension("post_process", ["post_process.c"],
#                   include_dirs=[numpy.get_include()]),
#     ],
# )
print("========= Compilation Done =========")

# # Or, if you use cythonize() to make the ext_modules list,
# # include_dirs can be passed to setup()

# setup(
#     ext_modules=cythonize("my_module.pyx"),
#     include_dirs=[numpy.get_include()]
# )    
