# import distutils

from code import compile_command


def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration('zjhnp_directory',
                           parent_package,
                           top_path)
    config.add_extension('zjhnp', ['py_reduce.c'])
    return config

if __name__ == "__main__":
    from distutils.core import setup, Extension
    import os
    os.system("nvcc reduce.cu -c -o reduce_cu.o -gencode=arch=compute_80,code=sm_80")
    setup(name='zjhnp', version='1.0', 
        ext_modules=[Extension('zjhnp', ['py_reduce.c'], 
        extra_compile_args=["-I/home/xianyun/anaconda3/lib/python3.8/site-packages/numpy/core/include", "-msse", "-msse2", "-msse3", "-O3"], 
        extra_link_args=["-msse", "-msse2", "-msse3", "-o out reduce_cu.o"])])
    # from numpy.distutils.core import setup
    # setup(configuration=configuration)