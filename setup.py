def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration('zjhnp_directory',
                           parent_package,
                           top_path)
    config.add_extension('zjhnp', ['py_reduce.c'])
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)