from distutils.core import setup
import setuptools

NAME = 'onedcelltrack'
AUTHOR = 'Miguel Atienza'
REQUIREMENTS=['trackpy', 'scikit-video', 'scikit-image', 'pandas', 'nd2reader', 'numba', 'omero-py', 'cellpose==2.0']
JUPYTER_REQUIREMENTS=['jupyterlab' , 'ipywidgets', 'ipympl']
LICENSE="MIT"
VERSION="0.1"

setup(
    name=NAME,
    license=LICENSE,
    version=VERSION,
    author=AUTHOR,
    install_requires=REQUIREMENTS,
    extras_require = {
        'jupyter': JUPYTER_REQUIREMENTS},
    tests_require=['pytest'],
    packages=setuptools.find_packages())
    #packages=setuptools.find_packages(exclude=('tests/')))
