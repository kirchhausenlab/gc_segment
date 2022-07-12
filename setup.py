from setuptools import setup, find_packages

setup(
    name='gc_segment',
    version='0.1',
    description='',
    url='https://github.com/kirchhausenlab/gc_segment',
    author='Benjamin Gallusser, Mihir Sahasrabudhe',
    author_email='benjamin.gallusser@epfl.ch',
    py_modules=[],
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'imageio',
        'scikit-image',
        'opencv-python',
        'pymaxflow',
        'scikit-learn',
        'ipython',
        'pyyaml',
        'cython',
        'simpleitk',
        'daisy',
        'zarr',
        'funlib.segment @ git+https://github.com/funkelab/funlib.segment@master',
    ],
)
