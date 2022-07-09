from setuptools import setup

setup(
    name='gc_segment',
    version='0.1',
    description='',
    url='https://github.com/kirchhausenlab/gc_segment',
    author='Benjamin Gallusser, Mihir Sahasrabudhe',
    author_email='benjamin.gallusser@epfl.ch',
    py_modules=[],
    install_requires=[
        'matplotlib',
        'imageio',
        'scikit-image',
        'opencv-python',
        'maxflow',
        'scikit-learn',
        'ipython',
        'pyyaml',
        'funlib.segment @ git+https://github.com/funkelab/funlib.segment@master',
    ],
    scripts=[
        "VolumeAnnotator"
    ],
)
