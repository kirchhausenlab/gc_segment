# gc_segment
Annotation-based organelle segmentation using graph cuts

## Installation

1. Clone this repository
1. Install the `boost` library: On Linux Debian/Ubuntu systems, you can run `sudo apt install libboost-dev && sudo apt install libboost-all-dev`
1. Install [anaconda/miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
1. Create a conda environment using the `environment.yml file` in this repo: `conda env create -f gc_segment/code/environment.yml`
1. Activate the environment with `conda activate [theEnvNameYouPicked]`
1. Run `pip install git+https://github.com/funkelab/funlib.segment` to install a required dependency directly from GitHub.
1. You are ready to go :)
