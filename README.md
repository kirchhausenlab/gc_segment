# gc_segment
Annotation-based organelle segmentation using graph cuts

## Getting started

1. Install some version of anaconda, e.g. [miniconda](https://docs.conda.io/en/latest/miniconda.html).
1. Open the terminal.
1. Create a new conda environment: `conda create -n gc_segment python` and activate it: `conda activate gc_segment`.
1. Install the complete C++ Boost Library
   * **Ubuntu**: `apt-get install libboost-all-dev`
   * **Mac**: `brew install boost`
   * **Windows**: Download from [boost.org](https://www.boost.org/users/download/). Unzip and follow [instructions](https://www.boost.org/doc/libs/1_83_0/more/getting_started/windows.html) provided by boost.org. 
1. Install the boost library: `conda install -c conda-forge boost`.
1. Pip install Cython: `pip install cython`.
1. Clone this git repository `git clone https://github.com/kirchhausenlab/gc_segment.git` and pip install it `pip install -e ./gc_segment`.

## <a name="generalusage"></a> General Usage
1. Move into the repository: `cd gc_segment`.
1. Adapt `configs/template.yaml` to your dataset (see [Config files](#configs) below for instructions)
1. Run 
   - `python VolumeAnnotator.py --cfg <path-to-config>` to create annotation seeds; and 
   - `python segment.py --cfg <path-to-config>` to generate a segmentation based on the annotation seeds.

## Available features

1. **Configuration files**: As each annotation task had associated options (ROI, number of supervoxels, etc.), configuration files are associated with each annotation to define GUI and segmentation options in a compact fashion.
1. **Axis views and sliders**: The tool offers the choice to view the volume along any axis. It further allows easy scrolling with a slider through planes perpendicular to this axis to visualize the volume.
1. **Pan mode**: To handle large volumes, the tool offers an option to pan through the volume. It was observed that if the volume size is large, it becomes more difficult for the annotator to maintain precision while annotating. To circumvent this problem, the tool is constrained to visualize a fixed size defined by number of voxels, with the added option of being able to pan through the entire larger volume.
1. **Annotation modes**: There are four annotation modes offered: organelle, background, eraser, and click. When in the organelle or background modes, any annotations made are automatically registered as ground truth for the appropriate class. The eraser mode allows deleting any registered annotation for a voxel. Finally, the click mode allows the user to click over the volume without specifying any annotations. This was introduced to reduce mistakes stemming from accidental clicks. 
1. **Brush size**: A variable brush size allows the user to define the area affected by each click. The brush has a square shape and the brush size is defined as the side length of this square minus one, times one-half. 
1. **Saving and resuming**: The user can pause and save the annotations at any moment. It is also possible to resume annotating from a previously saved state. 
1. **Superimposing previous result**: In case the annotator wishes to go through an iterative process of refinement of annotations, it is possible to overlay the result of a previous segmentation to visualize problematic regions which might need more attention. 

## Annotator GUI

The annotator graphical user interface (GUI) has the following layout.
![Example screen](/gc_segment/objects.png)

The user is shown a part of one plane of the volume in the viewing pane. The part shown can be determined using three things---
1. the **axis along which the view** is shown (`X`, `Y`, or `Z`); 
1. the **axes and coordinates** appearing in the viewing pane; and
1. the value in the corresponding **plane slider**. 

The user can left-click on pixels (which correspond to voxels in the 3D volume for the displayed plane) in the viewing pane to mark pixels. The user can also move around the displayed plane by right-clicking and dragging. Additionally to scroll through the volume, there are `Z`, `Y`, and `X` sliders provided. The viewing axis can be selected using three radio buttons on the top right. Once seeding is complete, the `Save` button on the bottom is used to save all of them to disk. Additionally, the current slice can be saved to disk using `Save Slice`. Seeds can be removed using the `Reset...` buttons. 

The user can scroll through the volume by using the mouse wheel. This will move the shown image along the chosen viewing axis. However, if the user hovers the mouse pointer over the `Brush size` slider, the mouse wheel will move this slider instead. Optionally, the user can click anywhere on the slider to directly choose a value of their liking.

Furthermore,
* "What to mark?" is a radio button which determines what effect a click has on the seeding.
  - `Object` marks a pixel as object;
  - `Background` marks a pixel as background;
  - `Erase` erases the mark at a pixel; and
  - `Click` has no effect at all.
* "Annotation/Segmentation mode" is something that can be used to refine previously made segmentations. In particular, if the tool finds a previously stored segmentation, it can be overlaid onto the current volume instead of the markings to examine the segmentation visually (see example figure below). *NOTE*: The "Segmentation" mode does **not** generate the segmentation, but simply overlays it on the volume. To generate the segmentation from the seeds themselves, please run `segment.py` as indicated above in [General Usage](#generalusage).


## <a name="configs"></a> Config files
Configuration files in `configs/` let the user determine the parameters of the annotation task. Below is an example configuration file with the meanings of the parameters---
![Example config file](/gc_segment/config.png)
Some points to note:
* The config file expects the volume to be saved as a set of images, each image showing an xy-plane for one particular z-coordinate. It further expects the names of these files to reflect the order of z-coordinates when the names are sorted with the python function `sorted()`. For example these file names can be `slice0000.png`, `slice0001.png`, ..., and so on.
* `coords` specifies a bounding box that will be extracted out of each image and stacked with other bounding boxes to construct the volume. The format of `coords` follows the standard format for bounding boxes in computer vision---it is a list of four values, with the first two values being the `x`- and `y`- coordinates of the top-left corner, and the next two being the width and the height of the box.
* `z_limits` specifies which `z`-slices to use. The value of `[100, 200]` here says the tool with use 100 images starting at slice number 101 and ending at slice number 200, after the `sorted()` function has been applied. The program assumes that the data will have no missing slices.
* `n_superpixels` specifies the target number of supervoxels after SLIC. The resulting number might not always be exactly this, but it is expected to be around this value.
* `n_jobs` specifies over how many parallel threads the computation is to be split. Splitting over multiple threads results in faster computation, but only on workstations that have multiple processing units.
* `use_gmm` and `n_gmm_components` refer to an optional post-processing step that uses a Gaussian mixture model to further remove outliers after maxflow.
* `delta`, `lambda`, etc. specify the energy function hyperparameters.
* Optionally, one can also specify `max_area_display_size` (200 pixels by default), which defines the maximum size of the rectangle that is displayed in the viewing area.

## Example 

The figure below shows an example of a raw volume, of seeds, and of the resulting segmentation overlaid on the volume with the "Segmentation" mode. Pixels marked objects appear in blue, while those marked background appear in red.
![Example steps](/gc_segment/modes.png)

## Best practices

We observed that certain practices helped in overall better ground truth, and we recommend the following practices for annotation.
1. **Tightly cropped ROIs**: To reduce the computation effort, ROIs should be as tightly cropped around the target organelle as possible. This reduces the inclusion of any background pixels not part of the organelle and hence reduces the size of the problem to be solved.
1. **No false negatives**: It is important not to have any parts of an organelle incorrectly annotated when the ground truth is used as training data for the deep learning pipeline. We observed that it is hence desirable to have tightly cropped ROIs when annotating an organelle and to ensure that there are no false negatives appearing in the ROI. 
1. **Large brush size for background**: As annotating background voxels does not have to be as rigorous as annotating organelle voxels, one can generally use a larger brush size to cover large portions of the background in a single stroke. 
1. **Annotating potentially confusing pixels**: The raw data are only grayscale, and hence, organelles, which are not the target of the annotation, might exhibit similar grayscale values. For such cases, we observed that it is essential to explicitly label such organelles as background.
1. **Resolving conflicts close to the boundary**: As an example, mitochondria and endoplasmic reticulum (ER) exhibit similar gray scale values. There exist several contact points between mitochondria and ER. At such points, it becomes particularly important to annotate the boundary between them as correctly as possible.  
