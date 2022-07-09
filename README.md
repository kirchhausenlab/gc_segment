# gc_segment
Annotation-based organelle segmentation using graph cuts

## Getting started

1. Install some version of anaconda, e.g. [miniconda](https://docs.conda.io/en/latest/miniconda.html).
1. Open the terminal.
1. Create a new conda environment: `conda create -n gc_segment python` and activate it: `conda activate gc_segment`.
1. Install the boost library: `conda install -c conda-forge boost`.
1. Pip install this repository: `pip install git+https://github.com/kirchhausenlab/gc_segment.git`.
1. Run "VolumeAnnotator".

## Available features

1. **Configuration files**: As each annotation task had associated options (ROI, number of supervoxels, etc.), configuration files are associated with each annotation to define GUI and segmentation options in a compact fashion.
1. **Axis views and sliders**: The tool offers the choice to view the volume along any axis. It further allows easy scrolling with a slider through planes perpendicular to this axis to visualize the volume.
1. **Pan mode**: To handle large volumes, the tool offers an option to pan through the volume. It was observed that if the volume size is large, it becomes more difficult for the annotator to maintain precision while annotating. To circumvent this problem, the tool is constrained to visualize a fixed size defined by number of voxels, with the added option of being able to pan through the entire larger volume.
1. **Annotation modes**: There are four annotation modes offered: organelle, background, eraser, and click. When in the organelle or background modes, any annotations made are automatically registered as ground truth for the appropriate class. The eraser mode allows deleting any registered annotation for a voxel. Finally, the click mode allows the user to click over the volume without specifying any annotations. This was introduced to reduce mistakes stemming from accidental clicks. 
1. **Brush size**: A variable brush size allows the user to define the area affected by each click. The brush has a square shape and the brush size is defined as the side length of this square minus one, times one-half. 
1. **Saving and resuming**: The user can pause and save the annotations at any moment. It is also possible to resume annotating from a previously saved state. 
1. **Superimposing previous result**: In case the annotator wishes to go through an iterative process of refinement of annotations, it is possible to overlay the result of a previous segmentation to visualize problematic regions which might need more attention. 

## Best practices

We observed that certain practices helped in overall better ground truth, and we recommend the following practices for annotation.
1. **Tightly cropped ROIs**: To reduce the computation effort, ROIs should be as tightly cropped around the target organelle as possible. This reduces the inclusion of any background pixels not part of the organelle and hence reduces the size of the problem to be solved.
1. **No false negatives**: It is important not to have any parts of an organelle incorrectly annotated when the ground truth is used as training data for the deep learning pipeline. We observed that it is hence desirable to have tightly cropped ROIs when annotating an organelle and to ensure that there are no false negatives appearing in the ROI. 
1. **Large brush size for background**: As annotating background voxels does not have to be as rigorous as annotating organelle voxels, one can generally use a larger brush size to cover large portions of the background in a single stroke. 
1. **Annotating potentially confusing pixels**: The raw data are only grayscale, and hence, organelles, which are not the target of the annotation, might exhibit similar grayscale values. For such cases, we observed that it is essential to explicitly label such organelles as background.
1. **Resolving conflicts close to the boundary**: As an example, mitochondria and endoplasmic reticulum (ER) exhibit similar gray scale values. There exist several contact points between mitochondria and ER. At such points, it becomes particularly important to annotate the boundary between them as correctly as possible.  
