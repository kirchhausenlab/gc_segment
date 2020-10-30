import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import imageio
import sys
import os
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from matplotlib.backend_bases import MouseButton
import time
import argparse
from utils import *


IMAGE_RECTANGLE = [0.03, 0.10, 0.60, 0.70]

KEYPRESS_MOVE_STEP  = 5

FOREGROUND = 'Object'
BACKGROUND = 'Background'
ERASE = 'Erase'
CLICK = 'Click'
ANNOTATION = 'Annotation'
SEGMENTATION = 'Segmentation'


TICK_STEP = 20


# Axes. 
X, Y, Z = 'X', 'Y', 'Z'

ANNO_CHOICES = [FOREGROUND, BACKGROUND, ERASE, CLICK]
ANNO_KEYPRESS_DICT = {
    'q':    [FOREGROUND, 0],
    'w':    [BACKGROUND, 1],
    'e':    [ERASE, 2],
    'r':    [CLICK, 3],
}


ANNO_SEG_KEYPRESS_DICT = {
    'a' :   [ANNOTATION, 0],
    's' :   [SEGMENTATION, 1],
}


AXIS_CHOICES = [X, Y, Z]
AXIS_KEYPRESS_DICT = {
    '3':    [Z, 2],
    '2':    [Y, 1],
    '1':    [X, 0],
}
PLANES_KEYPRESS_DICT = {
    'z':   1,
    'm': -1,
}
MOVE_KEYPRESS_DICT = {
    'up':       [ 0, -KEYPRESS_MOVE_STEP],
    'down':     [ 0,  KEYPRESS_MOVE_STEP],
    'left':     [-KEYPRESS_MOVE_STEP,  0],
    'right':    [ KEYPRESS_MOVE_STEP,  0],
}

RECOGNISED_KEYBOARD_SHORTCUTS = list(ANNO_KEYPRESS_DICT.keys()) + \
    list(PLANES_KEYPRESS_DICT.keys())  + \
    list(AXIS_KEYPRESS_DICT)           + \
    list(MOVE_KEYPRESS_DICT)            # + <other keyboard shorcuts>

FOREGROUND_NAMES = [FOREGROUND, 'Foreground']
ANNO_DIR = 'annotation/'
OUTPUT_DIR = 'segmentation_cropped/'
BLK = {
    FOREGROUND: 1,
    BACKGROUND: 10,
}

BOX_COORDS = {
    '/media/scratch/mihir/20190613/':   [1886, 1168, 458, 392],
    './data/20190613/':   [1886, 1168, 458, 392],
    '../data/20190613/':   [1886, 1168, 458, 392],
    './data/20190613_full/':   [1886, 1168, 458, 392],
    './data/20190821/':   [3768, 1829, 448, 294],
    '../data/20190821/':   [3768, 1829, 448, 294],
    '../data/20190919/':   [2160, 1018, 728, 252],
}


def bound_low(value, bound):
    """
    Lower-bound a value according to given bound. 
    """
    return max([value, bound])


def bound_high(value, bound):
    """
    Upper-bound a value according to a given bound.
    """
    return min([value, bound])


def slice_file_name(z_, label_):
    """ 
    Define a file name for slice number z_ for the foreground and background.
    """
    assert label_ in [FOREGROUND, BACKGROUND], 'slice_file_name: Supplied label must be \
                                        either {} or {}.'.format(FOREGROUND, BACKGROUND)

    return '%s_%d.tif' % (label_, z_)


def info_from_filename(filename):
    """
    Get information (z, label) from the filename.
    """

    z = int(os.path.split(filename)[-1].split('.')[0].split('_')[1])
    if any([f in filename for f in FOREGROUND_NAMES]):
        label = FOREGROUND
    else:
        label = BACKGROUND
    return z, label


class VolumeAnnotator(object):
    def __init__(
        self,
        cfg,
        **kwargs,
    ):
        """
        Init function for ForegroundBackground annotator. 

        Inputs
        ------
            data_path           Path to directory containing TIFF stack. 
            anno_dir            Directory in which to save annotations. 
                                Default: 'annotation/'
        """

        options = load_yaml(cfg)

        # Get experiment name
        self.experiment_name = '.'.join(os.path.split(cfg)[-1].split('.')[:-1])
        self.output_dir = os.path.join('output/', self.experiment_name)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.data_path = options.data_path
        assert os.path.exists(
            self.data_path), 'Specified directory {} does not exist!'.format(self.data_path)

        anno_dir = os.path.join(self.output_dir, 'annotation/')
        data_dir = os.path.join(self.output_dir, 'data/')

        self.anno_dir = anno_dir
        self.data_dir = data_dir

        # Make directories if they do not exist.
        for dir_ in [self.anno_dir, self.data_dir]:
            if not os.path.exists(dir_):
                os.makedirs(dir_)

#        while os.path.exists(anno_dir):
#            response                    = 'p'
#            while response not in ['', 'y', 'Y', 'n', 'N']:
#                response                = input('{} already exists. Annotations in this directory \
# will be overwritten. Continue (Y/n)? '.format(anno_dir))
#
#            _                           = os.popen('rm -rf {}'.format(anno_dir)).read()
#
#        self.anno_dir                   = anno_dir
#        os.makedirs(anno_dir)

        # Default kwargs.
        default_kwargs_ = {
            # Size of the figure
            'figure_size_':   [18, 10],

            # Colour map to display image.
            'im_cmap_':   'hsv',

            # Colour of the sliders and boxes.
            'face_colour_':   [0.85, 0.85, 0.85],
            'ax_colour_':   'lightgoldenrodyellow',
            'hover_colour_':   '0.8',

            # Margin left.
            'm_left_':   0.05,
            # Margin right.
            'm_right_':  0.95,
            # Margin top.
            'm_top_':    0.95,
            # Margin bottom.
            'm_bottom_': 0.05,

            # Maximum display size of the viewing area
            'max_area_display_size':    200,
        }
        # Set kwargs.
        for k in default_kwargs_:
            if hasattr(kwargs, k):
                setattr(self, k, kwargs[k])
            else:
                setattr(self, k, default_kwargs_[k])

        # Check that the data path and image exist.
        assert os.path.exists(self.data_path), \
            'Given data path {} does not exist!'.format(self.data_path)

        # Get box coordinates.
        if options.coords:
            self.X, self.Y, self.W, self.H = options.coords
        else:
            # BOX_COORDS[self.data_path]
            self.X, self.Y, self.W, self.H = None, None, None, None

        # Get image names.
        self.image_names = sorted(os.listdir(self.data_path))

        # Z slicing
        if not options.z_limits:
            self.z_limits = [0, len(self.image_names)]
        else:
            self.z_limits = options.z_limits

        # Number of slices
        self.n_slices = self.z_limits[1] - self.z_limits[0]

        # Read TIFF stack.
        self.stack = []
        for _z in range(self.z_limits[0], self.z_limits[1]):
            _f = self.image_names[_z]
            _img_path = os.path.join(self.data_path, _f)
            # Read slice.
            _slice = imageio.imread(_img_path)
            # Crop the slice if limits are specified.
            if self.X is not None:
                _slice = _slice[self.Y: self.Y + self.H,
                                self.X: self.X + self.W]

            # Save into data dir
            imageio.imsave(os.path.join(self.data_dir, _f), _slice)

            # Push into stack
            self.stack.append(_slice)

        # Concatenate into one volume.
        self.stack = np.stack(self.stack)
        # Image sizes.
        self.stack_depth = self.stack.shape[0]
        self.stack_height = self.stack.shape[1]
        self.stack_width = self.stack.shape[2]
        # Initialise empty annotation volumes.
        self.fg_annotation_ = np.zeros(self.stack.shape).astype(np.uint8)
        self.bg_annotation_ = np.zeros(self.stack.shape).astype(np.uint8)
        # Copy the stack into another image---this is the image that is displayed.
        self.disp_stack_ = self.stack.copy()
        self.disp_stack_ = np.stack(
            (self.disp_stack_, self.disp_stack_, self.disp_stack_)).transpose([1, 2, 3, 0])


        # Initialise the viewing rectangle. This is a new addition (2020-10-19)
        #   to help focus on smaller regions during annotation of big volumes. 
        #
        # In essence, this restricts the viewing rectangle to a maximum 
        #   size determined by self.max_area_display_size
        self.x_view_size    = min(self.max_area_display_size, self.W)
        self.y_view_size    = min(self.max_area_display_size, self.H)
        self.z_view_size    = min(self.max_area_display_size, self.n_slices)
        # Keep one tuple for each viewing axis. This tuple defines the top-left 
        #   corner of the viewing volume depending on the axis. The depth of 
        #   this volume is all planes. 
        self.viewing_rect   = {}
        for axis_ in AXIS_CHOICES:
            self.viewing_rect[axis_] = [0, 0]
        # When we use the move function to move around in the volume, these
        #    values will be changed. 
    
        # Define viewing rectangle sizes depending on the viewing axis. 
        self.view_sizes     = {}
        self.view_sizes[X]  = [self.y_view_size, self.z_view_size]
        self.view_sizes[Y]  = [self.x_view_size, self.z_view_size]
        self.view_sizes[Z]  = [self.x_view_size, self.y_view_size]

        # Define stack sizes according to viewing axis. 
        self.stack_sizes    = {}
        self.stack_sizes[X] = [self.stack_height, self.stack_depth]
        self.stack_sizes[Y] = [self.stack_width, self.stack_depth]
        self.stack_sizes[Z] = [self.stack_width, self.stack_height]


        # Dictionary to determine which annotation volume to contribute to.
        self.annotation_volume_dict = {
            FOREGROUND:   self.fg_annotation_,
            BACKGROUND:   self.bg_annotation_,
       }
        # Which channels to modify in disp_stack_ for FG and BG.
        self.fg_channel = 2         # Blue
        self.bg_channel = 0         # Red

        # View axis is Z by default. 
        self.view_axis_     = Z
        # Currently shown slice on the figure. Initialise this to zero for all axes. 
        self.x_ = 0
        self.y_ = 0
        self.z_ = 0

        # Currently chosen annotation mode. It is foreground by default.
        self.ann_mode_ = FOREGROUND

        # Create a dictionary to remember brush sizes. This allows for easy switching
        #   switching between annotation modes, as brush sizes are remembered.
        # Typically, objects use a smaller brush size.
        self.brush_sizes_ = {}
        for ann_mode_ in ANNO_CHOICES:
            self.brush_sizes_[ann_mode_] = 1
        # Currently chosen brush size. Initially is 1.
        self.brush_size_ = 1

        # Whether mouse is currently pressed.
        self.mouse_primary_pressed_ = False
        self.mouse_secondary_pressed_ = False

        # Set self.initialised_ to False, meaning the figure has not been initialised yet.
        self.initialised_ = False

        # Array determining whether there are annotations for a certain plane.
        self.annotated_planes_ = np.zeros(self.n_slices, dtype=bool)

        # If self.anno_dir already exists, it contains a previously saved annotation.
        # Load it.
        if os.path.exists(self.anno_dir):
            # Path to load from is in text.
            saved_files = os.listdir(self.anno_dir)
            # Load all files. The filename determines which annotation slice to set.
            for sf in saved_files:
                sf_path = os.path.join(self.anno_dir, sf)
                anno = imageio.imread(sf_path) / 255
                z, label = info_from_filename(sf_path)
                annotated_flag_ = anno.sum() > 0
                if label in FOREGROUND_NAMES:
                    self.fg_annotation_[z, :, :] = anno
                    ds_ = (1 - anno) * \
                        self.disp_stack_[z, :, :, self.fg_channel]
                    fg_ = anno * 255
                    self.disp_stack_[z, :, :, self.fg_channel] = ds_ + fg_
                else:
                    self.bg_annotation_[z, :, :] = anno
                    ds_ = (1 - anno) * \
                        self.disp_stack_[z, :, :, self.bg_channel]
                    bg_ = anno * 255
                    self.disp_stack_[z, :, :, self.bg_channel] = ds_ + bg_
                # Mark this plane as annotated.
                self.annotated_planes_[z] = annotated_flag_

        else:
            os.makedirs(self.anno_dir)

    # Update current figure.

    def _update_figure(self, axis_switch=False):
        """
        Redraws the figure for the current value of z_.
        axis_switch: whether viewing axis has been switched to make this call. 
        """
        if self.view_axis_ == X:
            # Determine the viewing rectangle. 
            y_min, z_min = self.viewing_rect[self.view_axis_]
            y_max        = y_min + self.y_view_size
            z_max        = z_min + self.z_view_size

            display_stack_ = self.disp_stack_[z_min:z_max, y_min:y_max, self.x_, :]

        elif self.view_axis_ == Y:
            # Determine the viewing rectangle. 
            x_min, z_min = self.viewing_rect[self.view_axis_]
            x_max        = x_min + self.x_view_size
            z_max        = z_min + self.z_view_size

            display_stack_ = self.disp_stack_[z_min:z_max, self.y_, x_min:x_max, :]

        elif self.view_axis_ == Z:
            # Determine the viewing rectangle. 
            x_min, y_min = self.viewing_rect[self.view_axis_]
            x_max        = x_min + self.x_view_size
            y_max        = y_min + self.y_view_size

            display_stack_ = self.disp_stack_[self.z_, y_min:y_max, x_min:x_max, :]



        if axis_switch:
            if self.ax_image_ is None: 
                self.ax_image_ = plt.axes(IMAGE_RECTANGLE)            # Was 0.05, 0.15, 0.55, 0.8
            else:
                self.ax_image_ = plt.axes(self.ax_image_)

            self.ax_image_.margins(x=0, y=0, tight=True)
            self.image_handle_ = self.ax_image_.imshow(display_stack_,
                                                   cmap=self.im_cmap_,
                                                   aspect='equal',)
        else:
            self.image_handle_.set_data(display_stack_)


        self._set_axis_ticks()
        self._set_axis_labels()
        self.figure_.canvas.draw_idle()
        return

    def _update_annotation_visual(self):
        """
        Updates annotation in self.disp_stack_ for current z.
        """
        self.disp_stack_[self.z_, :, :,
                         self.fg_channel][self.fg_annotation_[self.z_] > 0] = 255
        self.disp_stack_[self.z_, :, :,
                         self.bg_channel][self.bg_annotation_[self.z_] > 0] = 255
        return

    # Handle mouse press.

    def _handle_mouse_press(self,
                            event):
        """
        Function to handle mouse press for annotation. 
        Mouse press is registered for annotation only if it was pressed
        while over the plot showing the image. 
        """

        # First, make sure the figure has been initialised.
        assert self.initialised_, 'Figure has not been initialised yet!'

        # Check whether event occurs over ax_image_
        if event.inaxes == self.ax_image_:
            # Check if the primary or secondary button is pressed. 
            if event.button == MouseButton.LEFT:
                # This button press shall be used for annotation. 
                self.mouse_primary_pressed_ = True
                self.annotate_patch(event.xdata, event.ydata)
            # Check if secondary was pressed instead. 
            elif event.button == MouseButton.RIGHT:
                self.mouse_secondary_pressed_ = True
                # Save the location where this button was pressed. 
                # This will be used to determine how to move
                #   the image. 
                self.move_image_location_ = [int(np.floor(event.xdata)), 
                                           int(np.floor(event.ydata))]
                # Save the viewing rectangle at this point. 
                self.move_viewing_rect_ = self.viewing_rect[self.view_axis_]

        return

    def _handle_mouse_move(self,
                           event):
        """
        Handle how annotations occur when mouse is moved. 
        Annotations are recorded only if the button is pressed while the mouse is moving. 
        """

        # First, make sure the figure has been initialised.
        assert self.initialised_, 'Figure has not been initialised yet!'

        if (not self.mouse_primary_pressed_ and not self.mouse_secondary_pressed_) \
                or event.inaxes != self.ax_image_:
            return

        # If the primary button is pressed, annotate. 
        if self.mouse_primary_pressed_:
            # Do nothing if ann_mode_ is CLICK.
            if self.ann_mode_ == CLICK:
                return
    
            self.annotate_patch(event.xdata, event.ydata)
        # If the secondary button is pressed, move
        elif self.mouse_secondary_pressed_:
            # Call the move function 
            self.move_image(location=[event.xdata, event.ydata], displacement=None)

        return

    def _handle_mouse_release(self,
                              event):
        """
        Handle mouse release.
        """

        # First, make sure the figure has been initialised.
        assert self.initialised_, 'Figure has not been initialised yet!'

        # Check which mouse button was pressed. 
        if self.mouse_primary_pressed_:
            # The button had been pressed to annotate. 
            self.mouse_primary_pressed_ = False
            # Mark this plane as annotated if it still contains any annotations.
    
            if self.fg_annotation_[self.z_].sum() > 0 or self.bg_annotation_[self.z_].sum() > 0:
                self.annotated_planes_[self.z_] = True
            else:
                self.annotated_planes_[self.z_] = False
        elif self.mouse_secondary_pressed_:
            self.mouse_secondary_pressed_ = False
            # Just call _update_figure() one last time. 
            self._update_figure()

        return

    def annotate_patch(self, xloc, yloc):
        """
        Annotate patch centered at (xloc, yloc). 
        Uses current state to determine what kind of annotation will be done. 
        """

        # TODO: The annotation is dependent on the viewing rectangle. 

        # Find where the event occurred.
        loc_x_, loc_y_ = xloc, yloc
        loc_x_ = int(np.floor(loc_x_))
        loc_y_ = int(np.floor(loc_y_))

        # Get the viewing window. 
        ref_win_x, ref_win_y = self.viewing_rect[self.view_axis_]

        # Find coordinates of box to annotate.
        bs2_ = (self.brush_size_ + 1) // 2

        col_s_ = max([ref_win_x + loc_x_ - bs2_ + 1, 0])
        col_e_ = min([ref_win_x + loc_x_ + bs2_, self.stack_width])

        row_s_ = max([ref_win_y + loc_y_ - bs2_ + 1, 0])
        row_e_ = min([ref_win_y + loc_y_ + bs2_, self.stack_height])

        # Record annotation. This is dependent on what is the current viewing axis. 

        # If viewing axis is X, we annotate in the Y-Z plane. 
        if self.view_axis_ == X:
            if self.ann_mode_ == FOREGROUND:
                self.fg_annotation_[row_s_:row_e_, col_s_:col_e_, self.x_] = 1
                self.bg_annotation_[row_s_:row_e_, col_s_:col_e_, self.x_] = 0
                self.disp_stack_[row_s_:row_e_, col_s_:col_e_, 
                                 self.x_, self.fg_channel] = 255
                self.disp_stack_[row_s_:row_e_, col_s_:col_e_, 
                                 self.x_, self.bg_channel] = \
                    self.stack[row_s_:row_e_, col_s_:col_e_, self.x_]
            elif self.ann_mode_ == BACKGROUND:
                self.bg_annotation_[row_s_:row_e_, col_s_:col_e_, self.x_] = 1
                self.fg_annotation_[row_s_:row_e_, col_s_:col_e_, self.x_] = 0
                self.disp_stack_[row_s_:row_e_, col_s_:col_e_, 
                                 self.x_, self.bg_channel] = 255
                self.disp_stack_[row_s_:row_e_, col_s_:col_e_, 
                                 self.x_, self.fg_channel] = \
                    self.stack[row_s_:row_e_, col_s_:col_e_, self.x_]
            elif self.ann_mode_ == ERASE:
                self.bg_annotation_[row_s_:row_e_, col_s_:col_e_, self.x_] = 0
                self.fg_annotation_[row_s_:row_e_, col_s_:col_e_, self.x_] = 0
                self.disp_stack_[row_s_:row_e_, col_s_:col_e_, 
                                 self.x_, self.bg_channel] = \
                    self.stack[row_s_:row_e_, col_s_:col_e_, self.x_]
                self.disp_stack_[row_s_:row_e_, col_s_:col_e_, 
                                 self.x_, self.fg_channel] = \
                    self.stack[row_s_:row_e_, col_s_:col_e_, self.x_]

        # If viewing axis is Y, we annotate in the X-Z plane. 
        elif self.view_axis_ == Y:
            if self.ann_mode_ == FOREGROUND:
                self.fg_annotation_[row_s_:row_e_, self.y_, col_s_:col_e_] = 1
                self.bg_annotation_[row_s_:row_e_, self.y_, col_s_:col_e_] = 0
                self.disp_stack_[row_s_:row_e_, self.y_,
                                 col_s_:col_e_, self.fg_channel] = 255
                self.disp_stack_[row_s_:row_e_, self.y_, 
                                 col_s_:col_e_, self.bg_channel] = \
                    self.stack[row_s_:row_e_, self.y_, col_s_:col_e_]
            elif self.ann_mode_ == BACKGROUND:
                self.bg_annotation_[row_s_:row_e_, self.y_, col_s_:col_e_] = 1
                self.fg_annotation_[row_s_:row_e_, self.y_, col_s_:col_e_] = 0
                self.disp_stack_[row_s_:row_e_, self.y_,
                                 col_s_:col_e_, self.bg_channel] = 255
                self.disp_stack_[row_s_:row_e_, self.y_, col_s_:col_e_, self.fg_channel] = \
                    self.stack[row_s_:row_e_, self.y_, col_s_:col_e_]
            elif self.ann_mode_ == ERASE:
                self.bg_annotation_[row_s_:row_e_, self.y_, col_s_:col_e_] = 0
                self.fg_annotation_[row_s_:row_e_, self.y_, col_s_:col_e_] = 0
                self.disp_stack_[row_s_:row_e_, self.y_, col_s_:col_e_, self.bg_channel] = \
                    self.stack[row_s_:row_e_, self.y_, col_s_:col_e_]
                self.disp_stack_[row_s_:row_e_, self.y_, col_s_:col_e_, self.fg_channel] = \
                    self.stack[row_s_:row_e_, self.y_, col_s_:col_e_]


        # If viewing axis is Z, we annotate in the X-Y plane. 
        elif self.view_axis_ == Z:
            if self.ann_mode_ == FOREGROUND:
                self.fg_annotation_[self.z_, row_s_:row_e_, col_s_:col_e_] = 1
                self.bg_annotation_[self.z_, row_s_:row_e_, col_s_:col_e_] = 0
                self.disp_stack_[self.z_, row_s_:row_e_,
                                 col_s_:col_e_, self.fg_channel] = 255
                self.disp_stack_[self.z_, row_s_:row_e_, col_s_:col_e_, self.bg_channel] = \
                    self.stack[self.z_, row_s_:row_e_, col_s_:col_e_]
            elif self.ann_mode_ == BACKGROUND:
                self.bg_annotation_[self.z_, row_s_:row_e_, col_s_:col_e_] = 1
                self.fg_annotation_[self.z_, row_s_:row_e_, col_s_:col_e_] = 0
                self.disp_stack_[self.z_, row_s_:row_e_,
                                 col_s_:col_e_, self.bg_channel] = 255
                self.disp_stack_[self.z_, row_s_:row_e_, col_s_:col_e_, self.fg_channel] = \
                    self.stack[self.z_, row_s_:row_e_, col_s_:col_e_]
            elif self.ann_mode_ == ERASE:
                self.bg_annotation_[self.z_, row_s_:row_e_, col_s_:col_e_] = 0
                self.fg_annotation_[self.z_, row_s_:row_e_, col_s_:col_e_] = 0
                self.disp_stack_[self.z_, row_s_:row_e_, col_s_:col_e_, self.bg_channel] = \
                    self.stack[self.z_, row_s_:row_e_, col_s_:col_e_]
                self.disp_stack_[self.z_, row_s_:row_e_, col_s_:col_e_, self.fg_channel] = \
                    self.stack[self.z_, row_s_:row_e_, col_s_:col_e_]

        # Update display.
        self._update_figure()

    def move_image(self, 
                   location=None, displacement=None):
        """
        Move the viewing area depending on xloc and yloc. 

        location is used when moving with the mouse. It should be a list/tuple 
        [xloc, yloc]. 

        displacement is used when moving with the arrow keys. 
        It should be a tuple [xdisp, ydisp]
        """

        # First, make sure the figure has been initialised.
        assert self.initialised_, 'Figure has not been initialised yet!'

        if location is not None:
            xloc, yloc  = location

            # Convert xloc and yloc to integers
            xloc    = int(np.floor(xloc))
            yloc    = int(np.floor(yloc))
            # Get the original click location for the move. 
            ref_move_x, ref_move_y = self.move_image_location_
    
            # The amount to move along X or Y is now dependent on where the current
            #   location of the mouse is relative to this. 
            # After the image has been moved, the original location must 
            #   occur at the current location. That is to say, the mouse
            #   should "stick to" the original location. 
            #
            # To achieve this, we must move the coordinates of the viewing 
            #   rectangle. 
    
            # Get the current viewing rectangle. 
            view_rect_x, view_rect_y = self.move_viewing_rect_
            # Get the difference between xloc and ref_move_x, and similarly for y. 
            move_x      = ref_move_x - xloc
            move_y      = ref_move_y - yloc

            # Store xloc, yloc as the new move_location_
            self.move_location_ = [xloc, yloc]
 
        elif displacement is not None:
            view_rect_x, view_rect_y    = self.viewing_rect[self.view_axis_]
            move_x, move_y = displacement
            move_x      = int(np.floor(move_x))
            move_y      = int(np.floor(move_y))

        else:
            raise ValueError('One of location and displacement must be specified when calling move_image!')

        # The new location is the old location plus this difference
        new_loc_x   = view_rect_x + move_x
        new_loc_y   = view_rect_y + move_y

        # New locations are view_rect_x + move_x and view_rect_y + move_y. 
        # But these must be kept within bounds [0, self.self.<axis>_view_size]
        
        # Lower bound it first. 
        new_loc_x      = bound_low(new_loc_x, 0)
        new_loc_y      = bound_low(new_loc_y, 0)

        # Upper bound it now, depending on the axis. 
        x_view_size, y_view_size = self.view_sizes[self.view_axis_]
        x_size, y_size           = self.stack_sizes[self.view_axis_]
        new_loc_x  = bound_high(new_loc_x, x_size - x_view_size)
        new_loc_y  = bound_high(new_loc_y, y_size - y_view_size)

        # Store the new location
        self.viewing_rect[self.view_axis_] = [new_loc_x, new_loc_y]
            
       
        # Update the figure
        self._update_figure()
        return


    def _handle_ann_mode_radio(self,
                               label):
        """
        Handle choice of annotation mode---foreground or background.
        """

        # First, make sure the figure has been initialised.
        assert self.initialised_, 'Figure has not been initialised yet!'

        # If the chosen mode is the same as the current mode, nothing to do.
        if self.ann_mode_ == label:
            return

        # Update the brush size for the current ann_mode_ first.
        self.brush_sizes_[self.ann_mode_] = self.brush_size_
        # Switch ann mode
        self.ann_mode_ = label
        # Retrieve the saved brush size.
        self.brush_size_ = self.brush_sizes_[self.ann_mode_]
        # Set the brush size slider appropriately
        self.brush_size_slider_.set_val(self.brush_size_)

        return

    def _handle_axis_mode_radio(self, 
                                label):
        """
        Handle choice of viewing axis. 
        """

        # First make sure the figure is initialised. 
        assert self.initialised_, 'Figure has not been initialised yet!'

        # If the chosen mode is the same as the current mode, there is nothing to do. 
        if self.view_axis_ == label:
            return 

        # Else, switch the viewing axis and update the figure. 
        self.view_axis_ = label
        # axis_switch is set to True here because we must readjust for axes. 
        self._update_figure(axis_switch=True)
        return 

    def _handle_scroll_event(self,
                             event):
        """
        Move along the viewing axis, up and down depending on whether 
        the wheel is scrolled up or down. 
        """

        # First, make sure the figure is initialised. 
        assert self.initialised_, 'Figure has not been initialised yet!'


        # If the scroll happens on the brush size slider
        if event.inaxes == self.ax_brush_size_:
            if event.button == 'up':
                self.brush_size_ = bound_low(self.brush_size_ - 2, 1)
            elif event.button == 'down':
                self.brush_size_ = bound_high(self.brush_size_ + 2, self.max_brush_size)
            else:
                return 

            self.brush_size_slider_.set_val(self.brush_size_)
            return

        # Else scroll along planes. 
        if self.view_axis_ == X:
            if event.button == 'up':
                if self.x_ > 0:
                    self.x_ -= 1
            elif event.button == 'down':
                if self.x_ < self.W - 1:
                    self.x_ += 1
            else:
                return

            # Update the x slider as well. A '+ 1' is needed because on the slider, the
            #   slide numbers start from 0.
            self.x_slider_.set_val(self.x_ + 1)


        if self.view_axis_ == Y:
            if event.button == 'up':
                if self.y_ > 0:
                    self.y_ -= 1
            elif event.button == 'down':
                if self.y_ < self.H - 1: 
                    self.y_ += 1
            else:
                return

            # Update the y slider as well. A '+ 1' is needed because on the slider, the
            #   slide numbers start from 0.
            self.y_slider_.set_val(self.y_ + 1)    


        if self.view_axis_ == Z:
            if event.button == 'up':
                if self.z_ > 0:
                    self.z_ -= 1
            elif event.button == 'down':
                if self.z_ < self.z_limits[1] - self.z_limits[0] - 1:
                    self.z_ += 1
            else:
                return

            # Update the z slider as well. A '+ 1' is needed because on the slider, the
            #   slide numbers start from 0.
            self.z_slider_.set_val(self.z_ + 1)

        # Update figure to reflect the new image.
        self._update_figure()
        return

    # Handle keyboard shortcuts.
    def _handle_keyboard_shortcuts(self,
                                   event):
        """
        Handle keyboard shortcuts: Keyboard shortcuts let you switch 
        easily between annotation modes and other functions. 
        Currently, annotation modes are handled. 
        In the future, brush size is planned to be included.
        """

        # First, make sure the figure has been initialised.
        assert self.initialised_, 'Figure has not been initialised yet!'

        if event.key not in RECOGNISED_KEYBOARD_SHORTCUTS:
            return

        if event.key in ANNO_KEYPRESS_DICT:
            return self._handle_anno_keyboard_shorcuts(event.key)
        if event.key in PLANES_KEYPRESS_DICT:
            return self._handle_planes_keyboard_shortcuts(event.key)
        if event.key in AXIS_KEYPRESS_DICT:
            return self._handle_axis_keyboard_shortcuts(event.key)
        if event.key in MOVE_KEYPRESS_DICT:
            return self._handle_move_keyboard_shortcuts(event.key)

        # Can add other functions here.
        return

    def _handle_anno_keyboard_shorcuts(self,
                                       key):
        """
        Handle keyboard shortcuts for annotation modes. 
        """

        # First, make sure the figure has been initialised.
        assert self.initialised_, 'Figure has not been initialised yet!'

        label, index = ANNO_KEYPRESS_DICT[key]
        self.ann_mode_ = label
        self.ann_mode_radio_.set_active(index)

        return

    def _handle_axis_keyboard_shortcuts(self, 
                                        key):
        """
        Handle keyboard shortcuts for viewing axis modes. 
        """

        # First, make sure the figure is initialised.
        assert self.initialised_, 'Figure has not been initialised yet!'

        label, index = AXIS_KEYPRESS_DICT[key]
        if self.view_axis_ == label:
            # Current viewing is already the same as the chosen one. 
            # Do nothing.
            return 

        # Set the viewing axis. 
        self.view_axis_ = label
        self.axis_mode_radio_.set_active(index)
        # The figure needs to be updated too. 
        self._update_figure(axis_switch=True)
        return 


    def _handle_planes_keyboard_shortcuts(self,
                                          key):
        """
        Handle keyboard shortcuts for planes. 
        Move along annotated planes depending on what shortcut was used. 
        """

        if key == 'z':
            if self.z_ > 0:
                z__ = self.z_ - 1
                while z__ >= 0 and not self.annotated_planes_[z__]:
                    z__ -= 1

                if z__ >= 0 and self.annotated_planes_[z__]:
                    self.z_ = z__

        elif key == 'm':
            if self.z_ < self.n_slices - 1:
                z__ = self.z_ + 1
                while z__ < self.n_slices and not self.annotated_planes_[z__]:
                    z__ += 1
                if z__ < self.n_slices and self.annotated_planes_[z__]:
                    self.z_ = z__

        # Update the z slider as well. A '+ 1' is needed because on the slider, the
        #   slide numbers start from 0.
        self.z_slider_.set_val(self.z_ + 1)
        # Update figure to reflect the new image.
        self._update_figure()

        return

    def _handle_move_keyboard_shortcuts(self,
                                        key):

        """
        Handle moving along the image using keyboard shortcuts. 
        'up' and 'down' keys move in -1 and +1 along Y, respectively. 
        'left', and 'right' keys move in -1 and +1 along X, respectively.
        """

        # First, make sure the figure is initialised.
        assert self.initialised_, 'Figure has not been initialised yet!'

        # Get the displacement from the MOVE_KEYPRESS_DICT
        displacement    = MOVE_KEYPRESS_DICT[key]
        # Move in the image according to the displacement. 
        self.move_image(displacement=displacement, location=None)
        return 
            


    # Handle x slider.

    def _handle_x_slider(self,
                         x_val):
        """
        Function to handle the X-slider. 
        Updates the image shown in the image panel according to the value of x. 
        Function name starts with an underscore in order to be invisible to the outside world.
        """

        # First, make sure the figure has been initialised.
        assert self.initialised_, 'Figure has not been initialised yet!'

        # Update the index of the currently shown slice.
        # Takes '- 1' because slices are numbered starting from 1 in the GUI.
        self.x_ = int(x_val) - 1
        # Show new image.
        self._update_figure()
        return

    # Handle y slider.

    def _handle_y_slider(self,
                         y_val):
        """
        Function to handle the Y-slider. 
        Updates the image shown in the image panel according to the value of y. 
        Function name starts with an underscore in order to be invisible to the outside world.
        """

        # First, make sure the figure has been initialised.
        assert self.initialised_, 'Figure has not been initialised yet!'

        # Update the index of the currently shown slice.
        # Takes '- 1' because slices are numbered starting from 1 in the GUI.
        self.y_ = int(y_val) - 1
        # Show new image.
        self._update_figure()
        return

    # Handle z slider.

    def _handle_z_slider(self,
                         z_val):
        """
        Function to handle the Z-slider. 
        Updates the image shown in the image panel according to the value of z. 
        Function name starts with an underscore in order to be invisible to the outside world.
        """

        # First, make sure the figure has been initialised.
        assert self.initialised_, 'Figure has not been initialised yet!'

        # Update the index of the currently shown slice.
        # Takes '- 1' because slices are numbered 1,...,z_limits[1]-z_limits[0] in the GUI.
        self.z_ = int(z_val) - 1
        # Show new image.
        self._update_figure()
        return

    # Handle brush size slider.

    def _handle_brush_size_slider(self,
                                  bs_val):
        """
        Function to handle the size of the brush which annotates. 
        """

        # First, make sure the figure has been initialised.
        assert self.initialised_, 'Figure has not been initialised yet!'

        self.brush_size_ = int(bs_val)
        return

    # Handle reset slide button.

    def _handle_reset_slide_button(self,
                                   event):
        """
        Reset any annotations that have been made for the current slide.
        This also takes into account what the current viewing axis is. 
        """

        # First, make sure the figure has been initialised.
        assert self.initialised_, 'Figure has not been initialised yet!'

        if self.view_axis_ == X:
            # Reset annotations.
            self.fg_annotation_[:, :, self.x_] = 0
            self.bg_annotation_[:, :, self.x_] = 0
            # Copy data from original slice into disp_slice. Need only reset the G and B channels.
            self.disp_stack_[:, :, self.x_, self.fg_channel] = self.stack[:, :, self.x_]
            self.disp_stack_[:, :, self.x_, self.bg_channel] = self.stack[:, :, self.x_]

        if self.view_axis_ == Y:
            # Reset annotations.
            self.fg_annotation_[:, self.y_, :] = 0
            self.bg_annotation_[:, self.y_, :] = 0
            # Copy data from original slice into disp_slice. Need only reset the G and B channels.
            self.disp_stack_[:, self.y_, :, self.fg_channel] = self.stack[:, self.y_, :]
            self.disp_stack_[:, self.y_, :, self.bg_channel] = self.stack[:, self.y_, :]

        if self.view_axis_ == Z:
            # Reset annotations.
            self.fg_annotation_[self.z_, :, :] = 0
            self.bg_annotation_[self.z_, :, :] = 0
            # Copy data from original slice into disp_slice. Need only reset the G and B channels.
            self.disp_stack_[self.z_, :, :, self.fg_channel] = self.stack[self.z_, :, :]
            self.disp_stack_[self.z_, :, :, self.bg_channel] = self.stack[self.z_, :, :]


        # Redraw current figure.
        self._update_figure()
        return

    # Handle reset slide button.

    def _handle_reset_all_button(self,
                                 event):
        """
        Reset any annotations that have been made for the current slide.
        """

        # First, make sure the figure has been initialised.
        assert self.initialised_, 'Figure has not been initialised yet!'

        # Reset annotations.
        self.fg_annotation_[:] = 0
        self.bg_annotation_[:] = 0
        # Copy data from original slice into disp_slice. Need only reset the G and B channels.
        self.disp_stack_[:, :, :, self.fg_channel] = self.stack
        self.disp_stack_[:, :, :, self.bg_channel] = self.stack
        # Redraw current figure.
        self._update_figure()
        return

    # Handle save slice button
    def _handle_save_slice_button(self,
                                  event):
        """
        Save current slice's annotation to ANNO_DIR.
        """

        # First, make sure the figure has been initialised.
        assert self.initialised_, 'Figure has not been initialised yet!'

        align_left('Saving slice {} annotations to {}'.format(
            self.z_ + 1, self.anno_dir))

        # Create savenames
        fg_save_name = os.path.join(
            self.anno_dir, slice_file_name(self.z_, FOREGROUND))
        bg_save_name = os.path.join(
            self.anno_dir, slice_file_name(self.z_, BACKGROUND))
        # Save as TIFF images.
        imageio.imsave(fg_save_name, self.fg_annotation_[self.z_, :, :] * 255)
        imageio.imsave(bg_save_name, self.bg_annotation_[self.z_, :, :] * 255)

        write_done()
        return

    # Handle save button.

    def _handle_save_button(self,
                            event):
        """
        Save current annotation for all slices
        """

        # First, make sure the figure has been initialised.
        assert self.initialised_, 'Figure has not been initialised yet!'

        align_left('Saving all annotations to {}'.format(self.anno_dir))

        for z_ in range(self.stack_depth):
            # Create savenames
            fg_save_name = os.path.join(
                self.anno_dir, slice_file_name(z_, FOREGROUND))
            bg_save_name = os.path.join(
                self.anno_dir, slice_file_name(z_, BACKGROUND))
            # Save as TIFF images.
            imageio.imsave(fg_save_name, self.fg_annotation_[z_, :, :] * 255)
            imageio.imsave(bg_save_name, self.bg_annotation_[z_, :, :] * 255)

        write_done()
        return

    # Handle loading previous annotation.

    def _handle_load_box(self,
                         load_path):
        """
        Handle loading previously saved annotation.
        """

        # First, make sure the figure has been initialised.
        assert self.initialised_, 'Figure has not been initialised yet!'

        # Make sure supplied path exists.
        if not os.path.exists(load_path):
            self.load_box_.set_val(
                load_path + ' | ERROR: File does not exist!')
            return

        for dir_ in [anno_dir, data_dir]:
            if not os.path.exists(dir_):
                os.makedirs(dir_)

        # Path to load from is in text.
        saved_files = os.listdir(load_path)
        # Load all files. The filename determines which annotation slice to set.
        for sf in saved_files:
            sf_path = os.path.join(load_path, sf)
            anno = imageio.imread(sf_path) / 255
            z, label = info_from_filename(sf_path)
            if label == FOREGROUND:
                self.fg_annotation_[z, :, :] = anno
            else:
                self.bg_annotation_[z, :, :] = anno

        # Update current figure with annotation.
        self._update_annotation_visual()
        self._update_figure()
        return

    def _set_axis_labels(self):
        """
        Set axis names for the figure according to the viewing axis. 
        """
        if self.view_axis_ == X:
            self.ax_image_.set_xlabel(Y)
            self.ax_image_.set_ylabel(Z)

        elif self.view_axis_ == Y:
            self.ax_image_.set_xlabel(X)
            self.ax_image_.set_ylabel(Z)

        elif self.view_axis_ == Z:
            self.ax_image_.set_xlabel(X)
            self.ax_image_.set_ylabel(Y)

    def _set_axis_ticks(self):
        """
        Set axis tick labels according to the viewing rectangle.

        The ticks are always set to be at multiples of five.
        """
        
        x_start, y_start = self.viewing_rect[self.view_axis_]

        rem_x = x_start % TICK_STEP
        rem_y = y_start % TICK_STEP

        x_start_tick = (TICK_STEP - rem_x) % TICK_STEP
        y_start_tick = (TICK_STEP - rem_y) % TICK_STEP

        x_view_size, y_view_size = self.view_sizes[self.view_axis_]    

        x_ticks = np.arange(x_start_tick, x_view_size, TICK_STEP)
        y_ticks = np.arange(y_start_tick, y_view_size, TICK_STEP)

        x_tick_labels = [str(x + x_start) for x in x_ticks]
        y_tick_labels = [str(y + y_start) for y in y_ticks]

        self.ax_image_.set_yticks(y_ticks)
        self.ax_image_.set_yticklabels(y_tick_labels)

        self.ax_image_.set_xticks(x_ticks)
        self.ax_image_.set_xticklabels(x_tick_labels)

        return



    # Initialise pyplot figure and run the GUI.

    def __call__(self):
        """
        Initialise a pyplot figure with this VolumeAnnotator
        and put it into motion. This function draws the initial plot, 
        defines the objects, and connects object handlers to the objects. 
        """

        # Close other figures
        plt.close()

        # Make new figure.
        self.figure_, self.axes_ = plt.subplots(figsize=self.figure_size_)
        # Adjust subplot---set margins.
        plt.subplots_adjust(left=self.m_left_, bottom=self.m_bottom_,
                            right=self.m_right_, top=self.m_top_)
        # Turn off axis.
        plt.axis('off')
        # Set background colour for plot.
        self.axes_.set_facecolor(self.face_colour_)

        # Create axes for image.
        # The above was replaced with the following
        self.ax_image_ = None
        self._update_figure(axis_switch=True)

        # ================================
        #   Create axes for X-slider

#        self.ax_x_slider_ = plt.axes([0.05, 0.05, 0.50, 0.025],
#                                     facecolor=self.ax_colour_)
        self.ax_x_slider_ = plt.axes([0.65, 0.3, 0.30, 0.025],
                                     facecolor=self.ax_colour_)
        # Add X-slider.
        self.x_slider_ = Slider(self.ax_x_slider_, 'X', 1,
                                self.W, 
                                valinit=self.x_+1, valstep=1)
        # Add the x-slide handler function to x_slider_
        self.x_slider_.on_changed(self._handle_x_slider)
        # ================================



        # ================================
        #   Create axes for Y-slider

#        self.ax_y_slider_ = plt.axes([0.05, 0.085, 0.50, 0.025],
#                                     facecolor=self.ax_colour_)
        self.ax_y_slider_ = plt.axes([0.65, 0.335, 0.30, 0.025],
                                     facecolor=self.ax_colour_)
        # Add Z-slider.
        self.y_slider_ = Slider(self.ax_y_slider_, 'Y', 1,
                                self.H, 
                                valinit=self.y_+1, valstep=1)
        # Add the y-slide handler function to y_slider_
        self.y_slider_.on_changed(self._handle_y_slider)
        # ================================



        # ================================
        #   Create axes for Z-slider

#        self.ax_z_slider_ = plt.axes([0.05, 0.12, 0.50, 0.025],
#                                     facecolor=self.ax_colour_)
        self.ax_z_slider_ = plt.axes([0.65, 0.37, 0.30, 0.025],
                                     facecolor=self.ax_colour_)
        # Add Z-slider.
        self.z_slider_ = Slider(self.ax_z_slider_, 'Z', 1,
                                self.z_limits[1] - self.z_limits[0],
                                valinit=self.z_+1, valstep=1)
        # Add the z-slide handler function to z_slider_
        self.z_slider_.on_changed(self._handle_z_slider)
        # ================================

        # ================================
        #   Create slider for brush size.
        self.ax_brush_size_ = plt.axes([0.68, 0.9, 0.27, 0.025],
                                       facecolor=self.ax_colour_)
        # Add slider.
        self.max_brush_size = 41
        self.brush_size_slider_ = Slider(self.ax_brush_size_, 'Brush size', 1, self.max_brush_size,
                                         valinit=1, valstep=2)
        # Add the brush size handler function.
        self.brush_size_slider_.on_changed(self._handle_brush_size_slider)
        # ================================

        # ================================
        #   Add radio buttons for foreground and background
        self.ax_ann_mode_radio_ = plt.axes([0.65, 0.7, 0.1, 0.1],
                                           facecolor=self.ax_colour_)
        # Add radio buttons
        self.ann_mode_radio_ = RadioButtons(self.ax_ann_mode_radio_,
                                            ANNO_CHOICES, active=0)
        # Add handler for radio buttons.
        self.ann_mode_radio_.on_clicked(self._handle_ann_mode_radio)
        # ================================

        # ================================
        #   Add radio buttons for viewing axis. 
        self.ax_axis_mode_ratio_ = plt.axes([0.85, 0.7, 0.1, 0.1],
                                           facecolor=self.ax_colour_)
        # Add radio buttons
        self.axis_mode_radio_ = RadioButtons(self.ax_axis_mode_ratio_,
                                            AXIS_CHOICES, active=2)
        # Add handler for radio buttons.
        self.axis_mode_radio_.on_clicked(self._handle_axis_mode_radio) # TODO
        # ================================

 

        # ================================
        #   Add buttons.
        betw = 0.01
        butw = 0.0675
        buth = 0.05

        self.ax_reset_slice_button_ = plt.axes([0.65, 0.05, butw, buth],
                                               facecolor=self.ax_colour_)
        self.ax_reset_all_button_ = plt.axes([0.65+butw+betw, 0.05, butw, buth],
                                             facecolor=self.ax_colour_)
        self.ax_save_slice_button_ = plt.axes([0.65+2*butw+2*betw, 0.05, butw, buth],
                                              facecolor=self.ax_colour_)
        self.ax_save_button_ = plt.axes([0.65+3*butw+3*betw, 0.05, butw, buth],
                                        facecolor=self.ax_colour_)
        self.ax_load_box_ = plt.axes([0.65, 0.05+betw+buth, butw*4+betw*3, buth],
                                     facecolor=self.ax_colour_)

        # Initialise buttons.
        self.reset_slice_button_ = Button(self.ax_reset_slice_button_, 'Reset Slice',
                                          color=self.ax_colour_, hovercolor=self.hover_colour_)
        self.reset_all_button_ = Button(self.ax_reset_all_button_, 'Reset All',
                                        color=self.ax_colour_, hovercolor=self.hover_colour_)
        self.save_slice_button_ = Button(self.ax_save_slice_button_, 'Save Slice',
                                         color=self.ax_colour_, hovercolor=self.hover_colour_)
        self.save_button_ = Button(self.ax_save_button_, 'Save',
                                   color=self.ax_colour_, hovercolor=self.hover_colour_)
        self.load_box_ = TextBox(self.ax_load_box_, 'Load',
                                 initial=ANNO_DIR)

        # Add handlers to buttons.
        self.reset_slice_button_.on_clicked(self._handle_reset_slide_button)
        self.reset_all_button_.on_clicked(self._handle_reset_all_button)
        self.save_slice_button_.on_clicked(self._handle_save_slice_button)
        self.save_button_.on_clicked(self._handle_save_button)
        self.load_box_.on_submit(self._handle_load_box)
        # ================================

        # ================================
        #   Add annotation handlers.
        plt.connect('button_press_event',       self._handle_mouse_press)
        plt.connect('motion_notify_event',      self._handle_mouse_move)
        plt.connect('button_release_event',     self._handle_mouse_release)
        # Move along on Z.
        plt.connect('scroll_event',             self._handle_scroll_event)
        # Keyboard shorcuts.
        plt.connect('key_press_event',
                    self._handle_keyboard_shortcuts)
        # ================================

        self.initialised_ = True
        plt.show()

    # ========================================================================================
    #   Wrappers for "private" functions. SHOULD BE REMOVED IN THE FINAL VERSION.
    def handle_x_slider(self,
                        *args,
                        **kwargs):
        """
        Wrapper for _handle_x_slider which can be removed later!
        """
        return self._handle_x_slider(*args, **kwargs)

    def handle_y_slider(self,
                        *args,
                        **kwargs):
        """
        Wrapper for _handle_y_slider which can be removed later!
        """
        return self._handle_y_slider(*args, **kwargs)

    def handle_z_slider(self,
                        *args,
                        **kwargs):
        """
        Wrapper for _handle_z_slider which can be removed later!
        """
        return self._handle_z_slider(*args, **kwargs)
    # ========================================================================================


if __name__ == '__main__':
    # Remove default matplotlib keyboard bindings
    for function in plt.rcParams:
        if function.startswith('keymap.'):
            plt.rcParams[function] = []

    # Ensure that keypresses have corresponding radio buttons.
    for key_ in ANNO_KEYPRESS_DICT:
        assert ANNO_KEYPRESS_DICT[key_][0] in ANNO_CHOICES, '{} not found in ANNO_CHOICES. All keyboard shortcuts \
for annotations must have associated radio buttons!'.format(ANNO_KEYPRESS_DICT[key_][0])
    for key_ in AXIS_KEYPRESS_DICT:
        assert AXIS_KEYPRESS_DICT[key_][0] in AXIS_CHOICES, '{} not found in AXIS_CHOICES. All keyboard shortcuts \
for axes must have associated radio buttons!'.format(AXIS_KEYPRESS_DICT[key_][0])

    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', default='configs/template.yaml',
                        help='Configuration file to use.')

    args = parser.parse_args()

    assert os.path.exists(
        args.cfg), 'Specified configuration file {} does not exist!'.format(args.cfg)

    volume_annotator = VolumeAnnotator(cfg=args.cfg)
    volume_annotator()
