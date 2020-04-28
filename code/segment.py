from utils import *
from VolumeAnnotator import *
import pickle
import time
import numpy as np

import threading
import time
import itertools

import argparse

import maxflow
import logging
import numpy as np
import SimpleITK as sitk

from funlib.segment.arrays import replace_values

import visualization_utils
from utils import get_supervoxel_size

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


anim_done_ = False
SAVED_VARS_FNAME = 'saved_vars.pkl'


def animate_loading(text):
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if anim_done_:
            break

        write_flush('\r{}  {}'.format(text, c))
        time.sleep(CLOCK_DELAY)

    write_flush('\r')
    align_left(text)
    return


class SegmentationModule(object):
    def __init__(
        self,
        cfg,
    ):

        super(SegmentationModule, self).__init__()

        # ====================================================================
        #   Load options, and assign default values.
        align_left('Reading and fixing configuration file')
        self.options = load_yaml(cfg)
        write_okay()
        # ====================================================================

        # Get experiment name
        self.experiment_name = '.'.join(os.path.split(cfg)[-1].split('.')[:-1])
        self.output_dir = os.path.join('output/', self.experiment_name)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.anno_dir = os.path.join(self.output_dir, 'annotation/')
        assert os.path.exists(
            self.anno_dir), 'No annotations found in {}. Please annotate first!'.format(
            self.anno_dir)

        self.data_dir = os.path.join(self.output_dir, 'data/')
        self.mask_dir = os.path.join(self.output_dir, 'mask/')
        self.seg_dir = os.path.join(self.output_dir, 'seg/')

        for dir_ in [self.data_dir, self.mask_dir, self.seg_dir]:
            if not os.path.exists(dir_):
                os.makedirs(dir_)

        # ====================================================================
        #   Make sure data path exists.
        self.data_path = self.options.data_path
        assert os.path.exists(
            self.data_path), 'Specified data_path {} does not exist!'.format(
            self.data_path)

        # ====================================================================
        #   Print configuration hyperparams.
        write_flush('\nThe configuration of the experiment is: \n')
        write_flush(
            '{:>30} : {}\n'.format(
                'Configuration file',
                os.path.abspath(cfg)))
        for _param in self.options:
            _val = self.options[_param]
            write_flush('{:>30} : {}\n'.format(_param, _val))
        write_flush('\n')
        # ====================================================================

        # ====================================================================
        #   Variables of this class to be saved.
        self.save_vars = [
            'labels1',
            'supervoxel_indices',
            'adj_mat',
            'spx_locations',
            'grey_v_px',
            'beta',
            'fg_unaries',
            'bg_unaries',
            'fg_inv_prob_spx',
            'bg_inv_prob_spx',
            'pw_costs',
        ]
        for member_ in self.save_vars:
            setattr(self, member_, None)

        return

    def make_save_dict(self):
        save_dict = {}
        for member_ in self.save_vars:
            save_dict[member_] = getattr(self, member_)

        return save_dict

    def save_variables(self):
        self.save_vars_path = os.path.join(self.output_dir, SAVED_VARS_FNAME)
        align_left('Saving variables to %s' % (self.save_vars_path))

        save_dict = self.make_save_dict()
        try:
            with open(self.save_vars_path, 'wb') as fp:
                pickle.dump(save_dict, fp)
        except Exception as E:
            write_fail()
            print('Could not save variables. Got error {}'.format(E))
            exit(1)

        write_okay()
        return

    def load_variables(self):
        """
        If previously saved variables exist, load them.
        """
        self.load_vars_path = os.path.join(self.output_dir, SAVED_VARS_FNAME)
        if not os.path.exists(self.load_vars_path):
            return

        align_left('Loading previously computed variables')
        try:
            with open(self.load_vars_path, 'rb') as fp:
                saved_vars = pickle.load(fp)
        except Exception as E:
            write_fail()
            print('Could not load variables. Got error {}'.format(E))
            exit(1)
        write_okay()

        write_flush('I found the following stored variables:\n')
        for key in saved_vars:
            print('{:>30}'.format(key))

        align_left('Using previously computed variables')
        for member_ in saved_vars:
            setattr(self, member_, saved_vars[member_])
        write_okay()

        return

    def __call__(
            self,
        **kwargs,
    ):

        # ====================================================================
        #   Load the TIFF stack.
        align_left('Reading TIFF stack')

        tif_stack = []
        if self.options.coords:
            X, Y, W, H = self.options.coords
        else:
            X, Y, W, H = None, None, None, None  # BOX_COORDS[self.data_path]

        file_list = sorted(os.listdir(self.data_path))
        n_files = len(file_list)

        if not self.options.z_limits:
            z_limits = [0, n_files]
        else:
            z_limits = self.options.z_limits

        for _z in range(z_limits[0], z_limits[1]):
            fname = file_list[_z]
            img = imageio.imread(os.path.join(self.data_path, fname))
            # Crop the image if limits are specified.
            if X is not None:
                img = img[Y:Y + H, X:X + W]
            tif_stack.append(img[None, ...])
        image = np.concatenate(tif_stack, axis=0)

        logger.info(f'image shape {image.shape}')
        write_okay()
        # ====================================================================

        self.load_variables()

        # ====================================================================
        #   Perform SLIC segmentation into superpixels.
        align_left('Discovering supervoxels')
        start_time = time.time()
        if self.labels1 is None:

            supervoxel_size = get_supervoxel_size(
                self.options.n_superpixels,
                image.shape
            )

            sitk_img = sitk.GetImageFromArray(image)
            logger.debug(
                f'sitk image size {sitk_img.GetSize()}')
            logger.debug(
                f'sitk image dtype {sitk_img.GetPixelIDTypeAsString()}')

            # TODO port relevant params into config
            slic = sitk.SLICImageFilter()
            slic.SetSpatialProximityWeight(self.options.compactness)
            slic.SetSuperGridSize([supervoxel_size] * 3)
            slic.SetEnforceConnectivity(True)
            slic.SetNumberOfThreads(self.options.n_jobs)
            slic.SetMaximumNumberOfIterations(10)

            sitk_labels = slic.Execute(sitk_img)
            numpy_labels = sitk.GetArrayFromImage(sitk_labels)

            # TODO not sure if type casting is necessary, this is simply
            # matching the sklearn output dtype
            self.labels1 = numpy_labels.astype('int64')

            debug_start_time = time.time()
            old_values = np.unique(self.labels1)
            # this currently needs a boost install via apt
            replace_values(
                in_array=self.labels1,
                inplace=True,
                old_values=old_values,
                new_values=np.arange(len(old_values), dtype=old_values.dtype))

            logger.debug((
                'Replacing supervoxel ids to be contiguous took '
                f'{time.time() - debug_start_time} s'
            ))

            logger.info(
                f'\nNumber of generated supervoxels: {len(np.unique(self.labels1))}'
            )

        write_done(start_time)
        print(
            'Size of the volume is Z: %d, Y: %d, X: %d' %
            (self.labels1.shape[0],
             self.labels1.shape[1],
             self.labels1.shape[2]))
        # ====================================================================

        # ====================================================================
        #   Compute adjacency matrix.
        align_left('Computing adjacency matrix')
        start_time = time.time()
        if self.adj_mat is None:
            self.adj_mat = make_adjacency_matrix(self.labels1)

        write_done(start_time)
        # ====================================================================

    #    # Close open plots.
    #    plt.close()
    #
    #    # Column width, row height.
    #    bc, br = 5, 4
    #
    #    # Number of columns, number of rows.
    #    nc, nr = 1, 1
    #
    #    # Make figure.
    #    fig = plt.figure(figsize=(nc*bc, nr*br))
    #
    #    # Make subplots and get axes.
    #    axes    = []
    #    handles = []
    #    axes.append(fig.add_subplot(111))
    #    plt.tight_layout()
    #    handles.append(axes[0].imshow(bd[0,:,:], cmap='gray'))
    #
    #    # Initialisation
    #    def init():
    #        handles.append(axes[0].imshow(bd[0,:,:], cmap='gray'))
    #        return handles
    #
    #    # animate function. Determines what to display for index i.
    #    def animate(i):
    #        bd_ = bd[i,:,:]
    #
    #        handles[0].set_data(bd_)
    #
    #        return handles
    #
    #    # call the animator. blit=True means only re-draw the parts that have changed.
    #    anim = animation.FuncAnimation(fig, animate,
    #                                   frames=range(image.shape[0]), interval=200, blit=True)
    #
    #
    #    HTML(anim.to_html5_video())

        # ====================================================================
        #   Get grey values for superpixels.
        align_left('Compting grey values for supervoxels')
        start_time = time.time()
        if self.grey_v_px is None:
            self.grey_v_px, self.supervoxel_indices = get_grey_values(
                image, self.labels1, n_jobs=self.options.n_jobs)
        write_done(start_time)
        # ====================================================================

        # ====================================================================
        #   Compute centres of superpixels.
        if self.spx_locations is None:
            start_time = time.time()
            X_, Z_, Y_ = np.meshgrid(range(image.shape[1]),
                                     range(image.shape[0]),
                                     range(image.shape[2]))
            align_left('Compting x-coordinate of centres of supervoxels')
            spx_x_coord, _ = get_grey_values(
                X_,
                self.labels1,
                n_jobs=self.options.n_jobs,
                indices=self.supervoxel_indices)
            write_done(start_time)

            start_time = time.time()
            align_left('Compting y-coordinate of centres of supervoxels')
            spx_y_coord, _ = get_grey_values(
                Y_,
                self.labels1,
                n_jobs=self.options.n_jobs,
                indices=self.supervoxel_indices)
            write_done(start_time)

            start_time = time.time()
            align_left('Compting z-coordinate of centres of supervoxels')
            spx_z_coord, _ = get_grey_values(
                Z_,
                self.labels1,
                n_jobs=self.options.n_jobs,
                indices=self.supervoxel_indices)
            write_done(start_time)
            self.spx_locations = np.concatenate((spx_x_coord[..., None],
                                                 spx_y_coord[..., None],
                                                 spx_z_coord[..., None]), axis=1)
        # ====================================================================

        # ====================================================================
        #   Computing beta
        align_left('Computing beta')
        start_time = time.time()
        if self.beta is None:
            self.beta = compute_beta(self.grey_v_px, self.adj_mat,
                                     n_jobs=self.options.n_jobs)
        write_done(start_time)
        # ====================================================================

        # ====================================================================
        #   Reading annotation.
        align_left('Reading annotations')
        start_time = time.time()
        fg_img = np.zeros(image.shape)
        bg_img = np.zeros(image.shape)

        anno_files = os.listdir(self.anno_dir)

        for f_ in anno_files:
            f_path = os.path.join(self.anno_dir, f_)

            anno_plane = int(os.path.split(f_path)
                             [-1].split('.')[0].split('_')[1])
            anno_img = imageio.imread(f_path) / 255

            if any([k in f_path for k in ['Foreground', 'Object']]
                   ) and anno_plane < image.shape[0]:
                fg_img[anno_plane, :, :] = anno_img
            elif 'Background' in f_path and anno_plane < image.shape[0]:
                bg_img[anno_plane, :, :] = anno_img
    #        else:
    # raise ValueError('Could not understand annotation file %s' %(f_path))
        write_done(start_time)
        # ====================================================================

        # fg_img_anno_first = imageio.imread('foreground_first.tif') / 255
        # bg_img_anno_first = imageio.imread('background_first.tif') / 255
        # fg_img_anno_last  = imageio.imread('foreground_last.tif') / 255
        # bg_img_anno_last  = imageio.imread('background_last.tif') / 255

        # fg_img[0,:,:]     = fg_img_anno_first
        # fg_img[-1,:,:]    = fg_img_anno_last
        # bg_img[0,:,:]     = bg_img_anno_first
        # bg_img[-1,:,:]    = bg_img_anno_last

        # Using gaussians - wrong.
        # fg_inv_energy = place_gaussians(fg_img, sigmax=50, sigmay=50)
        # bg_inv_energy = place_gaussians(bg_img, sigmax=50, sigmay=50)
        # fg_inv_prob = np.exp(-fg_inv_energy)
        # bg_inv_prob = np.exp(-bg_inv_energy)

        # Using distance transform
        # ====================================================================
        #   Compute distance transform
        align_left('Computing distance transform')
        start_time = time.time()
        fg_inv_energy = morphology.distance_transform_edt(1 - fg_img)
        bg_inv_energy = morphology.distance_transform_edt(1 - bg_img)
        fg_inv_prob = fg_inv_energy.astype(np.float32)
        bg_inv_prob = bg_inv_energy.astype(np.float32)
        write_done(start_time)
        # ====================================================================

        # ====================================================================
        #   Compute distance unaries for superpixels.
        align_left('Computing distance unaries for supervoxels')
        start_time = time.time()
        fg_inv_prob_spx = np.log(
            1 +
            get_grey_values(
                fg_inv_prob,
                self.labels1,
                n_jobs=self.options.n_jobs,
                indices=self.supervoxel_indices)[0])
        bg_inv_prob_spx = np.log(
            1 +
            get_grey_values(
                bg_inv_prob,
                self.labels1,
                n_jobs=self.options.n_jobs,
                indices=self.supervoxel_indices)[0])

        self.fg_inv_prob_spx = fg_inv_prob_spx
        self.bg_inv_prob_spx = bg_inv_prob_spx
        write_done(start_time)
        # ====================================================================

        # ====================================================================
        #   Compute histogram unaries
        align_left('Computing foreground and background histograms')
        start_time = time.time()
        fg_pix = image[fg_img > 0]
        bg_pix = image[bg_img > 0]

        bins = np.linspace(image.min(), image.max(), 6)
        fg_hist = 1 + np.histogram(fg_pix, bins=bins)[0]
        bg_hist = 1 + np.histogram(bg_pix, bins=bins)[0]

        fg_hist = fg_hist / np.sum(fg_hist)
        bg_hist = bg_hist / np.sum(bg_hist)
        write_done(start_time)
        # ====================================================================

        # ====================================================================
        #   Set unaries for annotated supervoxels
        align_left('Setting unaries for annotated voxels')
        start_time = time.time()
        fg_pix_unaries = -np.log(fg_hist[np.digitize(image.flatten(),
                                                     bins=bins,
                                                     right=True)
                                         - 1].reshape(image.shape))
        bg_pix_unaries = -np.log(bg_hist[np.digitize(image.flatten(),
                                                     bins=bins,
                                                     right=True)
                                         - 1].reshape(image.shape))

        K = 100 * (1 + np.max(np.concatenate((fg_pix_unaries,
                                              bg_pix_unaries))))

        fg_pix_unaries[fg_img > 0] = 0
        fg_pix_unaries[bg_img > 0] = K

        bg_pix_unaries[bg_img > 0] = 0
        bg_pix_unaries[fg_img > 0] = K

        write_done(start_time)
        # ====================================================================

        # ====================================================================
        align_left('Computing histogram unaries')
        start_time = time.time()
        fg_unaries, _ = get_grey_values(
            fg_pix_unaries,
            self.labels1,
            n_jobs=self.options.n_jobs,
            indices=self.supervoxel_indices)
        bg_unaries, _ = get_grey_values(
            bg_pix_unaries,
            self.labels1,
            n_jobs=self.options.n_jobs,
            indices=self.supervoxel_indices)
        write_done(start_time)

        self.fg_unaries = fg_unaries
        self.bg_unaries = bg_unaries
        # ====================================================================

        # ====================================================================
        #   Compute pairwise costs.
        align_left('Computing pairwise terms')
        start_time = time.time()
        pw_edges, pw_costs = get_pairwise_costs(
            self.grey_v_px, self.spx_locations, self.adj_mat, self.beta, image.shape[:3], sigma=50)
        pw_costs = np.array(pw_costs)
        self.pw_costs = pw_costs
        write_done(start_time)
        # ====================================================================

        # fd, hog_image = hog(img, orientations=16, voxels_per_cell=(8, 8),
        # cells_per_block=(1, 1), visualize=True, feature_vector=False,
        # multichannel=False)

        # fd_ex = cv2.resize(fd[:,:,0,0,0], (img.shape[1], img.shape[0]),
        # interpolation=cv2.INTER_NEAREST)

        # fd_ex = np.concatenate([cv2.resize(fd[:,:,0,0,i], (img.shape[1],
        # img.shape[0]), interpolation=cv2.INTER_NEAREST)[...,None] for i in
        # range(16)], axis=-1)

        # fd_sp = np.concatenate([get_grey_values(fd_ex[:,:,i],
        # self.labels1)[...,None] for i in range(16)], axis=-1)

        # _, hog_pw = get_pairwise_costs(fd_sp, self.spx_locations, self.adj_mat, 1, img.shape[:2], sigma=10)
        # hog_pw = np.array(hog_pw)

        # ====================================================================
        #   Set up the max-flow/min-cut problem.

        align_left('Setting up the max-flow/min-cut problem')
        start_time = time.time()
        n_superpixels = self.adj_mat.shape[0]
        n_edges = len(pw_edges)

        G = maxflow.Graph[float](n_superpixels, n_edges)

        delta = self.options.delta
        gamma = self.options.gamma

        lamda = self.options.lamda
        eta = self.options.eta

        G.add_nodes(n_superpixels)

        for n in range(n_superpixels):
            G.add_tedge(n,

                        delta * fg_unaries[n] +
                        gamma * fg_inv_prob_spx[n],

                        delta * bg_unaries[n] +
                        gamma * bg_inv_prob_spx[n]
                        )

        for e in range(n_edges):
            e0, e1 = pw_edges[e]
        #    k = lamda * (pw_costs[e] + eta * hog_pw[e])
            k = lamda * pw_costs[e]

            G.add_edge(e0, e1, k, k)

        write_done(start_time)
        # ====================================================================

        print(
            '{:<80}{:.4f}, {:.4f}'.format(
                'Limits of histogram-based foreground unaries',
                fg_unaries.min(),
                fg_unaries.max()))
        print(
            '{:<80}{:.4f}, {:.4f}'.format(
                'Limits of histogram-based background unaries',
                bg_unaries.min(),
                bg_unaries.max()))
        print(
            '{:<80}{:.4f}, {:.4f}'.format(
                'Limits of distance-based foreground unaries',
                fg_inv_prob_spx.min(),
                fg_inv_prob_spx.max()))
        print(
            '{:<80}{:.4f}, {:.4f}'.format(
                'Limits of distance-based background unaries',
                bg_inv_prob_spx.min(),
                bg_inv_prob_spx.max()))
        print(
            '{:<80}{:.4f}, {:.4f}'.format(
                'Limits of pairwise terms',
                pw_costs.min(),
                pw_costs.max()))
        # print(hog_pw.min(), hog_pw.max())

        # ====================================================================
        #   Compute max flow
        align_left('Solving ')
        start_time = time.time()
        G.maxflow()
        write_done(start_time)
        # ====================================================================

        #   Read hidden variables
        align_left('Retrieving values of hidden variables')
        labels = np.zeros(n_superpixels)
        label_img = np.zeros_like(image)
        for n in range(n_superpixels):
            labels[n] = G.get_segment(n)
        label_img = labels[self.labels1]
        write_okay()
        # ====================================================================

        # Extract foreground and background from volume.
        foreground = image * label_img
        background = image * (1 - label_img)

        # ====================================================================
        #   Save result.
        align_left('Saving masks and segmentation to %s' % (self.output_dir))

        # Whether to apply the GMM.
        mask_img = np.zeros(image.shape[1:], dtype=np.uint8)
        seg_img = np.zeros(image.shape[1:], dtype=np.uint8)

        orig_file_list = file_list

        try:
            if self.options.use_gmm:
                gmm = skmixture.GaussianMixture(
                    n_components=self.options.n_gmm_components
                )
                gmm.fit(image[label_img == 1][..., None])
                y = gmm.predict(image[label_img == 1][..., None])

                new_fg_mask = np.zeros_like(image)
                new_fg_mask[label_img == 1] = 1 + y

                cropped_mask_paths = []
                cropped_seg_paths = []

                for gmmc in range(self.options.n_gmm_components):
                    mask_path_ = os.path.join(
                        self.mask_dir, 'mask%d/' %
                        (gmmc))
                    seg_path_ = os.path.join(self.seg_dir, 'seg%d/' % (gmmc))

                    cropped_mask_paths.append(mask_path_)
                    cropped_seg_paths.append(seg_path_)

                for dir_ in cropped_mask_paths + cropped_seg_paths:
                    if not os.path.exists(dir_):
                        os.makedirs(dir_)

                for s in range(new_fg_mask.shape[0]):
                    for gmmc in range(self.options.n_gmm_components):
                        mask = (new_fg_mask[s, :, :] == (
                            1 + gmmc)).astype(np.uint8)
                        mask_img[:, :] = mask * 255
                        seg_img[:, :] = mask * image[s, :, :]
                        imageio.imsave(os.path.join(
                            cropped_mask_paths[gmmc], orig_file_list[s + z_limits[0]]), mask_img)
                        imageio.imsave(os.path.join(
                            cropped_seg_paths[gmmc], orig_file_list[s + z_limits[0]]), seg_img)

                    imageio.imsave(os.path.join(
                        self.data_dir, orig_file_list[s + z_limits[0]]), image[s, :, :])

            else:       # Do not use GMM.
                cropped_mask0_path = os.path.join(self.mask_dir, 'mask0/')
                cropped_seg0_path = os.path.join(self.seg_dir, 'seg0/')

                for dir_ in [cropped_mask0_path, cropped_seg0_path]:
                    if not os.path.exists(dir_):
                        os.makedirs(dir_)

                for s in range(image.shape[0]):
                    # oimg                =
                    # imageio.imread(os.path.join(self.data_path,
                    # orig_file_list[s+z_limits[0]]))
                    mask0 = label_img[s, :, :]

                    mask_img[:, :] = mask0 * 255
                    seg_img[:, :] = mask0 * image[s, :, :]
                    imageio.imsave(os.path.join(
                        cropped_mask0_path, orig_file_list[s + z_limits[0]]), mask_img)
                    imageio.imsave(os.path.join(
                        cropped_seg0_path, orig_file_list[s + z_limits[0]]), seg_img)
                    imageio.imsave(os.path.join(
                        self.data_dir, orig_file_list[s + z_limits[0]]), image[s, :, :])
        except Exception as E:
            write_fail()
            write_flush('Could not finish operation. Got error {}\n'.format(E))
            exit(1)

        write_okay()
        # ====================================================================

        # ====================================================================
        #   Save variables
        self.save_variables()
        # ====================================================================

        write_fini()
        exit()

        # Close open plots.
        plt.close()

        # Column width, row height.
        bc, br = 5, 4

        # Number of columns, number of rows.
        nc, nr = 2, 1

        # Make figure.
        fig = plt.figure(figsize=(nc * bc, nr * br))

        # Make subplots and get axes.
        axes = []
        handles = []
        axes.append(fig.add_subplot(121))
        axes.append(fig.add_subplot(122))
        plt.tight_layout()
        handles.append(axes[0].imshow(foreground[0, :, :]))
        handles.append(axes[1].imshow(background[0, :, :]))

        # Initialisation
        def init():
            handles.append(axes[0].imshow(foreground[0, :, :]))
            handles.append(axes[1].imshow(background[0, :, :]))
            return handles

        # animate function. Determines what to display for index i.
        def animate(i):
            fg_ = foreground[i, :, :]
            bg_ = background[i, :, :]

            handles[0].set_data(fg_)
            handles[1].set_data(bg_)

            return handles

        # call the animator. blit=True means only re-draw the parts that have
        # changed.
        anim = animation.FuncAnimation(
            fig, animate, frames=range(
                image.shape[0]), interval=200, blit=True)

        HTML(anim.to_html5_video())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', default='configs/template.yaml',
                        help='Configuration file to use.')

#    parser.add_argument('--data_path', default='/media/scratch/mihir/20190613/', \
#                type=str, help='Path containing TIFF files from stack.')
#    parser.add_argument('--anno_dir', default=ANNO_DIR, type=str, \
#                help='Path to directory in which to record annotations.')
#    parser.add_argument('--output_dir', default=OUTPUT_DIR, type=str, \
#                help='Path where to record cropped segmentation masks.')
#    parser.add_argument('--n_superpixels', default=4000, type=int, \
#                help='Number of superpixels')
#    parser.add_argument('--use_gmm', default=False, type=bool, \
#                help='Whether to use a GMM for further separation. Useful for Golgis')
#    parser.add_argument('--n_jobs', default=32, type=int, \
#                help='Number of CPU cores to use.')

    args = parser.parse_args()

    assert os.path.exists(
        args.cfg), 'Specified configuration file {} does not exist!'.format(
        args.cfg)

    seg_module = SegmentationModule(args.cfg)
    seg_module()
