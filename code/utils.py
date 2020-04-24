import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import skimage
from skimage import color as skcolor
from skimage.future import graph as skgraph
from skimage import data as skdata
from skimage import segmentation as sksegmentation
import imageio
import cv2
import maxflow
from skimage.feature import hog
from skimage import filters as skfilters
from scipy.ndimage import morphology
from sklearn import cluster
from sklearn import mixture as skmixture
from joblib import Parallel, delayed, parallel_backend

from matplotlib import animation, rc
from IPython.display import HTML
import argparse
from attr_dict import *

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


DEFAULT_HYPERPARAMS = {
    # Z-limits. Whether to crop the image along Z as well. If None,
    #    the entire stack is used.
    'z_limits': False,

    # Compactness for SLIC segmentation.
    'compactness': 10,
    # Initial isotropic superpixel size for slic segmentation.
    'superpixel_size': 16,


    # Multiplier for FG and BG histogram for unary terms.
    'delta': 1.0,
    # Multiplier for FG and BG distance for unary terms.
    'gamma': 1.0,
    # Multiplier for the smoothness prior.
    'lamda': 2.0,
    # Multiplier for the HOG-component of the smoothness prior.
    # Not used for now.
    'eta': 0.0,


    # Number of CPUs to use when using multiprocessing
    'n_jobs': 32,

    # Whether to use GMM to further segmentat the decoded mask.
    'use_gmm': True,
    'n_gmm_components': 2,

    # Ground truth, in case the score is to be calculated.
    'ground_truth': False,
}


CLOCK_DELAY = 0.1


class BColours:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def align_left(text):
    write_flush('{:<80}'.format(text))
    return


def write_okay():
    write_flush('[  %sOK%s  ]\n' % (BColours.OKGREEN, BColours.ENDC))
    return


def write_done(start_time=None):
    if start_time:
        duration = time.time() - start_time
        write_flush('[ %sDONE in %.2f s%s ]\n' %
                    (BColours.OKGREEN, duration, BColours.ENDC))
    else:
        write_flush('[ %sDONE%s ]\n' % (BColours.OKGREEN, BColours.ENDC))
    return


def write_fini():
    write_flush('[ %sFINI%s ]\n' % (BColours.OKBLUE, BColours.ENDC))
    return


def write_fail():
    write_flush('[ %sFAIL%s ]\n' % (BColours.FAIL, BColours.ENDC))
    return


def write_flush(text, stream=sys.stdout):
    stream.write(text)
    stream.flush()
    return


"""
Custom function to load options from a .yaml file.
"""


def load_yaml(file_name):
    with open(file_name, 'r') as fp:
        cfg = yaml.safe_load(fp)
    cfg = make_recursive_attr_dict(cfg)

    # Also fix for backward compatibility
    fix_for_backward_compatibility(cfg)
    return cfg


def fix_for_backward_compatibility(options, cmptbl_dict=DEFAULT_HYPERPARAMS):
    """
    Fixes options for backward compatibility. If options are added
    to the implementation later on, old configuration files can still
    be used by placing these new options into the variable DEFAULT_HYPERPARAMS.
    They will get default values, as specified in this dictionary.
    """
    for key in cmptbl_dict:
        val = cmptbl_dict[key]
        if key not in options:
            if isinstance(val, dict):
                fix_backward_compatibility(
                    options[key], cmptbl_dict=cmptbl_dict[key])
            else:
                options[key] = val


def bound_low(values, bounds):
    """
    Lower-bound values using bounds.
    """
    pos = (np.array(values) >= np.array(bounds))
    return pos * values + (1 - pos) * bounds


def bound_high(values, bounds):
    """
    Upper-bound values using bounds.
    """
    pos = (np.array(values) <= np.array(bounds))
    return pos * values + (1 - pos) * bounds


def get_neighbours(L, c_id):
    """
    Get a volume of thickness +1 around the blob of superpixel
    with ID c_id. This is used to find neighbouring superpixels
    and return the result to make_adjacency_matrix.
    """
    D, H, W = L.shape[:3]
    # Find volume.
    d, h, w = (L == c_id).nonzero()
    # Find bounds of the superpixel along x, y, z.
    dl, dh = d.min(), d.max()
    hl, hh = h.min(), h.max()
    wl, wh = w.min(), w.max()

    # Find bigger volume around the superpixel.
    dl, hl, wl = bound_low([dl - 1, hl - 1, wl - 1], [0, 0, 0])
    dh, hh, wh = bound_high([dh + 1, hh + 1, wh + 1], [D, H, W])
    # Get the IDs of superpixels inside this bigger volume.
    vol = L[dl:dh, hl:hh, wl:wh]
    U = np.unique(vol)

    return U


def make_adjacency_matrix(L, n_jobs=32):
    """
    Make adjacency matrix over the labels (superpixels).
    """
    uniq = np.unique(L)
    inv_map = {}
    for u, i in enumerate(uniq):
        inv_map[u] = i

    # Cluster === superpixel.
    n_clusters = uniq.size
    adj_mat = np.zeros((n_clusters, n_clusters), dtype=np.bool)

    D, H, W = L.shape[:3]
    # Get neighbours for all superpixels.
    with parallel_backend('threading', n_jobs=n_jobs):
        Us = Parallel()(delayed(get_neighbours)(L, c_id) for c_id in uniq)

    # Populate adjacency matrix.
    for c, U in enumerate(Us):
        c_id = uniq[c]
        for u in U:
            if u == c_id:
                continue
            adj_mat[c, inv_map[u]] = True
    # Make adjacency matrix symmetric.
    adj_mat = (adj_mat + adj_mat.T)
    return adj_mat


def get_grey_value_(I, L, c):
    """
    Helper function that computes the grey value for superpixel c.
    """
#    where                       = (L == c).nonzero()
    where = np.where(L == c)
    px = I[where].mean()
#    M                           = (L == c)
#    px                          = (I * M).sum() / (M.sum() * 1.0)
    return px


def get_grey_values(I, L, n_jobs=32):
    # Get average grey values for every superpixel, and call it the grey value
    # for the superpixel
    n_superpixels = np.unique(L).size
    I_ = I.astype(np.float32).ravel()
    L_ = L.ravel()
    # Parallelise over multiple CPUs.
    with parallel_backend('threading', n_jobs=n_jobs):
        grey_values = Parallel()(delayed(get_grey_value_)(I_, L_, c)
                                 for c in range(n_superpixels))

    return np.array(grey_values)


def compute_edge_contr(m, grey_values, N):
    zm = grey_values[m]
    neighs = [m + 1 + n for n in np.where(N == 1)[0]]
    if len(neighs) == 0:
        return [0.0, 0.0]

    zn = grey_values[neighs]
    n_edges = len(neighs)
    total_m = np.sum((zn - zm) ** 2)
    return [total_m * 1.0, n_edges * 1.0]


def compute_beta(grey_values, adj_mat, n_jobs=32):
    # Get beta from grey values and adjacency matrix using the expression
    # beta = 1 / (2 * E[(zm - zn)**2])
    # where zm and zn are values of superpixels m, n for all neighbouring
    # superpixels m, and n
    n_superpixels = adj_mat.shape[0]

    with parallel_backend('threading', n_jobs=n_jobs):
        w_l_contr = Parallel()(delayed(compute_edge_contr)(
            m, grey_values, adj_mat[m, m + 1:]) for m in range(n_superpixels - 1))

    w_l_contr = np.array(w_l_contr)
    exp_sq_diff = w_l_contr[:, 0].sum() / w_l_contr[:, 1].sum()
    beta = 1. / (2 * exp_sq_diff)
    return beta


def get_pairwise_costs(
        grey_values,
        px_locations,
        adj_mat,
        beta,
        imsize,
        sigma=10):
    edge_list = []
    n_superpixels = adj_mat.shape[0]

    D, H, W = imsize

    costs = []

    for m in range(n_superpixels - 1):
        neighs = np.array(
            [m + 1 + n for n in np.where(adj_mat[m, m + 1:] == 1)[0]])

        xm, ym, zm = px_locations[m, 0], px_locations[m, 1], px_locations[m, 2]

        for n in neighs:
            xn, yn, zn = px_locations[n,
                                      0], px_locations[n, 1], px_locations[n, 2]

            mult = np.exp(-beta * \
                          np.sum((grey_values[m] - grey_values[n])**2) / (2 * sigma**2))
            dist = np.sqrt((xm - xn) ** 2 + (ym - yn) ** 2 + (zm - zn) ** 2)
            assert dist > 0, 'dist = 0 for m = %d [%d, %d], n = %d [%d, %d]' % (
                m, ym, xm, n, yn, xn)
            cost = mult * (1. / dist)

            edge_list.append([m, n])
            costs.append(cost)

    return edge_list, costs


def place_gaussian(X, Y, sigmax=5, sigmay=5):
    mult = 1
    cx = np.int(X[0, :].max()) // 2
    cy = np.int(Y[:, 0].max()) // 2
    G = np.exp(-(X - cx)**2 / (2 * sigmax**2) - (Y - cy)**2 / (2 * sigmay**2))
    return G, cx, cy


def place_gaussians(fg, sigmax=5, sigmay=5):
    pos = np.where(fg > 0)
    n_pos = pos[0].size

    G = np.zeros(fg.shape, dtype=np.float32)
    H, W = fg.shape[:2]
    X, Y = np.meshgrid(range(2 * W + 1), range(2 * H + 1))
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    bell, mx, my = place_gaussian(X, Y, sigmax=sigmax, sigmay=sigmay)

    for i in range(n_pos):
        ty, lx = pos[0][i], pos[1][i]
        by, rx = H - ty, W - lx

        part = bell[my - ty:my + by, mx - lx:mx + rx]
        G += part

#        G += place_gaussian(X, Y, x, y, sigmax=sigmax, sigmay=sigmay)
    return G


def get_supervoxel_size(n_supervoxels, volume_size):
    supervoxel_size = int(
        np.floor((np.prod(volume_size) / n_supervoxels)**(1 / 3)))
    logger.debug((
        f'\n{n_supervoxels} requested, '
        f'leads to supervoxel edge length of {supervoxel_size}'
    ))
    return supervoxel_size
