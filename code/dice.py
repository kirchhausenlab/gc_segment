import numpy as np
import os
import imageio
import argparse
from utils import *


def dice(true, pred):
    return (true * pred).sum() * 2.0 / (true.sum() + pred.sum())


def jaccard(true, pred):
    return (true * pred).sum() * 1.0 / (true + pred).sum()


def dice_to_jaccard(D):
    return D / (2 - D)


def jaccard_to_dice(J):
    return 2 * J / (1 + J)


def main(sys_string=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--true', type=str, default=None,
                        help='Path to ground truth')
    parser.add_argument('--pred', type=str, default=None,
                        help='Path to prediction')
    parser.add_argument('--z_limits', type=int, nargs='+',
                        default=None, help='Optional limits on the Z.')

    if sys_string is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(sys_string.split(" "))

    assert os.path.exists(
        args.true), 'Ground truth path does not exist. Got {}'.format(args.true)
    assert os.path.exists(
        args.pred), 'Prediction path does not exist. Got {}'.format(args.pred)
    assert (args.z_limits is None) or (len(args.z_limits) == 2),\
        'Invalid z_limits. Got {}'.format(args.z_limits)

    true_vol = []
    align_left('Reading ground truth from {}'.format(args.true))
    for f in sorted(os.listdir(args.true)):
        img_path = os.path.join(args.true, f)
        img_ = imageio.imread(img_path) < 255
        true_vol.append(img_[None, ...])
    true = np.concatenate(true_vol, axis=0)

    if args.z_limits is not None:
        true = true[args.z_limits[0]:args.z_limits[1], :, :]

    write_done()

    pred_vol = []
    align_left('Reading prediction from {}'.format(args.pred))
    for f in sorted(os.listdir(args.pred)):
        img_path = os.path.join(args.pred, f)
        img_ = imageio.imread(img_path) > 0
        pred_vol.append(img_[None, ...])
    pred = np.concatenate(pred_vol, axis=0)
    if args.z_limits is not None:
        pred = pred[args.z_limits[0]:args.z_limits[1], :, :]
    write_done()

    align_left('Computing Dice coefficient')
    dice_score = dice(true, pred)
    write_done()

    print('Dice score is {}'.format(dice_score))
    jaccard_index = dice_to_jaccard(dice_score)
    print('Jaccard index is {}'.format(jaccard_index))

    return true, pred, dice_score, jaccard_index


if __name__ == '__main__':
    main()
