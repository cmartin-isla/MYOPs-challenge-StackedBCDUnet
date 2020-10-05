"""
Set of functions for evaluation segmentation accuracy

Author: Xenia Gkontra, UB, 2020
"""

import numpy as np

def dice(gt, seg):
    """Function to calculate dice coefficient between two segmenattions
    gt - groundtruth
    seg - segmentation
    """
    # Create image with common pixels on
    common = np.array(gt & seg)
    # True positives
    a = np.sum(common, axis=None)
    b = np.sum(gt, axis=None)
    c = np.sum(seg, axis=None)
    # Dice coefficient
    dice_coef = 2 * a / (b + c)

    return dice_coef

def dice_2d(gt_3d, seg_3d):
    """Function to calculate dice coefficient between two segmenattions
    gt - groundtruth
    seg - segmentation
    """
    dice_coef = []
    # Create image with common pixels on
    Nx, Ny, Nz = gt_3d.shape
    for i in range(0, Nz):
        gt = gt_3d[:, :, i]
        seg = seg_3d[:, :, i]
        if np.sum(gt) == 0 & np.sum(seg) == 0:
            dice_coef.append(1)
        else:
            common = np.array(gt & seg)
            # True positives
            a = np.sum(common, axis=None)
            b = np.sum(gt, axis=None)
            c = np.sum(seg, axis=None)
            # Dice coefficient
            dice_coef.append(2 * a / (b + c))

    dice_coef_mean = np.mean(dice_coef)

    return dice_coef_mean


def dice_corrected(mask1, mask2, gt):
    dice_net_myo = dice(mask1, gt)
    dice_cor_myo_1 = dice(mask2.astype(int), gt)
    # Add both dices in one list
    dice_all = [dice_net_myo, dice_cor_myo_1]

    return dice_all

def dice_corrected_2d(mask1, mask2, gt):
    dice_net_myo = dice_2d(mask1, gt)
    dice_cor_myo_1 = dice_2d(mask2.astype(int), gt)
    # Add both dices in one list
    dice_all = [dice_net_myo, dice_cor_myo_1]

    return dice_all