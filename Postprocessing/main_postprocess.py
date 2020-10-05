"""
Set of functions for post-processing

Author: Xenia Gkontra, UB, 2020
"""
import nibabel as nib
import numpy as np
import os
import pdb
import re
from seg_accuracy_metrics import dice_corrected
from skimage.morphology import medial_axis, disk
from scipy import ndimage
import cv2
from skimage.morphology import convex_hull_image, remove_small_objects, binary_closing, disk
from bwmorph import spur
from image_interpolation import interp_shape
import matplotlib.pyplot as plt
#from concave_shapes import concave_snakes, concave_hull_image

def replace_slices_3D(mask):
    """
    Check if  edema appears as full slice and this is neither the first nor the last slice. If this is true,
    replace with the interpolation between previous and next slices
    """
    # Close small gaps

    closed = binary_closing(mask, selem=np.ones((3, 3, 3)))
    Nx, Ny, Nz = closed.shape
    mask_cor = np.zeros((Nx, Ny, Nz))

    if Nz > 3:
        for sl in range(0, Nz):
            # Check if edema is a full circle
            se_skel, skeleton, max_radius = myo_from_skeleton(closed[:, :, sl], 1, 0)
            # Clean spur pixels from skeleton
            if (np.sum(skeleton, axis=None) > 5) & ((sl > 1) & (sl < (Nz - 1))):
                mask_cor[:, :, sl] = interp_shape(closed[:, :, sl - 1], closed[:, :, sl + 1], 0.5).astype(int)
            else:
                mask_cor[:, :, sl] = closed[:, :, sl]
    else:
        mask_cor = mask
    return mask_cor


def create_circle_from_radius(Nx, Ny, radi, cx, cy):
    """
    Create a circle with center cx, cy and radius radi in numpy array of size Nx x Ny
    """
    out2 = np.zeros((Nx, Ny))
    # Create 2d grid
    [x, y] = np.mgrid[:Nx, :Ny]

    # Circle
    out_circle = ((x - cx) ** 2 + (y - cy) ** 2) <= radi ** 2

    return out_circle


def myo_from_skeleton(mask, label_spur, thick_margin):
    """
    Function to reconstruct a circular myocardium based on each skeleton and supposing that its thickness is
    homogeneous at all point and equal to maximum radius+margin
    """
    # Calculate skeleton and distance on the skeleton
    skeleton, skel_dist = medial_axis(mask, return_distance=True)
    # Clean skeleton
    if label_spur == 1:
        skeleton = spur(skeleton)

    # Calculate max radius
    max_radius = np.max(np.array(skel_dist), axis=None)

    # Calculate distance transform in respect with the skeleton
    skel_dist_c = ndimage.distance_transform_edt(1 - skeleton)
 #   pdb.set_trace()
    # Create myocardium from skeleton using the maximum radius
    mask_skel = skel_dist_c <= (max_radius + thick_margin)

    return mask_skel, skeleton, max_radius


def myo_postprocess(myo):
    """
    Function to post-process myocardium so as to create appropriate mask to erase out of myocardium regions for scar and edema
    mask - myocardium groundtruth
    """
    Nx, Ny, Nz = myo.shape
    mask_skel = np.zeros((Nx, Ny, Nz))
    mask_cor = np.zeros((Nx, Ny, Nz))
    mask_cor_se = np.zeros((Nx, Ny, Nz))

    # Process each slice separately
    for sl in range(0, Nz):
        # Create myocardium with the same thickness in all regions using its skeleton and the fact that it is a circular object
        mask_skel[:, :, sl], skel, max_radius = myo_from_skeleton(myo[:, :, sl], 1, 0)

        # Check whether the myocardium is not a full circle and this is not and apex or base slice. If this happens
        # then complete the circle If skel is not closed then no object will be found here - maybe here
        skel2 = spur(np.copy(skel))
        if (np.sum(skel2, axis=None) <= 1) :

            # Find corners
            # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(myo[:, :, sl])

            # Convex hull
            chull = convex_hull_image(myo[:, :, sl])
            # Radius and center of convex hull
            hull_dist = ndimage.distance_transform_edt(chull, return_distances=True)
            radius_circle = np.max(hull_dist, axis=None)
            rx, ry = np.where(hull_dist == radius_circle)

            # Create image with circle
            inner_circle = create_circle_from_radius(Nx, Ny, radius_circle - max_radius, rx[0], ry[0])
            outer_circle = create_circle_from_radius(Nx, Ny, radius_circle, rx[0], ry[0])
            dif_circle = (outer_circle.astype(int) - inner_circle.astype(int)).astype(bool)
         #   pdb.set_trace()
            # Myocardium with the part of the circle on top
            myo_full_circle = dif_circle | myo[:, :, sl]
            # Recalculate skeleton
            myo_full, skel, max_radius = myo_from_skeleton(myo_full_circle, 1, 0)
            myo_full_2, _, _ = myo_from_skeleton(myo_full_circle, 1, 6)
            # pdb.set_trace()
            mask_cor[:, :, sl] = np.copy(myo_full)
            mask_skel[:, :, sl] = np.copy(myo_full_2)
        else:
            mask_skel[:, :, sl], _, _ = myo_from_skeleton(myo[:, :, sl], 1, 6)
            mask_cor[:, :, sl] = np.copy(myo[:, :, sl])


    return mask_skel, mask_cor


def main_postprocessing_scar(t2_init, myo, s_init):
    """
    Function to post-process scar segmentation so as that all components are connected by convex hull, on myocardium
    and components smaller than 300 voxels are eliminated
    """
    print("Starting post-processing of scar segmentation")

    Nx, Ny, Nz = s_init.shape

    # Clean from objects smaller than 300 voxels
    scar = np.copy(s_init).astype(int)
    scar1 = remove_small_objects(np.multiply(np.copy(scar), myo.astype(int)).astype(bool), min_size=100)
 #   scar2 = remove_small_objects(np.multiply(np.copy(scar), myo.astype(int)).astype(bool), min_size=100)
    scar = np.copy(scar1)
    temp = np.zeros((Nx, Ny, Nz))
    # Chech whether the scar is consisting of more that 1 objects, in that case join them
    for sl in range(0, Nz):
        labeled_array, num_features = ndimage.measurements.label(scar[:, :, sl])
        _, _, max_radius = myo_from_skeleton(myo[:, :, sl], 1, 0)
        if num_features > 1:
         #   ch = convex_hull_image(scar[:, :, sl])
            ch = convex_hull_image(scar[:, :, sl]) > 0
            myo_eroded = np.copy(myo[:, :, sl])
            myo_eroded = ndimage.morphology.binary_erosion(myo_eroded, disk(0.20*max_radius))
            tmp = ch.astype(int) + myo_eroded.astype(int) + scar[:, :, sl].astype(int)
            ch = (tmp >= 2).astype(int)
      #      pdb.set_trace()
            # Check if this results in a full circle. If yes ignore this
            temp_ch = np.multiply(ch > 0, myo[:, :, sl])
            _, ch_skel, _ = myo_from_skeleton(temp_ch, 1, 0)
            skel2 = spur(np.copy(ch_skel))
     #       pdb.set_trace()
            if (np.sum(skel2, axis=None) > 20):
                temp[:, :, sl] = ndimage.morphology.binary_closing(s_init[:, :, sl], disk(0.9*max_radius))

            else:
                temp[:, :, sl] = temp_ch
        elif num_features == 1:
            temp[:, :, sl] = ndimage.morphology.binary_closing(scar[:, :, sl], disk(0.9*max_radius))
        else:
            temp[:, :, sl] = scar[:, :, sl]
    # Keep only regions inside the myocardium
    #  pdb.set_trace()
    s_cor = np.multiply(temp, myo)

    return s_cor


def main_postprocessing_scar_edema(mask_myo, se_init):
    """
    Function to post-process scar+edema segmentation so as the mask is on the corrected myocardium
    and components smaller than 300 voxels are eliminated
    """
    print("Starting post-processing of scar+edema segmentation")
    # Keep only parts inside myocardium
    se = np.multiply(mask_myo, se_init)

    # Check whether you need to interpolate
 #   se_cor = replace_slices_3D(se)
    se_cor = remove_small_objects(np.multiply(se, mask_myo).astype(bool), min_size=300)

    return se_cor


def main_postprocess(input_path, input_path_s, input_path_se, name, filename_part, output_path, myo_label, edema_label,
                     scar_label):
    """
    Fucntion to call appropriate ones for post-processing myocardium, scar and edema+scar segmentation and calculate
    respective dice
    """
    # ==============================================================================================================
    #                                               Load input data
    # ==============================================================================================================

    # Get name of the image, i.e from 'myops_training_x_pred_se.nii.gz', get x
    full_name = os.path.basename(name)
    result = re.search(filename_part[0] + '(.*)' + filename_part[1], full_name)
    filename = result.group(1)

    # Load MRIs and groundtruth segmentations
    t2_nib = nib.load(os.path.join(input_path, filename_part[0] + filename + '_T2.nii.gz'))
    t2_nii = np.array(t2_nib.get_data())
    co_nib = nib.load(os.path.join(input_path, filename_part[0] + filename + '_C0.nii.gz'))
    co_nii = np.array(co_nib.get_data())
    de_nib = nib.load(os.path.join(input_path, filename_part[0] + filename + '_DE.nii.gz'))
    de_nii = np.array(de_nib.get_data())
    if os.path.isfile(os.path.join(input_path, filename_part[0] + filename + '_gt_scar.nii.gz')):
        gt_nib = nib.load(os.path.join(input_path, filename_part[0] + filename + '_gt_scar.nii.gz'))
        gt_nii = np.array(gt_nib.get_data())
        gt_s = gt_nii > 0
    # Load predictions for myocardium and scar
    myo_nib = nib.load(os.path.join(input_path_s, filename_part[0] + filename + '_pred_MY.nii.gz'))
    myo_nii = np.array((myo_nib.get_data()).astype(int))
    s_nib = nib.load(os.path.join(input_path_s, filename_part[0] + filename + '_pred_SE.nii.gz'))
    s_nii = np.array((s_nib.get_data()).astype(int))
  #  pdb.set_trace()
    # Load prediction for edema+scar, myocardium is considered to be the same as above
    se_nib = nib.load(os.path.join(input_path_se, filename_part[0] + filename + '_pred_SE.nii.gz'))
    se_nii = np.array((se_nib.get_data()).astype(int))

    myo_nii = myo_nii > 0
    s_nii = s_nii == scar_label
    se_nii = (se_nii == scar_label) | (se_nii == edema_label)
    # gt_s = gt_nii > 0
    # gt_se = (gt_nii == scar_label) | (gt_nii == edema_label)
    # gt_myo = (gt_nii == myo_label) | (gt_nii == scar_label) | (gt_nii == edema_label)
    # ==============================================================================================================
    #                                              Post-process
    # ==============================================================================================================

    # Call function to post-process myocardium mask and create one recreated from the skeleton using the radius
    myo_skel, myo_cor = myo_postprocess(myo_nii)



    # Call main function for edema+scar specific post-processing
    # Delete outside
    # Fill holes
    Nx, Ny, Nz = myo_nii.shape
    temp_myo = np.zeros((Nx, Ny, Nz))
    for sl in range(0, Nz):
        temp_myo[:, :, sl] = ndimage.binary_fill_holes(myo_cor[:, :, sl]).astype(int)
    myo_skel = np.multiply(myo_skel, temp_myo)
    #pdb.set_trace()
    se_cor = main_postprocessing_scar_edema(myo_skel, se_nii)

    # Call main function for scar specific post-processing
    scar_cor = main_postprocessing_scar(t2_nii, myo_cor.astype(int) | se_cor.astype(int), s_nii)


    # Join edema+scar and scar masks
    se_cor_final = (scar_cor.astype(int)) | (se_cor.astype(int))

    # ==============================================================================================================
    #                                           Save niftis
    # ==============================================================================================================

    out_img = nib.Nifti1Image(scar_cor, s_nib.affine, s_nib.header)
    nib.save(out_img, os.path.join(output_path, 'scar_' + filename_part[0] + filename + '.nii.gz'))
    out_img = nib.Nifti1Image(myo_cor, s_nib.affine, s_nib.header)
    nib.save(out_img, os.path.join(output_path, 'myo_' + filename_part[0] + filename + '.nii.gz'))
    out_img = nib.Nifti1Image(se_cor, s_nib.affine, s_nib.header)
    nib.save(out_img, os.path.join(output_path, 'se_' + filename_part[0] + filename + '.nii.gz'))
    out_img = nib.Nifti1Image(se_cor_final, s_nib.affine, s_nib.header)
    nib.save(out_img, os.path.join(output_path, 'se_scar_' + filename_part[0] + filename + '.nii.gz'))
    se_cor_final_nifti = np.multiply(edema_label*se_cor_final, 1-scar_cor).astype(int)
    scar_cor_nifti = (scar_cor > 0).astype(int)*scar_label
    out_img = nib.Nifti1Image(se_cor_final_nifti + scar_cor_nifti, s_nib.affine, s_nib.header)
    nib.save(out_img, os.path.join(output_path, filename_part[0] + filename + '_seg.nii.gz'))
    # ==============================================================================================================
    #                                          Dice calculation
    # ==============================================================================================================

    # Dice net for scar before and after correction
    if os.path.isfile(os.path.join(input_path, filename_part[0] + filename + '_gt_scar.nii.gz')):
        dice_s_all = dice_corrected(s_nii, scar_cor, gt_s)
    else:
        dice_s_all = ''
    # Dice net for myocardium before and after correction
    # dice_myo_all = dice_corrected(myo_nii, myo_cor, gt_myo)
    # # Dice net for edema+scar before and after correction
    # dice_se_all = dice_corrected(se_nii, se_cor, gt_se)
    # # Dice net for edema+scar corrected by scar before and after correction
    # _, dice_cor_se_1 = dice_corrected(se_nii, se_cor_final, gt_se)
    # dice_se_all.append(dice_cor_se_1)

    print(dice_s_all)
    # print(dice_myo_all)
    # print(dice_se_all)

    return dice_s_all, filename
