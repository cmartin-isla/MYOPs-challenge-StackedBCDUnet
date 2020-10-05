"""
Post-process output of neural network corresponding to automatic segmentation of (1) scar and edema, (2) scar so
that the automatic segmentations respect certain physiological constrains

Input:
           - path to nifti files with scar and myo segmentation. The following files are required:
               - nifti files with filename myops_training_*_MY.nii containing the myocardium segmentation
               - nifti files with filename myops_training_*_SE.nii containing segmentation of interest
            - path to nifti files with scar and edema tissue segmentation:
                - nifti files with filename myops_training_*_SE.nii
            - path to nifti files with mri from the three sequences and groundtruth
               - nifti files with filename myops_training_*_gt.nii containing the groundtruth segmentation to calculate the dice
               - nifti files with filename "_T2.nii.gz", "_CO.nii.gz", "_gd.nii.gz"
            - path for saving results
Output:
           - .csv with 3D dice for the different post-processing methods
           - .nifti files with corrected segmentations
               - scar
               - edema+scar
               - myo

Example run: python postprocess_segmentation.py E:\\Xenia_Projects\\MICCAI_Myops_Challenge\\Data\\18_07_2020\\pred_union_plus_scar\\preds_dil_union\\preds_dil\\

Author: Xenia Gkontra, UB, 2020
"""

import argparse
import glob
import os
import pdb
import re
import nibabel as nib
import pandas as pd
from main_postprocess import main_postprocessing_scar, main_postprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post-process segmentations masks.')
    parser.add_argument("input_path_s", type=str,
                        default='D:\\Xenia\\Myops_challenge\\Dataset\\preds_style\\preds_style\\',
                        nargs='?', help='path to predicted masks for scar tissue')
    parser.add_argument("input_path_se", type=str,
                        default='D:\\Xenia\\Myops_challenge\\Dataset\\preds_style\\preds_style\\',
                        nargs='?', help='path to predicted masks for scar and edema tissue')
    parser.add_argument("input_path_gt", type=str,
                        default='D:\\Xenia\\Myops_challenge\\Dataset\\preds_val\\training\\',
                        nargs='?', help='path to mri and groundthruth segmentation')
    parser.add_argument("output_path", type=str, default='D:\\Xenia\\Myops_challenge\\Results\\15models\\preds_style_2d\\',
                        nargs='?', help='path to output')

    # ==================================================================================================================
    #                                          Parse input and parameters
    # ==================================================================================================================

    args = vars(parser.parse_args())
    # Path to niftis to be processed
    input_path_s = args['input_path_s']
    input_path_se = args['input_path_se']
    # Path to MRI & groundtruth
    input_path = args['input_path_gt']

    # Output path
    output_path = args['output_path']
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # Labels - this could also be provided as input
    myo_label = 200
    edema_label = 1220
    scar_label = 2221
    filename_part = ['myops_training_', '_pred_se.nii.gz']

    # Initialize dataframes to save the results
    dice_net_s = pd.DataFrame()
    dice_net_myo = pd.DataFrame()
    dice_net_se = pd.DataFrame()
    # ==================================================================================================================
    #                                               Process
    # ==================================================================================================================

    # Process every image found in the input directory with scar tissue
    for name in glob.glob(os.path.join(input_path_s, filename_part[0] + '*' + filename_part[1])):
        dice_s_value, dice_se_value, dice_myo_value, filename = main_postprocess(input_path, input_path_s, input_path_se, name, filename_part, output_path, myo_label, edema_label, scar_label)
        dice_net_s[filename] = dice_s_value
        dice_net_se[filename] = dice_se_value
        dice_net_myo[filename] = dice_myo_value
        print("Processed subject: ", filename)
     #   pdb.set_trace()
    # ==================================================================================================================
    #                                              Save results
    # ==================================================================================================================
    # Write .csv with results for myo
    dice_net_myo_init = dice_net_myo.rename(index={0: 'Net', 1: 'Con'})
    dice_net_myo = dice_net_myo_init.copy()
    dice_net_myo['Mean'] = dice_net_myo_init.mean(axis=1)
    dice_net_myo['Std'] = dice_net_myo_init.std(axis=1, ddof=1)
    dice_net_myo.to_csv(os.path.join(output_path, 'Dice_results_myo.csv'))
    # Write .csv with results for scar
    dice_net_s_init = dice_net_s.rename(index={0: 'Net', 1: 'Convex hull'})
    dice_net_s = dice_net_s_init.copy()
    dice_net_s['Mean'] = dice_net_s_init.mean(axis=1)
    dice_net_s['Std'] = dice_net_s_init.std(axis=1, ddof=1)
    dice_net_s.to_csv(os.path.join(output_path, 'Dice_results_scar.csv'))
    # Write .csv with results for scar+edema
    dice_net_se_init = dice_net_se.rename(index={0: 'Net', 1: 'Corrected using myo', 2: 'Corrected with scar'})
    dice_net_se = dice_net_se_init.copy()
    dice_net_se['Mean'] = dice_net_se_init.mean(axis=1)
    dice_net_se['Std'] = dice_net_se_init.std(axis=1, ddof=1)
    dice_net_se.to_csv(os.path.join(output_path, 'Dice_results_edema.csv'))