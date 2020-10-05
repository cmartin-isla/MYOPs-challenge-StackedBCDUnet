# -*- coding: utf-8 -*-
# Authors:
# Víctor M. Campello (vicmancr@gmail.com)
# inspired by github.com/baumgach/acdc_segmenter.git

import nibabel as nib
import numpy as np
import argparse
import logging
import h5py
import glob
import sys
import gc
import re
import os

from skimage import transform
from PIL import Image

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/'.join(dir_path.split('/')[:-2]))
sys.path.append('../..')

import config.system as config

from utils import utils_gen, utils_nii, image_utils
from data.dataset import Dataset

parser = argparse.ArgumentParser(
    description="Prepare dataset and save in hdf5 format for training.")
data_folders = [n for n in os.listdir(config.data_base) \
                if os.path.isdir(os.path.join(config.data_base, n))]
parser.add_argument('-d', '--data', type=str, 
    help="Name of dataset to use. Possible datasets: {0}".format(data_folders))
parser.add_argument('-l', '--n_labels', type=int, 
    help="max number of unique labels")


# Dictionary to translate a diagnosis into a number
# NOR  - Normal
# MINF - Previous myiocardial infarction (EF < 40%)
# DCM  - Dilated Cardiomypopathy
# HCM  - Hypertrophic cardiomyopathy
# RV   - Abnormal right ventricle (high volume or low EF)
diagnosis_dict = {'NOR': 0, 'MINF': 1, 'DCM': 2, 'HCM': 3, 'RV': 4}

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5


def crop_or_pad_slice_to_size(slice, nx, ny):

    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped


def prepare_data(dataset, output_file, subset, mode, size, target_resolution,n_labels, split_test_train=True):
    '''
    Main function that prepares a dataset from the raw ACDC data to an hdf5 dataset
    '''
    hdf5_file = h5py.File(output_file, "w")

    diag_list = {'test': [], 'train': []}
    height_list = {'test': [], 'train': []}
    weight_list = {'test': [], 'train': []}
    patient_id_list = {'test': [], 'train': []}
    cardiac_phase_list = {'test': [], 'train': []}

    file_list = {'test': [], 'train': []}
    num_slices = {'test': 0, 'train': 0}

    logging.info('Counting files and parsing meta data...')
    data = Dataset(dataset, subset, mode, size, target_resolution)
    for idx, volume in enumerate(data):
        

        split_test_train = True
        if split_test_train: # Select for test 1 for every 5 images
            train_test = 'test' if (int(volume.patient_id) % 3 == 0) else 'train'
        else:
            train_test = 'train'

        file_list[train_test].append((volume.filepath_DE,volume.filepath_C0,volume.filepath_T2, volume.maskpath))

        if 'Group' in volume.info.keys():
            diag_list[train_test].append(diagnosis_dict[volume.info['Group']])
        if 'Weight' in volume.info.keys():
            weight_list[train_test].append(volume.info['Weight'])
        if 'Height' in volume.info.keys():
            height_list[train_test].append(volume.info['Height'])

        patient_id_list[train_test].append(volume.patient_id)

        frame = volume.phase
        if frame == 'ES':
            cardiac_phase_list[train_test].append(1)  # 1 == systole
        elif frame == 'ED':
            cardiac_phase_list[train_test].append(2)  # 2 == diastole
        else:
            cardiac_phase_list[train_test].append(0)  # 0 means other phase

        num_slices[train_test] += volume.shape[2]
        # num_slices[train_test] += 5
        # num_slices[train_test] += 1

    # Write the small datasets
    for tt in ['test', 'train']:
        hdf5_file.create_dataset('diagnosis_{}'.format(tt), data=np.asarray(diag_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('weight_{}'.format(tt), data=np.asarray(weight_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('height_{}'.format(tt), data=np.asarray(height_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('patient_id_{}'.format(tt), data=np.asarray(patient_id_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('cardiac_phase_{}'.format(tt), data=np.asarray(cardiac_phase_list[tt], dtype=np.uint8))

    
    if mode == '2D':
        nx, ny, nz = size
        n_train = num_slices['train']
        n_test = num_slices['test']
    else:
        raise AssertionError('Wrong mode setting. This should never happen.')

    # Create datasets for images and masks
    data = {}
    for tt, num_points in zip(['test', 'train'], [n_test, n_train]):

        if num_points > 0:
                
            data['data_{}'.format(tt)] = hdf5_file.create_dataset('data_{}'.format(tt), [num_points] + list(size), dtype=np.float32)
            data['pred_{}'.format(tt)] = hdf5_file.create_dataset('pred_{}'.format(tt), [num_points] + list(size)[:2], dtype=np.uint8)
            data['z_{}'.format(tt)] = hdf5_file.create_dataset('z_{}'.format(tt), [num_points], dtype=np.uint8)
            data['n_{}'.format(tt)] = hdf5_file.create_dataset('n_{}'.format(tt), [num_points], dtype=np.uint8)

    
    print('d_np',data,num_points)
    mask_list = {'test': [], 'train': [] }
    img_list = {'test': [], 'train': [] }
    z_list = {'test': [], 'train': [] }
    n_list = {'test': [], 'train': [] }


    logging.info('Parsing image files')

    train_test_range = ['test', 'train'] if split_test_train else ['train']
    for train_test in train_test_range:

        write_buffer = 0
        counter_from = 0

        for f_DE,f_C0,f_T2, m in file_list[train_test]:

            logging.info('-----------------------------------------------------------')
            logging.info('Doing: {}'.format(f_DE))


            img_dat_DE = utils_nii.load_nii(f_DE)
            img_dat_T2 = utils_nii.load_nii(f_T2)
            img_dat_C0 = utils_nii.load_nii(f_C0)

            if volume.info['type'] == '4D_img+ED_ES_seg':
                mask_dat = [utils_nii.load_nii(m[0]),utils_nii.load_nii(m[1])]
            else: 
                mask_dat = utils_nii.load_nii(m)

            img_DE = img_dat_DE[0].copy()
            img_T2 = img_dat_T2[0].copy()
            img_C0 = img_dat_C0[0].copy()
            mask = mask_dat[0].copy()

            
            print(np.array([mask[...,0],mask[...,0],mask[...,0]]).T.shape)
            """
            import matplotlib.pyplot as plt
            
            if 'myops_training_101_DE' in f_DE:
                
                multichannel = np.hstack((np.array([crop_or_pad_slice_to_size(img_DE[...,0]/np.max(img_DE[...,0]),200,200).T,
                                                    crop_or_pad_slice_to_size(img_C0[...,0]/np.max(img_C0[...,0]),200,200).T,
                                                    crop_or_pad_slice_to_size(img_T2[...,0]/np.max(img_T2[...,0]),200,200).T]).T,
                                     np.array([(crop_or_pad_slice_to_size(mask[...,0],200,200)/np.max(mask)*1.5).T,
                                               (crop_or_pad_slice_to_size(mask[...,0],200,200)/np.max(mask)*1.5).T,
                                               (crop_or_pad_slice_to_size(mask[...,0],200,200)/np.max(mask)*1.5).T]).T))
                
                DE = crop_or_pad_slice_to_size(img_DE[...,0],200,200)
                C0 = crop_or_pad_slice_to_size(img_C0[...,0],200,200)
                T2 = crop_or_pad_slice_to_size(img_T2[...,0],200,200)
                three_modalities = np.hstack((DE/np.max(DE),C0/np.max(C0),T2/np.max(T2)))
                plt.imshow(multichannel)
                plt.show()
            """

            for id,v in enumerate(np.unique(mask)):
                mask[mask==v] = id
            
           
            pixel_size = (1,1,1)
 
            if mode == '2D':
                
                scale_vector = [pixel_size[0] / target_resolution[0], pixel_size[1] / target_resolution[1]]

                nn = img_DE.shape[2]
                    
                for zz in range(nn):
                    

                    slice_img_DE = np.squeeze(img_DE[:, :, zz])
                    
                    slice_rescaled_DE = transform.rescale(slice_img_DE,
                                                       scale_vector,
                                                       order=1,
                                                       preserve_range=True,
                                                       multichannel=False,
                                                       mode = 'constant')
                    
                    
                    
                    slice_img_C0 = np.squeeze(img_C0[:, :, zz])                    
                    slice_rescaled_C0 = transform.rescale(slice_img_C0,
                                                       scale_vector,
                                                       order=1,
                                                       preserve_range=True,
                                                       multichannel=False,
                                                       mode = 'constant')
                    
                    
                    slice_img_T2 = np.squeeze(img_T2[:, :, zz])                   
                    slice_rescaled_T2 = transform.rescale(slice_img_T2,
                                                       scale_vector,
                                                       order=1,
                                                       preserve_range=True,
                                                       multichannel=False,
                                                       mode = 'constant')
                    
                    

                    slice_mask = np.squeeze(mask[:, :, zz])

                    mask_rescaled = transform.rescale(slice_mask,
                                                      scale_vector,
                                                      order=0,
                                                      preserve_range=True,
                                                      multichannel=False,
                                                      mode='constant')
                
                    
                    
                


                    slice_cropped_DE = crop_or_pad_slice_to_size(slice_rescaled_DE, nx, ny)
                    slice_cropped_T2 = crop_or_pad_slice_to_size(slice_rescaled_T2, nx, ny)
                    slice_cropped_C0 = crop_or_pad_slice_to_size(slice_rescaled_C0, nx, ny)
                    
                    
                    mask_cropped = crop_or_pad_slice_to_size(mask_rescaled, nx, ny)
                    
                    slice_cropped_DE = slice_cropped_DE/np.max(slice_cropped_DE)
                    slice_cropped_T2 = slice_cropped_T2/np.max(slice_cropped_T2)
                    slice_cropped_C0 = slice_cropped_C0/np.max(slice_cropped_C0)
                    img_list[train_test].append(np.array([slice_cropped_DE.T,slice_cropped_T2.T,slice_cropped_C0.T]).T)
                    mask_list[train_test].append(mask_cropped)
                    z_list[train_test].append(int(f_DE.split('_')[-2]))
                    print('______________________________',f_DE.split('_')[-2])

                    write_buffer += 1

                    # Writing needs to happen inside the loop over the slices
                    if write_buffer >= MAX_WRITE_BUFFER:
                        counter_to = counter_from + write_buffer

                        _write_range_to_hdf5(data, train_test, img_list, mask_list, counter_from, counter_to, z_list)
                        _release_tmp_memory(img_list, mask_list, train_test, z_list)

                        # reset stuff for next iteration
                        counter_from = counter_to
                        write_buffer = 0
                        
            

        # after file loop: Write the remaining data

        logging.info('Writing remaining data')
        counter_to = counter_from + write_buffer
        _write_range_to_hdf5(data, train_test, img_list, mask_list, counter_from, counter_to,z_list)
        _release_tmp_memory(img_list, mask_list, train_test,z_list)


    # After test train loop:
    hdf5_file.close()


def _write_range_to_hdf5(hdf5_data, train_test, img_list, mask_list, counter_from, counter_to, z_list = {'train': [],'test': [] }):
    '''
    Helper function to write a range of data to the hdf5 datasets
    '''

    import matplotlib.pyplot as plt
    """
    for img,msk in zip(img_list['train'] + img_list['test'],mask_list['train'] + mask_list['test']):
        
        img = (img - np.min(img))/np.max((img - np.min(img)))*255
        img = (img / 16).astype(int)*16
        plt.imshow(np.hstack((img,msk/np.max(msk)*255)))
        plt.show()
        
        #image_utils.detect_circle(img)
    """

    logging.info('Writing data from {0} to {1}'.format(counter_from, counter_to))

    img_arr = np.asarray(img_list[train_test], dtype=np.float32)
    mask_arr = np.asarray(mask_list[train_test], dtype=np.uint8)
    
    """
    for msk in range(mask_arr.shape[0]):
    

        plt.imshow(mask_arr[msk,...])
        plt.show()
    """


    hdf5_data['data_{}'.format(train_test)][counter_from:counter_to, ...] = img_arr
    hdf5_data['pred_{}'.format(train_test)][counter_from:counter_to, ...] = mask_arr
    hdf5_data['z_{}'.format(train_test)][counter_from:counter_to, ...] = z_list[train_test]


def _release_tmp_memory(img_list, mask_list, train_test,z_list = {'train': [],'test': [] }):
    '''
    Helper function to reset the tmp lists and free the memory
    '''

    img_list[train_test].clear()
    mask_list[train_test].clear()
    z_list[train_test].clear()
    gc.collect()



if __name__ == "__main__":

    args = parser.parse_args()
    dataset = args.data
    n_labels = args.n_labels
    if '-' in dataset:
        datasets = dataset.split('-')
        logging.info('Evaluating more than one dataset at the same time: {0}'.format(datasets))
        for ds in datasets:
            assert ds in data_folders, 'Dataset "{0}" not available. \
                Please, choose one of the following: {1}'.format(ds, data_folders)
    else:
        assert dataset in data_folders, 'Dataset "{0}" not available. \
                Please, choose one of the following: {1}'.format(dataset, data_folders)

    preprocessing_folder = os.path.join(config.data_base, args.data, 'preproc_data')
    if not os.path.exists(preprocessing_folder):
        os.makedirs(preprocessing_folder)

    split_test_train = True
    mode = '2D'
    #mode = '25D'
    subset = 'training'
    #size = (384, 384,3)
    size = (256, 256, 3)
    target_resolution = (1, 1)
    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])
    if split_test_train:
        data_file_name = 'data_{0}_{1}_size_{2}_res_{3}_onlytrain.hdf5'.format(dataset, mode, size_str, res_str)
    else:
        data_file_name = 'data_{0}_{1}_size_{2}_res_{3}.hdf5'.format(dataset, mode, size_str, res_str)
    output_file = os.path.join(preprocessing_folder, data_file_name)

    database = dataset if '-' not in dataset else datasets
    prepare_data(database, output_file, subset, mode, size, target_resolution,n_labels)
