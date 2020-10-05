# -*- coding: utf-8 -*-
# Authors:
# Víctor M. Campello (vicmancr@gmail.com)

import os,sys
import re
import glob
import importlib
import numpy as np
import nibabel as nib

from skimage import transform
from PIL import Image

from utils import image_utils


class Slice(object):
    '''
    Class for a 2D slice.
    '''
    def __init__(self, image, mask, scale_vector, target_size=None):
        '''
        Constructor. It accounts for image, mask and properties.
        Parameters:
            image: Image data.
            mask: Mask data.
        '''
        self.img          = image
        self.mask         = mask
        self.scale_vector = scale_vector
        if target_size is not None:
            self.target_size  = target_size
        else:
            self.target_size = self.img.shape
        self.rescale_and_crop()

    def rescale_and_crop(self):
        '''Rescale and crop 2D slice.'''
        slice_img = np.squeeze(self.img)
        slice_rescaled = transform.rescale(slice_img, self.scale_vector, order=1,
                                           preserve_range=True, mode='constant')

        self.shape = slice_rescaled.shape
        x, y = self.shape
        nx, ny = self.target_size[:2]

        x_s = (x - nx) // 2
        y_s = (y - ny) // 2
        x_c = (nx - x) // 2
        y_c = (ny - y) // 2
        self.cropped_boundaries = (x_s, y_s, x_c, y_c)

        # Crop section of image for prediction
        if x > nx and y > ny:
            slice_cropped = slice_rescaled[x_s:x_s+nx, y_s:y_s+ny]
        else:
            slice_cropped = np.zeros((nx,ny))
            if x <= nx and y > ny:
                slice_cropped[x_c:x_c+ x, :] = slice_rescaled[:,y_s:y_s + ny]
            elif x > nx and y <= ny:
                slice_cropped[:, y_c:y_c + y] = slice_rescaled[x_s:x_s + nx, :]
            else:
                slice_cropped[x_c:x_c+x, y_c:y_c + y] = slice_rescaled[:, :]

        self.img_cropped = slice_cropped


class Volume(object):
    '''
    Class for a single volume.
    '''
    def __init__(self, fileinfo, image_size=None, image_resolution=None, pid=None):
        '''
        Constructor. It defines the configuration for the volume.
        Parameters:
            filepath: Path to volume.
            maskpath: Path to mask if it exists.
            image_size: Desired final image size for cropping.
                If None, original size is preserved.
            image_resolution: Desired final image resolution.
                If None, image is not rescaled.
        '''

        self.filepath_DE   = fileinfo['image_DE']
        self.filepath_C0   = fileinfo['image_C0']
        self.filepath_T2   = fileinfo['image_T2']
        self.maskpath   = fileinfo['mask'] if 'mask' in fileinfo.keys() else [fileinfo['mask_ED'],fileinfo['mask_ES']]
        self.patient_id = fileinfo['patient_id'] 
        self.info       = fileinfo['info']
        self.phase      = fileinfo['phase'] if 'phase' in fileinfo.keys() else 'unk'
        self.image_size = image_size
        self.image_resolution = image_resolution
        self.slices  = []
        self.current = 0
        self.img_dat = []
        self.mask_dat = []
        
        if pid is not None:
            self.patient_id = pid
            

        self.shape = nib.load(self.filepath_DE).get_data().shape

    def __iter__(self):
        return self

    def __next__(self):
        '''Returns next file.'''
        if self.current == 0: # Initialize when called
            self.process_slices()
            self.high = len(self.slices) - 1
        if self.current > self.high:
            self.slices = [] # Clear files
            self.img_dat = []
            self.mask_dat = []
            raise StopIteration
        else:
            self.current += 1
            return self.slices[self.current-1]

    def process_slices(self):
        '''Process slices of 3D images.'''
        if '.png' in self.filepath:
            img = np.zeros(self.shape)
            mask = np.zeros(self.shape, dtype=np.uint8)
            for idx, slc in enumerate(sorted(glob.iglob(self.filepath))):
                img[...,idx] = np.asarray(Image.open(slc).convert('L'))
                #sp = slc.split('frame')
                #mskname = re.sub(r'img', 'msk', sp[0]) + 'frame' + sp[1][:2] + '_gt' + sp[1][2:]
                mskname = self.maskpath
                if os.path.exists(mskname):
                    mask[...,idx] = (np.asarray(Image.open(mskname).convert('L'))/256*4).astype(int)
                if os.path.exists(mskname.split('.')[0]+'_gt.png'):
                    mask[...,idx] = (np.asarray(Image.open(mskname.split('.')[0]+'_gt.png').convert('L'))/256*4).astype(int)
            #self.img_dat = nimg.get_data(), nimg.affine, nimg.header
            
            self.img_dat = img, np.eye(4), None
            img = self.img_dat[0].copy()
            img = image_utils.normalise_image(img, 0.5, 0.5)
            self.mask_dat = mask, np.eye(4), None
            #import matplotlib.pyplot as plt;plt.imshow(np.squeeze(self.mask_dat[0]));plt.show()
            
            mask = self.mask_dat[0]        
            """
            t = re.sub(r'full_mesh_acdc', 'miccai2017', self.filepath)
            patient_id = self.filepath.split('_')[0][-3:]
            t = re.sub(r'img', 'patient{}'.format(patient_id), t)
            niif = '_'.join(t.split('_')[:2]) + '.nii.gz'
            niim = '_'.join(t.split('_')[:2]) + '_gt.nii.gz'
            nimg = nib.load(niif)
            self.img_dat = img, nimg.affine, nimg.header
            nimg = nib.load(niim)
            self.mask_dat = mask, nimg.affine, nimg.header
            """
            pixel_size = (1,1)

        else:
            nimg = nib.load(self.filepath)
            self.img_dat = nimg.get_data(), nimg.affine, nimg.header
            img = self.img_dat[0].copy()
            img = image_utils.normalise_image(img, 0.5, 0.5)

            if self.maskpath != '':
                nimg = nib.load(self.maskpath)
                self.mask_dat = nimg.get_data(), nimg.affine, nimg.header
                mask = self.mask_dat[0]
            else:
                mask = np.zeros(img.shape)

            pixel_size = (self.img_dat[2].structarr['pixdim'][1], self.img_dat[2].structarr['pixdim'][2])
        if self.image_resolution is not None:
            scale_vector = (pixel_size[0] / self.image_resolution[0], pixel_size[1] / self.image_resolution[1])
        else:
            scale_vector = (1, 1)
        self.rescale_and_crop(img, mask, scale_vector)

    def rescale_and_crop(self, img, mask, scale_vector):
        '''Rescale and crop slices of a 3D image.'''
        for zz in range(img.shape[2]):
            new_slc = Slice(img[:,:,zz], mask[:,:,zz], scale_vector, self.image_size)
            self.slices.append(new_slc)


class Dataset(object):
    '''
    Class for handling data files for a given dataset.
    '''
    def __init__(self, dataset_name, subset='training', mode='2D', image_size=None, image_resolution=None):
        '''
        Constructor. It defines the configuration for the dataset handler.
        Parameters:
            data_base: Path to datasets.
            dataset_name: The dataset name to consider.
            subset: Set to return: train or test.
            mode: Type of data: 2D or 3D.
            image_size: Desired final image size for cropping.
            image_resolution: Desired final image resolution.
        '''
        self.image_size = image_size
        self.image_resolution = image_resolution
        if isinstance(dataset_name, str):
            module = importlib.import_module('data.{0}.{0}'.format(dataset_name))
            handler = getattr(module, dataset_name[0].upper() + dataset_name[1:])
            generator = handler(subset)
            # File dictionaries containing data images.
            self.files = generator.get_files()

        else:
            self.files = []
            for db in dataset_name:
                module = importlib.import_module('data.{0}.{0}'.format(db))
                handler = getattr(module, db[0].upper() + db[1:])
                generator = handler(subset)
                # File dictionaries containing data images.
                self.files += generator.get_files()
        self.volumes = []
        self.process_volumes()
        self.high    = len(self.volumes) - 1
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        '''Returns next file.'''
        if self.current > self.high:
            raise StopIteration
        else:
            self.current += 1
            return self.volumes[self.current-1]

    def process_volumes(self):
        '''Process slices of 3D images.'''
        
        files_copy = []
        for f in self.files:
                
            try:
                print('Creating dataset:',f)
                self.volumes.append(Volume(f, self.image_size, self.image_resolution))
                files_copy.append(f)
            except Exception:
                print(sys.exc_info()[1])
                print('Errors in ', f['image_DE'])
                
        self.files =  files_copy
                    
