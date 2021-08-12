from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tsne import estimate_sne, tsne_grad, symmetric_sne_grad, q_tsne, q_joint
from tsne import p_joint
import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
from util.util import tensor2im
import matplotlib.pyplot as plt
import numpy as np


import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from skimage import img_as_ubyte
from skimage.transform import resize
from sklearn.cluster import KMeans
    
    
    
    
def bbox(msk, margin):
    a = np.where(msk != 0)
    bbox = np.min(a[0])-margin, np.max(a[0]) + margin, np.min(a[1])- margin, np.max(a[1]) + margin
    return bbox

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

def crop_or_pad_bbox(img,msk,margin= 5, crop_or_pad_size = 200):
    
    bb = bbox(msk,margin)
    
    if len(img.shape) == 3:  
        img = img[bb[0]:bb[1],bb[2]:bb[3],...]
        ph = np.zeros((crop_or_pad_size,crop_or_pad_size,3))
        for c in range(3):
            ph[...,c] = img_as_ubyte(crop_or_pad_slice_to_size(img[...,c],crop_or_pad_size,crop_or_pad_size))
        img = ph
    else:
        img = img[bb[0]:bb[1],bb[2]:bb[3]]
    
    msk = crop_or_pad_slice_to_size(msk[bb[0]:bb[1],bb[2]:bb[3]],crop_or_pad_size,crop_or_pad_size)
    
    
    #img = img_as_ubyte(img)
    return img,msk
            
    


# Set global parameters
NUM_POINTS = 50            # Number of samples from MNIST
CLASSES_TO_USE = [0]  # MNIST classes to use
PERPLEXITY = 20
SEED = 1                    # Random seed
MOMENTUM = 0.9
LEARNING_RATE = 10.
NUM_ITERS = 500             # Num iterations to train for
TSNE = False                # If False, Symmetric SNE
NUM_PLOTS =5             # Num. times to plot in training






def main():
    
    
    
    
       
            
            
              
    ds = './datasets/myops_ds/data_myops_2D_size_212_212_3_res_1_1_onlytrain.hdf5'
    
    
    data = h5py.File(ds, 'r')
    
    images_train = data['data_train']
    labels_train = data['pred_train']
    
    images_val = data['data_test']
    labels_val = data['pred_test']
    
    
    processed_images_train = []
    processed_masks_train = []
    
    processed_images_val = []
    processed_masks_val = []
    
    for img in range(images_train.shape[0]):
     
        test_crop_img,test_crop_msk =  crop_or_pad_bbox(images_train[img,...],labels_train[img,...])
    
        im = test_crop_img.astype('uint8')
        msk = test_crop_msk.astype('uint8')
        processed_images_train.append(np.ravel(im[...,0]))
        processed_masks_train.append(msk)
        
        
    for img in range(images_val.shape[0]):
    
        test_crop_img,test_crop_msk =  crop_or_pad_bbox(images_val[img,...],labels_val[img,...])
        
        im = test_crop_img.astype('uint8')
        msk = test_crop_msk.astype('uint8')
        processed_images_val.append(np.ravel(im[...,0]))
        processed_masks_val.append(msk)
    
    msks_resized_train = [resize(img,(28,28),order = 0,preserve_range=True, anti_aliasing=False) for img in processed_masks_train]
    msks_resized_val= [resize(img,(28,28),order = 0,preserve_range=True, anti_aliasing=False) for img in processed_masks_val]
    
    
    
    X = np.array([np.ravel(msk) for msk in msks_resized_train])
    kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
    
    fit = kmeans.fit(X)
    y = fit.labels_
    centroids = y = fit.centroids
    print()
    
    
    
    # numpy RandomState for reproducibility
    rng = np.random.RandomState(SEED)

    # Obtain matrix of joint probabilities p_ij
    P = p_joint(X, PERPLEXITY)

    # Fit SNE or t-SNE
    Y = estimate_sne(X, y, P, rng,
                     num_iters=NUM_ITERS,
                     q_fn=q_tsne if TSNE else q_joint,
                     grad_fn=tsne_grad if TSNE else symmetric_sne_grad,
                     learning_rate=LEARNING_RATE,
                     momentum=MOMENTUM,
                     plot=NUM_PLOTS)

if __name__ == "__main__":
    

    main()