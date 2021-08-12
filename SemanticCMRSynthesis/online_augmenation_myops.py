"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import sys
sys.path.append('..')
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from collections import OrderedDict

import data
from util.image_utils import *
from util.util import tensor2im
from util.mesh_deform import *
import numpy as np
import torch
import h5py
import random
import matplotlib.pyplot as plt
import joblib




def get_latent_tensors(labels_train,imgs_train):
    #get the style tensors
    latent_tensors = []
    with torch.no_grad():
        for img,lbl in zip(imgs_train,labels_train):
            label_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(lbl,axis=0),axis=0))
            img_tensor = torch.from_numpy(np.zeros((1, 3, 256, 256),dtype=np.float32))
            inst_tensor = torch.from_numpy(np.zeros([0],dtype=np.float32))
    
            
            data_i = {'image':img_tensor,
                      'label':label_tensor,
                      'instance':inst_tensor,
                      'path':None} 
            
    
            generated = model(data_i, mode='encode_only')
            latent_tensors.append(generated[0])
    print('Generated style latent tensors for {0} subjects...'.format(labels_train.shape[0]))
    return latent_tensors
        

"""
opt = TestOptions().parse()
model = Pix2PixModel(opt)
joblib.dump(model, 'spage_generator.myops')
"""

model = joblib.load('spade_generator.myops')





model.eval()

ds = 'datasets/myops_ds/original_dataset.hdf5'
style_dataset = h5py.File(ds, 'r')
labels_train = style_dataset['pred_train_my']
imgs_train = style_dataset['data_train']


latent_tensors = get_latent_tensors(labels_train,imgs_train)


def label2SPADEimg(lbl,rot=True,warp =True,dil_ero=True,change_labels = False):
    
    global latent_tensors,labels_train
    
    print('label2SPADEimg')
    if np.random.randint(3) > 0: # augment with 2/3 chance, if not, only style transfer
        try:
            if rot:
                degrees = np.random.randint(45,275)
                lbl = label_rotation(lbl,degrees)
            
            if dil_ero:
                lbl = dilate_erode_scar(lbl,np.random.randint(3),np.random.randint(2,5))
                
            
            if warp and np.random.randint(3) == 2 : #warp with chance 1/3 
                
                    subject = np.random.randint(len(latent_tensors))
                    _,lbl=adapt_contour_full(lbl,lbl,labels_train[subject,...],lbl)
        except:
                print('SPADE unable to rotate, dilate or warp... just applying style style transfer')
    

    
    lbl = add_bb_label(lbl)
    

    
    label_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(lbl,axis=0),axis=0))
    img_tensor = torch.from_numpy(np.zeros((1, 3, 256, 256),dtype=np.float32))
    inst_tensor = torch.from_numpy(np.zeros([0],dtype=np.float32))
    

    
    data_i = {'image':img_tensor,
          'label':label_tensor,
          'instance':inst_tensor,
          'path':None} 
    
    style = np.random.randint(len(latent_tensors))
    
    with torch.no_grad():
        fake_img = model(data_i, mode='decode_only', z = latent_tensors[style]*5)
        
    fake_img = np.squeeze(tensor2im(fake_img))
    lbl[lbl == 6] = 0
    
    if change_labels:
        lbl = map_labels(lbl,[1,2,3,4,5],[0,0,0,1,2])
    
    return lbl,fake_img/np.max(fake_img)




ds = 'datasets/myops_ds/original_dataset.hdf5'
style_dataset = h5py.File(ds, 'r')
labels_train = style_dataset['pred_train_my']
imgs_train = style_dataset['data_train']


np.random.seed(3)
random.seed(3)

for x in range(70):
    
    s = np.random.randint(70)
    subject_msk = labels_train[s,...]
    subject_img = imgs_train[s,...]
    fake_lbl,fake_img = label2SPADEimg(subject_msk,change_labels=True)
    subject_msk = map_labels(subject_msk,[1,2,3,4,5,6],[0,0,0,1,2,0])
    r1 = np.hstack((subject_msk/2,subject_img[...,0],subject_img[...,1],subject_img[...,2]))
    r2 = np.hstack((fake_lbl/2,fake_img[...,0],fake_img[...,1],fake_img[...,2]))
    plt.imshow(np.vstack((r1,r2)))
    plt.show(block = False)
    plt.pause(0.5)



