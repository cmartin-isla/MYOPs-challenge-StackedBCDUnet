import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from skimage import img_as_ubyte
import mesh_deform as md
   

        


          
ds = 'original_dataset.hdf5'


data = h5py.File(ds, 'r')

images_train = data['data_train']
labels_train = data['pred_train']
labels_train_orig = data['pred_train_my']

images_val = data['data_test']
labels_val = data['pred_test']
labels_val_orig = data['pred_train_my']



# Training with warping between subjects
for img in range(images_train.shape[0]):

    for im_warp in range(images_train.shape[0]):
        
        img_from = images_train[img,...]
        msk_from = labels_train_orig[img,...]
        msk_to = labels_train_orig[im_warp,...]
        try:
            img_w, lbl_w = md.adapt_contour_full(img_from,msk_from,msk_to,msk_from)
            


            #plt.imshow(test_crop_img)
            #plt.show()
        
            crop_region = md.bbox_crop(lbl_w,lbl_w,15,256)
            crop_region[lbl_w != 0] = 0
            msk_crop = lbl_w+ crop_region
            img_w[msk_crop ==0,:] = 0

            test_crop_img,test_crop_msk =  img_as_ubyte(img_w[...,:3]),msk_crop

            


            
        except:
            print("unable to interp")
        im = Image.fromarray((test_crop_img).astype('uint8'), 'RGB')
        im.save(os.path.join('train_img',str(img)+'_'+str(im_warp)+".png"))
        im = Image.fromarray(test_crop_msk.astype('uint8'))
        im.save(os.path.join('train_label',str(img)+'_'+str(im_warp)+".png"))
            
            

            
            

    
for img in range(images_train.shape[0],images_train.shape[0]+images_val.shape[0]):

    lbl_w = labels_val_orig[79-img,...]
    img_w = images_val[79-img,...]
    crop_region = bbox_crop(lbl_w,lbl_w,15,256)
    crop_region[lbl_w != 0] = 0
    msk_crop = lbl_w+ crop_region
    img_w[crop_region ==0,:] = 0
    
    test_crop_img,test_crop_msk =  img_as_ubyte(img_w[...,:3]),msk_crop
    

    
    im = Image.fromarray((test_crop_img).astype('uint8'), 'RGB')
    im.save(os.path.join('val_img',str(img)+'_'+str(img)+".png"))
    im = Image.fromarray(test_crop_msk.astype('uint8'))
    im.save(os.path.join('val_label',str(img)+'_'+str(img)+".png"))
    


