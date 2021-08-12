import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import measure
from mpl_toolkits.axes_grid1 import ImageGrid
import random
from scipy.ndimage import rotate

def show_imagegrid(img_lst,n_r,n_c):
    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(n_r, n_c),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    
    
    for ax, im in zip(grid, img_lst):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
    plt.show()



def bbox(msk, margin):
    a = np.where(msk != 0)
    bbox = np.min(a[0])-margin, np.max(a[0]) + margin, np.min(a[1])- margin, np.max(a[1]) + margin
    return bbox


def bbox_crop(im,msk,margin_bb,padding_crop):
    
    bb = md.bbox(msk,margin_bb)
    img = im.copy()
    img = img[bb[0]:bb[1],bb[2]:bb[3]]
    img[:,:] = 6
    img = md.crop_or_pad_slice_to_size(img,padding_crop, padding_crop)
    return img

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






def euclideanDistance(coordinate1, coordinate2):
    return pow(pow(coordinate1[0] - coordinate2[0], 2) + pow(coordinate1[1] - coordinate2[1], 2), .5)


def create_chessboard(h,w,s):
    cb = np.zeros((h,w))
    
    for x in range(w):
        for y in range(h):
            if (x%(2*s) < s and y%(2*s) < s) or (x%(2*s) >= s and y%(2*s) >= s):
                cb[y,x] = 1
                
                
    return cb            
    
def contour_distance(contour):
    distance = 0
    for i,p in enumerate(contour[:-1]):
        
        distance += euclideanDistance(p, contour[i+1])
    return distance
    

def equidistant_landmarks(contour,n_points):
    
    contour = [p for p in list(contour)]
    distance = contour_distance(contour)
    step = distance/n_points

    cummulative = []
    
    distance = 0
    for i,p in enumerate(contour[:-1]):
        distance += euclideanDistance(p, contour[i+1])
        if distance > step:
            distance = 0
        
        cummulative.append(distance)
    cummulative.append(0)
        
    landmarks = [contour[n] for n in np.where(np.array(cummulative) == 0)[0].tolist()]

     
    return landmarks

def align_landmarks(l_in,l_out):
    
    first_point = l_in[0]
    distances = []
    
    for p in l_out:
        distances.append(euclideanDistance(first_point,p))
        
    starting_point = np.argmin(distances)
    
    
    return l_out[starting_point:] +l_out[:starting_point] 


def align_landmarks2(l_in):
    
    distances = []
    
    for p in l_in:
        distances.append(euclideanDistance((0,0),p))
        
    starting_point = np.argmin(distances)
    
    print('l_in',l_in)
    print(starting_point)
    return l_in[starting_point:] + l_in[:starting_point]


def adapt_contour(img_in,mask_in,mask_out=None):
    
    
    contours_in = measure.find_contours(mask_in,0)
    landmarks_in = equidistant_landmarks(contours_in[0],24)
    
    contours_out = measure.find_contours(mask_out,0)
    landmarks_out = equidistant_landmarks(contours_out[0],24)
    

    if len(landmarks_in) > len(landmarks_out):
        landmarks_in = landmarks_in[:len(landmarks_out)]
    else:
        landmarks_out = landmarks_out[:len(landmarks_in)]
        
         
    tform = PiecewiseAffineTransform()
    tform.estimate(np.fliplr(np.array(landmarks_out)), np.fliplr(np.array(landmarks_in)))
    img_out = warp(img_in, tform, output_shape=img_in.shape)

    return img_out, landmarks_in,landmarks_out


def adapt_double_contour(img_in,mask_in,mask_out=None,n_points = 24):
    
    
    contours_in = measure.find_contours(mask_in,1)
    landmarks_in_epi = equidistant_landmarks(contours_in[0],n_points)
    landmarks_in_end = equidistant_landmarks(contours_in[1],n_points)
    contours_out = measure.find_contours(mask_out,0)
    landmarks_out_epi = equidistant_landmarks(contours_out[0],n_points)
    landmarks_out_end = equidistant_landmarks(contours_out[1],n_points)
    

    landmarks_out = []    
    landmarks_out.extend(landmarks_out_end)
    landmarks_out.extend(landmarks_out_epi)    
    
    
    landmarks_in = []    
    landmarks_in.extend(landmarks_in_end)
    landmarks_in.extend(landmarks_in_epi)
    
    
    tform = PiecewiseAffineTransform()
    tform.estimate(np.fliplr(np.array(landmarks_out)), np.fliplr(np.array(landmarks_in)))
    img_out = warp(img_in, tform, output_shape=img_in.shape,order = 0)
    
    
    return img_out, landmarks_in,landmarks_out_epi,landmarks_out_end

def adapt_contour(img_in,mask_in,mask_out=None,n_points = 8, bbox_in = None, bbox_out = None,interp = 0):
    
    
    contours_in = measure.find_contours(mask_in,0)
    landmarks_in_epi = equidistant_landmarks(contours_in[0],n_points)
    landmarks_in_end = equidistant_landmarks(contours_in[1],n_points)
    contours_out = measure.find_contours(mask_out,0)
    landmarks_out_epi = equidistant_landmarks(contours_out[0],n_points)
    landmarks_out_end = equidistant_landmarks(contours_out[1],n_points)
    

    if len(landmarks_in_epi) > len(landmarks_out_epi):
        landmarks_in_epi = landmarks_in_epi[:len(landmarks_out_epi)]
    else:
        landmarks_out_epi = landmarks_out_epi[:len(landmarks_in_epi)]
        
    if len(landmarks_in_end) > len(landmarks_out_end):
        landmarks_in_end = landmarks_in_end[:len(landmarks_out_end)]
    else:
        landmarks_out_end = landmarks_out_end[:len(landmarks_in_end)]
        
    landmarks_out = []   
    
    
    #landmarks_out_end = align_landmarks2(landmarks_out_end) 
    #landmarks_out_epi = align_landmarks2(landmarks_out_epi) 


    #landmarks_out.extend(landmarks_out_end)
    landmarks_out.extend(landmarks_out_epi)  
    
    #landmarks_in_end = align_landmarks2(landmarks_in_end) 
    #landmarks_in_epi = align_landmarks2(landmarks_in_epi) 
    

    landmarks_in = []    
    #landmarks_in.extend(landmarks_in_end)
    landmarks_in.extend(landmarks_in_epi)
    #landmarks_in = [(p[0]+random.randint(-2,1),p[1]+random.randint(-2,1)) for p in landmarks_in]
    #landmarks_out = [(p[0]+random.randint(-2,1),p[1]+random.randint(-2,1)) for p in landmarks_out]

    if bbox_in is None:
        landmarks_out.append(np.array([0, 0]))
        landmarks_out.append(np.array([0, img_in.shape[1]]))
        landmarks_out.append(np.array([img_in.shape[0], img_in.shape[1]]))
        landmarks_out.append(np.array([img_in.shape[0], 0]))  
        
        landmarks_in.append(np.array([0, 0]))
        landmarks_in.append(np.array([0, img_in.shape[1]]))
        landmarks_in.append(np.array([img_in.shape[0], img_in.shape[1]]))
        landmarks_in.append(np.array([img_in.shape[0], 0]))
    else:
        landmarks_out.append(np.array([bbox_out[0], bbox_out[2]]))
        landmarks_out.append(np.array([bbox_out[0], bbox_out[3]]))
        landmarks_out.append(np.array([bbox_out[1], bbox_out[2]]))
        landmarks_out.append(np.array([bbox_out[1], bbox_out[3]]))  
        
        landmarks_in.append(np.array([bbox_in[0], bbox_in[2]]))
        landmarks_in.append(np.array([bbox_in[0], bbox_in[3]]))
        landmarks_in.append(np.array([bbox_in[1], bbox_in[2]]))
        landmarks_in.append(np.array([bbox_in[1], bbox_in[3]]))
    
    
    tform = PiecewiseAffineTransform()
    tform.estimate(np.fliplr(np.array(landmarks_out)), np.fliplr(np.array(landmarks_in)))
    img_out = warp(img_in, tform, output_shape=img_in.shape,order = interp)
    
    
    return img_out, landmarks_in,landmarks_out

def map_labels(msks,in_lst,out_lst):
    msks_out = msks.copy()
    for i,o in zip(in_lst,out_lst):
        msks_out[msks == i] = o
        

    return msks_out


def adapt_contour_full(im_in,m_in,m_out,msk_warp):
    
    bbox_in = bbox(map_labels(m_in,[2],[0]),10)
    bbox_out = bbox(map_labels(m_out,[2],[0]),10)
    m_in  = map_labels(m_in,[2,3,4,5],[0,0,1,1])
    m_out = map_labels(m_out,[2,3,4,5],[0,0,1,1])
    """
    plt.imshow(np.hstack((m_in,m_out)))
    plt.show()
    """ 
    im_out,lm_in,lm_out =adapt_contour(im_in,m_in,m_out,bbox_in=bbox_in,bbox_out=bbox_out,interp = 2)
    mask_out,_,_ =adapt_contour(msk_warp,m_in,m_out,bbox_in=bbox_in,bbox_out=bbox_out)
    mask_out = (mask_out/np.max(mask_out))*np.max(np.unique(msk_warp)).astype(np.uint8)
    
    """
    plt.imshow(np.hstack((m_in,m_out)))
    
    for p in lm_in:
        plt.scatter(p[1],p[0])
    
    for p in lm_out:
        plt.scatter(p[1]+256,p[0])
    plt.show()
    """

    return im_out,mask_out

def generate_circle(img, radius, cx, cy):
    
    y, x = np.ogrid[-radius: radius, -radius: radius]
    index = x**2 + y**2 <= radius**2
    img[cy-radius:cy+radius, cx-radius:cx+radius][index] = 1
    
    return img


def generate_ring(img, radius_1,radius_2, cx, cy):
    
    im1 = generate_circle(img.copy(), radius_1, cx, cy)
    im2 = generate_circle(img.copy(), radius_2, cx, cy)
    

    return im1 - im2
    

def label_rotation(label,degrees):
    
    half_size= int(label.shape[0]/2)
    labels_without_scar = label.copy()
    labels_without_scar[labels_without_scar == 4] = 1 
    labels_without_scar[labels_without_scar == 5] = 1 
    labels_without_scar[labels_without_scar == 2] = 0
    labels_without_scar[labels_without_scar == 3] = 0
    labels_without_scar[labels_without_scar == 6] = 0
    labels_without_scar[labels_without_scar == 1] = 1
    
    
    #decopupling rv for rotation
    label_wo_rv = label.copy()
    label_rv = label.copy()
    label_wo_rv[label_wo_rv==3] = 0
    label_rv[label!=3] = 0
    #adapt to a ring
    msk_in_bas = labels_without_scar
    ring = generate_ring(np.zeros_like(msk_in_bas),half_size-60,half_size-100,half_size,half_size)
    binary_rounded,msk_wo_rv_rounded= adapt_contour_full(msk_in_bas,msk_in_bas,ring,label_wo_rv)
    bb = bbox(msk_wo_rv_rounded,10)
    msk_wo_rv_rounded = crop_or_pad_slice_to_size(msk_wo_rv_rounded[bb[0]:bb[1],bb[2]:bb[3]],half_size*2,half_size*2)
    
    # rotate ring
    msk_wo_rv_rot = rotate(msk_wo_rv_rounded,degrees,reshape = False, order = 0)
    
    """"
    plt.imshow(msk_wo_rv_rot)
    plt.show()
    """
    #re adapt to original shape

    _,msk_rewarped_rot  = adapt_contour_full(msk_wo_rv_rot,msk_wo_rv_rot,labels_without_scar,msk_wo_rv_rot) 
    

    msk_rewarped_rot[label_rv != 0] = 3
    
    
    return msk_rewarped_rot



def dilation(x, radius=3):
    """ Return greyscale morphological dilation of an image,
    see `skimage.morphology.dilation <http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.dilation>`_.

    Parameters
    -----------
    x : 2D array image.
    radius : int for the radius of mask.
    """
    from skimage.morphology import disk, dilation, erosion
    mask = disk(radius)
    x = dilation(x, selem=mask)
    return x

def erosion(x, radius=2):
    """ Return greyscale morphological dilation of an image,
    see `skimage.morphology.dilation <http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.dilation>`_.

    Parameters
    -----------
    x : 2D array image.
    radius : int for the radius of mask.
    """
    from skimage.morphology import disk, dilation, erosion
    mask = disk(radius)
    x = erosion(x, selem=mask)
    return x

def binarize(mask):
    bin_mask = mask.copy()
    bin_mask[mask > 1] = 1
    
    return bin_mask



    
def dilate_erode_scar(mask,aug,rad, with_deletion = True):   
    
    bin_mask = map_labels(mask,[2,3,4,5],[0,0,1,1])
    edema_mask = mask.copy()
    scar_mask = mask.copy()
    
    edema_mask[edema_mask != 4] = 0
    scar_mask[scar_mask != 5] = 0

    #aug = random.choice([0,1,2])
    
    if aug == 0:
        edema_mask = dilation(edema_mask,rad)
        scar_mask = erosion(scar_mask,int(rad/2))
    elif aug == 1:
        edema_mask = erosion(edema_mask,rad)
        scar_mask = dilation(scar_mask,rad)
    else:
        return mask

    edema_scar_msk = edema_mask + scar_mask
    edema_scar_msk[edema_scar_msk == 6] = 4
    
    
    if with_deletion:
        rand = np.random.randint(10)
        if rand > 7:
            edema_scar_msk[edema_scar_msk==5] = 0
            edema_scar_msk = dilation(edema_scar_msk,np.random.randint(8))
        if rand < 3:
            edema_scar_msk[edema_scar_msk==4] = 0
             
    pl = map_labels(mask,[4,5],[1,1])
    edema_scar_msk[edema_scar_msk > 5] = 4
    pl[edema_scar_msk != 0] = edema_scar_msk[edema_scar_msk != 0]
    
    
    removed_dead_eroded_pixels = pl.copy()
    for x in range(1,pl.shape[0]-1):
        for y in range(1,pl.shape[1]-1):
            if pl[x,y] == 1:
                if pl[x+1,y] == 2  or pl[x-1,y] == 2 or pl[x,y+1] == 2 or pl[x,y-1] == 2 or pl[x-1,y-1] == 2 or pl[x+1,y+1] == 2 or pl[x-1,y+1] == 2 or pl[x+1,y-1] == 2:
                    removed_dead_eroded_pixels[x,y] = 2
    final_mask = removed_dead_eroded_pixels


    return final_mask
    



"""
import h5py

data = h5py.File('../data/myops_ds_zoo/original_dataset_subs.hdf5', 'r')

    # the following are HDF5 datasets, not numpy arrays
images_train = data['data_train']
labels_train = data['pred_train_my']
labels_train_orig = data['pred_train']


ws = 55
im_in  = images_train[ws,...]
m_in =  labels_train[ws,...]

for s in range(70):
    
    m_out =  labels_train[s,...]
    

    im_out,mask_out, =adapt_contour_full(im_in,m_in,m_out,labels_train_orig[ws,...])
    #mask_out,_, =adapt_contour_full(labels_train_orig[0,...],m_in,m_out,labels_train[s,...])
    
    plt.imshow(np.hstack((im_in[...,3]+labels_train_orig[ws,...],im_out[...,3]+mask_out)))
    plt.show()

"""


 
