import numpy as np

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


def add_bb_label(label):
    
    bb  = bbox(label,15)
    label = label[bb[0]:bb[1],bb[2]:bb[3]]
    label[label==0] = 6
    label = crop_or_pad_slice_to_size(label,256,256)
    return label
    