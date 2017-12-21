from skimage.morphology import skeletonize, thin, skeletonize_3d
from skimage import data
import matplotlib.pyplot as plt
from skimage.morphology import disk, rectangle, closing, opening, remove_small_objects, dilation
from skimage.measure import label, regionprops
import numpy as np

def process_cgt(img):
    
    r = remove_small_objects(img.astype(bool), min_size = 20)
    
    r = closing(r, disk(5))
    r = closing(r, rectangle(1, 15))
    r = closing(r, rectangle(15, 1))
    
    label_image = label(r)
    
    for i, region in enumerate(regionprops(label_image)):
        minr, minc, maxr, maxc = region.bbox
        if minr == 0 or minc == 0 or maxr == img.shape[0] or maxc == img.shape[0]:
            continue
        if region.area > 200:
            continue

        label_image[label_image == i+1] = 0
        
    return label_image >= 1

def process_all(ypreds, size_image=400):

    pred_proc = np.reshape(ypreds, (-1, size_image, size_image)).copy()
    
    for i in range(pred_proc.shape[0]):
        pred_proc[i] = process_cgt(pred_proc[i])

    return pred_proc.flatten()  

