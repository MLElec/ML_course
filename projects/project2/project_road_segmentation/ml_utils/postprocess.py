from skimage.morphology import skeletonize, thin, skeletonize_3d
from skimage import data
import matplotlib.pyplot as plt
from skimage.morphology import disk, rectangle, closing, opening, remove_small_objects, dilation
from skimage.measure import label, regionprops
import numpy as np

def process_cgt(img):
    """
    Performs post processing on one image
    """
    
    # Remove small objects
    r = remove_small_objects(img.astype(bool), min_size = 20)
    
    # Connect roads that are close to each other in 2 directions (horizontal and vertical)
    r = closing(r, disk(5))
    r = closing(r, rectangle(1, 15))
    r = closing(r, rectangle(15, 1))
    
    # Label each blob in the image
    label_image = label(r)
    
    # Iterate over blobs
    for i, region in enumerate(regionprops(label_image)):
        
        # If the blob touch a border we should not remove it since we do not know if it is an external road
        minr, minc, maxr, maxc = region.bbox
        if minr == 0 or minc == 0 or maxr == img.shape[0] or maxc == img.shape[0]:
            continue
        # Here we check if the area is not too small.
        if region.area > 200:
            continue
        # If the blob do not meet the conditions it is removed from image
        label_image[label_image == i+1] = 0
        
    return label_image >= 1

def process_all(ypreds, size_image=400):
    """
    Performs post processing on all prediction (should be a vector and will be reshaped). size_image is used to set the
    final size of the images
    """
    pred_proc = np.reshape(ypreds, (-1, size_image, size_image)).copy()
    
    # Iterate over all images
    for i in range(pred_proc.shape[0]):
        pred_proc[i] = process_cgt(pred_proc[i])

    return pred_proc.flatten()  

