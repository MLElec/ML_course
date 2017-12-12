import skimage.measure
import skimage.draw
import skimage.morphology
import numpy as np

def post_processing(imgs, size_min):
    imgs_post = np.zeros(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_post[i] = skimage.morphology.remove_small_objects(imgs[i].astype(bool), min_size = size_min)   
    return imgs_post