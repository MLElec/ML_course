# Helper functions
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature
import os


        
def add_canny_as_4rth_color(img, sigma):
    canny_layer = np.zeros(img.shape[:2])
    for color in range(3):
        canny_layer += feature.canny(img[:,:,color], sigma=sigma)
    img = np.dstack((img, canny_layer/3))
    return img

def add_canny_layer_imgs(images, sigma, plot_random = False, nr_plot_imgs=5):
    results = np.zeros([images.shape[0], images.shape[1], images.shape[2],images.shape[3]+1])
    for image_nr in range(images.shape[0]):
        proc_img = images[image_nr,:,:,:]
        results[image_nr,:,:,:] = add_canny_as_4rth_color(proc_img,sigma)
    
    if(plot_random == True):
        nrs_to_plot = np.random.permutation(images.shape[0])[:nr_plot_imgs+1]
        plt.figure(figsize=(20,20))
        for i in range( nr_plot_imgs+1):
            plt.subplot(nr_plot_imgs+2, 2, 2*i+1)
            plt.imshow(results[nrs_to_plot[i],:,:,:3])
            plt.title('3 colors')
            plt.subplot(nr_plot_imgs+2, 2, 2*i+2)
            plt.set_cmap('Greys')
            plt.imshow(results[nrs_to_plot[i],:,:,3])
            plt.title('Canny layer')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return results;