# Helper functions
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage.morphology import distance_transform_cdt, binary_closing
import Augmentor as ag

    
def extract_from_path(path, is_label=False):
    files_data = np.array( [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))] )
    id_sort = np.argsort([ int(filename[9:12]) for filename in files_data])
    files_data = files_data[id_sort]
    
    shape_img = mpimg.imread(os.path.join(path, files_data[0])).shape
    x_data = np.zeros((len(files_data),) + shape_img)
    for i, file_ in enumerate(files_data):
        if is_label:
            x_data[i] = np.round(mpimg.imread(os.path.join(path, file_)))
        else:
            x_data[i] = mpimg.imread(os.path.join(path, file_))
        
    return x_data


def load_train_set(dir_, data='images', label='groundtruth', train_sub='aug_train', val_sub='aug_val', diplay_log=False):
        
    path_train_img = os.path.join(dir_, data, train_sub)
    path_train_label = os.path.join(dir_, label, train_sub)
    path_val_img = os.path.join(dir_, data, val_sub)
    path_val_label = os.path.join(dir_, label, val_sub)
    
    if diplay_log:
        print('Loading train images ...')
    x_train_data = extract_from_path(path_train_img)
    if diplay_log:
        print('Loading train labels ...')
    y_train_label = extract_from_path(path_train_label, True)
    if diplay_log:
        print('Loading validation images ...')
    x_val_data = extract_from_path(path_val_img)
    if diplay_log:
        print('Loading validation labels ...')
    y_val_label = extract_from_path(path_val_label, True)
    
    return x_train_data, y_train_label, x_val_data, y_val_label

def load_train_set_from_id(dir_, id_train, data='images', label='groundtruth'):

    shape_img = mpimg.imread(os.path.join(dir_, data, id_train[0])).shape
    x_data = np.zeros((len(id_train),) + shape_img)
    
    for i, file_ in enumerate(id_train):
        x_data[i] = mpimg.imread(os.path.join(dir_, data, file_))
    
    return x_data

def load_test_set(path_data='data/test_set_images'):
    # Look for all file sin subfolder, store filename and path to filename (each test file is in a separate folder)
    files_data = []
    path_test = []
    for path, subdirs, files in os.walk(path_data):
        for name in files:
            files_data.append(name)
            path_test.append(path)
    # Get file ids to sort them (usedul for submission)
    id_files = [int(file.replace('test_', '').replace('.png', '')) for file in files_data]
    files_data = np.array(files_data)[np.argsort(id_files)]
    path_test = np.array(path_test)[np.argsort(id_files)]
    # Load firt file to get shape of images test and create dummy empty vector
    shape_test = mpimg.imread(os.path.join(path_test[0], files_data[0])).shape
    x_data = np.zeros((len(files_data),) + shape_test)
    # Load all test images in subfolders
    for i, (path, file) in enumerate(zip(path_test, files_data)):
        x_data[i] = mpimg.imread(os.path.join(path, file))
    
    return x_data

def get_patches_all(x_data, y_label=None, patch_size=16):
    # Compute final siz of array of patches
    n_patch_h = x_data.shape[1] // patch_size
    n_patch_w = x_data.shape[2] // patch_size
    x_patches = np.zeros((x_data.shape[0]*n_patch_h*n_patch_w, patch_size, patch_size) + x_data.shape[3:])
    y_patches = np.zeros((x_data.shape[0]*n_patch_h*n_patch_w, patch_size, patch_size))
    
    # Iterate over all images to convert them to patches
    for i in range(x_data.shape[0]):
        y_temp = None
        if y_label is not None:
            y_temp = y_label[i]
        x_patches[i*(n_patch_h*n_patch_w):(i+1)*(n_patch_h*n_patch_w)], \
        y_patches[i*(n_patch_h*n_patch_w):(i+1)*(n_patch_h*n_patch_w)] = get_patches(x_data[i], y_temp, patch_size)
        
    # Compute label for gt (Each patch is either road or background)
    y_batch_label = get_patches_label(y_patches)
    return x_patches, y_patches, y_batch_label

def get_patches(x_data, y_label=None, patch_size=16):
    # Check if dimension are ok (x_data shapes are mulitples of patch)
    assert(x_data.shape[0] % patch_size == 0 and x_data.shape[1] % patch_size == 0)
    # Create new array that will contain all the patches
    n_patch_h = x_data.shape[0] // patch_size
    n_patch_w = x_data.shape[1] // patch_size
    x_patches = np.zeros((n_patch_h, n_patch_w, patch_size, patch_size) + x_data.shape[2:])
    y_patches = np.zeros((n_patch_h, n_patch_w, patch_size, patch_size))
    # Iterate over all batches
    for i in range(n_patch_h):
        for j in range(n_patch_w):
            x_patches[i, j] = x_data[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            if y_label is not None:
                y_patches[i, j] = y_label[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
    # Reshape tu have first dimension as linear feature instead of n_patch_h X n_patch_w
    x_patches = np.reshape(x_patches, (-1, patch_size, patch_size) + x_data.shape[2:])
    y_patches = np.reshape(y_patches, (-1, patch_size, patch_size))
    return x_patches, y_patches

def get_patches_label(y_batch):
    # Create vector of one and zeros (1 is road, 0 is background)
    y_label_unique = np.zeros((y_batch.shape[0]))
    # Take mean value of pixel to estiamte road yes or not
    for i in range(y_batch.shape[0]):
        y_label_unique[i] = int(np.mean(y_batch[i]) > 0.25)
    return y_label_unique

def get_useful_patches(patch_x, patch_y, min_threshold, max_threshold):
    """ From an array of [Num_patches x H x W x Num_channels] and its associated pixelwise labels,
    return patches with label mean > threshold"""
    means = np.mean(patch_y, axis=(1,2))
    id_keep = np.logical_and(means >= min_threshold, means <= max_threshold)
    
    useful_patches_x = patch_x[id_keep]
    useful_patches_y = patch_y[id_keep]
    
    return useful_patches_x, useful_patches_y

def normalize_data(data, mode='image_wise', **kwargs):
    # Data pre-processing, Normalize each image with itself
    if mode == 'image_wise':
        n = data.shape[0]
        for i in range(n):
            xx = data[i]
            xx -= np.mean(xx) # Centering in 0
            #xx /= np.linalg.norm(xx) # Normalizing to 1
            data[i] = xx # Affect value
        return data
    elif mode == 'all':
        if 'mean_ref' not in kwargs or 'std_ref' not in kwargs:
            kwargs['mean_ref'] = np.mean(data, axis=tuple(np.arange(data.ndim-1)))
            kwargs['std_ref'] = np.std(data-kwargs['mean_ref'], axis=tuple(np.arange(data.ndim-1)))
        data_norm = ((data-kwargs['mean_ref'])/kwargs['std_ref']).copy()
        return data_norm, kwargs['mean_ref'], kwargs['std_ref']


def display_predictions(y_pred, img_ref, img_cgt=None, n_display=3):
    
    if img_cgt is None:
        display_predictions_nocgt(y_pred, img_ref, n_display)
        return
    
    im_pred = np.reshape(y_pred, img_ref.shape[:3]).astype(np.float32)
    # id_display = np.random.permutation(len(img_ref))[:n_display]
    id_display = np.arange(n_display)
        
    plt.figure(figsize=(16, 5*n_display))
    for i in range(n_display):
        plt.subplot(n_display,3,3*i+1)
        plt.imshow(img_ref[id_display[i]]); plt.axis('off');
        if img_cgt is not None:
            plt.subplot(n_display,3,3*i+2)
            plt.imshow(img_ref[id_display[i]]); plt.imshow(img_cgt[id_display[i]], alpha=0.3); plt.axis('off');
        plt.subplot(n_display,3,3*i+3)
        plt.imshow(img_ref[id_display[i]]); plt.imshow(im_pred[id_display[i]], alpha=0.3); plt.axis('off');
        
def display_predictions_nocgt(y_pred, img_ref, n_display=3):
    
    if len(y_pred.shape) == 1:
        im_pred = np.reshape(y_pred, img_ref.shape[:3]).astype(np.float32)
    else:
        im_pred = y_pred
    
    # id_display = np.random.permutation(len(img_ref))[:n_display]
    id_display = np.arange(n_display)
        
    plt.figure(figsize=(10, 5*n_display))
    for i in range(n_display):
        plt.subplot(n_display,2,2*i+1)
        plt.imshow(img_ref[id_display[i]]); plt.axis('off');
        plt.subplot(n_display,2,2*i+2)
        plt.imshow(img_ref[id_display[i]]); plt.imshow(im_pred[id_display[i]], alpha=0.3); plt.axis('off');
        
        
def create_submission(y_pred, submission_filename, images_size=608, patch_size=16):
    n_patches = images_size//patch_size
    text = 'id,prediction'
    with open(submission_filename, 'w') as f:
        f.write('id,prediction')
        for i in range(y_pred.shape[0]):
            im = y_pred[i]
            for j in range(0, im.shape[1], patch_size):
                for k in range(0, im.shape[0], patch_size):
                    patch = im[k:k + patch_size, j:j + patch_size]
                    label = patch_to_label(patch)
                    name = '{:03d}_{}_{},{}'.format(i+1, j, k, label)
                    f.write('\n'+name)
                    
                    
def patch_to_label(patch, foreground_threshold=0.25):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0
    
    
def get_distance_map(ims):
    ims_d = np.zeros(ims.shape)
    for i in range(ims.shape[0]):
        d = binary_closing(ims[i])
        d = distance_transform_cdt(1-d)
        ims_d[i] = d
    return ims_d


def get_penalize_values(ims, lambda_=0.3):
    ims_f = get_distance_map(ims)
    for i in range(ims_f.shape[0]):
        max_d = np.max(ims_f[i])
        T = lambda_*max_d
        ims_f[i] = np.minimum(ims_f[i]/max_d, lambda_) # Threshold to lambda_
        ims_f[i] = np.exp(-ims_f[i])
    return ims_f


def get_one_hot_penalization(imgs_val):
    dist_maps = get_penalize_values(imgs_val)
    dist_maps = np.reshape(dist_maps, [-1])
    pen = np.ones((len(dist_maps), 2))
    pen[:, 0] = dist_maps
    return pen