import os
import numpy as np
import matplotlib.image as mpimg


def create_features(data):
    # RGB color features
    data_feat = np.std(data, axis=(1,2))
    data_feat = np.concatenate( (data_feat, np.mean(data, axis=(1,2))), axis=1)
    # All color features
    data_feat = np.concatenate( (data_feat, np.expand_dims(np.std(data, axis=(1,2,3)), axis=1)), axis=1)
    data_feat = np.concatenate( (data_feat, np.expand_dims(np.mean(data, axis=(1,2,3)), axis=1)), axis=1)
    return data_feat


def create_submission(y_pred, submission_filename, images_size=608, patch_size=16):
    n_patches = images_size//patch_size
    y_pred_shape = np.reshape(y_pred, (-1, n_patches, n_patches))
    text = 'id,prediction'
    with open(submission_filename, 'w') as f:
        f.write('id,prediction')
        for i in range(y_pred_shape.shape[0]):
            for j in range(y_pred_shape.shape[1]):
                for k in range(y_pred_shape.shape[2]):
                    name = '{:03d}_{}_{},{}'.format(i+1, j*patch_size, k*patch_size, int(y_pred_shape[i,j,k]))
                    f.write('\n'+name)

def load_train_set(dir_, data='images', label='groundtruth', ratio=0.8, seed=0):
    np.random.seed(seed)
    # Define path to data
    path_data = os.path.join(dir_, data)
    path_label = os.path.join(dir_, label)
    # Load train data files and sort them according to satImage_xxx.png where xxx is the id number
    files_data = np.array(os.listdir(path_label))
    id_sort = np.argsort([ int(filename[9:12]) for filename in files_data])
    files_data = files_data[id_sort]
    # Check first image to check the size (different for data and label)
    shape_train = mpimg.imread(os.path.join(path_data, files_data[0])).shape
    shape_label = mpimg.imread(os.path.join(path_label, files_data[0])).shape
    # Create empty matrices and load images (data and label)
    x_data = np.zeros((len(files_data),) + shape_train)
    y_label = np.zeros((len(files_data),) + shape_label)
    for i, file in enumerate(files_data):
        x_data[i] = mpimg.imread(os.path.join(path_data, file))
        y_label[i] = np.round(mpimg.imread(os.path.join(path_label, file)))
        
    # Id split for Valisation and test
    ids = np.random.permutation(x_data.shape[0])
    id_train = ids[:int(x_data.shape[0]*ratio)]
    id_valid = ids[int(x_data.shape[0]*ratio):]
    
    return x_data[id_train], y_label[id_train], x_data[id_valid], y_label[id_valid], id_train, id_valid

def load_test_set(dir_, data='test_set_images'):
    # Build path to root folder defined by data
    path_data = os.path.join(dir_, data)
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

def normalize_data(data, mean_ref=None, std_ref=None):
    
    if mean_ref is None or std_ref is None:
        mean_ref = np.mean(data, axis=tuple(np.arange(data.ndim-1)))
        std_ref = np.std(data-mean_ref, axis=tuple(np.arange(data.ndim-1)))
    data = (data-mean_ref)/std_ref
    return data, mean_ref, std_ref

def normalize_data_imwise(data):
    # Data pre-processing, Normalize each image with itself
    n = data.shape[0]
    for i in range(n):
        xx = data[i,:,:]
        xx -= np.mean(xx, axis=(0,1)) 
        xx /= np.std(xx, axis=(0,1)) 
        data[i] = xx # Affect value
    return data

def get_patches_all(x_data, y_label=None, patch_size=16):
    # Ccompute final siz of array of patches
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
        y_patches[i*(n_patch_h*n_patch_w):(i+1)*(n_patch_h*n_patch_w)] = get_patches(x_data[i], y_temp)
        
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

def load_image_train_by_id(_id, dir_, data='images', label='groundtruth'):
    path_data = os.path.join(dir_, data)
    path_label = os.path.join(dir_, label)
    # Load train data files and sort them according to satImage_xxx.png where xxx is the id number
    files_data = np.array(os.listdir(path_label))
    id_sort = np.argsort([ int(filename[9:12]) for filename in files_data])
    files_data = files_data[id_sort]
    file_load = files_data[_id]
    
    im = mpimg.imread(os.path.join(path_data, file_load))
    cgt = np.round(mpimg.imread(os.path.join(path_label, file_load)))
    
    return im, cgt

def load_image_test_by_id(_id, dir_, data='test_set_images'):
    path_data = os.path.join(dir_, data)
    path_file = os.path.join('test_{}'.format(_id), 'test_{}.png'.format(_id))
    im = mpimg.imread(os.path.join(path_data, path_file))
    
    return im

def patches_to_img(patches, im_shape, patch_size=16):
    n_patch_h = im_shape[0] // patch_size
    n_patch_w = im_shape[1] // patch_size
    
    im_patch = np.zeros(im_shape[:2])
    for i in range(n_patch_h):
        for j in range(n_patch_w):
            is_road = patches[i*n_patch_h+j]
            im_patch[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = is_road
    return im_patch