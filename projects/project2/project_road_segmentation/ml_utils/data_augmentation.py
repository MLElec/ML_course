import numpy as np
import datetime
import Augmentor as ag
import os
import shutil
from os import listdir
from os.path import isfile, join
from os import rename
from PIL import Image


def genererate_data_from_id(file_train, file_val, path_train_dir, path_image='images', 
                            path_gt='groundtruth', n_aug=400, seed=0, display_log=True):
    
        # Copy file to repective folders
    if display_log:
        print('\nCopy to sub folders ...')
    path_train_gen, path_val_gen = _copy_data(os.path.join(path_train_dir, path_image), file_train, file_val)
    path_train_gen, path_val_gen = _copy_data(os.path.join(path_train_dir, path_gt), file_train, file_val)
    
    # Generate augmented features
    if display_log:
        print('\nGenerate augmented features - + {} Train ...'.format(n_aug))
    _create_augmented_features(os.path.join(path_train_dir, path_image, path_train_gen), seed=seed, n_aug=n_aug)
    if display_log:
        print('\nGenerate augmented features - + {} Labels ...'.format(n_aug))
    _create_augmented_features(os.path.join(path_train_dir, path_gt, path_train_gen), seed=seed, n_aug=n_aug)

    if display_log:
        print('\nRemove augmentation source images from train ...')
    _remove_ids(file_train, os.path.join(path_train_dir, path_image, path_train_gen))
    _remove_ids(file_train, os.path.join(path_train_dir, path_gt, path_train_gen))

def genererate_data(path_train_dir, path_image='images', path_gt='groundtruth', ratio=0.8, n_aug=400, seed=0, display_log=True):
    
    # Generate output folders and copy data to them
    if display_log:
        print('\nSplit train set train/validation ...')
    file_train, file_val = _get_ids_train_val(os.path.join(path_train_dir, path_image), ratio)
    
    # Copy file to repective folders
    if display_log:
        print('\nCopy to sub folders ...')
    path_train_gen, path_val_gen = _copy_data(os.path.join(path_train_dir, path_image), file_train, file_val)
    path_train_gen, path_val_gen = _copy_data(os.path.join(path_train_dir, path_gt), file_train, file_val)
    
    # Generate augmented features
    if display_log:
        print('\nGenerate augmented features - Train ...')
    _create_augmented_features(os.path.join(path_train_dir, path_image, path_train_gen), n_aug=n_aug, seed=seed)
    if display_log:
        print('\nGenerate augmented features - Labels ...')
    _create_augmented_features(os.path.join(path_train_dir, path_gt, path_train_gen), n_aug=n_aug, seed=seed)

    if display_log:
        print('\nRemove augmentation source images from train ...') 
    _remove_ids(file_train, os.path.join(path_train_dir, path_image, path_train_gen))
    _remove_ids(file_train, os.path.join(path_train_dir, path_gt, path_train_gen))
    
    return file_train, file_val

def _remove_ids(file_id, path_image=''):
    for file_ in file_id:
        filename = os.path.join(path_image, file_)
        if os.path.exists(filename):
            os.remove(filename)
    
def _get_ids_train_val(path, ratio=0.8, seed=2):
    
    np.random.seed(seed)
    files_data = np.array( [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))] )
    id_sort = np.argsort([ int(filename[9:12]) for filename in files_data])
    files_data = files_data[id_sort]
    
        # Id split for Valisation and test
    ids = np.random.permutation(files_data.shape[0])
    id_train = ids[:int(files_data.shape[0]*ratio)]
    id_valid = ids[int(files_data.shape[0]*ratio):]
    
    return files_data[id_train], files_data[id_valid]
    
def _copy_data(path_src, filenames_train, filenames_val, path_train='aug_train', path_val='aug_val'):
    
    path_dest_train = os.path.join(path_src, path_train)
    path_dest_val = os.path.join(path_src, path_val)
    
    # Reset data is already exists
    if os.path.exists(path_dest_train):
        shutil.rmtree(path_dest_train)
    if os.path.exists(path_dest_val):
        shutil.rmtree(path_dest_val)
      
    # Make dir and copy files to new directory
    os.mkdir(path_dest_train)
    os.mkdir(path_dest_val)
    for file_ in filenames_train:
        shutil.copyfile(os.path.join(path_src, file_), os.path.join(path_dest_train, file_))
    for file_ in filenames_val:
        shutil.copyfile(os.path.join(path_src, file_), os.path.join(path_dest_val, file_))
        
    return path_train, path_val
    
def _create_augmented_features(path_images, path_output='', n_aug=10, max_roatations=25, prob=0.8, seed=0):
  
    # Build pipeline and set input/output directories
    p = ag.Pipeline(save_format='png')
    ag.Pipeline.set_seed(seed)
    p.add_further_directory(path_images, path_output)
    
    # Define transformations
    p.flip_left_right(probability=prob)
    p.flip_top_bottom(probability=prob)
    p.rotate(probability=prob, max_left_rotation=max_roatations, max_right_rotation=max_roatations)
    
    # Generate samples
    _pipeline(p, n_aug)

def _pipeline(pipe, n):
    
    sample_count = 1
    while sample_count <= n:
        for augmentor_image in pipe.augmentor_images:
            if sample_count <= n:
                img = pipe._execute(augmentor_image, save_to_disk=False)

                #if img.mode != "RGB":
                #    img = img.convert("RGB")
                
                file_name = 'satImage_{:03d}'.format(100+sample_count) + '.' + pipe.save_format
                img.save(os.path.join(augmentor_image.output_directory, file_name))
            sample_count += 1