
# coding: utf-8

# In[1]:


import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from PIL import Image
from sklearn.metrics import f1_score
import tensorflow as tf
import datetime

import ml_utils.road_seg as rs
import ml_utils.model as model


# # 1. Loading and Preprocessing
# 
# Load raw images and ground truth

# In[2]:

path_data = 'data'
path_train_dir = os.path.join(path_data, 'training')
path_test = os.path.join(path_data, 'test_set_images')
path_models = 'model'

if not os.path.exists(path_models):
    os.mkdir(path_models)
train_imgs, train_gt, val_imgs, val_gt, id_train, id_valid = rs.load_train_set(path_train_dir, ratio=0.8)

print('Shapes train: {},  test: {}'.format(train_imgs.shape, val_imgs.shape))

# Generate smaller images for training

# In[3]:

train_imgs = rs.normalize_data(train_imgs, mode='image_wise') 
val_imgs = rs.normalize_data(val_imgs, mode='image_wise') 

patch_size = 80
patch_tr, lab_tr,_ = rs.get_patches_all(train_imgs, train_gt, patch_size)

print('Shapes train: {}'.format(patch_tr.shape))


# Take only part of train and validation set (should at least contain a part of the road)

# In[4]:

useful_patches_tr, useful_lab_tr = rs.get_useful_patches(patch_tr, lab_tr, 0.2, 0.7)
useful_lab_tr = useful_lab_tr.astype(int)

print('Shapes train: {}'.format(useful_patches_tr.shape))


# Compute distance map

# In[5]:

useful_lab_tr_dn = rs.get_penalize_values(useful_lab_tr)


# Display patches example with label ground truth and distance map


# In[ ]:


m = model.Model()
m.train_model(useful_patches_tr, useful_lab_tr,
              train_imgs, train_gt, val_imgs, val_gt, n_epoch=80)



