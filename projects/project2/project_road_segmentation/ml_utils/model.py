import tensorflow as tf
import datetime
import numpy as np
from sklearn.metrics import f1_score
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import ml_utils.road_seg as rs
import time
import ml_utils.data_augmentation as d_aug
import ml_utils.road_seg as rs


class Model:
    
    def __init__(self, reg = 1e-2, n_filters = 64, kernel_size=3, display_log=True, 
                 path_models='model', model_type='cnn'):
                
        # Settings model
        self.reg = reg
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.file_train = None
        self.file_val = None
        
        # Path
        self.save_path_model = None
        self.save_path_stats = None
        self.path_models = path_models
        self.model_type = model_type
        
        if not os.path.exists(path_models):
            os.mkdir(path_models)
        
        if self.model_type == 'cnn_bn':
            self.build_model_cnn_bn(display_log)
        else:
            self.build_model_cnn(display_log)
        
    def bn_conv_relu(self, inputs,kernel_size, n_filters, is_training, regularizer):
        return tf.layers.conv2d(inputs=tf.layers.batch_normalization(inputs, training=is_training), 
                                filters=n_filters,kernel_size=kernel_size, kernel_regularizer=regularizer,
                                activation=tf.nn.relu,padding='SAME')
                
        
    def bn_upconv_relu(self, inputs,kernel_size, n_filters, is_training, regularizer):
        return tf.layers.conv2d_transpose(inputs=tf.layers.batch_normalization(inputs, training=is_training), 
                                          filters=n_filters, kernel_size=kernel_size, strides=2, 
                                          kernel_regularizer=regularizer, activation=tf.nn.relu, padding='SAME')
    
    
    def build_model_cnn_bn(self, display_log, seed=0):
        
        tf.reset_default_graph()
        tf.set_random_seed(seed)

        # Plae holders
        self.learning_rate = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)
        self.tf_data = tf.placeholder(tf.float32,[None, None, None, 3])
        self.tf_labels = tf.placeholder(tf.int32,[None,2])
        self.tf_pen_road = tf.placeholder(tf.float32,[None,2])
        
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg)
        class_weights = tf.constant([[1.0,1.0]]) 
        weights = tf.reduce_sum(class_weights * tf.cast(self.tf_labels, tf.float32), axis=1)

        self.conv1_1 =tf.layers.conv2d(inputs=self.tf_data,filters=self.n_filters,kernel_size=self.kernel_size, 
                                 kernel_regularizer=regularizer,activation=tf.nn.relu,padding='SAME')
        self.conv1_2 = self.bn_conv_relu(self.conv1_1, self.kernel_size, self.n_filters, self.is_training, regularizer)
        
        self.conv1_3 = self.bn_conv_relu(self.conv1_2, self.kernel_size, self.n_filters, self.is_training, regularizer)

        pool1 = tf.contrib.layers.max_pool2d(inputs=self.conv1_3, kernel_size=2, stride=2)

        self.conv2_1 = self.bn_conv_relu(pool1, self.kernel_size, self.n_filters, self.is_training, regularizer)
        
        self.conv2_2 = self.bn_conv_relu(self.conv2_1, self.kernel_size, self.n_filters, self.is_training, regularizer)
        
        self.conv2_3 = self.bn_conv_relu(self.conv2_2, self.kernel_size, self.n_filters, self.is_training, regularizer)

        pool2 = tf.contrib.layers.max_pool2d(inputs=self.conv2_3, kernel_size=2, stride=2)

        self.conv3_1 = self.bn_conv_relu(pool2, self.kernel_size, self.n_filters, self.is_training, regularizer)
        
        self.conv3_2 = self.bn_conv_relu(self.conv3_1, self.kernel_size, self.n_filters, self.is_training, regularizer)
        
        self.conv3_3 = self.bn_conv_relu(self.conv3_2, self.kernel_size, self.n_filters, self.is_training, regularizer)

        pool3 = tf.contrib.layers.max_pool2d(inputs=self.conv3_3, kernel_size=2, stride=2)

        self.conv4_1 = self.bn_conv_relu(pool3, self.kernel_size, self.n_filters, self.is_training, regularizer)
        
        self.conv4_2 = self.bn_conv_relu(self.conv4_1, self.kernel_size, self.n_filters, self.is_training, regularizer)
        
        self.conv4_3 = self.bn_conv_relu(self.conv4_2, self.kernel_size, self.n_filters, self.is_training, regularizer)

        pool4 = tf.contrib.layers.max_pool2d(inputs=self.conv4_3, kernel_size=2, stride=2)
        
        self.conv5_1 = self.bn_conv_relu(pool4, self.kernel_size, self.n_filters, self.is_training, regularizer)
        
        self.conv5_2 = self.bn_conv_relu(self.conv5_1, self.kernel_size, self.n_filters, self.is_training, regularizer)

        self.deconv1 = self.bn_upconv_relu(self.conv5_2, self.kernel_size, self.n_filters, self.is_training, regularizer)

   
        deconv1_c = tf.concat([self.deconv1, self.conv4_2], axis=3)
    
        # deconv1_c = 0.5*tf.add(self.deconv1, self.conv4)

        self.conv6_1 = self.bn_conv_relu(deconv1_c, self.kernel_size, 1.5*self.n_filters, self.is_training, regularizer)
        
        self.conv6_2 = self.bn_conv_relu(self.conv6_1, self.kernel_size, self.n_filters, self.is_training, regularizer)
        
        self.deconv2 = self.bn_upconv_relu(self.conv6_2, self.kernel_size, self.n_filters, self.is_training, regularizer)

        
        deconv2_c = tf.concat([self.deconv2, self.conv3_2], axis=3)
        # deconv2_c = 0.5*tf.add(self.deconv2, self.conv3)
        
        self.conv7_1 = self.bn_conv_relu(deconv2_c, self.kernel_size, 1.5*self.n_filters, self.is_training, regularizer)
        
        self.conv7_2 = self.bn_conv_relu(self.conv7_1, self.kernel_size, self.n_filters, self.is_training, regularizer)
        
        self.deconv3 = self.bn_upconv_relu(self.conv7_2, self.kernel_size, self.n_filters, self.is_training, regularizer)

        
        deconv3_c = tf.concat([self.deconv3, self.conv2_2], axis=3)
        # deconv3_c = 0.5*tf.add(self.deconv3, self.conv2)
            
        self.conv8_1 = self.bn_conv_relu(deconv3_c, self.kernel_size, 1.5*self.n_filters, self.is_training, regularizer)
        
        self.conv8_2 = self.bn_conv_relu(self.conv8_1, self.kernel_size, self.n_filters, self.is_training, regularizer)
        
        self.deconv4 = self.bn_upconv_relu(self.conv8_2, self.kernel_size, self.n_filters, self.is_training, regularizer)

        
        deconv4_c = tf.concat([self.deconv4, self.conv1_2], axis=3)
        # deconv4_c = 0.5*tf.add(self.deconv4, self.conv1)

        self.conv9_1 = self.bn_conv_relu(deconv4_c, self.kernel_size, 1.5*self.n_filters, self.is_training, regularizer)
        
        self.conv9_2 = self.bn_conv_relu(self.conv9_1, self.kernel_size, self.n_filters, self.is_training, regularizer)
                
        self.score_layer = tf.layers.conv2d(inputs=self.conv9_2, filters=2, kernel_size=1,kernel_regularizer=regularizer)
        
        logits = tf.reshape(self.score_layer, (-1,2))
        #y = tf.nn.softmax(logits)
        
        #y_label = tf.multiply(tf.cast(self.tf_labels, tf.float32), self.tf_pen_road)
        #self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.multiply(y_label, tf.log(y)), 1))

        self.cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=self.tf_labels, logits=logits, weights=weights)
        self.reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_term = tf.contrib.layers.apply_regularization(regularizer, self.reg_variables)
        self.loss = self.reg_term + self.cross_entropy

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.preds = tf.argmax(logits,axis=1,output_type=tf.int32)
        
        if display_log:
            print("conv1_1 size", self.conv1_1.shape)
            print("conv1_2 size", self.conv1_2.shape)
            print("conv1_3 size", self.conv1_3.shape)
            print("pool1 size", pool1.shape)
            print("conv2_1 size", self.conv2_1.shape)
            print("conv2_2 size", self.conv2_2.shape)
            print("conv2_3 size", self.conv2_3.shape)
            print("pool2 size", pool2.shape)
            print("conv3_1 size", self.conv3_1.shape)
            print("conv3_2 size", self.conv3_2.shape)
            print("conv3_3 size", self.conv3_3.shape)
            print("pool3 size", pool3.shape)
            print("conv4_1 size", self.conv4_1.shape)
            print("conv4_2 size", self.conv4_2.shape)
            print("conv4_3 size", self.conv4_3.shape)
            print("pool4 size", pool4.shape)
            print("conv5_1 size", self.conv5_1.shape)
            print("conv5_2 size", self.conv5_2.shape)
            print("deconv1 size", self.deconv1.shape)
            print("deconv1_c size", deconv1_c.shape)
            print("conv6_1 size", self.conv6_1.shape)
            print("conv6_2 size", self.conv6_2.shape)
            print("deconv2 size", self.deconv2.shape)
            print("deconv2_c size", deconv2_c.shape)
            print("conv7_1 size", self.conv7_1.shape)
            print("conv7_2 size", self.conv7_2.shape)
            print("deconv3 size", self.deconv3.shape)
            print("deconv3_c size", deconv3_c.shape)
            print("conv8_1 size", self.conv8_1.shape)
            print("conv8_2 size", self.conv8_2.shape)
            print("deconv4 size", self.deconv4.shape) 
            print("deconv4_c size", deconv4_c.shape)
            print("conv9_1 size", self.conv9_1.shape)
            print("conv9_2 size", self.conv9_2.shape)
            print("score size", self.score_layer.shape)
       
    
    def build_model_cnn(self, display_log, seed=0):
        
        tf.reset_default_graph()
        tf.set_random_seed(seed)

        # Plae holders
        self.learning_rate = tf.placeholder(tf.float32)
        self.tf_data = tf.placeholder(tf.float32,[None, None, None, 3])
        self.tf_labels = tf.placeholder(tf.int32,[None,2])
        self.tf_pen_road = tf.placeholder(tf.float32,[None,2])
        
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg)
        class_weights = tf.constant([[1.0,1.0]]) 
        weights = tf.reduce_sum(class_weights * tf.cast(self.tf_labels, tf.float32), axis=1)

        self.conv1 = tf.layers.conv2d(inputs=self.tf_data, filters=self.n_filters, kernel_size=self.kernel_size,
                                 kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, padding='SAME')

        pool1 = tf.contrib.layers.max_pool2d(inputs=self.conv1, kernel_size=2, stride=2)

        self.conv2 = tf.layers.conv2d(inputs=pool1, filters=self.n_filters*2, kernel_size=self.kernel_size, 
                                 kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, padding='SAME')

        pool2 = tf.contrib.layers.max_pool2d(inputs=self.conv2, kernel_size=2, stride=2)

        self.conv3 = tf.layers.conv2d(inputs=pool2, filters=self.n_filters*4, kernel_size=self.kernel_size, 
                                    kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, padding='SAME')
                                  

        pool3 = tf.contrib.layers.max_pool2d(inputs=self.conv3, kernel_size=2, stride=2)

        self.conv4 = tf.layers.conv2d(inputs=pool3, filters=self.n_filters*8, kernel_size=self.kernel_size, 
                                      kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, padding='SAME')
        
        pool4 = tf.contrib.layers.max_pool2d(inputs=self.conv4, kernel_size=2, stride=2)

        self.deconv1 = tf.layers.conv2d_transpose(inputs=pool4, filters=self.n_filters*8, kernel_size=4, strides=2,
                                                 kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, padding='SAME')
   
        # deconv1_c = tf.concat([self.deconv1, self.conv4], axis=3)
        # deconv1_c = 0.5*tf.add(self.deconv1, self.conv4)

        self.deconv2 = tf.layers.conv2d_transpose(inputs=self.deconv1, filters=self.n_filters*4, kernel_size=4,
                                                  strides=2, kernel_regularizer=regularizer, 
                                                  activation=tf.nn.leaky_relu, padding='SAME')
        
        # deconv2_c = tf.concat([self.deconv2, self.conv3], axis=3)
        # deconv2_c = 0.5*tf.add(self.deconv2, self.conv3)
        
        self.deconv3 = tf.layers.conv2d_transpose(inputs=self.deconv2, filters=self.n_filters*2, kernel_size=4, strides=2, 
                                             kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, padding='SAME')        
        
        # deconv3_c = tf.concat([self.deconv3, self.conv2], axis=3)
        # deconv3_c = 0.5*tf.add(self.deconv3, self.conv2)
            
        self.deconv4 = tf.layers.conv2d_transpose(inputs=self.deconv3, filters=self.n_filters, kernel_size=4, strides=2, 
                                             kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, padding='SAME')
        
        # deconv4_c = tf.concat([self.deconv4, self.conv1], axis=3)
        # deconv4_c = 0.5*tf.add(self.deconv4, self.conv1)

        self.score_layer = tf.layers.conv2d(inputs=self.deconv4, filters=2, kernel_size=1,kernel_regularizer=regularizer)
        
        logits = tf.reshape(self.score_layer, (-1,2))
        y = tf.nn.softmax(logits)
        
        y_label = tf.multiply(tf.cast(self.tf_labels, tf.float32), self.tf_pen_road)
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.multiply(y_label, tf.log(y)), 1))

        # self.cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=self.tf_labels, logits=logits, weights=weights)
        self.reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_term = tf.contrib.layers.apply_regularization(regularizer, self.reg_variables)
        self.loss = self.reg_term + self.cross_entropy

        self.train_step = tf.train.AdamOptimizer(self.learning_rate, epsilon=1e-4).minimize(self.loss)
        self.preds = tf.argmax(logits,axis=1,output_type=tf.int32)
        
        if display_log:
            print("conv1 size", self.conv1.shape)
            print("pool1 size", pool1.shape)
            print("conv2 size", self.conv2.shape)
            print("pool2 size", pool2.shape)
            print("conv3 size", self.conv3.shape)
            print("pool3 size", pool3.shape)
            print("conv4 size", self.conv4.shape)
            print("pool4 size", pool4.shape)
            print("deconv1 size", self.deconv1.shape)
            #print("deconv1_c size", deconv1_c.shape)
            print("deconv2 size", self.deconv2.shape)
            #print("deconv2_c size", deconv2_c.shape)
            print("deconv3 size", self.deconv3.shape)
            #print("deconv3_c size", deconv3_c.shape)
            print("deconv4 size", self.deconv4.shape) 
            #print("deconv4_c size", deconv4_c.shape)
            print("score size", self.score_layer.shape)
        
        
    def _get_base_sets(self, path_train_dir='data/training', path_image='images', ratio=0.8):
        
        self.file_train, self.file_val = d_aug._get_ids_train_val(os.path.join(path_train_dir, path_image), ratio)
        x_train, y_train = rs.load_set_from_id(path_train_dir, self.file_train)
        x_valid, y_valid = rs.load_set_from_id(path_train_dir, self.file_val)
        _, self.mean, self.std = rs.normalize_data(x_train, mode='all') 
        
        return x_train, y_train, x_valid, y_valid
            
    def _get_train_validation(self, path_train_dir, epoch, ratio=0.8, n_aug=400):
    
        # Augment data. If first time. get id_train and validation

        if self.file_train is None or self.file_val is None:
            # Split data in train and validation with ratio and create n_aug new images
            self.file_train, self.file_val = d_aug.genererate_data(path_train_dir, ratio=ratio, 
                                                                   n_aug=n_aug, display_log=False, seed=0)
            # Load base train (no augmentation) and take mean and std values
            data_trainset, _ = rs.load_set_from_id(path_train_dir, self.file_train)
            _, self.mean, self.std = rs.normalize_data(data_trainset, mode='all') 
        else:
            d_aug.genererate_data_from_id(self.file_train, self.file_val, path_train_dir, n_aug=n_aug, 
                                          seed=epoch, display_log=False)

        train_imgs, train_gt, val_imgs, val_gt = rs.load_train_set(path_train_dir)
        train_imgs, _, _ = rs.normalize_data(train_imgs, mode='all', mean_ref = self.mean, std_ref = self.std) 
        val_imgs, _, _ = rs.normalize_data(val_imgs, mode='all', mean_ref = self.mean, std_ref = self.std) 

        return train_imgs, train_gt.astype(int), val_imgs, val_gt.astype(int)
    
    
    def _get_worst_predictions_img(self, train_imgs, train_gt, f1s, n_keep=50):
            id_sorted = np.argsort(f1s)[:n_keep]
            print('f1 worst, mean: {:.4f}'.format(np.mean(f1s[id_sorted])))
            return train_imgs[id_sorted], train_gt[id_sorted]
            
            
    def train_model(self, path_train_dir, 
                    n_epoch = 4, batch_size = 5, learning_rate_val = 1e-3, nmax=10, 
                    seed=0, display_epoch=10, n_aug=400, n_worst=50):

        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        np.random.seed(seed)
        
        loss_tr_time = np.empty(0)
        f1_tr_time = np.empty(0)
        loss_ts_time = np.empty(0)
        f1_ts_time = np.empty(0)
        

        with tf.Session() as sess:

            sess.run(init)

            for epoch in range(1, n_epoch+1):
                start_epoch = time.time()
                
                if epoch != 1:
                    # If it's not the first epoch we keep worst prediction and add them to set (improve preformences on thoses).
                    # The rest of the set is a new generation
                    train_imgs_worst, train_gt_worst = self._get_worst_predictions_img(
                        train_imgs, train_gt, f1s_all, n_keep=n_worst)
                    train_imgs, train_gt, val_imgs, val_gt = self._get_train_validation(path_train_dir, epoch, n_aug=n_aug)
                    train_imgs = np.concatenate((train_imgs, train_imgs_worst), axis=0) 
                    train_gt = np.concatenate((train_gt, train_gt_worst), axis=0)
                else:
                    # First epoch, get validation and train images/labels.
                    train_imgs, train_gt, val_imgs, val_gt = self._get_train_validation(path_train_dir, epoch, n_aug=n_aug)
                       
                # Train over train set using mutliple batches
                print('\nStart train with data shape: {}'.format(train_imgs.shape))
                indices = np.random.permutation(train_gt.shape[0])
                for batch_iter in range(int(np.ceil(train_gt.shape[0]/batch_size))):
                       
                    batch_idx = indices[batch_iter*batch_size:min((batch_iter+1)*batch_size, indices.shape[0])]
                    batch_x = train_imgs[batch_idx]
                    r = np.reshape(train_gt[batch_idx], [-1])
                    batch_y = self.one_hot_convert(np.reshape(train_gt[batch_idx], [-1]), 2)
                    batch_pen_road = rs.get_one_hot_penalization(train_gt[batch_idx])
                    
                    _, batch_loss, train_cross_entropy, train_reg_term = sess.run(
                        [self.train_step, self.loss, self.cross_entropy, self.reg_term], # Returns
                        feed_dict={self.tf_data : batch_x, self.tf_labels : batch_y, # Feed variables
                                   self.tf_pen_road : batch_pen_road,
                                   self.learning_rate : learning_rate_val, #})
                                   self.is_training : True})
                    
                    
                # Test prediction to select worst predictions
                loss_train, f1_train, f1s_all = self.predict_model_cgt(sess, train_imgs, train_gt, nmax=nmax)
                    
                # Save losses and f1 for training sets
                loss_tr_time = np.concatenate((loss_tr_time, [loss_train]), axis=0)
                f1_tr_time = np.concatenate((f1_tr_time, [f1_train]), axis=0)

                # Feed back trainset
                print("Recap epoch {} is {:.4f}s".format(epoch , time.time() - start_epoch))
                print("\t last minibatch, loss : ", batch_loss, "cross entropy : ", train_cross_entropy, 
                          "reg term : ", train_reg_term)
                print("\t Train set loss : ", loss_train, ", f1 : ", f1_train)
                
                if epoch % display_epoch == 0:
                    
                    loss_val, f1_val, _ = self.predict_model_cgt(sess, val_imgs, val_gt, nmax=nmax)
                    loss_ts_time = np.concatenate((loss_ts_time, [loss_val]), axis=0)
                    f1_ts_time = np.concatenate((f1_ts_time, [f1_val]), axis=0)
                    
                    print("\t Validation set loss : ", loss_val, ", f1 : ", f1_val)
                    

                if epoch == 120:
                    learning_rate_val = 1e-1*learning_rate_val
                    
                print("\n")
                
            # Save tensorflow variables
            str_date = datetime.datetime.now().strftime("%Y_%m_%d_%Hh%M")
            self.save_path_model =  os.path.join(self.path_models, str_date + '_' + self.model_type + '_model.ckpt')
            saver.save(sess, self.save_path_model)
            print("Model saved in file: %s" % self.save_path_model)
            
            # Save train and val f1 score / loss evolution
            self.save_path_stats =  os.path.join(self.path_models, str_date + '_' + self.model_type + '_stats.npy')
            np.save(self.save_path_stats, {'loss_tr': loss_tr_time, 'f1_tr': f1_tr_time, 
                                           'loss_ts': loss_ts_time, 'f1_ts': f1_ts_time,
                                           'epoch': display_epoch})
            print("Stats saved to file: %s" % self.save_path_stats)

            
    def predict_model_cgt(self, sess, imgs, gt, nmax=10):
        imgs_size = gt.shape[0]
        splits = np.linspace(0, imgs_size, 1+imgs_size//nmax).astype(int)

        pred_tot = np.empty(0)
        loss_tot = np.empty(0)

        for i in range(splits.shape[0]-1):
            # Get batches
            batch_img = imgs[splits[i]:splits[i+1]]
            batch_gt = gt[splits[i]:splits[i+1]]
            batch_len = splits[i+1]-splits[i]
            batch_labels = self.one_hot_convert(np.reshape(batch_gt, -1).astype(int),2)
            batch_pen_road = rs.get_one_hot_penalization(batch_gt)
            
            # Run model on batch
            pred, loss = sess.run([self.preds, self.loss], 
                               feed_dict={self.tf_data : batch_img, self.tf_labels : batch_labels, 
                                          self.tf_pen_road : batch_pen_road, #})
                                          self.is_training : False})
            # Concatenate prediction and loss
            pred_tot = np.concatenate((pred_tot, pred), axis=0)
            loss_tot = np.concatenate((loss_tot, [loss]), axis=0)

        f1_tot = f1_score(np.reshape(gt, -1), np.reshape(pred_tot, -1), average='macro')
        loss = np.mean(loss_tot)
        
        f1s = np.zeros(imgs.shape[0])
        for i in range(imgs.shape[0]):
            id_start = i*(imgs.shape[1]*imgs.shape[2])
            id_end = (i+1)*(imgs.shape[1]*imgs.shape[2])
            f1s[i] = f1_score(np.reshape(gt, -1)[id_start:id_end], np.reshape(pred_tot, -1)[id_start:id_end], average='macro')
        
        return loss, f1_tot, f1s
    
    
    def predict_model(self, sess, imgs, nmax=10):
        imgs_size = imgs.shape[0]
        splits = np.linspace(0, imgs_size, 1+imgs_size//nmax).astype(int)

        pred_tot = np.empty(0)

        for i in range(splits.shape[0]-1):
            # Get batches
            batch_img = imgs[splits[i]:splits[i+1]]
            # Run model on batch
            pred = sess.run(self.preds, feed_dict={self.tf_data : batch_img ,#})
                                                   self.is_training : False})
            # Concatenate prediction and loss
            pred_tot = np.concatenate((pred_tot, pred), axis=0)

        return pred_tot
            
        
    def apply_model(self, img, path, nmax=5):
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            saver.restore(sess, path)
            pred = self.predict_model(sess, img, nmax=nmax)

        return pred
    
    def get_model_layers(self, img, path):
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            saver.restore(sess, path)
          
            conv_1, conv_2, conv_3, conv_4, deconv_1, deconv_2, deconv_3, deconv_4, score = sess.run(
                [self.conv1, self.conv2, self.conv3, self.conv4, 
                 self.deconv1, self.deconv2, self.deconv3, self.deconv4, self.score_layer], 
                 feed_dict={self.tf_data : [img]})
            layers = {'conv_1': conv_1, 'conv_2': conv_2, 'conv_3': conv_3, 'conv_4': conv_4, 
                      'deconv_1': deconv_1, 'deconv_2': deconv_2, 'deconv_3': deconv_3, 'deconv_4': deconv_4, 
                      'score': score}
            layers = OrderedDict(sorted(layers.items()))
        return layers

    def predict_f1(self, gt, pred):
        return f1_score(np.reshape(gt, -1), np.reshape(pred, -1), average='macro') 
            
    def one_hot_convert(self, vector, num_classes=None):
        """ (From https://stackoverflow.com/questions/29831489/numpy-1-hot-array)
        Converts an input 1-D vector of integers into an output
        2-D array of one-hot vectors, where an i'th input value
        of j will set a '1' in the i'th row, j'th column of the
        output array.

        Example:
            v = np.array((1, 0, 4))
            one_hot_v = convertToOneHot(v)
            print one_hot_v

            [[0 1 0 0 0]
             [1 0 0 0 0]
             [0 0 0 0 1]]
        """

        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0

        if num_classes is None:
            num_classes = np.max(vector)+1
        else:
            assert num_classes > 0
            assert num_classes >= np.max(vector)

        result = np.zeros(shape=(len(vector), num_classes))
        result[np.arange(len(vector)), vector] = 1
        return result
    
    
    def plot_layers(self, im_src, layers):
        n_lines = np.ceil((len(layers)+1)/3).astype(int)
        plt.figure(figsize=(16, 5*n_lines))
        # Base image (one that went through the layers)
        plt.subplot(n_lines, 3, 1)
        plt.imshow(im_src); plt.title('Source image'); plt.axis('off')
        # Display all sub layers for viz
        for i, key in enumerate(layers.keys()):
            plt.subplot(n_lines, 3, i+2)
            # Take mean of all layers ... better solution ?
            im_layer = layers[key].mean(axis=3).squeeze()
            # Normalize in rnage [-1, 1] for viz coherence
            plt.imshow(im_layer/np.max(np.abs(im_layer)), cmap='viridis', vmin=-1, vmax=1); plt.title(key); plt.axis('off')
            
            
    def plot_stats(self, _file=None):
        
        if _file is not None:
            pass
        elif self.save_path_stats is not None:
            _file = self.save_path_stats
        else:
            print('No model path found')
            return
        
        r = np.load(_file)[()]
        plt.figure(figsize=(16,5))
        plt.suptitle('Statitics: {}'.format(_file))
        plt.subplot(1,2,1)
        plt.plot(np.arange(len(r['f1_tr'])), r['f1_tr'] ,'-g', label='train F1')
        plt.plot(r['epoch']*np.arange(len(r['f1_ts'])), r['f1_ts'] ,'-b', label='vlaidation F1')
        plt.xlabel('Epochs')
        plt.ylabel('F1')
        plt.ylim([0,1])
        plt.grid()
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(np.arange(len(r['loss_tr'])), r['loss_tr'] ,'-g', label='train loss')
        plt.plot(r['epoch']*np.arange(len(r['loss_ts'])), r['loss_ts'] ,'-b', label='vlaidation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.ylim([0,1])
        plt.grid()
        plt.legend()
        
        print('Max f1 train: {:.4f}, valid: {:.4f}'.format(np.max(r['f1_tr']), np.max(r['f1_ts'])))
