import tensorflow as tf
import datetime
import numpy as np
from sklearn.metrics import f1_score
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import ml_utils.road_seg as rs
import time

class Model:
    
    def __init__(self, reg = 1e-2, n_filters = 64, kernel_size=3, display_log=True, path_models='model'):
                
        # Settings model
        self.reg = reg
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        
        # Path
        self.save_path_model = None
        self.save_path_stats = None
        self.path_models = path_models
        
        if not os.path.exists(path_models):
            os.mkdir(path_models)
        
        self.build_model(display_log)
        
        
    def build_model(self, display_log, seed=0):
        
        tf.reset_default_graph()
        tf.set_random_seed(seed)

        # Plae holders
        self.learning_rate = tf.placeholder(tf.float32)
        self.tf_data = tf.placeholder(tf.float32,[None, None, None, 3])
        self.tf_labels = tf.placeholder(tf.int32,[None,2])
        self.keep_prob = tf.placeholder(tf.float32)
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

        self.conv3 = tf.nn.dropout(tf.layers.conv2d(inputs=pool2, filters=self.n_filters*4, kernel_size=self.kernel_size, 
                                    kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, padding='SAME'), 
                                   self.keep_prob)
                                  

        pool3 = tf.contrib.layers.max_pool2d(inputs=self.conv3, kernel_size=2, stride=2)

        self.conv4 = tf.nn.dropout(tf.layers.conv2d(inputs=pool3, filters=self.n_filters*8, kernel_size=self.kernel_size, 
                                      kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, padding='SAME'), 
                                   self.keep_prob)
        
        pool4 = tf.contrib.layers.max_pool2d(inputs=self.conv4, kernel_size=2, stride=2)

        self.deconv1 = tf.nn.dropout(tf.layers.conv2d_transpose(inputs=pool4, filters=self.n_filters*8, kernel_size=4, strides=2,
                                                 kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, padding='SAME'),
                                     self.keep_prob)
   
        # deconv1_c = tf.concat([self.deconv1, self.conv4], axis=3)
        # deconv1_c = 0.5*tf.add(self.deconv1, self.conv4)

        self.deconv2 = tf.nn.dropout(tf.layers.conv2d_transpose(inputs=self.deconv1, filters=self.n_filters*4, kernel_size=4,
                                                  strides=2, kernel_regularizer=regularizer, 
                                                  activation=tf.nn.leaky_relu, padding='SAME'), 
                                     self.keep_prob)
        
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
        #y = tf.nn.softmax(logits)
        
        #y_label = tf.multiply(tf.cast(self.tf_labels, tf.float32), self.tf_pen_road)
        #self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.multiply(y_label, tf.log(y)), 1))

        self.cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=self.tf_labels, logits=logits, weights=weights)
        self.reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_term = tf.contrib.layers.apply_regularization(regularizer, self.reg_variables)
        self.loss = self.reg_term + self.cross_entropy

        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
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
        
        
    def train_model(self, useful_patches_tr, useful_lab_tr, train_imgs, train_gt, val_imgs, val_gt, 
                    n_epoch = 4, batch_size = 5, learning_rate_val = 5e-4, nmax=10, seed=0, display_epoch=10):
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
        loss_time = np.empty((0, 2))
        f1_time = np.empty((0, 2))

        with tf.Session() as sess:

            sess.run(init)

            for epoch in range(1, n_epoch+1):
                indices = np.random.permutation(useful_lab_tr.shape[0])
                start_epoch = time.time()
                
                for batch_iter in range(int(np.ceil(useful_lab_tr.shape[0]/batch_size))):

                    batch_idx = indices[batch_iter*batch_size:min((batch_iter+1)*batch_size, indices.shape[0])]
                    batch_x = useful_patches_tr[batch_idx]
                    batch_y = self.one_hot_convert(np.reshape(useful_lab_tr[batch_idx], [-1]))
                    batch_pen_road = rs.get_one_hot_penalization(useful_lab_tr[batch_idx])

                    _, batch_loss, train_cross_entropy, train_reg_term = sess.run(
                        [self.train_step, self.loss, self.cross_entropy, self.reg_term], # Returns
                        feed_dict={self.tf_data : batch_x, self.tf_labels : batch_y, # Feed variables
                                   self.keep_prob : 1, self.tf_pen_road : batch_pen_road,
                                   self.learning_rate : learning_rate_val})

                if epoch % display_epoch == 0:

                    loss_train, f1_train = self.predict_model_cgt(sess, train_imgs, train_gt, nmax=nmax)
                    loss_val, f1_val = self.predict_model_cgt(sess, val_imgs, val_gt, nmax=nmax)

                    print("Recap epoch {} is {:.4f}s".format(epoch , time.time() - start_epoch))
                    print("\t last minibatch, loss : ", batch_loss, "cross entropy : ", train_cross_entropy, 
                          "reg term : ", train_reg_term)
                    print("\t val_loss : ", loss_val, ", train_loss : ", loss_train)
                    print("\t val f1 : ", f1_val, ", train f1 : ", f1_train)
                    loss_time = np.concatenate((loss_time, [[loss_train, loss_val]]), axis=0)
                    f1_time = np.concatenate((f1_time, [[f1_train, f1_val]]), axis=0)

                if epoch == 180:
                    learning_rate_val = 1e-1*learning_rate_val

            # Save tensorflow variables
            str_date = datetime.datetime.now().strftime("%Y_%m_%d_%Hh%M")
            self.save_path_model =  os.path.join(self.path_models, str_date + '_model.ckpt')
            saver.save(sess, self.save_path_model)
            print("Model saved in file: %s" % self.save_path_model)
            
            # Save train and val f1 score / loss evolution
            self.save_path_stats =  os.path.join(self.path_models, str_date + '_stats.npy')
            np.save(self.save_path_stats, {'loss': loss_time, 'f1': f1_time})
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
                                          self.tf_pen_road : batch_pen_road, self.keep_prob : 1})
            # Concatenate prediction and loss
            pred_tot = np.concatenate((pred_tot, pred), axis=0)
            loss_tot = np.concatenate((loss_tot, [loss]), axis=0)

        f1 = f1_score(np.reshape(gt, -1), np.reshape(pred_tot, -1), average='macro') 
        loss = np.mean(loss_tot)
        return loss, f1
    
    
    def predict_model(self, sess, imgs, nmax=10):
        imgs_size = imgs.shape[0]
        splits = np.linspace(0, imgs_size, 1+imgs_size//nmax).astype(int)

        pred_tot = np.empty(0)

        for i in range(splits.shape[0]-1):
            # Get batches
            batch_img = imgs[splits[i]:splits[i+1]]
            # Run model on batch
            pred = sess.run(self.preds, feed_dict={self.tf_data : batch_img, self.keep_prob : 1})
            # Concatenate prediction and loss
            pred_tot = np.concatenate((pred_tot, pred), axis=0)

        return pred_tot
            
        
    def apply_model(self, img, path, nmax=10):
        
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
                 feed_dict={self.tf_data : [img], self.keep_prob : 1})
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
        plt.plot(np.arange(len(r['f1'])), r['f1'][:,0] ,'-g', label='train F1')
        plt.plot(np.arange(len(r['f1'])), r['f1'][:,1] ,'-b', label='vlaidation F1')
        plt.xlabel('Epochs')
        plt.ylabel('F1')
        plt.grid()
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(np.arange(len(r['loss'])), r['loss'][:,0] ,'-g', label='train loss')
        plt.plot(np.arange(len(r['loss'])), r['loss'][:,1] ,'-b', label='vlaidation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()
