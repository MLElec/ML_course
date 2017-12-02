import tensorflow as tf
import datetime
import numpy as np
from sklearn.metrics import f1_score
import os

class Model:
    
    def __init__(self, reg = 1e-3, n_filters = 64, kernel_size=3):
                
        # Settings model
        self.reg = reg
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        
        self.build_model()
        
        
    def build_model(self):
        
        tf.reset_default_graph()

        # Plae holders
        self.learning_rate = tf.placeholder(tf.float32)
        self.tf_data = tf.placeholder(tf.float32,[None, None, None, 3])
        self.tf_labels = tf.placeholder(tf.int32,[None,2])
        self.keep_prob = tf.placeholder(tf.float32)
        
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg)
        class_weights = tf.constant([[1.0,1.0]]) 
        weights = tf.reduce_sum(class_weights * tf.cast(self.tf_labels, tf.float32), axis=1)

        conv1 = tf.layers.conv2d(inputs=self.tf_data, filters=self.n_filters, kernel_size=self.kernel_size,
                                 kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, padding='SAME')
        print("conv1 size", conv1.shape)

        pool1 = tf.contrib.layers.max_pool2d(inputs=conv1, kernel_size=2, stride=2)
        print("pool1 size", pool1.shape)

        conv2 = tf.layers.conv2d(inputs=pool1, filters=self.n_filters*2, kernel_size=self.kernel_size, 
                                 kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, padding='SAME')
        print("conv2 size", conv2.shape)

        pool2 = tf.contrib.layers.max_pool2d(inputs=conv2, kernel_size=2, stride=2)
        print("pool2 size", pool2.shape)

        conv3 =tf.nn.dropout(tf.layers.conv2d(inputs=pool2, filters=self.n_filters*4, kernel_size=self.kernel_size, 
                                              kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, padding='SAME'),
                             self.keep_prob) 
        print("conv3 size", conv3.shape)

        pool3 = tf.contrib.layers.max_pool2d(inputs=conv3, kernel_size=2, stride=2)
        print("pool3 size", pool3.shape)

        conv4 = tf.nn.dropout(tf.layers.conv2d(inputs=pool3, filters=self.n_filters*4, kernel_size=self.kernel_size, 
                                               kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, padding='SAME'),
                              self.keep_prob)
        print("conv4 size", conv4.shape)

        pool4 = tf.contrib.layers.max_pool2d(inputs=conv4, kernel_size=2, stride=2)
        print("pool4 size", pool4.shape)

        deconv1 = tf.nn.dropout(tf.layers.conv2d_transpose(inputs=pool4, filters=self.n_filters*4, kernel_size=4, strides=2,
                                                           kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, 
                                                           padding='SAME'), self.keep_prob)
        print("deconv1 size", deconv1.shape)

        deconv2 = tf.nn.dropout(tf.layers.conv2d_transpose(inputs=deconv1, filters=self.n_filters*4, kernel_size=4, strides=2,
                                                           kernel_regularizer=regularizer, activation=tf.nn.leaky_relu,
                                                           padding='SAME'), self.keep_prob)
        print("deconv2 size", deconv2.shape)

        deconv3 = tf.layers.conv2d_transpose(inputs=deconv2, filters=self.n_filters*2, kernel_size=4, strides=2, 
                                             kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, padding='SAME')
        print("deconv3 size", deconv3.shape)

        deconv4 = tf.layers.conv2d_transpose(inputs=deconv3, filters=self.n_filters, kernel_size=4, strides=2, 
                                             kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, padding='SAME')
        print("deconv4 size", deconv4.shape)

        score_layer = tf.layers.conv2d(inputs=deconv4, filters=2, kernel_size=1,kernel_regularizer=regularizer)
        print("score size", score_layer.shape)

        logits = tf.reshape(score_layer, (-1,2))

        self.cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=self.tf_labels, logits=logits, weights=weights)
        self.reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_term = tf.contrib.layers.apply_regularization(regularizer, self.reg_variables)
        self.loss = self.reg_term + self.cross_entropy

        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.preds = tf.argmax(logits,axis=1,output_type=tf.int32)
        
        
    def train_model(self, useful_patches_tr, useful_lab_tr, train_imgs, train_gt, val_imgs, val_gt, 
                    n_epoch = 4, batch_size = 5, learning_rate_val = 1e-3, path_models='model'):
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
        loss_time = np.empty((0, 2))
        f1_time = np.empty((0, 2))

        with tf.Session() as sess:

            sess.run(init)

            for epoch in range(1, n_epoch+1):
                indices = np.random.permutation(useful_lab_tr.shape[0])

                for batch_iter in range(int(np.ceil(useful_lab_tr.shape[0]/batch_size))):

                    batch_idx = indices[batch_iter*batch_size:min((batch_iter+1)*batch_size, indices.shape[0])]
                    batch_x = useful_patches_tr[batch_idx]
                    batch_y = self.one_hot_convert(np.reshape(useful_lab_tr[batch_idx], [-1]))

                    _, batch_loss, train_cross_entropy, train_reg_term = sess.run(
                        [self.train_step, self.loss, self.cross_entropy, self.reg_term], # Returns
                        feed_dict={self.tf_data : batch_x, self.tf_labels : batch_y, # Feed variables
                                   self.keep_prob : 0.8, self.learning_rate : learning_rate_val})

                last_f1 =0
                if epoch % 1 ==0:

                    loss_train, f1_train = self.predict_model_cgt(sess, train_imgs, train_gt)
                    loss_val, f1_val = self.predict_model_cgt(sess, val_imgs, val_gt)

                    print("Recap epoch ", epoch)
                    print("\t last minibatch, cross entropy : ", train_cross_entropy, "reg term : ", train_reg_term)
                    print("\t val_loss : ", loss_val, ", train_loss : ", loss_train)
                    print("\t val f1 : ", f1_val, ", train f1 : ", f1_train)
                    loss_time = np.concatenate((loss_time, [[loss_train, loss_val]]), axis=0)
                    f1_time = np.concatenate((loss_time, [[f1_train, f1_val]]), axis=0)

                    if f1_train < last_f1:
                        learning_rate_val/=2
                    last_f1 = f1_train

            str_date = datetime.datetime.now().strftime("%Y_%m_%d_%Hh%M")
            self.save_path =  os.path.join(path_models, str_date + '_model.ckpt')
            saver.save(sess, self.save_path)
            print("Model saved in file: %s" % self.save_path)

            
    def predict_model_cgt(self, sess, imgs, gt, nmax=5):
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
            # Run model on batch
            pred, loss = sess.run([self.preds, self.loss], 
                               feed_dict={self.tf_data : batch_img, self.tf_labels : batch_labels, self.keep_prob : 1})
            # Concatenate prediction and loss
            pred_tot = np.concatenate((pred_tot, pred), axis=0)
            loss_tot = np.concatenate((loss_tot, [loss]), axis=0)

        f1 = f1_score(np.reshape(gt, -1), np.reshape(pred_tot, -1), average='macro') 
        loss = np.mean(loss_tot)
        return loss, f1
    
    
    def predict_model(self, sess, imgs, nmax=5):
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
            
        
    def apply_model(self, img, path):
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            saver.restore(sess, path)
            pred = self.predict_model(sess, img)

        return pred

            
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
        