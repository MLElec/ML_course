import tensorflow as tf
import datetime
import numpy as np
from sklearn.metrics import f1_score
import os

def one_hot_convert(vector, num_classes=None):
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

def run_model(useful_patches_tr, useful_lab_tr, train_imgs, train_gt, val_imgs, val_gt, path_models='model', 
              n_epoch = 4, batch_size = 5, learning_rate_val = 1e-3, reg = 1e-3, n_filters = 64, kernel_size=3):
    
    train_size = train_gt.shape[0]
    val_size = val_gt.shape[0]

    learning_rate = tf.placeholder(tf.float32)
    
    tf_data = tf.placeholder(tf.float32,[None, None, None, 3])
    tf_labels = tf.placeholder(tf.int32,[None,2])
    keep_prob = tf.placeholder(tf.float32)
    regularizer = tf.contrib.layers.l2_regularizer(scale=reg)

    class_weights = tf.constant([[1.0,1.0]]) 
    weights = tf.reduce_sum(class_weights * tf.cast(tf_labels, tf.float32), axis=1)

    conv1 = tf.layers.conv2d(inputs=tf_data, filters=n_filters, kernel_size=kernel_size,
                             kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, padding='SAME')
    print("conv1 size", conv1.shape)

    pool1 = tf.contrib.layers.max_pool2d(inputs=conv1, kernel_size=2, stride=2)
    print("pool1 size", pool1.shape)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=n_filters*2, kernel_size=kernel_size, 
                             kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, padding='SAME')
    print("conv2 size", conv2.shape)

    pool2 = tf.contrib.layers.max_pool2d(inputs=conv2, kernel_size=2, stride=2)
    print("pool2 size", pool2.shape)

    conv3 =tf.nn.dropout(tf.layers.conv2d(inputs=pool2, filters=n_filters*4, kernel_size=kernel_size, 
                                          kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, padding='SAME'),keep_prob) 
    print("conv3 size", conv3.shape)

    pool3 = tf.contrib.layers.max_pool2d(inputs=conv3, kernel_size=2, stride=2)
    print("pool3 size", pool3.shape)

    conv4 = tf.nn.dropout(tf.layers.conv2d(inputs=pool3, filters=n_filters*4, kernel_size=kernel_size, 
                                           kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, padding='SAME'),keep_prob)
    print("conv4 size", conv4.shape)

    pool4 = tf.contrib.layers.max_pool2d(inputs=conv4, kernel_size=2, stride=2)
    print("pool4 size", pool4.shape)

    deconv1 = tf.nn.dropout(tf.layers.conv2d_transpose(inputs=pool4, filters=n_filters*4, kernel_size=4, strides=2,
                                                       kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, 
                                                       padding='SAME'), keep_prob)
    print("deconv1 size", deconv1.shape)

    deconv2 = tf.nn.dropout(tf.layers.conv2d_transpose(inputs=deconv1, filters=n_filters*4, kernel_size=4, strides=2,
                                                       kernel_regularizer=regularizer, activation=tf.nn.leaky_relu,
                                                       padding='SAME'),keep_prob)
    print("deconv2 size", deconv2.shape)

    deconv3 = tf.layers.conv2d_transpose(inputs=deconv2, filters=n_filters*2, kernel_size=4, strides=2, 
                                         kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, padding='SAME')
    print("deconv3 size", deconv3.shape)

    deconv4 = tf.layers.conv2d_transpose(inputs=deconv3, filters=n_filters, kernel_size=4, strides=2, 
                                         kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, padding='SAME')
    print("deconv4 size", deconv4.shape)

    score_layer = tf.layers.conv2d(inputs=deconv4, filters=2, kernel_size=1,kernel_regularizer=regularizer)
    print("score size", score_layer.shape)

    logits = tf.reshape(score_layer, (-1,2))

    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=tf_labels, logits=logits, weights=weights)
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    loss = reg_term + cross_entropy

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    preds = tf.argmax(logits,axis=1,output_type=tf.int32)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(1,n_epoch+1):
            indices = np.random.permutation(useful_lab_tr.shape[0])

            for batch_iter in range(int(np.ceil(useful_lab_tr.shape[0]/batch_size))):

                batch_idx = indices[batch_iter*batch_size:min((batch_iter+1)*batch_size, indices.shape[0])]
                batch_x = useful_patches_tr[batch_idx]
                batch_y = one_hot_convert(np.reshape(useful_lab_tr[batch_idx], [-1]))

                _, train_loss, train_cross_entropy, train_reg_term = sess.run([train_step, loss, cross_entropy, reg_term], feed_dict={tf_data : batch_x, tf_labels : batch_y, keep_prob : 0.8, learning_rate : learning_rate_val})

            last_f1 =0
            if epoch % 1 ==0:
                
                loss_train, f1_train = predict_model(sess, preds, loss, tf_data, tf_labels, keep_prob, train_imgs, train_gt)
                loss_val, f1_val = predict_model(sess, preds, loss, tf_data, tf_labels, keep_prob, val_imgs, val_gt)
                
                print("Recap epoch ", epoch)
                print("\t last minibatch, cross entropy : ", train_cross_entropy, "reg term : ", train_reg_term)
                print("\t val_loss : ", loss_val, ", train_loss : ", loss_train)
                print("\t val f1 : ", f1_val, ", train f1 : ", f1_train)

                if f1_train < last_f1:
                    learning_rate_val/=2
                last_f1 = f1_train

        str_date = datetime.datetime.now().strftime("%Y_%m_%d_%Hh%M")
        save_path = saver.save(sess, os.path.join(path_models, str_date + '_model.ckpt'))
        print("Model saved in file: %s" % save_path)
        
        
def predict_model(sess, preds_tf, loss_tf, tf_data, tf_labels, keep_prob, imgs, gt, nmax=5, size_out=400):
    imgs_size = gt.shape[0]
    splits = np.linspace(0, imgs_size, 1+imgs_size//nmax).astype(int)
    
    pred_tot = np.empty(0)
    loss_tot = np.empty(0)
    
    for i in range(splits.shape[0]-1):
        # Get batches
        batch_img = imgs[splits[i]:splits[i+1]]
        batch_gt = gt[splits[i]:splits[i+1]]
        batch_len = splits[i+1]-splits[i]
        batch_labels = one_hot_convert(np.reshape(batch_gt, batch_len*size_out*size_out).astype(int),2)
        # Run model on batch
        pred, loss = sess.run([preds_tf, loss_tf], 
                           feed_dict={tf_data : batch_img, tf_labels : batch_labels, keep_prob : 1})
        # Concatenate prediction and loss
        pred_tot = np.concatenate((pred_tot, pred), axis=0)
        loss_tot = np.concatenate((loss_tot, [loss]), axis=0)
        
    f1 = f1_score(np.reshape(gt, imgs_size*size_out*size_out), np.reshape(pred_tot, [-1]), average='macro') 
    loss = np.mean(loss_tot)
    return loss, f1
        