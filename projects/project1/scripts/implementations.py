import numpy as np


def least_squares_GD(y, tx, initial_w=None, max_iters=100, gamma=0.1, loss_name='mse'):
    if initial_w is not None:
        w = initial_w
    else:
        w = np.zeros(np.shape(tx)[1])
        
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx.dot(w), loss_name)
        grad = _compute_gradient(y, tx, w, loss_name)
        w = w-gamma*grad

    print("Gradient Descent {} iter: loss={}".format(max_iters, loss))
    return loss, w

def least_squares_SGD(y, tx, initial_w=None, max_iters=100, gamma=0.5, loss_name='mse'):
    batch_size=100
    if initial_w is not None:
        w = initial_w
    else:
        w = np.zeros(np.shape(tx)[1])
        
    for n_iter in range(max_iters):
        (y_st, tx_st) = [batch for batch in _batch_iter(y, tx, batch_size)][0]
        loss = compute_loss(y, tx.dot(w), loss_name)
        grad = _compute_gradient(y_st, tx_st, w, loss_name)
        w = w-gamma*grad

    print("SGD (batch, iter) = ({}, {}): loss={}".format(batch_size, n_iter, loss))
    return loss, w

def least_squares(y, tx):
    w = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    loss = compute_loss(y, tx.dot(w))
    return loss, w

def ridge_regression(y, tx, lambda_):
    Lambda_ = lambda_*np.eye(tx.shape[1])*(2*len(y))
    w = np.linalg.inv(tx.T.dot(tx) + Lambda_.dot(Lambda_)).dot(tx.T).dot(y)
    loss = compute_loss(y, tx.dot(w))
    return loss, w

def build_poly(x, degree):
    """ polynomial basis functions for input data x, for j=0 up to j=degree. If x as multiple columns (feaures) 
        it build [ 1 x0 x1 ... xn x0**2 x1**2 ... xn**j]"""
    _x = np.ones((np.shape(x)[0], 1 + np.shape(x)[1]*degree))
    for i in range(0, degree):
        for k in range(0, np.shape(x)[1]):
            _id = 1 + i*np.shape(x)[1] + k
            _x[:, _id] = x[:, k]**(i+1)
    return _x

def accuracy(y_cgt, y_pred):
    y_pred_s = np.sign(y_pred)
    tp = np.sum(y_cgt == y_pred_s)
    return tp/len(y_cgt)

def compute_loss(y, ty, loss_name='mse'):
    if loss_name == 'mse':
        return _mse(y, ty)
    elif loss_name == 'mae':
        return _mae(y, ty)
    else:
        raise NotImplementedError
    
def _mse(y, ty):
    e = y - ty
    loss = 1/(2*np.shape(y)[0])*e.dot(e.T)
    return loss

def _mae(y, ty):
    e = np.abs(y-ty)
    loss = 1/(np.shape(y)[0])*np.sum(e)
    return loss

def _compute_gradient(y, tx, w, loss_name='mse'):
    if loss_name == 'mse':
        e = y-tx.dot(w)
        return -1/(np.shape(y)[0])*tx.T.dot(e)
    elif loss_name == 'mae':
        e = y-tx.dot(w)
        return -1/(np.shape(y)[0])*(tx.T).dot(np.sign(e))
    else:
        raise NotImplementedError

def _compute_stoch_gradient(y, tx, w, loss_name='mse'):
    return compute_gradient(y, tx, w, loss_name)


def _batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]