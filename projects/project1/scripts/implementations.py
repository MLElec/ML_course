import numpy as np


def least_squares_GD(y, tx, initial_w=None, max_iters=100, gamma=0.1):
    if initial_w is not None:
        w = initial_w
    else:
        w = np.zeros(np.shape(tx)[1])
        
    for n_iter in range(max_iters):
        loss = _mse(y, tx.dot(w))
        grad = _compute_gradient(y, tx, w)
        w = w-gamma*grad

    print("Gradient Descent {} iter: loss={}".format(max_iters, loss))
    return loss, w

def least_squares_SGD(y, tx, initial_w=None, max_iters=100, gamma=0.5):
    batch_size=100
    if initial_w is not None:
        w = initial_w
    else:
        w = np.zeros(np.shape(tx)[1])
        
    for n_iter in range(max_iters):
        (y_st, tx_st) = [batch for batch in _batch_iter(y, tx, batch_size)][0]
        loss = _mse(y, tx.dot(w))
        grad = _compute_gradient(y_st, tx_st, w)
        w = w-gamma*grad

    print("SGD (batch, iter) = ({}, {}): loss={}".format(batch_size, n_iter, loss))
    return loss, w

def least_squares(y, tx):
    w = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    loss = _mse(y, tx.dot(w))
    return loss, w

def ridge_regression(y, tx, lambda_):
    Lambda_ = lambda_*np.eye(tx.shape[1])*(2*len(y))
    w = np.linalg.inv(tx.T.dot(tx) + Lambda_.dot(Lambda_)).dot(tx.T).dot(y)
    loss = _mse(y, tx.dot(w))
    return loss, w

def _mse(y, ty):
    e = y - ty
    loss = 1/(2*np.shape(y)[0])*e.dot(e.T)
    return loss

def _compute_gradient(y, tx, w):
    e = y-tx.dot(w)
    return -1/(np.shape(y)[0])*tx.T.dot(e)

def _compute_stoch_gradient(y, tx, w):
    return compute_gradient(y, tx, w)

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

