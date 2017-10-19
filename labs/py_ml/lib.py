import numpy as np


## *********** Loss function ***********
def compute_loss(y, ty):
    """Compute loss"""
    return mse(y, ty)

def mse(y, ty):
    e = y - ty
    loss = e.dot(e.T) / (2 * len(e))
    return loss

## *********** Grid Search ***********
def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    losses = np.zeros((len(w0), len(w1)))
    
    for i in range(len(w0)):
        for j in range(len(w1)):
            w = np.array([w0[i], w1[j]]).T
            losses[i,j] = compute_loss(y, tx.dot(w))
    return losses

## *********** Gradient Descent ***********
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y-tx.dot(w)
    return -1/(np.shape(y)[0])*tx.T.dot(e)

def gradient_descent(y, tx, initial_w=None, max_iters=100, gamma=0.1, display_output=False):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    if initial_w is not None:
        w = initial_w
    else:
        w = np.zeros(np.shape(tx)[1])
        
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx.dot(w))
        grad = compute_gradient(y, tx, w)
        w = w-gamma*grad
        if display_output:
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return loss, w

## *********** Stochastic Gradient Descent ***********
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    return compute_gradient(y, tx, w)
    

def stochastic_gradient_descent(y, tx, initial_w=None, batch_size=100, max_iters=100, gamma=0.5, display_output=False):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    if initial_w is not None:
        w = initial_w
    else:
        w = np.zeros(np.shape(tx)[1])
        
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx.dot(w))
        (y_st, tx_st) = [batch for batch in batch_iter(y, tx, batch_size)][0]
        grad = compute_gradient(y_st, tx_st, w)
        w = w-gamma*grad
        if display_output:
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter+1, ti=max_iters, l=loss, w0=w[0], w1=w[1]))
    return loss, w

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
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

def least_squares(y, tx):
    w = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    #w = np.linalg.solve(tx, y)
    loss = mse(y, tx.dot(w))
    return loss, w

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    _x = np.ones((np.shape(x)[0], degree+1))
    for i in range(1, degree+1):
        _x[:, i] = x**i
    return _x

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    Lambda_ = lambda_*np.eye(tx.shape[1])*(2*len(y))
    w = np.linalg.inv(tx.T.dot(tx) + Lambda_.dot(Lambda_)).dot(tx.T).dot(y)
    loss = mse(y, tx.dot(w))
    return loss, w

def split_data(x, y, ratio, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row * ratio)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    return x[indices[:interval]], x[indices[interval:]], y[indices[:interval]], y[indices[interval:]]