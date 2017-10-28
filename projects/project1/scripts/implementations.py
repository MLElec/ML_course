import numpy as np
import matplotlib.pyplot as plt

def least_squares_GD(y, tx, initial_w=None, max_iters=100, gamma=0.1, loss_name='mse'):
    if initial_w is not None:
        w = initial_w
    else:
        w = np.zeros(np.shape(tx)[1])
        
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx.dot(w), loss_name)
        grad = _compute_gradient(y, tx, w, loss_name)
        w = w-gamma*grad/(n_iter+1)

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
        w = w-gamma*grad/(n_iter+1)

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

def plot_train_test(train_errors, test_errors, lambdas, degree):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set
    
    degree is just used for the title of the plot.
    """
    plt.semilogx(lambdas, train_errors, color='b', marker='*', label="Train error")
    plt.semilogx(lambdas, test_errors, color='r', marker='*', label="Test error")
    plt.xlabel("lambda")
    plt.ylabel("RMSE")
    plt.title("Ridge regression for polynomial degree " + str(degree))
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig("ridge_regression")

def test_least_squares(x, y, x_val, y_val, degrees, mode='normal'):
    
    best_acc = 0
    best_degree = 0
    #best_rmse_tr = []
    #best_rmse_te = []
    best_weights = []
    rmse_tr = []
    rmse_te = []
    for ind,degree in enumerate(degrees):
        degree = int(degree)

        # Get ploynomial
        phi_train = build_poly(x, degree)
        phi_test = build_poly(x_val, degree)


        #update_rmse = False

        if mode=='normal':
            mse_tr, weights = least_squares(y, phi_train)
        elif mode =='GD':
            mse_tr, weights = least_squares_GD(y, phi_train, initial_w=None, max_iters=200, gamma=0.01, loss_name='mse')
        elif mode =='SGD':
            mse_tr, weights = least_squares_SGD(y, phi_train, initial_w=None, max_iters=200, gamma=0.01, loss_name='mse')


        mse_te = compute_loss(y_val, phi_test.dot(weights))
        rmse_tr.append(np.sqrt(2*mse_tr))
        rmse_te.append(np.sqrt(2*mse_te))

        print("degree={d}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}".format(
                d=degree, tr=rmse_tr[ind], te=rmse_te[ind]))
        print('train acc : ', accuracy(y, phi_train.dot(weights)))
        val_acc = accuracy(y_val, phi_test.dot(weights))
        print('validation acc : ', val_acc)
        
        if(val_acc > best_acc):
            best_acc = val_acc
            best_degree = degree
            best_weights = weights
            #update_rmse = True
        
    # if(update_rmse):
    #    best_rmse_tr = rmse_tr
    #    best_rmse_te = rmse_te

        # Plot the best obtained results
    #plot_train_test(best_rmse_tr, best_rmse_te, lambdas, best_degree)

    print('Best params for Least Squares : degree = ',best_degree, ', accuracy = ', best_acc)
    
    return best_weights, best_degree
    
def test_ridge_regression(x, y, x_val, y_val, degrees, lambdas):
    
    best_acc = 0
    best_degree = 0
    best_lambda = 0
    best_rmse_tr = []
    best_rmse_te = []
    best_weights = []
    for degree in degrees:
        degree = int(degree)
        #lambdas = np.logspace(-7, 2, 20)

        # Split sets
        #x_train, x_test, y_train, y_test = split_data(x, y, ratio, seed)

        # Get ploynomial
        phi_train = build_poly(x, degree)
        phi_test = build_poly(x_val, degree)

        rmse_tr = []
        rmse_te = []
        update_rmse = False

        for ind, lambda_ in enumerate(lambdas):

            mse_tr, weights = ridge_regression(y, phi_train, lambda_)
            mse_te = compute_loss(y_val, phi_test.dot(weights))
            rmse_tr.append(np.sqrt(2*mse_tr))
            rmse_te.append(np.sqrt(2*mse_te))

            print("degree={d}, lambda={l:.3f}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}".format(
                    d=degree, l=lambda_, tr=rmse_tr[ind], te=rmse_te[ind]))
            print('train acc : ', accuracy(y, phi_train.dot(weights)))
            val_acc = accuracy(y_val, phi_test.dot(weights))
            print('validation acc : ', val_acc)

            if(val_acc > best_acc):
                best_acc = val_acc
                best_degree = degree
                best_lambda = lambda_
                best_weights = weights
                update_rmse = True
        
        if(update_rmse):
            best_rmse_tr = rmse_tr
            best_rmse_te = rmse_te

        # Plot the best obtained results
    plot_train_test(best_rmse_tr, best_rmse_te, lambdas, best_degree)

    print('Best params for Ridge regression : degree = ',best_degree, ', lambda = ',best_lambda,', accuracy = ', best_acc)
    
    return best_weights, best_degree, best_lambda


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


def sep_valid_train_data(x, y, ratio, seed=0):
    """Separates data x, y into training(size: 1-ratio) and validation(size: ratio) set """
    
    np.random.seed(seed)
    ids = np.random.permutation(y.shape[0])
    
    id_train = ids[:int(y.shape[0]*ratio)]
    id_validation = ids[int(y.shape[0]*ratio):]
    
    return y[id_train], x[id_train], y[id_validation], x[id_validation]