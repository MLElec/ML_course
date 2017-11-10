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

def sigmoid(xt, cst_exp=1e-15):
    """Sigmoid - fixed numerical errors"""
    #return 1.0 / (1 + np.exp(-xt))
    sig = np.exp(-np.logaddexp(0,-xt))
    sig[sig > 1 - cst_exp] =  1 - cst_exp
    sig[sig < cst_exp] = cst_exp
    return sig

def learning_by_gradient_descent(y, tx, w, gamma):
    """
     1 step of GD
    """
    loss = compute_loss(np.squeeze(y), np.squeeze(tx.dot(w)), loss_name='neg_log_likelihood')
    y = np.expand_dims(y, axis=0)
    gradient = _compute_gradient(y, tx, np.squeeze(w), loss_name='neg_log_likelihood')
    result =  gamma * gradient
    result = np.expand_dims(result, axis=1)
    w = w- result
    return loss, w

def penalized_logistic_regression(y, tx, w, lambda_, gamma):

    loss = compute_loss(np.squeeze(y), np.squeeze(tx.dot(w)), loss_name='neg_log_likelihood') + lambda_ * np.squeeze(w.T.dot(w))
    y = np.expand_dims(y, axis=0)
    gradient = _compute_gradient(y, tx, np.squeeze(w), loss_name='neg_log_likelihood');
    gradient = gradient + np.squeeze(2 * lambda_ * w)
    result =  gamma * gradient
    result = np.expand_dims(result, axis=1)
    w = w- result
    return loss, w

def logistic_regression_GD(y, x,max_iter, threshold, gamma, initial_w=None ):
    # init parameters
    losses = []

    tx = x
    if(initial_w == None):
        w = np.zeros((tx.shape[1], 1))
    else:
        w = initial_w;

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        if( np.isnan(loss)):
            break;
        # log info

        #if iter % 100 == 0:
        #    print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break;
    #print("STOP it={i}, loss={l} ".format(i=iter,l=loss))
    return loss, w;

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return logistic_regression_GD(y,tx, max_iter=max_iters, threshold=1e-8, gamma=gamma, initial_w=initial_w);

def logistic_regression_penalized_GD(y, x, max_iter, threshold, gamma, lambda_, initial_w=None):
    losses = []

    tx = x
    if(initial_w == None):
        w = np.zeros((tx.shape[1], 1))
    else:
        w = initial_w;
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = penalized_logistic_regression(y, tx, w, gamma, lambda_)
        if( np.isnan(loss)):
            break;
        # log info
        #if iter % 100 == 0:
        #    print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break;
    #print("STOP it={i}, loss={l} ".format(i=iter,l=loss))
    return loss, w;

def reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma):
    return logistic_regression_penalized_GD(y, tx, max_iter=max_iters, threshold=1e-8, gamma=gamma, lambda_=lambda_, initial_w=initial_w)

def test_logistic_GD(x, y, x_val, y_val, degrees, gamma):
    
    best_acc = 0
    best_degree = 0
    best_weights = []
    rmse_tr = []
    rmse_te = []
    for ind,degree in enumerate(degrees):
        degree = int(degree)

        # Get ploynomial
        phi_train = build_poly(x, degree)
        phi_test = build_poly(x_val, degree)

        loss, weights = logistic_regression_GD(y, phi_train, max_iter=500, gamma=gamma, threshold=1e-8)
        
        print("degree={d}".format(  d=degree))
        y_range_train = (1+np.sign(np.squeeze(phi_train.dot(weights))))/2
        y_range_val = (1+np.sign(np.squeeze(phi_test.dot(weights))))/2
        val_train = accuracy(np.squeeze(y), y_range_train)
        val_acc = accuracy(np.squeeze(y_val), y_range_val)
        
        print('train acc : ', val_train)
        print('validation acc : ', val_acc)
        
        if(val_acc > best_acc):
            best_acc = val_acc
            best_degree = degree
            best_weights = weights

    print('Best params for Least Squares : degree = ',best_degree, ', accuracy = ', best_acc)
    
    return best_weights, best_degree

def test_penalized_logistic_GD(x, y, x_val, y_val, degrees, gamma, lambdas):
    
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

        rmse_te = []
        update_rmse = False

        for ind, lambda_ in enumerate(lambdas):

            loss, weights = logistic_regression_penalized_GD(y, phi_train, max_iter=500, gamma=gamma, threshold=1e-8, lambda_=lambda_)

            print("degree={d}, lambda={l:.8f},".format(
                    d=degree, l=lambda_))

            print("degree={d}".format(  d=degree))
            y_range_train = (1+np.sign(np.squeeze(phi_train.dot(weights))))/2
            y_range_val = (1+np.sign(np.squeeze(phi_test.dot(weights))))/2
            val_train = accuracy(np.squeeze(y), y_range_train)
            val_acc = accuracy(np.squeeze(y_val), y_range_val)
            
            print('train acc : ', val_train)
            print('validation acc : ', val_acc)
        
            if(val_acc > best_acc):
                best_acc = val_acc
                best_degree = degree
                best_lambda = lambda_
                best_weights = weights
        

        # Plot the best obtained results

    print('Best params for Penalized Logistic regression : degree = ',best_degree, ', lambda = ',best_lambda,', accuracy = ', best_acc)
    
    return best_weights, best_degree, best_lambda

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function. - smart multiplication"""
    pred = sigmoid(tx.dot(w))
    pred = (pred - pred*pred) 
    tx2 =tx.T.copy();
    for i in range(0, len(pred)):
        tx2[:,i] =  pred[i] * tx2[:,i];
    return tx2.dot(tx)

def logistic_regression_Newton_step(y, tx, w):
    """Netwon method for logistic regression"""
    y = np.expand_dims(y, axis=0)
    loss = compute_loss(y, np.squeeze(tx.dot(w)) )
    gradient = _compute_gradient(y, tx, np.squeeze(w), loss_name='neg_log_likelihood')
    hessian = calculate_hessian(y, tx, w)
    return loss, gradient, hessian

def learning_by_newton_method(y, tx, w,gamma):
    """
    ONe step on Newton's method.
    """
    loss, gradient, hessian = logistic_regression_Newton_step(y, tx, w)
    result =gamma*np.linalg.solve(np.squeeze(hessian), np.squeeze(gradient.T))
    result = np.expand_dims(result, axis=1)
    w = w- result
    return loss, w

def logistic_regression_Newton(y, x, max_iter, threshold, gamma):
    # init parameters
    losses = []

    #    tx = np.c_[np.ones((y.shape[0], 1)), x]
    tx = x
    w = np.zeros((tx.shape[1], 1))
    # start the logistic regression
    for iter in range(max_iter):
        loss, w = learning_by_newton_method(y, tx, w, gamma)
        #break;
        if( np.isnan(loss)):
            break;
        # log info
        #if iter % 1 == 0:
            #print("Current iteration={i}, loss={l}".format(i=iter, l=np.squeeze(loss)))
        # converge criterion
        losses.append(loss)
        #print(loss)
        if len(losses) > 1 and losses[-2] - losses[-1] < threshold:
            break;
    print("STOP it={i}, loss={l} ".format(i=iter,l=loss))
    return w;

def test_logistic_Newton(x, y, x_val, y_val, degrees, gamma):
    
    best_acc = 0
    best_degree = 0
    best_weights = []
    rmse_tr = []
    rmse_te = []
    for ind,degree in enumerate(degrees):
        degree = int(degree)

        # Get ploynomial
        phi_train = build_poly(x, degree)
        phi_test = build_poly(x_val, degree)

        weights = logistic_regression_Newton(y, phi_train, max_iter=200, gamma=gamma, threshold=1e-8)

        mse_te = compute_loss(y_val, np.squeeze(phi_test.dot(weights)))
        rmse_te.append(np.sqrt(2*mse_te))

        print("degree={d}, Testing RMSE={te:.3f}".format(
                d=degree,  te=rmse_te[ind]))
        print('train acc : ', accuracy(y, np.squeeze(phi_train.dot(weights))))
        val_acc = accuracy(y_val, np.squeeze(phi_test.dot(weights)))
        print('validation acc : ', val_acc)
        
        if(val_acc > best_acc):
            best_acc = val_acc
            best_degree = degree
            best_weights = weights

    print('Best params for Least Squares : degree = ',best_degree, ', accuracy = ', best_acc)
    
    return best_weights, best_degree

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
    elif loss_name == 'neg_log_likelihood':
        return _logistic(y, ty) 
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

def _logistic(y, ty):
    y_pred = sigmoid(ty)
    return -np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))

def _compute_gradient(y, tx, w, loss_name='mse'):
    if loss_name == 'mse':
        e = y-tx.dot(w)
        return -1/(np.shape(y)[0])*tx.T.dot(e)
    elif loss_name == 'mae':
        e = y-tx.dot(w)
        return -1/(np.shape(y)[0])*(tx.T).dot(np.sign(e))
    elif loss_name == 'neg_log_likelihood':
        pred = sigmoid(tx.dot(w))
        grad = tx.T.dot(np.squeeze(pred - y))
        return grad
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
