import numpy as np
from scripts.implementations import build_poly, compute_loss, least_squares, accuracy, ridge_regression

def augmented_feat_angle(x, id_col, labels):
    feat_angle = x[:, id_col]
    feat_sin = np.sin(feat_angle)
    feat_cos = np.cos(feat_angle)
    x_angle = np.concatenate((feat_cos, feat_sin), axis=1)
    
    label_cos_new = [ label + '_cos' for label in labels[id_col]]
    label_sin_new = [ label + '_sin' for label in labels[id_col]]
    label_cos_new.extend(label_sin_new)
    return np.concatenate((x, x_angle), axis=1), np.concatenate((labels, label_cos_new))


def add_nan_feature(tx, feature_number, labels):
    
    label_nan = []
    for i, feature in enumerate(feature_number):  
        feat_nan = np.expand_dims(2*np.isnan(tx.T[feature,:])-1,axis=1)
        tx = np.concatenate((tx, feat_nan),axis=1)
        label_nan.append('NAN-'+str(i))
    return tx,  np.concatenate((labels, label_nan))
   

def _best_lambda(y, x, degree=7, k_fold = 4, seed = 1):
    lambdas = np.logspace(-4, 0, 5)
    
    # split data in k fold
    k_indices = _build_k_indices(y, k_fold, seed)
    # define lists to tore the loss of training data and test data
    rmse_tr = np.empty(len(lambdas))
    rmse_te = np.empty(len(lambdas))
    
    for i, _lambda in enumerate(lambdas):
        _rmse_tr = np.empty(k_fold)
        _rmse_te = np.empty(k_fold)
        _acc = np.empty(k_fold)
        # Performs k-fold validation
        for k in range(k_fold):
            _acc[k], _rmse_tr[k], _rmse_te[k] = _cross_validation_step_ridge(y, x, k_indices, k, _lambda, degree)
            _rmse_tr[k], _rmse_te[k] = np.sqrt(_rmse_tr[k]), np.sqrt(_rmse_te[k])
        # Take mean value
        rmse_tr[i] = np.mean(_rmse_tr)
        rmse_te[i] = np.mean(_rmse_te)

    # cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    return lambdas[np.argmin(rmse_te)]


def cross_validation_ls(y, x, k_fold = 4, degree = 1, seed = 0):
    # split data in k fold
    k_indices = _build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    acc_tr = np.empty(k_fold)
    acc_te = np.empty(k_fold)
    rmse_tr = np.empty(k_fold)
    rmse_te = np.empty(k_fold)

    # Performs k-fold validation
    for k in range(k_fold):
        acc_tr[k], acc_te[k], rmse_tr[k], rmse_te[k] = _cross_validation_step_ls(y, x, k_indices, k, degree)
        rmse_tr[k], rmse_te[k] = np.sqrt(rmse_tr[k]), np.sqrt(rmse_te[k])
    # Take mean value
    return np.mean(acc_te), np.mean(rmse_tr), np.mean(rmse_te)


def _cross_validation_step_ls(y, x, k_indices, k, degree):
    """return the loss of ridge regression."""
    
    # Get indices for train and test sets
    _ind_train = np.delete(k_indices, k, axis=0).flatten()
    _ind_test = k_indices[k]
    
    # Build polynomial matrix
    _phi_train = build_poly(x[_ind_train], degree)
    _phi_test = build_poly(x[_ind_test], degree)
                            
    loss_tr, weights = least_squares(y[_ind_train], _phi_train)
    loss_te = compute_loss(y[_ind_test], _phi_test.dot(weights))
    acc_tr = accuracy(y[_ind_test], _phi_test.dot(weights))
    acc_te = accuracy(y[_ind_train], _phi_train.dot(weights))
    return acc_tr, acc_te, loss_tr, loss_te


def cross_validation_ridge(y, x, k_fold = 4, degree = 1, seed = 0):
    # split data in k fold
    k_indices = _build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    acc = np.empty(k_fold)
    rmse_tr = np.empty(k_fold)
    rmse_te = np.empty(k_fold)

    # Performs k-fold validation
    for k in range(k_fold):
        lambda_ = _best_lambda(y, x, degree, k_fold, seed)
        acc[k], rmse_tr[k], rmse_te[k] = _cross_validation_step_ridge(y, x, k_indices, k, lambda_, degree)
        rmse_tr[k], rmse_te[k] = np.sqrt(rmse_tr[k]), np.sqrt(rmse_te[k])
        print(lambda_, acc[k])
    # Take mean value
    return np.mean(acc), np.mean(rmse_tr), np.mean(rmse_te)


def _cross_validation_step_ridge(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    
    # Get indices for train and test sets
    _ind_train = np.delete(k_indices, k, axis=0).flatten()
    _ind_test = k_indices[k]
    
    # Build polynomial matrix
    _phi_train = build_poly(x[_ind_train], degree)
    _phi_test = build_poly(x[_ind_test], degree)
    
    # Compute ridge regression and loss for both train and test set
    loss_tr, weights = ridge_regression(y[_ind_train], _phi_train, lambda_)
    loss_te = compute_loss(y[_ind_test], _phi_test.dot(weights))
    
    acc = accuracy(y[_ind_test], _phi_test.dot(weights))
    return acc, loss_tr, loss_te


def _build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
    
