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

def add_feature(x_in, id_feat1, id_feat2):
    if id_feat1 != id_feat2:
        new_feat = np.expand_dims(x_in[:, id_feat1]*x_in[:, id_feat2], axis=1)
    else:
        new_feat = np.expand_dims(x_in[:, id_feat1], axis=1)
    return np.concatenate((x_in, new_feat), axis=1)

def add_features(x_in, id_feats):
    for id_feat in id_feats:
        x_in = add_feature(x_in, id_feat[0], id_feat[1])
    return x_in

def remove_useless(x_in, id_useless):
    id_left = [ i for i in range(x_in.shape[1]) if i not in id_useless]
    return x_in[:, id_left]
   
def norm_poisson(feature, perc_threshold=0.01):
    length = feature.shape[0];
    idx_val = np.int(np.ceil(length*perc_threshold))
    maxval = np.sort(feature)[-idx_val]
    
    #idx_outliers = np.argsort(feature)[-idx_val:]
    #maxval = feature[np.argsort(feature)[ -(idx_val+1) ]]
    
    mean = np.nanmean(feature[feature < maxval])
    std = np.nanstd(feature[feature < maxval])
    feature[feature > maxval] = maxval
    feature -= mean
    feature /= std
    return feature, mean, std, maxval

def norm_poisson_feed(feature, mean_ref, std_ref, maxval):    
    feature[feature > maxval] = maxval
    feature -= mean_ref
    feature /= std_ref
    return feature

def norm_gaussian(feature, n_std=2.5):
    
    feat_cent = feature-np.nanmean(feature)
    std_thresh = np.nanstd(feat_cent, axis=0)
    maxval = n_std*std_thresh
    
    mean_update = np.nanmean(feature[np.abs(feat_cent) < maxval])
    std_update = np.nanstd(feature[np.abs(feat_cent) < maxval])
    feat_final = feature-mean_update
    feat_final[feat_final > maxval] = maxval
    feat_final[feat_final < -maxval] = -maxval
    feat_final /= std_update
    
    #feature[feature > maxval] = maxval
    #feature[feature < -maxval] = -maxval
    
    #mean_update = np.nanmean(feature)
    #std_update = np.nanstd(feature)
    #feature = (feature-mean_update)/std_update
        
    #return feature, mean_update, std_update, maxval

    return feat_final, mean_update, std_update, maxval

def norm_gaussian_feed(feature, mean_ref, std_ref, maxval):    
    feat_final = feature - mean_ref
    feat_final[feat_final > maxval]  = maxval
    feat_final[feat_final < -maxval] = -maxval
    feat_final /= std_ref
    return feat_final
    
def normalize_outliers(x_in, dist_type):
    # 1. Substract mean
    # 2. Compute std and detect ouliers
    # 3. Compute std and mean witout ouliers
    mean_corr = []
    std_corr = []
    max_val_corr = []
                
    for i, feat in enumerate(x_in.T):
        # Normalize according to distribution
        if dist_type[i] == 'g' or dist_type[i] == 'i' or dist_type[i] == 'u' \
                or dist_type[i] == 'f' or dist_type[i] == 'd':
            feat_new, mean_new, std_new, max_val_new = norm_gaussian(feat)
        elif dist_type[i] == 'p':
            feat_new, mean_new, std_new, max_val_new = norm_poisson(feat)
        else:
            feat_new, mean_new, std_new, max_val_new = (feat, 0, 1, np.inf)
        # Affect new values
        mean_corr.append(mean_new)
        std_corr.append(std_new)
        max_val_corr.append(max_val_new)
        x_in[:, i] = feat_new
    return x_in, mean_corr, std_corr, max_val_corr

def normalize_outliers_feed(x_in, mean_ref, std_ref, max_ref, dist_type):
    # 1. Substract mean
    # 2. Compute std and detect ouliers
    # 3. Compute std and mean witout ouliers
    
    for i, feat in enumerate(x_in.T):
        # Normalize according to distribution
        if dist_type[i] == 'g' or dist_type[i] == 'i' or dist_type[i] == 'u' \
                or dist_type[i] == 'f' or dist_type[i] == 'd':
            feat_new = norm_gaussian_feed(feat, mean_ref[i], std_ref[i], max_ref[i])
        elif dist_type[i] == 'p':
            feat_new = norm_poisson_feed(feat, mean_ref[i], std_ref[i], max_ref[i])
        else:
            feat_new = feat
        # Affect new value
        x_in[:, i] = feat_new
    return x_in

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
    
