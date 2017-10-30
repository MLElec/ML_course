
# coding: utf-8

# # CS-433 Machine Learning
# ## Project 1 : The Higgs Boson Challenge
# 
# Christian Abbet, Patryk Oleniuk, Gaétan Ramet

# We first start by loading our training data and splitting it in a training set (80%) and a validation set (20%)

# In[ ]:

import matplotlib
matplotlib.use('PS')

import os
import numpy as np
import scripts.implementations as lib  # Add personal library
import scripts.proj1_helpers as helper  # Add personal library
import scripts.ml as ml # Add personal library

np.set_printoptions(precision=4)

DATA_FOLDER = 'data'
DATA_TRAIN = os.path.join(DATA_FOLDER, 'train.csv')
DATA_TEST = os.path.join(DATA_FOLDER, 'test.csv')

y, x, ids, header = helper.load_csv_data(DATA_TRAIN)
y_train, x_train,  y_validation, x_validation = lib.sep_valid_train_data(x,y, 0.8);


# # 1. Data Exploration and Cleaning 
# 
# 
# ## 1.1 Data Exploration
# 
# We first load the data to see what is the repartition of the data. The two possible classes in for the measurements are `s` for signal, indicating the presence of a Higgs boson and `b` for backgroud noise. In this case around 2/3 of the data (65.73%) is labeled as background.

# In[ ]:


print('Repartition of {} labels, s: {:.2f}%, b: {:.2f}%'.format(
    len(y_train), np.mean(y_train==1)*100, np.mean(y_train==-1)*100))


# According to [the Higgs boson machine learning challenge](https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf) some variable are indicated as "may be undefined" when it can happen that they are meaning-
# less or cannot be computed. In this case, their value is set to -999.0, which is outside the normal range of all variables. We will set them to NaN so they will be easier to handle.

# In[ ]:


x_train[x_train == -999] = np.nan
x_validation[x_validation == -999] = np.nan


# Does the NaN value gave us any information (`s` or `b`) ? We can see that for some features, the presence of Nan values seems to change the repartition of signal and background measurement : this means that we can use information from the presence of NaNs for classification

# In[ ]:


for i,feature in enumerate(x_train.T):
    print('Feature {:d} : NaN is present, s: {:.2f}, b: {:.2f}'.format(i,
        np.mean((y_train[(np.isnan(feature))] == 1)),
        np.mean((y_train[(np.isnan(feature))] == -1))))
    print('Feature {:d} : NaN is NOT present, s: {:.2f}, b: {:.2f}'.format(i,
        np.mean((y_train[~(np.isnan(feature))] == 1)),
        np.mean((y_train[~(np.isnan(feature))] == -1))))
    pass
print('NaN is present, s: {:.2f}, b: {:.2f}'.format(
     np.mean(y_train[np.any(np.isnan(x_train), axis=1)] == 1), 
     np.mean(y_train[np.any(np.isnan(x_train), axis=1)] == -1)))
print('NaN is not present, s: {:.2f}, b: {:.2f}'.format(
     np.mean(y_train[~np.any(np.isnan(x_train), axis=1)] == 1), 
     np.mean(y_train[~np.any(np.isnan(x_train), axis=1)] == -1)))


# We can also take a look at the feature ranges. it can give us insights of the data. We can see that features (16), (19), (21), (26) and (29) are actually angles (range in $[-\pi, \pi]$). To be certain we checked it directly on the documentation. We decided to also use cosines and sines of these angles as features, as they might be relevant for classification. Note that we are ignoring the NaN values to compute the min and max.
# 
# We have to be careful with those results since the output gives us no imformation about the data distribution!

# In[ ]:


print("Base Features: \n")
for i, feature in enumerate(x_train.T):
    print('Feature {} - {} has range: [{:.4f}, {:.4f}]'.format(
        i+1, header[i], np.nanmin(feature), np.nanmax(feature)))


# ## 1.4 Feature augmentation

# We will now use the information gathered in previous section to generate new features that are more helpful for classification. First, we will start working with the angles, by adding new features that are the cosines and sines of each angle.

# In[ ]:


id_angle = [15, 18, 20, 25, 28] # The ids of the features that are angles
x_aug, header = ml.augmented_feat_angle(x_train, id_angle, header)

for i, feature in enumerate(x_aug.T):
    print('Feature {} - {} has range: [{:.4f}, {:.4f}]'.format(
        i+1, header[i], np.nanmin(feature), np.nanmax(feature)))


# We will now try to generate new meaningful features by multiplying together the basic features and evaluating the difference between the median of the signal data and the background data for each new feature to determine which one are relevant:

# In[ ]:


dim = x_aug.shape[1]
m_b = np.zeros((dim, dim)) # the background medians
m_s = np.zeros((dim, dim)) # the signal medians
m_d = np.zeros((dim, dim)) # the difference medians

for i, f1 in enumerate(x_aug.T):
    for j, f2 in enumerate(x_aug.T):
        if i == j:
            f_res = f1
        else:
            f_res = f1*f2
        id_keep = ~np.isnan(f_res)
        id_b = np.logical_and(y_train == -1, id_keep)
        id_s = np.logical_and(y_train == 1, id_keep)
        f_norm = (f_res-np.nanmean(f_res))/np.nanstd(f_res) # We normalize the features
        m_b[i,j] = np.median(f_norm[id_b])
        m_s[i,j] = np.median(f_norm[id_s])
        m_d[i,j] = np.median(f_norm[id_b]) - np.median(f_norm[id_s])


# Having a look at the new features distributions allows us to see that they separate signal from background much better than the base features.

# In[ ]:


tresh_id = np.nonzero((np.abs(m_d)-np.abs(np.diag(m_d))).flatten() > 0.4)
res = np.unravel_index(tresh_id, (dim, dim))
combs = np.array(res).reshape((2,-1)).T

# Take only unique pairs
combs = np.sort(combs, axis=1)
combs = list(set([tuple(row) for row in combs]))


# Now, we will also add the features based on the nan distributions as discussed above and normalize the data. For each of the 3 types of Nan distributions we will set the value of the NAN-feature to 1 if the initial feature is nan, and to -1 if it was not a Nan

# In[ ]:


id_nan = [0, 24, 27]
x_aug, header = ml.add_nan_feature(x_aug, id_nan, header)

for i, feature in enumerate(x_aug.T):
    print('Feature {} - {} has range: [{:.4f}, {:.4f}]'.format(
        i+1, header[i], np.nanmin(feature), np.nanmax(feature)))


# Finally, we will add the combined features we designed earlier and remove the angles as we already use their cos and sin that are more meaningful

# In[ ]:


x_aug = ml.add_features(x_aug, combs) # add features that are product of other features
x_aug = ml.remove_useless(x_aug, id_useless=id_angle) #remove the angles as we used their cos and sin


# As a last step we will normalize all our features as if they were normally distributed, even though some are not and set the nan measures to 0 (mean value) so that the don't influence the prediction. We deal with outliers by thresholding them to 2.5 $\sigma$ (the standard deviation)

# In[ ]:


distrib = ['g']*(x_aug.shape[1])
x_aug, mean_train, std_train, max_train = ml.normalize_outliers(x_aug, distrib) #normalize all the features as gaussian
x_aug = np.nan_to_num(x_aug) #put all nans to 0 (mean value) so that they don't influence the predictions
print('\nStd:', np.std(x_aug, axis=0), '\nn_feat', x_aug.shape[1])


# Let's quickly apply the same treatment to our validation set :

# In[ ]:


# normalize features
x_aug_val = x_validation.copy()
x_aug_val, _ = ml.augmented_feat_angle(x_aug_val, id_angle, header)
x_aug_val, _ = ml.add_nan_feature(x_aug_val, id_nan, header)
x_aug_val = ml.add_features(x_aug_val, combs)
x_aug_val = ml.remove_useless(x_aug_val, id_useless=id_angle)
x_aug_val = ml.normalize_outliers_feed(x_aug_val, mean_train, std_train, max_train, distrib)
x_aug_val = np.nan_to_num(x_aug_val)
print('\nStd:', np.std(x_aug_val, axis=0), '\nn_feat', x_aug_val.shape[1])


# # 2. Model fitting

# ## 2.1 Least Squares (Normal equations)

# In[ ]:


weights_ls, degree_ls = lib.test_least_squares(
    x_aug, y_train, x_aug_val, y_validation, degrees = np.linspace(1,3,3))


# ## 2.2 Least Squares (Gradient Descent)

# In[ ]:


weights_ls, degree_ls = lib.test_least_squares(
    x_aug, y_train, x_aug_val, y_validation, degrees = np.linspace(3,5,3), mode='GD')


# ## 2.3 Least Squares (Stochastic Gradient Descent)

# In[ ]:


weights_ls, degree_ls = lib.test_least_squares(
    x_aug, y_train, x_aug_val, y_validation, degrees = np.linspace(2,4,3), mode='SGD')


# ## 2.4 Ridge Regression

# For Ridge Regression, we sweeped the degrees from 6 to 13 and the regularize $\lambda$ from $10^{-8}$ to $10^2$. Here, for the sake of speed, we execute a smaller sweep which gives us 83.34% validation accuracy at degree 10 and $\lambda=5.45 * 10^{-6}$

# In[ ]:


weights_ridge, degree_ridge, lambda_ridge = lib.test_ridge_regression(
    x_aug, y_train, x_aug_val, y_validation, degrees = np.linspace(9,11,3), lambdas=np.logspace(-7,-5, 10))


# ## 2.5 Logistic Regression (by Newton Method)
# For Logistic Regression with  Newton Method, we sweeped the degrees from 1 to 13. Here we just show sweep from 9 to 11 for fas execution. $\gamma$=1e-2.

# In[ ]:


y_train_log = y_train.copy();
y_train_log[y_train_log == 0] = -1;
weights_log_newton, degree_log_newton = lib.test_logistic_Newton(
    x_aug, y_train_log, x_aug_val, y_validation, degrees = np.linspace(9,11,3), gamma=1e-2)


# ## 2.5 Logistic Regression (by Gradient Descent Method)
# For Logistic Regression with  Newton Method, we sweeped the degrees from 1 to 13. Here we just show sweep from 1 to 4 for fas execution. $\gamma$=1e-8.

# In[ ]:


y_train_log = y_train.copy();
y_validation_log = y_validation.copy()
y_validation_log[y_validation_log <= 0] = 0
y_train_log[y_train_log <= 0] = 0

weights_log_gd, degree_log_gd = lib.test_logistic_GD(
    x_aug, y_train_log, x_aug_val, y_validation_log, degrees = np.linspace(2,4,3), gamma=5e-8)


# ## 2.5 Penalized Logistic Regression (by Gradient Descent Method)
# For Logistic Regression with  Newton Method, we sweeped the degrees from 1 to 13 and lambda from 1e-8 to 1e-4.     <br> Here we just show sweep from 1 to 4 and $\lambda$ = <1e-8,1e-6> for fast execution. $\gamma$=1e-2.

# In[ ]:


y_train_log = y_train.copy();
y_validation_log = y_validation.copy()
y_validation_log[y_validation_log <= 0] = 0
y_train_log[y_train_log <= 0] = 0

weights_log_pengd, degree_log_pengd, labda_log_pengd = lib.test_penalized_logistic_GD(
    x_aug, y_train_log, x_aug_val, y_validation_log, degrees = np.linspace(2,4,3), gamma=5e-8, 
    lambdas = np.logspace(-7, -8, 1))


# # 3. Model Comparison
# The maximum accuracies from different methods have been gathered for different seeds, in order to see what is the variation of the accuracies.

# In[ ]:


# The maximum accuracies have been gathered for different seeds 
seeds =     [0     , 2      , 3 ,     4,     5,     6]
ls_res =    [0.7771, 0.7742 , 0.776,  0.776 ,0.775, 0.775] # degree =1
lsgd_res =  [0.7719, 0.770  , 0.769,  0.769, 0.769, 0.766] # degree =4
lssdg_res = [0.7640, 0.7693 , 0.7632, 0.764, 0.763, 0.762] # degree =3
ridge_res = [0.8334, 0.8312 , 0.8310, 0.831, 0.832, 0.832] # degree =10
log_New  =  [0.781,  0.777  , 0.7781, 0.776, 0.779, 0.775] # degree =10
log_GD   =  [0.812,  0.813  , 0.815,  0.810, 0.809, 0.812] # degree =3
log_pen_GD= [0.817,  0.816  , 0.813,  0.818, 0.8, 0.810] # degree =3

def print_val(name,val):
    print(name,' \t: ', np.mean(val), ' +-',np.abs(np.max(val) - np.min(val)))
    
print_val('Least Squares ',ls_res)
print_val('Least Squares GD',lsgd_res)
print_val('Least Squares SGD',lssdg_res)
print_val('Ridge Regr.   ',ridge_res)
print_val('Logistic Regr. (Ntn) ',log_New)
print_val('Logistic Regr. (GD) ',log_GD)
print_val('Pen. Logistic(GD) ',log_pen_GD)


data = [ls_res, lsgd_res, lssdg_res, ridge_res, log_New, log_GD, log_pen_GD]

# # 4. Submission test

# We will here take our best model, use it to classify the test data and generate the csv file for submission

# In[ ]:


#Load data, add feature and normalize
y_test, x_test, ids_test, header = helper.load_csv_data(DATA_TEST)
x_test[x_test == -999] = np.nan

x_aug_test = x_test.copy()
x_aug_test, _ = ml.augmented_feat_angle(x_aug_test, id_angle, header)
x_aug_test, _ = ml.add_nan_feature(x_aug_test, id_nan, header)
x_aug_test = ml.add_features(x_aug_test, combs)
x_aug_test = ml.remove_useless(x_aug_test, id_useless=id_angle)
x_aug_test = ml.normalize_outliers_feed(x_aug_test, mean_train, std_train, max_train, distrib)
x_aug_test = np.nan_to_num(x_aug_test)
print('\nStd:', np.std(x_aug_test, axis=0),'\nn_feat', x_aug_test.shape[1])


# In[ ]:


#Prediction

degree_opt = degree_ridge
weights_opt = weights_ridge

_phi_test = lib.build_poly(x_aug_test, degree_opt)
y_pred = helper.predict_labels(weights_opt, _phi_test)


# In[ ]:


#Submission
helper.create_csv_submission(ids_test, y_pred, 'final_submit.csv')
print('Results saved ...')

