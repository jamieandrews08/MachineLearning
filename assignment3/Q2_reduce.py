# QUESTION 2 - Apply Dimension Reduction Algorithms
# Jamie Andrews 
# March 23 2019
# Adapted from code written by Jonathan Tay & Chad Maron
# source : https://github.com/JonathanTay/CS-7641-assignment-3
# source: https://github.com/cmaron/CS-7641-assignments/tree/master/assignment3

#%% Imports
import pandas as pd
import numpy as np
from helpers import   pairwiseDistCorr,nn_reg,nn_arch,reconstructionError
from itertools import product
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#from helpers import  nn_arch,nn_reg
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

import os 
import sklearn.model_selection as ms

# for d in ['RP','PCA','ICA','RF']:
#     n = '.{}/'.format(d)
#     if not os.path.exists(n):
#         os.makedirs(n)

out = './output/'
cmap = cm.get_cmap('Spectral') 

np.random.seed(42)

# get orginal data (not reduced yet)
credit = pd.read_hdf('./data/datasets.hdf','credit')
creditX = credit.drop('Class',1).copy().values
creditY = credit['Class'].copy().values

wine = pd.read_hdf('./data/datasets.hdf','wine')        
wineX = wine.drop('Class',1).copy().values
wineY = wine['Class'].copy().values


wineX = StandardScaler().fit_transform(wineX)
creditX= StandardScaler().fit_transform(creditX)

# EXPLORE DIFFERENT DIMENSIONS WITH THE MODELS ===================

wine_dims = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
credit_dims = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


# PCA =============================================================

pca = PCA(random_state=42)
pca.fit(wineX)
tmp = pd.Series(data = pca.explained_variance_ratio_,index = range(1,13))
tmp.to_csv(out+'PCA/wine_scree.csv', header=False)


pca = PCA(random_state=42)
pca.fit(creditX)
tmp = pd.Series(data = pca.explained_variance_ratio_,index = range(1,15))
tmp.to_csv(out+'PCA/credit_scree.csv', header=False)


# ICA ==========================

ica = FastICA(random_state=42)
kurt = {}
for dim in wine_dims:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(wineX)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt) 
kurt.to_csv(out+'ICA/wine_scree.csv',header=False)


ica = FastICA(random_state=42)
kurt = {}
for dim in credit_dims:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(creditX)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt) 
kurt.to_csv(out+'ICA/credit_scree.csv',header=False)
#raise




# Randomized projections ========================

tmp = defaultdict(dict)
for i,dim in product(range(10),wine_dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(wineX), wineX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'RP/wine_scree1.csv')


tmp = defaultdict(dict)
for i,dim in product(range(10),credit_dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(creditX), creditX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'RP/credit_scree1.csv')


tmp = defaultdict(dict)
for i,dim in product(range(10),wine_dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(wineX)    
    tmp[dim][i] = reconstructionError(rp, wineX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'RP/wine_scree2.csv')


tmp = defaultdict(dict)
for i,dim in product(range(10),credit_dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(creditX)  
    tmp[dim][i] = reconstructionError(rp, creditX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'RP/credit_scree2.csv')




# RF ====================================

rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=42,n_jobs=1)
fs_wine = rfc.fit(wineX,wineY).feature_importances_ 
fs_credit = rfc.fit(creditX,creditY).feature_importances_ 

# returns fearure importance - the higher the number the more important the feature
tmp = pd.Series(np.sort(fs_wine)[::-1])
tmp.to_csv(out+'RF/wine_scree.csv', header=False, index = range(1,13))

# save unsorted version to see exactly which features are important
tmp = pd.Series(fs_wine)
tmp.to_csv(out+'RF/wine_scree2.csv', header=False, index = range(1,13))

#tmp = pd.Series(fs_credit)
tmp = pd.Series(np.sort(fs_credit)[::-1]) 
tmp.to_csv(out+'RF/credit_scree.csv', header=False, index = range(1,15))

# save unsorted version to see exactly which features are important
tmp = pd.Series(fs_credit)
tmp.to_csv(out+'RF/credit_scree2.csv', header=False, index = range(1,15))




# NOW REDUCE THE DATA WITH DETERMINED BEST NUMBER OF DIMENSIONS

# PCA reductions =======
out='./output/PCA/'

# wine quality data
dim = 4
pca = PCA(n_components=dim,random_state=10)

wineX2 = pca.fit_transform(wineX)
wine2 = pd.DataFrame(np.hstack((wineX2,np.atleast_2d(wineY).T)))
cols = list(range(wine2.shape[1]))
cols[-1] = 'Class'
wine2.columns = cols
wine2.to_hdf(out+'datasets.hdf','wine',complib='blosc',complevel=9, format='table')

# credit default data
dim = 4
pca = PCA(n_components=dim,random_state=10)
creditX2 = pca.fit_transform(creditX)
credit2 = pd.DataFrame(np.hstack((creditX2,np.atleast_2d(creditY).T)))
cols = list(range(credit2.shape[1]))
cols[-1] = 'Class'
credit2.columns = cols
credit2.to_hdf(out+'datasets.hdf','credit',complib='blosc',complevel=9, format='table')

# ICA Reductions ===========
out='./output/ICA/'
dim = 5
ica = FastICA(n_components=dim,random_state=10)

wineX2 = ica.fit_transform(wineX)
wine2 = pd.DataFrame(np.hstack((wineX2,np.atleast_2d(wineY).T)))
cols = list(range(wine2.shape[1]))
cols[-1] = 'Class'
wine2.columns = cols
wine2.to_hdf(out+'datasets.hdf','wine',complib='blosc',complevel=9, format='table')

dim = 7
ica = FastICA(n_components=dim,random_state=10)
creditX2 = ica.fit_transform(creditX)
credit2 = pd.DataFrame(np.hstack((creditX2,np.atleast_2d(creditY).T)))
cols = list(range(credit2.shape[1]))
cols[-1] = 'Class'
credit2.columns = cols
credit2.to_hdf(out+'datasets.hdf','credit',complib='blosc',complevel=9, format='table')


# RP Reductions ============
out='./output/RP/'
dim = 3
rp = SparseRandomProjection(n_components=dim,random_state=5)

wineX2 = rp.fit_transform(wineX)
wine2 = pd.DataFrame(np.hstack((wineX2,np.atleast_2d(wineY).T)))
cols = list(range(wine2.shape[1]))
cols[-1] = 'Class'
wine2.columns = cols
wine2.to_hdf(out+'datasets.hdf','wine',complib='blosc',complevel=9, format='table')

dim = 5
rp = SparseRandomProjection(n_components=dim,random_state=5)
creditX2 = rp.fit_transform(creditX)
credit2 = pd.DataFrame(np.hstack((creditX2,np.atleast_2d(creditY).T)))
cols = list(range(credit2.shape[1]))
cols[-1] = 'Class'
credit2.columns = cols
credit2.to_hdf(out+'datasets.hdf','credit',complib='blosc',complevel=9, format='table')


# RF Reductions ============
out='./output/RF/'
rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=42,n_jobs=7)


#wine quality data
dim = 4
filtr = ImportanceSelect(rfc,dim)
    
wineX2 = filtr.fit_transform(wineX,wineY)
wine2 = pd.DataFrame(np.hstack((wineX2,np.atleast_2d(wineY).T)))
cols = list(range(wine2.shape[1]))
cols[-1] = 'Class'
wine2.columns = cols
wine2.to_hdf(out+'datasets.hdf','wine',complib='blosc',complevel=9, format='table')

# credit default data
dim = 3
filtr = ImportanceSelect(rfc,dim)
creditX2 = filtr.fit_transform(creditX,creditY)
credit2 = pd.DataFrame(np.hstack((creditX2,np.atleast_2d(creditY).T)))
cols = list(range(credit2.shape[1]))
cols[-1] = 'Class'
credit2.columns = cols
credit2.to_hdf(out+'datasets.hdf','credit',complib='blosc',complevel=9, format='table')


