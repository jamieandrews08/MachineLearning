# PREP: get original data from CSV files, clean, and save out as HDF files
"""
Jamie Andrews
March 23 2019


Adapted from work by Jonathan Tay
Created on Wed Mar 15 10:39:27 2017
"""
# code source: https://github.com/JonathanTay/CS-7641-assignment-3

import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
import os 
import sklearn.model_selection as ms

#for d in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
for d in ['benchmark','RP','PCA','ICA','RF']:
    n = './output/{}/'.format(d)
    if not os.path.exists(n):
        os.makedirs(n)

OUT = './data/'

# wine data
print("processing wine dataset...")
wine = pd.read_csv ('./data/wines.csv', sep =",", header=0)
wineX = wine.drop('quality',1).copy().values
wineX = wineX.astype(float)
wineY = wine['quality'].copy().values
wineY = wineY.astype(float)

wine_trgX, wine_tstX, wine_trgY, wine_tstY = ms.train_test_split(wineX, wineY, test_size=0.3, random_state=0,stratify=wineY)     

wineX = pd.DataFrame(wine_trgX)
wineY = pd.DataFrame(wine_trgY)
wineY.columns = ['Class']

wineX2 = pd.DataFrame(wine_tstX)
wineY2 = pd.DataFrame(wine_tstY)
wineY2.columns = ['Class'] # Note renamed target variable; was 'quality'

wine1 = pd.concat([wineX,wineY],1)
wine1 = wine1.dropna(axis=1,how='all')
wine1.to_hdf(OUT+'datasets.hdf','wine',complib='blosc',complevel=9, format='table')

wine2 = pd.concat([wineX2,wineY2],1)
wine2 = wine2.dropna(axis=1,how='all')
wine2.to_hdf(OUT+'datasets.hdf','wine_test',complib='blosc',complevel=9, format='table')
print('wine dataset done.')

print('processing credit dataset...')
# get credit data and write to HDF -- only using 40% of samples
credit = pd.read_csv ('./data/credit.csv', sep =",", header=0)
credit_subset = credit.sample(frac=0.4, replace=True, random_state=42)
credit = credit_subset.copy(deep=True)
creditX = credit.drop('default',1).copy().values
creditX = creditX.astype(float)
creditY = credit['default'].copy().values

credit_trgX, credit_tstX, credit_trgY, credit_tstY = ms.train_test_split(creditX, creditY, test_size=0.3, random_state=0,stratify=creditY)     

creditX = pd.DataFrame(credit_trgX)
creditY = pd.DataFrame(credit_trgY)
creditY.columns = ['Class'] # Note renamed target variable; was 'default'

creditX2 = pd.DataFrame(credit_tstX)
creditY2 = pd.DataFrame(credit_tstY)
creditY2.columns = ['Class']

credit1 = pd.concat([creditX,creditY],1)
credit1 = credit1.dropna(axis=1,how='all')
credit1.to_hdf(OUT+'datasets.hdf','credit',complib='blosc',complevel=9, format='table')

credit2 = pd.concat([creditX2,creditY2],1)
credit2 = credit2.dropna(axis=1,how='all')
credit2.to_hdf(OUT+'datasets.hdf','credit_test',complib='blosc',complevel=9, format='table')
print('credit dataset done')
print('GO DO THE THING!')
