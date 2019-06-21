# Jamie Andrews
# March 23 2019
# Adapted from code by Chad Maron & Jonathan Tay
# https://github.com/cmaron/CS-7641-assignments/tree/master/assignment3
# https://github.com/JonathanTay/CS-7641-assignment-3

import itertools
import logging
import os
import glob
import re
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.axes as maxes
import matplotlib.pyplot as plt

import matplotlib.cm as cm
from kneed import KneeLocator

from matplotlib import cycler
from matplotlib.ticker import NullFormatter, FormatStrFormatter
from os.path import basename


def plot_scree(title, df, problem_name, multiple_runs=False, xlabel='Number of Clusters', ylabel=None):
    if ylabel is None:
        ylabel = 'Kurtosis'
        if problem_name == 'PCA' or problem_name == 'SVD':
            ylabel = 'Variance'
        elif problem_name == 'RP':
            # ylabel = 'PDCC'  # 'Pairwise distance corrcoef'
            ylabel = 'Pairwise Distance CorrCoef'
        elif problem_name == 'RF':
            ylabel = 'Feature Importances'
    title = title.format(ylabel)

    plt.close()
    plt.figure(figsize=(5, 4))
    # style
    plt.style.use('seaborn-paper')
    # create a color palette
    #palette = plt.get_cmap('Set1')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()

    ax = plt.gca()
    
    x_points = df.index.values
    #x_points = df[0].values
    y_points = df[1]
    if multiple_runs:
        y_points = np.mean(df.iloc[:, 1:-1], axis=1)
        y_std = np.std(df.iloc[:, 1:-1], axis=1)
        plt.plot(x_points, y_points, 'o-', linewidth=1.5, markersize=4,
                 label=ylabel)
        plt.fill_between(x_points, y_points - y_std,
                         y_points + y_std, alpha=0.2)
    else:
        plt.plot(x_points, y_points, 'o-', linewidth=1.5, markersize=4,
                 label=ylabel)

    min_value = np.min(y_points)
    min_point = y_points.idxmin()
    max_value = np.max(y_points)
    max_point = y_points.idxmax()
    knee_point = find_knee(y_points)
    kl = KneeLocator(x_points, y_points)

    ax.axvline(x=min_point, linestyle="--", color='red', label="Min: {}".format(int(min_point)))
    ax.axvline(x=max_point, linestyle="--", color='orange', label="Max: {}".format(int(max_point)))
    if kl.knee_x is not None:
        ax.axvline(x=kl.knee_x, linestyle="--", color='green', label="Knee: {}".format(kl.knee_x))
    else:
        ax.axvline(x=knee_point, linestyle="--", color='green', label="Knee: {}".format(knee_point))

    ax.set_xticks(df.index.values, minor=False)
    
    plt.legend(loc="best", frameon=True)

    return plt


def find_knee(values):
    # get coordinates of all the points
    nPoints = len(values)
    allCoord = np.vstack((range(nPoints), values)).T
    # np.array([range(nPoints), values])

    # get the first point
    firstPoint = allCoord[0]
    # get vector between first and last point - this is the line
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec ** 2))

    # find the distance from each point to the line:
    # vector between all points and first point
    vecFromFirst = allCoord - firstPoint

    # To calculate the distance to the line, we split vecFromFirst into two
    # components, one that is parallel to the line and one that is perpendicular
    # Then, we take the norm of the part that is perpendicular to the line and
    # get the distance.
    # We find the vector parallel to the line by projecting vecFromFirst onto
    # the line. The perpendicular vector is vecFromFirst - vecFromFirstParallel
    # We project vecFromFirst by taking the scalar product of the vector with
    # the unit vector that points in the direction of the line (this gives us
    # the length of the projection of vecFromFirst onto the line). If we
    # multiply the scalar product by the unit vector, we have vecFromFirstParallel
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel

    # distance to line is the norm of vecToLine
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))

    # knee/elbow is the point with max distance value
    idxOfBestPoint = np.argmax(distToLine)
    return idxOfBestPoint

# Plot the dimesion reduction exploration graphs for ICA, PCA, and RP

datasets = ['credit', 'wine']
redux= ['PCA', 'ICA', 'RF', 'RP']

for ds in datasets:
    for r in redux:
        # make sure directory exists
        n = './output/images/{}/'.format(r)
        if not os.path.exists(n):
            os.makedirs(n)

        if r == 'RF':
            #Random Forest Reduction - plot feature importance
            df = pd.read_csv('./output/'+r+'/'+ds+'_scree2.csv', header=None, index_col=0)
            df = df.rename(columns={ df.columns[0]: "Importance" })
            df['Importance'].plot.bar()
            
            plt.title("RF: Importance by feature, "+ds+" data", loc='center', fontsize=10, fontweight=0) #, 
                      #color='darkblue')
            plt.axhline(y=0.08, linestyle="--", color='red') #, label="Min: {}".format(int(min_point)))
            plt.axhline(y=0.09, linestyle="--", color='green') #, label="Min: {}".format(int(min_point)))
            plt.axhline(y=0.10, linestyle="--", color='yellow') #, label="Min: {}".format(int(min_point)))
            plt.xlabel("Feature Number")
            plt.ylabel('Portion of Total Importance (sum=1.0)')
            #plt.legend(loc='best')
            plt.xticks(rotation=0)
            plt.savefig('./output/images/'+r+'/q2_'+ds+'_important_feature.png')
            plt.show()
            plt.close()
            
            # RF plot how many features are important
            df = pd.read_csv('./output/'+r+'/'+ds+'_scree.csv', header=None, index_col=0)
            plot_scree(title=r+' on '+ds+' data', df=df, 
                   problem_name=r, multiple_runs=False, xlabel='Number of Dimensions', 
                       ylabel='Portion of Total Importance (sum=1.0)')
            plt.savefig('./output/images/'+r+'/q2_'+ds+'_scree.png')
            plt.show()
        
        elif r == 'RP':
            df = pd.read_csv('./output/'+r+'/'+ds+'_scree1.csv', header=None, index_col=0)
            plot_scree(title=r+' on '+ds+' data', df=df, 
                   problem_name=r, multiple_runs=True, xlabel='Number of Dimensions', ylabel=None)
            plt.savefig('./output/images/'+r+'/q2_'+ds+'_scree1.png')
            plt.show()
            
            plt.close()
            df = pd.read_csv('./output/'+r+'/'+ds+'_scree2.csv', header=None, index_col=0)
            plot_scree(title=r+' on '+ds+' data', df=df, 
                   problem_name=r, multiple_runs=True, xlabel='Number of Dimensions', ylabel="Reconstruction Error")
            plt.savefig('./output/images/'+r+'/q2_'+ds+'_scree2.png')
            plt.show()
            
            
        else:
            df = pd.read_csv('./output/'+r+'/'+ds+'_scree.csv', header=None, index_col=0)
            plot_scree(title=r+' on '+ds+' data', df=df, 
                   problem_name=r, multiple_runs=False, xlabel='Number of Dimensions', ylabel=None)
            plt.savefig('./output/images/'+r+'/q2_'+ds+'_scree.png')
            plt.show()
    

