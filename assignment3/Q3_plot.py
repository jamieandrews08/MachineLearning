# Jamie Andrews
# March 23, 2019
# adapted from code by Chad Maron
# https://github.com/cmaron/CS-7641-assignments/tree/master/assignment3


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
#import matplotlib as plt
import matplotlib.cm as cm
from kneed import KneeLocator

from matplotlib import cycler
from matplotlib.ticker import NullFormatter, FormatStrFormatter
from os.path import basename
import seaborn as sns

# PLOT SIL SAMPLES 
redux = ['PCA', 'ICA', 'RP', 'RF']
datasets = ['credit']
pretty_name = ['Credit']
for r in redux:
    out = './output/images/{}/'.format(r)
    i=0
    while i in range (0, len(datasets)):
        data = pd.read_csv('./output/{}/clustering/wine_sil_samples.csv'.format(r, datasets[0] ), sep=',')
        clusters = data.k.unique()
        title = "test"
        #n_clusters = 5
        
        for k in clusters:
            print("k = {}".format(k))
            n_clusters = k
            plt.close()
            plt.figure(figsize=(5, 4))
            plt.style.use('seaborn')
            plt.title("Avg Silhouette Scores: Credit, {} clusters".format(k))
            plt.grid()
            plt.tight_layout()

            df = data[data['k'] == n_clusters]
            ax = plt.gca()
            sample_silhouette_values = df[df['type'] == 'Kmeans']['score'].astype(np.double)
            silhouette_avg = sample_silhouette_values.mean()

            x_min = float(min(sample_silhouette_values))
            x_max = float(max(sample_silhouette_values))
            ax.set_xlim([x_min - 0.05, x_max + 0.05])
            ax.set_ylim([0, df.shape[0]/2 + (n_clusters + 1) * 10])

            cluster_labels = df[df['type'] == 'Kmeans']['label'].astype(np.float).values
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i].values
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                ith_cluster_silhouette_values.sort()
                y_upper = y_lower + size_cluster_i
                color = cm.nipy_spectral(float(i) / n_clusters)
                ax.fill_betweenx(np.arange(y_lower, y_upper),
                                 x_min, ith_cluster_silhouette_values,
                                 facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax.text(x_min-0.02, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax.set_xlabel("Average Silhouette coefficient values")
            ax.set_ylabel("Cluster label")
            silhouette_avg = sample_silhouette_values.mean()
            ax.axvline(x=silhouette_avg, color="red", linestyle="--")
            ax.axvline(x=0, color="green", linestyle="--")
            
            ax.set_yticks([])  # Clear the yaxis labels / ticks
            ax.set_xticks(np.linspace(round(x_min, 2), round(x_max, 2), 7))
            plt.savefig(out+'Q3_credit_silh_analysis_{}.png'.format(datasets[0], k))
            plt.show()
            i=+1

# GET CLUSTER GRAPHS FOR Q3 TSNE FILES ===============
datasets = ['credit', 'wine']
redux = ['PCA', 'ICA', 'RP', 'RF']
algo=['GMM', 'Kmeans']
file_alg = ['gmm', 'kmeans']

for r in redux:
    n = './output/images/{}/'.format(r)
    if not os.path.exists(n):
        os.makedirs(n)
    for ds in datasets:
        # get the 2d data from the file
        d_file = './output/'+r+'/clustering/'+ds+'_2D.csv'
        df = pd.read_csv(d_file)
        x_axis = df['x']
        y_axis = df['y']
        for alg in algo: 
            clust =  pd.read_csv('./output/'+r+'/clustering/'+ds+'_'+alg+'_clusterlabels.csv')
            # Plotting 2d t-Sne
            col_names = list(clust)
            for col in col_names[1:]:
                plt.close()
                plt.figure()
                if r != 'benchmark':
                    plt.title(alg+" Clustering on "+ds+" data reduced using "+r+", k="+col)
                else: 
                    plt.title(alg+" Clustering on "+ds+" data without reduction, k="+col)
                plt.grid()
                plt.tight_layout()
                plt.scatter(x_axis, y_axis, c=clust[col], alpha=0.7, s=5)
                plt.xticks([])
                plt.yticks([])

                plt.axis('tight')
                if r == 'benchmark':
                    plt.savefig('./output/images/{}/Q1_{}_tsne_{}.png'.format(r, ds, col),
                                format='png', bbox_inches='tight', dpi=150)
                else: 
                    plt.savefig('./output/images/{}/Q3_{}_tsne_{}.png'.format(r, ds, col),
                                format='png', bbox_inches='tight', dpi=150)
                    
                plt.show()

# Plot accuracy for models for Q3 =======================================
redux = ['PCA', 'ICA', 'RP', 'RF']
datasets = ['credit', 'wine']
pretty_name = ['Credit', 'Wine']
for r in redux:
    out = './output/images/{}/'.format(r)
    i=0
    while i in range (0, len(datasets)):
        data = pd.read_csv('./output/{}/clustering/{}_acc.csv'.format(r, datasets[i] ), sep=',')

        plt.style.use('seaborn-whitegrid')
        plt.figure(figsize=(8, 5))

        plt.plot( 'k', 'GMM', data=data, marker='o', linewidth=2)
        plt.plot( 'k', 'Kmeans', data=data, marker='o', linewidth=2)
        #plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
        plt.title(r+" on "+pretty_name[i]+" Data, Accuracy by Number of Clusters/Components", loc='center', fontsize=16, fontweight=0, color='darkblue')
        plt.xlabel("Number of Components/Clusters", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.axis('tight')
        plt.legend(loc='best', frameon=True)
        plt.tight_layout()
        plt.savefig(out+'Q3_'+datasets[i]+'_acc_by_clusters.png')
        plt.show()
        plt.close()

# Create 2x2 plot of info from Q3 Kmeans models ===========================================
# Source = https://matplotlib.org/gallery/scales/log_demo.html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sns.reset_orig()
redux = ['PCA', 'ICA', 'RP', 'RF']
datasets = ['credit', 'wine']
pretty_name = ['Credit', 'Wine']

i=0

# Data for plotting
for r in redux:
    out = './output/images/{}/'.format(r)
    i=0
    while i in range (0, len(datasets)):

        data1 = pd.read_csv('./output/{}/clustering/{}_acc.csv'.format(r, datasets[i] ), sep=',')
        data2 = pd.read_csv('./output/{}/clustering/{}_sse.csv'.format(r, datasets[i] ), sep=',')
        data3 = pd.read_csv('./output/{}/clustering/{}_adj_mi.csv'.format(r, datasets[i] ), sep=',')
        data4 = pd.read_csv('./output/{}/clustering/{}_sil_scores.csv'.format(r, datasets[i] ), sep=',')

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        #fig.patch.set_facecolor('lightgray')
        plt.style.use('seaborn-whitegrid')

        # top left
        ax1.plot( 'k', 'GMM', data=data1[0:12], marker='o', linewidth=1.5, markersize=4, color='blue')
        ax1.set(title='Accuracy')
        #ax1.grid()

        # top right
        ax2.plot( 'k', 'SSE (left)', data=data2[0:12], marker='o', linewidth=1.5, markersize=4, color='red')
        ax2.set(title='Sum of Squared Error')
        
        #ax2.grid()

        # bottom left
        #ax3.plot( 'k', 'GMM', data=data3, marker='o', linewidth=2)
        ax3.plot( 'k', 'Kmeans', data=data3[0:12], marker='o', linewidth=1.5, markersize=4, color='green')
        ax3.set(title='Adjusted Mutual Information')
        #ax3.grid()

        # bottom right
        #ax4.plot( 'k', 'GMM', data=data4, marker='o', linewidth=2)
        ax4.plot( 'k', 'Kmeans sil_scores', data=data4[0:12], marker='o', linewidth=1.5, markersize=4, color='darkorange')
        ax4.set(title='Sihlouette Scores')
        #ax4.grid()


        #fig.tight_layout()
        if r == 'benchmark':
            fig.suptitle('Kmeans, '+pretty_name[i]+' Data', color='darkblue') # or plt.suptitle('Main title')  
        else:
            fig.suptitle(r+': '+pretty_name[i]+', Kmeans ', color='darkblue') # or plt.suptitle('Main title')
        
        #fig.suptitle('Kmeans, '+pretty_name[i]+' Data', color='darkblue') # or plt.suptitle('Main title')
        fig.tight_layout()
        
        if r == 'benchmark':
            plt.savefig(out+'Q1_'+datasets[i]+'_kmeans_2x2.png')
        else:
            plt.savefig(out+'Q3_'+datasets[i]+'_kmeans_2x2.png')
        plt.show()

        i+=1

# Plot BIC for Q3 models =======================================
redux = [ 'PCA', 'ICA', 'RP', 'RF']
datasets = ['credit', 'wine']
pretty_name = ['Credit', 'Wine']
for r in redux:
    out = './output/images/{}/'.format(r)
    i=0
    while i in range (0, len(datasets)):
        data = pd.read_csv('./output/{}/clustering/{}_bic.csv'.format(r, datasets[i] ), sep=',')

        plt.style.use('seaborn-darkgrid')
        plt.figure(figsize=(8, 5))

        plt.plot( 'k', 'BIC', data=data, marker='o', linewidth=2)
        #plt.plot( 'k', 'Kmeans', data=data, marker='o', linewidth=2)
        #plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
        plt.title("BIC by Number of Components, Original "+pretty_name[i]+" Data", loc='center', fontsize=16, fontweight=0, color='darkblue')
        plt.xlabel("Number of Components", fontsize=14)
        plt.ylabel("BIC Score", fontsize=14)
        plt.axis('tight')
        plt.legend(loc='best', frameon=True)
        plt.tight_layout()
        plt.savefig(out+'Q3_'+datasets[i]+'_bic_by_components.png')
        plt.show()
        i+=1

# Create 2x2 plot of info from Q3 GMM models ===========================================
# Source = https://matplotlib.org/gallery/scales/log_demo.html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sns.reset_orig()
redux = ['PCA', 'ICA', 'RP', 'RF']
datasets = ['credit', 'wine']
pretty_name = ['Credit', 'Wine']

i=0

# Data for plotting
for r in redux:
    out = './output/images/{}/'.format(r)
    i=0
    while i in range (0, len(datasets)):

        data1 = pd.read_csv('./output/{}/clustering/{}_acc.csv'.format(r, datasets[i] ), sep=',')
        data2 = pd.read_csv('./output/{}/clustering/{}_bic.csv'.format(r, datasets[i] ), sep=',')
        data3 = pd.read_csv('./output/{}/clustering/{}_adj_mi.csv'.format(r, datasets[i] ), sep=',')
        data4 = pd.read_csv('./output/{}/clustering/{}_logliklihood.csv'.format(r, datasets[i] ), sep=',')

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        #fig.patch.set_facecolor('lightgray')
        plt.style.use('seaborn-whitegrid')

        # top left
        ax1.plot( 'k', 'GMM', data=data1[0:12], marker='o', linewidth=1.5, markersize=4, color='blue')
        #ax1.plot( 'k', 'Kmeans', data=data1, marker='o', linewidth=2)
        ax1.set(title='Accuracy')
        #ax1.grid()

        # top right
        ax2.plot( 'k', 'BIC', data=data2[0:12], marker='o', linewidth=1.5, markersize=4, color='red')
        #ax2.plot( 'k', 'Kmeans', data=data1, marker='o', linewidth=2)
        ax2.set(title='BIC')
        
        #ax2.grid()

        # bottom left
        #ax3.plot( 'k', 'GMM', data=data3, marker='o', linewidth=2)
        ax3.plot( 'k', 'GMM', data=data3[0:12], marker='o', linewidth=1.5, markersize=4, color='green')
        ax3.set(title='Adjusted Mutual Information')
        #ax3.grid()

        # bottom right
        #ax4.plot( 'k', 'GMM', data=data4, marker='o', linewidth=2)
        ax4.plot( 'k', 'Log Likelihood', data=data4[0:12], marker='o', linewidth=1.5, markersize=4, color='darkorange')
        ax4.set(title='Average Log Likelihood')
        #ax4.grid()


        #fig.tight_layout()
        if r == 'benchmark':
            fig.suptitle('GMM, '+pretty_name[i]+' Data', color='darkblue') # or plt.suptitle('Main title')  
        else:
            fig.suptitle(r+': '+pretty_name[i]+', GMM Fit', color='darkblue') # or plt.suptitle('Main title')
        
        #fig.suptitle('Kmeans, '+pretty_name[i]+' Data', color='darkblue') # or plt.suptitle('Main title')
        fig.tight_layout()
        
        if r == 'benchmark':
            plt.savefig(out+'Q1_'+datasets[i]+'_gmm_2x2.png')
        else:
            plt.savefig(out+'Q3_'+datasets[i]+'_gmm_2x2.png')
        plt.show()

        i+=1

# Create 4x2 plot of SSE and BIC from Q3 clustering models ===========================================
# Source = https://matplotlib.org/gallery/scales/log_demo.html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#sns.reset_orig()

datasets = ['credit', 'wine']
pretty_name = ['Credit', 'Wine']

i=0

# Data for plotting

out = './output/images/'
i=0
while i in range (0, len(datasets)):

    data1 = pd.read_csv('./output/{}/clustering/{}_sse.csv'.format('PCA', datasets[i] ), sep=',')
    data2 = pd.read_csv('./output/{}/clustering/{}_bic.csv'.format('PCA', datasets[i] ), sep=',')
    data3 = pd.read_csv('./output/{}/clustering/{}_sse.csv'.format('ICA', datasets[i] ), sep=',')
    data4 = pd.read_csv('./output/{}/clustering/{}_bic.csv'.format('ICA', datasets[i] ), sep=',')
    data5 = pd.read_csv('./output/{}/clustering/{}_sse.csv'.format('RP', datasets[i] ), sep=',')
    data6 = pd.read_csv('./output/{}/clustering/{}_bic.csv'.format('RP', datasets[i] ), sep=',')
    data7 = pd.read_csv('./output/{}/clustering/{}_sse.csv'.format('RF', datasets[i] ), sep=',')
    data8 = pd.read_csv('./output/{}/clustering/{}_bic.csv'.format('RF', datasets[i] ), sep=',')

    # Create figure
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2)
    #fig.patch.set_facecolor('lightgray')
    plt.style.use('seaborn-darkgrid')

    # top left
    ax1.plot( 'k', 'SSE (left)', data=data1, marker='o', linewidth=1.5, markersize=4, color='blue')
    #ax1.plot( 'k', 'Kmeans', data=data1, marker='o', linewidth=2)
    ax1.set(title='SSE')
    ax1.set_yticklabels([])
    #ax1.set_xticklabels([])
    ax1.set_ylabel('PCA')
    #ax1.grid()

    # top right
    ax2.plot( 'k', 'BIC', data=data2, marker='o', linewidth=1.5, markersize=4, color='blue')
    #ax2.plot( 'k', 'Kmeans', data=data1, marker='o', linewidth=2)
    ax2.set_yticklabels([])
    ax2.set(title='BIC')

    #ax2.grid()

    # 2nd row left
    #ax3.plot( 'k', 'GMM', data=data3, marker='o', linewidth=2)
    ax3.plot( 'k', 'SSE (left)', data=data3, marker='o', linewidth=1.5, markersize=4, color='red')
    ax3.set_yticklabels([])
    ax3.set_ylabel('ICA')
    #ax3.set(title='Adjusted Mutual Information')
    #ax3.grid()

    # 2nd row right
    #ax4.plot( 'k', 'GMM', data=data4, marker='o', linewidth=2)
    ax4.plot( 'k', 'BIC', data=data4, marker='o', linewidth=1.5, markersize=4, color='red')
    ax4.set_yticklabels([])
    #ax4.set(title='Average Log Likelihood')
    #ax4.grid()

    # 3rd row left 
    #ax4.plot( 'k', 'GMM', data=data4, marker='o', linewidth=2)
    ax5.plot( 'k', 'SSE (left)', data=data5, marker='o', linewidth=1.5, markersize=4, color='green')
    ax5.set_yticklabels([])
    ax5.set_ylabel('RF')
    #ax5.set(title='Average Log Likelihood')
    #ax4.grid()

    # 3rd row right 
    #ax4.plot( 'k', 'GMM', data=data4, marker='o', linewidth=2)
    ax6.plot( 'k', 'BIC', data=data6, marker='o', linewidth=1.5, markersize=4, color='green')
    ax6.set_yticklabels([])
    #ax6.set(title='Average Log Likelihood')
    #ax4.grid()

    # bottom left
    #ax4.plot( 'k', 'GMM', data=data4, marker='o', linewidth=2)
    ax7.plot( 'k', 'SSE (left)', data=data7, marker='o', linewidth=1.5, markersize=4, color='purple')
    ax7.set_yticklabels([])
    ax7.set_ylabel('RP')
    #ax7.set(title='Average Log Likelihood')
    #ax4.grid()

    # bottom right
    #ax4.plot( 'k', 'GMM', data=data4, marker='o', linewidth=2)
    ax8.plot( 'k', 'BIC', data=data8, marker='o', linewidth=1.5, markersize=4, color='purple')
    ax8.set_yticklabels([])
    #ax8.set(title='Average Log Likelihood')
    #ax4.grid()


    
    fig.suptitle('Reduced '+pretty_name[i]+' Data', color='darkblue', fontsize=14) # or plt.suptitle('Main title')

    #fig.suptitle('Kmeans, '+pretty_name[i]+' Data', color='darkblue') # or plt.suptitle('Main title')
    fig.tight_layout()


    plt.savefig(out+'Q3_'+datasets[i]+'_sse_bic_4x2.png')
    plt.show()

    i+=1



        i+=1