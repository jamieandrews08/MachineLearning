# %load ja_clustering.py
# %load clustering.py
"""
Created on Thu Mar 16 10:38:28 2017

@author: jtay

source: https://github.com/JonathanTay/CS-7641-assignment-3
"""

#%% Imports
import os
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM 
from collections import defaultdict
from helpers import cluster_acc, myGMM,nn_arch,nn_reg
from sklearn.metrics import adjusted_mutual_info_score as ami, silhouette_score as sil_score, \
    silhouette_samples as sil_samples
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import sys

seed = 42
np.random.seed(42)
redux = ['benchmark']
datasets = ['credit', 'wine']

for r in redux:
    out = './output/'+r+'/clustering/'   #.format(sys.argv[1])
    if not os.path.exists(out):
        os.makedirs(out)
    for ds in datasets:
        # read from the initial data set that has not been reduced
        data = pd.read_hdf('./data/datasets.hdf',ds)
        dataX = data.drop('Class',1).copy().values
        dataY = data['Class'].copy().values
        dataX= StandardScaler().fit_transform(dataX)

        # try different cluster sizes and get the results
        clusters =  [2,3,4,5,6,7,8,9,10,12,15,20,25,30,35]

        sse = defaultdict(dict)
        bic = defaultdict(dict)
        ll = defaultdict(dict)
        acc = defaultdict(lambda: defaultdict(dict))
        adj_mi = defaultdict(lambda: defaultdict(dict))
        km = kmeans(random_state=seed)
        gmm = GMM(random_state=seed)
        # added to get sihlouette scores
        sil = defaultdict(lambda: defaultdict(dict))
        sil_samp = np.empty(shape=(2*len(clusters)*dataX.shape[0],4), dtype='<U21')
        labels = defaultdict(lambda: defaultdict(dict))

        j=0
        for k in clusters:
            st = clock()
            km.set_params(n_clusters=k)
            gmm.set_params(n_components=k)

            print('Now processing {} data with {} using {} clusters...'.format(ds, r, k))
            data_st = clock()

            # fit the credit data
            km.fit(dataX)
            km_labels = km.predict(dataX)  

            gmm.fit(dataX)
            gmm_labels = gmm.predict(dataX)
            #gmm_inertia = gmm.inertia_

            # save the labels
            labels[k]['Kmeans'] = km_labels
            labels[k]['GMM'] = gmm_labels
            inertia[k]['Kmeans'] = km_labels
            inertia[k]['GMM'] = gmm_labels
            

            sil[k]['Kmeans'] = sil_score(dataX, km_labels)
            sil[k]['GMM'] = sil_score(dataX, gmm_labels)
            km_sil_samples = sil_samples(dataX, km_labels)
            gmm_sil_samples = sil_samples(dataX, gmm_labels)
            for i, x in enumerate(km_sil_samples):
                sil_samp[j] = [k, 'Kmeans', round(x, 6), km_labels[i]]
                j += 1
            for i, x in enumerate(gmm_sil_samples):
                sil_samp[j] = [k, 'GMM', round(x, 6), gmm_labels[i]]
                j += 1
            sse[k] = km.score(dataX)
            ll[k] = gmm.score(dataX)
            bic[k] = gmm.bic(dataX)
            acc[k]['Kmeans'] = cluster_acc(dataY,km.predict(dataX))
            acc[k]['GMM'] = cluster_acc(dataY,gmm.predict(dataX))
            adj_mi[k]['Kmeans'] = ami(dataY,km.predict(dataX))
            adj_mi[k]['GMM'] = ami(dataY,gmm.predict(dataX))



        gmm_clusters = pd.DataFrame()
        kmeans_clusters = pd.DataFrame()

        for i in clusters:
            gmm_clusters[i] = labels[i]['GMM']
            kmeans_clusters[i] = labels[i]['Kmeans']

        bic = pd.DataFrame(bic, index=[0]).T
        bic.index.name = 'k'
        bic.rename(columns= {bic.columns[0]: 'BIC'}, inplace=True)
        

        sse = (-pd.DataFrame(sse, index=[0]).T)
        sse.index.name = 'k'
        sse.rename(columns={sse.columns[0]:'SSE (left)'}, inplace=True)    

        ll = pd.DataFrame(ll, index=[0]).T
        ll.index.name = 'k'
        ll.rename(columns={ll.columns[0]:'Log Likelihood'}, inplace=True) 

        
        acc = pd.DataFrame(acc).T
        adj_mi = pd.DataFrame(adj_mi).T

        sil = pd.DataFrame(sil).T
        sil.rename(columns = lambda x: x+' sil_scores',inplace=True)

        sil_samp = pd.DataFrame(sil_samp, columns=['k', 'type', 'score', 'label']).set_index('k')  #.T
        
        sil.index.name = 'k'
        sil_samp.index.name = 'k'
        acc.index.name = 'k'
        adj_mi.index.name = 'k'
        

        sse.to_csv(out+ds+'_sse.csv')
        bic.to_csv(out+ds+'_bic.csv')
        ll.to_csv(out+ds+'_logliklihood.csv')
        acc.to_csv(out+ds+'_acc.csv')
        adj_mi.to_csv(out+ds+'_adj_mi.csv')
        sil.to_csv(out+ds+'_sil_scores.csv')
        sil_samp.to_csv(out+ds+'_sil_samples.csv')
        gmm_clusters.to_csv(out+ds+'_GMM_clusterlabels.csv')
        kmeans_clusters.to_csv(out+ds+'_Kmeans_clusterlabels.csv')
        
        print("Processing t_SNE for {} and {} data...".format(r, ds))
        # t-distributed stochastic neighbor embedding for visualization of clusters
        dataX2D = TSNE(verbose=10,random_state=seed).fit_transform(dataX)
        data2D = pd.DataFrame(np.hstack((dataX2D,np.atleast_2d(dataY).T)),columns=['x','y','target'])
        data2D.to_csv(out+ds+'_2D.csv')

print("All done!")


