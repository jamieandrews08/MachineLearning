# Jamie Andrews
# March 23 2019
# Adapted from code by Chad Maron & Jonathan Tay
# https://github.com/cmaron/CS-7641-assignments/tree/master/assignment3
# https://github.com/JonathanTay/CS-7641-assignment-3

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

redux = ['PCA', 'ICA', 'RP', 'RF']
datasets = ['credit']
algo = ['Kmeans', 'GMM']
clusters =  [2,3,4,5,6,7,8,9,10,12,15,20,25]#,30,35]
seed=42

d = 4 #data.shape[1]
nn_reg = [10**-x for x in range(1,4)]
nn_arch = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]#,d*3]]
#clusters =  [2,3,4,5,6,7,8,9,10,12,15,20,25]#,30,35]

#results = pd.DataFrame()

for r in redux: 
    out = './output/{}/'.format(r)
    results = pd.DataFrame()
    for ds in datasets:
        results = pd.DataFrame()
        # get the reduced data (output from Q2)
        data = pd.read_hdf(out+'datasets.hdf',ds)
        
        for a in algo:
            
                # get the labels for data from previous clustering (output from Q3)
                labels = pd.read_csv(out+'/clustering/{}_{}_clusterlabels.csv'.format(ds, a), sep=',')
            
                for k in clusters:
                    # for each k, add label for k clusters to the data
                    column = str(k)
                    data['cluster_label'] = labels[column]

                    # split and standardize
                    dataX = data.drop('Class',1).copy().values
                    dataY = data['Class'].copy().values
                    print("Processing {} data reduced with {} and {} using {} clusters...".format(ds, r, a, k))


                    print("Starting {} GridSearch...".format(a))

                    # Run ANN on the reduced data with cluster labels
                    grid ={'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
                    mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=seed)
                    pipe = Pipeline([('NN',mlp)])
                    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

                    gs.fit(dataX,dataY)

                    # get the output
                    #tmp = pd.DataFrame(gs.cv_results_)
                    gs.cv_results_['k'] = np.full((len(gs.cv_results_['mean_fit_time'])), fill_value = k, dtype=int)                
                    tmp = pd.DataFrame(gs.cv_results_)
                    tmp['clust_alg'] = a
                    tmp['datset']= ds 
                    tmp['redux'] = r
                    
                    results = pd.concat([results, tmp],ignore_index=True)
                    #raise
                
        results.to_csv(out+'Q5_{}_{}_cluster_NN.csv'.format(r, ds))

# run some plots for the sake of gathering info
# ANN - Get Run times for different alphas and layer sizes ===================
models = ['PCA', 'ICA', 'RF', 'RP']
datasets = ['credit']
#param = 'param_KNN__n_neighbors'
params = ['param_NN__alpha', 'param_NN__hidden_layer_sizes', 'param_km__n_clusters']
img_name = ['alphas', 'layers', 'clusters']
axis = ["Alpha Values", 'Hidden Layer Sizes', "Number of Clusters"]
cl_algo = ['Kmeans', 'GMM']

for model in models:
    for cl in cl_algo: 
        for ds in datasets:
            for i in range(0, len(params)):
                param = params[i]
                # get the data
                file = './output/'+model+'/Q5_credit_'+cl+'_ANN.csv' 
                #file = './output/PCA/credit_dim_red.csv'
                reg = pd.read_csv (file, sep =",")
                # change param to group by to categorical variable
                reg[param] = reg[param].astype('category')
                # get mean test and train scores grouping by hidden layer
                reg_byHL = reg.groupby([param])['mean_fit_time','mean_score_time'].mean()
                # sort values before plotting
                reg_byHL = reg_byHL.sort_values('mean_fit_time')
                # plot it
                reg_byHL.plot.bar()
                #plt.ylim(ymax = 1.0, ymin =0)
                plt.title(model+': Process Time by '+axis[i]+' - Credit Default Data', loc='center', fontsize=12, fontweight=0, color='darkblue')
                plt.xlabel(axis[i])
                plt.ylabel('Fit Time')
                plt.legend(loc='best', ncol=2)
                plt.xticks(rotation=45)
                # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                #           fancybox=False, shadow=True, ncol=5)
                # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                #            ncol=2, mode="expand", borderaxespad=0.)

                plt.savefig('./output/images/'+model+'/Q5_'+cl+'_ANN_time_by_'+img_name[i]+'.png')
                plt.show()
                #plt.close()
                i+=1




