# Jamie Andrews
# March 23 2019
# Adapted from code by Chad Maron & Jonathan Tay
# https://github.com/cmaron/CS-7641-assignments/tree/master/assignment3
# https://github.com/JonathanTay/CS-7641-assignment-3


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import nn_arch, nn_reg
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA



np.random.seed(42)
alphas = [10**-x for x in range(1,4)]
d = 4
nn_arch = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]


redux = ['PCA', 'ICA', 'RP', 'RF']

for r in redux:
    out = './output/{}/'.format(r)

    #datasets = ['credit']
    
    credit = pd.read_hdf(out+'datasets.hdf','credit')
    creditX = credit.drop('Class',1).copy().values
    creditY = credit['Class'].copy().values
    
    creditX= StandardScaler().fit_transform(creditX)
    grid ={'NN__alpha':alphas,'NN__hidden_layer_sizes':nn_arch}   
    mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=seed)
    pipe = Pipeline([('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

    gs.fit(creditX,creditY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'Q4_credit_dim_red_ANN.csv')


# LETS MAKE SOME UGLY PLOTS FOR INFORMATION PURPOSES
# Accuracy by number of clusters for Kmeans and GMM models
plt.close()
data['k'] = data['k'].astype('category')
data.plot.bar() #(x='k', y='GMM', data=data)
plt.ylim(ymax = 1.0, ymin =0)
plt.title(model+' Accuracy by '+img_name+' - '+ds+" data", loc='center', fontsize=12, fontweight=0, color='darkblue')
plt.xlabel("Number of Clusters (k)")
plt.ylabel('Accuracy Score')
plt.legend(loc='best', ncol=2)
plt.xticks(rotation=0)
plt.savefig(output+'Q3_'+ds+'_acc_by_'+img_name+'.png')
plt.show()


# ANN - Get Accuracy for different values of K, metrics, and weights ===================
models = ['PCA', 'ICA', 'RP', 'RF']
datasets = ['credit']
ds_name = ['Credit Default']
#param = 'param_KNN__n_neighbors'
params = ['param_NN__alpha', 'param_NN__hidden_layer_sizes']#param_NN__hidden_layer_sizes
img_name = ['alphas', 'layers']
axis = ["Alpha Values", 'Hidden Layer Sizes']

for model in models:
    for ds in datasets:
        for i in range(0, len(params)):
            param = params[i]
            # get the data
            #file = './output/'+model+'/'+ds+'_dim_red.csv' 
            file = './output/PCA/Q4_credit_dim_red_ANN.csv'
            reg = pd.read_csv (file, sep =",")
            # change param to group by to categorical variable
            reg[param] = reg[param].astype('category')
            # get mean test and train scores grouping by hidden layer
            reg_byHL = reg.groupby([param])['mean_train_score','mean_test_score'].mean()
            # sort values before plotting
            reg_byHL = reg_byHL.sort_values('mean_test_score')
            # plot it
            reg_byHL.plot.bar()
            plt.style.use('seaborn-darkgrid')
            plt.ylim(ymax = 1.0, ymin =0)
            plt.title(model+': ANN Accuracy by '+axis[i]+' - Credit Default data', loc='center', fontsize=12, fontweight=0, color='darkblue')
            plt.xlabel(axis[i])
            plt.ylabel('Accuracy Score')
            plt.legend(loc='best', ncol=2)
            plt.xticks(rotation=30)
            plt.legend(loc='best', frameon=True)
                        #bbox_to_anchor=(0.5, -0.05),
            #           fancybox=False, shadow=True, ncol=5)
            # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            #            ncol=2, mode="expand", borderaxespad=0.)

            plt.savefig('./output/images/'+model+'/Q4_'+ds+'_acc_by_'+img_name[i]+'.png')
            plt.show()
            #plt.close()
        i+=1



    
