# MachineLearning
Machine Learning projects performed for CS7641 at Georgia Tech in Spring 2019

_Please note that analysis reports that details the results and findings of these projects is not included in this repository. Analysis reports have been omitted at the request of GT professors to insure academic honesty of future students. Analysis reports can be obtained by emailing jandrews63@gatech.edu_ 

This repository contains code used for ML projects that focused on:
 * Supervised Learning
 * Randomized Optimization
 * Unsupervised Learning and Dimensionality Reduction
 * Markov Decision Processes
 
 
 ### Supervised Learning: 
  - Uses two data sets obtained from [UCI Machine Learning Repository] (https://archive.ics.uci.edu/ml/index.php)
  - Both datasets were altered so they contain a binary target variable that algorithms seek to classify correctly
  - Classification performed using:
        1) Decision Trees
        2) Neural Network (ANN) 
        3) Boosting
        4) Support vector Machine (SVM)
        5) k-Nearest Neighbor (kNN)
        
        
### Randomized Optimization:
  1) Uses Randomized optimization to assign weights to an ANN (using one of the data sets from Supervised Learning analysis)
  2) Uses Randomized optimization techniques to solve classic optimization problems such as:
        - Travelling Salesman
        - Knapsack 
        - Four Peaks
  - Randomized Optimization algorithms used for ANN weights and classic problems:
        - Random Hill Climbing
        - Simulated Annealing
        - Genetic Algorithm
        
        
### Unsupervised learning and Dimensionality Reduction:
  Uses the same two data sets from Supervised Learning with a binary target variable in a series of experiments
   1) Cluster the data using:
        - k-Means
        - Expectation Maximization
   2) Perform Dimensionality Reduction using:
        - Principle Component Analysis (PCA)
        - Independent Component Analysis (ICA)
        - Randomized Projection (RP)
        - Random Forests
   3) Cluster on the reduced data from each method in part 2 using:
        - k-means
        - Expectation Maximization
   4) Create an ANN to classify the data 
        - Use the reduced data created by each method in Part 2 as input
        - Use the reduced data of each method from Part 2 and cluster labels from each algorithm in Part 3
  
   (The write up then compares the results from all 12 models to results of ANN created in Supervised Learning project.)
   
   
### Markov Decision Processes:
   

       

 
