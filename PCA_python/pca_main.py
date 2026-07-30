#!/usr/bin/env python3                        
# ↑ this is a shebang: https://stackoverflow.com/questions/7670303/purpose-of-usr-bin-python3-shebang 

# import required python packages:    
import pandas as pd  
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA     
import numpy as np     
import matplotlib.pyplot as plt  
from sklearn.cluster import DBSCAN  
import os  
from matplotlib.colors import to_rgb  
import bhils_functions as bhils 

# --------------- FUNCTIONS DEFINITIONS 


#---------------- LOADING DATASET 
# set up your input data n x p matrix here: 
# n -> number of samples (for each feature) 
# p -> number of features = maximum possible number of PCA principal components  

# Reading from external file: 
path_to_input_file = "./datasets/dataset_RM.xlsx" 
df = pd.read_excel(path_to_input_file) 			# this is a pandas.DataFrame 
df.info()										# print info about input data matrix 
M_std = StandardScaler().fit_transform(df)		# data normalization 
n = M_std.shape[0]  # number of samples (rows) 
p = M_std.shape[1]  # number of features (columns) 
print(str(n) + " x " + str(p) + " input data matrix with " + str(p) + " features and " + str(n) + " number of samples for each feature") 
# --------------- 

# --------------- PCA 
pca_full = PCA().fit(M_std) # model object from sklearn.decomposition that holds all the PCA results, considering the maximum possible number of PCA components (=p)  

optVarThreshold = 0.90 
pPCA = bhils.find_optimal_numPCA_components( pca_full , optVarThreshold ) # find minimum number of optimal PCA components 

# PCA with optimal number of principal components 
pca_reduced = PCA(n_components=pPCA) 
vecs = pca_reduced.fit_transform(M_std) 
# Reconstruct a pandas.DataFrame with the optimal number of PCA components: 
reduced_df = pd.DataFrame(data=vecs, columns=[f'P{i+1}' for i in range(pPCA)]) 

bhils.plot_covariance_matrix( pca_full ) # Plot the p x p covariance matrix between all possible pairs of features of the input data   
 
# Perform DBSCAN in PCA latent space:    
r = 1 																# radius for DBSCAN 
min_s = 5 															# minimum number of samples in a cluster 
clustered_labels = bhils.perform_clustering( vecs, r, min_s )  			# apply DBSCAN in the PCA latent space 

bhils.plot_PCA_latent_space( vecs, pPCA, clustered_labels, saveFrames = False )  # plots the 3 principal components (if they exist); 4th component is represented by the size of the points in the 3D scatter plot 

bhils.plot_remaining_components(vecs, pPCA, clustered_labels) # plots 1D lines of the remaining components (4th component onward), if they exist 

plt.show() 
# ---------------   


