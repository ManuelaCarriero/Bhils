To run it, open the command prompt inside this folder (same folder with pca_main.py, bhils_functions.py, etc) and then run: 
> python pca_main.py (Windows) 
or 
$ python3 pca_main.py (Linux) 

"pca_main.py" performs a principal component analysis (PCA) on the dataset contained in the "datasets" folder.   
User-defined functions used in the analysis are stored in "bhils_functions.py". 

In a nutshell, the steps are:
	1. In "bhils.find_optimal_numPCA_components(...)", the program determines the minimum number of principal components needed to 
	explain at least a certain percentage of the total variance in the data, as defined by the variable "optVarThreshold".
	The optimal number of components is stored in the integer "pPCA".
	2. The latent space of the input dataset is then created using "pPCA" principal components.
	3. DBSCAN clustering is performed on this latent space, according to the parameters "r" (max distance between points in a cluster) and "min_s" (min number of points in a cluster) 

The program generates and saves plots illustrating:
    - The covariance matrix of the dataset features
    - A bar plot showing the variance explained (eigenvalues) by each principal component
    - The cumulative explained variance as a function of the number of components
    - A 3D plot of the first three principal components, colored according to the DBSCAN clustering (with optional animation frame generation if saveFrames = True)
    - Several 1D plots of the remaining principal components (4th onward), colored according to the DBSCAN clustering, with points of each different cluster vertically offset for clearer visualization

All figures are automatically saved in the "figures" folder (created if it doesn't already exist), in both png and pdf formats.