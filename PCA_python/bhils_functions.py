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

# --------------- FUNCTIONS DEFINITIONS 


def save_figure(fig, filename):

    # Define output directories
    png_dir = os.path.join("./figures/png/")
    pdf_dir = os.path.join("./figures/pdf/")

    # Create folders if they don't exist
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    # Full paths
    png_path = os.path.join(png_dir, f"{filename}.png")
    pdf_path = os.path.join(pdf_dir, f"{filename}.pdf")

    # Save the figure
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight") 

def find_optimal_numPCA_components( pca_full , optVarThreshold ): 
	# pca_full -> model object from sklearn.decomposition that holds all the PCA results  
	# optVarThreshold -> threshold \in [0,1] to determine optimal number of PCA components 
	
	# --- Analysis to find the optimal number of PCA components 
	# (i.e. returns mininum number of components that give sum(variance) >= optVarThreshold) 
	eigenvalues = pca_full.explained_variance_ratio_  
	explained_var = np.cumsum(eigenvalues) 
	components_indexes = range(1, len(explained_var)+1) 

	# Plot cummulative variance 
	fig = plt.figure(figsize=(8, 5))
	plt.plot(components_indexes, explained_var, marker='o', linestyle='--', color='b')
	plt.axhline(y=optVarThreshold, color='r', linestyle='--', label=f'Threshold {int(optVarThreshold*100)}%')  
	plt.title('Explained cumulative variance')
	plt.xlabel('Number of principal components') 
	plt.ylabel('Cumulative variance')
	plt.xticks(range(1, len(explained_var)+1)) 
	plt.grid(True)
	plt.tight_layout() 
	
	# Bar plot of the explained variance (eigeinvalues) of each component: 	
	bar_fig = plt.figure()
	plt.title('Eigenvalues of PCA Components \n (considering all possible PCA components for the given data set)')  
	plt.bar(components_indexes, eigenvalues, color='tab:blue')   
	plt.xlabel('PCA principal components')
	plt.ylabel('Eigenvalue') 
	plt.xticks( components_indexes )   
	print(f'Sum of eigenvalues: { sum(eigenvalues) }') # Should always be 1 (I believe!)  

	# Determnines the optimal number of PCA components: 
	num_components = np.argmax(explained_var >= optVarThreshold) + 1
	print(f"\nNumber of optimal PCA components (≥ {int(optVarThreshold*100.0)}% of variance explained): {num_components}") 
	
	save_figure(fig, "cumulative-variance") 
	save_figure(bar_fig, "eigenvalues-barplot")  
	 
	return num_components     
	
def plot_covariance_matrix( pca ): 
	# pca -> model object from sklearn.decomposition that holds all the PCA results 

	cov_matrix = pca.get_covariance() # p x p covariance matrix: each entry cov_matrix[i, j] represents the covariance between feature i and feature j  
	
	num_features = cov_matrix.shape[0] 

	fig = plt.figure()
	plt.title('Covariance matrix')
	plt.imshow(cov_matrix, cmap='bwr')
	plt.colorbar()     
	# Set ticks at positions 0 to num_features-1, with labels starting at 1
	ticks = range(num_features)
	labels = [str(i + 1) for i in ticks]
	plt.xticks(ticks, labels)
	plt.yticks(ticks, labels) 	

	plt.xlabel('Feature index') 
	plt.ylabel('Feature index')  
	
	save_figure(fig, "covariance-matrix")    
	   
def perform_clustering( vecs , r , min_s ):  
	# vecs is the 2D array of the reduced dataset (the latent space) 

	clustering = DBSCAN(eps=r, min_samples=min_s).fit(vecs) 
	clustered_labels = clustering.labels_  
	
	return clustered_labels 
	# clustered_labels holds the cluster labels for each data point  
	
def plot_remaining_components(vecs, pPCA, clustered_labels):
	unique_labels = np.unique(clustered_labels)
	num_clusters = len(unique_labels)
    
    # Define colors (black for noise, then tab10)
	color_map = plt.get_cmap('tab10')
	color_list = ['black'] + [color_map(i) for i in range(10)]
    
	for comp_idx in range(3, pPCA):  # from Component 4 onward
		fig, ax = plt.subplots(figsize=(8, 1.5 + 0.4 * num_clusters))

		for cluster_idx, label in enumerate(unique_labels):
			mask = clustered_labels == label
			x_vals = vecs[mask, comp_idx]
			y_vals = np.full_like(x_vals, fill_value=cluster_idx)
			ax.scatter(x_vals, y_vals, color=color_list[label + 1 if label != -1 else 0], alpha=0.6, label=f"Cluster {label+1}" if label != -1 else "Noise")

		# Style
		ax.set_yticks(range(num_clusters))
		ax.set_yticklabels([f"Cluster {label+1}" if label != -1 else "Noise" for label in unique_labels])
		ax.set_xlabel(f'Component {comp_idx + 1}')
		ax.set_title(f'1D Projection: Component {comp_idx + 1}')
		ax.set_xlim(vecs[:, comp_idx].min() - 1e-2, vecs[:, comp_idx].max() + 1e-2)
		ax.grid(False)
		ax.spines[['top', 'right']].set_visible(False)

		save_figure(fig, "component-"+str(comp_idx + 1))  
		plt.close(fig) 
			
      	
	
def plot_PCA_latent_space( vecs, pPCA, clustered_labels , saveFrames = False ):   
	# vecs -> 2D array of the reduced dataset (the latent space)  
	# pPCA -> number of PCA components considered   
	# saveFrames -> if True, saves png files for each different rotation angle of the 3D plot set below, that then can be used as frames for an animation  
	# clustered_labels holds the cluster labels for each data point   
	
	# Get the number of unique clusters (including -1 for noise)
#	clust_num_labels = np.unique(clustered_labels)
	# Define a color map (with enough distinct colors)
#	cmap = plt.get_cmap("tab10", len(clust_num_labels)) 
	# Create a color list by mapping labels to colors  
#	clust_color_list = np.array([cmap(i) for i in range(len(clust_num_labels))])       
	# Legend: first label is "Noise", others are "Cluster i" 
#	clust_label_list = ['Noise' if label == -1 else f'Cluster {label+1}' for label in clust_num_labels]  
	# num_labels gives the distinct cluster index labels, including -1 for noise (e.g. for 3 clusters: num_labels = [-1, 0, 1], where -1 is always noise) 
	# color_list is the list of colors of each cluster  
	# label_list is the list of strings for the labels of each cluster (same info as num_labels; e.g. for 3 clusters: label_list = ['Noise', 'Cluster 1', 'Cluster 2']) 

	
	
	   
    # Define colors: black for noise (-1) and other colors for clusters
	color_list = [to_rgb('black')] + list(plt.cm.tab10.colors)  # Create a color list to be used in the next array; First color is black for noise
	points_color_list = [color_list[i + 1] if i != -1 else to_rgb('black') for i in clustered_labels] # Color information for each point 

	if pPCA >= 2: 
	# Plot 2D PCA dimension-reduced data: 
		fig2D =	plt.figure()
		ax2D = fig2D.add_subplot()
		plt.title('PCA with DBSCAN clustering')
		plt.scatter(vecs[:, 0], vecs[:, 1], c=points_color_list)  
		plt.xlabel('Component 1') 
		plt.ylabel('Component 2')  

		# Legend: first label is "Noise", others are "Cluster i"
		unique_clust_indexes = np.unique(clustered_labels) 
		for i in range(0, len(unique_clust_indexes)):   
			ax2D.scatter([], [], [], c=color_list[i], label= "Noise" if unique_clust_indexes[i] == -1 else f"Cluster {int(i)}")     	 	 
		plt.legend(markerscale=5) 
				
		save_figure(fig2D, "pca-first-2-components") 	

	fourthComponentSizeValues = [] 
	if pPCA >= 4:  
		fourthComponentSizeValues = (vecs[:,3]+np.abs(min(vecs[:,3]))+0.5)*25  

	if pPCA >= 3:  
		
		defaultAngle = 30 # default angle for the 3D plot  
		
		os.makedirs("./figures/animation_frames/", exist_ok=True)  
		 
		iframe = 1 
		minAngle = 0 
		maxAngle = 360  
		totalFrames = 360  
		angleStep = (maxAngle-minAngle)/totalFrames  
		if saveFrames: 
			for angle in np.arange(minAngle, maxAngle, angleStep):   # Loop over the 3D plot rotation angle from minAngle to maxAngle, with steps of angleStep  
			#    df['traget'] = df['Altezza'] >= 180 
			#	print((vecs[:,3]+min(vecs[:,3]))*10)
			#	print(f'DBSCAN Clustered Labels: {cluestered_labels}')
			#    print(f'Clustered Labels MLSE: {np.sqrt(np.sum((true_labels-cluestered_labels)*+2))}')
				fig = plt.figure()
				ax = fig.add_subplot(projection='3d') 
				if pPCA >= 4: 
					ax.scatter(vecs[:, 0], vecs[:, 1], vecs[:,2], c=points_color_list, s = fourthComponentSizeValues ) 
				else: 
					ax.scatter(vecs[:, 0], vecs[:, 1], vecs[:,2], c=points_color_list )   
				
				ax.set_xlabel('Component 1')
				ax.set_ylabel('Component 2')
				ax.set_zlabel('Component 3') 
			
				# Rotate view: Here we fix the elevation to 30 degrees (adjust as needed)
				ax.view_init(elev=30, azim=angle) 
				
				# Save the frame as a PNG file, the file name corresponds to the current angle 
				filename = f"{iframe}.png" 
				iframe += 1 
				plt.savefig("./figures/animation_frames/"+filename) # dpi=300     
			
				plt.close(fig)  # Close the figure to free memory 

				if saveFrames: 
					print(f"Saving animation frames ({float(iframe)/float(totalFrames) * 100.0: .2f} % )", end='\r', flush=True)    
	
		# Plot the 3D figure with legend:  
		fig3D = plt.figure()    
		ax3D = fig3D.add_subplot(projection='3d') 
		fig3D.tight_layout()
		if pPCA >= 4: 
			ax3D.scatter(vecs[:, 0], vecs[:, 1], vecs[:,2], c=points_color_list, s = fourthComponentSizeValues ) 
		else: 
			ax3D.scatter(vecs[:, 0], vecs[:, 1], vecs[:,2], c=points_color_list )   
							
		# Legend: first label is "Noise", others are "Cluster i"
		unique_clust_indexes = np.unique(clustered_labels) 
		for i in range(0, len(unique_clust_indexes)):   
			ax3D.scatter([], [], [], c=color_list[i], label= "Noise" if unique_clust_indexes[i] == -1 else f"Cluster {int(i)}")     	 	 
		plt.legend() 
		
		ax3D.set_xlabel('Component 1')
		ax3D.set_ylabel('Component 2')
		ax3D.set_zlabel('Component 3')   
			
		# Rotate view: Here we fix the elevation to 30 degrees (adjust as needed)
		ax3D.view_init(elev=30, azim=defaultAngle) 	
		save_figure(fig3D, "pca-first-3-components")   

		

	
# --------------- 


