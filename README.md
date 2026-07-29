# Bhils (BILLS)
Project repository for BrainHack Rome 2025.

BHILS stands for Brain Hacking in Latent Space. To make the acronym easier to remember, the project was later renamed BILLS.

</p>
<p align="center">
  <img 
    width="400"
    src="https://github.com/ManuelaCarriero/Bhils/blob/main/documentation_images/comics.jpg"
  >

## Introduction to the repository

For the brain hackathon event, we analysed both a toy dataset and a MRI dataset in the PCA and PLS latent spaces. This repository contains the Matlab and Python codes used for the analysis, that are [apply_bhils.m](https://github.com/ManuelaCarriero/Bhils/blob/main/apply_bhils.m) and [https://github.com/ManuelaCarriero/Bhils/tree/tsansimoes-patch-1](https://github.com/ManuelaCarriero/Bhils/tree/tsansimoes-patch-1). 

A detailed description of the analysis will be provided in the accompanying paper, which is under revision.

In the meantime, this repository includes the complete analysis performed on a toy dataset, which was used to develop and validate our methodology. The results obtained from the toy dataset also helped confirm the findings observed in the MRI dataset.

The repository additionally contains an MRI dataset section, which illustrates how the samples assigned to each cluster in the unsupervised clustering analysis are distributed across the model parameter distributions.

## Analysis of the toydataset

The toy dataset is made of two independent variables X (height $X_1$ and distance to work $X_2$), 
two dependent variables (weight $Y_1$ and shoe size $Y_2$) and nine samples. 

You can find the dataset in the video of this tutorial: [Partial Least Squares Regression 1 Introduction (1/4)](https://www.youtube.com/watch?v=AxmqUKYeD-U).

Fig.1 shows the analysis of Principal Components, both for the PCA and PLS method. We can notice that, in both cases, the $X_1$ variable height contributes the most to the first principal component. 

Fig.2 reports the toy dataset in the PLS latent space considering monovariate X and Y models. The toy dataset X and Y variables are related through different linear correlation coefficients values. 


The toy dataset X and Y variables are related through different linear correlation
coefficients values: $r_{Y1X1}=0.89$, $p-value=0.0012$; $r_{Y1X2}=0.06$, $p-value=0.87$; 
$r_{Y2X1}=0.91$, $p-value=0.0006$; $r_{Y2X2}=0.05$, $p-value=0.89$; 
$r_{X1X2}=0.07$, $p-value=0.84$; $r_{Y1Y2}=0.78$, $p-value=0.01$.


We can notice that clustering is not affected by removing one of the two Y variables, which are both highly correlated at least to one of the X variables. We can instead notice that removing X1, which most explains the data variance, breaks the clear clustering pattern along the diagonal. Hence, the addition or removal of variables can modify latent space patterns.
The toy dataset in PLS latent space, considering multivariate X and Y, is reported in Fig.3 next to the PCA latent space plot. The toy dataset in PLS latent space is characterized by clear separation between clusters along the diagonal, differently from PCA. Thus, patterns in the latent space depend on the function to be maximized.




<p align="center">
  <img 
    width="800"
    src="https://github.com/ManuelaCarriero/Bhils/blob/main/documentation_images/pc_analysis.svg"
  >
  
Figure 1.


<p align="center">
  <img 
    width="800"
    src="https://github.com/ManuelaCarriero/Bhils/blob/main/documentation_images/latent_space_pls.svg"
  >
  
Figure 2.

<p align="center">
  <img 
    width="800"
    src="https://github.com/ManuelaCarriero/Bhils/blob/main/documentation_images/latent_space.svg"
  >
  
Figure 3.

  ## Analysis of the MRI dataset 

This section investigates the location of the clusters within the distribution of each parameter.

<p align="center">
  <img 
    width="400"
    src="https://github.com/ManuelaCarriero/Bhils/blob/main/documentation_images/dist_cmro2.svg"
  >
</p>
<p align="center">
  <img 
    width="400"
    src="https://github.com/ManuelaCarriero/Bhils/blob/main/documentation_images/dist_Rsoma.svg"
  >
</p>
<p align="center">
  <img 
    width="400"
    src="https://github.com/ManuelaCarriero/Bhils/blob/main/documentation_images/dist_fsoma.svg"
  >
</p>
<p align="center">
  <img 
    width="400"
    src="https://github.com/ManuelaCarriero/Bhils/blob/main/documentation_images/dist_fneurite.svg"
  >
</p>
<p align="center">
  <img 
    width="400"
    src="https://github.com/ManuelaCarriero/Bhils/blob/main/documentation_images/dist_fextra.svg"
  >
  </p>
<p align="center">
  <img 
    width="400"
    src="https://github.com/ManuelaCarriero/Bhils/blob/main/documentation_images/dist_Din.svg"
  >
    </p>
<p align="center">
  <img 
    width="400"
    src="https://github.com/ManuelaCarriero/Bhils/blob/main/documentation_images/dist_De.svg"
  >
