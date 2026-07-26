# Bhils
Project repository for brainhack in Rome 2025. Bhils stands for "Brain Hacking In Latent Space". 

For the brain hackathon event, we analysed both a toy dataset and a MRI dataset. This repository contain the Matlab and Python codes used for the analysis, that are YOUR_CODE and TIAGOS'CODE. The full explanation of the analysis done will be available in the PAPER coming soon!

Below, you can find the analysis we did using the toydataset, in order to clear our ideas up. The toydataset results have been actually useful also to confirm the results regarding the MRI dataset.

This repo contains also the "MRI dataset" section that shows where the samples of each cluster are located in the parameters distributions.

## Analysis of the toydataset

The toy dataset is made of two independent variables X (height $X_1$ and distance to work $X_2$), 
two dependent variables (weight $Y_1$ and shoe size $Y_2$) and nine samples. 

You can find the dataset in the video of this tutorial: [Partial Least Squares Regression 1 Introduction (1/4)](https://www.youtube.com/watch?v=AxmqUKYeD-U).

The toy dataset X and Y variables are related through different linear correlation
coefficients values: $r_{Y1X1}=0.89$, $p-value=0.0012$; $r_{Y1X2}=0.06$, $p-value=0.87$; 
$r_{Y2X1}=0.91$, $p-value=0.0006$; $r_{Y2X2}=0.05$, $p-value=0.89$; 
$r_{X1X2}=0.07$, $p-value=0.84$; $r_{Y1Y2}=0.78$, $p-value=0.01$.

<p align="center">
  <img 
    width="400"
    src="https://github.com/ManuelaCarriero/Bhils/blob/main/documentation_images/work_in_progress.png"
  >
</p>

## Analysis of the MRI dataset 

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
