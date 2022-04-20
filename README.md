# Gradient Difference Loss (GDL) in PyTorch
A simple implementation of the Gradient Difference Loss function in PyTorch, and its custom formulation with MSE loss function, for the training of Convolutional Neural Networks. 

First proposed in [[1]](#1).

Expression of the Mean Squared Error (already implemented in PyTorch).
<img src="https://render.githubusercontent.com/render/math?math=\displaystyle MSE\left(\mathbf{Y}_k,\hat{\mathbf{Y}}_k\right) = \frac{1}{N_x N_y}\sum_{i=1}^{N_x}\sum_{j=1}^{N_y}(\hat{y}_{i,j} - y_{i,j})^2">

Expression of the Gradient Difference Loss (from [[1]](#1)):
<img src="https://render.githubusercontent.com/render/math?math=\displaystyle GDL \left(\mathbf{Y}_k,\hat{\mathbf{Y}}_k\right) = \frac{1}{N_x N_y } \sum_{i=1}^{N_x}\sum_{j=1}^{N_y}[\left((\hat{y}_{i+1,j} - \hat{y}_{i,j}) -  (y_{i+1,j} - y_{i,j}) \right)^2 - \left(( \hat{y}_{i,j+1} - \hat{y}_{i,j} )- ( y_{i,j+1} - y_{i,j} )\right)^2]">

Hybrid Loss function combining GDL and MSE. Lambdas are weighting coefficients (scalars) used to balance the participation of the GDL loss and MSE loss. For a given 2D prediciton, the GDL loss is typically much higher than the MSE Loss, so lambdaGDL is set smaller than lambdaMSE. Their value can be set by testing on the available data (from [[2]](#2)).
<img src="https://render.githubusercontent.com/render/math?math=\displaystyle \mathcal{J}\left(\mathbf{Y}_k,\hat{\mathbf{Y}}_k\right) = \lambda_{MSE} MSE\left(\mathbf{Y}_k,\hat{\mathbf{Y}}_k\right) + \lambda_{GDL} GDL\left(\mathbf{Y}_k,\hat{\mathbf{Y}}_k\right)">

## References
<a id="1">[1]</a> 
Mathieu et al. (2015). 
Deep multi-scale video prediction beyond mean square error. 
https://arxiv.org/abs/1511.05440

<a id="2">[2]</a> 
Alguacil et al. (2021). 
Predicting the Propagation of Acoustic Waves using Deep Convolutional Neural Networks. 
https://arc.aiaa.org/doi/10.2514/6.2020-2513



