# ImageNet Classification with Deep Convolutional Neural Networks

## The Architecture

### ReLU Nonlinearity

> In terms of traning time with gradient descent, these **saturating nonlinearities are much slower than the non-saturating nonlinearities**

**Saturating**

$f$ is non-saturating 
$$
iff\quad (|lim_{z->-\infty}f(z)| = + \infty \ \lor |lim_{z->+\infty}f(z)| = +\infty)
$$
$f$ is saturating iff $f$ is not non-saturating

For ReLU $lim_{z->+\infty}f(z) = +\infty$, hence it is non-saturating, while sigmoid or tanh is saturating.

### Local Response Normalization

ReLUs have the desirable property that **they do not require input normalization to prevent them from saturating.** If at least some traning examples produce a positive input to a ReLU, learning will happen in that neuron. However, we still find that the **following local normalization scheme aids generalization**.

**Using BN or LN instead**

### Overlapping Pooling

A pooling layer can be thought of as consisting of a grid of pooling units spaced $s$ pixels apart, each summarizing a neighborhood of size $z\times z$ centered at the location of the pooling unit. 

If we set $s=z$, we obtain traditional local pooling, and if we set $s < z$, we obtain overlapping pooling. This is what we use throughout our network, with $s=2$ and $z=3$

We generally observe during training that **models with overlapping pooling find it slightly more diffcult to overfit**

### Overall Architecture

All architecture would be discussed in one blog

## Reducing Overfitting

### Data Augmentation

> We employ two distinct forms of data augmentation, both of which allow transformed images to be produced from the original images with **very little computation**, so the **transformed images do not need to be stored on disk**. In our implementation, the transformed images are generated in Python code on the CPU while the GPU is training on the previous batch of images. So these data agumentation schemes are **computationally free**

#### Image translations and horizontal reflections

$\color{red}{???}$

#### Altering the intensities of the RGB channels in training images

We perform PCA on the set of RGB pixel values throughout the ImageNet training set. To each training image, we add multiples of the found principal components, with magnitudes proportional to the corresponding eigenvalues times a random vairable drawn from a Gaussian with mean zero and standard deviation 0.1. There fore to each RGB image pixel $I_{xy} = [I_{xy}^R, I_{xy}^G, I_{xy}^B]$, we add the following quantity
$$
[\mathbf p_1, \mathbf p_2, \mathbf p_3][\alpha_1\lambda_1,\alpha_1\lambda_2,\alpha_1\lambda_3]^T
$$
where $\mathbf p_i$ and $\lambda_i$ are $ith$ eigenvector and eigenvalue of the $3\times3$ covariance matrix of RGB pixel values, respectively, and $\alpha_i$ is the aforementioned random variable.

### Dropout

**See dropout paper**

