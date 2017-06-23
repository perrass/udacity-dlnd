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

![](/assets/AlexNet.png)

1. 96 kernels with size $11\times 11\times 3$  -> $(11\times11\times3+1)\times96 = 34966$
2. 256 kernels with size $5\times5\times48$ -> $(5\times5\times48+1)\times256 = 307456$
3. 384 kernels with size $3\times3\times256$ -> $(3 \times 3\times256+1) \times 384 = 885120$
4. 384 kernels with size $3\times3\times192$ -> $(3\times3\times192+1)\times384=663936$
5. 256 kernels with size $3\times3\times192$ -> $(3\times3\times192+1)\times256 = 442624$
6. cnn to fully connected layers with 4096 neurons -> $13\times13\times128\times4096 = 88604672$
7. fully connected -> $4096 \times 4096 = 16777216$
8. fully connected to softmax -> $4096 \times 1000 = 4096000$

**The ratio of the weigths of CNN is $2334102/111811990 \approx 2.09\% $

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

