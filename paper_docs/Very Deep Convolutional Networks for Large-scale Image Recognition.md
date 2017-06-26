# Very Deep Convolutional Networks for Large-scale Image Recognition

## ConvNet Configurations

### Architecture

The only preprocessing we do is substracting the mean RGB value, computed on the training set, from each pixel.

![](/assets/vgg_conf.png)

The field of convolutional filter is $3\times3$, with the stride and padding of 1.

Max-pooling is performed over a $2\times2$ pixel window, with stride 2. 

The activation function is ReLU and the that in the last layer is soft-max.

### Discussion

The stacks of two $3\times3$ layers has an effective receptive field of $5\times5$, and the stacks of three has an effective receptive field of $7\times7$. Hence, **which is better?**

1. We incorporate three non-linear rectification layers instead of a single one, which **makes the decision function more discriminative**
2. The amount of parameters of $7\times7$ conv is $7^2C^2 = 49C^2$, where C is the number of channels, whereas that of the $3\times3$ is $3(3^2C^2) = 27C^2$. **This can be seen as imposing a *regularisation* on the $7\times7$ conv filters**, forcing them to have a decomposition through the $3\times3$ filters with non-linearity injected in between.

## Training

### Parameters

* The batch size was set to 256
* Momentum was 0.9
* The training was regularised by weight decay (**the L2 penalty multiplier set to $5\times10^{-4}$**)
* The dropout regularization for the first two fully-connected layers (p is 0.5)
* The learning rate was initially set to $10^{-2}$, and then decreased by a factor of 10 when the validation set accuracy stopped improving
* The weights from a normal distribution with the zero mean and $10^{-2}$ variance
* The biases were initialised with zero

### Scaling

We consider two approaches for setting the training scale $S$.

The first is to fix $S$ with $S = 256$ and $S = 384$. Given a ConvNet configuration, we first trained the network using $S = 256$. To speed-up training of the $S=384$ network, it was initialised with the weights pre-trained with $S=256$, and we use a smaller initial learning rate of $10^-3$

The second approach to setting $S$ is multi-scale training, where each training image is individually rescaled by randomly sampling $S$ from a certain range $[S_{min}, S_{max}]$ (we use $S_{min} = 256$ and $S_{max} = 512$). Since objects in images can be of different size, it is beneficial to take this into account during training. This can also be seen as **training set augmentation by scal jittering**, where a single model is trained to recognise objects over a wide range of scales. For speed reasons, we trained multi-scale models by fine-tuning all layers of a single-scale model with the same configuration, pre-trained with fixed $S=384$

