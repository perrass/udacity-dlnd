# Deep Residual Learning for Image Recognition

## Intro

Deep networks naturally integrate low/mid/high-level features and classifiers in an end-to-end multi-layer fashion, and the levels of features can be enriched by the number of stacked layers.

Then, a question arises: **Is learning better networks as easy as stacking more layers?**

Obstacles:

* Vanishing/exploding gradients, but this problem can be solved by normalized initialization and intermediate normalization layers, which enable networks with tens of layers to start converging for stochastic gradient descent with back-propagation
* Degradation problem: with the network depth increasing, accuracy get saturated and then degrades rapidly. Unexpectedly, such degradation **is not caused by overfitting, and adding more layers ot a suitable deep model leads to higher training error**. This indicates that **not all systems are similarly easy to optimize**

Solution is deep residual learning framework. Formally, denoting the desired underlying mapping as $H(x)$, we let the stacked nonlinear layers fit another mapping of $F(x) = H(x) - x$. The original mapping is recast into $F(x)+x$. We hypothesize that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping. **To the extreme, if an identity mapping were optimal, it would be easier to push the residual to zero than to fit an identity mapping by a stack of nonlinear layers** (PS: **$H(x)$ is already stacked**)

## Deep Residual Learning

![](/assets/res_learning.png)

We consider a building block defined as:
$$
\mathbf y = F(\mathbf x, {W_i}) + \mathbf x \qquad(1)
$$
Here $\mathbf x$ and $\mathbf y$ are the input and output vectors of the layers considered. The function $F(\mathbf x, {W_i})$ represents the residual mapping to be learned. In the former figure, **$F=W_2\sigma(W_1\mathbf x)$ in which $\sigma$ denotes ReLU and the biases are omitted for simplifying notations. The operation $F+\mathbf x$ is preformed by a shortcut connection and element-wise addtion. We adopt the second nonlinearity after the addtion (i.e., $\sigma(\mathbf y)$)**

The dimensions of $\mathbf x$ and $F$ must be equal. If this is not the case, we can perform a linear projection $W_s$ by the shortcut connections to match the dimensions.
$$
\mathbf y = F(\mathbf x, {W_i}) + W_s\mathbf x \qquad(2)
$$
We can also use a square matrix $W_s$ in (1). But the identity mapping is sufficient for addressing the degradation problem and is economical and thus $W_s$ is only used when matching dimensions.

The form of the residual function $F$ is flexible. In the paper, $F$ has two or three layers, but more is possible. However, **if there is only one layer, (1) is similar to a linear layer** $\mathbf y = W_1 \mathbf x + \mathbf x$, which is not useful.

## Network Architectures

### VGG19

Simple design rules:

1. For the same output feature map size, the layers have the same number of filters
2. If the feature map is halved, the number of filters is doubled so as to preserve the time complexity per layer

We perform **downsampling** directly by convolutional layers that have a stride of 2. The network ends with a global average pooling layer and a 1000-way fully-connected layer with softmax.

### 34-layer Residual

The identity shortcuts can be directly used when the input and output are of the same dimensions. When the dimensions increase, we consider two options:

1. The shortcut still performs identity mapping, with extra zero entries padded for increasing dimensions. (Padding with zero)
2. The projection shortcut is used to match dimensions

The second method is better, and authors argue

>Zero-padded dimensions in A indeed have no residual learning

### Implementation

* **Scale augmentation and color augmentation** are used. 
* BN is adopted after each convolution and before activation. 
* The mini-bath size for SGD is 256. 
* The learning rate starts from 0.1 and is divided by 10 when the error plateaus, and the models are trained for up to $60\times10^4$ iterations. 
* A weight decay of 0.0001 and a momentum of 0.9
* Dropout is not used

### Bottleneck Architectures

Because of concerns on the training time that we can afford, we modify the building block as a **bottleneck** design. For each residual function $F$, we use $1\times1$, $3\times3$, $1\times1$ convolutions, where **$1\times1$ layers are responsible for reducing and then increasing dimensions, leaving the $3\times3$ layer a bottleneck with smaller input/output dimensions.

![](/assets/res_bottleneck.png)

## Overall

![](/assets/resnet.png)

