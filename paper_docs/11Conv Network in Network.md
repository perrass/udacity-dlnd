# Network in Network

## Intro

Convolutional neural networks consist of alternating convolution layers and pooling layers. Convolution layers take inner product of the linear filter and the underlying receptive field followed by a nonlinear activation function at every local portion of the input.The resulting outputs are called **feature maps**

The **convolution filter in CNN is a generalized linear model (GLM)** for the underlying data patch, and we argue that the level of abstraction is low with GLM (GLM对数据的抽象能力较低). By abstraction we mean that the feature is invariant to the variants of the same concept (**??**). Replacing the GLM with a more potent nonlinear function approximator can enhance the abstraction ability of the local model. GLM can achieve a good extent of abstraction when the samples of the latent concepts are linearly separable (the variants of the concepts all live on one side of the separation plane defined by the GLM). Thus conventional CNN implicitly makes the assumption that **the latent concepts are linearly separable**. However, the data for the same concept often live on a nonlinear manifold, therefore the representations that capture these concepts are generally highly nonlinear function of the input. In NIN, the GLM is replaced with a "micro network" structure which is a general nonlinear function approximator.

We use MLPconv Layer and global average pooling to achieve non-linearity.

In traditional CNN, it is difficult to interpret how the category level information from the objective cost layer is passed back to the previous convolution layer due to the fully connected layers which act as a black box in between. In contrast, global average pooling is more meaningful and interpretable as it enforces correspondance between feature maps and categories, which is made possible by a stronger local modeling using the micro network (**PS: NIN使连接feature maps和categories变得可能?**). Furthermore, the fully connected layers are prone to overfitting and heavily depend on dropout regularization, while **global average pooling is itself a structural regularizer, which natively prevents overfitting for the overall structure** (PS: 所以MLAPP里GLM和Sparsity那两章还是要看啊)

## Convolutional Neural Network

Assume $(i, j)$ is the pixel index in the feature map, $x_{ij}$ stands for the input patch centered at location $(i, j)$, and $k$ is used to index the channels of the feature map.

---

**Q: is ReLU or max pooling before? is ReLU or bn before???**

The answer of the first question is same.
$$
f_{i, j, k} = max(bn(w^T_kx_{i,j}), 0) \\
f_{i, j, k} = bn(max(w^T_kx_{i,j}, 0))
$$
The answer of the second question is ReLU before, because **bn** can introduce negative value, which means more information.

---

## MLPCONV Layer

Both the linear convolutional layer and the mlpconv layer map the local receptive field to an output feature vector. The mlpconv maps the input local patch to the output feature vector with a multilayer perceptron consisting of multiple fully connected layers with nonlinear activation functions.

![](/assets/1_1_conv.png)

## Global Average Pooling

**Global average pooling is a layer to replace the trainditional fully connected layers in CNN**. The idea is to **generate one feature map for each corresponding category of the classification task** in the last mlpconv layer. **Instead of adding fully connected layers on top of the feature maps, we take the average of each feature map, and the resulting vector is fed directly into the softmax layer**

Benifits

* The feature maps can be easily interpreted as categories confidence maps
* No parameter to optimize in this layer, hence overfitting is avoid
* **Global average pooling sums out the spatial information, thus it is more robust to spatial translations of the input**

![](/assets/1_1_conv_2.png)

**PS: Why 1_1 conv is a regularizer?**

## Network In Network Structure

![](/assets/1_1_conv_3.png)

**The overall structure of NIN is a stack of mlpconv layers, on top of which lie the global average pooling and the objective cost layer.** Sub-sampling layers can be added in between the mlpconv layers as in CNNand maxout networks. The figure is an example.

## Summary of NIN and Maxout

1. Maxout wants to simulate all convex activation function using $max(z_1, ..., z_n)$
2. NIN wants to simulate all activation function using MLP (multi-layer perceptrons) and using global average pooling to regularize the results and to avoid overfitting.