# Going Deeper with Convolutions

## Intro and Related Work

In GoogLeNet, the purpose of $1\times1$ convolutions is to reduce dimension and to remove computational bottlenecks. This allows for not just **increasing the depth, but also the width of our networks** without a significant performance penalty.

## Motivation and High Level Considerations

The most straightforward way of improving the performance of deep neural networks is by increasing their size. This includes both increasing the depth - **the number of network levels**, and its width - **the number of units at each level**. However, this simple solution comes with two major drawbacks.

1. Bigger size typically means a larger number of parameters, which makes the enlarged network more prone to overfitting, especially if the number of labeled examples in the training set is limited.
2. The uniform increase of network size is the dramatically increased use of computational resources.

## Inception Architecture

The main idea of the Inception architecture is to consider how an optimal local sparse structure of a convolutional vision network can be approximated and covered by readily available dense components.

The Inception architecture are restricted to filter sizes $1\times1$, $3\times3$, $5\times5$. This decision was based more on **convenience rather than necessity**. In addition, a $3\times3$  max pooling with stride of 2 module is added to the structure.

![](/assets/inception_naive.png)

As these "Inception modules" are stacked on top of each other, their output correlation statistics are bound to vary: as features of higher abstraction are captured by higher layers, their spatial concentration is expected to decrease. This suggests that **the ratio of $3\times3$ and $5\times5$ convolutions should increase as we move to higher layers**. 

However, a big problem issued by the stack strategy. That is, when stacking is used the depth of convolutional layer is increased linearly, and the parameters in $3\times3$ and $5\times5$ would be increased quadratically. Hence, it is prohibitively expensive for naive Inception. The solution to this is to add $1\times1$ conv layers before $3\times3$, $5\times5$ modules and after max pooling. This leads to keep the same depth with the increase of layers.

![](/assets/inception_1_1.png)

Besides being used as reductions, they also include the use of rectified linear activation **making them dual-purpose**.

## GoogLeNet

![](/assets/googlenet.png)

The network is 22 layers deep when counting only layers with parameters (or 27 layers if we also count pooling). The overall number of layers used for the construction of the network is about 100. **The use of average pooling before the classifier is based on NIN, although our implementation has an additional linear layer.** The linear layer enables us to easily adapt our networks to other label sets, however it is used mostly for convenience and we do not expect it to have a major effect. We found that a move from fully connected layers to average pooling improved the top-1 accuracy by about 0.6%, however the use of dropout remained essential even after removing the fully connected layers.

The exact structure of the extra network on the side, including the auxiliary classifier, is as follows:

* An average pooling layer with $5\times5$ filter size and stride 3, resulting in an $4\times4\times512$ output for the (4a), and $4\times4\times528$ for the 4d stage.
* A $1\times1$ convolution with 128 filters for dimension reduction and recified linear activation
* A fully connected layer with 1024 units and rectified linear activation
* A dropout layer with 70% ratio of dropped outputs.
* A linear layer with softmax loss as the classifier (pre-dicting the same 100 classes as the main classifier, but removed at inference time)