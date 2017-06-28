# Going Deeper with Convolutions

## Intro and Related Work

In GoogLeNet, the purpose of $1\times1$ convolutions is to reduce dimension and to remove computational bottlenecks. This allows for not just **increasing the depth, but also the width of our networks** without a significant performance penalty.

## Motivation and High Level Considerations

The most straightforward way of improving the performance of deep neural networks is by increasing their size. This includes both increasing the depth - **the number of network levels**, and its width - **the number of units at each level**. However, this simple solution comes with two major drawbacks.

1. Bigger size typically means a larger number of parameters, which makes the enlarged network more prone to overfitting, especially if the number of labeled examples in the training set is limited.
2. The uniform increase of network size is the dramatically increased use of computational resources.

## Inception Architecture





