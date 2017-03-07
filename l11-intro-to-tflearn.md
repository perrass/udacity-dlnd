## Intro to TFLearn

### ReLU

**Rectified linear units** activations are the simplest **non-linear** activation function $$f(x) = max(x, 0)$$. The derivative of this is 1, if the input is positive. Hence, it's much faster than sigmoid function and deliminate gradient vanishing and time-costing from sigmoid.

#### Drawbacks

From CS231n

> Unfortunately, ReLU units can be fragile during training and can “die”. For example, a large gradient flowing through a ReLU neuron could cause the weights to update in such a way that the neuron will never activate on any datapoint again. The ReLU units can irreversibly die during training since they can get knocked off the data manifold. For example, you may find that as much as 40% of your network can be “dead” (i.e. neurons that never activate across the entire training dataset) if the learning rate is set too high.

### Softmax

**Softmax** is a generalization of logistic function that “squashes”(maps) a K-dimensional vector z of arbitrary real values to a K-dimensional vector σ(z) of real values in the range (0, 1) that add up to 1. While, **sigmoid** maps **one** arbitrary real values to a value in range (0, 1)

This is more suitable for multi-classification, and mathematically, it is

​								$$\sigma(\mathbf z)_j \ = \ {{e^{z_j}}\over{\sum^K_{k=1}\ e^{z_k}}}\qquad for j = 1,...,K$$

### Tanh

The function of tanh is 

​										$$tanh(x) = {{e^{2x}-1} \over {e^{2x} +1}}$$

The range is [-1, 1] and usually used in RNN and LSTM

[Why use tanh for activation function of MLP?](http://stackoverflow.com/questions/24282121/why-use-tanh-for-activation-function-of-mlp) in stackflow

[tanh activation](https://www.quora.com/search?q=tanh+activation) in quora

### Cross entropy

The error between prediction vector and real value is **proportional to how far apart these vectors are**. Cross entropy is used to calculate this distance. Then, the goal of training network is to minimize cross entropy. $$D(\hat y, y) = -\sum_j y_j ln{\hat y_j}$$

