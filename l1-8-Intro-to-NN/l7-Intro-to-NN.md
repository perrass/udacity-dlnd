## Introduction to Neural Networks

### Perceptron

#### Definition

The interconnected nodes in a network are called **perceptrons** or **neurons**

#### Weights

When building a neural network, we don't know which node is important. The network can learn for itself which data is most important and adjust how it considers that data (feature selection??). That is, weights of perceptrons are self-adjusted.

#### Activation function

Activation function is to transfer the weighted data into an expected output.

If the activation function is `f(x) = x`, the algorithm of network is regression, and if that is sigmoid function, the algorithm of network is logistic regression.

#### And, or, not perceptron

* The activate function is the most basic one (0, 1)
* Two ways to go from an AND to an OR are: increase the weights, **decrease** the magnitude of the bias

### Gradient Descent

[Sebastian Ruder's article: An overview of gradient descent](http://sebastianruder.com/optimizing-gradient-descent/)

#### Overview

Gradient descent is a way to minimize an objective function $J(\theta)$ parameterized by a model's parameters $\theta \in \mathbb R^d$ by updating the parameters in the opposite direction of the gradient of the objective function $\nabla_{\theta} J(\theta)$. **The learning rate** $\eta$ determines the size of the steps we take to earch a (local) minimum.

#### Batch Gradient Descent

Computes the gradient of the cost function w.r.t to the parameters \theta for the **entire training dataset**:

$$ \theta = \theta - \eta * \nabla_{\eta} J(\theta) $$

Batch gradient descent **is guaranteed to** the global minumum for convex error suerfaces and to a local minumum for non-convex surfaces

#### Stochastic Gradient Descent

Performing one update at a time to decrease the redundant computations for large datasets when using batch gradient descent:

$$\theta = \theta - \eta * \nabla_{\theta} J(\theta; x^{(i)};y^{(i)})$$

If the learning rate is low, SGD shows the same convergence behaviour as batch gradient descent !!!

#### Mini-batch Gradient Descent

$$ \theta = \theta - \eta * \nabla_{\theta} J(\theta;x^{i:i+n};y^{i:i+n})$$

**Mini-batch gradient descnet is typically the algorithms of choice when training a neural network and the term SGD ususally is employed also when mini-batches are used.**

#### Example

To calculate the gradient of SSE $E = \frac 1 2 \sum(y - \hat y)^2$
$$
\begin{align} \frac {\partial E} {\partial w_i} & = \frac {\partial} {\partial w_i} \frac 12  (y-\hat y(w_i))^2 = (y-\hat y(w_i))\frac {\partial} {\partial w_i}(y-\hat y(w_i)) \\
	& = -(y - \hat y) \frac {\partial \hat y} {\partial w_i} = -(y - \hat y) f'(h)\frac {\partial} {\partial w_i}\sum w_ix_i \\
	& = -(y - \hat y) f'(h) x_i
\end{align} \\
Where,
\frac {\partial} {\partial w_i} \sum_i w_ix_i = \frac {\partial} {\partial w_i}[w_1x_1 + ... + w_ix_i + ... + w_nx_n]
$$

### Challenges

* Choosing a proper learning rate can be difficult
* Chossing a proper schedules for dynanmic learning rate
* The same learning rate applies to all paramter updates. If our data is sparse and our features have very different frequencies, we might not want to update all of them to the same extent
* Avoiding getting trapped in their **numerous suboptimal local minima**, when minimizing highly non-convex error functions common for neural networks.

### Forward Pass

Input -> Hidden\_1 -> Hidden\_2 -> Output

In detail, if we choose linear combination for the weights calculation

Input -> `np.dot(input, weights_of_input_to_hidden_1)` -> activation function (get hidden\_1 output) -> `np.dot(hidden\_1_output, weights_of_hidden_1_to_hidden_2)` -> activation function -> output

### Backpropagation

The backpropagation algorithm uses the **chain rule** to **find the error** with the respect to the weights connecting the input layer to the hidden layer (for a two layer network)

The error is going forward, and the output is known, so we can using the **final output errors and the weights** to obtain the error of each neurons from former layers. This process is going backward. Hence, there exists the problem of **vanishing gradient**

[Yes, you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.6xszsfjee)

Except for **vanishing gradient**, the sigmoid is that its local gradient (z(1-z)) achieves a maximum at 0.25, when z = 0.5. That means that **every time the gradient signal flows through a sigmoid gate, its magnitude always diminishes by one quarter (or more)**. If you're using basic SGD, this would make the lower layers of a network train much slower than the higher ones.

**If you're using sigmoids or tanh non-linearities in network and you understand backpropagation you should always be nervous about making sure that the initialization doesn't cause them to be fully saturated**

[如何理解back propgation - 知乎](https://www.zhihu.com/question/27239198)

#### Example1

There are **one** hidden layer in a neural network

1. the output error is `error = y - final_outputs`
2. hidden layer's contribution to the error `hidden_error = np.dot(self.weights_hidden_to_output, error)`
3. if the activation function is sigmoid, `output_error_term = error * final_outputs * (1-final_outputs)` 
4. `hidden_error_term = hidden_error * hidden_outputs *(1 - hidden_outputs)`
5. `delta_weights_h_o = output_error_term * hidden_outputs[:, None]`
6. `delta_weights_i_h = hidden_error_term * inputs[:, None]`
7. For one epoch, `weights_hidden_to_output = learning_rate * delta_weights_h_o`
8. `weights_input_to_hidden = learning_rate * delta_weights_i_h`

From this we can get, the gradient step of the weights from hidden to output is 
$$
\Delta w_{jk} = \delta^k_jx_j = (y-\hat y) \times f'(\hat y) \times x_j
$$
the gradient step of the weights from input to hidden is 
$$
\Delta w_{ij} = \delta^j_ix_i = [\sum w_{jk}(y-\hat y)]f'(h_j)x_i
$$

### Summary

In conclusion, the forward pass is the process to get the prediction, using linear combination and sigmoid function in this case. When we have the prediction, we use the error from true value and prediction, then use the gradient of this error to update the weights from hidden to output. 

Then we should calculate the update of the weights from input to hidden. First, we should find the contribution of each nodes in hidden layer to the error, the error of each node is from the weights from hidden to output, this is the **hidden error**, then using the gradient formular in *Gradient Descent* part, we can get the gradient step of weights from input to hidden.

Then we sum all the gradient in one epoch and multiply learning_rate (if we use SGD) to update the weights from input to hidden and hidden to output.