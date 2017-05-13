# Therotical Intro to RNN 

### What are RNNs?

The following figure shows what are RNNs.

![](/asset/RNN-Recurrent.png)

In detial, unfolding(unrolling) a RNN means that we write out the network for a complete sequence. For example, **if the sequence we care about is a sentence of 5 words, the network would be unrolled into a 5-layer neural network, one layer for each word.** The formulas that govern the computation happening in a RNN are as follows:

* $x_t$ is the input at time step t. For example, $x_1$ could be a one-hot vector corresponding to the second word of a sentence.
* $s_t$ is the hidden state at time step t. It's the memory of the network and **captures information about what happened in all the previous time steps, but if we use gradient clip, there would be some loss due to the $\mathbf {min(clip, w^n)}$**. $s_t$ is calculated based on the previous hidden state and the input at the current step: $s_t = f(Ux_t + Ws_{t-1})$. $f$ is the nonlinear activation function, such as ReLU and tanh. $s_{t-1}$ is required to calculate the first hidden state, is typically  initialized to all zeroes.
* $o_t$ is the output at step t, which is **solely based on the memory at time t**. For example, if we wanted to predict the next word in a sentene it would be a vector of probabilities across our vocabulary. $o_t = softmax(Vs_t)$

Notes:

* a RNN **shares the same parameters ($U, V, W$) across all steps. This means we are performing the same task at each step, just with different inputs
* For some tasks, we should not output at each time step, and the only thing to output might be connecting to the fully connected layer.

### Backpropagation 

Compared backpropagation of neuron network, the bp for training rnns is a little twist. Because the parameters are shared by all time steps in the network, the gradient at each output depends **not only on the calculations of the current time step, but also the previous time steps**. For example, in order to calculate the gradient at t = 4 we would need to backpropagate 3 steps and **sum up the gradients**. This is called **backpropagation through time (BPTT)**

#### Define loss

$$
E(y, \hat y) = \sum_t E_t(y_t, \hat y_t) = -\sum_t y_t log \hat y_t
$$

#### Derivative of softmax

$$
y_t = {e^{z_t}\over {\sum_k e^{z_t}}}
$$

if j = i, this is a sigmoid function
$$
{\partial y_t\over \partial {z_t}} = y_t (1-y_t)
$$
if j $\neq$ i
$$
{\partial y_t \over \partial z_i}  = {\partial \over \partial z_i} ({e^{z_t}\over {\sum_k e^{z_k}}})
= {{0 \cdot \sum _k e^{z_k} - e^{z_t} \cdot e^{z_i}}\over {(\sum_k e^{z_k})^2}} 
= -{e^{z_t}\over {\sum_k e^{z_k}}}\cdot {e^{z_i}\over {\sum_k e^{z_k}}}
= -y_ty_i
$$

#### Activation function and classifier

$$
s_t = tanh(Ux_t + Ws_{t-1})
$$

$$
y_t = softmax(Vs_t)
$$

#### The gradient of V

$$
\begin{align} {\partial E_t \over \partial V} & = {\partial E_t \over \partial  \hat y_t} {\partial \hat y_t \over \partial  V} = {\partial E_t \over \partial  \hat y_t} {\partial \hat y_t \over \partial \hat z_t} {\partial \hat z_t \over \partial  V} \\
& = [-{y_t\over {\hat y_t}}\cdot \hat y_t(1-\hat y_t) - \sum_{k \neq t} {y_k\over {\hat y_k}}\cdot (-\hat {y_t}\hat {y_k})]{\partial \hat z_t \over \partial  V} \\
& =  (-y_t + y_t\hat y_t + \sum_{k \neq t} y_k \hat y_t){\partial \hat z_t \over \partial  V}\\
& = (\hat y_t - y_t) \bullet s_t 
\end{align}
$$

#### The gradient of W

$$
{\partial E_t \over \partial W} = {\partial E_t \over \partial  \hat y_t} {\partial \hat y_t \over \partial  W} 
= {\partial E_t \over \partial  \hat y_t} {\partial \hat y_t \over \partial  z_t}  {\partial z_t \over \partial  s_t}{\partial  s_t \over \partial  W} 
= \sum_{k=0}^t{\partial E_t \over \partial  \hat y_t}{\partial \hat y_t \over \partial  z_t}  {\partial z_t \over \partial  s_t}{\partial s_t \over \partial s_k}{\partial  s_k \over \partial  W}
$$

$s_t = tanh(Ux_t + Ws_{t-1})$, depends on $W, s_{t-1}, s_{t-2}, ...s_0$ 

$\partial s_t \over \partial s_k$ is a chain rule itself, this leads to a **Jacobian matrix **whose elements are all the pointwise derivatives

Hence, finally
$$
{\partial E_t \over \partial W} = {\partial E_t \over \partial  \hat y_t} {\partial \hat y_t \over \partial  W} 
= {\partial E_t \over \partial  \hat y_t} {\partial \hat y_t \over \partial s_t} {\partial s_t \over \partial  W} 
= \sum_{k=0}^t{\partial E_t \over \partial  \hat y_t} {\partial \hat y_t \over \partial s_t}{\partial s_t \over \partial s_k}{\partial  s_k \over \partial  W}
= \sum_{k=0}^t{\partial E_t \over \partial  \hat y_t} {\partial \hat y_t \over \partial s_t}(\prod_{j=k+1}^t {\partial s_j \over \partial s_{j-1}}){\partial  s_k \over \partial  W}
$$

#### The gradient of U	

It is simlar with the gradient of W due to the activation function $s_t$

### LSTM

![LSTM](/asset/LSTM.png)

#### The core idea behind LSTMs

The keys to LSTMs is the **cell state**, which is kind of like a conveyor belt. It runs straight down the entire chain, and **carefully regulated by gates to remove or add information**

![](/asset/Cell-State.png)



#### The first step

The first step in LSTM is to decide **what information we're going to throw away from the cell state**. This decision is made by a sigmoid layer called the "forget gate layer", and the output is between 0 and 1. If the output of this layer is 0, which means "completely get rid of the input", while a 1 means "completely keep the input"

![](/asset/LSTM-Step1.png)

#### The second step

The next step is to decide **what new information we're going to store in the cell state**.

1. a sigmoid layer called the "input gate layer" decides which values we'll update.
2. a tanh layer creates a vector of new candidate values, $\hat C$, that could be added to the state. **This provides more information of inputs, instead of a value between 0 and 1**
3. Update old cell state $C_{t-1}$ into the new cell state $C_t$

![](/asset/LSTM-Step2-1.png)

![](/asset/LSTM-Step2-2.png)

#### The third step

Finally, we need to decide **what we're going to output**

1. A sigmoid layer to decide what parts of the cell state to output
2. A tanh layer which inputs are the cell state to provide information from cell state 

![](/asset/LSTM-Step3.png)

### GRU

Major improvements from LSTMs

* Combines the forget and input gates into a single "update gate"
* Merges cell state and hidden state 

Interpretations

* $r_t$ is a reset gate, which decides how to combine the new input with the previous memory 
* $z_t$ is a update gate, which defines how much of the previous memory to keep around, if $z_t$ is small, $h_t$ would be more close to $h_{t-1}$, and this means the updated information is small 

![](/asset/GRU.png)

### Reference

[RECURRENT NEURAL NETWORKS TUTORIAL, PART 1 – INTRODUCTION TO RNNS](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)

[RECURRENT NEURAL NETWORKS TUTORIAL, PART 3 – BACKPROPAGATION THROUGH TIME AND VANISHING GRADIENTS](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/)

[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

[Written Memories: Understanding, Deriving and Extending the LSTM](http://r2rt.com/written-memories-understanding-deriving-and-extending-the-lstm.html)

[Recurrent Neural Networks in Tensorflow I](http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html)

[Recurrent Neural Networks in Tensorflow II](http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html)
