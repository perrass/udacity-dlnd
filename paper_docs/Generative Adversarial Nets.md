# Generative Adversarial Nets

## Intro

**Adversarial Process**: We simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training proccedure for G is to maximize the probability of D making a mistake.

**Adversarial nets**: The generative model generates samples by passing random noise through a multiplayer perceptron, and the discriminative model is also a multilayer perceptron. This kind of net is adversarial nets. We can **train both models using only the highly successful backpropagation and dropout algorithms, and sample from the generative model using only forward propagation**

## Adversarial nets

To learn the generator's distribution $p_g$ over data $x$, we define a prior on input noise variable $p_z(z)$, then represent a mapping to data space as $G(z; \theta_g)$. where $G$ is a **differentiable** function represented by a multilayer perceptron with parameter $\theta_g$. We also define a second multilayer perceptron $D(x;\theta_d)$ that outputs a single scalar. $D(x)$ represents the probability that $x$ came from the data rather than $p_g$. We train $D$ to maximize the probability of assigning the correct label to both training examples and samples from $G$. We simultaneously train $G$ to minimize $log(1-D(G(z)))$ (PS: 同时做两个最优化，D的优化包括训练样本和模拟样本，G的优化是特定的函数)

In other words, $D$ and $G$ play the following two-player minimax game with value function $V(G, D)$:
$$
min_Gmax_DV(D,G) = E_{x\sim p_{data}(x)}[logD(x)] + E_{z\sim p_z(z)}[log(1-D(G(z)))]
$$
When the discriminiator is unable to differentiate between the two distributions, $D(x) = \frac 1 2$

## Theoretical Results

The stochastic gradient of the discriminator is
$$
\nabla_{\theta_d} \frac 1 m \sum^m_{i=1}[logD(x^i) +log(1-D(G(z^i)))]
$$
That of the generator is 
$$
\nabla_{\theta_g} \frac 1 m \sum^m_{i=1} log(1-D(G(z^i)))
$$
**Proposition 1**

For $G$ fixed, the optimal discriminator $D$ is 
$$
D^*_G(x) = \frac {p_{data}(x)} {p_{data}(x)+p_g(x)}
$$
**Proposition 2**

If $G$ and $D$ have enough capacity, and at each step of Algorithm1, the discriminator is allowed to reach its optimum given G, and $p_g$ is updated so as to improve the criterion
$$
E_{x\sim p_{data}(x)}[logD^*_G(x)] + E_{z\sim p_g}[log(1-D^*_G(x)]
$$
then $p_g$ converges to $p_{data}$

