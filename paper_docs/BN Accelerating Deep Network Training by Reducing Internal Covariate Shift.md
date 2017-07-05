# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

Batch normalization performs the normalization for each training mini-batch.

## Towards Reducing Internal Covariate Shift

We define **Internal Covariate Shift** as the change in the distribution of network activations due to the change in network parameters during training. (PS: 反向传播更新参数后，激活函数的结果的分布的变化). By **fixing the distribution of the layer inputs x as the training progresses**, we expect to improve the training speed, because if the network inputs are **whitened (linearly transformed to have zero means and unit variances, and decorrelated)** , the convergence of the network training would be faster. (PS: **This means BN should be followed by ReLU**).

## Normalization via Mini-Batch Statistics

For a layer with d-dimensional input $x=(x^{(1)}...x^{(d)})$, we will normalize each dimension:
$$
\hat x^{(k)} = \frac {x^{(k)} - E[x^{(k)}]} {\sqrt {Var[x^{(k)}]}}
$$
where the expection and variance are computed over the training data set.

Note that simply normalizing each input of a layer may change what the layer can represent. For instance, **normalizing the inputs of a sigmoid would constrain them to the linear regime of the nonlinearity** (PS: 可能是因为，输入被改为靠近0附近的数，近似线性). To address this, we make sure that the transformation inserted in the network can represent the identity transform. To accomplish this, we introduce, for each activation $x^{(k)}$, a pair of parameters $\gamma^{(k)}$, $\beta^{(k)}$, which scale and shift the normalized value:
$$
y^k = \gamma^k\hat x^k + \beta^k
$$
These parameters are learned along with the original model parameters, and restore the **representation power of the network**. $\gamma = \sqrt {Var[x^{(k)}]}$, $\beta = E(x^k)$

---

**Algorithms**

我们的到整个样本的$\gamma, \beta$，然后用对mini-batch做正则，然后再转换

Input: Values of x over a mini-batch: $B = {x_1,...m}$; Parameters to be learned: $\gamma, \beta$

Output: $\{y_i = BN_{\gamma, \beta}(x_i) \}$

$\mu_B \leftarrow \frac 1 m \sum^m_{i=1} x_i$ 				// mini-batch mean

$\sigma^2_B \leftarrow \frac 1 m \sum^m_{i=1} (x_i - \mu_B)^2$		       // mini-batch variance

$\hat x_i \leftarrow \frac {x_i - \mu_B} {\sqrt{\sigma^2_B + \epsilon}}$					      // normalize

$y_i \leftarrow \gamma\hat x_i + \beta (BN_{\gamma, \beta}(x_i))$ 		     // scale and shift

---

**Algorithms for training NN**

![](/assets/bn.png)

The transformations of BN (scaling and shifting) are different at **train** and **inference** step. The train step using **Algorithm 1**, and the inference step using **un-bias** statistics

---

### Batch-Normalized CNN

> We add the BN transform immediately before the nonlinearity, by normalizing $x = Wu+b$

For convolutional layers, we additionally want the normalization to obey the convoltional property, so that **different elements of the same feature map, at different locations, are normalized in the same way.** To achieve this, we jointly normalize all the activations in a mini-batch, over all locations. In Alg.1 we let $B$ be the set of all values in a feature map across both the elements of a mini-batch and spatial locations - so for a mini-batch of size m and feature maps of size $p\times q$, we use the effective mini-batch of size $m' = m \cdot pq$. And we learn **a pair of parameters $\gamma$, $\beta$ per feature map**.

### BN enables higher learning rates

If we add a scaler $a$
$$
BN(Wu) = BN((aW)u)
$$
Then
$$
\frac {\partial BN((aW)u)} {\partial u} = \frac {\partial BN(Wu)} {\partial u} \\
\frac {\partial BN((aW)u)} {\partial {(aW)}} = \frac 1 a \frac {\partial BN(Wu)} {\partial W}\\
\color{red}{WHY???}
$$

1. The scale does not affect $u$ (**the layer Jacobian**), nor, consequently, the gradient propagation
2. Larger weights lead to smaller gradients, and BN will stablize the parameter growth.

### Regularization

When training with Batch Normalization, a **training example is seen in conjunction with other examples in the mini-batch**, and the training network no longer producing **deterministic values** for a given training example. This increase the ability of generalization. Hence, BN produces kind of regularization. 