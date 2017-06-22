# Dropout: A Simple Way to Prevent Neural Networks from Overfitting

## Intro

The key idea of dropout is to randomly drop units from the neural network during training. **This prevents units from co-adapting too much**. This means we prevent a special set of neurons to learning a specific pattern from the training set

PS: **Why does the specific pattern concern?**

* The specific pattern might be the **result of sampling noise**.
* The specific pattern is produced by **complex co-adaptations**, this kind of complex co-adapations is not fit well in other cases.

> In a standard neural network, the derivative received by each parameter tells it **how it should change so the final loss function is reduced**, given what all other units are doing. Therefore, units may change in a way that they six up the mistakes of the other units. This may lead to **complex co-adaptations**. This in turn leads to overfitting because these co-adaptations do not generalize to unseen data

## Motivation

> A motivation for dropout comes from a theory of the **role of sex in evolution**. Sexual reproduction involves taking half the genes of one parent and half of the other, adding a very small amount of random mutation, and combining them to produce an offspring.
>
> Sexual reprodcution is likely to break up these co-adapted sets fo genes, especially if these sets are large and intuitively.

## Related Work

Dropout can be interpreted as a way of **regularizing a neural network by adding noise to its hidden units**. Hence dropout can be seen as a **stochastic regularization** technique, and it is natural to consider its **deterministic** counterpart which is obtained by **marginalizing out the noise** (排除/使边缘化噪音)

## Model Description

![](/assets/dropout_network.png)

$r_i^{(l)}$ is with **Bernoulli distribution**, usually set $p=0.5$

## Learning Dropout Nets

### Backpropagation

One particular form of regularization was found to be especially useful for dropout - **constraining the *norm* of the incoming weight vector at each hidden unit to be upper bounded by a fixed constant $c$**. In other words, if $\mathbf w$ represents the vector of weights incident on any hidden unit, the neural network was optimized under the constraint $||\mathbf w||_2 \le c$. This constraint was imposed during optimization by **projecting $\mathbf w$ onto the surface of a ball of radius $c$, whenever $\mathbf w$ went out of it**. This is also called **max-norm regularization since it implies that the maximum value that the norm of any weight can take is $c$**

---

$\color{red}{调参经验!!!}$

Although dropout alone gives significant improvements, using dropout along with maxnorm regularization, **large decaying learning rates** and **high momentum** provides a significant boost over just using dropout.  

A possible justification is that constraining weight vectors to lie inside a ball of fixed radius makes it possible to **use a huge learning rate** without the possibility of weights blowing up.

**Why?**

**Dropout introduces a significant amount of noise in the gradients compared to standard stochastic gradient descent**. Therefore, a lot of gradients tend to cancel each other. In order to make up for this, **a dropout net should typically use 10-100 times the learning rate that was optimal for a standard neural net. Or setting the momentum to 0.95 to 0.99, rather than 0.9**

**Then?**

If the learning rate and the momentum is large, **the weights to network would grow very large**. Hence, we use **max-norm regularization**. Typically, the value of range is from 3 to 4.

---

## Salient Features

### Effect on Features

**Why does dropout work ?**

> We hypothesize that for each hidden unit, dropout prevents co-adaptions by **making the presence of other hidden units unreliable**. Therefore, a hidden unit **cannot rely on other specific units to correct its mistakes**. It must perform well in a wide variety of different contexts provided by the other hidden units.

PS: 意思是说，不适用dropout，神经元会联合起来学习特定的特征，但是如果使用dropout，被激活的神经元是不确定的，因此无法利用特定的一组神经元来学习特定特征，只能用被给定的特征来学习。就像一个不很会做饭的人买菜，如果是去超市，大概率是买他会做的材料。但如果给出来的材料不确定，那么他会被迫学习新的菜谱来适应材料。

![](/assets/dropout_co_adaptation.png)

### Effect on Dropout Rate

An interesting setting can be used is **holding the $pn$ to be constant where n is the number of of hidden units in any particular layer**. In the paper, the optimal $p$ is 0.6 in this case, rather than 0.5.

For real-valued inputs, a typical value is 0.8

For hidden layers, **the choice of $p$ is coupled with the choice of number of hidden units $n$**. 

*  Smaller $p$ requires big $n$ which slows down the training and leads to underfitting
* Large $p$ may not produce enough dropout to prevent overfitting



