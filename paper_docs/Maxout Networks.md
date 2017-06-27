# Maxout Networks

## Intro

We argue that rather than using dropout as a slight performance enhancement applied to arbitrary models, the best performance may be obtained by directly designing a model that enhances dropout's abilities as a model averaging technique.

**Maxout has benefical characteristics both for optimization and model averaging with dropout**

## Review of dropout

**Dropout training is similar to bagging, where many different models are trained on different subsets of the data**. Dropout training differs from bagging in that each model is trained for only one step and all of the models share parameters.

## Description of maxout

The maxout model uses a new type of **activation function: the maxout unit**. A maxout hidden layer implements the function
$$
h_i(x) = max_{j\in[1,k]}z_{ij}
$$
where
$$
z_{ij} = x^TW_{...ij} + b_{ij}
$$
For basic forward neural networks, the formular for activation function is $z = W*X+b$, and then $out = f(z)$

![](/assets/maxout_1.png)

If we add a **maxout hidden layer** and set $k=5$, the formular is changed to 
$$
z_1 = w_1*x + b_1\\
z_2 = w_2*x + b_2 \\
z_3 = w_3*x + b_3\\
z_4 = w_4*x + b_4\\
z_5 = w_5*x + b_5\\
out = max(z_1, z_2, z_3, z_4, z_5)
$$
![](/assets/maxout_2.png)

A single maxout unit can be interpreted as **making a piecewise linear approximation to an arbitrary convec function** (PS: 可以利用Max的分段函数性质，逼近处任何凸的激活函数). Maxout networks learn not just the relationship between hidden units, but also the activation function of each hidden unit.