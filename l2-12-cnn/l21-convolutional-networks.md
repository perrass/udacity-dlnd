## Convolutional Networks

### Concepts

* **Statistical Invariance**, if the two imputs contain the same kind of information, we can set the same weights and train the weights jointly for those input. This is **weight sharing**. E.g., the location of a same word in a document, or the color of a same image.
* Patch (Kernel), the window
* Depth, the number of feature map, or filter depth, the amount of filters in a convolutional layer. If we have a depth of k, we connect each patch of pixels to k neurons in the next layer, also a hyper-parameter
* Feature Map, one stack in depth, a map in one layer to capture one feature. E.g. if one patch might include some white teeth, some blonde whiskers, and a part of a red tongue. The layer at least should have three feature maps for teeth, whisker and tongue, and the filter depth is 3.
* Stride, the number of pixels when shifting the window, a hyper-parameter
* Valid Padding, the size of output is smaller than the input
* Same Padding, the size of output is the same as the input, but the patch or the window must **go off the edge**

#### Example

For a image with $28 \times 28$ pixels, a patch is $3 \times 3$, input's depth is 3, and output's depth is 8. Given padding and stride, what are the width, height and depth?

| Padding | Stride | Width | Height | Depth |
| ------- | ------ | ----- | ------ | ----- |
| Same    | 1      | 28    | 28     | 8     |
| Valid   | 1      | 26    | 26     | 8     |
| Valid   | 2      | 13    | 13     | 8     |



### Convolutional Networks

A image has width, height, and depth (RGB). Assigning a window to pass the image, like passing a number in a $2 * 2$ matrix, then we get a output with new width, height, and depth (not just RGB, but K of channels). **This operation is called  convolution**

The change of the matrix in a process of convolution is 
$$
256 \times 256 \times RGB \to 128 \times 128 \times 16 \to 64 \times 64 \times 64 \to 32 \times 32 \times256
$$
One important thing is **CNN groups together adjacent pixels and treating them as a collective**

#### Parameter sharing

The weights, $\mathbf w$, are shared across patches for a given layer in a CNN to detect the cat regardless of where in the image it is located. Practically, The weights and biases we learn for a given output layer are shared across all patches in a given input layer.

**Benefit**

1. Deliminate the influence of locatione
2. Get a smaller and more scalable model, because we would not have to learn new parameters for every single patch and hidden layer neuron pair

#### How many neurons in each layer in CNN

Given:

* our input layer has a width of $W$ and a height of $H$
* our convolutional layer has a filter size $F$
* a stride of $S$
* a padding of $P$
* and the number of filters $K$

Get:

* the width of the next layer: $W_{out} = (W - F+ 2P)/ S \ + 1$
* the height of the next layer: $H_{out} = (H -F +2P)/S \ + 1$
* the output depth: $D_{out} = K$
* the output volumn: $W_{out} * H_{out} * D_{out}$ 

E.g:

* An input of shape $32 \times 32 \times 3$
* 20 filters of shape $8 \times 8 \times 3$
* A stride of 2 for both width and height
* Padding of size 1
* Output size is $14 \times 14 \times 20$

---

$\color{red}{具体的计算过程!!!}$

假设$12\times12$的图片，用$3\times3$的核，步长是1，补全的0，卷积的基本方法是求和并没有激活函数。通过卷积层后，会变成$10\times10$的隐藏层，隐藏层的深度是卷积核的深度

![](/assets/conv_3_3_1_0.png)

用$3\times3$的核，步长是2，补全是0，会变成$5\times5$的隐藏层。**注意: 如果不用padding，最右边一列和最下边一行会失去，但可以学到更有规律的特征**

![](/assets/conv_3_3_2_0.png)

用$3\times3$的核，步长是2，对右下做补全，会变成$6\times6$的隐藏层。**注意：如果加入padding，会得到更多的特征，但学习到的规律可能会变弱**

![](/assets/conv_3_3_2_right_down.png)

---

In Tensorflow,

```python
input = tf.placeholder(tf.float32, (None, 32, 32, 3))
filter_weights = tf.Variable(tf.truncated_normal((8, 8, 3, 20)))
filter_bias = tf.Variable(tf.zeros(20))
strides = [1, 2, 2, 1]
padding = 'SAME'
conv = tf.nn.conv2d(input, filter_weights, strides, padding) + filter_bias
```

If the nn is fully connected without sharing weights, the number of parameters is 
$$
(8 * 8 * 3 + 1) * (14 * 14 * 20) = 756560
$$
If sharing weights is used in nn, the number of parameters is 
$$
(8 * 8 * 3 + 1) * 20 = 3860
$$
