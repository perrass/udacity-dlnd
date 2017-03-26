## Intro to Tensorflow

In TensorFlow, data isn't stored as integers, floats, or strings. These values are encapsulated in an object called a **tensor**

```Python
hello_constant = tf.constant("Hello World!")
```

TensorFlow API is built around the idea of **computational graph**, and a TensorFlow Session is an environment for running a graph.

```Python
with tf.Session() as sess:
    output = sess.run(hello_constant)
```

### Input

#### `tf.placeholder()`

The $X$ of dataset cannot be put into TensoFlow, ``tf.placeholder()`` returns a tensor that gets its value from data passed to the `tf.session.run()` function. And `feed_dict` parameter in`tf.seesion.run()` set the placeholder tensor.

```Python
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: "Test String", y: 123, z: 45.67})
```

### Linear functions

#### `tf.Variable`

The `tf.Variable` **class** creates a tensor with an initial value that can be modified. Normally, `tf.global_variables_initializer()` is used to initialize the state of all the Variable tensor, and it returns an **operation to initialize all TensorFlow variables from the graph**

### Softmax



### One-Hot Encoding



### Cross Entropy



### Normalization (Why)



### Stochastic Gradient Descent

* Inputs
* Initial weights
* Momentum Using average of current direction instead of the random gradient
* Learning rate decay Decreasing Learning rate with steps

#### Hyper-parameters

* Initial learning rate **small first**
* Learning rate decay
* Momentum
* Batch size
* Weigh Initialization

#### ADAGRAD

* Batch size
* Weigh Initialization

### Mini-batch



### Epochs

An **epoch** is a single forward and backward pass of the whole dataset