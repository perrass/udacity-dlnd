# RNN

### Key_words

* n_seqs: number of sequences per batch
* n_steps: number of sequence steps per batch, 移动windows的步长
* batch_size: number of chars per batch
* n_batches: number of batches of dataset
* lstm_size
* num_layers

### Input

* Starting sequence: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
* 确定n_seqs, n_steps得到batch_size和n_batches，去除末尾不足一个batch_size的文字
* 将文本转化为一个n_seqs行，n_steps列的矩阵
  * **n_seqs有时也被称为batch_size的原因是, len(batch) = n_seqs，也是batch_size**
* 得到目标矩阵 (y)，向右走一各字节，比如得到[2, 3, 4, ..., 13]，然后再按相同的方法的到和x相同格式的y
* 这时数据的形状是n_seqs * (n_steps * batch_size)的矩阵

#### Get batches

```python
def get_batches(arr, n_seqs, n_steps):
    batch_size = n_seqs * n_steps
    n_batches = len(arr) // batch_size
    arr = arr[: n_batches * batch_size]
    arr = arr.reshape((n_seqs, -1))
    
    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:n+n_steps]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y
```

#### Build inputs

```python
def build_inputs(batch_size, num_steps):
    inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
    targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, keep_prob
```

  ### Build LSTM

```python
def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)  # Would change in May
    initial_state = cell.zero_state(batch_size, tf.float32)
    
    return cell, initial_state
```

### Output

If our input has batch size N, number of steps M, and the hidden layer has L hidden units, then, **output of each LSTM cell has size L**, we have **M outputs for each sequence step**, and we have **N sequence**. So the total size is $N \times M \times L$ 

#### LSTM -> DNN (fully connected)

需要做一层转换，当然卷积理论上可以，$N\times M\times L \to (N\times M) \times L$

```python
def build_output(lstm_output, in_size, out_size):
    seq_output = tf.concat(lstm_output, axis=1)
    x = tf.reshape(seq_output, [-1, in_size])
    #在LSTM层已经创建了weights和biases的variables，当我们再全连接生成weights和biases变量时会出现冲突，因此需要声明变量范围
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))
    logits = tf.matmul(x, softmax_w) + softmax_b
    out = tf.nn.softmax(logits, name='predictions')
    return out, logits
```

### Training loss

我们得到logits并利用已有的targets，来计算$softmax~cross~entropy~loss$，为了和logits的shape对齐，需要将y变化为$(M\times N) \times C$. Logits的shape是因为LSTM对接的全连接层，这个全连接层有C个units.

```python
def build_loss(logits, targets, lstm_size, num_classes):
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    return loss
```

### Optimizer

因为RNN的循环过程，因此就算我们用LSTM解决了梯度消失的问题，依然会有**梯度膨胀**的问题

> 梯度膨胀问题
>
> 带有Recurrent的隐藏层的输出是 $h_t = f(W_{hh}h_{t-1} +Ｗ_{xh}x_t) = f(W_{hh}f(W_{h_{t-1}h_{t-1}}h_{t-2} + W_{xh_{t-1}}x_{t-1})+Ｗ_{xh}x_t)=...$
>
> 因此，会得到$W_{hh}$的连乘，因此如果W的值过大，就会使连乘的结果过大，影响预测
>
> 因此，我们会限定一个连乘的上限，这个方法叫做Gradient Clip
>
> 但是为什么LSTM没有解决梯度膨胀问题还没搞明白！！！

```python
def build_optimizer(loss, learning_rate, grad_clip):
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_gloabl_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    return optimizer
```

### Build network

```python
class CharRNN:
    def __init__(self, num_classes, batch_size=64, num_steps=50, lstm_size=128,
                 num_layers=2, learning_rate=0.001, grad_cli5, sampling=False):
        if sampling:
            batch_size, num_steps = 1, 1
        tf.reset_default_graph()
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)
        x_one_hot = tf.one_hot(self.inputs, num_classes)
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)
```

