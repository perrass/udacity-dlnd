## Mini-Flow

### Intro

In Google TensorFlow Whitepaper

>  In TensorFlow graph, each node has zero or more inputs and zero or more outputs, and represents an instantiation of an **operation**. Values that flow along normal edges in the graph are **tensors**, arbitrary dimensionality **arrays** where the underlying element **type is specified or inferred at graph-construction time**  

Hence, using this idea, we can build a MiniFlow with the same basic architecture

```python
class None(object):
    def __init__(self, inbound_nodes=[]):
        # Nodes from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Nodes to which this Node passes values
        self.outbound_nodes = []
        # For each inbound Node here, add this Node as an outbound Node to _that_ Node
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
        
        # A calculated value
        self.value = None
        
    def forward(self):
        raise NotImplemented
        
    def passward(self):
        raise NotImplemented
```

1. 每个节点都会有一些输入，包括普通的Nodes和**多卡或分布式时会存在的Receive**
2. 每个节点都会有一些输出，**该节点输入的输出一定含有该节点**
3. 每个节点都会传递出一个值
4. 每个节点都会有向前和向后传播的方法，用于神经网络的计算

### 图表中会出现的类型

`Node`定义了每个节点都具有的基本属性，但只有`Node`的**特殊子类会出现在计算图中**

#### Input

```python
class Input(Node):
    def __init__(self):
        # An Input node has no inbound nodes, so no need to pass anything to the Node instantiator
        Node.__init__(self)
        
    def forward(self, value=None):
        # Overwrite the value if one is passed in
        if value is not None:
            self.value = value
```

* `Input` 子类实际上不计算任何内容，仅仅存储Value，比如数据特征或模型参数
* `Input`子类是**唯一一个**value可以做为参数传递到`forward`方法中的子类

---

#### Unpacking argument lists

\* and \*\* are **unpacking-argument-lists operators** ([Unpacking Argument Lists](https://docs.python.org/3/tutorial/controlflow.html#unpacking-argument-lists)) 

The reverse situation occurs when the arguments are already in a list or tuple but need to be unpacked for a function call requiring separate positional arguments. 

\* operator unpacks the arguments out of a list or tuple

```python
args = [3, 6]
list(range(*args)) # [3, 4, 5]
```

Dictionaries can deliver keyword arguments with the \*\* operator

```python
def parrot(voltage, state="a stiff", action="voom"):
    print("-- This parrot wouldn't", action, end=' ')
    print("if you put", voltage, "volts through it.", end=' ')
    print("E's", state, "!")
    
d = {"voltage": "four million", "state": "bleedin' demised", "action": "VOOM"}
parrot(**d)
# -- This parrot wouldn't VOOM if you put four million volts through it. E's bleedin' demised !
```

---

#### Add 

```python
class Add(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)
        
    def forward(self):
        value_array = [node.value for node in self.inbound_nodes]
        self.value = sum(value_array)
```

#### Linear

```python
class Linear(Node):
    # inputs, weights, bias are all numpy array
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])
        
    def forward(self):
        inputs = self.inbound_nodes[0].value
        weights = self.inbound_nodes[1].value
        bias = self.inbound_nodes[2].value
        self.value = np.dot(inputs, weights) + bias
```

#### Sigmoid

```python
class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])
        
    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))
    
    def forward(self):
        input_value = self.inbound_nodes[0].value
        self.value = self._sigmoid(input_value)
```

#### MSE

```python
class MSE(Node):
    def __init__(self, y, a):
        Node.__init__(self, [y, a])
        
    def forward(self):
        # Making both arrays (3, 1) insures the result is (3, 1) and does an elementwise subtraction as expected
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)
        m = self.inbound_nodes[0].value.shape[0]
        
        diff = y - a
        self.value = np.mean(np.sum(np.square(diff)))
```

**Using reshape to avoid possible matrix/vector broadcast errors**. For example, if we subtract an array of shape (3, ) from an array of shape (3, 1) we get an array of shape (3, 3) as the result when we want an array of shape (3, 1) instead.

If we set the second parameters of `reshape` to -1 means flatten, and to 1 means set the number of column to 1.

```python
w = np.array([[1, 2], [3, 4]])
w_flat = np.reshape(w, -1) # [1, 2, 3, 4]

w_lists = np.reshape(-1, 1) # [[1], [2], [3], [4]]
```

### 加入反向传播

#### Input

```python
class Input(Node):
    def __init__(self):
        # An Input node has no inbound nodes, so no need to pass anything to the Node instantiator
        Node.__init__(self)
        
    def forward(self, value=None):
        # Overwrite the value if one is passed in
        if value is not None:
            self.value = value
            
    def passward(self):
        # There is no input
        self.gradients = {self: 0}
        # Cycle through the outputs. The gradient will change depending on each output,
        # so the gradient are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node
            grad_cost = n.gradients[self]
            # Add all partial costs to this node
            self.gradients[self] += grad_cost * 1
```

#### Linear

```python
class Linear(Node):
    def __init__(self, X, W, b):
        # The base class (Node) constructor. Weights and bias
        # are treated like inbound nodes.
        Node.__init__(self, [X, W, b])

    def forward(self):
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value
        self.value = np.dot(X, W) + b

    def backward(self):
        """
        Calculates the gradient based on the output values.
        """
        # Initialize a partial for each of the inbound_nodes.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]
            # Set the partial of the loss with respect to this node's inputs.
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
            # Set the partial of the loss with respect to this node's weights.
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            # Set the partial of the loss with respect to this node's bias.
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)
```

#### Sigmoid

```python
class Sigmoid(Node):
    """
    Represents a node that performs the sigmoid activation function.
    """
    def __init__(self, node):
        # The base class constructor.
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def forward(self):
        input_value = self.inbound_nodes[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        # Initialize the gradients to 0.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]
            sigmoid = self.value
            self.gradients[self.inbound_nodes[0]] += sigmoid * (1 - sigmoid) * grad_cost
```

#### MSE

```python
class MSE(Node):
    def __init__(self, y, a):
        Node.__init__(self, [y, a])
        
    def forward(self):
        # Making both arrays (3, 1) insures the result is (3, 1) and does an elementwise subtraction as expected
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)
        m = self.inbound_nodes[0].value.shape[0]
        
        diff = y - a
        self.value = np.mean(np.sum(np.square(diff)))
        
    def backward(self):
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff
```

### 辅助函数

#### 拓扑排序

```python
def topological_sort(feed_dict):
    """
    Sort the nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Node and the value is the respective value feed to that Node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L
```

#### 向前向后传播

```Python

def forward_and_backward(graph):
    """
    Performs a forward pass and a backward pass through a list of sorted Nodes.

    Arguments:

        `graph`: The result of calling `topological_sort`.
    """
    # Forward pass
    for n in graph:
        n.forward()

    # Backward pass
    # see: https://docs.python.org/2.3/whatsnew/section-slices.html
    for n in graph[::-1]:
        n.backward()
```

#### SGD更新

```python
def sgd_update(trainables, learning_rate=1e-2):
    """
    Updates the value of each trainable with SGD.

    Arguments:

        `trainables`: A list of `Input` Nodes representing weights/biases.
        `learning_rate`: The learning rate.
    """
    # TODO: update all the `trainables` with SGD
    # You can access and assign the value of a trainable with `value` attribute.
    # Example:
    # for t in trainables:
    #   t.value = your implementation here
    for trainable in trainables:
        partial = trainable.gradients[trainable]
        trainable.value -= learning_rate * partial
```

