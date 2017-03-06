## RNN and LSTM

### Bag of Words

A bag (dict) of distinct word and corresponding count. However, it loses the information related to the **order of words**, and then loses the information for understanding.

```python
from collections import Counter

def bag_of_words(text):
    return Counter(text.split(' '))
```

### Word embedding

A tech to map words or phrases to vectors

### Word2vec

Word2vec is a neural network model that trains on text to create embeddings. And there exists two architectures to produce a distributed representation of words: **continuous bag-of-words (CBOW)** or continuous **skip-gram**

+ CBOW: the model predicts the current word from a window of surronding context words. The order of context words does not influence prediction
+ skip-gram: the model uses the current word the predict the surronding window of context words. This weighs nearby context words more heavily than more distant context words

[The details of CBOW and skip-gram](http://blog.csdn.net/u014595019/article/details/51884529)

### RNN

[Anyone Can Learn To Code an LSTM-RNN in Python (Part 1: RNN)](http://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/)

```Python
import copy, numpy as np
np.random.seed(0)

# choose sigmoid as activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# get the derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)

# a map to convert int to bit str
int2binary = {}

# max length of binary number
binary_dim = 8

largest_number = pow(2, binary_dim)

# map an int to its binary represetation
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

# learning rate
alpha = 0.1
input_dim = 2

# The size of hidden layer that will be storing our carry bit
hidden_dim = 16
output_dim = 1

# initialize neural network weights
synapse_0 = 2*np.random.random((input_dim, hidden_dim)) - 1  # 2 * 16
synapse_1 = 2*np.random.random((hidden_dim, output_dim)) - 1  # 16 * 1

# The matrix of weights that connects the hidden layer in the previous time-step to the hidden layer in the current timstep. It also connects the hidden layer in the current timestep to the hidden layer in the next timestep
synapse_h = 2*np.random.random((hidden_dim, hidden_dim)) - 1

# The updates of weights
synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# training logic
for j in range(100000):

    # generate a simple addition problem (a + b = c)
    # the upper bound is largest_number / 2
    a_int = np.random.randint(largest_number/2)
    a = int2binary[a_int]

    b_int = np.random.randint(largest_number/2)
    b = int2binary[b_int]

    c_int = a_int + b_int
    c = int2binary[c_int]

    d = np.zeros_like(c)

    overall_error = 0

    # keep track of the layer2 derivatives and layer1 values
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))
    
    ### PASS FORWARD

    for position in range(binary_dim):
		
        # layer_0 inputs
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T
		
        # propagate from the input to the hidden layer
        # propagate from the previous hidden layer to the current hidden layer
        # layer_1_values[-1] indicates the prev hidden layer
        # sum these two vectors and pass through the sigmoid
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

        # propagates the hidden layer to the output to make a prediction
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        # error
        layer_2_error = y - layer_2
        
        # store the derivative in a list, holding the derivative at each timestep 
        layer_2_deltas.append((layer_2_error) * sigmoid_output_to_derivative(layer_2))
        
        # calculate the sum of the absolute errors
        overall_error += np.abs(layer_2_error[0])
	
    	# round the output and stores in d
        d[binary_dim - position - 1] = np.round(layer_2[0][0])

        # Copies the layer_1 value into an array so that at the next time step we can apply the hidden layer at the current one
        layer_1_values.append(copy.deepcopy(layer_1))

    future_layer_1_delta = np.zeros(hidden_dim)
    
    ### BACK PROPAGATION

    for position in range(binary_dim):
		
        # Index the input data
        X = np.array([[a[position], b[position]]])
        
        # select the current hidden layer from the list
        layer_1 = layer_1_values[-position-1]
        
        # select the previous hidden layer from the list
        prev_layer_1 = layer_1_values[-position-2]

        # select the current output error from the list
        layer_2_delta = layer_2_deltas[-position-1]
        
        # compute the current hidden layer error given error at
        # the hidden layer from the future and
        # error at the current output layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) +
            layer_2_delta.dot(synapse_1.T)) * \
            sigmoid_output_to_derivative(layer_1)

        # We don't actually update our weight matrices until after we've fully backpropagetd everything
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)

        # set the future_layer_1_delta to the current one due to the back propagetion
        future_layer_1_delta = layer_1_delta

    # update weights
    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0

    if (j % 1000 == 0):
        print ("Error:" + str(overall_error))
        print ("Pred:" + str(d))
        print ("True:" + str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print (str(a_int) + " + " + str(b_int) + " = " + str(out))
        print ("-------------")

```

**Note1: How the the size of hidden layer affects the speed of convergence.**

* Do larger hidden dimensions make things train faster or slower?
* More iterations or fewer?

**Note2: numpy notes**

* `np.zeros` and `np.zeros_like`

**Note3: How to combine the information from the previous hidden layer and the  input?**

After each has been propagated through its various matrices (**interpretations**), we sum the information

**Note4: What is delta?**

**Note5: What is the distinction between a common neural network and a RNN?**

**Note6: Why don't we actually update our weight matrices until after we've fully back-propagated everything?**

[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

### LSTM

[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)



