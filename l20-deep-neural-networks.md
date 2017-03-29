## Deep Neural Networks

### ReLUs

```python
hidden_layer = tf.add(tf.matmul(features, hidden_weights), hidden_biases)
hidden_layer = tf.nn.relu(hidden_layer)
output = tf.add(tf.matmul(hidden_layer, output_weights), output_biases)
```

### Train and save a model



### Fine tune

When a model has already trained and saved, you can also **finetune** it

### Over-fitting

The ways to prevent over-fitting

1. Early Termination means stopping to train, as soon as  we stop improving in validation set
2. Regularization
   1. Norm (L1, L2, Other)
   2. Dropout, for each sample passing each activations, setting half of them to zero. This means, you destroy half of the data completely and randomly, and then **double** the remain value passing the activations. **This ensures that, in evaluation, there is no information loss** 