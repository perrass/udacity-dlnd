import tensorflow as tf


# Remove the pervious weights and bias
tf.reset_default_graph()

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

n_hidden_layer = 256 # layer number of features

# Two variables
weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer]), name='weights_h'),
    'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]), name='weights_o')
}
biases = {
    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer]), name='bias_h'),
    'out': tf.Variable(tf.random_normal([n_classes]), name='bias_o')
}

save_file = './l20-train-model.ckpt'
saver = tf.train.Saver()

with tf.Session() as sess:
    # Load the weights and bias - No Error
    saver.restore(sess, save_file)

print('Loaded Weights and Bias successfully.')
