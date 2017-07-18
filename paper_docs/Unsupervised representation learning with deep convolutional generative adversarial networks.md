# Unsupervised representation learning with deep convolutional generative adversarial networks

## Notes

GANs provide an attractive alternative to maximum likelihood techniques.

We found global average pooling increased model stability but hurt convergence speed.

## Approach and model architecture

Core to our approach is adopting and modifying three recently demonstrated chagnes to CNN architectures.

The **first** is the all convolutional net which replaces deterministic spatial pooling functions (such as maxpooling) with strided convolutions, allowing the network to learn its own spatial downsampling. We use this approach in our generator, allowing it to learn its own spatial upsampling, and discriminator.

**Second** is the trend towards eliminating fully connected layers on top of convolutional features. The first layer of the GAN, which takes a uniform noise distribution $Z$ as input, could be called fully connected as it is just a matrix multiplication, but the result is reshaped into a 4-dimensional tensor and used as the start of the convolutional stack. For the discriminator, the last convolution layer is flattened and then fed into a single sigmoid output.

**Third** is **BN** which stabilizes learning by normalizing the input to each unit to have zero mean and unit variance. This helps deal with training problems that arise due to **poor initialization** and **helps gradient flow in deeper models**. This proved critical to get deep generators to begin learning, preventing the generator from collapsing all samples to a single point which is a common failure mode observed in GANs.  **There is no BN on the generator output layer and the discriminator input layer**

The ReLU activation is used in the generator with the exception of the output layer which uses the Tanh function. We observed that using a bounded activation allowed the model to learn more quickly to **saturate and cover the color space of the training dsitribution**. Within the distriminator we found the **leaky rectified activation** to work well

Overrall,

* Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator)
* Use batchnorm in both the generator and the discriminator
* Remove fully connected hidden layers for deeper architectures
* Use ReLU activation in generator for all layers except for the output, which uses Tanh
* Use LeakyReLU activation in the discriminator for all layers.