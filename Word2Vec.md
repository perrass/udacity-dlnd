# Word2Vec

**Word embedding**

Word embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing where words or phrases from the vocabulary are mapped to vectors of real numbers.

The goal is actually just to learn the **weights of the hidden layer** we'll see that these weights are actually the "word vectors" that we're trying to learn

Given a specific word in the middle of a sentence, look at the words nearby and pick one at random. The network is going to tell us the **probability for every word in our vocabulary of being the "nearby word" that we chose**. "Nearby" is a "window size" parameter to the algorithms

### Step1: transform a single word to one-hot vector

There is no activation function on the hidden layer neurons, but the output neurons use softmax

We can set word vectors with 300 features, so the hidden layer is going to be represented by a weight matrix with 10000 rows and 300 columns

And for one input (1 * 10000) is multiplies by a 10000 * 300 matrx, return a 1 * 300 vector

![](/asset/word2vec.png)

This means that the hidden layer of this model is really just operating as a **lookup table**. The output of the hidden layer is just the **word vector** for the input word (**learn features automatically**)

The 1 * 300 word vector is the input of the output layer, and the output layer is a softmax regression classifier (300 * 1). The return is **the proability that if you randomly pick a word nearby "input", that it is "output"

**However, the purpose of word2vec is to train the hidden layer weight matrix to find efficient representations for our words. We can discard the softmax layer because we don't really care about making predictions with this network. We just want the embedding matrix so we can use it in other networks we build from the dataset**

## Negative sampling

The number of weights of former neural network is 3 million. 

Improvment of word2vec

1. Treating common word pairs on phrases as single "words" in their model.
2. Subsampling frequent words to decrease the number of training examples
3. Modifying the optimization object with a technique they called "Negative sampling", which causes each training sample to update only a small percentage of the model's weights

#### Subsampling

To delete the occurance of the redundent words, like "the"
$$
P(w_i) = (\sqrt{z(w_i)\over 0.001} + 1)\cdot{0.001\over z(w_i)}
$$

* if $z(w_i) <= 0.0026$, all words (single word) should be kept
* if $z(w_i) = 0.00746$, half of words should be kept
* if $z(w_i) = 1$, 3.3% of word would be kept, but it is impossible

#### Negative sampling

Only randomly select a small number of "negative" words to update the weights for. (In this context, a "negative" word is one for which we want the network to output a 0 for). We will also still update the weights for out 'positive' words.

Recall that the output layer of our model has a weight matrix that's 300 * 10000. So we will just be updating the weights for our positive word, plus the weights for 5 other words that we want to output 0. That's a total of 6 output neurons, and 1800 weight values total. That's only 0.06% of the 3M weights in the output layer.

The probability for a selecting a word is just it's weight divided by the sum of weights for all words
$$
P(w_i) = {f(w_i)^{3/4}\over{\sum_{j=0}^n(f(w_j)^{3/4})}}
$$


### References

[Word2Vec Tutorial Part 1 - Word2vec Tutorial](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

[Word2Vec Tutorial Part 2 - Negative Sampling](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)

[Word2Vec (Part 1): NLP With Deep Learning with Tensorflow (Skip-gram)](http://www.thushv.com/natural_language_processing/word2vec-part-1-nlp-with-deep-learning-with-tensorflow-skip-gram/)

[Word2Vec (Part 2): NLP With Deep Learning with Tensorflow (CBOW)](http://www.thushv.com/natural_language_processing/word2vec-part-2-nlp-with-deep-learning-with-tensorflow-cbow/)