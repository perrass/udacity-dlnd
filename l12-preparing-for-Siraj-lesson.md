## Preparing for Siraj's Lesson

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

[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

### LSTM

[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)



