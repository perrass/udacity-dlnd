# Thrones2Vec

## Packages

### codecs

This module defines base classes for standard Python codecs (encoders and decoders) and provides access to the internal Python codec registry, which manages the codec and error handling lookup process. Most standard codecs are text encodings, which encode text to bytes.

```python
corpus_raw = u""
for book_filename in book_filenames:
    print("Reading '{0}'...".format(book_filename))
    with codecs.open(book_filename, "r", "utf-8") as book_file:
        corpus_raw += book_file.read()
    print("Corpus is now {0} characters long".format(len(corpus_raw)))
    print()
```

### glob

The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell, although results are returned in arbitrary order.

`glob.glob(pathname, *, recursive=False)`: pathname can be either absolute or relative, and can contain shell-style wildcards

### logging

[Basic Logging Tutorial](https://docs.python.org/3/howto/logging.html#logging-basic-tutorial)

[Advanced Logging Tutorial](https://docs.python.org/3/howto/logging.html#logging-advanced-tutorial)

### multiprocessing

[`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html?highlight=multiprocessing#module-multiprocessing) is a package that supports spawning processes using an API similar to the [`threading`](https://docs.python.org/3/library/threading.html#module-threading) module. The [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html?highlight=multiprocessing#module-multiprocessing) package offers both local and remote concurrency, effectively side-stepping the [Global Interpreter Lock](https://docs.python.org/3/glossary.html#term-global-interpreter-lock) by using subprocesses instead of threads. Due to this, the [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html?highlight=multiprocessing#module-multiprocessing) module allows the programmer to fully leverage multiple processors on a given machine. It runs on both Unix and Windows.

The [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html?highlight=multiprocessing#module-multiprocessing) module also introduces APIs which do not have analogs in the [`threading`](https://docs.python.org/3/library/threading.html#module-threading) module. A prime example of this is the [`Pool`](https://docs.python.org/3/library/multiprocessing.html?highlight=multiprocessing#multiprocessing.pool.Pool) object which offers a convenient means of parallelizing the execution of a function across multiple input values, distributing the input data across processes (data parallelism). The following example demonstrates the common practice of defining such functions in a module so that child processes can successfully import that module. This basic example of data parallelism using [`Pool`](https://docs.python.org/3/library/multiprocessing.html?highlight=multiprocessing#multiprocessing.pool.Pool),

### os

`os.path.join("trained", "thrones2vec.w2v")` to get the relative path `..\trained\thrones2vec.w2v`

### nltk

### 	genism.models.word2vec

### sklearn.manifold