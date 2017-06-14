# Sentiment Classification

`open` built-in function: 	return a `file object` with specific mode, such as 'r' or 'w'. You can use `io.IOBase.readlines()` to get a **list** of lines from the stream, or use `for line in file:...` for interation

`map` is a built-in function: return an **iterator** that applies function to every item of iterable.

```python
g = open('reviews.txt', 'r')
reviews = list(map(lambda x: x[:-1], g.readlines()))
```

