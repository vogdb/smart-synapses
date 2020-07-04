### About

This project is a collection of different smart synapse mechanisms in ANN. The name "smart synapse"
I took from the article [Beyond Hebb: exclusive-OR and biological learning. Klemm 2000.](https://doi.org/10.1103/physrevlett.84.3013)
Thus the first script `klemm/original` in this collection is an implementation of that article.
I didn't use any neural networks frameworks. Everything is done in plain Numpy.
`klemm/experimental` is just a place to experiment on the original. It shows that an intuitive
plasticity rule works pretty bad. Errors get increased very fast.