Probabilistic Models in Informatics - Report
=============================================

This is an implementation in python of Dual Online learning algorithm for latent Dirichlet allocation (LDA).  

Dependencies
-------------

To use the code, you must install all of these packages:

- Linux OS (Stable on Ubuntu)
- Mac OS (Version 10.13 or later)
- Python version 2 or 3 (stable on version 2.7 and 3.6)
- Numpy >= 1.8

Learning Algorithm
------------------

Estimate a model by executing:

```sh
$ python run.py [train file] [setting file] [model folder] [test data folder]
```

| Environmental Variables | Meaning |
| ------ | ------ |
| [train file] | path of the training data |
| [setting file] | path of setting file that provides parameters for learning |
| [model folder] | path of the folder for saving the learned model |
| [test data folder] | path of the folder contains data for computing perplexity |

The model folder will contain some more files. These files contain some statistics of how the model is after a mini-batch is processed. These statistics include perplexity of the model, and time for finishing the E and M steps.

Example of execution:

```sh
$ python run.py ./data/nyt_50k.txt settings.txt ./models/nyt ./data
```

**Setting file**

See settings.txt for a sample.

**Data format**

The implementations only support reading data type in LDA. Please refer to the following site for instructions.

<http://www.cs.columbia.edu/~blei/lda-c/>

Under LDA, the words of each document are assumed exchangeable.  Thus, each document is succinctly represented as a sparse vector of word counts. The data is a file where each line is of the form:

[M] [term_1]:[count] [term_2]:[count] ...  [term_N]:[count]

where [M] is the number of unique terms in the document, and the [count] associated with each term is how many times that term appeared in the document.  Note that [term_1] is an integer which indexes the term; it is not a string.

Measure
-------

Perplexity is a popular measure to see predictiveness and generalization of a topic model.

In order to compute perplexity of the model, the testing data is needed. Each document in testing data is randomly divided into two disjoint part w_obs and w_ho with the ratio 70:30. They are stored in [test data folder] with corresponding file name is of the form:

data_test_part_1.txt and data_test_part_2.txt

Printing Topics
---------------

The Python script topics.py lets you print out the top N
words from each topic in a .topic file.  Usage is:

```sh
$ python topics.py [beta file] [vocab file] [n words] [result file]
```

License
----

MIT



