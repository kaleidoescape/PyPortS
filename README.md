PyPortS
=======

A Python3 implementation of [Keshava and Pitler's (2006) ``RePortS''][https://pdfs.semanticscholar.org/2b4e/de57fbd0ccf243c1763da2f882130ee292d2.pdf] algorithm for unsupervised morpheme induction. 

Running PyPortS
---------------

PyPortS is trained on a corpus of words. It accepts text files with one word per line for the training corpus. Multiple text files can be included.

It also requires a test corpus with a matching gold standard corpus. The test corpus is a text file with one word per line. The gold standard corpus has the same words as the test corpus, in the same order, but with plus signs (+) between the morphemes.

When training pyports.py, a version number needs to be included. This is used for saving the model for reuse in the future. 

To train pyports:

```
$ python3 pyports.py train ver_1.0.0 test.txt gold.txt train1.txt train2.txt
``` 

To test pyports:

```
$ python3 pyports.py test ver_1.0.0 test.txt gold.txt
```

Some datasets are already included from the original project (Russian, English, Japanese (kanji), Japanese (kunrei)). To run all of them in the same configuration as in the original project:

```
python3 pyports.py standard
```
