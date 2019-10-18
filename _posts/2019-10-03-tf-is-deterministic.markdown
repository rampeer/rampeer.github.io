---
layout: post
title:  "TF is Determinisic (finally)! and my other thoughts on TF2.0"
date:   2019-10-03 00:00:00 +0300
categories: 
---

# Tensorflow is Determinisic (finally)! and my other thoughts on TF2.0

TF 2.0 was released a few months ago. There are people who had first-hand experience with it.
I'm starting a new project, so I decided to invest time into digging into TF 2.0 and join this circle of pioneers.

My findings do not agree with general sentiment about TF, and I found a few peculiar things. Because of that I decided 
to share it with you.

## TF2.0 is a bootleg Torch?

Yes, the code looks suspiciously like PyTorch. Just look [at that](https://www.tensorflow.org/guide/keras/custom_layers_and_models):

```python
class Linear(layers.Layer):

  def __init__(self, units=32, input_dim=32):
    super(Linear, self).__init__()
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                              dtype='float32'),
                         trainable=True)
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(initial_value=b_init(shape=(units,),
                                              dtype='float32'),
                         trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b
```

Similarity is clear. However, I would like to point out a few things:

1) There are some extra features added. For example, it is possible to [decorate functions](https://www.tensorflow.org/api_docs/python/tf/function)
to let TensorFlow construct a graph from it and optimize its execution (sounds like magic). As far as I know, PyTorch
follows a slightly different paradigm, and cannot do any precomputations or optimizations.

2) Similarity is not a flaw. Wrapping network logic into a class is well-established technique which 
has been proven to be effective and reliable. There is no point in inventing a square wheel just to be distinct. 
And to be specific, [Lasagne](https://github.com/Lasagne/Lasagne) and [Chainer](https://github.com/chainer/chainer/) 
did it before PyTorch (Lasagne seems to be older, but most resources point out Chainer to be the first to use dynamic 
graphs).

To sum - yep, they are similar, and it's great.

## Determinism

TF 2.0 was released a few months ago. One of the release notes points attracted my attention:

>Add environment variable TF_CUDNN_DETERMINISTIC. Setting to TRUE or "1" forces the selection 
of deterministic cuDNN convolution and max-pooling algorithms. When this is enabled, the algorithm selection 
procedure itself is also deterministic.

Inability to train models in deterministic fashion was a [major pain]({% post_url 2019-05-29-random-seeds %}) in TF 1.0.

With this new flag, it is relieved.

## Swift opens a new DL era

Oh my, if I would be given a dollar whenever I read/hear this phrase, I would have bought a yacht by now.

Swift TensorFlow is a distinct project from TF2.0, but I (like many others) confused them. This project is about
adding TF support into Swift.

*But why Swift*?

I had this burning question for a while. Python is slow, yes - but there are a lot of different languages, and picking
one that is known for mobile development is a peculiar choice.
There is a [document](https://github.com/tensorflow/swift/blob/master/docs/WhySwiftForTensorFlow.md)
describing cons and pros. To sum up, they had to choose a language that would allow some static analysis for
optimizations and graph extractions on one hand, and won't be too cumbersome to use (and to switch from Python) on the other hand.
Arguments seem to be fairly convincing, until the very end:

>[We] picked Swift over Julia because Swift has a much larger community, 
>is syntactically closer to Python, 
>and because we were more familiar with its internal implementation details - which allowed us to implement a prototype much faster.

Larger community, huh? It's mostly mobile developers who won't contribute or use TF in a meaningful way. The document
author even admits that a few paragraphs later:

>Swift does not have much in the way of a data science community. It has a small community that we hope to engage with

Julia is also a relatively new language with quickly-growing ML community, but for some reason they decided not to use it.
Moreover,

>It is worth noting that as of our launch in April 2018, Swift for TensorFlow is not yet solving some of these goals

Whoops.

I felt that there is something else going on, something is amiss. My hunch was right. 
[Chris Lattner](https://en.wikipedia.org/wiki/Chris_Lattner) who is responsible for creating Swift, was invited to
work in Google to work on "TensorFlow Accelerator Support and Compiler Infrastructure", 
and later - on porting TF to a more suitable language. And of course, among all languages he 
decided to choose the one he created himself. (His homepage)[http://nondot.org/sabre/].

So, it seems that the last argument (about internal implementation details) is the most important one, despite looking
inconsiderable.

P.S. Personally, I do not think that Swift is bad / this choice is wrong, but I do find this choice to be peculiar and unusual.
