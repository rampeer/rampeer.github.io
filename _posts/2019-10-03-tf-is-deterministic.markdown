---
layout: post
title:  "TF is Determinisic (finally)! and my other thoughts on TF2.0"
date:   2019-10-03 00:00:00 +0300
categories: 
---

# Tensorflow is Determinisic (finally)! and my other thoughts on TF2.0

TF 2.0 was released a few months ago. There are people who had the first-hand experience with it.
I'm starting a new project, so I decided to invest time in digging into TF 2.0 and join this circle of pioneers.

My findings do not agree with the general sentiment about TF in the community, and I found a 
few peculiar things. Because of that, I decided to share it with you.

## TF2.0 is a bootleg Torch?

Yes, the interfaces and overall code style have suspicious PyTorch vibes. Just look [at that](https://www.tensorflow.org/guide/keras/custom_layers_and_models):

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

The similarity is uncanny. However, I would like to point out a few things:

1) There are some extra features added. For example, it is possible to 
[decorate functions](https://www.tensorflow.org/api_docs/python/tf/function) 
to let TensorFlow construct a graph from it and optimize its execution (sounds like magic). 
As far as I know, PyTorch follows a slightly different paradigm, and cannot do any precomputations or optimizations.

2) The similarity is not a flaw. Wrapping network logic into a class is a well-established technique 
that has been proven to be effective. There is no point in inventing the square wheel just to be unique. 
And, in fact, [Lasagne](https://github.com/Lasagne/Lasagne) and [Chainer](https://github.com/chainer/chainer/) 
did it before PyTorch (Lasagne seems to be older, but most resources point out Chainer to be the first to use dynamic graphs).

To sum - yep, they are similar, and it’s okay.

## Swift opens a new DL era

Oh my, if I would be given a dollar whenever I read/hear this phrase, I would have bought a yacht by now.

Swift TensorFlow is a distinct project from TF2.0, but I (like many others) confused them. 
This project is about adding TF support into Swift, which will set up a new cornerstone in ML and will open a gateway 
to the better AI development (judging by numerous ecstatic news articles).

No, it won't. It is just a language. At best, it will improve quality of your code in your solution and
performance of your models, as authors claim. In industry, modelling is a part of larger process. Data has to come 
from somewhere, and trained models must be stored and used elsewhere. Right now Swift does not have huge choice of
libraries for ETL, data munging, preprocessing, visualization (Python neither ;), experiment tracking, 
model deployment and other necessary stuff, so you will have to rely on other tools for that. 
In addition to that, ML is not just a bunch of libraries; it is ecosystem that also consists of best practices, 
educational materials, pieces of code scattered across Internet, thousand of answered questions in StackOverflow, and, of course,
people who know and use said libraries. DL Swift does not have all that. Using it will cost you a lot
since you have to support your solution, but ML engineers familiar with Swift are scarce and expensive (think of Scala 5 years ago). 
It is not clear whether increment of quality of product from using Swift will worth all that hassle.

You may argue that researchers and academia may benefit from this project more. Maybe, but in my experience
researchers eat RTXes on breakfast, lunch and dinner, and they prefer to throw in more computational power than to
clean/optimize the code.

But what's even more intriguing, why *Swift?*

I had this burning question for a while. 
Python is slow, yes - but there are a lot of different languages, and picking one that is known for mobile development 
is a strange choice.

There is a [document](https://github.com/tensorflow/swift/blob/master/docs/WhySwiftForTensorFlow.md)
describing the cons and pros. To sum up, they had to choose a language that 
would allow some static analysis for optimizations and graph extractions on one hand, and won’t be too 
cumbersome to use (and to switch from Python) on the other hand. 
Arguments seem to be fairly convincing, until the very end:

>[We] picked Swift over Julia because Swift has a much larger community, 
>is syntactically closer to Python, 
>and because we were more familiar with its internal implementation details - which allowed us to implement a prototype much faster.

Larger community, huh? It's mostly mobile developers who won't contribute or use TF in a meaningful way. The document
the author even admits that a few paragraphs later:

>Swift does not have much in the way of a data science community. It has a small community that we hope to engage with

Julia is also a relatively new language with the quickly-growing ML community, but for some reason, they decided not to use it.
Moreover,

>It is worth noting that as of our launch in April 2018, Swift for TensorFlow is not yet solving some of these goals

Whoops.

I felt that there is something else going on, something is amiss. My hunch was right. 
[Chris Lattner](https://en.wikipedia.org/wiki/Chris_Lattner) ([his homepage](http://nondot.org/sabre/) ) who is 
responsible for creating Swift, was invited to work in Google to work on "TensorFlow Accelerator Support and 
Compiler Infrastructure", and later - on porting TF to a more suitable language. 
And of course, among all languages he decided to choose the one he created himself.

So, it seems that the last argument (about internal implementation details) is the most important one, despite looking
inconsiderable.

Added: Personally, I do not think that Swift is bad / this choice is wrong, but I have predict that it will have
troubles displacing conventional instruments and building a community, and I find this choice to be peculiar and unusual.

## Determinism

TF 2.0 was released a few months ago. One of the release notes points attracted my attention:

>Add environment variable TF_CUDNN_DETERMINISTIC. Setting to TRUE or "1" forces the selection 
of deterministic cuDNN convolution and max-pooling algorithms. When this is enabled, the algorithm selection 
procedure itself is also deterministic.

Inability to train models in deterministic fashion was a [major pain]({% post_url 2019-05-29-random-seeds %}) in TF 1.0.

With this new flag, it is relieved. I'll make a run tests and write a post about determinism and performance.

## To sum up

I'm excited to try out TF2.0 on one of my pet projects. They have greatly improved the syntax and implemented features
that we've been waiting for ages. But since it is a relatively new technology, 
I would be cautious when using it in a full-production environment.