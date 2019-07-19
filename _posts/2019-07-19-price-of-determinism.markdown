---
layout: post
title:  "Price of determinism"
date:   2019-07-19 00:00:00 +0300
categories: 
---

This is a short update to [the previous post]({% post_url 2019-06-13-keras-pain %}). It's tl;dr: in order to
make deterministic pipelines and reproducible modules, we have to enable a flag in PyTorch. This flag tells 
CuDNN to disable some optimizations, which in turn guarantees that each time computations will return the same
results.

Reproducibility comes with a price of longer training times. But how much? 

Well, I made a script to test that.

[**Here it is!**](https://github.com/rampeer/rampeer.github.io/blob/master/sources/reproducibility/reproducibility_cnn_torch_timing.py)

It trains simple model several times, and outputs average training time.

For me, its output looks like

```
Non-deterministic training times: 1.1445602107048034 +- 0.07025331596594957
Deterministic training times: 1.1756325054168701 +- 0.07699960712582016
```

In other words, the slowdown is negligible in comparison with intrinsic training time variation.

Similar results were acquired on several machines with different hardware and software versions.

More complex models may have different, and more noticeable effect. But my point here: we are not speaking about 10x
performance decreases, it's unlikely to be an issue, and it worth to try.
