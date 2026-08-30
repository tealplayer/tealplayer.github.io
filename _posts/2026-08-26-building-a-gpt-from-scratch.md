---
layout: post
title: "Building a GPT from Scratch"
date: 2026-08-26
---

<script>
  window.MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']]
    }
  };
</script>

<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async>
</script>

The modern AI boom all sprung from an architecture originally meant for translating text from one language into another, called the transformer. Introduced by Vaswani et al.[1] in 2017, it's the basis for AI models such as Claude and ChatGPT. In this, I reimplement a decoder-only version from scratch, trained it on Shakespeare, investigated model behavior based on the one-layer interpretability in [2], and went looking for induction heads inspired by the two-head circuit that [2] identifies as the mechanism behind in-context learning. Although I didn't find any induction heads, there are some interesting implications of this. This post is about what I built, what I found instead, and why I think character-level tokenization is the reason the circuit never formed

**What I Built and Why**

My model is a decoder-only GPT built from scratch. The architecture is from [1] and the decoder-only aspect is from the GPT lineage, specifically from GPT-1[3] and GPT-2[4]. I wrote every component myself rather than importing a transformer, learning the parts from 3Blue1Brown YouTube videos and readings, and [1]. Encoder-decoder models are for when there's a distinct source and target. For example, [1] used this type of model for translation, where you have maybe an English string as the source where you can view all the tokens(the encoder part), and the decoder part then generates, say, French tokens one at a time which is the target, and its cross attention queries into a matrix formed by the encoded English. That part where it predicts the French words is basically a decoder-only model plus the cross attention into the encoder outputs. For decoder-only models(mine), there is only one residual stream. Given some Shakespeare, the model must predict the next character. Before the current token, everything is context for the model. The token we're on soon becomes that context for the tokens after it. So with each token, the context increases by one token(i.e. token 5 has 5 tokens of context and token 200 has 200). The difference between encoder-decoder and decoder-only is that in the former, cross-attention always reaches back into the same fixed amount of tokens, the encoded output regardless of whether you're generating token 1 or token 5 for translation. Here, the source doesn't change, while the decoder performs self attention over its own tokens like how my model does. But, the decoder has one thing that grows and one thing that's fixed. In a decoder-only model, nothing is fixed. The only thing each position attends over is its own growing context. This is why the decoder-only model doesn't have a clear source, and why you can't say "it's the context". 

**Architecture and Choices I Made**

First, the attention. Attention is where the transformer looks back at the context to see how well our query tokens attend to the key tokens. Queries ask the question: "What am I looking for?" at each query token, which are produced by multiplying the token's residual stream vector by the learned matrix $W_Q$. Keys determine whether a token gets attended to, and are produced the same way as queries are, by multiplying the same vector by $W_K$. How do these query and key vectors interact? They interact by forming a matrix that holds the scores going into the Softmax function. We'll talk about the Softmax in a second, but first: the scores matrix. The rows are the queries that have the tokens asking "What am I looking for?", and the keys are tokens in the context possibly answering that question. Each dot product between the query and key represents how well a query attends to a key. After these dot products are computed, each entry(dot product) in this scores matrix gets scaled by being divided by the square root of the key dimension $\sqrt{d_k}$. Why by $\sqrt{d_k}$? Well, when we compute the dot product by a query and a key it looks like: $q * k = q_1k_1 + q_2k_2 + ... + q_{d_k}k_{d_k}$, so d_k is the number of products summed together which in my model is 64 products summed together. Each score in the scores matrix is the sum of 64 terms. So suppose each $q_i$ and $k_i$ have a mean of 0 and a variance of 1, which is approximately true at initialization. That means each product has a mean of 0 and a variance of 1 as well, which add as you sum them since each $q_i$ and $k_i$ is an independent random variable, so the total variance is 64, which means a standard deviation of $sqrt{64} = 8$ and the scores spread over a range around ±8. This scales with $d_k$, so if you double $d_k$ then you double the variance and the spread grows, so we divide by $\sqrt{d_k}$ to bring the variance back to 1. Now the reason we even do this at all is because of Softmax. Softmax is an exponential function that normalizes the scores to range from 0-1 and sum to 1 for each row/query. If we don't scale the scores before Softmax, then one key will essentially get all the attention, and everything else rounds to 0. The attention weights produced by this matrix wouldn't be a weighted blend that represents each key fairly. Another problem is that the Softmax gradients vanish when the output is saturated like that, so the model barely learns. So, scaling by $frac{1}{sqrt{d_k}}$ controls the spread caused by the dot product computations. The scores matrix is then:

$\frac{QK^T}{\sqrt{d_k}}$,

where Q and K are the matrices holding all the query and key vectors, one per token position.
