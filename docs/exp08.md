# Aim

To implement Transformer-based language models - a decoder-only **GPT** for text generation and an encoder-only
**BERT** for masked-language modelling - and to train and evaluate miniature versions on Shakespeare's text.

# Theory

The **Transformer** (Vaswani et al., 2017) replaces recurrence by *self-attention*:
$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,$$
computed in several parallel *heads*. A block consists of multi-head attention and a position-wise feed-forward network, each with
a residual connection and layer normalisation; learned *positional embeddings* inform the model about word order.

* **GPT** (Generative Pre-trained Transformer) is a *decoder-only* model with a **causal mask**: position $t$ attends only
  to positions $\le t$. It is trained to maximise $\prod_t p(x_t\mid x_{<t})$ and generates text autoregressively, using
  temperature and top-$k$ sampling to control randomness.
* **BERT** (Bidirectional Encoder Representations from Transformers) is an *encoder-only* model without a causal mask. It
  is pre-trained with **masked language modelling**: 15\% of the tokens are hidden (80\% replaced by `[MASK]`, 10\% by a random
  token, 10\% unchanged) and predicted from the context on *both* sides, which makes it well suited to understanding tasks
  (classification, question answering) after fine-tuning.
* **Perplexity** $=\exp(\text{cross-entropy})$ measures how surprised a language model is by held-out text.

# Steps

1. Tokenise the corpus into words and punctuation, build a 5000-word vocabulary and encode the text.
2. Implement a Transformer block (multi-head attention, feed-forward, residuals, layer-norm) and a model class that can
   be causal (GPT) or bidirectional (BERT).
3. Create next-token training windows (context 48) and compute a bigram-model baseline perplexity.
4. Train the mini-GPT, plot its loss and perplexity and generate text from prompts with different temperatures.
5. Create masked inputs for the mini-BERT and train it with the masked-language-modelling objective.
6. Evaluate masked-token accuracy, solve fill-in-the-blank examples, and visualise the attention maps of both models.

# Output

@@OUTPUT@@

# Conclusion

Both Transformer families were implemented from the same attention block and trained on Shakespeare (5000-word vocabulary, about 1.9 million parameters per model,
words rather than characters). The **mini-GPT** (8 epochs, 732 s) reduced the validation loss from 4.36 to 3.36, a perplexity of 28.9 compared with 85.7 for a
bigram baseline; training and validation loss fell together, so it was still under-trained rather than over-fitted. It generates text in the correct play format
(speaker names, line breaks, period vocabulary such as "Lady Capulet", "Bolingbroke") with locally plausible phrases, but, as expected from such a small model, the
sentences are not coherent over longer spans. The **mini-BERT** reached a masked-token accuracy of 33.4\% on held-out text after 8 epochs (always guessing the most frequent word gives 14.6\%).
In the fill-in-the-blank examples it predicted grammatically sensible words (for example a verb such as "make/take/bring" for "I will \_\_ you to the king"), but it did not recover the famous
completions ("Romeo, wherefore art thou \_\_"), so its accuracy is limited by the small corpus and short training. The attention maps show the causal (lower-triangular) pattern of GPT
against the full bidirectional pattern of BERT. Pre-trained GPT-2 or BERT weights, not available offline in this environment, would be needed for
fluent text and for downstream fine-tuning.
