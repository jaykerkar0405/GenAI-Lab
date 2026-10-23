# Aim

To build character-level text generators using a Recurrent Neural Network (RNN) and a Long Short-Term Memory (LSTM)
network, to train them on Shakespeare's plays, and to generate new text with controllable randomness.

# Theory

A **recurrent neural network** processes a sequence $x_1,\dots,x_T$ while carrying a hidden state
$h_t=\tanh(W_x x_t+W_h h_{t-1}+b)$. A **language model** estimates $p(x_t\mid x_{<t})$ and is trained with
cross-entropy on next-token prediction; its quality is measured by *perplexity* $=\exp(\text{cross-entropy})$.

Plain RNNs suffer from vanishing/exploding gradients over long sequences. The **LSTM** adds a cell state $c_t$ and gates:
$$f_t=\sigma(W_f[h_{t-1},x_t]),\; i_t=\sigma(W_i[h_{t-1},x_t]),\; o_t=\sigma(W_o[h_{t-1},x_t]),$$
$$c_t=f_t\odot c_{t-1}+i_t\odot\tanh(W_c[h_{t-1},x_t]),\qquad h_t=o_t\odot\tanh(c_t),$$
so information can be remembered or forgotten selectively.

To generate text, the softmax distribution of the next character is sampled repeatedly and the sampled character is fed back.
A *temperature* $T$ rescales the logits, $p_i\propto\exp(z_i/T)$: $T<1$ makes the output conservative, $T>1$ more random.

# Steps

1. Load the Shakespeare corpus (about 1.1 million characters), build the character vocabulary and integer encoding.
2. Create overlapping training windows of 100 characters (stride 50) with targets shifted by one character; split into train/validation.
3. Define two models: Embedding $\rightarrow$ SimpleRNN(256) $\rightarrow$ Dense, and Embedding $\rightarrow$ LSTM(256) $\rightarrow$ Dense.
4. Train both with Adam and gradient clipping; plot loss and perplexity.
5. Write a sampling function with temperature and generate text from a seed.
6. Compare the models and the temperatures using the share of generated words that are valid.

# Output

@@OUTPUT@@

# Conclusion

Both recurrent models learned the format of the plays (speaker names in capitals followed by a colon, line breaks, capitalisation) and
mostly real English words from raw characters. The LSTM (12 epochs, 489 s) reached a validation loss of 1.375, i.e. a perplexity of
3.96, compared with 1.510 (perplexity 4.53) for the SimpleRNN (6 epochs, 98 s), so the gated memory gives a clearly better model, although the
LSTM also trained for more epochs and took about five times longer. Generated text is locally fluent but has little global meaning, as expected from
a character-level model. Temperature controls the trade-off: at $T=0.7$ 95.0\% of the LSTM's words were valid vocabulary words, at $T=1.0$ 77.8\%
and at $T=1.3$ only 69.5\%, while low temperatures give safe but repetitive text. (At $T=0.3$ the SimpleRNN scored slightly higher, 99.3\% versus
97.7\%, because its output is more repetitive.)
