# Aim

To generate music with AI by training sequence models - an LSTM and a Transformer - on the melodies of
J. S. Bach's chorales, and to evaluate the generated music with objective metrics.

# Theory

Symbolic music (notes rather than audio) can be modelled like text. Each melody is converted into a sequence of *tokens*;
here one token per eighth-note time step: a **pitch** token at a note onset, a **hold** token while the note continues
and a **rest** token. A generative model then learns $p(x_t\mid x_{<t})$ and music is produced by sampling the next token
repeatedly (with temperature and top-$k$ filtering).

* The **LSTM** keeps a recurrent memory of the recent melody and is a classical model for music generation (e.g. early
  Magenta models).
* The **Transformer** attends directly to all earlier steps with causal self-attention, which captures repeated motifs and
  phrase structure at longer ranges (as in Music Transformer).
* Because there is no single right answer, evaluation combines *perplexity/next-token accuracy* with musical
  statistics: the share of notes in the key (diatonic consistency), the average melodic leap, the pitch range, the
  pitch-class histogram, and the *novelty* (share of 6-grams not present in the training data).
* Output can be exported as **MIDI** (note events) and rendered to audio.

# Steps

1. Parse 140 Bach chorales with **music21**, transpose them to C major / A minor and convert the soprano line to tokens.
2. Visualise a real melody as a piano roll; create training and validation windows (split by chorale).
3. Define a 2-layer LSTM and a 3-layer causal Transformer and train both.
4. Compare the learning curves, perplexity and next-token accuracy.
5. Generate melodies from a short seed with both models and plot piano rolls.
6. Compute the musical metrics and the pitch-class histograms, and export MIDI and WAV files.

# Output

@@OUTPUT@@

# Conclusion

The 140 Bach chorales gave a small dataset of 15,228 eighth-note steps (91 tokens: rest, hold and 33 distinct pitches). Both models learned the
melodic grammar: the LSTM reached a validation perplexity of 3.06 and a next-step accuracy of 64.6\%, the Transformer 4.38 and 63.1\%. Although the
Transformer fitted the training data better (training loss 0.574 against 0.775 for the LSTM), it generalised worse (validation loss 1.478 against
1.118), a sign of over-fitting on such a small corpus; Transformers usually need more data than recurrent models to show their advantage. The
generated melodies stayed in key (100\% of notes belong to the C-major/A-minor scale set, though this metric is lenient), had a mean melodic leap of 1.91
(LSTM) and 1.95 (Transformer) semitones, close to the 2.04 of real chorale melodies, and were largely original: 11.5\% (LSTM) and 23.5\% (Transformer) of their
6-grams do not occur in the training data. The Transformer is therefore the more "creative" but less faithful model. The melodies were exported as MIDI and WAV
files and could be extended to polyphony by modelling all four chorale voices.
