# Aim

To prepare raw tabular, image, text and audio data so that it can be fed to generative models: cleaning, imputation,
scaling, encoding, normalisation, augmentation, tokenisation, padding and building efficient input pipelines.

# Theory

Generative models are very sensitive to how data is represented, so *preprocessing* is a core part of the workflow.

* **Cleaning** removes duplicates, fixes missing values (*imputation* by median / most-frequent value) and limits
  outliers, e.g. with the inter-quartile-range rule $[Q_1-1.5\,\mathrm{IQR},\;Q_3+1.5\,\mathrm{IQR}]$.
* **Scaling** brings features to a comparable range: standardisation $z=(x-\mu)/\sigma$ or min--max scaling.
  Statistics must be learned on the training split only, to avoid data leakage.
* **Encoding** converts categorical variables into numbers (one-hot encoding).
* **Images** are normalised, e.g. to $[-1,1]$ so that they match the $\tanh$ output of a GAN generator, resized, and
  *augmented* (flips, crops, brightness, noise) to enlarge the dataset.
* **Text** is cleaned, split into tokens (characters or words), mapped to integer ids using a vocabulary, and padded
  to a fixed length. For language models, inputs are windows and the targets are the same windows shifted by one token.
* **Audio** is peak-normalised and converted into time--frequency representations (STFT, mel-spectrogram).
* `tf.data` builds fast pipelines: `shuffle`, `map`, `batch`, `prefetch`.

# Steps

1. Build a messy tabular dataset; inspect missing values, duplicates and outliers.
2. Remove duplicates, clip outliers by the IQR rule, and impute, scale and one-hot encode with a `ColumnTransformer`
   pipeline fitted on the training split only.
3. Load Fashion-MNIST, rescale pixels to $[-1,1]$, and compare the histograms before and after.
4. Apply resizing and augmentations with `tf.image`.
5. Build a `tf.data` pipeline (shuffle, map, batch, prefetch) and inspect a batch.
6. Load the Shakespeare corpus, clean it, tokenise it into words with `TextVectorization` and build character windows
   for sequence models.
7. Normalise a synthetic audio waveform and compute its log-mel spectrogram.

# Output

@@OUTPUT@@

# Conclusion

Raw data of four different types was converted into model-ready tensors. The tabular pipeline removed all missing
values and produced standardised numeric features (mean $\approx 0$, std $\approx 1$) plus one-hot categorical columns;
Fashion-MNIST was normalised to $[-1,1]$ and served through an efficient `tf.data` pipeline with augmentation;
the Shakespeare text was cleaned, tokenised and windowed for next-token prediction; and the audio signal was turned into
a log-mel spectrogram. These steps are reused by the generative models in the following experiments.
