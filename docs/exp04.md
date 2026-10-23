# Aim

To implement a Generative Adversarial Network (DCGAN) that generates realistic handwritten-digit images from random
noise, to train it with a custom adversarial loop in TensorFlow, and to evaluate the results.

# Theory

A **Generative Adversarial Network** (Goodfellow et al., 2014) consists of two networks trained against each other:

* the **generator** $G(z)$ maps random noise $z\sim\mathcal{N}(0,I)$ to a synthetic sample;
* the **discriminator** $D(x)$ estimates the probability that $x$ is real rather than generated.

They play the minimax game
$$\min_G \max_D \; V(D,G)=\mathbb{E}_{x\sim p_{data}}[\log D(x)]+\mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))].$$
At the optimum the generator distribution equals the data distribution and $D(x)=\tfrac12$. In practice the generator
minimises the *non-saturating* loss $-\log D(G(z))$, which gives stronger gradients early in training.

A **DCGAN** uses convolutional layers: strided transposed convolutions upsample in the generator, strided
convolutions downsample in the discriminator, batch-normalisation and LeakyReLU stabilise training, and the generator
ends with $\tanh$ so images live in $[-1,1]$. Tricks used here: Adam with $\beta_1=0.5$, one-sided label smoothing
(real label $0.9$) and dropout in the discriminator. Typical failure modes are *mode collapse* (the generator
produces few distinct outputs) and an unbalanced discriminator.

# Steps

1. Load MNIST and scale the pixels to $[-1,1]$; create a shuffled, batched `tf.data` pipeline.
2. Define the generator (Dense $\rightarrow$ 2 transposed-convolution blocks $\rightarrow$ $\tanh$ conv) and the
   discriminator (2 strided-convolution blocks $\rightarrow$ 1 logit).
3. Define the binary cross-entropy losses and two Adam optimisers.
4. Write a `tf.function` training step that updates both networks with `GradientTape`.
5. Train for 15 epochs while logging the losses and discriminator accuracy, saving samples from a fixed noise batch.
6. Plot the loss curves and the evolution of the generated digits.
7. Generate a final grid of 100 digits and interpolate between two latent vectors.
8. Evaluate the generated digits with an independent MNIST classifier and check class diversity.

# Output

@@OUTPUT@@

# Conclusion

A DCGAN was implemented and trained on MNIST for 15 epochs. Starting from pure noise, the generator learned to produce
digit-like images within a few epochs (see the evolution grid). Training stayed in a balanced adversarial regime: the
discriminator accuracy settled at about 0.57 and both losses were stable, with no collapse. Interpolating between two latent
vectors produced a smooth transition (a 9 gradually becoming a 1), which indicates a meaningful latent space. An independent classifier
(97.9\% accurate on real digits) assigned a mean confidence of 0.86 to the generated digits and was more than 90\% sure for 60.3\%
of them, and all ten classes were generated in comparable numbers (126 to 256 out of 2000), so there was no mode collapse. The
remaining blurry or malformed samples show that a small DCGAN trained for a short time on a CPU does not yet match the realism of
real digits; longer training would improve this.
