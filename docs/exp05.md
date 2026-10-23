# Aim

To fine-tune a GAN progressively from low to higher image resolution ($8\times 8$ up to $64\times 64$) and to analyse how
fine-tuning from a pre-trained model and the learning rate affect the quality of the generated images.

# Theory

Training a GAN directly at high resolution is slow and unstable, because the discriminator can easily tell real
images from blurry fakes. **Progressive growing** (Karras et al., 2018) trains the generator and discriminator on
low-resolution images first and adds layers that double the resolution, one stage at a time. This is a form of
*transfer learning / fine-tuning*: the large-scale structure learned at low resolution is reused, and each new layer
only has to learn finer detail.

* **Fade-in:** when a new block is added, the output is blended,
  $\text{out}=(1-\alpha)\,\text{up}(x_{old})+\alpha\,x_{new}$, with $\alpha:0\to1$, so the network is not shocked by the new layers.
* **Pixel normalisation** of feature vectors keeps the generator's signal magnitudes bounded. The generator output is *linear* (no $\tanh$): a first attempt with a $\tanh$ output saturated, the generator collapsed to a constant image and the discriminator loss fell to zero.
* **Least-squares GAN loss:** $L_D=\tfrac12\mathbb{E}[(D(x)-1)^2]+\tfrac12\mathbb{E}[D(G(z))^2]$, $L_G=\tfrac12\mathbb{E}[(D(G(z))-1)^2]$,
  which gives smoother gradients than the log loss.
* **Fine-tuning hyper-parameters:** the learning rate and the number of fade-in steps. Too small a rate adapts the
  pre-trained layers too slowly, too large a rate destroys the transferred features.
* **Evaluation:** the Fréchet distance $\|\mu_r-\mu_g\|^2+\mathrm{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$ between
  Gaussian fits of real and generated *feature* vectors (the principle of FID; here the features come from a small
  classifier trained on Fashion-MNIST, since the Inception network is not used).

# Steps

1. Resize Fashion-MNIST (20,000 images) to 8, 16, 32 and 64 pixels.
2. Implement the progressive generator and discriminator with fade-in layers and pixel normalisation.
3. Train a classifier on the dataset and use its features to compute the Fréchet distance.
4. Train progressively: $8\times 8$, then 16, 32 and 64 with 350 fade-in and 350 stabilising steps each; save a checkpoint
   of the $16\times 16$ model.
5. Visualise samples at each resolution and the loss history.
6. Fine-tuning study at $32\times 32$ with the same step budget: training from scratch versus fine-tuning the $16\times 16$ checkpoint with
   learning rates $2.5\cdot10^{-5}$, $10^{-4}$ and $5\cdot10^{-4}$; compare the Fréchet distances and samples.

# Output

@@OUTPUT@@

# Conclusion

A progressively growing GAN was trained from $8\times 8$ up to $64\times 64$ on Fashion-MNIST. At every stage the generator and discriminator losses stayed balanced (both about
0.2), and the samples became sharper and more detailed with each added resolution: blurry blobs at $8\times 8$, recognisable shoes, boots, shirts and dresses at $16\times 16$ and $32\times 32$, and
finer shapes at $64\times 64$. The Fréchet distance (measured after resizing to $32\times 32$) fell from 114.3 ($8\times 8$) to 60.3 ($16\times 16$) and 31.2 ($32\times 32$); it was slightly higher, 38.7, at $64\times 64$, so the
$64\times 64$ stage with only 700 steps did not improve this metric, even though its images look more detailed. In the fine-tuning study with the same 700-step budget, training the $32\times 32$ model
from scratch failed badly (Fréchet distance 277.8; its samples collapsed to almost the same image), whereas fine-tuning the pre-trained $16\times 16$ model reached 52.7 (learning rate $2.5\cdot10^{-5}$),
56.1 ($10^{-4}$) and 37.7 ($5\cdot10^{-4}$), a reduction of up to 86\% relative to scratch. The highest rate gave the best Fréchet distance but visibly blobbier shapes, so the metric and
the images should be read together, and each configuration was run once, so small differences between the fine-tuning rates should not be over-interpreted. Overall the results show that transferring
low-resolution weights with fade-in makes training faster and more stable than training at the target resolution from scratch. It must also be said that the first attempt of this experiment failed:
the generator used a $\tanh$ output, which saturated, the generator collapsed to a constant image and the discriminator loss fell to zero. Replacing it by a linear output (as in ProGAN) and lowering the learning
rate fixed this. The images are still far from photo-realistic, because the models were trained for only a few thousand CPU steps.
