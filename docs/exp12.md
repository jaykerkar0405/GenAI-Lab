# Aim

To implement image inpainting - filling missing regions of an image with plausible content - using a Generative Adversarial Network
(context encoder), and to compare it with a purely reconstruction-based model.

# Theory

**Inpainting** restores a damaged or masked region $M$ of an image $x$ from its context. A **context encoder** (Pathak et al., 2016) is an
encoder--decoder generator $G$ that receives the damaged image $x\odot(1-M)$ (and the mask) and predicts the missing pixels; the final image is
the composite $\hat x=G(\cdot)\odot M+x\odot(1-M)$. Two losses are combined:

* **Reconstruction loss:** the L1 distance on the hole, $\mathcal{L}_{rec}=\|M\odot(G(\cdot)-x)\|_1$. It gives the correct overall
  structure but, being an average over plausible fillings, leads to *blurry* results.
* **Adversarial loss:** a discriminator $D$ tries to tell real images from completed ones, $\mathcal{L}_{adv}=-\log D(\hat x)$,
  which pushes the generator towards sharp, realistic texture.

The total generator loss is $\lambda\mathcal{L}_{rec}+\mathcal{L}_{adv}$ with $\lambda=100$. A **U-Net** with skip connections lets the generator reuse
detailed context from the encoder. Quality is measured by the **PSNR** $=10\log_{10}(1/\mathrm{MSE})$ and the **SSIM** (structural similarity)
between the completed and the original image, and, since many fillings are valid, also by visual inspection.

# Steps

1. Load Fashion-MNIST, pad it to $32\times 32$, scale to $[-1,1]$ and create a pool of random rectangular and stroke-shaped masks.
2. Build the U-Net generator (input: damaged image + mask) and the convolutional discriminator.
3. Train two generators with identical settings for 8 epochs: one with the L1 loss only, one with L1 + adversarial loss.
4. Plot the training curves.
5. Inpaint unseen test images with centre, rectangular and stroke masks and compare the models visually.
6. Compute PSNR, SSIM and the hole L1 error on 2000 test images and compare with a trivial baseline.

# Output

@@OUTPUT@@

# Conclusion

A U-Net context-encoder generator was trained for 8 epochs on 30,000 Fashion-MNIST images, once with the L1 loss alone and once with L1 plus an adversarial
loss. Both models filled centre holes, random rectangles and irregular strokes with plausible clothing shapes: on 2000 unseen test images the
PSNR was 24.4/26.3/26.0 dB (centre/rectangle/strokes) for the L1-only model and 23.8/25.9/25.7 dB for the L1 + GAN model, compared with only 12.4/14.8/14.1 dB
when the hole is simply filled with black. SSIM showed the same picture (0.82/0.89/0.88 against 0.81/0.89/0.88). The adversarial model therefore scored
slightly *lower* on these pixel-wise metrics (by about 0.4 to 0.6 dB), which is expected, because PSNR and SSIM reward the blurry average of all possible fillings whereas
the discriminator pushes the generator to commit to one sharper, realistic filling; the numbers alone do not capture sharpness, so visual inspection of
the side-by-side results is also needed. With only 8 epochs on a CPU the discriminator stayed fairly balanced (generator adversarial loss about 1.08), and longer
training and larger images would be needed to show a clear perceptual benefit of the GAN loss.
