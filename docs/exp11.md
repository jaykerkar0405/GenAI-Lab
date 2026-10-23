# Aim

To implement neural style transfer with a pre-trained convolutional network, creating images that combine the content of a photograph
with the artistic style of a painting.

# Theory

**Neural style transfer** (Gatys, Ecker and Bethge, 2015) treats a pre-trained image-classification CNN (VGG19) as a feature extractor.
Deep layers respond to *what* is in the image (objects and layout), whereas the *correlations* between feature channels in several layers
describe the *texture and style*. For a layer with feature map $F\in\mathbb{R}^{N\times C}$ the **Gram matrix** is $G=F^\top F/N$.

A new image $x$ is obtained by gradient descent **on the pixels** (the network weights stay frozen) to minimise
$$\mathcal{L}(x)=\alpha\,\|F^{l}(x)-F^{l}(c)\|^2+\beta\sum_{l}w_l\,\|G^{l}(x)-G^{l}(s)\|^2+\gamma\,TV(x),$$
where $c$ is the content image, $s$ the style image, and the total-variation term $TV$ suppresses noise. The ratio $\beta/\alpha$ controls
how strongly the style is imposed. The method is *generative* because it synthesises a new image, but it needs an optimisation for every
image pair; later feed-forward models (fast style transfer, AdaIN) learn to stylise in one pass.

# Steps

1. Download two content photographs and one painting (Kandinsky, *Composition 7*) and resize them to 256 pixels.
2. Load VGG19 with ImageNet weights (without the classifier) and select `block5_conv2` for content and five `conv1` layers for style.
3. Implement Gram matrices and the content, style and total-variation losses.
4. Optimise the pixels of the image with Adam for 250 steps; log the losses and save intermediate images.
5. Repeat with a second content image and with three different style weights.
6. Quantify the style distance to the painting and the content distance to the photo.

# Output

@@OUTPUT@@

# Conclusion

Neural style transfer was implemented with a frozen VGG19. In 250 optimisation steps (about 2 minutes per image on a CPU) the style loss fell from
$2.8\times10^{11}$ to $1.4\times10^{9}$ while the content loss stayed at 1865, so the Labrador and the turtle kept their shapes but acquired
Kandinsky's colours, strokes and textures. The style weight controls the balance: raising it from $10^{-3}$ to $10^{-1}$ lowered the style distance to the painting
from $2.76\times10^{9}$ to $1.01\times10^{9}$ and raised the content distance from 473 to 6207, i.e. a stronger style costs a less recognisable photograph.
The method needs an optimisation for every image pair, which motivates the feed-forward style-transfer networks used in practice.
