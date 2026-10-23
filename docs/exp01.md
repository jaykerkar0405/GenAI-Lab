# Aim

To get familiar with Python and the TensorFlow/Keras framework, and to use them for the basic operations that every
generative-AI model is built from: tensors, automatic differentiation, training a neural network, and sampling.

# Theory

**Python** is the language of the machine-learning ecosystem. NumPy provides n-dimensional arrays, Matplotlib provides
plotting, and **TensorFlow** provides GPU/CPU-accelerated *tensors* together with *automatic differentiation*.
**Keras** is the high-level API of TensorFlow for assembling layers into models.

A *tensor* is an n-dimensional array with a shape and a dtype. TensorFlow records operations on a `tf.GradientTape`
and can then compute the gradient of any scalar loss $L(\theta)$ with respect to the parameters $\theta$, which gives
the gradient-descent update
$$\theta \leftarrow \theta - \eta \,\nabla_\theta L(\theta).$$

A *generative model* learns the data distribution $p(x)$ and draws new samples from it, typically by transforming
simple random noise $z \sim \mathcal{N}(0, I)$ with a neural network $G_\theta(z)$. Understanding tensors, gradients and
sampling is therefore the foundation for GANs, VAEs and Transformers studied in the later experiments.

# Steps

1. Verify the Python / NumPy / Matplotlib / TensorFlow installation and hardware.
2. Practise core Python constructs (functions, dictionaries, comprehensions).
3. Create tensors and perform arithmetic, matrix multiplication, reshaping, reduction and broadcasting.
4. Use `tf.GradientTape` to compute a derivative and to train a linear regression model with a manual training loop.
5. Load the MNIST dataset and visualise some images.
6. Build a Keras `Sequential` network, compile it with Adam and cross-entropy loss, train, and evaluate it.
7. Plot the learning curves and sample predictions.
8. Sample from a Gaussian mixture and pass random noise through a generator network to see the generative pipeline.

# Output

@@OUTPUT@@

# Conclusion

TensorFlow and Keras were set up and used successfully. Tensor operations and broadcasting behaved as expected, and
automatic differentiation returned the analytic gradient ($dy/dx = 29$ at $x=3$). The hand-written training loop
recovered the line parameters, and the Keras network learned MNIST digits reaching a test accuracy of 97.2\\% after five epochs.
Finally, drawing noise vectors and mapping them through a network demonstrated the
sample-from-noise pipeline on which the generative models in the following experiments are built.
