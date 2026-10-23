# %% [markdown]
# # Experiment 1: Introduction to Python and TensorFlow for Generative AI
#
# **Aim:** Become familiar with Python scientific libraries and the TensorFlow / Keras API, and use them to build the
# first building blocks of a generative model (tensors, automatic differentiation, a trainable network, and sampling).

# %% [markdown]
# ## 1. Environment check

# %% title: Library versions and hardware
import os, sys, platform
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

print("Python     :", sys.version.split()[0])
print("Platform   :", platform.platform())
print("NumPy      :", np.__version__)
print("Matplotlib :", matplotlib.__version__)
print("TensorFlow :", tf.__version__)
print("Keras      :", keras.__version__)
print("GPUs       :", tf.config.list_physical_devices("GPU") or "none (running on CPU)")

# %% [markdown]
# ## 2. Python basics used throughout the lab

# %% title: Python basics - lists, dicts, functions, comprehensions
def normalise(xs):
    lo, hi = min(xs), max(xs)
    return [(x - lo) / (hi - lo) for x in xs]

pixels = [0, 64, 128, 192, 255]
print("scaled pixels :", normalise(pixels))
config = {"latent_dim": 100, "batch_size": 64, "epochs": 5}
for k, v in config.items():
    print(f"{k:12s} = {v}")
squares = {n: n ** 2 for n in range(1, 6)}
print("squares       :", squares)

# %% [markdown]
# ## 3. Tensors, shapes and broadcasting

# %% title: Creating and manipulating tensors
a = tf.constant([[1., 2., 3.], [4., 5., 6.]])
b = tf.ones((2, 3)) * 2
print("a.shape, a.dtype :", a.shape, a.dtype)
print("a + b =\n", (a + b).numpy())
print("a @ a^T =\n", tf.matmul(a, a, transpose_b=True).numpy())
print("reshape(3,2) =\n", tf.reshape(a, (3, 2)).numpy())
print("column-wise mean :", tf.reduce_mean(a, axis=0).numpy())
print("broadcast a + [10,20,30] =\n", (a + tf.constant([10., 20., 30.])).numpy())
v = tf.Variable(tf.random.normal((2, 2), seed=1))
v.assign_add(tf.ones((2, 2)))
print("Variable after assign_add =\n", v.numpy())

# %% [markdown]
# ## 4. Automatic differentiation with `GradientTape`
# Generative models are trained by gradient descent on a loss, so we first check that TensorFlow's gradients match the
# analytic derivative of $y = x^3 + 2x$, and then fit a straight line with a hand-written training loop.

# %% title: Automatic differentiation
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x ** 3 + 2 * x
print("y        =", y.numpy())
print("dy/dx    =", tape.gradient(y, x).numpy(), "(analytic: 3x^2 + 2 = 29)")

# %% title: Linear regression from scratch with GradientTape
tf.random.set_seed(0); np.random.seed(0)
X = np.random.rand(200, 1).astype("float32")
Y = 3.0 * X + 0.5 + 0.05 * np.random.randn(200, 1).astype("float32")
w, c = tf.Variable([[0.0]]), tf.Variable([0.0])
opt = keras.optimizers.SGD(0.5)
losses = []
for step in range(150):
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean((X @ w + c - Y) ** 2)
    g = tape.gradient(loss, [w, c])
    opt.apply_gradients(zip(g, [w, c]))
    losses.append(loss.numpy())
    if step % 30 == 0:
        print(f"step {step:3d}  loss {loss.numpy():.5f}")
print(f"learned w = {w.numpy()[0,0]:.3f} (true 3.0), b = {c.numpy()[0]:.3f} (true 0.5)")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot(losses); ax[0].set_title("Training loss"); ax[0].set_xlabel("step"); ax[0].set_ylabel("MSE")
ax[1].scatter(X, Y, s=8, label="data")
xs = np.linspace(0, 1, 10)
ax[1].plot(xs, w.numpy()[0, 0] * xs + c.numpy()[0], "r", label="fitted line"); ax[1].legend()
ax[1].set_title("Fitted line")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 5. A first neural network with Keras (MNIST classifier)
# Keras `Sequential` API: define layers, compile with an optimiser and loss, then `fit`.

# %% title: Loading and inspecting the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print("train images:", x_train.shape, x_train.dtype, " labels:", y_train.shape)
print("test images :", x_test.shape)
print("pixel range :", x_train.min(), "-", x_train.max())
fig, axes = plt.subplots(2, 8, figsize=(11, 3))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(x_train[i], cmap="gray"); ax.set_title(int(y_train[i]), fontsize=9); ax.axis("off")
plt.suptitle("Sample MNIST digits"); plt.tight_layout(); plt.show()

# %% title: Building and training the Keras model
keras.utils.set_random_seed(42)
x_tr = (x_train / 255.0).astype("float32"); x_te = (x_test / 255.0).astype("float32")
model = keras.Sequential([
    keras.layers.Input((28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation="softmax"),
])
model.summary(print_fn=print)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hist = model.fit(x_tr, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=2)
test_loss, test_acc = model.evaluate(x_te, y_test, verbose=0)
print(f"\nTest loss {test_loss:.4f}  |  Test accuracy {test_acc:.4f}")

# %% title: Training curves and sample predictions
fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
ax[0].plot(hist.history["loss"], label="train"); ax[0].plot(hist.history["val_loss"], label="val")
ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(hist.history["accuracy"], label="train"); ax[1].plot(hist.history["val_accuracy"], label="val")
ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout(); plt.show()
pred = model.predict(x_te[:10], verbose=0)
fig, axes = plt.subplots(1, 10, figsize=(12, 1.8))
for i, ax in enumerate(axes):
    ax.imshow(x_te[i], cmap="gray"); ax.axis("off")
    p = pred[i].argmax(); ax.set_title(f"{p}", color="g" if p == y_test[i] else "r")
plt.suptitle("Predictions (green = correct)"); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6. First generative step: sampling from distributions
# A generative model learns a distribution and then **samples** from it. Here we sample from a hand-built 2-D Gaussian
# mixture with TensorFlow Probability-style operations written in plain TensorFlow, and then use a latent noise vector
# fed through a (still untrained) network to see what "generation" means mechanically.

# %% title: Sampling from a Gaussian mixture and from a random generator network
tf.random.set_seed(7)
means = tf.constant([[-2., -2.], [2., 2.], [-2., 2.]])
comp = tf.random.categorical(tf.math.log([[0.4, 0.4, 0.2]]), 1500)[0]
samples = tf.gather(means, comp) + 0.5 * tf.random.normal((1500, 2))
print("sample batch shape:", samples.shape, "| empirical mean:", samples.numpy().mean(0).round(3))

z = tf.random.normal((16, 64))
gen = keras.Sequential([keras.layers.Input((64,)), keras.layers.Dense(256, activation="relu"),
                        keras.layers.Dense(784, activation="sigmoid"), keras.layers.Reshape((28, 28))])
imgs = gen(z).numpy()
print("untrained generator output:", imgs.shape, "(noise z of size 64 -> 28x28 image)")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
ax[0].scatter(*samples.numpy().T, s=4, c=comp.numpy(), cmap="viridis"); ax[0].set_title("Gaussian-mixture samples")
grid = np.concatenate([np.concatenate(imgs[r * 8:(r + 1) * 8], axis=1) for r in range(2)], axis=0)
ax[1].imshow(grid, cmap="gray"); ax[1].axis("off"); ax[1].set_title("Untrained generator: structured noise")
plt.tight_layout(); plt.show()
