# Aim

To visualise datasets, training behaviour, predictions and internal representations of AI models using Matplotlib,
Seaborn, PCA and t-SNE.

# Theory

Visualisation turns numbers into insight and is essential at every stage of building a model:

* **Data exploration:** class histograms, mean images, pixel histograms, correlation heat-maps and pair plots reveal
  imbalance, ranges and relationships between features.
* **Dimensionality reduction:** *PCA* projects data onto the directions of maximum variance (linear); *t-SNE* preserves
  local neighbourhoods in a non-linear 2-D map and is widely used to inspect embeddings and latent spaces.
* **Training monitoring:** loss/accuracy curves of the training and validation sets show under- and over-fitting.
* **Evaluation:** the confusion matrix $C_{ij}$ counts samples of true class $i$ predicted as $j$; precision, recall
  and F1 are derived from it. Confidence histograms show calibration.
* **Interpretability:** convolution filters, feature maps and *occlusion sensitivity* (the drop in class probability
  when a patch is masked) show what a CNN responds to.

# Steps

1. Load MNIST; plot the class distribution, sample images, the average image per class and the pixel histogram.
2. Load the Wine dataset; plot a correlation heat-map and a Seaborn pair plot.
3. Reduce 3000 digits to 2-D with PCA and t-SNE and compare the maps.
4. Train a small CNN and plot learning curves.
5. Plot the confusion matrix, the classification report, misclassified examples and the confidence distribution.
6. Visualise learned filters, feature maps, an occlusion-sensitivity map and a t-SNE of the learned features.

# Output

@@OUTPUT@@

# Conclusion

A range of visualisations was used to understand both the data and the model. The class plots showed an almost balanced
dataset, PCA vs t-SNE showed that non-linear embedding separates digit classes much better than a linear projection,
and the learning curves, confusion matrix and misclassified samples characterised the CNN's accuracy (98.65\% on the
test set) and its few confusions. Feature maps, occlusion maps and the t-SNE of learned features showed that the network
focuses on the digit strokes and separates the classes in its feature space.
