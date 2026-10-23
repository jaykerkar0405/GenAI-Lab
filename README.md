# Generative AI Lab

Thirteen Generative AI experiments implemented in Python and TensorFlow/Keras. Each `Experiment N` folder contains
the executed Jupyter notebook and the PDF implementation report (Aim, Theory, Steps, Code, Output, Conclusion).

| # | Experiment |
|---|------------|
| 1 | Introduction to Python and TensorFlow for Generative AI |
| 2 | Data Preprocessing for Generative AI Applications |
| 3 | Data Visualization for AI Models |
| 4 | Implementing a Generative Adversarial Network (GAN) for Image Generation |
| 5 | Fine-Tuning GAN Models for High-Resolution Image Generation |
| 6 | Variational Autoencoders (VAEs) for Image Generation |
| 7 | Text Generation using RNNs and LSTMs |
| 8 | Transformer-Based Models for Text Generation (GPT and BERT) |
| 9 | Music Generation using AI (LSTM and Transformers) |
| 10 | Deployment of a Generative AI Application using Gradio/Streamlit |
| 11 | Style Transfer Using Generative AI |
| 12 | Image Inpainting using GANs |
| 13 | Implementing Responsible AI and Bias Mitigation Techniques |

## Layout

```
Experiment N/   executed notebook + PDF report
src/            experiment sources (percent-format scripts, one per experiment)
docs/           report text (Aim, Theory, Steps, Conclusion) with an output placeholder
template/       pandoc LaTeX template and Lua filter for the report layout
tools/          build.py and the experiment list
```

## Rebuilding

Requires Python 3 with TensorFlow, nbformat/nbclient/ipykernel, plus pandoc and a LaTeX installation.

```
python3 tools/build.py N              # execute the notebook and build the PDF
python3 tools/build.py N --no-exec    # rebuild only the PDF from the saved notebook
```

Experiments ran on a 4-core CPU without a GPU, so the models are small. Notes: Experiment 8 trains a small GPT and BERT
from scratch (no pretrained weights were available offline); Experiment 5 uses a linear generator output because an
earlier `tanh` version collapsed during training.
