# Aim

To deploy a generative-AI model as an interactive web application using **Gradio** and **Streamlit**, and to test the running
applications through their API and in a browser.

# Theory

A trained model only becomes useful when others can use it. *Deployment* wraps the model in an application with a user
interface and a serving back-end:

* **Gradio** builds a web UI directly around a Python function (`gr.Interface` / `gr.Blocks`): components such as dropdowns,
  sliders and images are mapped to the function's arguments and results. Every Gradio app also exposes an HTTP/Python **API**
  (`gradio_client`), and it can be hosted on Hugging Face Spaces.
* **Streamlit** re-runs a Python script from top to bottom on every interaction; widgets (`st.selectbox`, `st.slider`,
  `st.button`) return their current values, and `@st.cache_resource` keeps the loaded model in memory between runs.
* Typical deployment steps: save the trained model (`.keras` file), write the app, pin the dependencies
  (`requirements.txt`), run the server, test it, and publish it.

The model used here is a **conditional VAE**: its decoder receives the latent vector $z$ and a one-hot class label $c$, so
$x=\mathrm{Dec}(z,c)$ can be generated for any requested digit; the latent noise scale controls the variety of styles.

# Steps

1. Train the conditional VAE on MNIST and save the decoder as `digit_decoder.keras`.
2. Check generation for all ten digits.
3. Write `app_gradio.py` (dropdown, two sliders, number input, image output) and `app_streamlit.py` (select box, sliders,
   button, image grid).
4. Start the Gradio server in the background, wait until it answers, and call it through the `gradio_client` API.
5. Start the Streamlit server and wait for its health check.
6. Use a headless Chromium browser (Playwright) to open each app, press the button and take a screenshot.
7. Shut the servers down.

# Output

@@OUTPUT@@

# Conclusion

A conditional VAE (final loss about 108 nats after 12 epochs) was trained, saved as an 830 KB \texttt{.keras} file and deployed as two web
applications. The Gradio server started and answered HTTP 200 in about 6 seconds, its API endpoint \texttt{/generate} returned a $448\times 224$
image of eight digit-7 samples in 0.4 s, and the Streamlit server passed its health check. Browser automation confirmed that a user can pick a digit,
set the number of samples and the style variation, press the button and see newly generated handwriting in both frameworks. Gradio is the
quickest way to expose a model function with a ready-made API, while Streamlit gives more freedom over page layout; either can be published
on a hosting service (Hugging Face Spaces, Streamlit Community Cloud).
