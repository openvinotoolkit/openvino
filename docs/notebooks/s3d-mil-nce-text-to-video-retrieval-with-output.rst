Text-to-Video retrieval with S3D MIL-NCE and OpenVINO
=====================================================

This tutorial based on `the TensorFlow
tutorial <https://www.tensorflow.org/hub/tutorials/text_to_video_retrieval_with_s3d_milnce>`__
that demonstrates how to use the `S3D
MIL-NCE <https://tfhub.dev/deepmind/mil-nce/s3d/1>`__ model from
TensorFlow Hub to do text-to-video retrieval to find the most similar
videos for a given text query.

MIL-NCE inherits from Multiple Instance Learning (MIL) and Noise
Contrastive Estimation (NCE). The method is capable of addressing
visually misaligned narrations from uncurated instructional videos. Two
model variations are available with different 3D CNN backbones: I3D and
S3D. In this tutorial we use S3D variation. More details about the
training and the model can be found in `End-to-End Learning of Visual
Representations from Uncurated Instructional
Videos <https://arxiv.org/abs/1912.06430>`__ paper.

This tutorial demonstrates step-by-step instructions on how to run and
optimize S3D MIL-NCE model with OpenVINO. An additional part
demonstrates how to run quantization with
`NNCF <https://github.com/openvinotoolkit/nncf/>`__ to speed up the
inference.

The tutorial consists of the following steps:


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `The original inference <#the-original-inference>`__
-  `Convert the model to OpenVINO
   IR <#convert-the-model-to-openvino-ir>`__
-  `Compiling models <#compiling-models>`__
-  `Inference <#inference>`__
-  `Optimize model using NNCF Post-training Quantization
   API <#optimize-model-using-nncf-post-training-quantization-api>`__

   -  `Prepare dataset <#prepare-dataset>`__
   -  `Perform model quantization <#perform-model-quantization>`__

-  `Run quantized model inference <#run-quantized-model-inference>`__
    


This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Prerequisites
-------------



.. code:: ipython3

    import platform
    
    %pip install -Uq pip
    %pip install --upgrade --pre openvino-tokenizers "openvino>=2024.2.0" --extra-index-url "https://storage.openvinotoolkit.org/simple/wheels/nightly"
    %pip install -q "tensorflow-macos>=2.5; sys_platform == 'darwin' and platform_machine == 'arm64' and python_version > '3.8'" # macOS M1 and M2
    %pip install -q "tensorflow>=2.5; sys_platform == 'darwin' and platform_machine != 'arm64' and python_version > '3.8'" # macOS x86
    %pip install -q "tensorflow>=2.5; sys_platform != 'darwin' and python_version > '3.8'"
    
    %pip install -q tensorflow_hub tf_keras numpy "opencv-python" "nncf>=2.10.0"
    if platform.system() != "Windows":
        %pip install -q "matplotlib>=3.4"
    else:
        %pip install -q "matplotlib>=3.4,<3.7"

.. code:: ipython3

    import os
    from pathlib import Path
    
    import tensorflow as tf
    import tensorflow_hub as hub
    
    import numpy as np
    import cv2
    from IPython import display
    import math
    
    os.environ["TFHUB_CACHE_DIR"] = str(Path("./tfhub_modules").resolve())

Download the model

.. code:: ipython3

    hub_handle = "https://www.kaggle.com/models/deepmind/mil-nce/TensorFlow1/s3d/1"
    hub_model = hub.load(hub_handle)

The model has 2 signatures, one for generating video embeddings and one
for generating text embeddings. We will use these embedding to find the
nearest neighbors in the embedding space as in the original tutorial.
Below we will define auxiliary functions

.. code:: ipython3

    def generate_embeddings(model, input_frames, input_words):
        """Generate embeddings from the model from video frames and input words."""
        # Input_frames must be normalized in [0, 1] and of the shape Batch x T x H x W x 3
        vision_output = model.signatures["video"](tf.constant(tf.cast(input_frames, dtype=tf.float32)))
        text_output = model.signatures["text"](tf.constant(input_words))
    
        return vision_output["video_embedding"], text_output["text_embedding"]

.. code:: ipython3

    # @title Define video loading and visualization functions  { display-mode: "form" }
    
    
    # Utilities to open video files using CV2
    def crop_center_square(frame):
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]
    
    
    def load_video(video_url, max_frames=32, resize=(224, 224)):
        path = tf.keras.utils.get_file(os.path.basename(video_url)[-128:], video_url)
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)
    
                if len(frames) == max_frames:
                    break
        finally:
            cap.release()
        frames = np.array(frames)
        if len(frames) < max_frames:
            n_repeat = int(math.ceil(max_frames / float(len(frames))))
            frames = frames.repeat(n_repeat, axis=0)
        frames = frames[:max_frames]
        return frames / 255.0
    
    
    def display_video(urls):
        html = "<table>"
        html += "<tr><th>Video 1</th><th>Video 2</th><th>Video 3</th></tr><tr>"
        for url in urls:
            html += "<td>"
            html += '<img src="{}" height="224">'.format(url)
            html += "</td>"
        html += "</tr></table>"
        return display.HTML(html)
    
    
    def display_query_and_results_video(query, urls, scores):
        """Display a text query and the top result videos and scores."""
        sorted_ix = np.argsort(-scores)
        html = ""
        html += "<h2>Input query: <i>{}</i> </h2><div>".format(query)
        html += "Results: <div>"
        html += "<table>"
        html += "<tr><th>Rank #1, Score:{:.2f}</th>".format(scores[sorted_ix[0]])
        html += "<th>Rank #2, Score:{:.2f}</th>".format(scores[sorted_ix[1]])
        html += "<th>Rank #3, Score:{:.2f}</th></tr><tr>".format(scores[sorted_ix[2]])
        for i, idx in enumerate(sorted_ix):
            url = urls[sorted_ix[i]]
            html += "<td>"
            html += '<img src="{}" height="224">'.format(url)
            html += "</td>"
        html += "</tr></table>"
    
        return html

.. code:: ipython3

    # @title Load example videos and define text queries  { display-mode: "form" }
    
    video_1_url = "https://upload.wikimedia.org/wikipedia/commons/b/b0/YosriAirTerjun.gif"  # @param {type:"string"}
    video_2_url = "https://upload.wikimedia.org/wikipedia/commons/e/e6/Guitar_solo_gif.gif"  # @param {type:"string"}
    video_3_url = "https://upload.wikimedia.org/wikipedia/commons/3/30/2009-08-16-autodrift-by-RalfR-gif-by-wau.gif"  # @param {type:"string"}
    
    video_1 = load_video(video_1_url)
    video_2 = load_video(video_2_url)
    video_3 = load_video(video_3_url)
    all_videos = [video_1, video_2, video_3]
    
    query_1_video = "waterfall"  # @param {type:"string"}
    query_2_video = "playing guitar"  # @param {type:"string"}
    query_3_video = "car drifting"  # @param {type:"string"}
    all_queries_video = [query_1_video, query_2_video, query_3_video]
    all_videos_urls = [video_1_url, video_2_url, video_3_url]
    display_video(all_videos_urls)




.. raw:: html

    <table><tr><th>Video 1</th><th>Video 2</th><th>Video 3</th></tr><tr><td><img src="https://upload.wikimedia.org/wikipedia/commons/b/b0/YosriAirTerjun.gif" height="224"></td><td><img src="https://upload.wikimedia.org/wikipedia/commons/e/e6/Guitar_solo_gif.gif" height="224"></td><td><img src="https://upload.wikimedia.org/wikipedia/commons/3/30/2009-08-16-autodrift-by-RalfR-gif-by-wau.gif" height="224"></td></tr></table>



The original inference
----------------------



.. code:: ipython3

    # Prepare video inputs.
    videos_np = np.stack(all_videos, axis=0)
    
    # Prepare text input.
    words_np = np.array(all_queries_video)
    
    # Generate the video and text embeddings.
    video_embd, text_embd = generate_embeddings(hub_model, videos_np, words_np)
    
    # Scores between video and text is computed by dot products.
    all_scores = np.dot(text_embd, tf.transpose(video_embd))

.. code:: ipython3

    # Display results.
    html = ""
    for i, words in enumerate(words_np):
        html += display_query_and_results_video(words, all_videos_urls, all_scores[i, :])
        html += "<br>"
    display.HTML(html)




.. raw:: html

    <h2>Input query: <i>waterfall</i> </h2><div>Results: <div><table><tr><th>Rank #1, Score:4.71</th><th>Rank #2, Score:-1.63</th><th>Rank #3, Score:-4.17</th></tr><tr><td><img src="https://upload.wikimedia.org/wikipedia/commons/b/b0/YosriAirTerjun.gif" height="224"></td><td><img src="https://upload.wikimedia.org/wikipedia/commons/3/30/2009-08-16-autodrift-by-RalfR-gif-by-wau.gif" height="224"></td><td><img src="https://upload.wikimedia.org/wikipedia/commons/e/e6/Guitar_solo_gif.gif" height="224"></td></tr></table><br><h2>Input query: <i>playing guitar</i> </h2><div>Results: <div><table><tr><th>Rank #1, Score:6.50</th><th>Rank #2, Score:-1.79</th><th>Rank #3, Score:-2.67</th></tr><tr><td><img src="https://upload.wikimedia.org/wikipedia/commons/e/e6/Guitar_solo_gif.gif" height="224"></td><td><img src="https://upload.wikimedia.org/wikipedia/commons/b/b0/YosriAirTerjun.gif" height="224"></td><td><img src="https://upload.wikimedia.org/wikipedia/commons/3/30/2009-08-16-autodrift-by-RalfR-gif-by-wau.gif" height="224"></td></tr></table><br><h2>Input query: <i>car drifting</i> </h2><div>Results: <div><table><tr><th>Rank #1, Score:8.78</th><th>Rank #2, Score:-1.07</th><th>Rank #3, Score:-2.17</th></tr><tr><td><img src="https://upload.wikimedia.org/wikipedia/commons/3/30/2009-08-16-autodrift-by-RalfR-gif-by-wau.gif" height="224"></td><td><img src="https://upload.wikimedia.org/wikipedia/commons/b/b0/YosriAirTerjun.gif" height="224"></td><td><img src="https://upload.wikimedia.org/wikipedia/commons/e/e6/Guitar_solo_gif.gif" height="224"></td></tr></table><br>



Convert the model to OpenVINO IR
--------------------------------

OpenVINO supports TensorFlow
models via conversion into Intermediate Representation (IR) format. We
need to provide a model object, input data for model tracing to
``ov.convert_model`` function to obtain OpenVINO ``ov.Model`` object
instance. Model can be saved on disk for next deployment using
``ov.save_model`` function.

.. code:: ipython3

    import openvino_tokenizers  # NOQA Need to import conversion and operation extensions
    import openvino as ov
    
    model_path = hub.resolve(hub_handle)
    # infer on random data
    images_data = np.random.rand(3, 32, 224, 224, 3).astype(np.float32)
    words_data = np.array(["First sentence", "Second one", "Abracadabra"], dtype=str)
    
    ov_model = ov.convert_model(model_path, input=[("words", [3]), ("images", [3, 32, 224, 224, 3])])

Compiling models
----------------



Only CPU is supported for this model due to strings as input.

.. code:: ipython3

    core = ov.Core()
    
    compiled_model = core.compile_model(ov_model, device_name="CPU")

Inference
---------



.. code:: ipython3

    # Redefine `generate_embeddings` function to make it possible to use the compile IR model.
    def generate_embeddings(model, input_frames, input_words):
        """Generate embeddings from the model from video frames and input words."""
        # Input_frames must be normalized in [0, 1] and of the shape Batch x T x H x W x 3
        output = compiled_model({"words": input_words, "images": tf.cast(input_frames, dtype=tf.float32)})
    
        return output["video_embedding"], output["text_embedding"]

.. code:: ipython3

    # Generate the video and text embeddings.
    video_embd, text_embd = generate_embeddings(compiled_model, videos_np, words_np)
    
    # Scores between video and text is computed by dot products.
    all_scores = np.dot(text_embd, tf.transpose(video_embd))

.. code:: ipython3

    # Display results.
    html = ""
    for i, words in enumerate(words_np):
        html += display_query_and_results_video(words, all_videos_urls, all_scores[i, :])
        html += "<br>"
    display.HTML(html)




.. raw:: html

    <h2>Input query: <i>waterfall</i> </h2><div>Results: <div><table><tr><th>Rank #1, Score:4.71</th><th>Rank #2, Score:-1.63</th><th>Rank #3, Score:-4.17</th></tr><tr><td><img src="https://upload.wikimedia.org/wikipedia/commons/b/b0/YosriAirTerjun.gif" height="224"></td><td><img src="https://upload.wikimedia.org/wikipedia/commons/3/30/2009-08-16-autodrift-by-RalfR-gif-by-wau.gif" height="224"></td><td><img src="https://upload.wikimedia.org/wikipedia/commons/e/e6/Guitar_solo_gif.gif" height="224"></td></tr></table><br><h2>Input query: <i>playing guitar</i> </h2><div>Results: <div><table><tr><th>Rank #1, Score:6.50</th><th>Rank #2, Score:-1.79</th><th>Rank #3, Score:-2.67</th></tr><tr><td><img src="https://upload.wikimedia.org/wikipedia/commons/e/e6/Guitar_solo_gif.gif" height="224"></td><td><img src="https://upload.wikimedia.org/wikipedia/commons/b/b0/YosriAirTerjun.gif" height="224"></td><td><img src="https://upload.wikimedia.org/wikipedia/commons/3/30/2009-08-16-autodrift-by-RalfR-gif-by-wau.gif" height="224"></td></tr></table><br><h2>Input query: <i>car drifting</i> </h2><div>Results: <div><table><tr><th>Rank #1, Score:8.78</th><th>Rank #2, Score:-1.07</th><th>Rank #3, Score:-2.17</th></tr><tr><td><img src="https://upload.wikimedia.org/wikipedia/commons/3/30/2009-08-16-autodrift-by-RalfR-gif-by-wau.gif" height="224"></td><td><img src="https://upload.wikimedia.org/wikipedia/commons/b/b0/YosriAirTerjun.gif" height="224"></td><td><img src="https://upload.wikimedia.org/wikipedia/commons/e/e6/Guitar_solo_gif.gif" height="224"></td></tr></table><br>



Optimize model using NNCF Post-training Quantization API
--------------------------------------------------------



`NNCF <https://github.com/openvinotoolkit/nncf>`__ provides a suite of
advanced algorithms for Neural Networks inference optimization in
OpenVINO with minimal accuracy drop. We will use 8-bit quantization in
post-training mode (without the fine-tuning pipeline). The optimization
process contains the following steps:

1. Create a Dataset for quantization.
2. Run ``nncf.quantize`` for getting an optimized model.
3. Serialize an OpenVINO IR model, using the ``ov.save_model`` function.

Prepare dataset
~~~~~~~~~~~~~~~



This model doesn’t require a big dataset for calibration. We will use
only example videos for this purpose. NNCF provides ``nncf.Dataset``
wrapper for using native framework dataloaders in quantization pipeline.
Additionally, we specify transform function that will be responsible for
preparing input data in model expected format.

.. code:: ipython3

    import nncf
    
    dataset = nncf.Dataset(((words_np, tf.cast(videos_np, dtype=tf.float32)),))


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, openvino
    

Perform model quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~



The ``nncf.quantize`` function provides an interface for model
quantization. It requires an instance of the OpenVINO Model and
quantization dataset. Optionally, some additional parameters for the
configuration quantization process (number of samples for quantization,
preset, ignored scope etc.) can be provided.

.. code:: ipython3

    MODEL_DIR = Path("model/")
    MODEL_DIR.mkdir(exist_ok=True)
    
    quantized_model_path = MODEL_DIR / "quantized_model.xml"
    
    
    if not quantized_model_path.exists():
        quantized_model = nncf.quantize(model=ov_model, calibration_dataset=dataset, model_type=nncf.ModelType.TRANSFORMER)
        ov.save_model(quantized_model, quantized_model_path)



.. parsed-literal::

    Output()






    







    



.. parsed-literal::

    Output()






    







    


.. parsed-literal::

    INFO:nncf:39 ignored nodes were found by name in the NNCFGraph
    


.. parsed-literal::

    Output()






    







    



.. parsed-literal::

    Output()






    







    


Run quantized model inference
-----------------------------



There are no changes in model usage after applying quantization. Let’s
check the model work on the previously used example.

.. code:: ipython3

    int8_model = core.compile_model(quantized_model_path, device_name="CPU")

.. code:: ipython3

    # Generate the video and text embeddings.
    video_embd, text_embd = generate_embeddings(int8_model, videos_np, words_np)
    
    # Scores between video and text is computed by dot products.
    all_scores = np.dot(text_embd, tf.transpose(video_embd))

.. code:: ipython3

    # Display results.
    html = ""
    for i, words in enumerate(words_np):
        html += display_query_and_results_video(words, all_videos_urls, all_scores[i, :])
        html += "<br>"
    display.HTML(html)




.. raw:: html

    <h2>Input query: <i>waterfall</i> </h2><div>Results: <div><table><tr><th>Rank #1, Score:4.71</th><th>Rank #2, Score:-1.63</th><th>Rank #3, Score:-4.17</th></tr><tr><td><img src="https://upload.wikimedia.org/wikipedia/commons/b/b0/YosriAirTerjun.gif" height="224"></td><td><img src="https://upload.wikimedia.org/wikipedia/commons/3/30/2009-08-16-autodrift-by-RalfR-gif-by-wau.gif" height="224"></td><td><img src="https://upload.wikimedia.org/wikipedia/commons/e/e6/Guitar_solo_gif.gif" height="224"></td></tr></table><br><h2>Input query: <i>playing guitar</i> </h2><div>Results: <div><table><tr><th>Rank #1, Score:6.50</th><th>Rank #2, Score:-1.79</th><th>Rank #3, Score:-2.67</th></tr><tr><td><img src="https://upload.wikimedia.org/wikipedia/commons/e/e6/Guitar_solo_gif.gif" height="224"></td><td><img src="https://upload.wikimedia.org/wikipedia/commons/b/b0/YosriAirTerjun.gif" height="224"></td><td><img src="https://upload.wikimedia.org/wikipedia/commons/3/30/2009-08-16-autodrift-by-RalfR-gif-by-wau.gif" height="224"></td></tr></table><br><h2>Input query: <i>car drifting</i> </h2><div>Results: <div><table><tr><th>Rank #1, Score:8.78</th><th>Rank #2, Score:-1.07</th><th>Rank #3, Score:-2.17</th></tr><tr><td><img src="https://upload.wikimedia.org/wikipedia/commons/3/30/2009-08-16-autodrift-by-RalfR-gif-by-wau.gif" height="224"></td><td><img src="https://upload.wikimedia.org/wikipedia/commons/b/b0/YosriAirTerjun.gif" height="224"></td><td><img src="https://upload.wikimedia.org/wikipedia/commons/e/e6/Guitar_solo_gif.gif" height="224"></td></tr></table><br>


