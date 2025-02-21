Convert a JAX Model to OpenVINO™ IR
===================================

`JAX <https://jax.readthedocs.io/en/latest>`__ is a Python library for
accelerator-oriented array computation and program transformation,
designed for high-performance numerical computing and large-scale
machine learning. JAX provides a familiar NumPy-style API for ease of
adoption by researchers and engineers.

In this tutorial we will show how to convert JAX
`ViT <https://github.com/google-research/vision_transformer?tab=readme-ov-file#available-vit-models>`__
and
`Mixer <https://github.com/google-research/vision_transformer?tab=readme-ov-file#mlp-mixer>`__
models in OpenVINO format.

.. raw:: html

   <details>

.. raw:: html

   <summary>

Click here for more detailed information about the models

.. raw:: html

   </summary>

Vision Transformer
~~~~~~~~~~~~~~~~~~

Overview of the model: authors split an image into fixed-size patches,
linearly embed each of them, add position embeddings, and feed the
resulting sequence of vectors to a standard Transformer encoder. In
order to perform classification, authors use the standard approach of
adding an extra learnable “classification token” to the sequence.

MLP-Mixer
~~~~~~~~~

MLP-Mixer (Mixer for short) consists of per-patch linear embeddings,
Mixer layers, and a classifier head. Mixer layers contain one
token-mixing MLP and one channel-mixing MLP, each consisting of two
fully-connected layers and a GELU nonlinearity. Other components
include: skip-connections, dropout, and linear classifier head.

.. raw:: html

   </details>


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Load and run the original model and a
   sample <#load-and-run-the-original-model-and-a-sample>`__
-  `Convert the model to OpenVINO
   IR <#convert-the-model-to-openvino-ir>`__
-  `Compiling the model <#compiling-the-model>`__
-  `Run OpenVINO model inference <#run-openvino-model-inference>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Prerequisites
-------------



.. code:: ipython3

    import requests
    from pathlib import Path
    
    if not Path("notebook_utilspy").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
        )
        open("notebook_utils.py", "w").write(r.text)
    
    if not Path("cmd_helper.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/cmd_helper.py",
        )
        open("cmd_helper.py", "w").write(r.text)
    
    # Read more about telemetry collection at https://github.com/openvinotoolkit/openvino_notebooks?tab=readme-ov-file#-telemetry
    from notebook_utils import collect_telemetry
    
    collect_telemetry("jax-classification-to-openvino.ipynb")

.. code:: ipython3

    from cmd_helper import clone_repo
    
    
    clone_repo("https://github.com/google-research/vision_transformer.git")

.. code:: ipython3

    %pip install --pre -Uq "openvino>=2024.5.0" --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
    %pip install -q "tensorflow-macos>=2.5 jax-metal>=0.4.2"; sys_platform == 'darwin' and platform_machine == 'arm64'" # macOS M1 and M2
    %pip install -q "tensorflow>=2.5 jax>=0.4.2"; sys_platform == 'darwin' and platform_machine != 'arm64'" # macOS x86
    %pip install -q "tensorflow-cpu>=2.5 jax>=0.4.2"; sys_platform != 'darwin'"
    %pip install -q Pillow "absl-py>=0.12.0" "flax>=0.6.4" "pandas>=1.1.0" tf_keras tqdm "einops>=0.3.0" "ml-collections>=0.1.0"

.. code:: ipython3

    import PIL
    import jax
    import numpy as np
    
    from vit_jax import checkpoint
    from vit_jax import models_vit
    from vit_jax import models_mixer
    from vit_jax.configs import models as models_config
    
    import openvino as ov

.. code:: ipython3

    import ipywidgets as widgets
    
    available_models = ["ViT-B_32", "Mixer-B_16"]
    
    
    model_to_use = widgets.Select(
        options=available_models,
        value=available_models[0],
        description="Select model:",
        disabled=False,
    )
    
    model_to_use




.. parsed-literal::

    Select(description='Select model:', options=('ViT-B_32', 'Mixer-B_16'), value='ViT-B_32')



Load and run the original model and a sample
--------------------------------------------



Download a pre-trained model.

.. code:: ipython3

    from notebook_utils import download_file
    
    
    model_name = model_to_use.value
    model_config = models_config.MODEL_CONFIGS[model_name]
    
    
    if model_name.startswith("Mixer"):
        # Download model trained on imagenet2012
        if not Path(f"{model_name}_imagenet2012.npz").exists():
            download_file(f"https://storage.googleapis.com/mixer_models/imagenet1k/{model_name}.npz", filename=f"{model_name}_imagenet2012.npz")
        model = models_mixer.MlpMixer(num_classes=1000, **model_config)
    else:
        # Download model pre-trained on imagenet21k and fine-tuned on imagenet2012.
        if not Path(f"{model_name}_imagenet2012.npz").exists():
            model_name_path = download_file(
                f"https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/{model_name}.npz", filename=f"{model_name}_imagenet2012.npz"
            )
        model = models_vit.VisionTransformer(num_classes=1000, **model_config)



.. parsed-literal::

    ViT-B_32_imagenet2012.npz:   0%|          | 0.00/337M [00:00<?, ?B/s]


Load and convert pretrained checkpoint.

.. code:: ipython3

    params = checkpoint.load(f"{model_name}_imagenet2012.npz")
    params["pre_logits"] = {}  # Need to restore empty leaf for Flax.

Get imagenet labels.

.. code:: ipython3

    from notebook_utils import download_file
    
    imagenet_labels_path = Path("ilsvrc2012_wordnet_lemmas.txt")
    if not imagenet_labels_path.exists():
        download_file("https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt")
    imagenet_labels = dict(enumerate(open(imagenet_labels_path)))



.. parsed-literal::

    ilsvrc2012_wordnet_lemmas.txt:   0%|          | 0.00/21.2k [00:00<?, ?B/s]


Get a random picture with the correct dimensions.

.. code:: ipython3

    resolution = 224 if model_name.startswith("Mixer") else 384
    url_224 = "https://github.com/user-attachments/assets/a9337f2b-20a5-4930-9fd1-75932154b285"
    url_384 = "https://github.com/user-attachments/assets/c07a0e72-b909-4521-b6f8-f22a7867071d"
    image_path = Path("img_{resolution}.jpg")
    if not image_path.exists():
        download_file(url_224 if resolution == 224 else url_384, filename="img_{resolution}.jpg")
    img = PIL.Image.open(image_path)



.. parsed-literal::

    picsum.jpg:   0%|          | 0.00/30.5k [00:00<?, ?B/s]


.. code:: ipython3

    img




.. image:: jax-classification-to-openvino-with-output_files/jax-classification-to-openvino-with-output_16_0.png



Run the original model inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # Predict on a batch with a single item
    data = (np.array(img) / 128 - 1)[None, ...]
    (logits,) = model.apply(dict(params=params), data, train=False)
    
    preds = np.array(jax.nn.softmax(logits))
    for idx in preds.argsort()[:-11:-1]:
        print(f"{preds[idx]:.5f} : {imagenet_labels[idx]}", end="")


.. parsed-literal::

    0.95251 : alp
    0.03884 : valley, vale
    0.00192 : cliff, drop, drop-off
    0.00173 : ski
    0.00059 : lakeside, lakeshore
    0.00049 : promontory, headland, head, foreland
    0.00036 : volcano
    0.00021 : snowmobile
    0.00017 : mountain_bike, all-terrain_bike, off-roader
    0.00017 : mountain_tent
    

Convert the model to OpenVINO IR
--------------------------------



OpenVINO supports JAX models via conversion to OpenVINO Intermediate
Representation (IR). `OpenVINO model conversion
API <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model>`__
should be used for these purposes. ``ov.convert_model`` function accepts
original JAX model instance and example input for tracing and returns
``ov.Model`` representing this model in OpenVINO framework. Converted
model can be used for saving on disk using ``ov.save_model`` function or
directly loading on device using ``core.complie_model``.

Before conversion we need to create the
`Jaxprs <https://jax.readthedocs.io/en/latest/key-concepts.html#jaxprs>`__
(JAX’s internal intermediate representation (IR) of programs) object by
tracing a Python function using the
`jax.make_jaxpr <https://jax.readthedocs.io/en/latest/_autosummary/jax.make_jaxpr.html>`__
function. [``jax.make_jaxpr``] take a function as argument, that should
perform the forward pass. In our case it is calling of ``model.apply``
method. But ``model.apply`` requires not only input data, but also
``params`` and keyword argument ``train=False`` in our case. To handle
it create a wrapper function ``model_apply`` that calls
``model.apply(params, x, train=False)``.

.. code:: ipython3

    from pathlib import Path
    
    
    model_path = Path(f"models/{model_name}.xml")
    
    
    def model_apply(x):
        return model.apply(dict(params=params), x, train=False)
    
    
    jaxpr = jax.make_jaxpr(model_apply)((np.array(img) / 128 - 1)[None, ...])
    
    converted_model = ov.convert_model(jaxpr)
    ov.save_model(converted_model, model_path)

Compiling the model
-------------------



Select device from dropdown list for running inference using OpenVINO.

.. code:: ipython3

    from notebook_utils import device_widget
    
    
    core = ov.Core()
    
    device = device_widget()
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    compiled_model = core.compile_model(model_path, device.value)

Run OpenVINO model inference
----------------------------

.. code:: ipython3

    (logits_ov,) = list(compiled_model(data).values())[0]
    
    preds = np.array(jax.nn.softmax(logits_ov))
    for idx in preds.argsort()[:-11:-1]:
        print(f"{preds[idx]:.5f} : {imagenet_labels[idx]}", end="")


.. parsed-literal::

    0.95255 : alp
    0.03881 : valley, vale
    0.00192 : cliff, drop, drop-off
    0.00173 : ski
    0.00059 : lakeside, lakeshore
    0.00049 : promontory, headland, head, foreland
    0.00036 : volcano
    0.00021 : snowmobile
    0.00017 : mountain_bike, all-terrain_bike, off-roader
    0.00017 : mountain_tent
    

.. code:: ipython3

    # Cleanup
    # %pip uninstall -q -y "tensorflow-cpu" tensorflow tf_keras
