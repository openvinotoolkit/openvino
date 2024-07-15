CLIP model with Jina CLIP and OpenVINO
--------------------------------------

`jina-clip-v1 <https://huggingface.co/jinaai/jina-clip-v1>`__ is a
state-of-the-art English multimodal(text-image) embedding model trained
by `Jina AI <https://aimodels.fyi/creators/huggingFace/jinaai>`__. It
bridges the gap between traditional text embedding models, which excel
in text-to-text retrieval but are incapable of cross-modal tasks, and
models that effectively align image and text embeddings but are not
optimized for text-to-text retrieval. jina-clip-v1 offers robust
performance in both domains. Its dual capability makes it an excellent
tool for multimodal retrieval-augmented generation (MuRAG) applications,
allowing seamless text-to-text and text-to-image searches within a
single model. jina-clip-v1 can be used for a variety of multimodal
applications, such as: image search by describing them in text,
multimodal question answering, multimodal content generation. Jina AI
has also provided the Embeddings API as an easy-to-use interface for
working with jina-clip-v1 and their other embedding models.

In this notebook we will load the model with Hugging Face Transformers,
convert it to OpenVINO IR format, optimize it with NNCF and show the
life demo.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prerequisites <#Prerequisites>`__
-  `Instantiate model <#Instantiate-model>`__

   -  `Prepare input data <#Prepare-input-data>`__
   -  `Run PyTorch model inference <#Run-PyTorch-model-inference>`__

-  `Run OpenVINO model inference <#Run-OpenVINO-model-inference>`__

   -  `Prepare input data <#Prepare-input-data>`__
   -  `Convert Model to OpenVINO IR
      format <#Convert-Model-to-OpenVINO-IR-format>`__
   -  `Select inference device <#Select-inference-device>`__
   -  `Compile model and run
      inference <#Compile-model-and-run-inference>`__

-  `Quantize model to INT8 using
   NNCF <#Quantize-model-to-INT8-using-NNCF>`__

   -  `Prepare datasets <#Prepare-datasets>`__

      -  `Dataset with text data <#Dataset-with-text-data>`__
      -  `Dataset with image data <#Dataset-with-image-data>`__

   -  `Perform quantization <#Perform-quantization>`__

      -  `Quantization of text model <#Quantization-of-text-model>`__
      -  `Quantization of image model <#Quantization-of-image-model>`__

   -  `Compare File Size <#Compare-File-Size>`__
   -  `Compare inference time of the FP16 IR and quantized
      models <#Compare-inference-time-of-the-FP16-IR-and-quantized-models>`__

-  `Gradio demo <#Gradio-demo>`__

Prerequisites
-------------

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    %pip install -q "openvino>=2024.2.0" "datasets>=2.20" "nncf>=2.11.0"
    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu "gradio>=4.19" "pillow" "einops" "timm" "transformers[torch]>=4.39" "torch>=2.1"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


Instantiate model
-----------------

`back to top ⬆️ <#Table-of-contents:>`__

Let’s load the
`jinaai/jina-clip-v1 <https://huggingface.co/jinaai/jina-clip-v1>`__
with Hugging Face Transformers. We creates PyTorch model class instance
with ``AutoModel``, load and initialize it with model configuration and
weights, using ``from_pretrained`` method.

.. code:: ipython3

    from transformers import AutoModel
    
    model = AutoModel.from_pretrained("jinaai/jina-clip-v1", trust_remote_code=True)


.. parsed-literal::

    2024-07-13 00:37:54.543689: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-07-13 00:37:54.579011: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-07-13 00:37:55.252367: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


Prepare input data
~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

The model can encode meaningful sentences in English as text input.
Image could be provided to model as local file path, URLs or directly
passing in the PIL.Image objects.

.. code:: ipython3

    from PIL import Image
    import requests
    
    # image input data
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    
    open("notebook_utils.py", "w").write(r.text)
    from notebook_utils import download_file
    
    download_file(
        "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/3f779fc1-c1b2-4dec-915a-64dae510a2bb",
        "furseal.png",
        directory="data",
    )
    
    img_furseal = Image.open("./data/furseal.png")
    
    image_path = download_file(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
        directory="data",
    )
    
    img_coco = Image.open("./data/coco.jpg")
    
    IMAGE_INPUTS = [img_furseal, img_coco]
    
    # text input data
    TEXT_INPUTS = ["Seal", "Cobra", "Rat", "Penguin", "Dog"]



.. parsed-literal::

    data/furseal.png:   0%|          | 0.00/2.55M [00:00<?, ?B/s]



.. parsed-literal::

    data/coco.jpg:   0%|          | 0.00/202k [00:00<?, ?B/s]


.. code:: ipython3

    from typing import List
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from scipy.special import softmax
    
    
    def calc_simularity_softmax(embeddings1, embeddings2, apply_softmax=True):
        simularity = []
        for emb1 in embeddings1:
            temp_simularity = []
            for emb2 in embeddings2:
                temp_simularity.append(emb1 @ emb2)
            temp_simularity = softmax(temp_simularity) if apply_softmax else temp_simularity
            simularity.append(temp_simularity)
    
        return simularity
    
    
    def visionize_result(image: Image, labels: List[str], probs: np.ndarray, top: int = 5):
        """
        Utility function for visionization classification results
        params:
          image: input image
          labels: list of classification labels
          probs: model predicted softmaxed probabilities for each label
          top: number of the highest probability results for visionization
        returns:
          None
        """
        plt.figure(figsize=(64, 64))
        top_labels = np.argsort(-probs)[: min(top, probs.shape[0])]
        top_probs = probs[top_labels]
        plt.subplot(8, 8, 1)
        plt.imshow(image)
        plt.axis("off")
    
        plt.subplot(8, 8, 2)
        y = np.arange(top_probs.shape[-1])
        plt.grid()
        plt.barh(y, top_probs)
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [labels[index] for index in top_labels])
        plt.xlabel("simularity")

We will use tokenizer and preprocess from jina-clip model. We will take
``tokenizer`` to encode text input data using ``model.get_tokenizer()``
and take ``preprocess`` for image data using ``model.get_preprocess()``.

.. code:: ipython3

    tokenizer = model.get_tokenizer()
    
    tokenizer_kwargs = dict()
    tokenizer_kwargs["padding"] = "max_length"
    tokenizer_kwargs["max_length"] = 512
    tokenizer_kwargs["truncation"] = True
    
    text_inputs = tokenizer(
        TEXT_INPUTS,
        return_tensors="pt",
        **tokenizer_kwargs,
    ).to("cpu")
    
    
    processor = model.get_preprocess()
    vision_inputs = processor(images=IMAGE_INPUTS, return_tensors="pt")

Run PyTorch model inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    text_embeddings = model.text_model(text_inputs["input_ids"])
    image_embeddings = model.vision_model(vision_inputs["pixel_values"])
    
    res = calc_simularity_softmax(image_embeddings.detach().numpy(), text_embeddings.detach().numpy())
    visionize_result(img_furseal, TEXT_INPUTS, np.array(res[0]))



.. image:: jina-clip-with-output_files/jina-clip-with-output_11_0.png


Run OpenVINO model inference
----------------------------

`back to top ⬆️ <#Table-of-contents:>`__

Convert Model to OpenVINO IR format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

OpenVINO supports PyTorch models via conversion to OpenVINO Intermediate
Representation (IR). OpenVINO model conversion API should be used for
these purposes. ``ov.convert_model`` function accepts original PyTorch
model instance and example input for tracing and returns ``ov.Model``
representing this model in OpenVINO framework. Converted model can be
used for saving on disk using ``ov.save_model`` function or directly
loading on device using ``core.complie_model``.

.. code:: ipython3

    import openvino as ov
    from pathlib import Path
    
    core = ov.Core()

.. code:: ipython3

    fp16_text_model_path = Path("jina-clip-text_v1_fp16.xml")
    
    if not fp16_text_model_path.exists():
        ov_text_model = ov.convert_model(model.text_model, example_input=text_inputs["input_ids"])
        ov.save_model(ov_text_model, fp16_text_model_path)


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4565: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(
    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/jinaai/jina-bert-flash-implementation/b78d1595de294f13ffe7b19d6cd63892a6e4e7a4/mha.py:333: TracerWarning: Converting a tensor to a Python float might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/jinaai/jina-bert-flash-implementation/b78d1595de294f13ffe7b19d6cd63892a6e4e7a4/mha.py:343: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if seqlen > self.linear_biases.shape[-1]:


.. code:: ipython3

    fp16_vision_model_path = Path("jina-clip-vision_v1_fp16.xml")
    
    if not fp16_vision_model_path.exists():
        ov_vision_model = ov.convert_model(model.vision_model, example_input=vision_inputs["pixel_values"])
        ov.save_model(ov_vision_model, fp16_vision_model_path)


.. parsed-literal::

    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/jinaai/jina-clip-implementation/952897b38094b9f6a47b3d9a1d8239523e374098/eva_model.py:468: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert H == self.img_size[0] and W == self.img_size[1], (


Select inference device
~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

For starting work, please select inference device from dropdown list.

.. code:: ipython3

    import ipywidgets as widgets
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value="AUTO",
        description="Device:",
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Compile model and run inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    compiled_text_model = core.compile_model(fp16_text_model_path, device.value)
    compiled_vision_model = core.compile_model(fp16_vision_model_path, device.value)

.. code:: ipython3

    text_ov_res = compiled_text_model(text_inputs["input_ids"])
    vis_ov_res = compiled_vision_model(vision_inputs["pixel_values"])
    
    res = calc_simularity_softmax(vis_ov_res[0], text_ov_res[0])
    visionize_result(img_furseal, TEXT_INPUTS, np.array(res[0]))



.. image:: jina-clip-with-output_files/jina-clip-with-output_21_0.png


Quantize model to INT8 using NNCF
---------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

Lets speed up the model by applying 8-bit post-training quantization
from `NNCF <https://github.com/openvinotoolkit/nncf/>`__ (Neural Network
Compression Framework) and infer quantized model via OpenVINO™ Toolkit.
`NNCF <https://github.com/openvinotoolkit/nncf/>`__ enables
post-training quantization by adding quantization layers into model
graph and then using a subset of the training dataset to initialize the
parameters of these additional quantization layers. Quantized operations
are executed in ``INT8`` instead of ``FP32``/``FP16`` making model
inference faster. The optimization process contains the following steps:

1. Prepare quantization dataset
2. Quantize the converted OpenVINO model with NNCF with
   ``nncf.quantize()``.
3. Save the ``INT8`` model using ``openvino.save_model()`` function.
4. Compare model size of converted and quantized models.
5. Compare performance of converted and quantized models.

..

   **Note:** quantization process may require additional time and memory
   for performing. You can disable it using widget below:

.. code:: ipython3

    to_quantize = widgets.Checkbox(
        value=True,
        description="Quantization",
        disabled=False,
    )
    
    to_quantize




.. parsed-literal::

    Checkbox(value=True, description='Quantization')



.. code:: ipython3

    # Fetch `skip_kernel_extension` module
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
    )
    open("skip_kernel_extension.py", "w").write(r.text)
    
    %load_ext skip_kernel_extension

Prepare datasets
~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

The `Conceptual
Captions <https://ai.google.com/research/ConceptualCaptions/>`__ dataset
consisting of ~3.3M images annotated with captions is used to quantize
model.

Dataset with text data
^^^^^^^^^^^^^^^^^^^^^^

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    %%skip not $to_quantize.value
    
    import torch
    from datasets import load_dataset
    from tqdm.notebook import tqdm
    import requests
    from io import BytesIO
    import numpy as np
    from PIL import Image
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    
    
    def check_text_data(data):
        """
        Check if the given data is text-based.
        """
        if isinstance(data, str):
            return True
        if isinstance(data, list):
            return all(isinstance(x, str) for x in data)
        return False
    
    
    def collate_fn_text(example, text_column="caption"):
        """
        Preprocesses an example by loading and transforming text data.
        Checks if the text data in the example is valid by calling the `check_text_data` function.
        If there is any error during the download process, returns None.
        Returns the preprocessed inputs with transformed image and text data.
        """
        assert len(example) == 1
        example = example[0]
    
        if not check_text_data(example[text_column]):
            raise ValueError("Text data is not valid")
    
        text_input = tokenizer(
            example[text_column],
            return_tensors='pt',
            **tokenizer_kwargs)
    
        return text_input
    
    
    def prepare_calibration_data_text(dataloader, init_steps):
        """
        This function prepares calibration data from a dataloader for a specified number of initialization steps.
        It iterates over the dataloader, fetching batches and storing the relevant data.
        """
        data = []
        print(f"Fetching {init_steps} samples for the initialization...")
        with tqdm(total=init_steps) as pbar:
            for batch in dataloader:
                if len(data) == init_steps:
                    break
                if batch:
                    pbar.update(1)
                    with torch.no_grad():
                        data.append(batch["input_ids"].to("cpu"))
        return data

.. code:: ipython3

    %%skip not $to_quantize.value
    
    import logging
    import nncf
    
    dataset = load_dataset("google-research-datasets/conceptual_captions", trust_remote_code=True)
    train_dataset = dataset["train"].shuffle(seed=42)
    
    dataloader_text = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn_text, batch_size=1)
    calibration_data_text = prepare_calibration_data_text(dataloader_text, 50)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino
    Fetching 50 samples for the initialization...



.. parsed-literal::

      0%|          | 0/50 [00:00<?, ?it/s]


Dataset with image data
^^^^^^^^^^^^^^^^^^^^^^^

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    %%skip not $to_quantize.value
    
    
    def get_pil_from_url(url):
        """
        Downloads and converts an image from a URL to a PIL Image object.
        """
        response = requests.get(url, verify=False, timeout=20)
        image = Image.open(BytesIO(response.content))
        return image.convert("RGB")
    
    
    def collate_fn_vision(example, image_column="image_url"):
        """
        Preprocesses an example by loading and transforming image data.
        Downloads the image specified by the URL in the image_column by calling the `get_pil_from_url` function.
        If there is any error during the download process, returns None.
        Returns the preprocessed inputs with transformed image and text data.
        """
        assert len(example) == 1
        example = example[0]
    
        url = example[image_column]
        try:
            image = get_pil_from_url(url)
            h, w = image.size
            if h == 1 or w == 1:
                return None
        except Exception:
            return None
    
        vision_input = processor(images=[image])
        return vision_input
    
    
    def prepare_calibration_data_vis(dataloader, init_steps):
        """
        This function prepares calibration data from a dataloader for a specified number of initialization steps.
        It iterates over the dataloader, fetching batches and storing the relevant data.
        """
        data = []
        print(f"Fetching {init_steps} samples for the initialization...")
        with tqdm(total=init_steps) as pbar:
            for batch in dataloader:
                if len(data) == init_steps:
                    break
                if batch:
                    pbar.update(1)
                    with torch.no_grad():
                        data.append(batch["pixel_values"].to("cpu"))
        return data

.. code:: ipython3

    %%skip not $to_quantize.value
    
    dataset = load_dataset("google-research-datasets/conceptual_captions", trust_remote_code=True)
    train_dataset = dataset["train"].shuffle(seed=42)
    
    dataloader_vis = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn_vision, batch_size=1)
    calibration_data_vision = prepare_calibration_data_vis(dataloader_vis, 50)


.. parsed-literal::

    Fetching 50 samples for the initialization...



.. parsed-literal::

      0%|          | 0/50 [00:00<?, ?it/s]


Perform quantization
~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

Create a quantized model from the pre-trained ``FP16`` model.

   **NOTE**: Quantization is time and memory consuming operation.
   Running quantization code below may take a long time.

Quantization of text model
^^^^^^^^^^^^^^^^^^^^^^^^^^

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    int8_text_model_path = "jina-clip-text_v1_int8.xml"

.. code:: ipython3

    %%skip not $to_quantize.value
    
    if len(calibration_data_text) == 0:
        raise RuntimeError(
            'Calibration dataset is empty. Please check internet connection and try to download images manually.'
        )
    
    ov_model_text = core.read_model(fp16_text_model_path)
    
    calibration_dataset = nncf.Dataset(calibration_data_text)
    quantized_model = nncf.quantize(
        model=ov_model_text,
        calibration_dataset=calibration_dataset
    )
    ov.save_model(quantized_model, int8_text_model_path)



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>




.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



Quantization of image model
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    int8_vision_model_path = "jina-clip-vision_v1_int8.xml"

.. code:: ipython3

    %%skip not $to_quantize.value
    
    if len(calibration_data_vision) == 0:
        raise RuntimeError(
            'Calibration dataset is empty. Please check internet connection and try to download images manually.'
        )
    
    ov_model_vision = core.read_model(fp16_vision_model_path)
    
    calibration_dataset = nncf.Dataset(calibration_data_vision)
    quantized_model = nncf.quantize(
        model=ov_model_vision,
        calibration_dataset=calibration_dataset
    )
    ov.save_model(quantized_model, int8_vision_model_path)



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>




.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



.. code:: ipython3

    %%skip not $to_quantize.value
    
    compiled_text_model_int8 = core.compile_model(int8_text_model_path, device.value)
    compiled_vision_model_int8 = core.compile_model(int8_vision_model_path, device.value)
    
    text_ov_res_int8 = compiled_text_model_int8(text_inputs["input_ids"])
    vis_ov_res_int8 = compiled_vision_model_int8(vision_inputs["pixel_values"])
    
    res = calc_simularity_softmax(vis_ov_res_int8[0], text_ov_res_int8[0])
    visionize_result(img_furseal, TEXT_INPUTS, np.array(res[0]))



.. image:: jina-clip-with-output_files/jina-clip-with-output_39_0.png


Compare File Size
~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    %%skip not $to_quantize.value
    
    from pathlib import Path
    
    fp16_ir_model_size = Path(fp16_text_model_path).with_suffix(".bin").stat().st_size / 1024 / 1024
    quantized_model_size = Path(int8_text_model_path).with_suffix(".bin").stat().st_size / 1024 / 1024
    print(
        f"Text model:   FP16 model size - {fp16_ir_model_size:.2f} MB; INT8 model size - {quantized_model_size:.2f} MB; Model compression rate: {fp16_ir_model_size / quantized_model_size:.3f}"
    )
    
    
    fp16_ir_model_size = Path(fp16_vision_model_path).with_suffix(".bin").stat().st_size / 1024 / 1024
    quantized_model_size = Path(int8_vision_model_path).with_suffix(".bin").stat().st_size / 1024 / 1024
    print(
        f"Vision model: FP16 model size - {fp16_ir_model_size:.2f} MB; INT8 model size - {quantized_model_size:.2f} MB;  Model compression rate: {fp16_ir_model_size / quantized_model_size:.3f}"
    )


.. parsed-literal::

    Text model:   FP16 model size - 266.88 MB; INT8 model size - 136.98 MB; Model compression rate: 1.948
    Vision model: FP16 model size - 163.83 MB; INT8 model size - 82.64 MB;  Model compression rate: 1.983


Compare inference time of the FP16 IR and quantized models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

To measure the inference performance of the ``FP16`` and ``INT8``
models, we use median inference time on calibration dataset. So we can
approximately estimate the speed up of the dynamic quantized models.

   **NOTE**: For the most accurate performance estimation, it is
   recommended to run ``benchmark_app`` in a terminal/command prompt
   after closing other applications with static shapes.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    import time
    
    
    def calculate_inference_time(model_path, calibration_data):
        model = core.compile_model(model_path, device.value)
        inference_time = []
        for batch in calibration_data:
            start = time.perf_counter()
            _ = model(batch)[0]
            end = time.perf_counter()
            delta = end - start
            inference_time.append(delta)
        return np.median(inference_time)

.. code:: ipython3

    %%skip not $to_quantize.value
    
    fp16_latency = calculate_inference_time(fp16_text_model_path, calibration_data_text)
    int8_latency = calculate_inference_time(int8_text_model_path, calibration_data_text)
    print(f"Performance speed up for text model: {fp16_latency / int8_latency:.3f}")
    
    
    fp16_latency = calculate_inference_time(fp16_vision_model_path, calibration_data_vision)
    int8_latency = calculate_inference_time(int8_vision_model_path, calibration_data_vision)
    print(f"Performance speed up for vision model: {fp16_latency / int8_latency:.3f}")


.. parsed-literal::

    Performance speed up for text model: 1.539
    Performance speed up for vision model: 1.509


Gradio demo
-----------

`back to top ⬆️ <#Table-of-contents:>`__

You can provide your own image and comma-separated list of labels for
zero-shot classification.

Feel free to upload an image, using the file upload window and type
label names into the text field, using comma as the separator (for
example, ``cat,dog,bird``)

.. code:: ipython3

    import gradio as gr
    
    core = ov.Core()
    
    compiled_text_model_int8 = None
    compiled_vision_model_int8 = None
    if Path(int8_text_model_path).exists and Path(int8_vision_model_path).exists:
        compiled_text_model_int8 = core.compile_model(int8_text_model_path, device.value)
        compiled_vision_model_int8 = core.compile_model(int8_vision_model_path, device.value)
    
    compiled_text_model_f16 = core.compile_model(fp16_text_model_path, device.value)
    compiled_vision_model_f16 = core.compile_model(fp16_vision_model_path, device.value)
    
    
    def image_text_sim(text, image, quantized_model):
        compiled_text_model = compiled_text_model_int8 if quantized_model else compiled_text_model_f16
        text = text.split(",")
        text_inputs = tokenizer(text, return_tensors="pt", **tokenizer_kwargs)
        emb1_res = compiled_text_model(text_inputs["input_ids"])
    
        compiled_vision_model = compiled_vision_model_int8 if quantized_model else compiled_vision_model_f16
        vision_input = processor(images=[image])
        emb2_res = compiled_vision_model(vision_input["pixel_values"])
    
        text_description = "Simularity: "
        simularity = calc_simularity_softmax(emb2_res[0], emb1_res[0], False)
        if len(text) == 1:
            text_description += f"{simularity[0]}"
        else:
            simularity_text = "\n".join([f"{text[i]} {sim:.4f}" for i, sim in enumerate(simularity[0])])
            text_description += f"\n{simularity_text}"
        return text_description
    
    
    def text_text_sim(text1, text2, quantized_model):
        compiled_text_model = compiled_text_model_int8 if quantized_model else compiled_text_model_f16
    
        text_inputs = tokenizer(text1, return_tensors="pt", **tokenizer_kwargs)
        emb1_res = compiled_text_model(text_inputs["input_ids"])
    
        text_inputs = tokenizer(text2, return_tensors="pt", **tokenizer_kwargs)
        emb2_res = compiled_text_model(text_inputs["input_ids"])
    
        return f"Simularity: {calc_simularity_softmax(emb1_res[0], emb2_res[0], False)[0][0]:.4f}"
    
    
    def image_image_sim(image1, image2, quantized_model):
        compiled_vision_model = compiled_vision_model_int8 if quantized_model else compiled_vision_model_f16
    
        vision_input = processor(images=[image1])
        emb1_res = compiled_vision_model(vision_input["pixel_values"])
    
        vision_input = processor(images=[image2])
        emb2_res = compiled_vision_model(vision_input["pixel_values"])
    
        return f"Simularity: {calc_simularity_softmax(emb1_res[0], emb2_res[0], False)[0][0]:.4f}"
    
    
    with gr.Blocks() as demo:
        gr.Markdown("Discover simularity of text or image files using this demo.")
        model_choice_visible = Path(int8_text_model_path).exists and Path(int8_vision_model_path).exists
        quantized_model = gr.Checkbox(
            label="Use quantized int8 model", info="Model type. FP16 model is used by default.", visible=model_choice_visible, value=False
        )
        with gr.Tab("Text-Image"):
            with gr.Row():
                image_text_vis = gr.Image(label="Image", type="pil")
                text_text_vis = gr.Textbox(label="Labels", info="Use comma to separate sentences")
            text_image_button = gr.Button("Submit")
            with gr.Row():
                gr.Examples([img_furseal], image_text_vis)
                gr.Examples(["seal,rat,cobra"], text_text_vis)
            text_image_output = gr.Textbox(label="Results")
        with gr.Tab("Text-Text"):
            with gr.Row():
                text_text_1 = gr.Textbox(label="Text")
                text_text_2 = gr.Textbox(label="Text")
            text_text_button = gr.Button("Submit")
            with gr.Row():
                gr.Examples(["The breeding season for fur seals is from May to the end of November"], text_text_1)
                gr.Examples(["Fur seals feed on fish and squid"], text_text_2)
            text_text_output = gr.Textbox(label="Results")
        with gr.Tab("Image-Image"):
            with gr.Row():
                image_image_1 = gr.Image(label="Image", type="pil")
                image_image_2 = gr.Image(label="Image", type="pil")
            image_image_button = gr.Button("Submit")
            text_output = gr.Textbox(label="Results")
            with gr.Row():
                gr.Examples([img_furseal], image_image_1)
                gr.Examples([img_coco], image_image_2)
            image_image_output = gr.Textbox(label="Results")
    
        text_image_button.click(image_text_sim, inputs=[text_text_vis, image_text_vis, quantized_model], outputs=text_image_output)
        text_text_button.click(text_text_sim, inputs=[text_text_1, text_text_2, quantized_model], outputs=text_text_output)
        image_image_button.click(image_image_sim, inputs=[image_image_1, image_image_2, quantized_model], outputs=image_image_output)
    
    
    if __name__ == "__main__":
        try:
            demo.queue().launch(debug=False)
        except Exception:
            demo.queue().launch(share=True, debug=False)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860
    
    To create a public link, set `share=True` in `launch()`.



.. raw:: html

    <div><iframe src="http://127.0.0.1:7860/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>

