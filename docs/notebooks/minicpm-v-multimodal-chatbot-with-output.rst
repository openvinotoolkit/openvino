Visual-language assistant with MiniCPM-V2 and OpenVINO
======================================================

MiniCPM-V 2 is a strong multimodal large language model for efficient
end-side deployment. MiniCPM-V 2.6 is the latest and most capable model
in the MiniCPM-V series. The model is built on SigLip-400M and Qwen2-7B
with a total of 8B parameters. It exhibits a significant performance
improvement over previous versions, and introduces new features for
multi-image and video understanding.

More details about model can be found in `model
card <https://huggingface.co/openbmb/MiniCPM-V-2_6>`__ and original
`repo <https://github.com/OpenBMB/MiniCPM-V>`__.

In this tutorial we consider how to convert and optimize MiniCPM-V2
model for creating multimodal chatbot. Additionally, we demonstrate how
to apply stateful transformation on LLM part and model optimization
techniques like weights compression using
`NNCF <https://github.com/openvinotoolkit/nncf>`__


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Convert model to OpenVINO Intermediate
   Representation <#convert-model-to-openvino-intermediate-representation>`__

   -  `Compress Language Model Weights to 4
      bits <#compress-language-model-weights-to-4-bits>`__

-  `Prepare model inference
   pipeline <#prepare-model-inference-pipeline>`__
-  `Run OpenVINO model inference <#run-openvino-model-inference>`__

   -  `Select device <#select-device>`__
   -  `Select language model variant <#select-language-model-variant>`__

-  `Interactive demo <#interactive-demo>`__

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

    %pip install -q "torch>=2.1" "torchvision" "timm>=0.9.2" "transformers>=4.40" "Pillow" "gradio>=4.19" "tqdm" "sentencepiece" "peft" "huggingface-hub>=0.24.0" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "openvino>=2024.3.0" "nncf>=2.12.0"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    import requests
    from pathlib import Path
    
    if not Path("minicpm_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/minicpm-v-multimodal-chatbot/minicpm_helper.py")
        open("minicpm_helper.py", "w").write(r.text)
    
    
    if not Path("gradio_helper.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks//minicpm-v-multimodal-chatbot//gradio_helper.py"
        )
        open("gradio_helper.py", "w").write(r.text)
    
    if not Path("notebook_utils.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
        open("notebook_utils.py", "w").write(r.text)

Convert model to OpenVINO Intermediate Representation
-----------------------------------------------------



OpenVINO supports PyTorch models via conversion to OpenVINO Intermediate
Representation (IR). `OpenVINO model conversion
API <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model>`__
should be used for these purposes. ``ov.convert_model`` function accepts
original PyTorch model instance and example input for tracing and
returns ``ov.Model`` representing this model in OpenVINO framework.
Converted model can be used for saving on disk using ``ov.save_model``
function or directly loading on device using ``core.complie_model``.

``minicpm_helper.py`` script contains helper function for model
conversion, please check its content if you interested in conversion
details.

.. raw:: html

   <details>

Click here for more detailed explanation of conversion steps
MiniCPM-V2.6 is autoregressive transformer generative model, it means
that each next model step depends from model output from previous step.
The generation approach is based on the assumption that the probability
distribution of a word sequence can be decomposed into the product of
conditional next word distributions. In other words, model predicts the
next token in the loop guided by previously generated tokens until the
stop-condition will be not reached (generated sequence of maximum length
or end of string token obtained). The way the next token will be
selected over predicted probabilities is driven by the selected decoding
methodology. You can find more information about the most popular
decoding methods in this
`blog <https://huggingface.co/blog/how-to-generate>`__. The entry point
for the generation process for models from the Hugging Face Transformers
library is the ``generate`` method. You can find more information about
its parameters and configuration in the
`documentation <https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/text_generation#transformers.GenerationMixin.generate>`__.
To preserve flexibility in the selection decoding methodology, we will
convert only model inference for one step.

The inference flow has difference on first step and for the next. On the
first step, model accept preprocessed input instruction and image, that
transformed to the unified embedding space using ``input_embedding`` and
``image encoder`` models, after that ``language model``, LLM-based part
of model, runs on input embeddings to predict probability of next
generated tokens. On the next step, ``language_model`` accepts only next
token id selected based on sampling strategy and processed by
``input_embedding`` model and cached attention key and values. Since the
output side is auto-regressive, an output token hidden state remains the
same once computed for every further generation step. Therefore,
recomputing it every time you want to generate a new token seems
wasteful. With the cache, the model saves the hidden state once it has
been computed. The model only computes the one for the most recently
generated output token at each time step, re-using the saved ones for
hidden tokens. This reduces the generation complexity from
:math:`O(n^3)` to :math:`O(n^2)` for a transformer model. More details
about how it works can be found in this
`article <https://scale.com/blog/pytorch-improvements#Text%20Translation>`__.

With increasing model size like in modern LLMs, we also can note an
increase in the number of attention blocks and size past key values
tensors respectively. The strategy for handling cache state as model
inputs and outputs in the inference cycle may become a bottleneck for
memory-bounded systems, especially with processing long input sequences,
for example in a chatbot scenario. OpenVINO suggests a transformation
that removes inputs and corresponding outputs with cache tensors from
the model keeping cache handling logic inside the model. Such models are
also called stateful. A stateful model is a model that implicitly
preserves data between two consecutive inference calls. The tensors
saved from one run are kept in an internal memory buffer called a
``state`` or a ``variable`` and may be passed to the next run, while
never being exposed as model output. Hiding the cache enables storing
and updating the cache values in a more device-friendly representation.
It helps to reduce memory consumption and additionally optimize model
performance. More details about stateful models and working with state
can be found in `OpenVINO
documentation <https://docs.openvino.ai/2024/openvino-workflow/running-inference/stateful-models.html>`__.

In LLMs, ``input_embedding`` is a part of language model, but for
multimodal case, the first step hidden state produced by this model part
should be integrated with image embeddings into common embedding space.
For ability to reuse this model part and avoid introduction of llm model
instance, we will use it separately.

``image_encoder`` is represented in MiniCPM-V by pretrained
`SigLIP <https://huggingface.co/google/siglip-so400m-patch14-384>`__
model. Additionally, MiniCPM uses perceiver ``resampler`` that
compresses the image representations. To preserve model ability to
process images of different size with respect aspect ratio combined in
batch, we will use ``image_encoder`` and ``resampler`` as separated
models.

To sum up above, model consists of 4 parts:

-  **Image Encoder** for encoding input images into embedding space. It
   includes SigLIP model.
-  **Resampler** for compression image representation.
-  **Input Embedding** for conversion input text tokens into embedding
   space.
-  **Language Model** for generation answer based on input embeddings
   provided by Image Encoder and Input Embedding models.

Let’s convert each model part.

.. raw:: html

   </details>

.. code:: ipython3

    from minicpm_helper import convert_minicpmv26
    
    # uncomment the line to see model conversion code
    # ??convert_minicpmv26


.. parsed-literal::

    2024-10-23 01:47:25.606377: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-10-23 01:47:25.640217: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-10-23 01:47:26.161344: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. code:: ipython3

    model_id = "openbmb/MiniCPM-V-2_6"
    
    model_dir = convert_minicpmv26(model_id)


.. parsed-literal::

    ⌛ openbmb/MiniCPM-V-2_6 conversion started. Be patient, it may takes some time.
    ⌛ Load Original model



.. parsed-literal::

    Fetching 24 files:   0%|          | 0/24 [00:00<?, ?it/s]



.. parsed-literal::

    image_processing_minicpmv.py:   0%|          | 0.00/16.6k [00:00<?, ?B/s]



.. parsed-literal::

    configuration_minicpm.py:   0%|          | 0.00/3.28k [00:00<?, ?B/s]



.. parsed-literal::

    .gitattributes:   0%|          | 0.00/1.64k [00:00<?, ?B/s]



.. parsed-literal::

    README.md:   0%|          | 0.00/15.5k [00:00<?, ?B/s]



.. parsed-literal::

    config.json:   0%|          | 0.00/1.36k [00:00<?, ?B/s]



.. parsed-literal::

    radar_final.png:   0%|          | 0.00/1.13M [00:00<?, ?B/s]



.. parsed-literal::

    added_tokens.json:   0%|          | 0.00/629 [00:00<?, ?B/s]



.. parsed-literal::

    generation_config.json:   0%|          | 0.00/121 [00:00<?, ?B/s]



.. parsed-literal::

    merges.txt:   0%|          | 0.00/1.67M [00:00<?, ?B/s]



.. parsed-literal::

    model-00003-of-00004.safetensors:   0%|          | 0.00/4.33G [00:00<?, ?B/s]



.. parsed-literal::

    model-00002-of-00004.safetensors:   0%|          | 0.00/4.93G [00:00<?, ?B/s]



.. parsed-literal::

    model.safetensors.index.json:   0%|          | 0.00/66.8k [00:00<?, ?B/s]



.. parsed-literal::

    modeling_minicpmv.py:   0%|          | 0.00/15.7k [00:00<?, ?B/s]



.. parsed-literal::

    modeling_navit_siglip.py:   0%|          | 0.00/41.8k [00:00<?, ?B/s]



.. parsed-literal::

    model-00001-of-00004.safetensors:   0%|          | 0.00/4.87G [00:00<?, ?B/s]



.. parsed-literal::

    model-00004-of-00004.safetensors:   0%|          | 0.00/2.06G [00:00<?, ?B/s]



.. parsed-literal::

    preprocessor_config.json:   0%|          | 0.00/714 [00:00<?, ?B/s]



.. parsed-literal::

    resampler.py:   0%|          | 0.00/34.7k [00:00<?, ?B/s]



.. parsed-literal::

    processing_minicpmv.py:   0%|          | 0.00/10.0k [00:00<?, ?B/s]



.. parsed-literal::

    tokenizer_config.json:   0%|          | 0.00/5.64k [00:00<?, ?B/s]



.. parsed-literal::

    special_tokens_map.json:   0%|          | 0.00/3.56k [00:00<?, ?B/s]



.. parsed-literal::

    tokenization_minicpmv_fast.py:   0%|          | 0.00/1.66k [00:00<?, ?B/s]



.. parsed-literal::

    tokenizer.json:   0%|          | 0.00/7.03M [00:00<?, ?B/s]



.. parsed-literal::

    vocab.json:   0%|          | 0.00/2.78M [00:00<?, ?B/s]



.. parsed-literal::

    Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]


.. parsed-literal::

    ✅ Original model successfully loaded


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/auto/image_processing_auto.py:513: FutureWarning: The image_processor_class argument is deprecated and will be removed in v4.42. Please use `slow_image_processor_class`, or `fast_image_processor_class` instead
      warnings.warn(


.. parsed-literal::

    ⌛ Convert Input embedding model
    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. parsed-literal::

    [ WARNING ]  Please fix your imports. Module %s has been moved to %s. The old module will be deleted in version %s.


.. parsed-literal::

    ✅ Input embedding model successfully converted
    ⌛ Convert Language model


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4713: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(
    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/qwen2/modeling_qwen2.py:100: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if sequence_length != 1:
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/qwen2/modeling_qwen2.py:165: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if seq_len > self.max_seq_len_cached:


.. parsed-literal::

    ✅ Language model successfully converted
    ⌛ Convert Image embedding model
    ✅ Image embedding model successfully converted
    ⌛ Convert Resamler model


.. parsed-literal::

    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/ckpt/resampler.py:421: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert embed_dim == embed_dim_to_check, \
    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/ckpt/resampler.py:428: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/ckpt/resampler.py:434: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"
    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/ckpt/resampler.py:520: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert key_padding_mask.shape == (bsz, src_len), \
    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/ckpt/resampler.py:539: TracerWarning: Converting a tensor to a Python float might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      q_scaled = q / math.sqrt(E)


.. parsed-literal::

    ✅ Resampler model successfully converted
    ✅ openbmb/MiniCPM-V-2_6 model sucessfully converted. You can find results in MiniCPM-V-2_6


Compress Language Model Weights to 4 bits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



For reducing memory consumption, weights compression optimization can be
applied using `NNCF <https://github.com/openvinotoolkit/nncf>`__.

.. raw:: html

   <details>

Click here for more details about weight compression Weight compression
aims to reduce the memory footprint of a model. It can also lead to
significant performance improvement for large memory-bound models, such
as Large Language Models (LLMs). LLMs and other models, which require
extensive memory to store the weights during inference, can benefit from
weight compression in the following ways:

-  enabling the inference of exceptionally large models that cannot be
   accommodated in the memory of the device;

-  improving the inference performance of the models by reducing the
   latency of the memory access when computing the operations with
   weights, for example, Linear layers.

`Neural Network Compression Framework
(NNCF) <https://github.com/openvinotoolkit/nncf>`__ provides 4-bit /
8-bit mixed weight quantization as a compression method primarily
designed to optimize LLMs. The main difference between weights
compression and full model quantization (post-training quantization) is
that activations remain floating-point in the case of weights
compression which leads to a better accuracy. Weight compression for
LLMs provides a solid inference performance improvement which is on par
with the performance of the full model quantization. In addition, weight
compression is data-free and does not require a calibration dataset,
making it easy to use.

``nncf.compress_weights`` function can be used for performing weights
compression. The function accepts an OpenVINO model and other
compression parameters. Compared to INT8 compression, INT4 compression
improves performance even more, but introduces a minor drop in
prediction quality.

More details about weights compression, can be found in `OpenVINO
documentation <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html>`__.

.. raw:: html

   </details>

..

   **Note:** weights compression process may require additional time and
   memory for performing. You can disable it using widget below:

.. code:: ipython3

    from minicpm_helper import compression_widget
    
    to_compress_weights = compression_widget()
    
    to_compress_weights




.. parsed-literal::

    Checkbox(value=True, description='Weights Compression')



.. code:: ipython3

    import nncf
    import gc
    import openvino as ov
    
    from minicpm_helper import llm_path, copy_llm_files
    
    
    compression_configuration = {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "ratio": 1.0, "all_layers": True}
    
    
    core = ov.Core()
    llm_int4_path = Path("language_model_int4") / llm_path.name
    if to_compress_weights.value and not (model_dir / llm_int4_path).exists():
        ov_model = core.read_model(model_dir / llm_path)
        ov_compressed_model = nncf.compress_weights(ov_model, **compression_configuration)
        ov.save_model(ov_compressed_model, model_dir / llm_int4_path)
        del ov_compressed_model
        del ov_model
        gc.collect()
        copy_llm_files(model_dir, llm_int4_path.parent)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino
    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              4 │ 100% (197 / 197)            │ 100% (197 / 197)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









Prepare model inference pipeline
--------------------------------



|image0|

As discussed, the model comprises Image Encoder and LLM (with separated
text embedding part) that generates answer. In ``minicpm_helper.py`` we
defined LLM inference class ``OvModelForCausalLMWithEmb`` that will
represent generation cycle, It is based on `HuggingFace Transformers
GenerationMixin <https://huggingface.co/docs/transformers/main_classes/text_generation>`__
and looks similar to `Optimum
Intel <https://huggingface.co/docs/optimum/intel/index>`__
``OVModelForCausalLM``\ that is used for LLM inference with only
difference that it can accept input embedding. In own turn, general
multimodal model class ``OvMiniCPMVModel`` handles chatbot functionality
including image processing and answer generation using LLM.

.. |image0| image:: https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/2727402e-3697-442e-beca-26b149967c84

.. code:: ipython3

    from minicpm_helper import OvModelForCausalLMWithEmb, OvMiniCPMV, init_model  # noqa: F401
    
    # uncomment the line to see model inference class
    # ??OVMiniCPMV
    
    # uncomment the line to see language model inference class
    # ??OvModelForCausalLMWithEmb

Run OpenVINO model inference
----------------------------



Select device
~~~~~~~~~~~~~



.. code:: ipython3

    from notebook_utils import device_widget
    
    device = device_widget(default="AUTO", exclude=["NPU"])
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Select language model variant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    from minicpm_helper import lm_variant_selector
    
    
    use_int4_lang_model = lm_variant_selector(model_dir / llm_int4_path)
    
    use_int4_lang_model




.. parsed-literal::

    Checkbox(value=True, description='INT4 language model')



.. code:: ipython3

    ov_model = init_model(model_dir, llm_path.parent if not use_int4_lang_model.value else llm_int4_path.parent, device.value)


.. parsed-literal::

    applied slice for lm head


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/auto/image_processing_auto.py:513: FutureWarning: The image_processor_class argument is deprecated and will be removed in v4.42. Please use `slow_image_processor_class`, or `fast_image_processor_class` instead
      warnings.warn(


.. code:: ipython3

    import requests
    from PIL import Image
    
    url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
    image = Image.open(requests.get(url, stream=True).raw)
    question = "What is unusual on this image?"
    
    print(f"Question:\n{question}")
    image


.. parsed-literal::

    Question:
    What is unusual on this image?




.. image:: minicpm-v-multimodal-chatbot-with-output_files/minicpm-v-multimodal-chatbot-with-output_17_1.png



.. code:: ipython3

    tokenizer = ov_model.processor.tokenizer
    
    msgs = [{"role": "user", "content": question}]
    
    
    print("Answer:")
    res = ov_model.chat(image=image, msgs=msgs, context=None, tokenizer=tokenizer, sampling=False, stream=True, max_new_tokens=50)
    
    generated_text = ""
    for new_text in res:
        generated_text += new_text
        print(new_text, flush=True, end="")


.. parsed-literal::

    Answer:
    The unusual aspect of this image is the cat's relaxed and vulnerable position. Typically, cats avoid exposing their bellies to potential threats or predators as it leaves them open for attack if necessary; however in domestic settings like homes where they feel safe from immediate

Interactive demo
----------------



.. code:: ipython3

    from gradio_helper import make_demo
    
    demo = make_demo(ov_model)
    
    try:
        demo.launch(debug=False, height=600)
    except Exception:
        demo.launch(debug=False, share=True, height=600)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860
    
    To create a public link, set `share=True` in `launch()`.







