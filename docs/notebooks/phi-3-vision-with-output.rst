Visual-language assistant with Phi3-Vision and OpenVINO
-------------------------------------------------------

The Phi-3-Vision is a lightweight, state-of-the-art open multimodal
model built upon datasets which include - synthetic data and filtered
publicly available websites - with a focus on very high-quality,
reasoning dense data both on text and vision. The model belongs to the
Phi-3 model family, and the multimodal version comes with 128K context
length (in tokens) it can support. The model underwent a rigorous
enhancement process, incorporating both supervised fine-tuning and
direct preference optimization to ensure precise instruction adherence
and robust safety measures. More details about model can be found in
`model blog
post <https://azure.microsoft.com/en-us/blog/new-models-added-to-the-phi-3-family-available-on-microsoft-azure/>`__,
`technical report <https://aka.ms/phi3-tech-report>`__,
`Phi-3-cookbook <https://github.com/microsoft/Phi-3CookBook>`__

In this tutorial we consider how to use Phi-3-Vision model to build
multimodal chatbot using `Optimum
Intel <https://github.com/huggingface/optimum-intel>`__. Additionally,
we optimize model to low precision using
`NNCF <https://github.com/openvinotoolkit/nncf>`__

**Table of contents:**

-  `Prerequisites <#prerequisites>`__
-  `Select Model <#select-model>`__
-  `Convert and Optimize model <#convert-and-optimize-model>`__

   -  `Compress model weights to
      4-bit <#compress-model-weights-to-4-bit>`__

-  `Select inference device <#select-inference-device>`__
-  `Run OpenVINO model <#run-openvino-model>`__
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



install required packages and setup helper functions.

.. code:: ipython3

    import platform

    %pip install -q -U "torch>=2.1" "torchvision" "transformers>=4.45" "protobuf>=3.20" "gradio>=4.26" "Pillow" "accelerate" "tqdm"  --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install --pre -qU "openvino>=2024.6.0" "openvino-tokenizers>=2024.6.0" --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
    %pip install -q -U "nncf>=2.14.0"
    %pip install -q "git+https://github.com/huggingface/optimum-intel.git" --extra-index-url https://download.pytorch.org/whl/cpu

    if platform.system() == "Darwin":
        %pip install -q "numpy<2.0"

.. code:: ipython3

    import requests
    from pathlib import Path

    if not Path("cmd_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/cmd_helper.py")
        open("cmd_helper.py", "w").write(r.text)


    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/phi-3-vision/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)

    if not Path("notebook_utils.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
        open("notebook_utils.py", "w").write(r.text)

    # Read more about telemetry collection at https://github.com/openvinotoolkit/openvino_notebooks?tab=readme-ov-file#-telemetry
    from notebook_utils import collect_telemetry

    collect_telemetry("phi-3-vision.ipynb")

Select Model
------------



The tutorial supports the following models from Phi-3 model family:

- `Phi-3.5-vision-instruct <https://huggingface.co/microsoft/Phi-3.5-vision-instruct>`__
- `Phi-3-vision-128k-instruct <https://huggingface.co/microsoft/Phi-3-vision-128k-instruct>`__

You can select one from the provided options below.

.. code:: ipython3

    import ipywidgets as widgets

    # Select model
    model_ids = [
        "microsoft/Phi-3.5-vision-instruct",
        "microsoft/Phi-3-vision-128k-instruct",
    ]

    model_dropdown = widgets.Dropdown(
        options=model_ids,
        value=model_ids[0],
        description="Model:",
        disabled=False,
    )

    model_dropdown




.. parsed-literal::

    Dropdown(description='Model:', options=('microsoft/Phi-3.5-vision-instruct', 'microsoft/Phi-3-vision-128k-inst…



.. code:: ipython3

    model_id = model_dropdown.value
    print(f"Selected {model_id}")
    MODEL_DIR = Path(model_id.split("/")[-1])


.. parsed-literal::

    Selected microsoft/Phi-3.5-vision-instruct


Convert and Optimize model
--------------------------



Phi-3-vision is PyTorch model. OpenVINO supports PyTorch models via
conversion to OpenVINO Intermediate Representation (IR). `OpenVINO model
conversion
API <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model>`__
should be used for these purposes. ``ov.convert_model`` function accepts
original PyTorch model instance and example input for tracing and
returns ``ov.Model`` representing this model in OpenVINO framework.
Converted model can be used for saving on disk using ``ov.save_model``
function or directly loading on device using ``core.compile_model``.

For convenience, we will use OpenVINO integration with HuggingFace
Optimum. `Optimum
Intel <https://huggingface.co/docs/optimum/intel/index>`__ is the
interface between the Transformers and Diffusers libraries and the
different tools and libraries provided by Intel to accelerate end-to-end
pipelines on Intel architectures.

Among other use cases, Optimum Intel provides a simple interface to
optimize your Transformers and Diffusers models, convert them to the
OpenVINO Intermediate Representation (IR) format and run inference using
OpenVINO Runtime. ``optimum-cli`` provides command line interface for
model conversion and optimization.

General command format:

.. code:: bash

   optimum-cli export openvino --model <model_id_or_path> --task <task> <output_dir>

where task is task to export the model for, if not specified, the task
will be auto-inferred based on the model. You can find a mapping between
tasks and model classes in Optimum TaskManager
`documentation <https://huggingface.co/docs/optimum/exporters/task_manager>`__.
Additionally, you can specify weights compression using
``--weight-format`` argument with one of following options: ``fp32``,
``fp16``, ``int8`` and ``int4``. For int8 and int4
`nncf <https://github.com/openvinotoolkit/nncf>`__ will be used for
weight compression. More details about model export provided in `Optimum
Intel
documentation <https://huggingface.co/docs/optimum/intel/openvino/export#export-your-model>`__.

Compress model weights to 4-bit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For reducing memory
consumption, weights compression optimization can be applied using
`NNCF <https://github.com/openvinotoolkit/nncf>`__ during run Optimum
Intel CLI.

.. raw:: html

   <details>

.. raw:: html

   <summary>

Click here for more details about weight compression

.. raw:: html

   </summary>

Weight compression aims to reduce the memory footprint of a model. It
can also lead to significant performance improvement for large
memory-bound models, such as Large Language Models (LLMs). LLMs and
other models, which require extensive memory to store the weights during
inference, can benefit from weight compression in the following ways:

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

.. code:: ipython3

    to_compress = widgets.Checkbox(value=True, description="Compress model", disabled=False)

    to_compress

.. code:: ipython3

    from cmd_helper import optimum_cli

    model_dir = MODEL_DIR / "INT4" if to_compress.value else MODEL_DIR / "FP16"
    if not model_dir.exists():
        optimum_cli(model_id, model_dir, additional_args={"weight-format": "int4" if to_compress.value else "fp16", "trust-remote-code": ""})



**Export command:**



``optimum-cli export openvino --model microsoft/Phi-3.5-vision-instruct Phi-3.5-vision-instruct/INT4 --weight-format int4 --trust-remote-code``


.. parsed-literal::

    2024-12-24 08:39:28.193255: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-12-24 08:39:28.205380: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1735015168.220063  230613 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1735015168.224457  230613 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-12-24 08:39:28.238718: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    Loading checkpoint shards: 100%|██████████| 2/2 [00:04<00:00,  2.14s/it]
    The class `optimum.bettertransformers.transformation.BetterTransformer` is deprecated and will be removed in a future release.
    WARNING:root:Cannot apply model.to_bettertransformer because of the exception:
    The model type phi3_v is not yet supported to be used with BetterTransformer. Feel free to open an issue at https://github.com/huggingface/optimum/issues if you would like this model type to be supported. Currently supported models are: dict_keys(['albert', 'bark', 'bart', 'bert', 'bert-generation', 'blenderbot', 'bloom', 'camembert', 'blip-2', 'clip', 'codegen', 'data2vec-text', 'deit', 'distilbert', 'electra', 'ernie', 'fsmt', 'gpt2', 'gptj', 'gpt_neo', 'gpt_neox', 'hubert', 'layoutlm', 'm2m_100', 'marian', 'markuplm', 'mbart', 'opt', 'pegasus', 'rembert', 'prophetnet', 'roberta', 'roc_bert', 'roformer', 'splinter', 'tapas', 't5', 'vilt', 'vit', 'vit_mae', 'vit_msn', 'wav2vec2', 'xlm-roberta', 'yolos']).. Usage model with stateful=True may be non-effective if model does not contain torch.functional.scaled_dot_product_attention
    `loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
    /home/ea/work/py311/lib/python3.11/site-packages/transformers/cache_utils.py:458: TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.
      or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
    /home/ea/work/py311/lib/python3.11/site-packages/transformers/modeling_attn_mask_utils.py:116: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
    /home/ea/work/py311/lib/python3.11/site-packages/optimum/exporters/onnx/model_patcher.py:306: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if past_key_values_length > 0:
    /home/ea/.cache/huggingface/modules/transformers_modules/microsoft/Phi-3.5-vision-instruct/4a0d683eba9f1d0cbfb6151705d1ee73c25a80ca/modeling_phi3_v.py:444: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      seq_len = seq_len or torch.max(position_ids) + 1
    /home/ea/.cache/huggingface/modules/transformers_modules/microsoft/Phi-3.5-vision-instruct/4a0d683eba9f1d0cbfb6151705d1ee73c25a80ca/modeling_phi3_v.py:445: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if seq_len > self.original_max_position_embeddings:
    /home/ea/work/py311/lib/python3.11/site-packages/nncf/torch/dynamic_graph/wrappers.py:85: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      op1 = operator(\*args, \*\*kwargs)
    /home/ea/work/py311/lib/python3.11/site-packages/transformers/cache_utils.py:443: TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.
      elif len(self.key_cache[layer_idx]) == 0:  # fills previously skipped layers; checking for tensor causes errors
    /home/ea/work/py311/lib/python3.11/site-packages/transformers/models/clip/modeling_clip.py:243: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if not interpolate_pos_encoding and (height != self.image_size or width != self.image_size):


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │ Weight compression mode   │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │ int8_asym                 │ 3% (1 / 129)                │ 0% (0 / 128)                           │
    ├───────────────────────────┼─────────────────────────────┼────────────────────────────────────────┤
    │ int4_asym                 │ 97% (128 / 129)             │ 100% (128 / 128)                       │
    ┕━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙
    [2KApplying Weight Compression [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m100%[0m • [38;2;0;104;181m0:01:58[0m • [38;2;0;104;181m0:00:00[0m;0;104;181m0:00:01[0m181m0:00:05[0m
    [?25hINFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │ Weight compression mode   │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │ int8_sym                  │ 100% (139 / 139)            │ 100% (139 / 139)                       │
    ┕━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙
    [2KApplying Weight Compression [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m100%[0m • [38;2;0;104;181m0:00:01[0m • [38;2;0;104;181m0:00:00[0m01[0m • [38;2;0;104;181m0:00:01[0m
    [?25hINFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │ Weight compression mode   │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │ int8_sym                  │ 100% (1 / 1)                │ 100% (1 / 1)                           │
    ┕━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙
    [2KApplying Weight Compression [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m100%[0m • [38;2;0;104;181m0:00:00[0m • [38;2;0;104;181m0:00:00[0m
    [?25hINFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │ Weight compression mode   │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │ int8_sym                  │ 100% (2 / 2)                │ 100% (2 / 2)                           │
    ┕━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙
    [2KApplying Weight Compression [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m100%[0m • [38;2;0;104;181m0:00:00[0m • [38;2;0;104;181m0:00:00[0m
    [?25h

Select inference device
-----------------------



.. code:: ipython3

    from notebook_utils import device_widget

    device = device_widget(default="AUTO", exclude=["NPU"])

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Run OpenVINO model
------------------



OpenVINO integration with Optimum Intel provides ready-to-use API for
model inference that can be used for smooth integration with
transformers-based solutions. For loading model, we will use
``OVModelForVisualCausalLM`` class that have compatible interface with
Transformers LLaVA implementation. For loading a model,
``from_pretrained`` method should be used. It accepts path to the model
directory or model_id from HuggingFace hub (if model is not converted to
OpenVINO format, conversion will be triggered automatically).
Additionally, we can provide an inference device, quantization config
(if model has not been quantized yet) and device-specific OpenVINO
Runtime configuration. More details about model inference with Optimum
Intel can be found in
`documentation <https://huggingface.co/docs/optimum/intel/openvino/inference>`__.

.. code:: ipython3

    from optimum.intel.openvino import OVModelForVisualCausalLM

    model = OVModelForVisualCausalLM.from_pretrained(model_dir, device=device.value, trust_remote_code=True)

.. code:: ipython3

    import requests
    from PIL import Image

    image_path = Path("cat.png")

    if not image_path.exists():
        url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
        image = Image.open(requests.get(url, stream=True).raw)
        image.save(image_path)
    else:
        image = Image.open(image_path)

    print("Question:\n What is unusual on this picture?")
    image


.. parsed-literal::

    Question:
     What is unusual on this picture?




.. image:: phi-3-vision-with-output_files/phi-3-vision-with-output_14_1.png



.. code:: ipython3

    from transformers import AutoProcessor, TextStreamer

    messages = [
        {"role": "user", "content": "<|image_1|>\nWhat is unusual on this picture?"},
    ]

    processor = AutoProcessor.from_pretrained(MODEL_DIR / "INT4" if to_compress.value else "FP16", trust_remote_code=True)

    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(prompt, [image], return_tensors="pt")

    generation_args = {"max_new_tokens": 50, "do_sample": False, "streamer": TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)}

    print("Answer:")
    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)


.. parsed-literal::

    Answer:
    A cat is lying in a box.


Interactive demo
----------------



.. code:: ipython3

    from gradio_helper import make_demo

    demo = make_demo(model, processor)

    try:
        demo.launch(debug=True, height=600)
    except Exception:
        demo.launch(debug=True, share=True, height=600)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/
