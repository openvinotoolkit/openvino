Visual-language assistant with Qwen2VL and OpenVINO
===================================================

Qwen2VL is the latest addition to the QwenVL series of multimodal large
language models.

**Key Enhancements of Qwen2VL:** \* **SoTA understanding of images of
various resolution & ratio**: Qwen2-VL achieves state-of-the-art
performance on visual understanding benchmarks, including MathVista,
DocVQA, RealWorldQA, MTVQA, etc. \* **Understanding videos of 20min+**:
Qwen2-VL can understand videos over 20 minutes for high-quality
video-based question answering, dialog, content creation, etc. \*
**Agent that can operate your mobiles, robots, etc.:** with the
abilities of complex reasoning and decision making, Qwen2-VL can be
integrated with devices like mobile phones, robots, etc., for automatic
operation based on visual environment and text instructions. \*
**Multilingual Support:** to serve global users, besides English and
Chinese, Qwen2-VL now supports the understanding of texts in different
languages inside images, including most European languages, Japanese,
Korean, Arabic, Vietnamese, etc.

**Model Architecture Details:**

-  **Naive Dynamic Resolution**: Qwen2-VL can handle arbitrary image
   resolutions, mapping them into a dynamic number of visual tokens,
   offering a more human-like visual processing experience.

.. raw:: html

   <p align="center">

.. raw:: html

   <p>

-  **Multimodal Rotary Position Embedding (M-ROPE)**: Decomposes
   positional embedding into parts to capture 1D textual, 2D visual, and
   3D video positional information, enhancing its multimodal processing
   capabilities.

.. raw:: html

   <p align="center">

.. raw:: html

   <p>

More details about model can be found in `model
card <https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct>`__,
`blog <https://qwenlm.github.io/blog/qwen2-vl/>`__ and original
`repo <https://github.com/QwenLM/Qwen2-VL>`__.

In this tutorial we consider how to convert and optimize Qwen2VL model
for creating multimodal chatbot using `Optimum
Intel <https://github.com/huggingface/optimum-intel>`__. Additionally,
we demonstrate how to apply model optimization techniques like weights
compression using `NNCF <https://github.com/openvinotoolkit/nncf>`__

**Table of contents:**

-  `Prerequisites <#prerequisites>`__
-  `Select model <#select-model>`__
-  `Convert and Optimize model <#convert-and-optimize-model>`__

   -  `Compress model weights to
      4-bit <#compress-model-weights-to-4-bit>`__

-  `Prepare model inference
   pipeline <#prepare-model-inference-pipeline>`__

   -  `Select inference device <#select-inference-device>`__

-  `Run model inference <#run-model-inference>`__
-  `Interactive Demo <#interactive-demo>`__

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

    %pip install -q "transformers>=4.45" "torch>=2.1" "torchvision" "qwen-vl-utils" "Pillow" "gradio>=4.36" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install  -q -U "openvino>=2024.6.0" "openvino-tokenizers>=2024.6.0" "nncf>=2.14.0"
    %pip install -q "git+https://github.com/huggingface/optimum-intel.git" --extra-index-url https://download.pytorch.org/whl/cpu

.. code:: ipython3

    from pathlib import Path
    import requests

    if not Path("cmd_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/cmd_helper.py")
        open("cmd_helper.py", "w").write(r.text)

    if not Path("notebook_utils.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
        open("notebook_utils.py", "w").write(r.text)

    # Read more about telemetry collection at https://github.com/openvinotoolkit/openvino_notebooks?tab=readme-ov-file#-telemetry
    from notebook_utils import collect_telemetry

    collect_telemetry("qwen2-vl.ipynb")

Select model
------------



There are multiple Qwen2VL models available in `models
collection <https://huggingface.co/collections/OpenGVLab/internvl-20-667d3961ab5eb12c7ed1463e>`__.
You can select one of them for conversion and optimization in notebook
using widget bellow:

.. code:: ipython3

    import ipywidgets as widgets

    model_ids = ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct"]

    model_id = widgets.Dropdown(
        options=model_ids,
        default=model_ids[0],
        description="Model:",
    )

    model_id




.. parsed-literal::

    Dropdown(description='Model:', options=('Qwen/Qwen2-VL-2B-Instruct', 'Qwen/Qwen2-VL-7B-Instruct'), value='Qwen…



.. code:: ipython3

    print(f"Selected {model_id.value}")
    pt_model_id = model_id.value
    model_dir = Path(pt_model_id.split("/")[-1])


.. parsed-literal::

    Selected Qwen/Qwen2-VL-2B-Instruct


Convert and Optimize model
--------------------------



Qwen2VL is PyTorch model. OpenVINO supports PyTorch models via
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
`NNCF <https://github.com/openvinotoolkit/nncf>`__.

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

    from cmd_helper import optimum_cli

    if not (model_dir / "INT4").exists():
        optimum_cli(pt_model_id, model_dir / "INT4", additional_args={"weight-format": "int4"})



**Export command:**



``optimum-cli export openvino --model Qwen/Qwen2-VL-2B-Instruct Qwen2-VL-2B-Instruct/INT4 --weight-format int4``


.. parsed-literal::

    2024-12-24 18:27:51.174286: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-12-24 18:27:51.186686: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1735050471.201093  340500 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1735050471.205249  340500 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-12-24 18:27:51.219846: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    Downloading shards: 100%|██████████| 2/2 [00:00<00:00,  2.73it/s]
    `Qwen2VLRotaryEmbedding` can now be fully parameterized by passing the model config through the `config` argument. All other arguments will be removed in v4.46
    Loading checkpoint shards: 100%|██████████| 2/2 [00:02<00:00,  1.46s/it]
    `loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
    /home/ea/work/py311/lib/python3.11/site-packages/transformers/cache_utils.py:458: TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.
      or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
    /home/ea/work/py311/lib/python3.11/site-packages/transformers/modeling_attn_mask_utils.py:281: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      elif sliding_window is None or key_value_length < sliding_window:
    /home/ea/work/py311/lib/python3.11/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py:1329: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attention_mask.shape[-1] > target_length:
    /home/ea/work/py311/lib/python3.11/site-packages/transformers/cache_utils.py:443: TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.
      elif len(self.key_cache[layer_idx]) == 0:  # fills previously skipped layers; checking for tensor causes errors


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │ Weight compression mode   │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │ int8_asym                 │ 15% (1 / 197)               │ 0% (0 / 196)                           │
    ├───────────────────────────┼─────────────────────────────┼────────────────────────────────────────┤
    │ int4_asym                 │ 85% (196 / 197)             │ 100% (196 / 196)                       │
    ┕━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙
    [2KApplying Weight Compression [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m100%[0m • [38;2;0;104;181m0:00:45[0m • [38;2;0;104;181m0:00:00[0m;0;104;181m0:00:01[0m181m0:00:02[0m
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
    │ int8_sym                  │ 100% (1 / 1)                │ 100% (1 / 1)                           │
    ┕━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙
    [2KApplying Weight Compression [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m100%[0m • [38;2;0;104;181m0:00:01[0m • [38;2;0;104;181m0:00:00[0m
    [?25hINFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │ Weight compression mode   │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │ int8_sym                  │ 100% (130 / 130)            │ 100% (130 / 130)                       │
    ┕━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙
    [2KApplying Weight Compression [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m100%[0m • [38;2;0;104;181m0:00:03[0m • [38;2;0;104;181m0:00:00[0m02[0m • [38;2;0;104;181m0:00:01[0m
    [?25h

Prepare model inference pipeline
--------------------------------



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


.. parsed-literal::

    2024-12-24 18:30:03.136274: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-12-24 18:30:03.148865: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1735050603.163311  340474 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1735050603.167677  340474 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-12-24 18:30:03.182551: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    from notebook_utils import device_widget

    device = device_widget(default="AUTO", exclude=["NPU"])

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    model = OVModelForVisualCausalLM.from_pretrained(model_dir / "INT4", device.value)


.. parsed-literal::

    Could not infer whether the model was already converted or not to the OpenVINO IR, keeping `export=AUTO`.
    unsupported operand type(s) for ^: 'bool' and 'str'


Run model inference
-------------------



.. code:: ipython3

    from PIL import Image
    from transformers import AutoProcessor, AutoTokenizer
    from qwen_vl_utils import process_vision_info
    from transformers import TextStreamer


    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(model_dir / "INT4", min_pixels=min_pixels, max_pixels=max_pixels)

    if processor.chat_template is None:
        tok = AutoTokenizer.from_pretrained(model_dir)
        processor.chat_template = tok.chat_template

    example_image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    example_image_path = Path("demo.jpeg")

    if not example_image_path.exists():
        Image.open(requests.get(example_image_url, stream=True).raw).save(example_image_path)

    image = Image.open(example_image_path)
    question = "Describe this image."

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{example_image_path}",
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    display(image)
    print("Question:")
    print(question)
    print("Answer:")

    generated_ids = model.generate(**inputs, max_new_tokens=100, streamer=TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True))



.. image:: qwen2-vl-with-output_files/qwen2-vl-with-output_15_0.png


.. parsed-literal::

    Question:
    Describe this image.
    Answer:
    The image depicts a woman sitting on a sandy beach with a large dog. The dog is wearing a harness and is sitting on its hind legs, reaching up to give a high-five to the woman. The woman is smiling and appears to be enjoying the moment. The background shows the ocean with gentle waves, and the sky is clear with a soft light, suggesting it might be either sunrise or sunset.


.. code:: ipython3

    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/qwen2-vl/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)

Interactive Demo
----------------



Now, you can try to chat with model. Upload image or video using
``Upload`` button, provide your text message into ``Input`` field and
click ``Submit`` to start communication.

.. code:: ipython3

    from gradio_helper import make_demo


    demo = make_demo(model, processor)

    try:
        demo.launch(debug=True)
    except Exception:
        demo.launch(debug=True, share=True)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/
