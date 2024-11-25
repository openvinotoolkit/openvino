Visual-language assistant with InternVL2 and OpenVINO
=====================================================

InternVL 2.0 is the latest addition to the InternVL series of multimodal
large language models. InternVL 2.0 features a variety of
instruction-tuned models, ranging from 1 billion to 108 billion
parameters. Compared to the state-of-the-art open-source multimodal
large language models, InternVL 2.0 surpasses most open-source models.
It demonstrates competitive performance on par with proprietary
commercial models across various capabilities, including document and
chart comprehension, infographics QA, scene text understanding and OCR
tasks, scientific and mathematical problem solving, as well as cultural
understanding and integrated multimodal capabilities.

More details about model can be found in `model
card <https://huggingface.co/OpenGVLab/InternVL2-4B>`__,
`blog <https://internvl.github.io/blog/2024-07-02-InternVL-2.0/>`__ and
original `repo <https://github.com/OpenGVLab/InternVL>`__.

In this tutorial we consider how to convert and optimize InternVL2 model
for creating multimodal chatbot. Additionally, we demonstrate how to
apply stateful transformation on LLM part and model optimization
techniques like weights compression using
`NNCF <https://github.com/openvinotoolkit/nncf>`__


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Select model <#select-model>`__
-  `Convert and Optimize model <#convert-and-optimize-model>`__

   -  `Compress model weights to
      4-bit <#compress-model-weights-to-4-bit>`__

-  `Select inference device <#select-inference-device>`__
-  `Prepare model inference
   pipeline <#prepare-model-inference-pipeline>`__
-  `Run model inference <#run-model-inference>`__
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

    import platform
    
    %pip install -q "transformers>4.36" "torch>=2.1" "torchvision" "einops" "timm" "Pillow" "gradio>=4.36"  --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "nncf>=2.14.0" "datasets"
    %pip install -q "git+https://github.com/huggingface/optimum-intel.git" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q -U "openvino>=2024.5" "openvino-tokenizers>=2024.5" "openvino-genai>=2024.5"
    
    if platform.system() == "Darwin":
        %pip install -q "numpy<2.0.0"

.. code:: ipython3

    from pathlib import Path
    import requests
    
    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/internvl2/gradio_helper.py")
        open("gradio_helper.py", "w", encoding="utf-8").write(r.text)
    
    if not Path("notebook_utils.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
        open("notebook_utils.py", "w", encoding="utf-8").write(r.text)
    
    if not Path("cmd_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/cmd_helper.py")
        open("cmd_helper.py", "w", encoding="utf-8").write(r.text)

Select model
------------



There are multiple InternVL2 models available in `models
collection <https://huggingface.co/collections/OpenGVLab/internvl-20-667d3961ab5eb12c7ed1463e>`__.
You can select one of them for conversion and optimization in notebook
using widget bellow:

.. code:: ipython3

    model_ids = ["OpenGVLab/InternVL2-1B", "OpenGVLab/InternVL2-2B", "OpenGVLab/InternVL2-4B", "OpenGVLab/InternVL2-8B"]
    
    
    def model_selector(default=model_ids[0]):
        import ipywidgets as widgets
    
        model_checkpoint = widgets.Dropdown(
            options=model_ids,
            default=default,
            description="Model:",
        )
        return model_checkpoint
    
    
    model_id = model_selector()
    
    model_id




.. parsed-literal::

    Dropdown(description='Model:', options=('OpenGVLab/InternVL2-1B', 'OpenGVLab/InternVL2-2B', 'OpenGVLab/InternV…



.. code:: ipython3

    print(f"Selected {model_id.value}")
    pt_model_id = model_id.value
    model_dir = Path(pt_model_id.split("/")[-1])


.. parsed-literal::

    Selected OpenGVLab/InternVL2-1B
    

Convert and Optimize model
--------------------------



Our model conversion and optimization consist of following steps: 1.
Download original PyTorch model. 2. Convert model to OpenVINO format. 3.
Compress model weights using NNCF.

Let’s consider each step more deeply.

Convert model to OpenVINO IR format using Optimum CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



OpenVINO supports PyTorch models via conversion to OpenVINO Intermediate
Representation format. For convenience, we will use OpenVINO integration
with HuggingFace Optimum. `Optimum
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
``fp16``, ``int8`` and ``int4``. Fro int8 and int4
`nncf <https://github.com/openvinotoolkit/nncf>`__ will be used for
weight compression. More details about model export provided in `Optimum
Intel
documentation <https://huggingface.co/docs/optimum/intel/openvino/export#export-your-model>`__.

Compress model weights to 4-bit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For reducing memory
consumption, weights compression optimization can be applied using
`NNCF <https://github.com/openvinotoolkit/nncf>`__ via ``optimum-cli``
command. In this tutorial we will demonstrates how to apply accurate
int4 weight quantization using AWQ method.

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

Usually 4-bit compression allows to get maximal speedup and minimal
memory footprint comparing with 8-bit compression, but in the same time
it may significantly drop model accuracy. `Activation-aware Weight
Quantization <https://arxiv.org/abs/2306.00978>`__ (AWQ) is an algorithm
that tunes model weights for more accurate INT4 compression. It slightly
improves generation quality of compressed models, but requires
additional time for tuning weights on a calibration dataset.

More details about weights compression, can be found in `OpenVINO
documentation <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html>`__.

.. raw:: html

   </details>

.. code:: ipython3

    from cmd_helper import optimum_cli
    
    if not model_dir.exists():
        optimum_cli(
            model_id.value, model_dir, additional_args={"trust-remote-code": "", "weight-format": "int4", "dataset": "contextual", "awq": "", "num-samples": "32"}
        )



**Export command:**



``optimum-cli export openvino --model OpenGVLab/InternVL2-1B InternVL2-1B --trust-remote-code --weight-format int4 --dataset contextual --awq --num-samples 32``


.. parsed-literal::

    2024-11-20 12:30:38.063041: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-11-20 12:30:38.076313: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1732091438.091128  419590 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1732091438.095600  419590 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-11-20 12:30:38.110828: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    Attempt to save config using standard API has failed with 'architectures'. There may be an issue with model config, please check its correctness before usage.
    The class `optimum.bettertransformers.transformation.BetterTransformer` is deprecated and will be removed in a future release.
    WARNING:root:Cannot apply model.to_bettertransformer because of the exception:
    The model type qwen2 is not yet supported to be used with BetterTransformer. Feel free to open an issue at https://github.com/huggingface/optimum/issues if you would like this model type to be supported. Currently supported models are: dict_keys(['albert', 'bark', 'bart', 'bert', 'bert-generation', 'blenderbot', 'bloom', 'camembert', 'blip-2', 'clip', 'codegen', 'data2vec-text', 'deit', 'distilbert', 'electra', 'ernie', 'fsmt', 'gpt2', 'gptj', 'gpt_neo', 'gpt_neox', 'hubert', 'layoutlm', 'm2m_100', 'marian', 'markuplm', 'mbart', 'opt', 'pegasus', 'rembert', 'prophetnet', 'roberta', 'roc_bert', 'roformer', 'splinter', 'tapas', 't5', 'vilt', 'vit', 'vit_mae', 'vit_msn', 'wav2vec2', 'xlm-roberta', 'yolos']).. Usage model with stateful=True may be non-effective if model does not contain torch.functional.scaled_dot_product_attention
    `loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
    We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class (https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)
    /home/ea/work/py311/lib/python3.11/site-packages/transformers/cache_utils.py:458: TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.
      or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
    /home/ea/work/py311/lib/python3.11/site-packages/optimum/exporters/openvino/model_patcher.py:506: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if sequence_length != 1:
    /home/ea/work/py311/lib/python3.11/site-packages/transformers/cache_utils.py:443: TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.
      elif len(self.key_cache[layer_idx]) == 0:  # fills previously skipped layers; checking for tensor causes errors
    /home/ea/work/py311/lib/python3.11/site-packages/transformers/models/qwen2/modeling_qwen2.py:329: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
    /home/ea/.cache/huggingface/modules/transformers_modules/OpenGVLab/InternVL2-1B/a84c71e158b16180df4fd1c5fe963fdf54b2cd43/modeling_internvl_chat.py:195: TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      h = w = int(vit_embeds.shape[1] ** 0.5)
    /home/ea/.cache/huggingface/modules/transformers_modules/OpenGVLab/InternVL2-1B/a84c71e158b16180df4fd1c5fe963fdf54b2cd43/modeling_internvl_chat.py:169: TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    /home/ea/.cache/huggingface/modules/transformers_modules/OpenGVLab/InternVL2-1B/a84c71e158b16180df4fd1c5fe963fdf54b2cd43/modeling_internvl_chat.py:173: TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      x = x.view(n, int(h * scale_factor), int(w * scale_factor),
    /home/ea/.cache/huggingface/modules/transformers_modules/OpenGVLab/InternVL2-1B/a84c71e158b16180df4fd1c5fe963fdf54b2cd43/modeling_internvl_chat.py:174: TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      int(c / (scale_factor * scale_factor)))
    

.. parsed-literal::

    FlashAttention2 is not installed.
    

.. parsed-literal::

    Generating test split: 100%|██████████| 506/506 [00:00<00:00, 22956.39 examples/s]
    Collecting calibration dataset: 100%|██████████| 32/32 [04:18<00:00,  8.08s/it]
    

.. parsed-literal::

    [2KStatistics collection [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m100%[0m [38;2;0;104;181m32/32[0m • [38;2;0;104;181m0:02:34[0m • [38;2;0;104;181m0:00:00[0m181m0:00:04[0m181m0:00:08[0m:19[0m
    [?25hINFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 28% (1 / 169)               │ 0% (0 / 168)                           │
    ├────────────────┼─────────────────────────────┼────────────────────────���───────────────┤
    │              4 │ 72% (168 / 169)             │ 100% (168 / 168)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙
    [2KApplying AWQ [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m100%[0m [38;2;0;104;181m24/24[0m • [38;2;0;104;181m0:01:54[0m • [38;2;0;104;181m0:00:00[0m54[0m • [38;2;0;104;181m0:00:06[0m;2;97;53;69m━[0m[38;2;123;51;77m━[0m[38;2;153;48;86m━[0m[38;2;183;44;94m━[0m[38;2;209;42;102m━[0m[38;2;230;39;108m━[0m[38;2;244;38;112m━[0m[38;2;249;38;114m━[0m[38;2;244;38;112m━[0m[38;2;230;39;108m━[0m[38;2;209;42;102m━[0m[38;2;183;44;94m━[0m[38;2;153;48;86m━[0m[38;2;123;51;77m━[0m[38;2;97;53;69m━[0m[38;2;76;56;63m━[0m[38;2;62;57;59m━[0m[38;2;58;58;58m━[0m[38;2;62;57;59m━[0m[38;2;76;56;63m━[0m[38;2;97;53;69m━[0m[38;2;123;51;77m━[0m[38;2;153;48;86m━[0m[38;2;183;44;94m━[0m[38;2;209;42;102m━[0m[38;2;230;39;108m━[0m[38;2;244;38;112m━[0m[38;2;249;38;114m━[0m[38;2;244;38;112m━[0m[38;2;230;39;108m━[0m[38;2;209;42;102m━[0m[38;2;183;44;94m━[0m[38;2;153;48;86m━[0m[38;2;123;51;77m━[0m[38;2;97;53;69m━[0m[38;2;76;56;63m━[0m[38;2;62;57;59m━[0m[38;2;58;58;58m━[0m[38;2;62;57;59m━[0m[38;2;76;56;63m━[0m[38;2;97;53;69m━[0m[38;2;123;51;77m━[0m[38;2;153;48;86m━[0m   • [38;2;0;104;181m0:00:00[0m  
    [2KApplying Weight Compression [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m100%[0m • [38;2;0;104;181m0:00:17[0m • [38;2;0;104;181m0:00:00[0m;0;104;181m0:00:01[0m181m0:00:01[0m
    [?25hINFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (99 / 99)              │ 100% (99 / 99)                         │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━���━━━━━━━━━━━━━━━┙
    [2KApplying Weight Compression [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m100%[0m • [38;2;0;104;181m0:00:01[0m • [38;2;0;104;181m0:00:00[0m• [38;2;0;104;181m0:00:01[0m:01[0m
    [?25hINFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (1 / 1)                │ 100% (1 / 1)                           │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━���━━━━━━━━━━━━━━━┙
    [2KApplying Weight Compression [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m100%[0m • [38;2;0;104;181m0:00:00[0m • [38;2;0;104;181m0:00:00[0m
    [?25h

.. parsed-literal::

    Attempt to save config using standard API has failed with 'architectures'. There may be an issue with model config, please check its correctness before usage.
    

Select inference device
-----------------------



.. code:: ipython3

    from notebook_utils import device_widget
    
    device = device_widget(default="AUTO", exclude=["NPU"])
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Prepare model inference pipeline
--------------------------------



`OpenVINO™ GenAI <https://github.com/openvinotoolkit/openvino.genai>`__
is a library of the most popular Generative AI model pipelines,
optimized execution methods, and samples that run on top of highly
performant `OpenVINO
Runtime <https://github.com/openvinotoolkit/openvino>`__.

This library is friendly to PC and laptop execution, and optimized for
resource consumption. It requires no external dependencies to run
generative models as it already includes all the core functionality
(e.g. tokenization via openvino-tokenizers). OpenVINO™ GenAI is a flavor
of OpenVINO™, aiming to simplify running inference of generative AI
models. It hides the complexity of the generation process and minimizes
the amount of code required.

Inference Visual language models can be implemented using OpenVINO GenAI
``VLMPipeline`` class. Similarly to LLMPipeline, that we discussed in
this
`notebook <https://openvinotoolkit.github.io/openvino_notebooks/?search=Create+an+LLM-powered+Chatbot+using+OpenVINO+Generate+API>`__.
It supports chat mode with preserving conversational history inside
pipeline, that allows us effectively implements chatbot that supports
conversation about provided images content. For pipeline initialization
we should provide path to model directory and inference device.

.. code:: ipython3

    import openvino_genai as ov_genai
    
    ov_model = ov_genai.VLMPipeline(model_dir, device=device.value)

Run model inference
-------------------



For preparing input data, ``VLMPipeline`` use tokenizer and image
processor inside, we just need to convert image to input OpenVINO tensor
and provide question as string. Additionally, we can provides options
for controlling generation process (e.g. number of maximum generated
tokens or using multinomial sampling for decoding instead of greedy
search approach) using ``GenerationConfig``.

Generation process for long response may be time consuming, for
accessing partial result as soon as it is generated without waiting when
whole process finished, Streaming API can be used. Token streaming is
the mode in which the generative system returns the tokens one by one as
the model generates them. This enables showing progressive generations
to the user rather than waiting for the whole generation. Streaming is
an essential aspect of the end-user experience as it reduces latency,
one of the most critical aspects of a smooth experience.

.. code:: ipython3

    import requests
    from PIL import Image
    from io import BytesIO
    import numpy as np
    import openvino as ov
    
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 100
    
    
    def load_image(image_file):
        if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.byte)
        return image, ov.Tensor(image_data)
    
    
    EXAMPLE_IMAGE = Path("examples_image1.jpg")
    EXAMPLE_IMAGE_URL = "https://huggingface.co/OpenGVLab/InternVL2-2B/resolve/main/examples/image1.jpg"
    
    if not EXAMPLE_IMAGE.exists():
        img_data = requests.get(EXAMPLE_IMAGE_URL).content
        with EXAMPLE_IMAGE.open("wb") as handler:
            handler.write(img_data)
    
    
    def streamer(subword: str) -> bool:
        """
    
        Args:
            subword: sub-word of the generated text.
    
        Returns: Return flag corresponds whether generation should be stopped.
    
        """
        print(subword, end="", flush=True)
    
    
    question = "Please describe the image shortly"
    
    
    image, image_tensor = load_image(EXAMPLE_IMAGE)
    display(image)
    print(f"User: {question}\n")
    print("Assistant:")
    output = ov_model.generate(question, image=image_tensor, generation_config=config, streamer=streamer)



.. image:: internvl2-with-output_files/internvl2-with-output_14_0.png


.. parsed-literal::

    User: Please describe the image shortly
    
    Assistant:
    .
    
    The image shows a red panda, a type of mammal known for its distinctive red fur and white markings. The animal is resting on a wooden structure, possibly a platform or a platform-like object, with its head turned slightly towards the camera. The background is a natural setting, with trees and foliage visible, suggesting that the red panda is in a forested or wooded area. The red panda's eyes are large and expressive, and its ears are perked up, indicating that it is alert

Interactive demo
----------------



.. code:: ipython3

    from gradio_helper import make_demo
    
    demo = make_demo(ov_model)
    try:
        demo.launch(debug=True, height=600)
    except Exception:
        demo.launch(debug=True, share=True, height=600)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/
