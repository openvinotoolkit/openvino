LLM Instruction-following pipeline with OpenVINO
================================================

LLM stands for “Large Language Model,” which refers to a type of
artificial intelligence model that is designed to understand and
generate human-like text based on the input it receives. LLMs are
trained on large datasets of text to learn patterns, grammar, and
semantic relationships, allowing them to generate coherent and
contextually relevant responses. One core capability of Large Language
Models (LLMs) is to follow natural language instructions.
Instruction-following models are capable of generating text in response
to prompts and are often used for tasks like writing assistance,
chatbots, and content generation.

In this tutorial, we consider how to run an instruction-following text
generation pipeline using popular LLMs and OpenVINO. We will use
pre-trained models from the `Hugging Face
Transformers <https://huggingface.co/docs/transformers/index>`__
library. To simplify the user experience, the `Hugging Face Optimum
Intel <https://huggingface.co/docs/optimum/intel/index>`__ library
converts the models to OpenVINO™ IR format.

The tutorial consists of the following steps:

-  Install prerequisites
-  Download and convert the model from a public source using the
   `OpenVINO integration with Hugging Face
   Optimum <https://huggingface.co/blog/openvino>`__.
-  Compress model weights to INT8 and INT4 with `OpenVINO
   NNCF <https://github.com/openvinotoolkit/nncf>`__
-  Create an instruction-following inference pipeline
-  Run instruction-following pipeline

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prerequisites <#Prerequisites>`__
-  `Select model for inference <#Select-model-for-inference>`__
-  `Instantiate Model using Optimum
   Intel <#Instantiate-Model-using-Optimum-Intel>`__
-  `Compress model weights <#Compress-model-weights>`__

   -  `Weights Compression using Optimum
      Intel <#Weights-Compression-using-Optimum-Intel>`__
   -  `Weights Compression using
      NNCF <#Weights-Compression-using-NNCF>`__

-  `Select device for inference and model
   variant <#Select-device-for-inference-and-model-variant>`__
-  `Create an instruction-following inference
   pipeline <#Create-an-instruction-following-inference-pipeline>`__

   -  `Setup imports <#Setup-imports>`__
   -  `Prepare template for user
      prompt <#Prepare-template-for-user-prompt>`__
   -  `Main generation function <#Main-generation-function>`__
   -  `Helpers for application <#Helpers-for-application>`__

-  `Run instruction-following
   pipeline <#Run-instruction-following-pipeline>`__

Prerequisites
-------------

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    %pip install -Uq pip
    %pip uninstall -q -y optimum optimum-intel
    %pip install --pre -Uq openvino openvino-tokenizers[transformers] --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
    %pip install -q "torch>=2.1" "nncf>=2.7" "transformers>=4.36.0" onnx "optimum>=1.16.1" "accelerate" "datasets>=2.14.6" "gradio>=4.19" "git+https://github.com/huggingface/optimum-intel.git" --extra-index-url https://download.pytorch.org/whl/cpu

Select model for inference
--------------------------

`back to top ⬆️ <#Table-of-contents:>`__

The tutorial supports different models, you can select one from the
provided options to compare the quality of open source LLM solutions.
>\ **Note**: conversion of some models can require additional actions
from user side and at least 64GB RAM for conversion.

The available options are:

-  **tiny-llama-1b-chat** - This is the chat model finetuned on top of
   `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T <https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T>`__.
   The TinyLlama project aims to pretrain a 1.1B Llama model on 3
   trillion tokens with the adoption of the same architecture and
   tokenizer as Llama 2. This means TinyLlama can be plugged and played
   in many open-source projects built upon Llama. Besides, TinyLlama is
   compact with only 1.1B parameters. This compactness allows it to
   cater to a multitude of applications demanding a restricted
   computation and memory footprint. More details about model can be
   found in `model
   card <https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0>`__
-  **phi-2** - Phi-2 is a Transformer with 2.7 billion parameters. It
   was trained using the same data sources as
   `Phi-1.5 <https://huggingface.co/microsoft/phi-1_5>`__, augmented
   with a new data source that consists of various NLP synthetic texts
   and filtered websites (for safety and educational value). When
   assessed against benchmarks testing common sense, language
   understanding, and logical reasoning, Phi-2 showcased a nearly
   state-of-the-art performance among models with less than 13 billion
   parameters. More details about model can be found in `model
   card <https://huggingface.co/microsoft/phi-2#limitations-of-phi-2>`__.
-  **dolly-v2-3b** - Dolly 2.0 is an instruction-following large
   language model trained on the Databricks machine-learning platform
   that is licensed for commercial use. It is based on
   `Pythia <https://github.com/EleutherAI/pythia>`__ and is trained on
   ~15k instruction/response fine-tuning records generated by Databricks
   employees in various capability domains, including brainstorming,
   classification, closed QA, generation, information extraction, open
   QA, and summarization. Dolly 2.0 works by processing natural language
   instructions and generating responses that follow the given
   instructions. It can be used for a wide range of applications,
   including closed question-answering, summarization, and generation.
   More details about model can be found in `model
   card <https://huggingface.co/databricks/dolly-v2-3b>`__.
-  **red-pajama-3b-instruct** - A 2.8B parameter pre-trained language
   model based on GPT-NEOX architecture. The model was fine-tuned for
   few-shot applications on the data of
   `GPT-JT <https://huggingface.co/togethercomputer/GPT-JT-6B-v1>`__,
   with exclusion of tasks that overlap with the HELM core
   scenarios.More details about model can be found in `model
   card <https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1>`__.
-  **mistral-7b** - The Mistral-7B-v0.2 Large Language Model (LLM) is a
   pretrained generative text model with 7 billion parameters. You can
   find more details about model in the `model
   card <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2>`__,
   `paper <https://arxiv.org/abs/2310.06825>`__ and `release blog
   post <https://mistral.ai/news/announcing-mistral-7b/>`__.
-  **llama-3-8b-instruct** - Llama 3 is an auto-regressive language
   model that uses an optimized transformer architecture. The tuned
   versions use supervised fine-tuning (SFT) and reinforcement learning
   with human feedback (RLHF) to align with human preferences for
   helpfulness and safety. The Llama 3 instruction tuned models are
   optimized for dialogue use cases and outperform many of the available
   open source chat models on common industry benchmarks. More details
   about model can be found in `Meta blog
   post <https://ai.meta.com/blog/meta-llama-3/>`__, `model
   website <https://llama.meta.com/llama3>`__ and `model
   card <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>`__.
   >\ **Note**: run model with demo, you will need to accept license
   agreement. >You must be a registered user in 🤗 Hugging Face Hub.
   Please visit `HuggingFace model
   card <https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>`__,
   carefully read terms of usage and click accept button. You will need
   to use an access token for the code below to run. For more
   information on access tokens, refer to `this section of the
   documentation <https://huggingface.co/docs/hub/security-tokens>`__.
   >You can login on Hugging Face Hub in notebook environment, using
   following code:

.. code:: python

       ## login to huggingfacehub to get access to pretrained model 

       from huggingface_hub import notebook_login, whoami

       try:
           whoami()
           print('Authorization token already provided')
       except OSError:
           notebook_login()

.. code:: ipython3

    from pathlib import Path
    import requests
    
    # Fetch `notebook_utils` module
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)
    from notebook_utils import download_file
    
    if not Path("./config.py").exists():
        download_file(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/llm-question-answering/config.py")
    from config import SUPPORTED_LLM_MODELS
    import ipywidgets as widgets

.. code:: ipython3

    model_ids = list(SUPPORTED_LLM_MODELS)
    
    model_id = widgets.Dropdown(
        options=model_ids,
        value=model_ids[1],
        description="Model:",
        disabled=False,
    )
    
    model_id




.. parsed-literal::

    Dropdown(description='Model:', index=1, options=('tiny-llama-1b', 'phi-2', 'dolly-v2-3b', 'red-pajama-instruct…



.. code:: ipython3

    model_configuration = SUPPORTED_LLM_MODELS[model_id.value]
    print(f"Selected model {model_id.value}")


.. parsed-literal::

    Selected model llama-3-8b-instruct


Instantiate Model using Optimum Intel
-------------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

Optimum Intel can be used to load optimized models from the `Hugging
Face Hub <https://huggingface.co/docs/optimum/intel/hf.co/models>`__ and
create pipelines to run an inference with OpenVINO Runtime using Hugging
Face APIs. The Optimum Inference models are API compatible with Hugging
Face Transformers models. This means we just need to replace
``AutoModelForXxx`` class with the corresponding ``OVModelForXxx``
class.

Below is an example of the RedPajama model

.. code:: diff

   -from transformers import AutoModelForCausalLM
   +from optimum.intel.openvino import OVModelForCausalLM
   from transformers import AutoTokenizer, pipeline

   model_id = "togethercomputer/RedPajama-INCITE-Chat-3B-v1"
   -model = AutoModelForCausalLM.from_pretrained(model_id)
   +model = OVModelForCausalLM.from_pretrained(model_id, export=True)

Model class initialization starts with calling ``from_pretrained``
method. When downloading and converting the Transformers model, the
parameter ``export=True`` should be added. We can save the converted
model for the next usage with the ``save_pretrained`` method. Tokenizer
class and pipelines API are compatible with Optimum models.

To optimize the generation process and use memory more efficiently, the
``use_cache=True`` option is enabled. Since the output side is
auto-regressive, an output token hidden state remains the same once
computed for every further generation step. Therefore, recomputing it
every time you want to generate a new token seems wasteful. With the
cache, the model saves the hidden state once it has been computed. The
model only computes the one for the most recently generated output token
at each time step, re-using the saved ones for hidden tokens. This
reduces the generation complexity from :math:`O(n^3)` to :math:`O(n^2)`
for a transformer model. More details about how it works can be found in
this
`article <https://scale.com/blog/pytorch-improvements#Text%20Translation>`__.
With this option, the model gets the previous step’s hidden states
(cached attention keys and values) as input and additionally provides
hidden states for the current step as output. It means for all next
iterations, it is enough to provide only a new token obtained from the
previous step and cached key values to get the next token prediction.

Compress model weights
----------------------

`back to top ⬆️ <#Table-of-contents:>`__ The Weights Compression
algorithm is aimed at compressing the weights of the models and can be
used to optimize the model footprint and performance of large models
where the size of weights is relatively larger than the size of
activations, for example, Large Language Models (LLM). Compared to INT8
compression, INT4 compression improves performance even more but
introduces a minor drop in prediction quality.

Weights Compression using Optimum Intel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

Optimum Intel supports weight compression via NNCF out of the box. For
8-bit compression we pass ``load_in_8bit=True`` to ``from_pretrained()``
method of ``OVModelForCausalLM``. For 4 bit compression we provide
``quantization_config=OVWeightQuantizationConfig(bits=4, ...)`` argument
containing number of bits and other compression parameters. An example
of this approach usage you can find in `llm-chatbot
notebook <../llm-chatbot>`__

Weights Compression using NNCF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

You also can perform weights compression for OpenVINO models using NNCF
directly. ``nncf.compress_weights`` function accepts the OpenVINO model
instance and compresses its weights for Linear and Embedding layers. We
will consider this variant in this notebook for both int4 and int8
compression.

   **Note**: This tutorial involves conversion model for FP16 and
   INT4/INT8 weights compression scenarios. It may be memory and
   time-consuming in the first run. You can manually control the
   compression precision below. **Note**: There may be no speedup for
   INT4/INT8 compressed models on dGPU

.. code:: ipython3

    from IPython.display import display
    
    prepare_int4_model = widgets.Checkbox(
        value=True,
        description="Prepare INT4 model",
        disabled=False,
    )
    prepare_int8_model = widgets.Checkbox(
        value=False,
        description="Prepare INT8 model",
        disabled=False,
    )
    prepare_fp16_model = widgets.Checkbox(
        value=False,
        description="Prepare FP16 model",
        disabled=False,
    )
    
    display(prepare_int4_model)
    display(prepare_int8_model)
    display(prepare_fp16_model)



.. parsed-literal::

    Checkbox(value=True, description='Prepare INT4 model')



.. parsed-literal::

    Checkbox(value=False, description='Prepare INT8 model')



.. parsed-literal::

    Checkbox(value=False, description='Prepare FP16 model')


.. code:: ipython3

    from pathlib import Path
    import logging
    import openvino as ov
    import nncf
    from optimum.intel.openvino import OVModelForCausalLM, OVWeightQuantizationConfig
    import gc
    
    
    nncf.set_log_level(logging.ERROR)
    
    pt_model_id = model_configuration["model_id"]
    fp16_model_dir = Path(model_id.value) / "FP16"
    int8_model_dir = Path(model_id.value) / "INT8_compressed_weights"
    int4_model_dir = Path(model_id.value) / "INT4_compressed_weights"
    
    core = ov.Core()
    
    
    def convert_to_fp16():
        if (fp16_model_dir / "openvino_model.xml").exists():
            return
        ov_model = OVModelForCausalLM.from_pretrained(pt_model_id, export=True, compile=False, load_in_8bit=False)
        ov_model.half()
        ov_model.save_pretrained(fp16_model_dir)
        del ov_model
        gc.collect()
    
    
    def convert_to_int8():
        if (int8_model_dir / "openvino_model.xml").exists():
            return
        ov_model = OVModelForCausalLM.from_pretrained(pt_model_id, export=True, compile=False, load_in_8bit=True)
        ov_model.save_pretrained(int8_model_dir)
        del ov_model
        gc.collect()
    
    
    def convert_to_int4():
        compression_configs = {
            "mistral-7b": {
                "sym": True,
                "group_size": 64,
                "ratio": 0.6,
            },
            "red-pajama-3b-instruct": {
                "sym": False,
                "group_size": 128,
                "ratio": 0.5,
            },
            "dolly-v2-3b": {"sym": False, "group_size": 32, "ratio": 0.5},
            "llama-3-8b-instruct": {"sym": True, "group_size": 128, "ratio": 1.0},
            "default": {
                "sym": False,
                "group_size": 128,
                "ratio": 0.8,
            },
        }
    
        model_compression_params = compression_configs.get(model_id.value, compression_configs["default"])
        if (int4_model_dir / "openvino_model.xml").exists():
            return
        ov_model = OVModelForCausalLM.from_pretrained(
            pt_model_id,
            export=True,
            compile=False,
            quantization_config=OVWeightQuantizationConfig(bits=4, **model_compression_params),
        )
        ov_model.save_pretrained(int4_model_dir)
        del ov_model
        gc.collect()
    
    
    if prepare_fp16_model.value:
        convert_to_fp16()
    if prepare_int8_model.value:
        convert_to_int8()
    if prepare_int4_model.value:
        convert_to_int4()


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    2024-04-19 10:35:50.012050: I tensorflow/core/util/port.cc:111] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-04-19 10:35:50.025002: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-04-19 10:35:50.060073: E tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:9342] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-04-19 10:35:50.060108: E tensorflow/compiler/xla/stream_executor/cuda/cuda_fft.cc:609] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-04-19 10:35:50.060134: E tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:1518] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-04-19 10:35:50.068691: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-04-19 10:35:50.069448: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-04-19 10:35:51.045741: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
    Framework not specified. Using pt to export the model.



.. parsed-literal::

    Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]


.. parsed-literal::

    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    Using framework PyTorch: 2.2.2+cpu
    Overriding 1 configuration item(s)
    	- use_cache -> True
    /home/ea/miniconda3/lib/python3.11/site-packages/transformers/modeling_utils.py:4225: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(
    The cos_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class
    The sin_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class
    /home/ea/miniconda3/lib/python3.11/site-packages/optimum/exporters/openvino/model_patcher.py:311: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if sequence_length != 1:



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



.. parsed-literal::

    Configuration saved in llama-3-8b-instruct/INT4_compressed_weights/openvino_config.json


Let’s compare model size for different compression types

.. code:: ipython3

    fp16_weights = fp16_model_dir / "openvino_model.bin"
    int8_weights = int8_model_dir / "openvino_model.bin"
    int4_weights = int4_model_dir / "openvino_model.bin"
    
    if fp16_weights.exists():
        print(f"Size of FP16 model is {fp16_weights.stat().st_size / 1024 / 1024:.2f} MB")
    for precision, compressed_weights in zip([8, 4], [int8_weights, int4_weights]):
        if compressed_weights.exists():
            print(f"Size of model with INT{precision} compressed weights is {compressed_weights.stat().st_size / 1024 / 1024:.2f} MB")
        if compressed_weights.exists() and fp16_weights.exists():
            print(f"Compression rate for INT{precision} model: {fp16_weights.stat().st_size / compressed_weights.stat().st_size:.3f}")


.. parsed-literal::

    Size of model with INT4 compressed weights is 4435.75 MB


Select device for inference and model variant
---------------------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

   **Note**: There may be no speedup for INT4/INT8 compressed models on
   dGPU.

.. code:: ipython3

    core = ov.Core()
    
    support_devices = core.available_devices
    if "NPU" in support_devices:
        support_devices.remove("NPU")
    
    device = widgets.Dropdown(
        options=support_devices + ["AUTO"],
        value="CPU",
        description="Device:",
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'GPU.0', 'GPU.1', 'AUTO'), value='CPU')



.. code:: ipython3

    available_models = []
    if int4_model_dir.exists():
        available_models.append("INT4")
    if int8_model_dir.exists():
        available_models.append("INT8")
    if fp16_model_dir.exists():
        available_models.append("FP16")
    
    model_to_run = widgets.Dropdown(
        options=available_models,
        value=available_models[0],
        description="Model to run:",
        disabled=False,
    )
    
    model_to_run




.. parsed-literal::

    Dropdown(description='Model to run:', options=('INT4',), value='INT4')



.. code:: ipython3

    from transformers import AutoTokenizer
    
    if model_to_run.value == "INT4":
        model_dir = int4_model_dir
    elif model_to_run.value == "INT8":
        model_dir = int8_model_dir
    else:
        model_dir = fp16_model_dir
    print(f"Loading model from {model_dir}")
    
    model_name = model_configuration["model_id"]
    ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}
    
    tok = AutoTokenizer.from_pretrained(model_name)
    
    ov_model = OVModelForCausalLM.from_pretrained(
        model_dir,
        device=device.value,
        ov_config=ov_config,
    )


.. parsed-literal::

    Loading model from llama-3-8b-instruct/INT4_compressed_weights


.. parsed-literal::

    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    Compiling the model to CPU ...


Create an instruction-following inference pipeline
--------------------------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

The ``run_generation`` function accepts user-provided text input,
tokenizes it, and runs the generation process. Text generation is an
iterative process, where each next token depends on previously generated
until a maximum number of tokens or stop generation condition is not
reached. To obtain intermediate generation results without waiting until
when generation is finished, we will use
```TextIteratorStreamer`` <https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.TextIteratorStreamer>`__,
provided as part of HuggingFace `Streaming
API <https://huggingface.co/docs/transformers/main/en/generation_strategies#streaming>`__.

The diagram below illustrates how the instruction-following pipeline
works

.. figure:: https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/e881f4a4-fcc8-427a-afe1-7dd80aebd66e
   :alt: generation pipeline)

   generation pipeline)

As can be seen, on the first iteration, the user provided instructions
converted to token ids using a tokenizer, then prepared input provided
to the model. The model generates probabilities for all tokens in logits
format The way the next token will be selected over predicted
probabilities is driven by the selected decoding methodology. You can
find more information about the most popular decoding methods in this
`blog <https://huggingface.co/blog/how-to-generate>`__.

There are several parameters that can control text generation quality:

-  | ``Temperature`` is a parameter used to control the level of
     creativity in AI-generated text. By adjusting the ``temperature``,
     you can influence the AI model’s probability distribution, making
     the text more focused or diverse.
   | Consider the following example: The AI model has to complete the
     sentence “The cat is \____.” with the following token
     probabilities:

   | playing: 0.5
   | sleeping: 0.25
   | eating: 0.15
   | driving: 0.05
   | flying: 0.05

   -  **Low temperature** (e.g., 0.2): The AI model becomes more focused
      and deterministic, choosing tokens with the highest probability,
      such as “playing.”
   -  **Medium temperature** (e.g., 1.0): The AI model maintains a
      balance between creativity and focus, selecting tokens based on
      their probabilities without significant bias, such as “playing,”
      “sleeping,” or “eating.”
   -  **High temperature** (e.g., 2.0): The AI model becomes more
      adventurous, increasing the chances of selecting less likely
      tokens, such as “driving” and “flying.”

-  ``Top-p``, also known as nucleus sampling, is a parameter used to
   control the range of tokens considered by the AI model based on their
   cumulative probability. By adjusting the ``top-p`` value, you can
   influence the AI model’s token selection, making it more focused or
   diverse. Using the same example with the cat, consider the following
   top_p settings:

   -  **Low top_p** (e.g., 0.5): The AI model considers only tokens with
      the highest cumulative probability, such as “playing.”
   -  **Medium top_p** (e.g., 0.8): The AI model considers tokens with a
      higher cumulative probability, such as “playing,” “sleeping,” and
      “eating.”
   -  **High top_p** (e.g., 1.0): The AI model considers all tokens,
      including those with lower probabilities, such as “driving” and
      “flying.”

-  ``Top-k`` is another popular sampling strategy. In comparison with
   Top-P, which chooses from the smallest possible set of words whose
   cumulative probability exceeds the probability P, in Top-K sampling K
   most likely next words are filtered and the probability mass is
   redistributed among only those K next words. In our example with cat,
   if k=3, then only “playing”, “sleeping” and “eating” will be taken
   into account as possible next word.

To optimize the generation process and use memory more efficiently, the
``use_cache=True`` option is enabled. Since the output side is
auto-regressive, an output token hidden state remains the same once
computed for every further generation step. Therefore, recomputing it
every time you want to generate a new token seems wasteful. With the
cache, the model saves the hidden state once it has been computed. The
model only computes the one for the most recently generated output token
at each time step, re-using the saved ones for hidden tokens. This
reduces the generation complexity from O(n^3) to O(n^2) for a
transformer model. More details about how it works can be found in this
`article <https://scale.com/blog/pytorch-improvements#Text%20Translation>`__.
With this option, the model gets the previous step’s hidden states
(cached attention keys and values) as input and additionally provides
hidden states for the current step as output. It means for all next
iterations, it is enough to provide only a new token obtained from the
previous step and cached key values to get the next token prediction.

The generation cycle repeats until the end of the sequence token is
reached or it also can be interrupted when maximum tokens will be
generated. As already mentioned before, we can enable printing current
generated tokens without waiting until when the whole generation is
finished using Streaming API, it adds a new token to the output queue
and then prints them when they are ready.

Setup imports
~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    from threading import Thread
    from time import perf_counter
    from typing import List
    import gradio as gr
    from transformers import AutoTokenizer, TextIteratorStreamer
    import numpy as np

Prepare template for user prompt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

For effective generation, model expects to have input in specific
format. The code below prepare template for passing user instruction
into model with providing additional context.

.. code:: ipython3

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_kwargs = model_configuration.get("toeknizer_kwargs", {})
    
    
    def get_special_token_id(tokenizer: AutoTokenizer, key: str) -> int:
        """
        Gets the token ID for a given string that has been added to the tokenizer as a special token.
    
        Args:
            tokenizer (PreTrainedTokenizer): the tokenizer
            key (str): the key to convert to a single token
    
        Raises:
            RuntimeError: if more than one ID was generated
    
        Returns:
            int: the token ID for the given key
        """
        token_ids = tokenizer.encode(key)
        if len(token_ids) > 1:
            raise ValueError(f"Expected only a single token for '{key}' but found {token_ids}")
        return token_ids[0]
    
    
    response_key = model_configuration.get("response_key")
    tokenizer_response_key = None
    
    if response_key is not None:
        tokenizer_response_key = next(
            (token for token in tokenizer.additional_special_tokens if token.startswith(response_key)),
            None,
        )
    
    end_key_token_id = None
    if tokenizer_response_key:
        try:
            end_key = model_configuration.get("end_key")
            if end_key:
                end_key_token_id = get_special_token_id(tokenizer, end_key)
            # Ensure generation stops once it generates "### End"
        except ValueError:
            pass
    
    prompt_template = model_configuration.get("prompt_template", "{instruction}")
    end_key_token_id = end_key_token_id or tokenizer.eos_token_id
    pad_token_id = end_key_token_id or tokenizer.pad_token_id


.. parsed-literal::

    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.


Main generation function
~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

As it was discussed above, ``run_generation`` function is the entry
point for starting generation. It gets provided input instruction as
parameter and returns model response.

.. code:: ipython3

    def run_generation(
        user_text: str,
        top_p: float,
        temperature: float,
        top_k: int,
        max_new_tokens: int,
        perf_text: str,
    ):
        """
        Text generation function
    
        Parameters:
          user_text (str): User-provided instruction for a generation.
          top_p (float):  Nucleus sampling. If set to < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for a generation.
          temperature (float): The value used to module the logits distribution.
          top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
          max_new_tokens (int): Maximum length of generated sequence.
          perf_text (str): Content of text field for printing performance results.
        Returns:
          model_output (str) - model-generated text
          perf_text (str) - updated perf text filed content
        """
    
        # Prepare input prompt according to model expected template
        prompt_text = prompt_template.format(instruction=user_text)
    
        # Tokenize the user text.
        model_inputs = tokenizer(prompt_text, return_tensors="pt", **tokenizer_kwargs)
    
        # Start generation on a separate thread, so that we don't block the UI. The text is pulled from the streamer
        # in the main thread. Adds timeout to the streamer to handle exceptions in the generation thread.
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=float(temperature),
            top_k=top_k,
            eos_token_id=end_key_token_id,
            pad_token_id=pad_token_id,
        )
        t = Thread(target=ov_model.generate, kwargs=generate_kwargs)
        t.start()
    
        # Pull the generated text from the streamer, and update the model output.
        model_output = ""
        per_token_time = []
        num_tokens = 0
        start = perf_counter()
        for new_text in streamer:
            current_time = perf_counter() - start
            model_output += new_text
            perf_text, num_tokens = estimate_latency(current_time, perf_text, new_text, per_token_time, num_tokens)
            yield model_output, perf_text
            start = perf_counter()
        return model_output, perf_text

Helpers for application
~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

For making interactive user interface we will use Gradio library. The
code bellow provides useful functions used for communication with UI
elements.

.. code:: ipython3

    def estimate_latency(
        current_time: float,
        current_perf_text: str,
        new_gen_text: str,
        per_token_time: List[float],
        num_tokens: int,
    ):
        """
        Helper function for performance estimation
    
        Parameters:
          current_time (float): This step time in seconds.
          current_perf_text (str): Current content of performance UI field.
          new_gen_text (str): New generated text.
          per_token_time (List[float]): history of performance from previous steps.
          num_tokens (int): Total number of generated tokens.
    
        Returns:
          update for performance text field
          update for a total number of tokens
        """
        num_current_toks = len(tokenizer.encode(new_gen_text))
        num_tokens += num_current_toks
        per_token_time.append(num_current_toks / current_time)
        if len(per_token_time) > 10 and len(per_token_time) % 4 == 0:
            current_bucket = per_token_time[:-10]
            return (
                f"Average generation speed: {np.mean(current_bucket):.2f} tokens/s. Total generated tokens: {num_tokens}",
                num_tokens,
            )
        return current_perf_text, num_tokens
    
    
    def reset_textbox(instruction: str, response: str, perf: str):
        """
        Helper function for resetting content of all text fields
    
        Parameters:
          instruction (str): Content of user instruction field.
          response (str): Content of model response field.
          perf (str): Content of performance info filed
    
        Returns:
          empty string for each placeholder
        """
        return "", "", ""

Run instruction-following pipeline
----------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

Now, we are ready to explore model capabilities. This demo provides a
simple interface that allows communication with a model using text
instruction. Type your instruction into the ``User instruction`` field
or select one from predefined examples and click on the ``Submit``
button to start generation. Additionally, you can modify advanced
generation parameters:

-  ``Device`` - allows switching inference device. Please note, every
   time when new device is selected, model will be recompiled and this
   takes some time.
-  ``Max New Tokens`` - maximum size of generated text.
-  ``Top-p (nucleus sampling)`` - if set to < 1, only the smallest set
   of most probable tokens with probabilities that add up to top_p or
   higher are kept for a generation.
-  ``Top-k`` - the number of highest probability vocabulary tokens to
   keep for top-k-filtering.
-  ``Temperature`` - the value used to module the logits distribution.

.. code:: ipython3

    examples = [
        "Give me a recipe for pizza with pineapple",
        "Write me a tweet about the new OpenVINO release",
        "Explain the difference between CPU and GPU",
        "Give five ideas for a great weekend with family",
        "Do Androids dream of Electric sheep?",
        "Who is Dolly?",
        "Please give me advice on how to write resume?",
        "Name 3 advantages to being a cat",
        "Write instructions on how to become a good AI engineer",
        "Write a love letter to my best friend",
    ]
    
    
    with gr.Blocks() as demo:
        gr.Markdown(
            "# Question Answering with " + model_id.value + " and OpenVINO.\n"
            "Provide instruction which describes a task below or select among predefined examples and model writes response that performs requested task."
        )
    
        with gr.Row():
            with gr.Column(scale=4):
                user_text = gr.Textbox(
                    placeholder="Write an email about an alpaca that likes flan",
                    label="User instruction",
                )
                model_output = gr.Textbox(label="Model response", interactive=False)
                performance = gr.Textbox(label="Performance", lines=1, interactive=False)
                with gr.Column(scale=1):
                    button_clear = gr.Button(value="Clear")
                    button_submit = gr.Button(value="Submit")
                gr.Examples(examples, user_text)
            with gr.Column(scale=1):
                max_new_tokens = gr.Slider(
                    minimum=1,
                    maximum=1000,
                    value=256,
                    step=1,
                    interactive=True,
                    label="Max New Tokens",
                )
                top_p = gr.Slider(
                    minimum=0.05,
                    maximum=1.0,
                    value=0.92,
                    step=0.05,
                    interactive=True,
                    label="Top-p (nucleus sampling)",
                )
                top_k = gr.Slider(
                    minimum=0,
                    maximum=50,
                    value=0,
                    step=1,
                    interactive=True,
                    label="Top-k",
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=5.0,
                    value=0.8,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )
    
        user_text.submit(
            run_generation,
            [user_text, top_p, temperature, top_k, max_new_tokens, performance],
            [model_output, performance],
        )
        button_submit.click(
            run_generation,
            [user_text, top_p, temperature, top_k, max_new_tokens, performance],
            [model_output, performance],
        )
        button_clear.click(
            reset_textbox,
            [user_text, model_output, performance],
            [user_text, model_output, performance],
        )
    
    if __name__ == "__main__":
        demo.queue()
        try:
            demo.launch(height=800)
        except Exception:
            demo.launch(share=True, height=800)
    
    # If you are launching remotely, specify server_name and server_port
    # EXAMPLE: `demo.launch(server_name='your server name', server_port='server port in int')`
    # To learn more please refer to the Gradio docs: https://gradio.app/docs/


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860
    
    To create a public link, set `share=True` in `launch()`.



.. raw:: html

    <div><iframe src="http://127.0.0.1:7860/" width="100%" height="800" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>

