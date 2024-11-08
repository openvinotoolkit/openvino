Create an LLM-powered Chatbot using OpenVINO
============================================

In the rapidly evolving world of artificial intelligence (AI), chatbots
have emerged as powerful tools for businesses to enhance customer
interactions and streamline operations. Large Language Models (LLMs) are
artificial intelligence systems that can understand and generate human
language. They use deep learning algorithms and massive amounts of data
to learn the nuances of language and produce coherent and relevant
responses. While a decent intent-based chatbot can answer basic,
one-touch inquiries like order management, FAQs, and policy questions,
LLM chatbots can tackle more complex, multi-touch questions. LLM enables
chatbots to provide support in a conversational manner, similar to how
humans do, through contextual memory. Leveraging the capabilities of
Language Models, chatbots are becoming increasingly intelligent, capable
of understanding and responding to human language with remarkable
accuracy.

Previously, we already discussed how to build an instruction-following
pipeline using OpenVINO and Optimum Intel, please check out `Dolly
example <240-dolly-2-instruction-following-with-output.html>`__ for reference. In this
tutorial, we consider how to use the power of OpenVINO for running Large
Language Models for chat. We will use a pre-trained model from the
`Hugging Face
Transformers <https://huggingface.co/docs/transformers/index>`__
library. To simplify the user experience, the `Hugging Face Optimum
Intel <https://huggingface.co/docs/optimum/intel/index>`__ library is
used to convert the models to OpenVINO™ IR format.

The tutorial consists of the following steps:

-  Install prerequisites
-  Download and convert the model from a public source using the
   `OpenVINO integration with Hugging Face
   Optimum <https://huggingface.co/blog/openvino>`__.
-  Compress model weights to 4-bit or 8-bit data types using
   `NNCF <https://github.com/openvinotoolkit/nncf>`__
-  Create a chat inference pipeline
-  Run chat pipeline

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prerequisites <#prerequisites>`__
-  `Select model for inference <#select-model-for-inference>`__
-  `login to huggingfacehub to get access to pretrained
   model <#login-to-huggingfacehub-to-get-access-to-pretrained-model>`__
-  `Instantiate Model using Optimum
   Intel <#instantiate-model-using-optimum-intel>`__
-  `Compress model weights <#compress-model-weights>`__

   -  `Weights Compression using Optimum
      Intel <#weights-compression-using-optimum-intel>`__
   -  `Weights Compression using
      NNCF <#weights-compression-using-nncf>`__

-  `Select device for inference and model
   variant <#select-device-for-inference-and-model-variant>`__
-  `Run Chatbot <#run-chatbot>`__

Prerequisites
-------------



Install required dependencies

.. code:: ipython3

    %pip uninstall -q -y openvino-dev openvino openvino-nightly optimum optimum-intel
    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu\
    "git+https://github.com/huggingface/optimum-intel.git"\
    "nncf>=2.8.0"\
    "datasets" \
    "accelerate"\
    "openvino-nightly"\
    "gradio"\
    "onnx" "einops" "transformers_stream_generator" "tiktoken" "transformers>=4.36.0"

Select model for inference
--------------------------



The tutorial supports different models, you can select one from the
provided options to compare the quality of open source LLM solutions.

   **NOTE**: conversion of some models can require additional actions
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

-  **red-pajama-3b-chat** - A 2.8B parameter pre-trained language model
   based on GPT-NEOX architecture. It was developed by Together Computer
   and leaders from the open-source AI community. The model is
   fine-tuned on OASST1 and Dolly2 datasets to enhance chatting ability.
   More details about model can be found in `HuggingFace model
   card <https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1>`__.
-  **llama-2-7b-chat** - LLama 2 is the second generation of LLama
   models developed by Meta. Llama 2 is a collection of pre-trained and
   fine-tuned generative text models ranging in scale from 7 billion to
   70 billion parameters. llama-2-7b-chat is 7 billions parameters
   version of LLama 2 finetuned and optimized for dialogue use case.
   More details about model can be found in the
   `paper <https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/>`__,
   `repository <https://github.com/facebookresearch/llama>`__ and
   `HuggingFace model
   card <https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>`__

   **NOTE**: run model with demo, you will need to accept license
   agreement. You must be a registered user in Hugging Face Hub.
   Please visit `HuggingFace model
   card <https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>`__,
   carefully read terms of usage and click accept button. You will need
   to use an access token for the code below to run. For more
   information on access tokens, refer to `this section of the
   documentation <https://huggingface.co/docs/hub/security-tokens>`__.
   You can login on Hugging Face Hub in notebook environment, using
   following code:

.. code:: python

       ## login to huggingfacehub to get access to pretrained model

       from huggingface_hub import notebook_login, whoami

       try:
           whoami()
           print('Authorization token already provided')
       except OSError:
           notebook_login()

-  **mpt-7b-chat** - MPT-7B is part of the family of
   MosaicPretrainedTransformer (MPT) models, which use a modified
   transformer architecture optimized for efficient training and
   inference. These architectural changes include performance-optimized
   layer implementations and the elimination of context length limits by
   replacing positional embeddings with Attention with Linear Biases
   (`ALiBi <https://arxiv.org/abs/2108.12409>`__). Thanks to these
   modifications, MPT models can be trained with high throughput
   efficiency and stable convergence. MPT-7B-chat is a chatbot-like
   model for dialogue generation. It was built by finetuning MPT-7B on
   the
   `ShareGPT-Vicuna <https://huggingface.co/datasets/jeffwan/sharegpt_vicuna>`__,
   `HC3 <https://huggingface.co/datasets/Hello-SimpleAI/HC3>`__,
   `Alpaca <https://huggingface.co/datasets/tatsu-lab/alpaca>`__,
   `HH-RLHF <https://huggingface.co/datasets/Anthropic/hh-rlhf>`__, and
   `Evol-Instruct <https://huggingface.co/datasets/victor123/evol_instruct_70k>`__
   datasets. More details about the model can be found in `blog
   post <https://www.mosaicml.com/blog/mpt-7b>`__,
   `repository <https://github.com/mosaicml/llm-foundry/>`__ and
   `HuggingFace model
   card <https://huggingface.co/mosaicml/mpt-7b-chat>`__.
-  **qwen-7b-chat** - Qwen-7B is the 7B-parameter version of the large
   language model series, Qwen (abbr. Tongyi Qianwen), proposed by
   Alibaba Cloud. Qwen-7B is a Transformer-based large language model,
   which is pretrained on a large volume of data, including web texts,
   books, codes, etc. For more details about Qwen, please refer to the
   `GitHub <https://github.com/QwenLM/Qwen>`__ code repository.
-  **chatglm3-6b** - ChatGLM3-6B is the latest open-source model in the
   ChatGLM series. While retaining many excellent features such as
   smooth dialogue and low deployment threshold from the previous two
   generations, ChatGLM3-6B employs a more diverse training dataset,
   more sufficient training steps, and a more reasonable training
   strategy. ChatGLM3-6B adopts a newly designed `Prompt
   format <https://github.com/THUDM/ChatGLM3/blob/main/PROMPT_en.md>`__,
   in addition to the normal multi-turn dialogue. You can find more
   details about model in the `model
   card <https://huggingface.co/THUDM/chatglm3-6b>`__
-  **mistral-7b** - The Mistral-7B-v0.1 Large Language Model (LLM) is a
   pretrained generative text model with 7 billion parameters. You can
   find more details about model in the `model
   card <https://huggingface.co/mistralai/Mistral-7B-v0.1>`__,
   `paper <https://arxiv.org/abs/2310.06825>`__ and `release blog
   post <https://mistral.ai/news/announcing-mistral-7b/>`__.
-  **zephyr-7b-beta** - Zephyr is a series of language models that are
   trained to act as helpful assistants. Zephyr-7B-beta is the second
   model in the series, and is a fine-tuned version of
   `mistralai/Mistral-7B-v0.1 <https://huggingface.co/mistralai/Mistral-7B-v0.1>`__
   that was trained on on a mix of publicly available, synthetic
   datasets using `Direct Preference Optimization
   (DPO) <https://arxiv.org/abs/2305.18290>`__. You can find more
   details about model in `technical
   report <https://arxiv.org/abs/2310.16944>`__ and `HuggingFace model
   card <https://huggingface.co/HuggingFaceH4/zephyr-7b-beta>`__.
-  **neural-chat-7b-v3-1** - Mistral-7b model fine-tuned using Intel
   Gaudi. The model fine-tuned on the open source dataset
   `Open-Orca/SlimOrca <https://huggingface.co/datasets/Open-Orca/SlimOrca>`__
   and aligned with `Direct Preference Optimization (DPO)
   algorithm <https://arxiv.org/abs/2305.18290>`__. More details can be
   found in `model
   card <https://huggingface.co/Intel/neural-chat-7b-v3-3>`__ and `blog
   post <https://medium.com/@NeuralCompressor/the-practice-of-supervised-finetuning-and-direct-preference-optimization-on-habana-gaudi2-a1197d8a3cd3>`__.
-  **notus-7b-v1** - Notus is a collection of fine-tuned models using
   `Direct Preference Optimization
   (DPO) <https://arxiv.org/abs/2305.18290>`__. and related
   `RLHF <https://huggingface.co/blog/rlhf>`__ techniques. This model is
   the first version, fine-tuned with DPO over zephyr-7b-sft. Following
   a data-first approach, the only difference between Notus-7B-v1 and
   Zephyr-7B-beta is the preference dataset used for dDPO. Proposed
   approach for dataset creation helps to effectively fine-tune Notus-7b
   that surpasses Zephyr-7B-beta and Claude 2 on
   `AlpacaEval <https://tatsu-lab.github.io/alpaca_eval/>`__. More
   details about model can be found in `model
   card <https://huggingface.co/argilla/notus-7b-v1>`__.
-  **youri-7b-chat** - Youri-7b-chat is a Llama2 based model. `Rinna
   Co., Ltd. <https://rinna.co.jp/>`__ conducted further pre-training
   for the Llama2 model with a mixture of English and Japanese datasets
   to improve Japanese task capability. The model is publicly released
   on Hugging Face hub. You can find detailed information at the
   `rinna/youri-7b-chat project
   page <https://huggingface.co/rinna/youri-7b>`__.

.. code:: ipython3

    from config import SUPPORTED_LLM_MODELS
    import ipywidgets as widgets

.. code:: ipython3

    model_ids = list(SUPPORTED_LLM_MODELS)

    model_id = widgets.Dropdown(
        options=model_ids,
        value=model_ids[0],
        description="Model:",
        disabled=False,
    )

    model_id




.. parsed-literal::

    Dropdown(description='Model:', options=('tiny-llama-1b-chat', 'red-pajama-3b-chat', 'llama-2-chat-7b', 'mpt-7b…



.. code:: ipython3

    model_configuration = SUPPORTED_LLM_MODELS[model_id.value]
    print(f"Selected model {model_id.value}")


.. parsed-literal::

    Selected model tiny-llama-1b-chat


Instantiate Model using Optimum Intel
-------------------------------------



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
method. When downloading and converting Transformers model, the
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

In our case, MPT, Qwen and ChatGLM model currently is not covered by
Optimum Intel, we will convert it manually and create wrapper compatible
with Optimum Intel.

.. code:: ipython3

    from transformers import AutoModelForCausalLM, AutoConfig
    from optimum.intel import OVQuantizer
    from optimum.intel.openvino import OVModelForCausalLM
    import openvino as ov
    from pathlib import Path
    import shutil
    import torch
    import logging
    import nncf
    import gc
    from converter import converters


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino


.. parsed-literal::

    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/torch/cuda/__init__.py:138: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 11080). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.)
      return torch._C._cuda_getDeviceCount() > 0
    No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'


Compress model weights
----------------------



The Weights Compression algorithm is aimed at compressing the weights of
the models and can be used to optimize the model footprint and
performance of large models where the size of weights is relatively
larger than the size of activations, for example, Large Language Models
(LLM). Compared to INT8 compression, INT4 compression improves
performance even more, but introduces a minor drop in prediction
quality.

Weights Compression using Optimum Intel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



To enable weights compression via NNCF for models supported by Optimum
Intel ``OVQuantizer`` class should be used for ``OVModelForCausalLM``
model.
``OVQuantizer.quantize(save_directory=save_dir, weights_only=True)``
enables weights compression. We will consider how to do it on RedPajama,
LLAMA and Zephyr examples.

   **NOTE**: Weights Compression using Optimum Intel currently supports
   only INT8 compression. We will apply INT4 compression for these model
   using NNCF API described below.

..

   **NOTE**: There may be no speedup for INT4/INT8 compressed models on
   dGPU.

Weights Compression using NNCF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



You also can perform weights compression for OpenVINO models using NNCF
directly. ``nncf.compress_weights`` function accepts OpenVINO model
instance and compresses its weights for Linear and Embedding layers. We
will consider this variant based on MPT model.

   **NOTE**: This tutorial involves conversion model for FP16 and
   INT4/INT8 weights compression scenarios. It may be memory and
   time-consuming in the first run. You can manually control the
   compression precision below.

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


We can now save floating point and compressed model variants

.. code:: ipython3

    nncf.set_log_level(logging.ERROR)

    pt_model_id = model_configuration["model_id"]
    pt_model_name = model_id.value.split("-")[0]
    model_type = AutoConfig.from_pretrained(pt_model_id, trust_remote_code=True).model_type
    fp16_model_dir = Path(model_id.value) / "FP16"
    int8_model_dir = Path(model_id.value) / "INT8_compressed_weights"
    int4_model_dir = Path(model_id.value) / "INT4_compressed_weights"


    def convert_to_fp16():
        if (fp16_model_dir / "openvino_model.xml").exists():
            return
        if not model_configuration["remote"]:
            ov_model = OVModelForCausalLM.from_pretrained(
                pt_model_id, export=True, compile=False, load_in_8bit=False
            )
            ov_model.half()
            ov_model.save_pretrained(fp16_model_dir)
            del ov_model
        else:
            model_kwargs = {}
            if "revision" in model_configuration:
                model_kwargs["revision"] = model_configuration["revision"]
            model = AutoModelForCausalLM.from_pretrained(
                model_configuration["model_id"],
                torch_dtype=torch.float32,
                trust_remote_code=True,
                **model_kwargs
            )
            converters[pt_model_name](model, fp16_model_dir)
            del model
        gc.collect()


    def convert_to_int8():
        if (int8_model_dir / "openvino_model.xml").exists():
            return
        int8_model_dir.mkdir(parents=True, exist_ok=True)
        if not model_configuration["remote"]:
            if fp16_model_dir.exists():
                ov_model = OVModelForCausalLM.from_pretrained(fp16_model_dir, compile=False)
            else:
                ov_model = OVModelForCausalLM.from_pretrained(
                    pt_model_id, export=True, compile=False, load_in_8bit=False
                )
                ov_model.half()
            quantizer = OVQuantizer.from_pretrained(ov_model)
            quantizer.quantize(save_directory=int8_model_dir, weights_only=True)
            del quantizer
            del ov_model
        else:
            convert_to_fp16()
            ov_model = ov.Core().read_model(fp16_model_dir / "openvino_model.xml")
            shutil.copy(fp16_model_dir / "config.json", int8_model_dir / "config.json")
            configuration_file = fp16_model_dir / f"configuration_{model_type}.py"
            if configuration_file.exists():
                shutil.copy(
                    configuration_file, int8_model_dir / f"configuration_{model_type}.py"
                )
            compressed_model = nncf.compress_weights(ov_model)
            ov.save_model(compressed_model, int8_model_dir / "openvino_model.xml")
            del ov_model
            del compressed_model
        gc.collect()


    def convert_to_int4():
        compression_configs = {
            "zephyr-7b-beta": {
                "mode": nncf.CompressWeightsMode.INT4_SYM,
                "group_size": 64,
                "ratio": 0.6,
            },
            "mistral-7b": {
                "mode": nncf.CompressWeightsMode.INT4_SYM,
                "group_size": 64,
                "ratio": 0.6,
            },
            "notus-7b-v1": {
                "mode": nncf.CompressWeightsMode.INT4_SYM,
                "group_size": 64,
                "ratio": 0.6,
            },
            "neural-chat-7b-v3-1": {
                "mode": nncf.CompressWeightsMode.INT4_SYM,
                "group_size": 64,
                "ratio": 0.6,
            },
            "llama-2-chat-7b": {
                "mode": nncf.CompressWeightsMode.INT4_SYM,
                "group_size": 128,
                "ratio": 0.8,
            },
            "chatglm2-6b": {
                "mode": nncf.CompressWeightsMode.INT4_SYM,
                "group_size": 128,
                "ratio": 0.72,
            },
            "qwen-7b-chat": {
                "mode": nncf.CompressWeightsMode.INT4_SYM,
                "group_size": 128,
                "ratio": 0.6
            },
            'red-pajama-3b-chat': {
                "mode": nncf.CompressWeightsMode.INT4_ASYM,
                "group_size": 128,
                "ratio": 0.5,
            },
            "default": {
                "mode": nncf.CompressWeightsMode.INT4_ASYM,
                "group_size": 128,
                "ratio": 0.8,
            },
        }

        model_compression_params = compression_configs.get(
            model_id.value, compression_configs["default"]
        )
        if (int4_model_dir / "openvino_model.xml").exists():
            return
        int4_model_dir.mkdir(parents=True, exist_ok=True)
        if not model_configuration["remote"]:
            if not fp16_model_dir.exists():
                model = OVModelForCausalLM.from_pretrained(
                    pt_model_id, export=True, compile=False, load_in_8bit=False
                ).half()
                model.config.save_pretrained(int4_model_dir)
                ov_model = model._original_model
                del model
                gc.collect()
            else:
                ov_model = ov.Core().read_model(fp16_model_dir / "openvino_model.xml")
                shutil.copy(fp16_model_dir / "config.json", int4_model_dir / "config.json")

        else:
            convert_to_fp16()
            ov_model = ov.Core().read_model(fp16_model_dir / "openvino_model.xml")
            shutil.copy(fp16_model_dir / "config.json", int4_model_dir / "config.json")
            configuration_file = fp16_model_dir / f"configuration_{model_type}.py"
            if configuration_file.exists():
                shutil.copy(
                    configuration_file, int4_model_dir / f"configuration_{model_type}.py"
                )
        compressed_model = nncf.compress_weights(ov_model, **model_compression_params)
        ov.save_model(compressed_model, int4_model_dir / "openvino_model.xml")
        del ov_model
        del compressed_model
        gc.collect()


    if prepare_fp16_model.value:
        convert_to_fp16()
    if prepare_int8_model.value:
        convert_to_int8()
    if prepare_int4_model.value:
        convert_to_int4()

Let’s compare model size for different compression types

.. code:: ipython3

    fp16_weights = fp16_model_dir / "openvino_model.bin"
    int8_weights = int8_model_dir / "openvino_model.bin"
    int4_weights = int4_model_dir / "openvino_model.bin"

    if fp16_weights.exists():
        print(f"Size of FP16 model is {fp16_weights.stat().st_size / 1024 / 1024:.2f} MB")
    for precision, compressed_weights in zip([8, 4], [int8_weights, int4_weights]):
        if compressed_weights.exists():
            print(
                f"Size of model with INT{precision} compressed weights is {compressed_weights.stat().st_size / 1024 / 1024:.2f} MB"
            )
        if compressed_weights.exists() and fp16_weights.exists():
            print(
                f"Compression rate for INT{precision} model: {fp16_weights.stat().st_size / compressed_weights.stat().st_size:.3f}"
            )


.. parsed-literal::

    Size of FP16 model is 2098.68 MB
    Size of model with INT8 compressed weights is 1050.99 MB
    Compression rate for INT8 model: 1.997
    Size of model with INT4 compressed weights is 696.99 MB
    Compression rate for INT4 model: 3.011


Select device for inference and model variant
---------------------------------------------



   **NOTE**: There may be no speedup for INT4/INT8 compressed models on
   dGPU.

.. code:: ipython3

    core = ov.Core()
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value="CPU",
        description="Device:",
        disabled=False,
    )

    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'GPU.0', 'GPU.1', 'AUTO'), value='CPU')



The cell below create ``OVMPTModel``, ``OVQWENModel`` and
``OVCHATGLM2Model`` wrapper based on ``OVModelForCausalLM`` model.

.. code:: ipython3

    from ov_llm_model import model_classes

The cell below demonstrates how to instantiate model based on selected
variant of model weights and inference device

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

    Dropdown(description='Model to run:', options=('INT4', 'INT8', 'FP16'), value='INT4')



.. code:: ipython3

    from transformers import AutoTokenizer

    if model_to_run.value == "INT4":
        model_dir = int4_model_dir
    elif model_to_run.value == "INT8":
        model_dir = int8_model_dir
    else:
        model_dir = fp16_model_dir
    print(f"Loading model from {model_dir}")

    ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}

    # On a GPU device a model is executed in FP16 precision. For red-pajama-3b-chat model there known accuracy
    # issues caused by this, which we avoid by setting precision hint to "f32".
    if model_id.value == "red-pajama-3b-chat" and "GPU" in core.available_devices and device.value in ["GPU", "AUTO"]:
        ov_config["INFERENCE_PRECISION_HINT"] = "f32"

    model_name = model_configuration["model_id"]
    class_key = model_id.value.split("-")[0]
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model_class = (
        OVModelForCausalLM
        if not model_configuration["remote"]
        else model_classes[class_key]
    )
    ov_model = model_class.from_pretrained(
        model_dir,
        device=device.value,
        ov_config=ov_config,
        config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
        trust_remote_code=True,
    )


.. parsed-literal::

    Loading model from tiny-llama-1b-chat/INT4_compressed_weights


.. parsed-literal::

    The argument `trust_remote_code` is to be used along with export=True. It will be ignored.
    Compiling the model to CPU ...


.. code:: ipython3

    tokenizer_kwargs = model_configuration.get("tokenizer_kwargs", {})
    test_string = "2 + 2 ="
    input_tokens = tok(test_string, return_tensors="pt", **tokenizer_kwargs)
    answer = ov_model.generate(**input_tokens, max_new_tokens=2)
    print(tok.batch_decode(answer, skip_special_tokens=True)[0])


.. parsed-literal::

    Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.


.. parsed-literal::

    2 + 2 = 4


Run Chatbot
-----------



Now, when model created, we can setup Chatbot interface using
`Gradio <https://www.gradio.app/>`__. The diagram below illustrates how
the chatbot pipeline works

.. figure:: https://user-images.githubusercontent.com/29454499/255523209-d9336491-c7ba-4dc1-98f0-07f23743ce89.png
   :alt: generation pipeline

   generation pipeline

As can be seen, the pipeline very similar to instruction-following with
only changes that previous conversation history additionally passed as
input with next user question for getting wider input context. On the
first iteration, the user provided instructions joined to conversation
history (if exists) converted to token ids using a tokenizer, then
prepared input provided to the model. The model generates probabilities
for all tokens in logits format The way the next token will be selected
over predicted probabilities is driven by the selected decoding
methodology. You can find more information about the most popular
decoding methods in this
`blog <https://huggingface.co/blog/how-to-generate>`__. The result
generation updates conversation history for next conversation step. it
makes stronger connection of next question with previously provided and
allows user to make clarifications regarding previously provided
answers.

| There are several parameters that can control text generation quality:
  \* ``Temperature`` is a parameter used to control the level of
  creativity in AI-generated text. By adjusting the ``temperature``, you
  can influence the AI model’s probability distribution, making the text
  more focused or diverse.
| Consider the following example: The AI model has to complete the
  sentence “The cat is \____.” with the following token probabilities:

::

   playing: 0.5
   sleeping: 0.25
   eating: 0.15
   driving: 0.05
   flying: 0.05

   - **Low temperature** (e.g., 0.2): The AI model becomes more focused and deterministic, choosing tokens with the highest probability, such as "playing."
   - **Medium temperature** (e.g., 1.0): The AI model maintains a balance between creativity and focus, selecting tokens based on their probabilities without significant bias, such as "playing," "sleeping," or "eating."
   - **High temperature** (e.g., 2.0): The AI model becomes more adventurous, increasing the chances of selecting less likely tokens, such as "driving" and "flying."

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

-  ``Top-k`` is an another popular sampling strategy. In comparison with
   Top-P, which chooses from the smallest possible set of words whose
   cumulative probability exceeds the probability P, in Top-K sampling K
   most likely next words are filtered and the probability mass is
   redistributed among only those K next words. In our example with cat,
   if k=3, then only “playing”, “sleeping” and “eating” will be taken
   into account as possible next word.
-  ``Repetition Penalty`` This parameter can help penalize tokens based
   on how frequently they occur in the text, including the input prompt.
   A token that has already appeared five times is penalized more
   heavily than a token that has appeared only one time. A value of 1
   means that there is no penalty and values larger than 1 discourage
   repeated tokens.

.. code:: ipython3

    from threading import Event, Thread
    from uuid import uuid4
    from typing import List, Tuple
    import gradio as gr
    from transformers import (
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
        TextIteratorStreamer,
    )


    model_name = model_configuration["model_id"]
    start_message = model_configuration["start_message"]
    history_template = model_configuration.get("history_template")
    current_message_template = model_configuration.get("current_message_template")
    stop_tokens = model_configuration.get("stop_tokens")
    roles = model_configuration.get("roles")
    tokenizer_kwargs = model_configuration.get("tokenizer_kwargs", {})

    chinese_examples = [
        ["你好!"],
        ["你是谁?"],
        ["请介绍一下上海"],
        ["请介绍一下英特尔公司"],
        ["晚上睡不着怎么办？"],
        ["给我讲一个年轻人奋斗创业最终取得成功的故事。"],
        ["给这个故事起一个标题。"],
    ]

    english_examples = [
        ["Hello there! How are you doing?"],
        ["What is OpenVINO?"],
        ["Who are you?"],
        ["Can you explain to me briefly what is Python programming language?"],
        ["Explain the plot of Cinderella in a sentence."],
        ["What are some common mistakes to avoid when writing code?"],
        [
            "Write a 100-word blog post on “Benefits of Artificial Intelligence and OpenVINO“"
        ],
    ]

    japanese_examples = [
        ["こんにちは！調子はどうですか?"],
        ["OpenVINOとは何ですか?"],
        ["あなたは誰ですか?"],
        ["Pythonプログラミング言語とは何か簡単に説明してもらえますか?"],
        ["シンデレラのあらすじを一文で説明してください。"],
        ["コードを書くときに避けるべきよくある間違いは何ですか?"],
        ["人工知能と「OpenVINOの利点」について100語程度のブログ記事を書いてください。"],
    ]

    examples = (
        chinese_examples
        if ("qwen" in model_id.value or "chatglm" in model_id.value)
        else japanese_examples
        if ("youri" in model_id.value)
        else english_examples
    )

    max_new_tokens = 256


    class StopOnTokens(StoppingCriteria):
        def __init__(self, token_ids):
            self.token_ids = token_ids

        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            for stop_id in self.token_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False


    if stop_tokens is not None:
        if isinstance(stop_tokens[0], str):
            stop_tokens = tok.convert_tokens_to_ids(stop_tokens)

        stop_tokens = [StopOnTokens(stop_tokens)]


    def default_partial_text_processor(partial_text: str, new_text: str):
        """
        helper for updating partially generated answer, used by default

        Params:
          partial_text: text buffer for storing previosly generated text
          new_text: text update for the current step
        Returns:
          updated text string

        """
        partial_text += new_text
        return partial_text


    text_processor = model_configuration.get(
        "partial_text_processor", default_partial_text_processor
    )


    def convert_history_to_token(history: List[Tuple[str, str]], roles=None):
        """
        function for conversion history stored as list pairs of user and assistant messages to tokens according to model expected conversation template
        Params:
          history: dialogue history
        Returns:
          history in token format
        """
        if roles is None:
            text = start_message + "".join(
                [
                    "".join(
                        [
                            history_template.format(
                                num=round, user=item[0], assistant=item[1]
                            )
                        ]
                    )
                    for round, item in enumerate(history[:-1])
                ]
            )
            text += "".join(
                [
                    "".join(
                        [
                            current_message_template.format(
                                num=len(history) + 1,
                                user=history[-1][0],
                                assistant=history[-1][1],
                            )
                        ]
                    )
                ]
            )
            input_token = tok(text, return_tensors="pt", **tokenizer_kwargs).input_ids
        else:
            input_ids = []
            input_ids.extend(tok.build_single_message(roles[0], "", start_message))
            for old_query, response in history[:-1]:
                input_ids.extend(tok.build_single_message(roles[1], "", old_query))
                input_ids.extend(tok.build_single_message(roles[2], "", response))
            input_ids.extend(tok.build_single_message(roles[1], "", history[-1][0]))
            input_ids.extend([tok.get_command(f"<|{roles[2]}|>")])
            input_token = tok.batch_encode_plus(
                [input_ids], return_tensors="pt", is_split_into_words=True
            ).input_ids
        return input_token


    def user(message, history):
        """
        callback function for updating user messages in interface on submit button click

        Params:
          message: current message
          history: conversation history
        Returns:
          None
        """
        # Append the user's message to the conversation history
        return "", history + [[message, ""]]


    def bot(history, temperature, top_p, top_k, repetition_penalty, conversation_id):
        """
        callback function for running chatbot on submit button click

        Params:
          history: conversation history
          temperature:  parameter for control the level of creativity in AI-generated text.
                        By adjusting the `temperature`, you can influence the AI model's probability distribution, making the text more focused or diverse.
          top_p: parameter for control the range of tokens considered by the AI model based on their cumulative probability.
          top_k: parameter for control the range of tokens considered by the AI model based on their cumulative probability, selecting number of tokens with highest probability.
          repetition_penalty: parameter for penalizing tokens based on how frequently they occur in the text.
          conversation_id: unique conversation identifier.

        """

        # Construct the input message string for the model by concatenating the current system message and conversation history
        # Tokenize the messages string
        input_ids = convert_history_to_token(history, roles)
        if input_ids.shape[1] > 2000:
            history = [history[-1]]
            input_ids = convert_history_to_token(history, roles)
        streamer = TextIteratorStreamer(
            tok, timeout=30.0, skip_prompt=True, skip_special_tokens=True
        )
        generate_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
        )
        if stop_tokens is not None:
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList(stop_tokens)

        stream_complete = Event()

        def generate_and_signal_complete():
            """
            genration function for single thread
            """
            global start_time
            ov_model.generate(**generate_kwargs)
            stream_complete.set()

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()

        # Initialize an empty string to store the generated text
        partial_text = ""
        for new_text in streamer:
            partial_text = text_processor(partial_text, new_text)
            history[-1][1] = partial_text
            yield history


    def get_uuid():
        """
        universal unique identifier for thread
        """
        return str(uuid4())


    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=".disclaimer {font-variant-caps: all-small-caps;}",
    ) as demo:
        conversation_id = gr.State(get_uuid)
        gr.Markdown(f"""<h1><center>OpenVINO {model_id.value} Chatbot</center></h1>""")
        chatbot = gr.Chatbot(height=500)
        with gr.Row():
            with gr.Column():
                msg = gr.Textbox(
                    label="Chat Message Box",
                    placeholder="Chat Message Box",
                    show_label=False,
                    container=False,
                )
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit")
                    stop = gr.Button("Stop")
                    clear = gr.Button("Clear")
        with gr.Row():
            with gr.Accordion("Advanced Options:", open=False):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            temperature = gr.Slider(
                                label="Temperature",
                                value=0.1,
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                interactive=True,
                                info="Higher values produce more diverse outputs",
                            )
                    with gr.Column():
                        with gr.Row():
                            top_p = gr.Slider(
                                label="Top-p (nucleus sampling)",
                                value=1.0,
                                minimum=0.0,
                                maximum=1,
                                step=0.01,
                                interactive=True,
                                info=(
                                    "Sample from the smallest possible set of tokens whose cumulative probability "
                                    "exceeds top_p. Set to 1 to disable and sample from all tokens."
                                ),
                            )
                    with gr.Column():
                        with gr.Row():
                            top_k = gr.Slider(
                                label="Top-k",
                                value=50,
                                minimum=0.0,
                                maximum=200,
                                step=1,
                                interactive=True,
                                info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                            )
                    with gr.Column():
                        with gr.Row():
                            repetition_penalty = gr.Slider(
                                label="Repetition Penalty",
                                value=1.1,
                                minimum=1.0,
                                maximum=2.0,
                                step=0.1,
                                interactive=True,
                                info="Penalize repetition — 1.0 to disable.",
                            )
        gr.Examples(
            examples, inputs=msg, label="Click on any example and press the 'Submit' button"
        )

        submit_event = msg.submit(
            fn=user,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            fn=bot,
            inputs=[
                chatbot,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                conversation_id,
            ],
            outputs=chatbot,
            queue=True,
        )
        submit_click_event = submit.click(
            fn=user,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            fn=bot,
            inputs=[
                chatbot,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                conversation_id,
            ],
            outputs=chatbot,
            queue=True,
        )
        stop.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[submit_event, submit_click_event],
            queue=False,
        )
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue(max_size=2)
    # if you are launching remotely, specify server_name and server_port
    #  demo.launch(server_name='your server name', server_port='server port in int')
    # if you have any issue to launch on your platform, you can pass share=True to launch method:
    # demo.launch(share=True)
    # it creates a publicly shareable link for the interface. Read more in the docs: https://gradio.app/docs/
    demo.launch()

.. code:: ipython3

    # please run this cell for stopping gradio interface
    demo.close()
