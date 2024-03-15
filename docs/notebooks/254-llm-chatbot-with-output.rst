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
    "git+https://github.com/openvinotoolkit/nncf.git"\
    "datasets" \
    "accelerate"\
    "openvino-nightly"\
    "gradio"\
    "onnx" "einops" "transformers_stream_generator" "tiktoken" "transformers>=4.38.1" "bitsandbytes"


.. parsed-literal::

    WARNING: Skipping openvino-dev as it is not installed.
    WARNING: Skipping openvino as it is not installed.
    WARNING: Skipping openvino-nightly as it is not installed.
    WARNING: Skipping optimum as it is not installed.
    WARNING: Skipping optimum-intel as it is not installed.
    Note: you may need to restart the kernel to use updated packages.
    
    [notice] A new release of pip is available: 23.1.2 -> 24.0
    [notice] To update, run: pip install --upgrade pip
    Note: you may need to restart the kernel to use updated packages.


Select model for inference
--------------------------



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
-  **mini-cpm-2b-dpo** - MiniCPM is an End-Size LLM developed by
   ModelBest Inc. and TsinghuaNLP, with only 2.4B parameters excluding
   embeddings. After Direct Preference Optimization (DPO) fine-tuning,
   MiniCPM outperforms many popular 7b, 13b and 70b models. More details
   can be found in
   `model_card <https://huggingface.co/openbmb/MiniCPM-2B-dpo-fp16>`__.
-  **gemma-2b-it** - Gemma is a family of lightweight, state-of-the-art
   open models from Google, built from the same research and technology
   used to create the Gemini models. They are text-to-text, decoder-only
   large language models, available in English, with open weights,
   pre-trained variants, and instruction-tuned variants. Gemma models
   are well-suited for a variety of text generation tasks, including
   question answering, summarization, and reasoning. This model is
   instruction-tuned version of 2B parameters model. More details about
   model can be found in `model
   card <https://huggingface.co/google/gemma-2b-it>`__. >\ **Note**: run
   model with demo, you will need to accept license agreement. >You must
   be a registered user in Hugging Face Hub. Please visit `HuggingFace
   model card <https://huggingface.co/google/gemma-2b-it>`__, carefully
   read terms of usage and click accept button. You will need to use an
   access token for the code below to run. For more information on
   access tokens, refer to `this section of the
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

-  **red-pajama-3b-chat** - A 2.8B parameter pre-trained language model
   based on GPT-NEOX architecture. It was developed by Together Computer
   and leaders from the open-source AI community. The model is
   fine-tuned on OASST1 and Dolly2 datasets to enhance chatting ability.
   More details about model can be found in `HuggingFace model
   card <https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1>`__.
-  **gemma-7b-it** - Gemma is a family of lightweight, state-of-the-art
   open models from Google, built from the same research and technology
   used to create the Gemini models. They are text-to-text, decoder-only
   large language models, available in English, with open weights,
   pre-trained variants, and instruction-tuned variants. Gemma models
   are well-suited for a variety of text generation tasks, including
   question answering, summarization, and reasoning. This model is
   instruction-tuned version of 7B parameters model. More details about
   model can be found in `model
   card <https://huggingface.co/google/gemma-7b-it>`__. >\ **Note**: run
   model with demo, you will need to accept license agreement. >You must
   be a registered user in Hugging Face Hub. Please visit `HuggingFace
   model card <https://huggingface.co/google/gemma-7b-it>`__, carefully
   read terms of usage and click accept button. You will need to use an
   access token for the code below to run. For more information on
   access tokens, refer to `this section of the
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

-  **llama-2-7b-chat** - LLama 2 is the second generation of LLama
   models developed by Meta. Llama 2 is a collection of pre-trained and
   fine-tuned generative text models ranging in scale from 7 billion to
   70 billion parameters. llama-2-7b-chat is 7 billions parameters
   version of LLama 2 finetuned and optimized for dialogue use case.
   More details about model can be found in the
   `paper <https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/>`__,
   `repository <https://github.com/facebookresearch/llama>`__ and
   `HuggingFace model
   card <https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>`__.
   >\ **Note**: run model with demo, you will need to accept license
   agreement. >You must be a registered user in Hugging Face Hub.
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

-  **qwen1.5-0.5b-chat/qwen1.5-1.8b-chat/qwen1.5-7b-chat** - Qwen1.5 is
   the beta version of Qwen2, a transformer-based decoder-only language
   model pretrained on a large amount of data. Qwen1.5 is a language
   model series including decoder language models of different model
   sizes. It is based on the Transformer architecture with SwiGLU
   activation, attention QKV bias, group query attention, mixture of
   sliding window attention and full attention. You can find more
   details about model in the `model
   repository <https://huggingface.co/Qwen>`__.
-  **qwen-7b-chat** - Qwen-7B is the 7B-parameter version of the large
   language model series, Qwen (abbr. Tongyi Qianwen), proposed by
   Alibaba Cloud. Qwen-7B is a Transformer-based large language model,
   which is pretrained on a large volume of data, including web texts,
   books, codes, etc. For more details about Qwen, please refer to the
   `GitHub <https://github.com/QwenLM/Qwen>`__ code repository.
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
   card <https://huggingface.co/Intel/neural-chat-7b-v3-1>`__ and `blog
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
-  **baichuan2-7b-chat** - Baichuan 2 is the new generation of
   large-scale open-source language models launched by `Baichuan
   Intelligence inc <https://www.baichuan-ai.com/home>`__. It is trained
   on a high-quality corpus with 2.6 trillion tokens and has achieved
   the best performance in authoritative Chinese and English benchmarks
   of the same size.
-  **internlm2-chat-1.8b** - InternLM2 is the second generation InternLM
   series. Compared to the previous generation model, it shows
   significant improvements in various capabilities, including
   reasoning, mathematics, and coding. More details about model can be
   found in `model repository <https://huggingface.co/internlm>`__.

.. code:: ipython3

    from config import SUPPORTED_LLM_MODELS
    import ipywidgets as widgets

.. code:: ipython3

    model_languages = list(SUPPORTED_LLM_MODELS)
    
    model_language = widgets.Dropdown(
        options=model_languages,
        value=model_languages[0],
        description="Model Language:",
        disabled=False,
    )
    
    model_language




.. parsed-literal::

    Dropdown(description='Model Language:', options=('English', 'Chinese', 'Japanese'), value='English')



.. code:: ipython3

    model_ids = list(SUPPORTED_LLM_MODELS[model_language.value])
    
    model_id = widgets.Dropdown(
        options=model_ids,
        value=model_ids[0],
        description="Model:",
        disabled=False,
    )
    
    model_id




.. parsed-literal::

    Dropdown(description='Model:', options=('qwen1.5-0.5b-chat', 'qwen1.5-1.8b-chat', 'qwen1.5-7b-chat', 'qwen-7b-…



.. code:: ipython3

    model_configuration = SUPPORTED_LLM_MODELS[model_language.value][model_id.value]
    print(f"Selected model {model_id.value}")


.. parsed-literal::

    Selected model qwen-7b-chat


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
    from optimum.intel.openvino import OVModelForCausalLM
    import openvino as ov
    from pathlib import Path
    import shutil
    import torch
    import logging
    import nncf
    import gc
    from converter import converters, register_configs
    
    register_configs()


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    2024-03-07 02:52:02.115283: I tensorflow/core/util/port.cc:111] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-03-07 02:52:02.118993: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-03-07 02:52:02.161204: E tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:9342] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-03-07 02:52:02.161239: E tensorflow/compiler/xla/stream_executor/cuda/cuda_fft.cc:609] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-03-07 02:52:02.161273: E tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:1518] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-03-07 02:52:02.169740: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-03-07 02:52:02.171079: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-03-07 02:52:03.108737: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


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



Optimum Intel supports weight compression via NNCF out of the box. For
8-bit compression we pass ``load_in_8bit=True`` to ``from_pretrained()``
method of ``OVModelForCausalLM``. For 4 bit compression we provide
``quantization_config=OVWeightQuantizationConfig(bits=4, ...)`` argument
containing number of bits and other compression parameters. We will
consider how to do it on RedPajama, LLAMA and Zephyr examples.

   **Note**: There may be no speedup for INT4/INT8 compressed models on
   dGPU.

Weights Compression using NNCF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



You also can perform weights compression for OpenVINO models using NNCF
directly. ``nncf.compress_weights`` function accepts OpenVINO model
instance and compresses its weights for Linear and Embedding layers. We
will consider this variant based on MPT model.

   **Note**: This tutorial involves conversion model for FP16 and
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

    from optimum.intel import OVWeightQuantizationConfig
    
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
            remote_code = model_configuration.get("remote_code", False)
            model_kwargs = {}
            if remote_code:
                model_kwargs = {
                    "trust_remote_code": True,
                    "config": AutoConfig.from_pretrained(pt_model_id, trust_remote_code=True)
                }
            ov_model = OVModelForCausalLM.from_pretrained(
                pt_model_id, export=True, compile=False, load_in_8bit=False, **model_kwargs
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
            remote_code = model_configuration.get("remote_code", False)
            model_kwargs = {}
            if remote_code:
                model_kwargs = {
                    "trust_remote_code": True,
                    "config": AutoConfig.from_pretrained(pt_model_id, trust_remote_code=True)
                }
            ov_model = OVModelForCausalLM.from_pretrained(
                pt_model_id, export=True, compile=False, load_in_8bit=True, **model_kwargs
            )
            ov_model.save_pretrained(int8_model_dir)
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
                "sym": True,
                "group_size": 64,
                "ratio": 0.6,
            },
            "mistral-7b": {
                "sym": True,
                "group_size": 64,
                "ratio": 0.6,
            },
            "minicpm-2b-dpo": {
                "sym": True,
                "group_size": 64,
                "ratio": 0.6,
            },
            "gemma-2b-it": {
                "sym": True,
                "group_size": 64,
                "ratio": 0.6,
            },
            "notus-7b-v1": {
                "sym": True,
                "group_size": 64,
                "ratio": 0.6,
            },
            "neural-chat-7b-v3-1": {
                "sym": True,
                "group_size": 64,
                "ratio": 0.6,
            },
            "llama-2-chat-7b": {
                "sym": True,
                "group_size": 128,
                "ratio": 0.8,
            },
            "gemma-7b-it": {
                "sym": True,
                "group_size": 128,
                "ratio": 0.8,
            },
            "chatglm2-6b": {
                "sym": True,
                "group_size": 128,
                "ratio": 0.72,
            },
            "qwen-7b-chat": {
                "sym": True,
                "group_size": 128, 
                "ratio": 0.6
            },
            'red-pajama-3b-chat': {
                "sym": False,
                "group_size": 128,
                "ratio": 0.5,
            },
            "default": {
                "sym": False,
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
            remote_code = model_configuration.get("remote_code", False)
            model_kwargs = {}
            if remote_code:
                model_kwargs = {
                    "trust_remote_code" : True,
                    "config": AutoConfig.from_pretrained(pt_model_id, trust_remote_code=True)
                }
            ov_model = OVModelForCausalLM.from_pretrained(
                pt_model_id, export=True, compile=False,
                quantization_config=OVWeightQuantizationConfig(bits=4, **model_compression_params),
                **model_kwargs
            )
            ov_model.save_pretrained(int4_model_dir)
            del ov_model
        else:
            convert_to_fp16()
            ov_model = ov.Core().read_model(fp16_model_dir / "openvino_model.xml")
            shutil.copy(fp16_model_dir / "config.json", int4_model_dir / "config.json")
            configuration_file = fp16_model_dir / f"configuration_{model_type}.py"
            if configuration_file.exists():
                shutil.copy(
                    configuration_file, int4_model_dir / f"configuration_{model_type}.py"
                )
            mode = nncf.CompressWeightsMode.INT4_SYM if model_compression_params["sym"] else \
                nncf.CompressWeightsMode.INT4_ASYM
            del model_compression_params["sym"]
            compressed_model = nncf.compress_weights(ov_model, mode=mode, **model_compression_params)
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



.. parsed-literal::

    modeling_qwen.py:   0%|          | 0.00/55.6k [00:00<?, ?B/s]


.. parsed-literal::

    A new version of the following files was downloaded from https://huggingface.co/Qwen/Qwen-7B-Chat:
    - cpp_kernels.py
    . Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
    A new version of the following files was downloaded from https://huggingface.co/Qwen/Qwen-7B-Chat:
    - qwen_generation_utils.py
    . Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
    A new version of the following files was downloaded from https://huggingface.co/Qwen/Qwen-7B-Chat:
    - modeling_qwen.py
    - cpp_kernels.py
    - qwen_generation_utils.py
    . Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.



.. parsed-literal::

    Downloading shards:   0%|          | 0/8 [00:00<?, ?it/s]



.. parsed-literal::

    Loading checkpoint shards:   0%|          | 0/8 [00:00<?, ?it/s]



.. parsed-literal::

    generation_config.json:   0%|          | 0.00/273 [00:00<?, ?B/s]


.. parsed-literal::

    /home/ethan/intel/openvino_notebooks/openvino_env/lib/python3.11/site-packages/transformers/modeling_utils.py:4193: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(
    /home/ethan/.cache/huggingface/modules/transformers_modules/Qwen/Qwen-7B-Chat/8d24619bab456ea5abe2823c1d05fc5edec19174/modeling_qwen.py:811: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if batch_size <= 0:
    /home/ethan/.cache/huggingface/modules/transformers_modules/Qwen/Qwen-7B-Chat/8d24619bab456ea5abe2823c1d05fc5edec19174/modeling_qwen.py:835: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      elif kv_seq_len != hidden_states.size()[1]:
    /home/ethan/.cache/huggingface/modules/transformers_modules/Qwen/Qwen-7B-Chat/8d24619bab456ea5abe2823c1d05fc5edec19174/modeling_qwen.py:1288: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if seqlen > self._seq_len_cached or ntk_alpha != self._ntk_alpha_cached:
    /home/ethan/.cache/huggingface/modules/transformers_modules/Qwen/Qwen-7B-Chat/8d24619bab456ea5abe2823c1d05fc5edec19174/modeling_qwen.py:482: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if key_size > self.seq_length and self.use_logn_attn and not self.training:
    /home/ethan/.cache/huggingface/modules/transformers_modules/Qwen/Qwen-7B-Chat/8d24619bab456ea5abe2823c1d05fc5edec19174/modeling_qwen.py:502: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if query.size(1) == key_size:
    /home/ethan/intel/openvino_notebooks/openvino_env/lib/python3.11/site-packages/torch/jit/_trace.py:160: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten/src/ATen/core/TensorBody.h:489.)
      if a.grad is not None:


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

    Size of FP16 model is 14729.26 MB


Select device for inference and model variant
---------------------------------------------



   **Note**: There may be no speedup for INT4/INT8 compressed models on
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

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



The cell below create ``OVMPTModel`` and ``OVCHATGLM2Model`` wrapper
based on ``OVModelForCausalLM`` model.

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

    Dropdown(description='Model to run:', options=('FP16',), value='FP16')



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

    Loading model from qwen-7b-chat/FP16


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

    2 + 2 = (1


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
        if (model_language.value == "Chinese")
        else japanese_examples
        if (model_language.value == "Japanese")
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
    
    
    def convert_history_to_token(history: List[Tuple[str, str]]):
        """
        function for conversion history stored as list pairs of user and assistant messages to tokens according to model expected conversation template
        Params:
          history: dialogue history
        Returns:
          history in token format
        """
        if pt_model_name == "baichuan2":
            system_tokens = tok.encode(start_message)
            history_tokens = []
            for (old_query, response) in history[:-1]:
                round_tokens = []
                round_tokens.append(195)
                round_tokens.extend(tok.encode(old_query))
                round_tokens.append(196)
                round_tokens.extend(tok.encode(response))
                history_tokens = round_tokens + history_tokens
            input_tokens = system_tokens + history_tokens
            input_tokens.append(195)
            input_tokens.extend(tok.encode(history[-1][0]))
            input_tokens.append(196)
            input_token = torch.LongTensor([input_tokens])
        elif history_template is None:
            messages = [{"role": "system", "content": start_message}]
            for idx, (user_msg, model_msg) in enumerate(history):
                if idx == len(history) - 1 and not model_msg:
                    messages.append({"role": "user", "content": user_msg})
                    break
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                if model_msg:
                    messages.append({"role": "assistant", "content": model_msg})
                    
            input_token = tok.apply_chat_template(messages,
                                                  add_generation_prompt=True,
                                                  tokenize=True,
                                                  return_tensors="pt")
        else:
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
        input_ids = convert_history_to_token(history)
        if input_ids.shape[1] > 2000:
            history = [history[-1]]
            input_ids = convert_history_to_token(history)
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
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                stop_tokens)
    
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
    
    
    def request_cancel():
        ov_model.request.cancel()
    
    
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
        gr.Markdown(
            f"""<h1><center>OpenVINO {model_id.value} Chatbot</center></h1>""")
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
            fn=request_cancel,
            inputs=None,
            outputs=None,
            cancels=[submit_event, submit_click_event],
            queue=False,
        )
        clear.click(lambda: None, None, chatbot, queue=False)
    
    # if you are launching remotely, specify server_name and server_port
    #  demo.launch(server_name='your server name', server_port='server port in int')
    # if you have any issue to launch on your platform, you can pass share=True to launch method:
    # demo.launch(share=True)
    # it creates a publicly shareable link for the interface. Read more in the docs: https://gradio.app/docs/
    demo.launch()

.. code:: ipython3

    # please uncomment and run this cell for stopping gradio interface
    # demo.close()


.. parsed-literal::

    Closing server running on port: 5467

