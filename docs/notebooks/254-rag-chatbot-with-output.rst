Create an LLM-powered RAG system using OpenVINO
===============================================

**Retrieval-augmented generation (RAG)** is a technique for augmenting
LLM knowledge with additional, often private or real-time, data. LLMs
can reason about wide-ranging topics, but their knowledge is limited to
the public data up to a specific point in time that they were trained
on. If you want to build AI applications that can reason about private
data or data introduced after a model’s cutoff date, you need to augment
the knowledge of the model with the specific information it needs. The
process of bringing the appropriate information and inserting it into
the model prompt is known as Retrieval Augmented Generation (RAG).

`LangChain <https://python.langchain.com/docs/get_started/introduction>`__
is a framework for developing applications powered by language models.
It has a number of components specifically designed to help build RAG
applications. In this tutorial, we’ll build a simple question-answering
application over a Markdown or CSV data source.

The tutorial consists of the following steps:

-  Install prerequisites
-  Download and convert the model from a public source using the
   `OpenVINO integration with Hugging Face
   Optimum <https://huggingface.co/blog/openvino>`__.
-  Compress model weights to 4-bit or 8-bit data types using
   `NNCF <https://github.com/openvinotoolkit/nncf>`__
-  Create a RAG chain pipeline
-  Run chat pipeline

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prerequisites <#prerequisites>`__
-  `Select model for inference <#select-model-for-inference>`__
-  `login to huggingfacehub to get access to pretrained
   model <#login-to-huggingfacehub-to-get-access-to-pretrained-model>`__
-  `Convert model <#convert-model>`__

   -  `Convert LLM model <#convert-llm-model>`__

-  `Compress model weights <#compress-model-weights>`__

   -  `Weights Compression using Optimum
      Intel <#weights-compression-using-optimum-intel>`__
   -  `Weights Compression using
      NNCF <#weights-compression-using-nncf>`__
   -  `Convert embedding model <#convert-embedding-model>`__

-  `Select device for inference and model
   variant <#select-device-for-inference-and-model-variant>`__

   -  `Select device for embedding model
      inference <#select-device-for-embedding-model-inference>`__
   -  `Select device for LLM model
      inference <#select-device-for-llm-model-inference>`__

-  `Load model <#load-model>`__

   -  `Load embedding model <#load-embedding-model>`__
   -  `Load LLM model <#load-llm-model>`__

-  `Run QA over Document <#run-qa-over-document>`__

Prerequisites
-------------



Install required dependencies

.. code:: ipython3

    %pip uninstall -q -y openvino-dev openvino openvino-nightly
    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu\
    "git+https://github.com/huggingface/optimum-intel.git"\
    "nncf>=2.8.0"\
    "datasets"\
    "accelerate"\
    "openvino-nightly"\
    "gradio"\
    "onnx" "chromadb" "sentence_transformers" "langchain" "langchainhub" "transformers>=4.34.0" "unstructured" "scikit-learn" "python-docx" "pdfminer.six"


.. parsed-literal::

    WARNING: Skipping openvino-dev as it is not installed.
    WARNING: Skipping openvino as it is not installed.
    Note: you may need to restart the kernel to use updated packages.

    [notice] A new release of pip is available: 23.3.1 -> 23.3.2
    [notice] To update, run: pip install --upgrade pip
    Note: you may need to restart the kernel to use updated packages.


Select model for inference
--------------------------



The tutorial supports different models, you can select one from the
provided options to compare the quality of open source LLM solutions.


   **NOTE**: conversion of some models can require additional actions
   from user side and at least 64GB RAM for conversion.

The available embedding model options are:

-  **all-mpnet-base-v2(All)** - This is a
   `sentence-transformers <https://huggingface.co/sentence-transformers>`__
   model: It maps sentences & paragraphs to a 768 dimensional dense
   vector space and can be used for tasks like clustering or semantic
   search. More details about model can be found in `model
   card <https://huggingface.co/sentence-transformers/all-mpnet-base-v2>`__
-  **text2vec-large-chinese(Chinese)** - This is a
   `CoSENT <https://github.com/bojone/CoSENT>`__ model. It can be used
   for tasks like sentence embeddings, text matching or semantic search.
   More details about model can be found in `model
   card <https://huggingface.co/GanymedeNil/text2vec-base-chinese>`__

The available LLM model options are:

-  **tiny-llama-1b-chat** - This is the chat model finetuned on top of
   `TinyLlama/TinyLlama-1.1B-intermediate-step-955k-2T <https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T>`__.
   The TinyLlama project aims to pretrain a 1.1B Llama model on 3
   trillion tokens with the adoption of the same architecture and
   tokenizer as Llama 2. This means TinyLlama can be plugged and played
   in many open-source projects built upon Llama. Besides, TinyLlama is
   compact with only 1.1B parameters. This compactness allows it to
   cater to a multitude of applications demanding a restricted
   computation and memory footprint. More details about model can be
   found in `model
   card <https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.6>`__

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

.. code:: ipython3

    from pathlib import Path
    from optimum.intel import OVQuantizer
    from optimum.intel.openvino import OVModelForCausalLM
    import openvino as ov
    import torch
    import nncf
    import logging
    import shutil
    import gc
    import ipywidgets as widgets
    from transformers import (
        AutoModelForCausalLM,
        AutoModel,
        AutoTokenizer,
        AutoConfig,
        TextIteratorStreamer,
        pipeline,
        StoppingCriteria,
        StoppingCriteriaList,
    )


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    2023-12-25 07:58:21.310297: I tensorflow/core/util/port.cc:111] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-12-25 07:58:21.312367: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2023-12-25 07:58:21.337757: E tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:9342] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2023-12-25 07:58:21.337778: E tensorflow/compiler/xla/stream_executor/cuda/cuda_fft.cc:609] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2023-12-25 07:58:21.337798: E tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:1518] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2023-12-25 07:58:21.343045: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2023-12-25 07:58:21.343941: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 AVX_VNNI AMX_TILE AMX_INT8 AMX_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-12-25 07:58:21.912373: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


Convert model
-------------



Convert LLM model
~~~~~~~~~~~~~~~~~



.. code:: ipython3

    from config import SUPPORTED_EMBEDDING_MODELS, SUPPORTED_LLM_MODELS

    llm_model_id = list(SUPPORTED_LLM_MODELS)

    llm_model_id = widgets.Dropdown(
        options=llm_model_id,
        value=llm_model_id[0],
        description="LLM Model:",
        disabled=False,
    )

    llm_model_id




.. parsed-literal::

    Dropdown(description='LLM Model:', options=('tiny-llama-1b-chat', 'red-pajama-3b-chat', 'llama-2-chat-7b', 'mp…



.. code:: ipython3

    llm_model_configuration = SUPPORTED_LLM_MODELS[llm_model_id.value]
    print(f"Selected LLM model {llm_model_id.value}")


.. parsed-literal::

    Selected LLM model chatglm3-6b


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


.. code:: ipython3

    from converter import converters

    nncf.set_log_level(logging.ERROR)

    pt_model_id = llm_model_configuration["model_id"]
    pt_model_name = llm_model_id.value.split("-")[0]
    model_type = AutoConfig.from_pretrained(pt_model_id, trust_remote_code=True).model_type
    fp16_model_dir = Path(llm_model_id.value) / "FP16"
    int8_model_dir = Path(llm_model_id.value) / "INT8_compressed_weights"
    int4_model_dir = Path(llm_model_id.value) / "INT4_compressed_weights"


    def convert_to_fp16():
        if (fp16_model_dir / "openvino_model.xml").exists():
            return
        if not llm_model_configuration["remote"]:
            ov_model = OVModelForCausalLM.from_pretrained(
                pt_model_id, export=True, compile=False, load_in_8bit=False
            )
            ov_model.half()
            ov_model.save_pretrained(fp16_model_dir)
            del ov_model
        else:
            model_kwargs = {}
            if "revision" in llm_model_configuration:
                model_kwargs["revision"] = llm_model_configuration["revision"]
            model = AutoModelForCausalLM.from_pretrained(
                llm_model_configuration["model_id"],
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
        if not llm_model_configuration["remote"]:
            if fp16_model_dir.exists():
                ov_model = OVModelForCausalLM.from_pretrained(fp16_model_dir, compile=False, load_in_8bit=False)
            else:
                ov_model = OVModelForCausalLM.from_pretrained(
                    pt_model_id, export=True, compile=False
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
            },"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1",
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
                "ratio": 0.72
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
            llm_model_id.value, compression_configs["default"]
        )
        if (int4_model_dir / "openvino_model.xml").exists():
            return
        int4_model_dir.mkdir(parents=True, exist_ok=True)
        if not llm_model_configuration["remote"]:
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

    Size of FP16 model is 11909.69 MB
    Size of model with INT4 compressed weights is 3890.41 MB
    Compression rate for INT4 model: 3.061


Convert embedding model
~~~~~~~~~~~~~~~~~~~~~~~



Since some embedding models can only support limited languages, we can
filter them out according the LLM you selected.

.. code:: ipython3

    embedding_model_id = list(SUPPORTED_EMBEDDING_MODELS)

    if "qwen" not in llm_model_id.value and "chatglm" not in llm_model_id.value:
        embedding_model_id = [x for x in embedding_model_id if "chinese" not in x]

    embedding_model_id = widgets.Dropdown(
        options=embedding_model_id,
        value=embedding_model_id[0],
        description="Embedding Model:",
        disabled=False,
    )

    embedding_model_id




.. parsed-literal::

    Dropdown(description='Embedding Model:', options=('all-mpnet-base-v2', 'text2vec-large-chinese'), value='all-m…



.. code:: ipython3

    embedding_model_configuration = SUPPORTED_EMBEDDING_MODELS[embedding_model_id.value]
    print(f"Selected {embedding_model_id.value} model")


.. parsed-literal::

    Selected all-mpnet-base-v2 model


.. code:: ipython3

    embedding_model_dir = Path(embedding_model_id.value)

    if not (embedding_model_dir / "openvino_model.xml").exists():
        model = AutoModel.from_pretrained(embedding_model_configuration["model_id"])
        converters[embedding_model_id.value](model, embedding_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(embedding_model_configuration["model_id"])
        tokenizer.save_pretrained(embedding_model_dir)
        del model

Select device for inference and model variant
---------------------------------------------



   **NOTE**: There may be no speedup for INT4/INT8 compressed models on
   dGPU.

Select device for embedding model inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    core = ov.Core()
    embedding_device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value="CPU",
        description="Device:",
        disabled=False,
    )

    embedding_device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'GPU', 'AUTO'), value='CPU')



.. code:: ipython3

    print(f"Embedding model will be loaded to {embedding_device.value} device for response generation")


.. parsed-literal::

    Embedding model will be loaded to CPU device for response generation


Select device for LLM model inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    llm_device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value="CPU",
        description="Device:",
        disabled=False,
    )

    llm_device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'GPU', 'AUTO'), value='CPU')



.. code:: ipython3

    print(f"LLM model will be loaded to {llm_device.value} device for response generation")


.. parsed-literal::

    LLM model will be loaded to CPU device for response generation


Load model
----------



Load embedding model
~~~~~~~~~~~~~~~~~~~~



Wrapper around a text embedding model for LangChain, used for converting
text to embeddings.

.. code:: ipython3

    from ov_embedding_model import OVEmbeddings

    embedding = OVEmbeddings.from_model_id(
        embedding_model_dir,
        do_norm=embedding_model_configuration["do_norm"],
        ov_config={
            "device_name": embedding_device.value,
            "config": {"PERFORMANCE_HINT": "THROUGHPUT"},
        },
        model_kwargs={
            "model_max_length": 512,
        },
    )

Load LLM model
~~~~~~~~~~~~~~



The cell below create ``OVMPTModel``, ``OVQWENModel`` and
``OVCHATGLM2Model`` wrapper based on ``OVModelForCausalLM`` model.

.. code:: ipython3

    from ov_llm_model import model_classes

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

    Dropdown(description='Model to run:', options=('INT4', 'FP16'), value='INT4')



.. code:: ipython3

    from langchain.llms import HuggingFacePipeline

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
    if llm_model_id.value == "red-pajama-3b-chat" and "GPU" in core.available_devices and llm_device.value in ["GPU", "AUTO"]:
        ov_config["INFERENCE_PRECISION_HINT"] = "f32"

    model_name = llm_model_configuration["model_id"]
    stop_tokens = llm_model_configuration.get("stop_tokens")
    class_key = llm_model_id.value.split("-")[0]
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

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

    model_class = (
        OVModelForCausalLM
        if not llm_model_configuration["remote"]
        else model_classes[class_key]
    )
    ov_model = model_class.from_pretrained(
        model_dir,
        device=llm_device.value,
        ov_config=ov_config,
        config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
        trust_remote_code=True,
    )


.. parsed-literal::

    Loading model from chatglm3-6b/INT4_compressed_weights


.. parsed-literal::

    The argument `trust_remote_code` is to be used along with export=True. It will be ignored.
    Compiling the model to CPU ...


Wrapper around a LLM/chat model for LangChain, used for generating the
response text. An OpenVINO compiled model can be run locally through the
``HuggingFacePipeline`` class.

.. code:: ipython3

    streamer = TextIteratorStreamer(
        tok, timeout=30.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        model=ov_model,
        tokenizer=tok,
        max_new_tokens=256,
        streamer=streamer,
        # temperature=1,
        # do_sample=True,
        # top_p=0.8,
        # top_k=20,
        # repetition_penalty=1.1,
    )
    if stop_tokens is not None:
        generate_kwargs["stopping_criteria"] = StoppingCriteriaList(stop_tokens)

    pipe = pipeline("text-generation", **generate_kwargs)
    llm = HuggingFacePipeline(pipeline=pipe)

Run QA over Document
--------------------



Now, when model created, we can setup Chatbot interface using
`Gradio <https://www.gradio.app/>`__.

A typical RAG application has two main components:

-  **Indexing**: a pipeline for ingesting data from a source and
   indexing it. This usually happen offline.

-  **Retrieval and generation**: the actual RAG chain, which takes the
   user query at run time and retrieves the relevant data from the
   index, then passes that to the model.

The most common full sequence from raw data to answer looks like:

**Indexing** 1. ``Load``: First we need to load our data. We’ll use
DocumentLoaders for this. 2. ``Split``: Text splitters break large
Documents into smaller chunks. This is useful both for indexing data and
for passing it in to a model, since large chunks are harder to search
over and won’t in a model’s finite context window. 3. ``Store``: We need
somewhere to store and index our splits, so that they can later be
searched over. This is often done using a VectorStore and Embeddings
model.

.. figure:: https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/dfed2ba3-0c3a-4e0e-a2a7-01638730486a
   :alt: Indexing pipeline

   Indexing pipeline

**Retrieval and generation** 1. ``Retrieve``: Given a user input,
relevant splits are retrieved from storage using a Retriever. 2.
``Generate``: A LLM produces an answer using a prompt that includes the
question and the retrieved data.

.. figure:: https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/f0545ddc-c0cd-4569-8c86-9879fdab105a
   :alt: Retrieval and generation pipeline

   Retrieval and generation pipeline

.. code:: ipython3

    from typing import List
    from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownTextSplitter
    from langchain.document_loaders import (
        CSVLoader,
        EverNoteLoader,
        PDFMinerLoader,
        TextLoader,
        UnstructuredEPubLoader,
        UnstructuredHTMLLoader,
        UnstructuredMarkdownLoader,
        UnstructuredODTLoader,
        UnstructuredPowerPointLoader,
        UnstructuredWordDocumentLoader, )


    class ChineseTextSplitter(CharacterTextSplitter):
        def __init__(self, pdf: bool = False, **kwargs):
            super().__init__(**kwargs)
            self.pdf = pdf

        def split_text(self, text: str) -> List[str]:
            if self.pdf:
                text = re.sub(r"\n{3,}", "\n", text)
                text = text.replace("\n\n", "")
            sent_sep_pattern = re.compile(
                '([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')
            sent_list = []
            for ele in sent_sep_pattern.split(text):
                if sent_sep_pattern.match(ele) and sent_list:
                    sent_list[-1] += ele
                elif ele:
                    sent_list.append(ele)
            return sent_list


    TEXT_SPLITERS = {
        "Character": CharacterTextSplitter,
        "RecursiveCharacter": RecursiveCharacterTextSplitter,
        "Markdown": MarkdownTextSplitter,
        "Chinese": ChineseTextSplitter,
    }


    LOADERS = {
        ".csv": (CSVLoader, {}),
        ".doc": (UnstructuredWordDocumentLoader, {}),
        ".docx": (UnstructuredWordDocumentLoader, {}),
        ".enex": (EverNoteLoader, {}),
        ".epub": (UnstructuredEPubLoader, {}),
        ".html": (UnstructuredHTMLLoader, {}),
        ".md": (UnstructuredMarkdownLoader, {}),
        ".odt": (UnstructuredODTLoader, {}),
        ".pdf": (PDFMinerLoader, {}),
        ".ppt": (UnstructuredPowerPointLoader, {}),
        ".pptx": (UnstructuredPowerPointLoader, {}),
        ".txt": (TextLoader, {"encoding": "utf8"}),
    }

.. code:: ipython3

    from langchain.prompts import PromptTemplate
    from langchain.vectorstores import Chroma
    from langchain.chains import RetrievalQA
    from langchain.docstore.document import Document
    from threading import Event, Thread
    import gradio as gr
    import re
    from uuid import uuid4


    def load_single_document(file_path: str) -> List[Document]:
        """
        helper for loading a single document

        Params:
          file_path: document path
        Returns:
          documents loaded

        """
        ext = "." + file_path.rsplit(".", 1)[-1]
        if ext in LOADERS:
            loader_class, loader_args = LOADERS[ext]
            loader = loader_class(file_path, **loader_args)
            return loader.load()

        raise ValueError(f"File does not exist '{ext}'")


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


    text_processor = llm_model_configuration.get(
        "partial_text_processor", default_partial_text_processor
    )


    def build_chain(docs, spliter_name, chunk_size, chunk_overlap, vector_search_top_k):
        """
        Initialize a QA chain

        Params:
          doc: orignal documents provided by user
          chunk_size:  size of a single sentence chunk
          chunk_overlap: overlap size between 2 chunks
          vector_search_top_k: Vector search top k

        """
        documents = []
        for doc in docs:
            documents.extend(load_single_document(doc.name))

        text_splitter = TEXT_SPLITERS[spliter_name](
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        texts = text_splitter.split_documents(documents)

        db = Chroma.from_documents(texts, embedding)
        retriever = db.as_retriever(search_kwargs={"k": vector_search_top_k})

        global rag_chain
        prompt = PromptTemplate.from_template(llm_model_configuration["prompt_template"])
        chain_type_kwargs = {"prompt": prompt}
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
        )

        return "Retriever is Ready"


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


    def bot(history, conversation_id):
        """
        callback function for running chatbot on submit button click

        Params:
          history: conversation history.
          conversation_id: unique conversation identifier.

        """
        stream_complete = Event()

        def infer(question):
            rag_chain.run(question)
            stream_complete.set()

        t1 = Thread(target=infer, args=(history[-1][0],))
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
        gr.Markdown("""<h1><center>QA over Document</center></h1>""")
        gr.Markdown(f"""<center>Powered by OpenVINO and {llm_model_id.value} </center>""")
        with gr.Row():
            with gr.Column(scale=1):
                docs = gr.File(
                    label="Load text files",
                    file_count="multiple",
                    file_types=[
                        ".csv",
                        ".doc",
                        ".docx",
                        ".enex",
                        ".epub",
                        ".html",
                        ".md",
                        ".odt",
                        ".pdf",
                        ".ppt",
                        ".pptx",
                        ".txt",
                    ],
                )
                load_docs = gr.Button("Build Retriever")
                retriever_argument = gr.Accordion("Retriever Configuration", open=False)
                with retriever_argument:
                    spliter = gr.Dropdown(
                        ["Character", "RecursiveCharacter", "Markdown", "Chinese"],
                        value="RecursiveCharacter",
                        label="Text Spliter",
                        info="Method used to splite the documents",
                        multiselect=False,
                    )

                    chunk_size = gr.Slider(
                        label="Chunk size",
                        value=1000,
                        minimum=100,
                        maximum=2000,
                        step=50,
                        interactive=True,
                        info="Size of sentence chunk",
                    )

                    chunk_overlap = gr.Slider(
                        label="Chunk overlap",
                        value=200,
                        minimum=0,
                        maximum=400,
                        step=10,
                        interactive=True,
                        info=("Overlap between 2 chunks"),
                    )

                    vector_search_top_k = gr.Slider(
                        1,
                        10,
                        value=6,
                        step=1,
                        label="Vector search top k",
                        interactive=True,
                    )
                langchain_status = gr.Textbox(
                    label="Status", value="Retriever is Not ready", interactive=False
                )
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=600)
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
                            clear = gr.Button("Clear")
        load_docs.click(
            build_chain,
            inputs=[docs, spliter, chunk_size, chunk_overlap, vector_search_top_k],
            outputs=[langchain_status],
            queue=False,
        )
        submit_event = msg.submit(
            user, [msg, chatbot], [msg, chatbot], queue=False, trigger_mode="once"
        ).then(bot, [chatbot, conversation_id], chatbot, queue=True)
        submit_click_event = submit.click(
            user, [msg, chatbot], [msg, chatbot], queue=False, trigger_mode="once"
        ).then(bot, [chatbot, conversation_id], chatbot, queue=True)
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue(max_size=2)
    # if you are launching remotely, specify server_name and server_port
    #  demo.launch(server_name='your server name', server_port='server port in int')
    # if you have any issue to launch on your platform, you can pass share=True to launch method:
    # demo.launch(share=True)
    # it creates a publicly shareable link for the interface. Read more in the docs: https://gradio.app/docs/
    demo.launch()


.. parsed-literal::

    Running on local URL:  http://10.3.233.70:4888

    To create a public link, set `share=True` in `launch()`.



.. .. raw:: html

..    <div><iframe src="http://10.3.233.70:4888/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>





.. parsed-literal::

    /home/ethan/intel/openvino_notebooks/openvino_env/lib/python3.10/site-packages/optimum/intel/openvino/modeling_decoder.py:388: FutureWarning: `shared_memory` is deprecated and will be removed in 2024.0. Value of `shared_memory` is going to override `share_inputs` value. Please use only `share_inputs` explicitly.
      self.request.start_async(inputs, shared_memory=True)
    /home/ethan/intel/openvino_notebooks/openvino_env/lib/python3.10/site-packages/optimum/intel/openvino/modeling_decoder.py:388: FutureWarning: `shared_memory` is deprecated and will be removed in 2024.0. Value of `shared_memory` is going to override `share_inputs` value. Please use only `share_inputs` explicitly.
      self.request.start_async(inputs, shared_memory=True)
    /home/ethan/intel/openvino_notebooks/openvino_env/lib/python3.10/site-packages/optimum/intel/openvino/modeling_decoder.py:388: FutureWarning: `shared_memory` is deprecated and will be removed in 2024.0. Value of `shared_memory` is going to override `share_inputs` value. Please use only `share_inputs` explicitly.
      self.request.start_async(inputs, shared_memory=True)


.. code:: ipython3

    # please run this cell for stopping gradio interface
    demo.close()
    del rag_chain


.. parsed-literal::

    Closing server running on port: 4888

