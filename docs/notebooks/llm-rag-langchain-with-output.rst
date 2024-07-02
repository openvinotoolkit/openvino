Create a RAG system using OpenVINO and LangChain
================================================

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
application over a text data source.

The tutorial consists of the following steps:

-  Install prerequisites
-  Download and convert the model from a public source using the
   `OpenVINO integration with Hugging Face
   Optimum <https://huggingface.co/blog/openvino>`__.
-  Compress model weights to 4-bit or 8-bit data types using
   `NNCF <https://github.com/openvinotoolkit/nncf>`__
-  Create a RAG chain pipeline
-  Run Q&A pipeline

In this example, the customized RAG pipeline consists of following
components in order, where embedding, rerank and LLM will be deployed
with OpenVINO to optimize their inference performance.

.. figure:: https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/0076f6c7-75e4-4c2e-9015-87b355e5ca28
   :alt: RAG

   RAG

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prerequisites <#Prerequisites>`__
-  `Select model for inference <#Select-model-for-inference>`__
-  `login to huggingfacehub to get access to pretrained
   model <#login-to-huggingfacehub-to-get-access-to-pretrained-model>`__
-  `Convert model and compress model
   weights <#convert-model-and-compress-model-weights>`__

   -  `LLM conversion and Weights Compression using
      Optimum-CLI <#LLM-conversion-and-Weights-Compression-using-Optimum-CLI>`__

      -  `Weight compression with AWQ <#Weight-compression-with-AWQ>`__

   -  `Convert embedding model using
      Optimum-CLI <#Convert-embedding-model-using-Optimum-CLI>`__
   -  `Convert rerank model using
      Optimum-CLI <#Convert-rerank-model-using-Optimum-CLI>`__

-  `Select device for inference and model
   variant <#Select-device-for-inference-and-model-variant>`__

   -  `Select device for embedding model
      inference <#Select-device-for-embedding-model-inference>`__
   -  `Select device for rerank model
      inference <#Select-device-for-rerank-model-inference>`__
   -  `Select device for LLM model
      inference <#Select-device-for-LLM-model-inference>`__

-  `Load model <#Load-model>`__

   -  `Load embedding model <#Load-embedding-model>`__
   -  `Load rerank model <#Load-rerank-model>`__
   -  `Load LLM model <#Load-LLM-model>`__

-  `Run QA over Document <#Run-QA-over-Document>`__

Prerequisites
-------------

`back to top ⬆️ <#Table-of-contents:>`__

Install required dependencies

.. code:: ipython3

    import os
    
    os.environ["GIT_CLONE_PROTECTION_ACTIVE"] = "false"
    
    %pip install -Uq pip
    %pip uninstall -q -y optimum optimum-intel
    %pip install --pre -Uq openvino openvino-tokenizers[transformers] --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu\
    "git+https://github.com/huggingface/optimum-intel.git"\
    "git+https://github.com/openvinotoolkit/nncf.git"\
    "datasets"\
    "accelerate"\
    "gradio"\
    "onnx" "einops" "transformers_stream_generator" "tiktoken" "transformers>=4.40" "bitsandbytes" "faiss-cpu" "sentence_transformers" "langchain>=0.2.0" "langchain-community>=0.2.0" "langchainhub" "unstructured" "scikit-learn" "python-docx" "pypdf" 


.. parsed-literal::

    WARNING: Skipping openvino-dev as it is not installed.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    import os
    from pathlib import Path
    import requests
    import shutil
    import io
    
    # fetch model configuration
    
    config_shared_path = Path("../../utils/llm_config.py")
    config_dst_path = Path("llm_config.py")
    text_example_en_path = Path("text_example_en.pdf")
    text_example_cn_path = Path("text_example_cn.pdf")
    text_example_en = "https://github.com/openvinotoolkit/openvino_notebooks/files/15039728/Platform.Brief_Intel.vPro.with.Intel.Core.Ultra_Final.pdf"
    text_example_cn = "https://github.com/openvinotoolkit/openvino_notebooks/files/15039713/Platform.Brief_Intel.vPro.with.Intel.Core.Ultra_Final_CH.pdf"
    
    if not config_dst_path.exists():
        if config_shared_path.exists():
            try:
                os.symlink(config_shared_path, config_dst_path)
            except Exception:
                shutil.copy(config_shared_path, config_dst_path)
        else:
            r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/llm_config.py")
            with open("llm_config.py", "w", encoding="utf-8") as f:
                f.write(r.text)
    elif not os.path.islink(config_dst_path):
        print("LLM config will be updated")
        if config_shared_path.exists():
            shutil.copy(config_shared_path, config_dst_path)
        else:
            r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/llm_config.py")
            with open("llm_config.py", "w", encoding="utf-8") as f:
                f.write(r.text)
    
    
    if not text_example_en_path.exists():
        r = requests.get(url=text_example_en)
        content = io.BytesIO(r.content)
        with open("text_example_en.pdf", "wb") as f:
            f.write(content.read())
    
    if not text_example_cn_path.exists():
        r = requests.get(url=text_example_cn)
        content = io.BytesIO(r.content)
        with open("text_example_cn.pdf", "wb") as f:
            f.write(content.read())


.. parsed-literal::

    LLM config will be updated


Select model for inference
--------------------------

`back to top ⬆️ <#Table-of-contents:>`__

The tutorial supports different models, you can select one from the
provided options to compare the quality of open source LLM solutions.

   **Note**: conversion of some models can require additional actions
   from user side and at least 64GB RAM for conversion.

The available embedding model options are:

-  `bge-small-en-v1.5 <https://huggingface.co/BAAI/bge-small-en-v1.5>`__
-  `bge-small-zh-v1.5 <https://huggingface.co/BAAI/bge-small-zh-v1.5>`__
-  `bge-large-en-v1.5 <https://huggingface.co/BAAI/bge-large-en-v1.5>`__
-  `bge-large-zh-v1.5 <https://huggingface.co/BAAI/bge-large-zh-v1.5>`__

BGE embedding is a general Embedding Model. The model is pre-trained
using RetroMAE and trained on large-scale pair data using contrastive
learning.

The available rerank model options are:

-  `bge-reranker-large <https://huggingface.co/BAAI/bge-reranker-large>`__
-  `bge-reranker-base <https://huggingface.co/BAAI/bge-reranker-base>`__

Reranker model with cross-encoder will perform full-attention over the
input pair, which is more accurate than embedding model (i.e.,
bi-encoder) but more time-consuming than embedding model. Therefore, it
can be used to re-rank the top-k documents returned by embedding model.

You can also find available LLM model options in
`llm-chatbot <../llm-chatbot/README.md>`__ notebook.

.. code:: ipython3

    from pathlib import Path
    import openvino as ov
    import torch
    import ipywidgets as widgets
    from transformers import (
        TextIteratorStreamer,
        StoppingCriteria,
        StoppingCriteriaList,
    )

Convert model and compress model weights
----------------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

The Weights Compression algorithm is aimed at compressing the weights of
the models and can be used to optimize the model footprint and
performance of large models where the size of weights is relatively
larger than the size of activations, for example, Large Language Models
(LLM). Compared to INT8 compression, INT4 compression improves
performance even more, but introduces a minor drop in prediction
quality.

.. code:: ipython3

    from llm_config import (
        SUPPORTED_EMBEDDING_MODELS,
        SUPPORTED_RERANK_MODELS,
        SUPPORTED_LLM_MODELS,
    )
    
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

    llm_model_ids = [model_id for model_id, model_config in SUPPORTED_LLM_MODELS[model_language.value].items() if model_config.get("rag_prompt_template")]
    
    llm_model_id = widgets.Dropdown(
        options=llm_model_ids,
        value=llm_model_ids[-1],
        description="Model:",
        disabled=False,
    )
    
    llm_model_id




.. parsed-literal::

    Dropdown(description='Model:', index=4, options=('qwen1.5-7b-chat', 'qwen-7b-chat', 'chatglm3-6b', 'baichuan2-…



.. code:: ipython3

    llm_model_configuration = SUPPORTED_LLM_MODELS[model_language.value][llm_model_id.value]
    print(f"Selected LLM model {llm_model_id.value}")


.. parsed-literal::

    Selected LLM model qwen1.5-1.8b-chat


🤗 `Optimum Intel <https://huggingface.co/docs/optimum/intel/index>`__ is
the interface between the 🤗
`Transformers <https://huggingface.co/docs/transformers/index>`__ and
`Diffusers <https://huggingface.co/docs/diffusers/index>`__ libraries
and OpenVINO to accelerate end-to-end pipelines on Intel architectures.
It provides ease-to-use cli interface for exporting models to `OpenVINO
Intermediate Representation
(IR) <https://docs.openvino.ai/2024/documentation/openvino-ir-format.html>`__
format.

The command bellow demonstrates basic command for model export with
``optimum-cli``

::

   optimum-cli export openvino --model <model_id_or_path> --task <task> <out_dir>

where ``--model`` argument is model id from HuggingFace Hub or local
directory with model (saved using ``.save_pretrained`` method),
``--task`` is one of `supported
task <https://huggingface.co/docs/optimum/exporters/task_manager>`__
that exported model should solve. For LLMs it will be
``text-generation-with-past``. If model initialization requires to use
remote code, ``--trust-remote-code`` flag additionally should be passed.

LLM conversion and Weights Compression using Optimum-CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

You can also apply fp16, 8-bit or 4-bit weight compression on the
Linear, Convolutional and Embedding layers when exporting your model
with the CLI by setting ``--weight-format`` to respectively fp16, int8
or int4. This type of optimization allows to reduce the memory footprint
and inference latency. By default the quantization scheme for int8/int4
will be
`asymmetric <https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Quantization.md#asymmetric-quantization>`__,
to make it
`symmetric <https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Quantization.md#symmetric-quantization>`__
you can add ``--sym``.

For INT4 quantization you can also specify the following arguments :

-  The ``--group-size`` parameter will define the group size to use for
   quantization, -1 it will results in per-column quantization.
-  The ``--ratio`` parameter controls the ratio between 4-bit and 8-bit
   quantization. If set to 0.9, it means that 90% of the layers will be
   quantized to int4 while 10% will be quantized to int8.

Smaller group_size and ratio values usually improve accuracy at the
sacrifice of the model size and inference latency.

   **Note**: There may be no speedup for INT4/INT8 compressed models on
   dGPU.

.. code:: ipython3

    from IPython.display import Markdown, display
    
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


Weight compression with AWQ
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`back to top ⬆️ <#Table-of-contents:>`__

`Activation-aware Weight
Quantization <https://arxiv.org/abs/2306.00978>`__ (AWQ) is an algorithm
that tunes model weights for more accurate INT4 compression. It slightly
improves generation quality of compressed LLMs, but requires significant
additional time for tuning weights on a calibration dataset. We use
``wikitext-2-raw-v1/train`` subset of the
`Wikitext <https://huggingface.co/datasets/Salesforce/wikitext>`__
dataset for calibration.

Below you can enable AWQ to be additionally applied during model export
with INT4 precision.

   **Note**: Applying AWQ requires significant memory and time.

..

   **Note**: It is possible that there will be no matching patterns in
   the model to apply AWQ, in such case it will be skipped.

.. code:: ipython3

    enable_awq = widgets.Checkbox(
        value=False,
        description="Enable AWQ",
        disabled=not prepare_int4_model.value,
    )
    display(enable_awq)

.. code:: ipython3

    pt_model_id = llm_model_configuration["model_id"]
    pt_model_name = llm_model_id.value.split("-")[0]
    fp16_model_dir = Path(llm_model_id.value) / "FP16"
    int8_model_dir = Path(llm_model_id.value) / "INT8_compressed_weights"
    int4_model_dir = Path(llm_model_id.value) / "INT4_compressed_weights"
    
    
    def convert_to_fp16():
        if (fp16_model_dir / "openvino_model.xml").exists():
            return
        remote_code = llm_model_configuration.get("remote_code", False)
        export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format fp16".format(pt_model_id)
        if remote_code:
            export_command_base += " --trust-remote-code"
        export_command = export_command_base + " " + str(fp16_model_dir)
        display(Markdown("**Export command:**"))
        display(Markdown(f"`{export_command}`"))
        ! $export_command
    
    
    def convert_to_int8():
        if (int8_model_dir / "openvino_model.xml").exists():
            return
        int8_model_dir.mkdir(parents=True, exist_ok=True)
        remote_code = llm_model_configuration.get("remote_code", False)
        export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format int8".format(pt_model_id)
        if remote_code:
            export_command_base += " --trust-remote-code"
        export_command = export_command_base + " " + str(int8_model_dir)
        display(Markdown("**Export command:**"))
        display(Markdown(f"`{export_command}`"))
        ! $export_command
    
    
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
            "llama-3-8b-instruct": {
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
            "qwen-7b-chat": {"sym": True, "group_size": 128, "ratio": 0.6},
            "red-pajama-3b-chat": {
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
    
        model_compression_params = compression_configs.get(llm_model_id.value, compression_configs["default"])
        if (int4_model_dir / "openvino_model.xml").exists():
            return
        remote_code = llm_model_configuration.get("remote_code", False)
        export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format int4".format(pt_model_id)
        int4_compression_args = " --group-size {} --ratio {}".format(model_compression_params["group_size"], model_compression_params["ratio"])
        if model_compression_params["sym"]:
            int4_compression_args += " --sym"
        if enable_awq.value:
            int4_compression_args += " --awq --dataset wikitext2 --num-samples 128"
        export_command_base += int4_compression_args
        if remote_code:
            export_command_base += " --trust-remote-code"
        export_command = export_command_base + " " + str(int4_model_dir)
        display(Markdown("**Export command:**"))
        display(Markdown(f"`{export_command}`"))
        ! $export_command
    
    
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
            print(f"Size of model with INT{precision} compressed weights is {compressed_weights.stat().st_size / 1024 / 1024:.2f} MB")
        if compressed_weights.exists() and fp16_weights.exists():
            print(f"Compression rate for INT{precision} model: {fp16_weights.stat().st_size / compressed_weights.stat().st_size:.3f}")


.. parsed-literal::

    Size of model with INT4 compressed weights is 1343.75 MB


Convert embedding model using Optimum-CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

Since some embedding models can only support limited languages, we can
filter them out according the LLM you selected.

.. code:: ipython3

    embedding_model_id = list(SUPPORTED_EMBEDDING_MODELS[model_language.value])
    
    embedding_model_id = widgets.Dropdown(
        options=embedding_model_id,
        value=embedding_model_id[0],
        description="Embedding Model:",
        disabled=False,
    )
    
    embedding_model_id




.. parsed-literal::

    Dropdown(description='Embedding Model:', options=('bge-small-zh-v1.5', 'bge-large-zh-v1.5'), value='bge-small-…



.. code:: ipython3

    embedding_model_configuration = SUPPORTED_EMBEDDING_MODELS[model_language.value][embedding_model_id.value]
    print(f"Selected {embedding_model_id.value} model")


.. parsed-literal::

    Selected bge-small-zh-v1.5 model


OpenVINO embedding model and tokenizer can be exported by
``feature-extraction`` task with ``optimum-cli``.

.. code:: ipython3

    export_command_base = "optimum-cli export openvino --model {} --task feature-extraction".format(embedding_model_configuration["model_id"])
    export_command = export_command_base + " " + str(embedding_model_id.value)
    
    if not Path(embedding_model_id.value).exists():
        ! $export_command

Convert rerank model using Optimum-CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    rerank_model_id = list(SUPPORTED_RERANK_MODELS)
    
    rerank_model_id = widgets.Dropdown(
        options=rerank_model_id,
        value=rerank_model_id[0],
        description="Rerank Model:",
        disabled=False,
    )
    
    rerank_model_id




.. parsed-literal::

    Dropdown(description='Rerank Model:', options=('bge-reranker-large', 'bge-reranker-base'), value='bge-reranker…



.. code:: ipython3

    rerank_model_configuration = SUPPORTED_RERANK_MODELS[rerank_model_id.value]
    print(f"Selected {rerank_model_id.value} model")


.. parsed-literal::

    Selected bge-reranker-large model


Since ``rerank`` model is sort of sentence classification task, its
OpenVINO IR and tokenizer can be exported by ``text-classification``
task with ``optimum-cli``.

.. code:: ipython3

    export_command_base = "optimum-cli export openvino --model {} --task text-classification".format(rerank_model_configuration["model_id"])
    export_command = export_command_base + " " + str(rerank_model_id.value)
    
    if not Path(rerank_model_id.value).exists():
        ! $export_command

Select device for inference and model variant
---------------------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

   **Note**: There may be no speedup for INT4/INT8 compressed models on
   dGPU.

Select device for embedding model inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    core = ov.Core()
    
    support_devices = core.available_devices
    
    embedding_device = widgets.Dropdown(
        options=support_devices + ["AUTO"],
        value="CPU",
        description="Device:",
        disabled=False,
    )
    
    embedding_device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'GPU', 'AUTO'), value='CPU')



.. code:: ipython3

    print(f"Embedding model will be loaded to {embedding_device.value} device for text embedding")


.. parsed-literal::

    Embedding model will be loaded to CPU device for text embedding


Optimize the BGE embedding model’s parameter precision when loading
model to NPU device.

.. code:: ipython3

    USING_NPU = embedding_device.value == "NPU"
    
    npu_embedding_dir = embedding_model_id.value + "-npu"
    npu_embedding_path = Path(npu_embedding_dir) / "openvino_model.xml"
    if USING_NPU and not Path(npu_embedding_dir).exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
        )
        with open("notebook_utils.py", "w") as f:
            f.write(r.text)
        import notebook_utils as utils
    
        shutil.copytree(embedding_model_id.value, npu_embedding_dir)
        utils.optimize_bge_embedding(Path(embedding_model_id.value) / "openvino_model.xml", npu_embedding_path)

Select device for rerank model inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    rerank_device = widgets.Dropdown(
        options=support_devices + ["AUTO"],
        value="CPU",
        description="Device:",
        disabled=False,
    )
    
    rerank_device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'GPU', 'AUTO'), value='CPU')



.. code:: ipython3

    print(f"Rerenk model will be loaded to {rerank_device.value} device for text reranking")


.. parsed-literal::

    Rerenk model will be loaded to CPU device for text reranking


Select device for LLM model inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    llm_device = widgets.Dropdown(
        options=support_devices + ["AUTO"],
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


Load models
-----------

`back to top ⬆️ <#Table-of-contents:>`__

Load embedding model
~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

Now a Hugging Face embedding model can be supported by OpenVINO through
```OpenVINOEmbeddings`` <https://python.langchain.com/docs/integrations/text_embedding/openvino>`__
and
```OpenVINOBgeEmbeddings`` <https://python.langchain.com/docs/integrations/text_embedding/openvino#bge-with-openvino>`__\ classes
of LangChain.

.. code:: ipython3

    from langchain_community.embeddings import OpenVINOBgeEmbeddings
    
    embedding_model_name = npu_embedding_dir if USING_NPU else embedding_model_id.value
    batch_size = 1 if USING_NPU else 4
    embedding_model_kwargs = {"device": embedding_device.value, "compile": False}
    encode_kwargs = {
        "mean_pooling": embedding_model_configuration["mean_pooling"],
        "normalize_embeddings": embedding_model_configuration["normalize_embeddings"],
        "batch_size": batch_size,
    }
    
    embedding = OpenVINOBgeEmbeddings(
        model_name_or_path=embedding_model_name,
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    if USING_NPU:
        embedding.ov_model.reshape(1, 512)
    embedding.ov_model.compile()
    
    text = "This is a test document."
    embedding_result = embedding.embed_query(text)
    embedding_result[:3]


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    2024-05-24 00:13:06.057342: I tensorflow/core/util/port.cc:111] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-05-24 00:13:06.061389: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-05-24 00:13:06.108453: E tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:9342] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-05-24 00:13:06.108490: E tensorflow/compiler/xla/stream_executor/cuda/cuda_fft.cc:609] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-05-24 00:13:06.108542: E tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:1518] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-05-24 00:13:06.120406: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-05-24 00:13:06.938926: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    Compiling the model to CPU ...




.. parsed-literal::

    [-0.04208654910326004, 0.06681869924068451, 0.007916687056422234]



Load rerank model
~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

Now a Hugging Face embedding model can be supported by OpenVINO through
```OpenVINOReranker`` <https://python.langchain.com/docs/integrations/document_transformers/openvino_rerank>`__
class of LangChain.

   **Note**: Rerank can be skipped in RAG.

.. code:: ipython3

    from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker
    
    rerank_model_name = rerank_model_id.value
    rerank_model_kwargs = {"device": rerank_device.value}
    rerank_top_n = 2
    
    reranker = OpenVINOReranker(
        model_name_or_path=rerank_model_name,
        model_kwargs=rerank_model_kwargs,
        top_n=rerank_top_n,
    )


.. parsed-literal::

    Compiling the model to CPU ...


Load LLM model
~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

OpenVINO models can be run locally through the ``HuggingFacePipeline``
class. To deploy a model with OpenVINO, you can specify the
``backend="openvino"`` parameter to trigger OpenVINO as backend
inference framework.

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



OpenVINO models can be run locally through the ``HuggingFacePipeline``
class in
`LangChain <https://python.langchain.com/docs/integrations/llms/openvino/>`__.
To deploy a model with OpenVINO, you can specify the
``backend="openvino"`` parameter to trigger OpenVINO as backend
inference framework.

.. code:: ipython3

    from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
    
    if model_to_run.value == "INT4":
        model_dir = int4_model_dir
    elif model_to_run.value == "INT8":
        model_dir = int8_model_dir
    else:
        model_dir = fp16_model_dir
    print(f"Loading model from {model_dir}")
    
    ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}
    
    if "GPU" in llm_device.value and "qwen2-7b-instruct" in llm_model_id.value:
        ov_config["GPU_ENABLE_SDPA_OPTIMIZATION"] = "NO"
    
    # On a GPU device a model is executed in FP16 precision. For red-pajama-3b-chat model there known accuracy
    # issues caused by this, which we avoid by setting precision hint to "f32".
    if llm_model_id.value == "red-pajama-3b-chat" and "GPU" in core.available_devices and llm_device.value in ["GPU", "AUTO"]:
        ov_config["INFERENCE_PRECISION_HINT"] = "f32"
    
    llm = HuggingFacePipeline.from_model_id(
        model_id=str(model_dir),
        task="text-generation",
        backend="openvino",
        model_kwargs={
            "device": llm_device.value,
            "ov_config": ov_config,
            "trust_remote_code": True,
        },
        pipeline_kwargs={"max_new_tokens": 2},
    )
    
    llm.invoke("2 + 2 =")


.. parsed-literal::

    The argument `trust_remote_code` is to be used along with export=True. It will be ignored.


.. parsed-literal::

    Loading model from neural-chat-7b-v3-1/INT4_compressed_weights


.. parsed-literal::

    Compiling the model to CPU ...




.. parsed-literal::

    '2 + 2 = 4'



Run QA over Document
--------------------

`back to top ⬆️ <#Table-of-contents:>`__

Now, when model created, we can setup Chatbot interface using
`Gradio <https://www.gradio.app/>`__.

A typical RAG application has two main components:

-  **Indexing**: a pipeline for ingesting data from a source and
   indexing it. This usually happen offline.

-  **Retrieval and generation**: the actual RAG chain, which takes the
   user query at run time and retrieves the relevant data from the
   index, then passes that to the model.

The most common full sequence from raw data to answer looks like:

**Indexing**

1. ``Load``: First we need to load our data. We’ll use DocumentLoaders
   for this.
2. ``Split``: Text splitters break large Documents into smaller chunks.
   This is useful both for indexing data and for passing it in to a
   model, since large chunks are harder to search over and won’t in a
   model’s finite context window.
3. ``Store``: We need somewhere to store and index our splits, so that
   they can later be searched over. This is often done using a
   VectorStore and Embeddings model.

.. figure:: https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/dfed2ba3-0c3a-4e0e-a2a7-01638730486a
   :alt: Indexing pipeline

   Indexing pipeline

**Retrieval and generation**

1. ``Retrieve``: Given a user input, relevant splits are retrieved from
   storage using a Retriever.
2. ``Generate``: A LLM produces an answer using a prompt that includes
   the question and the retrieved data.

.. figure:: https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/f0545ddc-c0cd-4569-8c86-9879fdab105a
   :alt: Retrieval and generation pipeline

   Retrieval and generation pipeline

.. code:: ipython3

    import re
    from typing import List
    from langchain.text_splitter import (
        CharacterTextSplitter,
        RecursiveCharacterTextSplitter,
        MarkdownTextSplitter,
    )
    from langchain.document_loaders import (
        CSVLoader,
        EverNoteLoader,
        PyPDFLoader,
        TextLoader,
        UnstructuredEPubLoader,
        UnstructuredHTMLLoader,
        UnstructuredMarkdownLoader,
        UnstructuredODTLoader,
        UnstructuredPowerPointLoader,
        UnstructuredWordDocumentLoader,
    )
    
    
    class ChineseTextSplitter(CharacterTextSplitter):
        def __init__(self, pdf: bool = False, **kwargs):
            super().__init__(**kwargs)
            self.pdf = pdf
    
        def split_text(self, text: str) -> List[str]:
            if self.pdf:
                text = re.sub(r"\n{3,}", "\n", text)
                text = text.replace("\n\n", "")
            sent_sep_pattern = re.compile('([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')
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
        ".pdf": (PyPDFLoader, {}),
        ".ppt": (UnstructuredPowerPointLoader, {}),
        ".pptx": (UnstructuredPowerPointLoader, {}),
        ".txt": (TextLoader, {"encoding": "utf8"}),
    }
    
    chinese_examples = [
        ["英特尔®酷睿™ Ultra处理器可以降低多少功耗？"],
        ["相比英特尔之前的移动处理器产品，英特尔®酷睿™ Ultra处理器的AI推理性能提升了多少？"],
        ["英特尔博锐® Enterprise系统提供哪些功能？"],
    ]
    
    english_examples = [
        ["How much power consumption can Intel® Core™ Ultra Processors help save?"],
        ["Compared to Intel’s previous mobile processor, what is the advantage of Intel® Core™ Ultra Processors for Artificial Intelligence?"],
        ["What can Intel vPro® Enterprise systems offer?"],
    ]
    
    if model_language.value == "English":
        text_example_path = "text_example_en.pdf"
    else:
        text_example_path = "text_example_cn.pdf"
    
    examples = chinese_examples if (model_language.value == "Chinese") else english_examples

We can build a RAG pipeline of LangChain through
```create_retrieval_chain`` <https://python.langchain.com/docs/modules/chains/>`__,
which will help to create a chain to connect RAG components including:

-  ```Vector stores`` <https://python.langchain.com/docs/modules/data_connection/vectorstores/>`__\ ，
-  ```Retrievers`` <https://python.langchain.com/docs/modules/data_connection/retrievers/>`__
-  ```LLM`` <https://python.langchain.com/docs/integrations/llms/>`__
-  ```Embedding`` <https://python.langchain.com/docs/integrations/text_embedding/>`__

.. code:: ipython3

    from langchain.prompts import PromptTemplate
    from langchain_community.vectorstores import FAISS
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.docstore.document import Document
    from langchain.retrievers import ContextualCompressionRetriever
    from threading import Thread
    import gradio as gr
    
    stop_tokens = llm_model_configuration.get("stop_tokens")
    rag_prompt_template = llm_model_configuration["rag_prompt_template"]
    
    
    class StopOnTokens(StoppingCriteria):
        def __init__(self, token_ids):
            self.token_ids = token_ids
    
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_id in self.token_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False
    
    
    if stop_tokens is not None:
        if isinstance(stop_tokens[0], str):
            stop_tokens = llm.pipeline.tokenizer.convert_tokens_to_ids(stop_tokens)
    
        stop_tokens = [StopOnTokens(stop_tokens)]
    
    
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
    
    
    text_processor = llm_model_configuration.get("partial_text_processor", default_partial_text_processor)
    
    
    def create_vectordb(docs, spliter_name, chunk_size, chunk_overlap, vector_search_top_k, vector_search_top_n, run_rerank, search_method, score_threshold):
        """
        Initialize a vector database
    
        Params:
          doc: orignal documents provided by user
          chunk_size:  size of a single sentence chunk
          chunk_overlap: overlap size between 2 chunks
          vector_search_top_k: Vector search top k
    
        """
        documents = []
        for doc in docs:
            documents.extend(load_single_document(doc.name))
    
        text_splitter = TEXT_SPLITERS[spliter_name](chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
        texts = text_splitter.split_documents(documents)
    
        global db
        db = FAISS.from_documents(texts, embedding)
    
        global retriever
        if search_method == "similarity_score_threshold":
            search_kwargs = {"k": vector_search_top_k, "score_threshold": score_threshold}
        else:
            search_kwargs = {"k": vector_search_top_k}
        retriever = db.as_retriever(search_kwargs=search_kwargs, search_type=search_method)
        if run_rerank:
            reranker.top_n = vector_search_top_n
            retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)
        prompt = PromptTemplate.from_template(rag_prompt_template)
    
        global combine_docs_chain
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    
        global rag_chain
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
        return "Vector database is Ready"
    
    
    def update_retriever(vector_search_top_k, vector_rerank_top_n, run_rerank, search_method, score_threshold):
        """
        Update retriever
    
        Params:
          vector_search_top_k: size of searching results
          vector_rerank_top_n:  size of rerank results
          run_rerank: whether run rerank step
          search_method: search method used by vector store
    
        """
        global retriever
        global db
        global rag_chain
        global combine_docs_chain
    
        if search_method == "similarity_score_threshold":
            search_kwargs = {"k": vector_search_top_k, "score_threshold": score_threshold}
        else:
            search_kwargs = {"k": vector_search_top_k}
        retriever = db.as_retriever(search_kwargs=search_kwargs, search_type=search_method)
        if run_rerank:
            retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)
            reranker.top_n = vector_rerank_top_n
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    
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
    
    
    def bot(history, temperature, top_p, top_k, repetition_penalty, hide_full_prompt, do_rag):
        """
        callback function for running chatbot on submit button click
    
        Params:
          history: conversation history
          temperature:  parameter for control the level of creativity in AI-generated text.
                        By adjusting the `temperature`, you can influence the AI model's probability distribution, making the text more focused or diverse.
          top_p: parameter for control the range of tokens considered by the AI model based on their cumulative probability.
          top_k: parameter for control the range of tokens considered by the AI model based on their cumulative probability, selecting number of tokens with highest probability.
          repetition_penalty: parameter for penalizing tokens based on how frequently they occur in the text.
          hide_full_prompt: whether to show searching results in promopt.
          do_rag: whether do RAG when generating texts.
    
        """
        streamer = TextIteratorStreamer(
            llm.pipeline.tokenizer,
            timeout=60.0,
            skip_prompt=hide_full_prompt,
            skip_special_tokens=True,
        )
        llm.pipeline._forward_params = dict(
            max_new_tokens=512,
            temperature=temperature,
            do_sample=temperature > 0.0,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
        )
        if stop_tokens is not None:
            llm.pipeline._forward_params["stopping_criteria"] = StoppingCriteriaList(stop_tokens)
    
        if do_rag:
            t1 = Thread(target=rag_chain.invoke, args=({"input": history[-1][0]},))
        else:
            input_text = rag_prompt_template.format(input=history[-1][0], context="")
            t1 = Thread(target=llm.invoke, args=(input_text,))
        t1.start()
    
        # Initialize an empty string to store the generated text
        partial_text = ""
        for new_text in streamer:
            partial_text = text_processor(partial_text, new_text)
            history[-1][1] = partial_text
            yield history
    
    
    def request_cancel():
        llm.pipeline.model.request.cancel()
    
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=".disclaimer {font-variant-caps: all-small-caps;}",
    ) as demo:
        gr.Markdown("""<h1><center>QA over Document</center></h1>""")
        gr.Markdown(f"""<center>Powered by OpenVINO and {llm_model_id.value} </center>""")
        with gr.Row():
            with gr.Column(scale=1):
                docs = gr.File(
                    label="Step 1: Load text files",
                    value=[text_example_path],
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
                load_docs = gr.Button("Step 2: Build Vector Store")
                db_argument = gr.Accordion("Vector Store Configuration", open=False)
                with db_argument:
                    spliter = gr.Dropdown(
                        ["Character", "RecursiveCharacter", "Markdown", "Chinese"],
                        value="RecursiveCharacter",
                        label="Text Spliter",
                        info="Method used to splite the documents",
                        multiselect=False,
                    )
    
                    chunk_size = gr.Slider(
                        label="Chunk size",
                        value=400,
                        minimum=50,
                        maximum=2000,
                        step=50,
                        interactive=True,
                        info="Size of sentence chunk",
                    )
    
                    chunk_overlap = gr.Slider(
                        label="Chunk overlap",
                        value=50,
                        minimum=0,
                        maximum=400,
                        step=10,
                        interactive=True,
                        info=("Overlap between 2 chunks"),
                    )
    
                langchain_status = gr.Textbox(
                    label="Vector Store Status",
                    value="Vector Store is Not ready",
                    interactive=False,
                )
                do_rag = gr.Checkbox(
                    value=True,
                    label="RAG is ON",
                    interactive=True,
                    info="Whether to do RAG for generation",
                )
                with gr.Accordion("Generation Configuration", open=False):
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
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    height=600,
                    label="Step 3: Input Query",
                )
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            msg = gr.Textbox(
                                label="QA Message Box",
                                placeholder="Chat Message Box",
                                show_label=False,
                                container=False,
                            )
                    with gr.Column():
                        with gr.Row():
                            submit = gr.Button("Submit")
                            stop = gr.Button("Stop")
                            clear = gr.Button("Clear")
                gr.Examples(examples, inputs=msg, label="Click on any example and press the 'Submit' button")
                retriever_argument = gr.Accordion("Retriever Configuration", open=True)
                with retriever_argument:
                    with gr.Row():
                        with gr.Row():
                            do_rerank = gr.Checkbox(
                                value=True,
                                label="Rerank searching result",
                                interactive=True,
                            )
                            hide_context = gr.Checkbox(
                                value=True,
                                label="Hide searching result in prompt",
                                interactive=True,
                            )
                        with gr.Row():
                            search_method = gr.Dropdown(
                                ["similarity_score_threshold", "similarity", "mmr"],
                                value="similarity_score_threshold",
                                label="Searching Method",
                                info="Method used to search vector store",
                                multiselect=False,
                                interactive=True,
                            )
                        with gr.Row():
                            score_threshold = gr.Slider(
                                0.01,
                                0.99,
                                value=0.5,
                                step=0.01,
                                label="Similarity Threshold",
                                info="Only working for 'similarity score threshold' method",
                                interactive=True,
                            )
                        with gr.Row():
                            vector_rerank_top_n = gr.Slider(
                                1,
                                10,
                                value=2,
                                step=1,
                                label="Rerank top n",
                                info="Number of rerank results",
                                interactive=True,
                            )
                        with gr.Row():
                            vector_search_top_k = gr.Slider(
                                1,
                                50,
                                value=10,
                                step=1,
                                label="Search top k",
                                info="Number of searching results, must >= Rerank top n",
                                interactive=True,
                            )
        load_docs.click(
            create_vectordb,
            inputs=[docs, spliter, chunk_size, chunk_overlap, vector_search_top_k, vector_rerank_top_n, do_rerank, search_method, score_threshold],
            outputs=[langchain_status],
            queue=False,
        )
        submit_event = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot,
            [chatbot, temperature, top_p, top_k, repetition_penalty, hide_context, do_rag],
            chatbot,
            queue=True,
        )
        submit_click_event = submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot,
            [chatbot, temperature, top_p, top_k, repetition_penalty, hide_context, do_rag],
            chatbot,
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
        vector_search_top_k.release(
            update_retriever,
            [vector_search_top_k, vector_rerank_top_n, do_rerank, search_method, score_threshold],
        )
        vector_rerank_top_n.release(
            update_retriever,
            [vector_search_top_k, vector_rerank_top_n, do_rerank, search_method, score_threshold],
        )
        do_rerank.change(
            update_retriever,
            [vector_search_top_k, vector_rerank_top_n, do_rerank, search_method, score_threshold],
        )
        search_method.change(
            update_retriever,
            [vector_search_top_k, vector_rerank_top_n, do_rerank, search_method, score_threshold],
        )
        score_threshold.change(
            update_retriever,
            [vector_search_top_k, vector_rerank_top_n, do_rerank, search_method, score_threshold],
        )
    
    
    demo.queue()
    # if you are launching remotely, specify server_name and server_port
    #  demo.launch(server_name='your server name', server_port='server port in int')
    # if you have any issue to launch on your platform, you can pass share=True to launch method:
    # demo.launch(share=True)
    # it creates a publicly shareable link for the interface. Read more in the docs: https://gradio.app/docs/
    demo.launch()

.. code:: ipython3

    # please run this cell for stopping gradio interface
    demo.close()
