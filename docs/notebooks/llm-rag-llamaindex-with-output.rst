Create a RAG system using OpenVINO and LlamaIndex
=================================================

**Retrieval-augmented generation (RAG)** is a technique for augmenting
LLM knowledge with additional, often private or real-time, data. LLMs
can reason about wide-ranging topics, but their knowledge is limited to
the public data up to a specific point in time that they were trained
on. If you want to build AI applications that can reason about private
data or data introduced after a model’s cutoff date, you need to augment
the knowledge of the model with the specific information it needs. The
process of bringing the appropriate information and inserting it into
the model prompt is known as Retrieval Augmented Generation (RAG).

`LlamaIndex <https://docs.llamaindex.ai/en/stable/>`__ is a framework
for building context-augmented generative AI applications with
LLMs.LlamaIndex imposes no restriction on how you use LLMs. You can use
LLMs as auto-complete, chatbots, semi-autonomous agents, and more. It
just makes using them easier.

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
    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu\
    "llama-index" "pymupdf" "llama-index-readers-file" "llama-index-vector-stores-faiss" "llama-index-llms-openvino" "llama-index-embeddings-openvino" "llama-index-postprocessor-openvino-rerank" "transformers>=4.40"\
    "git+https://github.com/huggingface/optimum-intel.git"\
    "git+https://github.com/openvinotoolkit/nncf.git"\
    "datasets"\
    "accelerate"\
    "gradio"
    %pip install --pre -Uq openvino openvino-tokenizers[transformers] --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    WARNING: Skipping optimum as it is not installed.
    WARNING: Skipping optimum-intel as it is not installed.
    Note: you may need to restart the kernel to use updated packages.
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    langchain-huggingface 0.0.1 requires huggingface-hub>=0.23.0, but you have huggingface-hub 0.20.3 which is incompatible.
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
    import ipywidgets as widgets

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

    Dropdown(description='Model:', index=12, options=('tiny-llama-1b-chat', 'gemma-2b-it', 'red-pajama-3b-chat', '…



.. code:: ipython3

    llm_model_configuration = SUPPORTED_LLM_MODELS[model_language.value][llm_model_id.value]
    print(f"Selected LLM model {llm_model_id.value}")


.. parsed-literal::

    Selected LLM model neural-chat-7b-v3-1


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



.. parsed-literal::

    Checkbox(value=False, description='Enable AWQ')


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

    Size of model with INT4 compressed weights is 5069.90 MB


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

    Dropdown(description='Embedding Model:', options=('bge-small-en-v1.5', 'bge-large-en-v1.5'), value='bge-small-…



.. code:: ipython3

    embedding_model_configuration = SUPPORTED_EMBEDDING_MODELS[model_language.value][embedding_model_id.value]
    print(f"Selected {embedding_model_id.value} model")


.. parsed-literal::

    Selected bge-small-en-v1.5 model


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
```OpenVINOEmbeddings`` <https://docs.llamaindex.ai/en/stable/examples/embeddings/openvino/>`__
class of LlamaIndex.

.. code:: ipython3

    from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
    
    
    embedding = OpenVINOEmbedding(folder_name=embedding_model_id.value, device=embedding_device.value)
    
    embeddings = embedding.get_text_embedding("Hello World!")
    print(len(embeddings))
    print(embeddings[:5])


.. parsed-literal::

    Compiling the model to CPU ...


.. parsed-literal::

    384
    [-0.003275666618719697, -0.01169075071811676, 0.04155930131673813, -0.03814813867211342, 0.02418304793536663]


Load rerank model
~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

Now a Hugging Face embedding model can be supported by OpenVINO through
```OpenVINORerank`` <https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/openvino_rerank/>`__
class of LlamaIndex.

   **Note**: Rerank can be skipped in RAG.

.. code:: ipython3

    from llama_index.postprocessor.openvino_rerank import OpenVINORerank
    
    reranker = OpenVINORerank(model=rerank_model_id.value, device=rerank_device.value, top_n=2)


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



OpenVINO models can be run locally through the ``OpenVINOLLM`` class in
`LlamaIndex <https://docs.llamaindex.ai/en/stable/examples/llm/openvino/>`__.
If you have an Intel GPU, you can specify ``device_map="gpu"`` to run
inference on it.

.. code:: ipython3

    from llama_index.llms.openvino import OpenVINOLLM
    
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
    
    llm = OpenVINOLLM(
        model_name=str(model_dir),
        tokenizer_name=str(model_dir),
        context_window=3900,
        max_new_tokens=2,
        model_kwargs={"ov_config": ov_config, "trust_remote_code": True},
        generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
        device_map=llm_device.value,
    )
    
    response = llm.complete("2 + 2 =")
    print(str(response))


.. parsed-literal::

    The argument `trust_remote_code` is to be used along with export=True. It will be ignored.


.. parsed-literal::

    Loading model from neural-chat-7b-v3-1/INT4_compressed_weights


.. parsed-literal::

    Compiling the model to CPU ...
    /home/ethan/intel/openvino_notebooks/openvino_env/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:492: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.7` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
      warnings.warn(
    /home/ethan/intel/openvino_notebooks/openvino_env/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:497: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.95` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
      warnings.warn(
    Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.


.. parsed-literal::

    4


Run QA over Document
--------------------

`back to top ⬆️ <#Table-of-contents:>`__

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

.. code:: ipython3

    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core import Settings
    from llama_index.readers.file import PyMuPDFReader
    from llama_index.vector_stores.faiss import FaissVectorStore
    import faiss
    
    if model_language.value == "English":
        text_example_path = "text_example_en.pdf"
    else:
        text_example_path = "text_example_cn.pdf"
    
    loader = PyMuPDFReader()
    documents = loader.load(file_path=text_example_path)
    
    # dimensions of embedding model
    d = embedding._model.request.outputs[0].get_partial_shape()[2].get_length()
    faiss_index = faiss.IndexFlatL2(d)
    Settings.embed_model = embedding
    llm.max_new_tokens = 2048
    Settings.llm = llm
    
    
    vector_store = FaissVectorStore(faiss_index=faiss_index)

**Retrieval and generation**

1. ``Retrieve``: Given a user input, relevant splits are retrieved from
   storage using a Retriever.
2. ``Generate``: A LLM produces an answer using a prompt that includes
   the question and the retrieved data.

.. figure:: https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/f0545ddc-c0cd-4569-8c86-9879fdab105a
   :alt: Retrieval and generation pipeline

   Retrieval and generation pipeline

.. code:: ipython3

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[SentenceSplitter(chunk_size=500, chunk_overlap=50)],
    )
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=10, node_postprocessors=[reranker])
    if model_language.value == "English":
        query = "What can Intel vPro® Enterprise systems offer?"
    else:
        query = "英特尔博锐® Enterprise系统提供哪些功能？"
    
    streaming_response = query_engine.query(query)
    streaming_response.print_response_stream()


.. parsed-literal::

    Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.


.. parsed-literal::

    Intel vPro® Enterprise systems offer a range of features and capabilities, including dynamic root of trust, system management mode (SMM) protections, memory encryption with multi-key support, OS kernel protection, out-of-band management with remote KVM control, unique device identifier, device history, and in-band manageability plug-ins. These features are part of the strong portfolio of security and manageability technologies that form the foundation of the Intel vPro platform, which delivers differentiated capabilities to organizations of all sizes.
