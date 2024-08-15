Create an Agentic RAG using OpenVINO and LlamaIndex
===================================================

An **agent** is an automated reasoning and decision engine. It takes in
a user input/query and can make internal decisions for executing that
query in order to return the correct result. The key agent components
can include, but are not limited to:

-  Breaking down a complex question into smaller ones
-  Choosing an external Tool to use + coming up with parameters for
   calling the Tool
-  Planning out a set of tasks
-  Storing previously completed tasks in a memory module

`LlamaIndex <https://docs.llamaindex.ai/en/stable/>`__ is a framework
for building context-augmented generative AI applications with
LLMs.LlamaIndex imposes no restriction on how you use LLMs. You can use
LLMs as auto-complete, chatbots, semi-autonomous agents, and more. It
just makes using them easier. You can build agents on top of your
existing LlamaIndex RAG pipeline to empower it with automated decision
capabilities. A lot of modules (routing, query transformations, and
more) are already agentic in nature in that they use LLMs for decision
making.

**Agentic RAG = Agent-based RAG implementation**

While standard RAG excels at simple queries across a few documents,
agentic RAG takes it a step further and emerges as a potent solution for
question answering. It introduces a layer of intelligence by employing
AI agents. These agents act as autonomous decision-makers, analyzing
initial findings and strategically selecting the most effective tools
for further data retrieval. This multi-step reasoning capability
empowers agentic RAG to tackle intricate research tasks, like
summarizing, comparing information across multiple documents and even
formulating follow-up questions -all in an orchestrated and efficient
manner.

.. figure:: https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/871cb90d-27fd-4a87-aa3c-f4cdb199a148
   :alt: agentic-rag

   agentic-rag

This example will demonstrate using RAG engines as a tool in an agent
with OpenVINO and LlamaIndex.

**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Download models <#download-models>`__

   -  `Download LLM <#download-llm>`__
   -  `Download Embedding model <#download-embedding-model>`__

-  `Create models <#create-models>`__

   -  `Create OpenVINO LLM <#create-openvino-llm>`__
   -  `Create OpenVINO Embedding <#create-openvino-embedding>`__

-  `Create tools <#create-tools>`__
-  `Run Agentic RAG <#run-agentic-rag>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Prerequisites
-------------



Install required dependencies

.. code:: ipython3

    import os

    os.environ["GIT_CLONE_PROTECTION_ACTIVE"] = "false"

    %pip uninstall -q -y optimum optimum-intel
    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu \
    "llama-index" "pymupdf" "llama-index-readers-file" "llama-index-llms-openvino>=0.2.0" "llama-index-embeddings-openvino>=0.2.0" "transformers>=4.40"

    %pip install -q "git+https://github.com/huggingface/optimum-intel.git" \
    "git+https://github.com/openvinotoolkit/nncf.git" \
    "datasets" \
    "accelerate"
    %pip install --pre -Uq "openvino>=2024.2.0" openvino-tokenizers[transformers] --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

.. code:: ipython3

    from pathlib import Path
    import requests
    import io

    text_example_en_path = Path("text_example_en.pdf")
    text_example_en = "https://github.com/user-attachments/files/16171326/xeon6-e-cores-network-and-edge-brief.pdf"

    if not text_example_en_path.exists():
        r = requests.get(url=text_example_en)
        content = io.BytesIO(r.content)
        with open("text_example_en.pdf", "wb") as f:
            f.write(content.read())

Download models
---------------



Download LLM
~~~~~~~~~~~~



To run LLM locally, we have to download the model in the first step. It
is possible to `export your
model <https://github.com/huggingface/optimum-intel?tab=readme-ov-file#export>`__
to the OpenVINO IR format with the CLI, and load the model from local
folder.

Large Language Models (LLMs) are a core component of agent. LlamaIndex
does not serve its own LLMs, but rather provides a standard interface
for interacting with many different LLMs. In this example, we select
``Meta-Llama-3-8B-Instruct`` as LLM in agent pipeline.

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
   agreement. >You must be a registered user in Hugging Face Hub.
   Please visit `HuggingFace model
   card <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>`__,
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

    llm_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm_model_path = "Meta-Llama-3-8B-Instruct-ov"

    if not Path(llm_model_path).exists():
        !optimum-cli export openvino --model {llm_model_id} --task text-generation-with-past --trust-remote-code --weight-format int4 {llm_model_path}

Download Embedding model
~~~~~~~~~~~~~~~~~~~~~~~~



Embedding model is another key component in RAG pipeline. It takes text
as input, and return a long list of numbers used to capture the
semantics of the text. An OpenVINO embedding model and tokenizer can be
exported by ``feature-extraction`` task with ``optimum-cli``. In this
tutorial, we use
`bge-small-en-v1.5 <https://huggingface.co/BAAI/bge-small-en-v1.5>`__ as
example.

.. code:: ipython3

    embedding_model_id = "BAAI/bge-small-en-v1.5"
    embedding_model_path = "bge-small-en-v1.5-ov"

    if not Path(embedding_model_path).exists():
        !optimum-cli export openvino --model {embedding_model_id} --task feature-extraction {embedding_model_path}

Create models
-------------



Create OpenVINO LLM
~~~~~~~~~~~~~~~~~~~



Select device for LLM model inference

.. code:: ipython3

    import ipywidgets as widgets
    import openvino as ov

    core = ov.Core()

    support_devices = core.available_devices

    llm_device = widgets.Dropdown(
        options=support_devices + ["AUTO"],
        value="CPU",
        description="Device:",
        disabled=False,
    )

    llm_device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



OpenVINO models can be run locally through the ``OpenVINOLLM`` class in
`LlamaIndex <https://docs.llamaindex.ai/en/stable/examples/llm/openvino/>`__.
If you have an Intel GPU, you can specify ``device_map="gpu"`` to run
inference on it.

.. code:: ipython3

    from llama_index.llms.openvino import OpenVINOLLM

    ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}


    def completion_to_prompt(completion):
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>user<|end_header_id|>{completion}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"


    llm = OpenVINOLLM(
        model_id_or_path=str(llm_model_path),
        context_window=3900,
        max_new_tokens=1000,
        model_kwargs={"ov_config": ov_config},
        device_map=llm_device.value,
        completion_to_prompt=completion_to_prompt,
    )


.. parsed-literal::

    Compiling the model to CPU ...
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.


Create OpenVINO Embedding
~~~~~~~~~~~~~~~~~~~~~~~~~



Select device for embedding model inference

.. code:: ipython3

    support_devices = core.available_devices

    embedding_device = widgets.Dropdown(
        options=support_devices + ["AUTO"],
        value="CPU",
        description="Device:",
        disabled=False,
    )

    embedding_device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



A Hugging Face embedding model can be supported by OpenVINO through
`OpenVINOEmbeddings <https://docs.llamaindex.ai/en/stable/examples/embeddings/openvino/>`__
class of LlamaIndex.

.. code:: ipython3

    from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding

    embedding = OpenVINOEmbedding(model_id_or_path=embedding_model_path, device=embedding_device.value)


.. parsed-literal::

    Compiling the model to CPU ...


Create tools
------------



In this examples, we will create 2 customized tools for ``multiply`` and
``add``.

.. code:: ipython3

    from llama_index.core.agent import ReActAgent
    from llama_index.core.tools import FunctionTool


    def multiply(a: float, b: float) -> float:
        """Multiply two numbers and returns the product"""
        return a * b


    multiply_tool = FunctionTool.from_defaults(fn=multiply)


    def add(a: float, b: float) -> float:
        """Add two numbers and returns the sum"""
        return a + b


    add_tool = FunctionTool.from_defaults(fn=add)

To demonstrate using RAG engines as a tool in an agent, we’re going to
create a very simple RAG query engine as one of the tools.

   **Note**: For a full RAG pipeline with OpenVINO, you can check the
   `RAG notebooks <llm-rag-llamaindex-with-output.html>`__

.. code:: ipython3

    from llama_index.readers.file import PyMuPDFReader
    from llama_index.core import VectorStoreIndex, Settings

    Settings.embed_model = embedding
    Settings.llm = llm
    loader = PyMuPDFReader()
    documents = loader.load(file_path=text_example_en_path)
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=2)

Now we turn our query engine into a tool by supplying the appropriate
metadata (for the python functions, this was being automatically
extracted so we didn’t need to add it):

.. code:: ipython3

    from llama_index.core.tools import QueryEngineTool

    rag_tool = QueryEngineTool.from_defaults(
        query_engine,
        name="Xeon6",
        description="A RAG engine with some basic facts about Intel Xeon 6 processors with E-cores",
    )

Run Agentic RAG
---------------



We modify our agent by adding this engine to our array of tools (we also
remove the llm parameter, since it’s now provided by settings):

.. code:: ipython3

    agent = ReActAgent.from_tools([multiply_tool, add_tool, rag_tool], llm=llm, verbose=True)

Ask a question using multiple tools.

.. code:: ipython3

    response = agent.chat("What's the maximum number of cores in an Intel Xeon 6 processor server with 4 sockets ? Go step by step, using a tool to do any math.")


.. parsed-literal::

    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


.. parsed-literal::

    Thought: The current language of the user is English. I need to use a tool to help me answer the question.
    Action: Xeon6
    Action Input: {'input': 'maximum cores in a single socket'}


.. parsed-literal::

    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


.. parsed-literal::

    Observation:

    According to the provided context information, the maximum cores in a single socket is 144.


.. parsed-literal::

    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


.. parsed-literal::

    Thought: The current language of the user is English. I need to use a tool to help me answer the question.
    Action: multiply
    Action Input: {'a': 144, 'b': 4}
    Observation: 576
    Thought: The current language of the user is English. I can answer without using any more tools. I'll use the user's language to answer
    Answer: The maximum number of cores in an Intel Xeon 6 processor server with 4 sockets is 576.

