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
    import requests
    
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/pip_helper.py",
    )
    open("pip_helper.py", "w").write(r.text)
    
    os.environ["GIT_CLONE_PROTECTION_ACTIVE"] = "false"
    
    from pip_helper import pip_install
    
    pip_install(
        "-q",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cpu",
        "llama-index",
        "llama-index-llms-huggingface==0.3.3",  # pin to keep compatibility due to https://github.com/run-llama/llama_index/commit/f037de8d0471b37f9c4069ebef5dfb329633d2c6
        "llama-index-readers-file",
        "llama-index-core",
        "llama-index-llms-huggingface",
        "llama-index-embeddings-huggingface",
        "transformers>=4.43.1",
        "llama-index-llms-huggingface>=0.3.0,<0.3.4",
        "llama-index-embeddings-huggingface>=0.3.0",
    )
    pip_install("-q", "git+https://github.com/huggingface/optimum-intel.git", "git+https://github.com/openvinotoolkit/nncf.git", "datasets", "accelerate")
    pip_install("--pre", "-Uq", "openvino>=2024.2.0", "--extra-index-url", "https://storage.openvinotoolkit.org/simple/wheels/nightly")
    pip_install("--pre", "-Uq", "openvino-tokenizers[transformers]", "--extra-index-url", "https://storage.openvinotoolkit.org/simple/wheels/nightly")
    pip_install("-q", "--no-deps", "llama-index-llms-openvino", "llama-index-embeddings-openvino")

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
for interacting with many different LLMs. In this example, we can select
``Phi3-mini-instruct`` or ``Meta-Llama-3-8B-Instruct`` as LLM in agent
pipeline. \* **phi3-mini-instruct** - The Phi-3-Mini is a 3.8B
parameters, lightweight, state-of-the-art open model trained with the
Phi-3 datasets that includes both synthetic data and the filtered
publicly available websites data with a focus on high-quality and
reasoning dense properties. More details about model can be found in
`model
card <https://huggingface.co/microsoft/Phi-3-mini-4k-instruct>`__,
`Microsoft blog <https://aka.ms/phi3blog-april>`__ and `technical
report <https://aka.ms/phi3-tech-report>`__. \*
**llama-3.1-8b-instruct** - The Llama 3.1 instruction tuned text only
models (8B, 70B, 405B) are optimized for multilingual dialogue use cases
and outperform many of the available open source and closed chat models
on common industry benchmarks. More details about model can be found in
`Meta blog post <https://ai.meta.com/blog/meta-llama-3-1/>`__, `model
website <https://llama.meta.com>`__ and `model
card <https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct>`__.
>\ **Note**: run model with demo, you will need to accept license
agreement. >You must be a registered user in Hugging Face Hub. Please
visit `HuggingFace model
card <https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct>`__,
carefully read terms of usage and click accept button. You will need to
use an access token for the code below to run. For more information on
access tokens, refer to `this section of the
documentation <https://huggingface.co/docs/hub/security-tokens>`__. >You
can login on Hugging Face Hub in notebook environment, using following
code:

.. code:: python

       ## login to huggingfacehub to get access to pretrained model 

       from huggingface_hub import notebook_login, whoami

       try:
           whoami()
           print('Authorization token already provided')
       except OSError:
           notebook_login()

.. code:: ipython3

    import ipywidgets as widgets
    
    llm_model_ids = ["OpenVINO/Phi-3-mini-4k-instruct-int4-ov", "meta-llama/Meta-Llama-3.1-8B-Instruct"]
    
    llm_model_id = widgets.Dropdown(
        options=llm_model_ids,
        value=llm_model_ids[0],
        description="Model:",
        disabled=False,
    )
    
    llm_model_id




.. parsed-literal::

    Dropdown(description='Model:', options=('OpenVINO/Phi-3-mini-4k-instruct-int4-ov', 'meta-llama/Meta-Llama-3.1-…



.. code:: ipython3

    from pathlib import Path
    import huggingface_hub as hf_hub
    
    llm_model_path = llm_model_id.value.split("/")[-1]
    repo_name = llm_model_id.value.split("/")[0]
    
    if not Path(llm_model_path).exists():
        if repo_name == "OpenVINO":
            hf_hub.snapshot_download(llm_model_id.value, local_dir=llm_model_path)
        else:
            !optimum-cli export openvino --model {llm_model_id.value} --task text-generation-with-past --trust-remote-code --weight-format int4 --group-size 128 --ratio 0.8 {llm_model_path}

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
    embedding_model_path = "bge-small-en-v1.5"
    
    if not Path(embedding_model_path).exists():
        !optimum-cli export openvino --model {embedding_model_id} --task feature-extraction {embedding_model_path}

Create models
-------------



Create OpenVINO LLM
~~~~~~~~~~~~~~~~~~~



Select device for LLM model inference

.. code:: ipython3

    from notebook_utils import device_widget
    
    llm_device = device_widget("CPU", exclude=["NPU"])
    
    llm_device

OpenVINO models can be run locally through the ``OpenVINOLLM`` class in
`LlamaIndex <https://docs.llamaindex.ai/en/stable/examples/llm/openvino/>`__.
If you have an Intel GPU, you can specify ``device_map="gpu"`` to run
inference on it.

.. code:: ipython3

    from llama_index.llms.openvino import OpenVINOLLM
    
    import openvino.properties as props
    import openvino.properties.hint as hints
    import openvino.properties.streams as streams
    
    
    ov_config = {hints.performance_mode(): hints.PerformanceMode.LATENCY, streams.num(): "1", props.cache_dir(): ""}
    
    
    def phi_completion_to_prompt(completion):
        return f"<|system|><|end|><|user|>{completion}<|end|><|assistant|>\n"
    
    
    def llama3_completion_to_prompt(completion):
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{completion}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    
    llm = OpenVINOLLM(
        model_id_or_path=str(llm_model_path),
        context_window=3900,
        max_new_tokens=1000,
        model_kwargs={"ov_config": ov_config},
        generate_kwargs={"do_sample": False, "temperature": None, "top_p": None},
        completion_to_prompt=phi_completion_to_prompt if llm_model_path == "Phi-3-mini-4k-instruct-int4-ov" else llama3_completion_to_prompt,
        device_map=llm_device.value,
    )


.. parsed-literal::

    Compiling the model to CPU ...


Create OpenVINO Embedding
~~~~~~~~~~~~~~~~~~~~~~~~~



Select device for embedding model inference

.. code:: ipython3

    embedding_device = device_widget()
    
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
    
    
    def divide(a: float, b: float) -> float:
        """Add two numbers and returns the sum"""
        return a / b
    
    
    divide_tool = FunctionTool.from_defaults(fn=divide)

To demonstrate using RAG engines as a tool in an agent, we’re going to
create a very simple RAG query engine as one of the tools.

   **Note**: For a full RAG pipeline with OpenVINO, you can check the
   `RAG notebooks <llm-rag-llamaindex-with-output.html>`__

.. code:: ipython3

    from llama_index.core import SimpleDirectoryReader
    from llama_index.core import VectorStoreIndex, Settings
    
    Settings.embed_model = embedding
    Settings.llm = llm
    
    reader = SimpleDirectoryReader(input_files=[text_example_en_path])
    documents = reader.load_data()
    index = VectorStoreIndex.from_documents(
        documents,
    )

Now we turn our query engine into a tool by supplying the appropriate
metadata (for the python functions, this was being automatically
extracted so we didn’t need to add it):

.. code:: ipython3

    from llama_index.core.tools import QueryEngineTool, ToolMetadata
    
    vector_tool = QueryEngineTool(
        index.as_query_engine(streaming=True),
        metadata=ToolMetadata(
            name="vector_search",
            description="Useful for searching for basic facts about 'Intel Xeon 6 processors'",
        ),
    )

Run Agentic RAG
---------------



We modify our agent by adding this engine to our array of tools (we also
remove the llm parameter, since it’s now provided by settings):

.. code:: ipython3

    agent = ReActAgent.from_tools([multiply_tool, divide_tool, vector_tool], llm=llm, verbose=True)

Ask a question using multiple tools.

.. code:: ipython3

    response = agent.chat("What's the maximum number of cores of 8 sockets of 'Intel Xeon 6 processors' ? Go step by step, using a tool to do any math.")


.. parsed-literal::

    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


.. parsed-literal::

    > Running step ee829c21-5642-423d-afcf-27e894aede35. Step input: What's the maximum number of cores of 8 sockets of 'Intel Xeon 6 processors' ? Go step by step, using a tool to do any math.


.. parsed-literal::

    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


.. parsed-literal::

    Thought: The current language of the user is English. I need to use a tool to help me answer the question.
    Action: vector_search
    Action Input: {'input': 'Intel Xeon 6 processors'}
    

.. parsed-literal::

    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


.. parsed-literal::

    Observation: According to the provided text, Intel Xeon 6 processors with Efficient-cores are described as having the following features and benefits:
    
    * Up to 144 cores per socket in 1- or 2-socket configurations, boosting processing capacity, accelerating service mesh performance, and decreasing transaction latency.
    * Improved power efficiency and lower idle power ISO configurations, contributing to enhanced sustainability with a TDP range of 205W-330W.
    * Intel QuickAssist Technology (Intel QAT) drives fast encryption/key protection, while Intel Software Guard Extensions (Intel SGX) and Intel Trust Domain Extensions (Intel TDX) enable confidential computing for regulated workloads.
    * Intel Xeon 6 processor-based platforms with Intel Ethernet 800 Series Network Adapters set the bar for maximum 5G core workload performance and lower operating costs.
    
    These processors are suitable for various industries, including:
    
    * Telecommunications: 5G core networks, control plane (CP), and user plane functions (UPF)
    * Enterprise: Network security appliances, secure access service edge (SASE), next-gen firewall (NGFW), real-time deep packet inspection, antivirus, intrusion prevention and detection, and SSL/TLS inspection
    * Media and Entertainment: Content delivery networks, media processing, video on demand (VOD)
    * Industrial/Energy: Digitalization of automation, protection, and control
    
    The processors are also mentioned to be suitable for various use cases, including:
    
    * 5G core networks
    * Network security appliances
    * Content delivery networks
    * Media processing
    * Video on demand (VOD)
    * Digitalization of automation, protection, and control in industrial and energy sectors
    > Running step c8d3f8b5-0a3e-4254-87a8-c13cd4f992ad. Step input: None


.. parsed-literal::

    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


.. parsed-literal::

    Thought: The current language of the user is English. I need to use a tool to help me answer the question.
    Action: multiply
    Action Input: {'a': 8, 'b': 144}
    Observation: 1152
    > Running step 437a7fcf-7f53-4d7c-b3d4-06b2714a1b9d. Step input: None
    Thought: The current language of the user is English. I can answer without using any more tools. I'll use the user's language to answer.
    Answer: The maximum number of cores of 8 sockets of 'Intel Xeon 6 processors' is 1152.
    

.. code:: ipython3

    agent.reset()
