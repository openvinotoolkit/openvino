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
    "llama-index" "llama-index-readers-file" "llama-index-llms-openvino>=0.2.2" "llama-index-embeddings-openvino>=0.2.0" "transformers>=4.40"

    %pip install -q "git+https://github.com/huggingface/optimum-intel.git" \
    "git+https://github.com/openvinotoolkit/nncf.git" \
    "datasets" \
    "accelerate"
    %pip install --pre -Uq "openvino>=2024.2.0" openvino-tokenizers[transformers] --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly


.. parsed-literal::

    WARNING: Skipping optimum as it is not installed.
    WARNING: Skipping optimum-intel as it is not installed.
    Note: you may need to restart the kernel to use updated packages.


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
report <https://aka.ms/phi3-tech-report>`__. \* **llama-3-8b-instruct**
- Llama 3 is an auto-regressive language model that uses an optimized
transformer architecture. The tuned versions use supervised fine-tuning
(SFT) and reinforcement learning with human feedback (RLHF) to align
with human preferences for helpfulness and safety. The Llama 3
instruction tuned models are optimized for dialogue use cases and
outperform many of the available open source chat models on common
industry benchmarks. More details about model can be found in `Meta blog
post <https://ai.meta.com/blog/meta-llama-3/>`__, `model
website <https://llama.meta.com/llama3>`__ and `model
card <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>`__.
>\ **Note**: run model with demo, you will need to accept license
agreement. >You must be a registered user in Hugging Face Hub. Please
visit `HuggingFace model
card <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>`__,
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

    llm_model_ids = ["OpenVINO/Phi-3-mini-4k-instruct-int4-ov", "meta-llama/Meta-Llama-3-8B-Instruct"]

    llm_model_id = widgets.Dropdown(
        options=llm_model_ids,
        value=llm_model_ids[0],
        description="Model:",
        disabled=False,
    )

    llm_model_id




.. parsed-literal::

    Dropdown(description='Model:', options=('OpenVINO/Phi-3-mini-4k-instruct-int4-ov', 'meta-llama/Meta-Llama-3-8B…



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



.. parsed-literal::

    Fetching 16 files:   0%|          | 0/16 [00:00<?, ?it/s]



.. parsed-literal::

    .gitattributes:   0%|          | 0.00/1.52k [00:00<?, ?B/s]



.. parsed-literal::

    config.json:   0%|          | 0.00/884 [00:00<?, ?B/s]



.. parsed-literal::

    README.md:   0%|          | 0.00/3.75k [00:00<?, ?B/s]



.. parsed-literal::

    openvino_detokenizer.xml:   0%|          | 0.00/3.25k [00:00<?, ?B/s]



.. parsed-literal::

    generation_config.json:   0%|          | 0.00/172 [00:00<?, ?B/s]



.. parsed-literal::

    configuration_phi3.py:   0%|          | 0.00/10.4k [00:00<?, ?B/s]



.. parsed-literal::

    openvino_model.xml:   0%|          | 0.00/3.04M [00:00<?, ?B/s]



.. parsed-literal::

    openvino_detokenizer.bin:   0%|          | 0.00/500k [00:00<?, ?B/s]



.. parsed-literal::

    special_tokens_map.json:   0%|          | 0.00/569 [00:00<?, ?B/s]



.. parsed-literal::

    added_tokens.json:   0%|          | 0.00/293 [00:00<?, ?B/s]



.. parsed-literal::

    openvino_tokenizer.xml:   0%|          | 0.00/12.7k [00:00<?, ?B/s]



.. parsed-literal::

    openvino_model.bin:   0%|          | 0.00/2.45G [00:00<?, ?B/s]



.. parsed-literal::

    openvino_tokenizer.bin:   0%|          | 0.00/500k [00:00<?, ?B/s]



.. parsed-literal::

    tokenizer.json:   0%|          | 0.00/1.84M [00:00<?, ?B/s]



.. parsed-literal::

    tokenizer_config.json:   0%|          | 0.00/3.34k [00:00<?, ?B/s]



.. parsed-literal::

    tokenizer.model:   0%|          | 0.00/500k [00:00<?, ?B/s]


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

    [ERROR] 09:10:57.134 [NPUBackends] Cannot find backend for inference. Make sure the device is available.




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
        return f"<|system|><|end|><|user|>{completion}<|end|><|assistant|>\n"


    def messages_to_prompt(messages):
        prompt = ""
        for message in messages:
            if message.role == "system":
                prompt += f"<|system|>{message.content}<|end|>"
            elif message.role == "user":
                prompt += f"<|user|>{message.content}<|end|>"
            elif message.role == "assistant":
                prompt += f"<|assistant|>{message.content}<|end|>"

        # ensure we start with a system prompt, insert blank if needed
        if not prompt.startswith("<|system|>"):
            prompt = "<|system|><|end|>" + prompt

        # add final assistant prompt
        prompt = prompt + "<|assistant|>\n"

        return prompt


    llm = OpenVINOLLM(
        model_id_or_path=str(llm_model_path),
        context_window=3900,
        max_new_tokens=1000,
        model_kwargs={"ov_config": ov_config},
        generate_kwargs={"do_sample": False, "temperature": None, "top_p": None},
        completion_to_prompt=completion_to_prompt,
        messages_to_prompt=messages_to_prompt,
        device_map=llm_device.value,
    )


.. parsed-literal::

    /home/ethan/intel/openvino_notebooks/openvino_env/lib/python3.11/site-packages/pydantic/_internal/_fields.py:161: UserWarning: Field "model_id" has conflict with protected namespace "model_".

    You may be able to resolve this warning by setting `model_config['protected_namespaces'] = ()`.
      warnings.warn(
    Compiling the model to CPU ...


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
            description="Useful for searching for basic facts about Intel Xeon 6 processors",
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

    response = agent.chat("What's the maximum number of cores of 6731 sockets of Intel Xeon 6 processors ? Go step by step, using a tool to do any math.")


.. parsed-literal::

    > Running step 49b4846c-74d1-4766-9321-06aa21df11e1. Step input: What's the maximum number of cores of 6731 sockets of Intel Xeon 6 processors ? Go step by step, using a tool to do any math.
    Thought: The current language of the user is English. I need to use a tool to help me answer the question.
    Action: vector_search
    Action Input: {'input': 'maximum number of cores of Intel Xeon 6 processors'}
    Observation: The Intel Xeon 6 processors with Efficient-cores have up to 144 cores per socket in 1- or 2-socket configurations.
    > Running step cb9c570c-09b4-4ac6-9121-cd1ef67e63cc. Step input: None
    Thought: I can answer without using any more tools. I'll use the user's language to answer.
    Answer: The maximum number of cores for Intel Xeon 6 processors is 144 cores per socket.
    Support: To calculate the maximum number of cores for 6731 sockets, you would multiply the number of cores per socket by the number of sockets.
    Action: multiply
    Action Input: {'a': 144, 'b': 6731}
    Observation: 969264
    > Running step 9601596a-63db-4791-92fe-750f3a0cd924. Step input: None
    Thought: I can answer without using any more tools. I'll use the user's language to answer.
    Answer: The maximum number of cores for 6731 sockets of Intel Xeon 6 processors is 969,264 cores.


.. code:: ipython3

    agent.reset()
