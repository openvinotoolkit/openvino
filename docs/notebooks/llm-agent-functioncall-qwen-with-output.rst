Create Function-calling Agent using OpenVINO and Qwen-Agent
===========================================================

LLM are limited to the knowledge on which they have been trained and the
additional knowledge provided as context, as a result, if a useful piece
of information is missing the provided knowledge, the model cannot “go
around” and try to find it in other sources. This is the reason why we
need to introduce the concept of Agents.

The core idea of agents is to use a language model to choose a sequence
of actions to take. In agents, a language model is used as a reasoning
engine to determine which actions to take and in which order. Agents can
be seen as applications powered by LLMs and integrated with a set of
tools like search engines, databases, websites, and so on. Within an
agent, the LLM is the reasoning engine that, based on the user input, is
able to plan and execute a set of actions that are needed to fulfill the
request.

.. figure:: https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/22fa5396-8381-400f-a78f-97e25d57d807
   :alt: agent

   agent

`Qwen-Agent <https://github.com/QwenLM/Qwen-Agent>`__ is a framework for
developing LLM applications based on the instruction following, tool
usage, planning, and memory capabilities of Qwen. It also comes with
example applications such as Browser Assistant, Code Interpreter, and
Custom Assistant.

This notebook explores how to create a Function calling Agent step by
step using OpenVINO and Qwen-Agent.


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Create a Function calling
   agent <#create-a-function-calling-agent>`__

   -  `Create functions <#create-functions>`__
   -  `Download model <#download-model>`__
   -  `Select inference device for
      LLM <#select-inference-device-for-llm>`__
   -  `Create LLM for Qwen-Agent <#create-llm-for-qwen-agent>`__
   -  `Create Function-calling
      pipeline <#create-function-calling-pipeline>`__

-  `Interactive Demo <#interactive-demo>`__

   -  `Create tools <#create-tools>`__
   -  `Create AI agent demo with Qwen-Agent and Gradio
      UI <#create-ai-agent-demo-with-qwen-agent-and-gradio-ui>`__

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

    import os

    os.environ["GIT_CLONE_PROTECTION_ACTIVE"] = "false"

    %pip install -Uq pip
    %pip uninstall -q -y optimum optimum-intel
    %pip install --pre -Uq "openvino>=2024.2.0" openvino-tokenizers[transformers] --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.1" \
    "datasets" \
    "accelerate" \
    "qwen-agent==0.0.7" "transformers>=4.38.1" "gradio==4.21.0", "modelscope-studio>=0.4.0" "langchain>=0.2.3" "langchain-community>=0.2.4" "wikipedia"
    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu \
    "git+https://github.com/huggingface/optimum-intel.git" \
    "git+https://github.com/openvinotoolkit/nncf.git"

Create a Function calling agent
-------------------------------



Function calling allows a model to detect when one or more tools should
be called and respond with the inputs that should be passed to those
tools. In an API call, you can describe tools and have the model
intelligently choose to output a structured object like JSON containing
arguments to call these tools. The goal of tools APIs is to more
reliably return valid and useful tool calls than what can be done using
a generic text completion or chat API.

We can take advantage of this structured output, combined with the fact
that you can bind multiple tools to a tool calling chat model and allow
the model to choose which one to call, to create an agent that
repeatedly calls tools and receives results until a query is resolved.

Create a function
~~~~~~~~~~~~~~~~~



First, we need to create a example function/tool for getting the
information of current weather.

.. code:: ipython3

    import json


    def get_current_weather(location, unit="fahrenheit"):
        """Get the current weather in a given location"""
        if "tokyo" in location.lower():
            return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
        elif "san francisco" in location.lower():
            return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
        elif "paris" in location.lower():
            return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
        else:
            return json.dumps({"location": location, "temperature": "unknown"})

Wrap the function’s name and description into a json list, and it will
help LLM to find out which function should be called for current task.

.. code:: ipython3

    functions = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ]

Download model
~~~~~~~~~~~~~~



Large Language Models (LLMs) are a core component of Agent. In this
example, we will demonstrate how to create a OpenVINO LLM model in
Qwen-Agent framework. Since Qwen2 can support function calling during
text generation, we select ``Qwen/Qwen2-7B-Instruct`` as LLM in agent
pipeline.

-  **Qwen/Qwen2-7B-Instruct** - Qwen2 is the new series of Qwen large
   language models. Compared with the state-of-the-art open source
   language models, including the previous released Qwen1.5, Qwen2 has
   generally surpassed most open source models and demonstrated
   competitiveness against proprietary models across a series of
   benchmarks targeting for language understanding, language generation,
   multilingual capability, coding, mathematics, reasoning, etc. `Model
   Card <https://huggingface.co/Qwen/Qwen2-7B-Instruct>`__

To run LLM locally, we have to download the model in the first step. It
is possible to `export your
model <https://github.com/huggingface/optimum-intel?tab=readme-ov-file#export>`__
to the OpenVINO IR format with the CLI, and load the model from local
folder.

.. code:: ipython3

    from pathlib import Path

    model_id = "Qwen/Qwen2-7B-Instruct"
    model_path = "Qwen2-7B-Instruct-ov"

    if not Path(model_path).exists():
        !optimum-cli export openvino --model {model_id} --task text-generation-with-past --trust-remote-code --weight-format int4 --ratio 0.72 {model_path}

Select inference device for LLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    import openvino as ov
    import ipywidgets as widgets

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

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



Create LLM for Qwen-Agent
~~~~~~~~~~~~~~~~~~~~~~~~~



OpenVINO has been integrated into the ``Qwen-Agent`` framework. You can
use following method to create a OpenVINO based LLM for a ``Qwen-Agent``
pipeline.

.. code:: ipython3

    from qwen_agent.llm import get_chat_model

    import openvino.properties as props
    import openvino.properties.hint as hints
    import openvino.properties.streams as streams


    ov_config = {hints.performance_mode(): hints.PerformanceMode.LATENCY, streams.num(): "1", props.cache_dir(): ""}
    llm_cfg = {
        "ov_model_dir": model_path,
        "model_type": "openvino",
        "device": device.value,
        "ov_config": ov_config,
        # (Optional) LLM hyperparameters for generation:
        "generate_cfg": {"top_p": 0.8},
    }
    llm = get_chat_model(llm_cfg)


.. parsed-literal::

    Compiling the model to CPU ...
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.


You can get additional inference speed improvement with `Dynamic
Quantization of activations and KV-cache quantization on
CPU <https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/llm-inference-hf.html#enabling-openvino-runtime-optimizations>`__.
These options can be enabled with ``ov_config`` as follows:

.. code:: ipython3

    ov_config = {
        "KV_CACHE_PRECISION": "u8",
        "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
        hints.performance_mode(): hints.PerformanceMode.LATENCY,
        streams.num(): "",
        props.cache_dir(): "",
    }

Create Function-calling pipeline
--------------------------------



After defining the functions and LLM, we can build the agent pipeline
with capability of function calling.

.. figure:: https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/3170ca30-23af-4a1a-a655-1d0d67df2ded
   :alt: functioncalling

   functioncalling

The workflow of Qwen2 function calling consists of several steps:

1. Role ``user`` sending the request.
2. Check if the model wanted to call a function, and call the function
   if needed
3. Get the observation from ``function``\ ’s results.
4. Consolidate the observation into final response of ``assistant``.

A typical multi-turn dialogue structure is as follows:

-  **Query**:
   ``{'role': 'user', 'content': 'create a picture of cute cat'},``

-  **Function calling**:
   ``{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt": "a cute cat"}'}},``

-  **Observation**:
   ``{'role': 'function', 'content': '{"image_url": "https://image.pollinations.ai/prompt/a%20cute%20cat"}', 'name': 'my_image_gen'}``

-  **Final Response**:
   ``{'role': 'assistant', 'content': "Here is the image of a cute cat based on your description:\n\n![](https://image.pollinations.ai/prompt/a%20cute%20cat)."}``

.. code:: ipython3

    print("# User question:")
    messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
    print(messages)

    print("# Assistant Response 1:")
    responses = []

    # Step 1: Role `user` sending the request
    responses = llm.chat(
        messages=messages,
        functions=functions,
        stream=False,
    )
    print(responses)

    messages.extend(responses)

    # Step 2: check if the model wanted to call a function, and call the function if needed
    last_response = messages[-1]
    if last_response.get("function_call", None):
        available_functions = {
            "get_current_weather": get_current_weather,
        }  # only one function in this example, but you can have multiple
        function_name = last_response["function_call"]["name"]
        function_to_call = available_functions[function_name]
        function_args = json.loads(last_response["function_call"]["arguments"])
        function_response = function_to_call(
            location=function_args.get("location"),
        )
        print("# Function Response:")
        print(function_response)

        # Step 3: Get the observation from `function`'s results
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )

        print("# Assistant Response 2:")
        # Step 4: Consolidate the observation from function into final response
        responses = llm.chat(
            messages=messages,
            functions=functions,
            stream=False,
        )
        print(responses)


.. parsed-literal::

    # User question:
    [{'role': 'user', 'content': "What's the weather like in San Francisco?"}]
    # Assistant Response 1:
    [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco, CA"}'}}]
    # Function Response:
    {"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}
    # Assistant Response 2:
    [{'role': 'assistant', 'content': 'The current weather in San Francisco is 72 degrees Fahrenheit.'}]


Interactive Demo
----------------



Let’s create a interactive agent using
`Gradio <https://www.gradio.app/>`__.

Create tools
~~~~~~~~~~~~



Qwen-Agent provides a mechanism for `registering
tools <https://github.com/QwenLM/Qwen-Agent/blob/main/docs/tool.md>`__.
For example, to register your own image generation tool:

-  Specify the tool’s name, description, and parameters. Note that the
   string passed to ``@register_tool('my_image_gen')`` is automatically
   added as the ``.name`` attribute of the class and will serve as the
   unique identifier for the tool.
-  Implement the ``call(...)`` function.

In this notebook, we will create 3 tools as examples:

- **image_generation**: AI painting (image generation) service, input text
 description, and return the image URL drawn based on text information.
- **get_current_weather**: Get the current weather in a given city name.
- **wikipedia**: A wrapper around Wikipedia. Useful for when you need to
  answer general questions about people, places, companies, facts,
  historical events, or other subjects.

.. code:: ipython3

    import urllib.parse
    import json5
    import requests
    from qwen_agent.tools.base import BaseTool, register_tool


    @register_tool("image_generation")
    class ImageGeneration(BaseTool):
        description = "AI painting (image generation) service, input text description, and return the image URL drawn based on text information."
        parameters = [{"name": "prompt", "type": "string", "description": "Detailed description of the desired image content, in English", "required": True}]

        def call(self, params: str, **kwargs) -> str:
            prompt = json5.loads(params)["prompt"]
            prompt = urllib.parse.quote(prompt)
            return json5.dumps({"image_url": f"https://image.pollinations.ai/prompt/{prompt}"}, ensure_ascii=False)


    @register_tool("get_current_weather")
    class GetCurrentWeather(BaseTool):
        description = "Get the current weather in a given city name."
        parameters = [{"name": "city_name", "type": "string", "description": "The city and state, e.g. San Francisco, CA", "required": True}]

        def call(self, params: str, **kwargs) -> str:
            # `params` are the arguments generated by the LLM agent.
            city_name = json5.loads(params)["city_name"]
            key_selection = {
                "current_condition": [
                    "temp_C",
                    "FeelsLikeC",
                    "humidity",
                    "weatherDesc",
                    "observation_time",
                ],
            }
            resp = requests.get(f"https://wttr.in/{city_name}?format=j1")
            resp.raise_for_status()
            resp = resp.json()
            ret = {k: {_v: resp[k][0][_v] for _v in v} for k, v in key_selection.items()}
            return str(ret)


    @register_tool("wikipedia")
    class Wikipedia(BaseTool):
        description = "A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects."
        parameters = [{"name": "query", "type": "string", "description": "Query to look up on wikipedia", "required": True}]

        def call(self, params: str, **kwargs) -> str:
            # `params` are the arguments generated by the LLM agent.
            from langchain.tools import WikipediaQueryRun
            from langchain_community.utilities import WikipediaAPIWrapper

            query = json5.loads(params)["query"]
            wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000))
            resutlt = wikipedia.run(query)
            return str(resutlt)

.. code:: ipython3

    tools = ["image_generation", "get_current_weather", "wikipedia"]

Create AI agent demo with Qwen-Agent and Gradio UI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



The Agent class serves as a higher-level interface for Qwen-Agent, where
an Agent object integrates the interfaces for tool calls and LLM (Large
Language Model). The Agent receives a list of messages as input and
produces a generator that yields a list of messages, effectively
providing a stream of output messages.

Qwen-Agent offers a generic Agent class: the ``Assistant`` class, which,
when directly instantiated, can handle the majority of Single-Agent
tasks. Features:

-  It supports role-playing.
-  It provides automatic planning and tool calls abilities.
-  RAG (Retrieval-Augmented Generation): It accepts documents input, and
   can use an integrated RAG strategy to parse the documents.

.. code:: ipython3

    from qwen_agent.agents import Assistant

    bot = Assistant(llm=llm_cfg, function_list=tools, name="OpenVINO Agent")


.. parsed-literal::

    Compiling the model to CPU ...
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.


.. code:: ipython3

    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/llm-agent-functioncall/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)

    from gradio_helper import make_demo

    demo = make_demo(bot=bot)

    try:
        demo.run()
    except Exception:
        demo.run(share=True)
    # If you are launching remotely, specify server_name and server_port
    # EXAMPLE: `demo.launch(server_name='your server name', server_port='server port in int')`
    # To learn more please refer to the Gradio docs: https://gradio.app/docs/

.. code:: ipython3

    # please uncomment and run this cell for stopping gradio interface
    # demo.close()
