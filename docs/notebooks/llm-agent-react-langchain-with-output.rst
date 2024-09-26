Create ReAct Agent using OpenVINO and LangChain
===============================================

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

`LangChain <https://python.langchain.com/docs/get_started/introduction>`__
is a framework for developing applications powered by language models.
LangChain comes with a number of built-in agents that are optimized for
different use cases.

This notebook explores how to create an AI Agent step by step using
OpenVINO and LangChain.


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Create tools <#create-tools>`__
-  `Create prompt template <#create-prompt-template>`__
-  `Create LLM <#create-llm>`__

   -  `Download model <#select-model>`__
   -  `Select inference device for
      LLM <#select-inference-device-for-llm>`__

-  `Create agent <#create-agent>`__
-  `Run the agent <#run-agent>`__
-  `Interactive Demo <#interactive-demo>`__

   -  `Use built-in tool <#use-built-in-tool>`__
   -  `Create customized tools <#create-customized-tools>`__
   -  `Create AI agent demo with Gradio
      UI <#create-ai-agent-demo-with-gradio-ui>`__

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
    "gradio>=4.19"
    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu "transformers>=4.38.1" "langchain>=0.2.3" "langchain-community>=0.2.4" "Wikipedia"
    %pip install -q "git+https://github.com/huggingface/optimum-intel.git" \
    "git+https://github.com/openvinotoolkit/nncf.git"

Create a tools
--------------



First, we need to create some tools to call. In this example, we will
create 3 custom functions to do basic calculation. For `more
information <https://python.langchain.com/docs/modules/tools/>`__ on
creating custom tools.

.. code:: ipython3

    from langchain_core.tools import tool
    
    
    @tool
    def multiply(first_int: int, second_int: int) -> int:
        """Multiply two integers together."""
        return first_int * second_int
    
    
    @tool
    def add(first_int: int, second_int: int) -> int:
        "Add two integers."
        return first_int + second_int
    
    
    @tool
    def exponentiate(base: int, exponent: int) -> int:
        "Exponentiate the base to the exponent power."
        return base**exponent

.. code:: ipython3

    print(f"name of `multiply` tool: {multiply.name}")
    print(f"description of `multiply` tool: {multiply.description}")


.. parsed-literal::

    name of `multiply` tool: multiply
    description of `multiply` tool: Multiply two integers together.


Tools are interfaces that an agent, chain, or LLM can use to interact
with the world. They combine a few things:

1. The name of the tool
2. A description of what the tool is
3. JSON schema of what the inputs to the tool are
4. The function to call
5. Whether the result of a tool should be returned directly to the user

Now that we have created all of them, and we can create a list of tools
that we will use downstream.

.. code:: ipython3

    tools = [multiply, add, exponentiate]

Create prompt template
----------------------



A prompt for a language model is a set of instructions or input provided
by a user to guide the model’s response, helping it understand the
context and generate relevant and coherent language-based output, such
as answering questions, completing sentences, or engaging in a
conversation.

Different agents have different prompting styles for reasoning. In this
example, we will use `ReAct agent <https://react-lm.github.io/>`__ with
its typical prompt template. For a full list of built-in agents see
`agent
types <https://python.langchain.com/docs/modules/agents/agent_types/>`__.

.. figure:: https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/a83bdf7f-bb9d-4b1f-9a0a-3fe4a76ba1ae
   :alt: react

   react

A ReAct prompt consists of few-shot task-solving trajectories, with
human-written text reasoning traces and actions, as well as environment
observations in response to actions. ReAct prompting is intuitive and
flexible to design, and achieves state-of-the-art few-shot performances
across a variety of tasks, from question answering to online shopping!

In an prompt template for agent, ``input`` is user’s query and
``agent_scratchpad`` should be a sequence of messages that contains the
previous agent tool invocations and the corresponding tool outputs.

.. code:: ipython3

    PREFIX = """[INST]Respond to the human as helpfully and accurately as possible. You have access to the following tools:"""
    
    FORMAT_INSTRUCTIONS = """Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
    
    Valid "action" values: "Final Answer" or {tool_names}
    
    Provide only ONE action per $JSON_BLOB, as shown:
    
    ```
    {{{{
      "action": $TOOL_NAME,
      "action_input": $INPUT
    }}}}
    ```
    
    Follow this format:
    
    Question: input question to answer
    Thought: consider previous and subsequent steps
    Action:
    ```
    $JSON_BLOB
    ```
    Observation: action result
    ... (repeat Thought/Action/Observation N times)
    Thought: I know what to respond
    Action:
    ```
    {{{{
      "action": "Final Answer",
      "action_input": "Final response to human"
    }}}}
    ```[/INST]"""
    
    SUFFIX = """Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.
    Thought:[INST]"""
    
    HUMAN_MESSAGE_TEMPLATE = "{input}\n\n{agent_scratchpad}"

Create LLM
----------



Large Language Models (LLMs) are a core component of LangChain.
LangChain does not serve its own LLMs, but rather provides a standard
interface for interacting with many different LLMs. In this example, we
select ``Mistral-7B-Instruct-v0.3`` as LLM in agent pipeline.

-  **Mistral-7B-Instruct-v0.3** - The Mistral-7B-Instruct-v0.3 Large
   Language Model (LLM) is an instruct fine-tuned version of the
   Mistral-7B-v0.3. You can find more details about model in the `model
   card <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3>`__,
   `paper <https://arxiv.org/abs/2310.06825>`__ and `release blog
   post <https://mistral.ai/news/announcing-mistral-7b/>`__.
   >\ **Note**: run model with demo, you will need to accept license
   agreement. >You must be a registered user in Hugging Face Hub.
   Please visit `HuggingFace model
   card <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3>`__,
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

Download model
~~~~~~~~~~~~~~



To run LLM locally, we have to download the model in the first step. It
is possible to `export your
model <https://github.com/huggingface/optimum-intel?tab=readme-ov-file#export>`__
to the OpenVINO IR format with the CLI, and load the model from local
folder.

.. code:: ipython3

    from pathlib import Path
    
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    model_path = "Mistral-7B-Instruct-v0.3-ov-int4"
    
    if not Path(model_path).exists():
        !optimum-cli export openvino --model {model_id} --task text-generation-with-past --trust-remote-code --weight-format int4 {model_path}

Select inference device for LLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)
    
    from notebook_utils import device_widget
    
    device = device_widget("CPU", exclude=["NPU"])




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'GPU', 'AUTO'), value='CPU')



OpenVINO models can be run locally through the ``HuggingFacePipeline``
class in LangChain. To deploy a model with OpenVINO, you can specify the
``backend="openvino"`` parameter to trigger OpenVINO as backend
inference framework. For `more
information <https://python.langchain.com/docs/integrations/llms/openvino/>`__.

.. code:: ipython3

    from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
    from transformers.generation.stopping_criteria import StoppingCriteriaList, StoppingCriteria
    
    import openvino.properties as props
    import openvino.properties.hint as hints
    import openvino.properties.streams as streams
    
    
    class StopSequenceCriteria(StoppingCriteria):
        """
        This class can be used to stop generation whenever a sequence of tokens is encountered.
    
        Args:
            stop_sequences (`str` or `List[str]`):
                The sequence (or list of sequences) on which to stop execution.
            tokenizer:
                The tokenizer used to decode the model outputs.
        """
    
        def __init__(self, stop_sequences, tokenizer):
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]
            self.stop_sequences = stop_sequences
            self.tokenizer = tokenizer
    
        def __call__(self, input_ids, scores, **kwargs) -> bool:
            decoded_output = self.tokenizer.decode(input_ids.tolist()[0])
            return any(decoded_output.endswith(stop_sequence) for stop_sequence in self.stop_sequences)
    
    
    ov_config = {hints.performance_mode(): hints.PerformanceMode.LATENCY, streams.num(): "1", props.cache_dir(): ""}
    stop_tokens = ["Observation:"]
    
    ov_llm = HuggingFacePipeline.from_model_id(
        model_id=model_path,
        task="text-generation",
        backend="openvino",
        model_kwargs={
            "device": device.value,
            "ov_config": ov_config,
            "trust_remote_code": True,
        },
        pipeline_kwargs={"max_new_tokens": 2048},
    )
    ov_llm = ov_llm.bind(skip_prompt=True, stop=["Observation:"])
    
    tokenizer = ov_llm.pipeline.tokenizer
    ov_llm.pipeline._forward_params["stopping_criteria"] = StoppingCriteriaList([StopSequenceCriteria(stop_tokens, tokenizer)])


.. parsed-literal::

    2024-06-07 23:17:16.804739: I tensorflow/core/util/port.cc:111] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-06-07 23:17:16.807973: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-06-07 23:17:16.850235: E tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:9342] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-06-07 23:17:16.850258: E tensorflow/compiler/xla/stream_executor/cuda/cuda_fft.cc:609] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-06-07 23:17:16.850290: E tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:1518] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-06-07 23:17:16.859334: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-06-07 23:17:17.692415: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers
    The argument `trust_remote_code` is to be used along with export=True. It will be ignored.
    Compiling the model to GPU ...


You can get additional inference speed improvement with `Dynamic
Quantization of activations and KV-cache quantization on
CPU <https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/llm-inference-hf.html#enabling-openvino-runtime-optimizations>`__.
These options can be enabled with ``ov_config`` as follows:

.. code:: ipython3

    ov_config = {
        "KV_CACHE_PRECISION": "u8",
        "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
        hints.performance_mode(): hints.PerformanceMode.LATENCY,
        streams.num(): "1",
        props.cache_dir(): "",
    }

Create agent
------------



Now that we have defined the tools, prompt template and LLM, we can
create the agent_executor.

The agent executor is the runtime for an agent. This is what actually
calls the agent, executes the actions it chooses, passes the action
outputs back to the agent, and repeats.

.. code:: ipython3

    from langchain.agents import AgentExecutor, StructuredChatAgent
    
    agent = StructuredChatAgent.from_llm_and_tools(
        ov_llm,
        tools,
        prefix=PREFIX,
        suffix=SUFFIX,
        human_message_template=HUMAN_MESSAGE_TEMPLATE,
        format_instructions=FORMAT_INSTRUCTIONS,
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

Run the agent
-------------



We can now run the agent with a math query. Before getting the final
answer, a agent executor will also produce intermediate steps of
reasoning and actions. The format of these messages will follow your
prompt template.

.. code:: ipython3

    agent_executor.invoke({"input": "Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result"})


.. parsed-literal::

    
    
    > Entering new AgentExecutor chain...
    Thought: I can use the exponentiate and add tools to solve the first part, and then use the multiply tool for the second part, and finally the exponentiate tool again to square the result.
    
    Action:
    ```
    {
      "action": "exponentiate",
      "action_input": {"base": 3, "exponent": 5}
    }
    ```
    Observation:
    Observation: 243
    Thought: Now I need to add twelve and three
    
    Action:
    ```
    {
      "action": "add",
      "action_input": {"first_int": 12, "second_int": 3}
    }
    ```
    Observation:
    Observation: 15
    Thought: Now I need to multiply the result by 243
    
    Action:
    ```
    {
      "action": "multiply",
      "action_input": {"first_int": 243, "second_int": 15}
    }
    ```
    Observation:
    Observation: 3645
    Thought: Finally, I need to square the result
    
    Action:
    ```
    {
      "action": "exponentiate",
      "action_input": {"base": 3645, "exponent": 2}
    }
    ```
    Observation:
    Observation: 13286025
    Thought: I know what to respond
    
    Action:
    ```
    {
      "action": "Final Answer",
      "action_input": "The final answer is 13286025"
    }
    ```
    
    > Finished chain.




.. parsed-literal::

    {'input': 'Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result',
     'output': 'The final answer is 13286025'}



Interactive Demo
----------------



Let’s create a interactive agent using
`Gradio <https://www.gradio.app/>`__.

Use built-in tools
~~~~~~~~~~~~~~~~~~



LangChain has provided a list of all `built-in
tools <https://python.langchain.com/docs/integrations/tools/>`__. In
this example, we will use ``Wikipedia`` python package to query key
words generated by agent.

.. code:: ipython3

    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain_core.callbacks import CallbackManagerForToolRun
    from typing import Optional
    
    from pydantic import BaseModel, Field
    
    
    class WikipediaQueryRunWrapper(WikipediaQueryRun):
        def _run(
            self,
            text: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
        ) -> str:
            """Use the Wikipedia tool."""
            return self.api_wrapper.run(text)
    
    
    api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
    
    
    class WikiInputs(BaseModel):
        """inputs to the wikipedia tool."""
    
        text: str = Field(description="query to look up on wikipedia.")
    
    
    wikipedia = WikipediaQueryRunWrapper(
        description="A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects. Input should be a search query.",
        args_schema=WikiInputs,
        api_wrapper=api_wrapper,
    )

.. code:: ipython3

    wikipedia.invoke({"text": "OpenVINO"})




.. parsed-literal::

    'Page: OpenVINO\nSummary: OpenVINO is an open-source software toolkit for optimizing and deploying deep learning models. It enables programmers to develop scalable and efficient AI solutions with relatively few lines of code. It supports several popular model formats and categories, such as large language models, computer vision, and generative AI.\nActively developed by Intel, it prioritizes high-performance inference on Intel hardware but also supports ARM/ARM64 processors and encourages contributors to add new devices to the portfolio.\nBased in C++, it offers the following APIs: C/C++, Python, and Node.js (an early preview).\nOpenVINO is cross-platform and free for use under Apache License 2.0.\n\nPage: Stable Diffusion\nSummary: Stable Diffusion is a deep learning, text-to-image model released in 2022 based on diffusion techniques. It is considered to be a part of the ongoing artificial intelligence boom.\nIt is primarily used to generate detailed images conditioned on text descriptions, t'



Create customized tools
~~~~~~~~~~~~~~~~~~~~~~~



In this examples, we will create 2 customized tools for
``image generation`` and ``weather qurey``.

.. code:: ipython3

    import urllib.parse
    import json5
    
    
    @tool
    def painting(prompt: str) -> str:
        """
        AI painting (image generation) service, input text description, and return the image URL drawn based on text information.
        """
        prompt = urllib.parse.quote(prompt)
        return json5.dumps({"image_url": f"https://image.pollinations.ai/prompt/{prompt}"}, ensure_ascii=False)
    
    
    painting.invoke({"prompt": "a cat"})




.. parsed-literal::

    '{image_url: "https://image.pollinations.ai/prompt/a%20cat"}'



.. code:: ipython3

    @tool
    def weather(
        city_name: str,
    ) -> str:
        """
        Get the current weather for `city_name`
        """
    
        if not isinstance(city_name, str):
            raise TypeError("City name must be a string")
    
        key_selection = {
            "current_condition": [
                "temp_C",
                "FeelsLikeC",
                "humidity",
                "weatherDesc",
                "observation_time",
            ],
        }
        import requests
    
        resp = requests.get(f"https://wttr.in/{city_name}?format=j1")
        resp.raise_for_status()
        resp = resp.json()
        ret = {k: {_v: resp[k][0][_v] for _v in v} for k, v in key_selection.items()}
    
        return str(ret)
    
    
    weather.invoke({"city_name": "London"})




.. parsed-literal::

    "{'current_condition': {'temp_C': '9', 'FeelsLikeC': '8', 'humidity': '93', 'weatherDesc': [{'value': 'Sunny'}], 'observation_time': '04:39 AM'}}"



Create AI agent demo with Gradio UI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    tools = [wikipedia, painting, weather]
    
    agent = StructuredChatAgent.from_llm_and_tools(
        ov_llm,
        tools,
        prefix=PREFIX,
        suffix=SUFFIX,
        human_message_template=HUMAN_MESSAGE_TEMPLATE,
        format_instructions=FORMAT_INSTRUCTIONS,
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

.. code:: ipython3

    def partial_text_processor(partial_text, new_text):
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
    
    
    def run_chatbot(history):
        """
        callback function for running chatbot on submit button click
    
        Params:
          history: conversation history
    
        """
        partial_text = ""
    
        for new_text in agent_executor.stream(
            {"input": history[-1][0]},
        ):
            if "output" in new_text.keys():
                partial_text = partial_text_processor(partial_text, new_text["output"])
                history[-1][1] = partial_text
                yield history
    
    
    def request_cancel():
        ov_llm.pipeline.model.request.cancel()

.. code:: ipython3

    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/llm-agent-react/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)
    
    from gradio_helper import make_demo
    
    demo = make_demo(run_fn=run_chatbot, stop_fn=request_cancel)
    
    try:
        demo.launch()
    except Exception:
        demo.launch(share=True)
    # If you are launching remotely, specify server_name and server_port
    # EXAMPLE: `demo.launch(server_name='your server name', server_port='server port in int')`
    # To learn more please refer to the Gradio docs: https://gradio.app/docs/


.. parsed-literal::

    
    
    > Entering new AgentExecutor chain...
    Thought: I need to use the weather tool to get the current weather in London, then use the painting tool to generate a picture of Big Ben based on the weather information.
    
    Action:
    ```
    {
      "action": "weather",
      "action_input": "London"
    }
    ```
    
    Observation:
    Observation: {'current_condition': {'temp_C': '9', 'FeelsLikeC': '8', 'humidity': '93', 'weatherDesc': [{'value': 'Sunny'}], 'observation_time': '04:39 AM'}}
    Thought: I have the current weather in London. Now I can use the painting tool to generate a picture of Big Ben based on the weather information.
    
    Action:
    ```
    {
      "action": "painting",
      "action_input": "Big Ben, sunny day"
    }
    ```
    
    Observation:
    Observation: {image_url: "https://image.pollinations.ai/prompt/Big%20Ben%2C%20sunny%20day"}
    Thought: I have the image URL of Big Ben on a sunny day. Now I can respond to the human with the image URL.
    
    Action:
    ```
    {
      "action": "Final Answer",
      "action_input": "Here is the image of Big Ben on a sunny day: https://image.pollinations.ai/prompt/Big%20Ben%2C%20sunny%20day"
    }
    ```
    Observation:
    
    > Finished chain.


.. code:: ipython3

    # please uncomment and run this cell for stopping gradio interface
    # demo.close()
