Create LLM Agent using OpenVINO
===============================

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

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prerequisites <#Prerequisites>`__
-  `Create tools <#Create-tools>`__
-  `Create prompt template <#Create-prompt-template>`__
-  `Create LLM <#Create-LLM>`__

   -  `Download model <#Select-model>`__
   -  `Select inference device for
      LLM <#Select-inference-device-for-LLM>`__

-  `Create agent <#Create-agent>`__
-  `Run the agent <#Run-agent>`__
-  `Interactive Demo <#Interactive-Demo>`__

   -  `Use built-in tool <#Use-built-in-tool>`__

Prerequisites
-------------

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    import os
    
    os.environ["GIT_CLONE_PROTECTION_ACTIVE"] = "false"
    
    %pip install -Uq pip
    %pip uninstall -q -y optimum optimum-intel
    %pip install --pre -Uq openvino openvino-tokenizers[transformers] --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu\
    "git+https://github.com/huggingface/optimum-intel.git"\
    "git+https://github.com/openvinotoolkit/nncf.git"\
    "torch>=2.1"\
    "datasets"\
    "accelerate"\
    "gradio"\
    "transformers>=4.38.1" "langchain>=0.2.0" "langchain-community>=0.2.0" "wikipedia"

Create a tools
--------------

`back to top ⬆️ <#Table-of-contents:>`__

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

Tools are interfaces that an agent, chain, or LLM can use to interact
with the world. They combine a few things:

1. The name of the tool
2. A description of what the tool is
3. JSON schema of what the inputs to the tool are
4. The function to call
5. Whether the result of a tool should be returned directly to the user

.. code:: ipython3

    print(f"name of `multiply` tool: {multiply.name}")
    print(f"description of `multiply` tool: {multiply.description}")


.. parsed-literal::

    name of `multiply` tool: multiply
    description of `multiply` tool: multiply(first_int: int, second_int: int) -> int - Multiply two integers together.


Now that we have created all of them, and we can create a list of tools
that we will use downstream.

.. code:: ipython3

    tools = [multiply, add, exponentiate]

Create prompt template
----------------------

`back to top ⬆️ <#Table-of-contents:>`__

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

In an prompt template for agent, ``agent_scratchpad`` should be a
sequence of messages that contains the previous agent tool invocations
and the corresponding tool outputs.

.. code:: ipython3

    from langchain.prompts import PromptTemplate
    
    prompt = PromptTemplate.from_template(
        """Answer the following questions as best you can. You have access to the following tools:
    
        {tools}
    
        Use the following format:
    
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action\nObservation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
    
        Begin!
    
        Question: {input}
        Thought:{agent_scratchpad}"""
    )

Create LLM
----------

`back to top ⬆️ <#Table-of-contents:>`__

Large Language Models (LLMs) are a core component of LangChain.
LangChain does not serve its own LLMs, but rather provides a standard
interface for interacting with many different LLMs. In this example, we
select ``neural-chat-7b-v3-1`` as LLM in agent pipeline.

**neural-chat-7b-v3-1** - Mistral-7b model fine-tuned using Intel Gaudi.
The model fine-tuned on the open source dataset
`Open-Orca/SlimOrca <https://huggingface.co/datasets/Open-Orca/SlimOrca>`__
and aligned with `Direct Preference Optimization (DPO)
algorithm <https://arxiv.org/abs/2305.18290>`__. More details can be
found in `model
card <https://huggingface.co/Intel/neural-chat-7b-v3-1>`__ and `blog
post <https://medium.com/@NeuralCompressor/the-practice-of-supervised-finetuning-and-direct-preference-optimization-on-habana-gaudi2-a1197d8a3cd3>`__.

Download model
~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

To run LLM locally, we have to download the model in the first step. It
is possible to `export your
model <https://github.com/huggingface/optimum-intel?tab=readme-ov-file#export>`__
to the OpenVINO IR format with the CLI, and load the model from local
folder.

.. code:: ipython3

    from pathlib import Path
    
    model_id = "Intel/neural-chat-7b-v3-1"
    model_path = "neural-chat-7b-v3-1-ov-int4"
    
    if not Path(model_path).exists():
        !optimum-cli export openvino --model {model_id} --weight-format int4 {model_path}

Select inference device for LLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

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

    Dropdown(description='Device:', options=('CPU', 'GPU.0', 'GPU.1', 'AUTO'), value='CPU')



OpenVINO models can be run locally through the ``HuggingFacePipeline``
class in LangChain. To deploy a model with OpenVINO, you can specify the
``backend="openvino"`` parameter to trigger OpenVINO as backend
inference framework. For `more
information <https://python.langchain.com/docs/integrations/llms/openvino/>`__.

.. code:: ipython3

    from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
    
    ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}
    
    ov_llm = HuggingFacePipeline.from_model_id(
        model_id=model_path,
        task="text-generation",
        backend="openvino",
        model_kwargs={"device": device.value, "ov_config": ov_config},
        pipeline_kwargs={"max_new_tokens": 1024},
    )


.. parsed-literal::

    2024-05-01 12:57:42.013703: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-05-01 12:57:42.015389: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-05-01 12:57:42.049792: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-05-01 12:57:42.050591: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-05-01 12:57:42.819557: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/bitsandbytes/cextension.py:34: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
      warn("The installed version of bitsandbytes was compiled without GPU support. "


.. parsed-literal::

    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so: undefined symbol: cadam32bit_grad_fp32
    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
    WARNING[XFORMERS]: xFormers can't load C++/CUDA extensions. xFormers was built for:
        PyTorch 2.0.1+cu118 with CUDA 1108 (you have 2.1.2+cpu)
        Python  3.8.18 (you have 3.8.10)
      Please reinstall xformers (see https://github.com/facebookresearch/xformers#installing-xformers)
      Memory-efficient attention, SwiGLU, sparse and more won't be available.
      Set XFORMERS_MORE_DETAILS=1 for more details
    Compiling the model to CPU ...


You can get additional inference speed improvement with [Dynamic
Quantization of activations and KV-cache quantization] on
CPU(https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/llm-inference-hf.html#enabling-openvino-runtime-optimizations).
These options can be enabled with ``ov_config`` as follows:

.. code:: ipython3

    ov_config = {
        "KV_CACHE_PRECISION": "u8",
        "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
        "PERFORMANCE_HINT": "LATENCY",
        "NUM_STREAMS": "1",
        "CACHE_DIR": "",
    }

Create agent
------------

`back to top ⬆️ <#Table-of-contents:>`__

Now that we have defined the tools, prompt template and LLM, we can
create the agent_executor.

The agent executor is the runtime for an agent. This is what actually
calls the agent, executes the actions it chooses, passes the action
outputs back to the agent, and repeats.

.. code:: ipython3

    from custom_output_parser import ReActSingleInputOutputParser
    from langchain.agents import AgentExecutor, create_react_agent
    
    output_parser = ReActSingleInputOutputParser()
    
    agent = create_react_agent(ov_llm, tools, prompt, output_parser=output_parser)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

Run the agent
-------------

`back to top ⬆️ <#Table-of-contents:>`__

We can now run the agent with a math query. Before getting the final
answer, a agent executor will also produce intermediate steps of
reasoning and actions. The format of these messages will follow your
prompt template.

.. code:: ipython3

    agent_executor.invoke({"input": "Take 3 to the fifth power and multiply that by the sum of twelve and three"})


.. parsed-literal::

    
    
    > Entering new AgentExecutor chain...
    Answer the following questions as best you can. You have access to the following tools:
    
        multiply: multiply(first_int: int, second_int: int) -> int - Multiply two integers together.
    add: add(first_int: int, second_int: int) -> int - Add two integers.
    exponentiate: exponentiate(base: int, exponent: int) -> int - Exponentiate the base to the exponent power.
    
        Use the following format:
    
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [multiply, add, exponentiate]
        Action Input: the input to the action
    Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
    
        Begin!
    
        Question: Take 3 to the fifth power and multiply that by the sum of twelve and three
        Thought: We need to exponentiate 3 to the power of 5, then multiply the result by the sum of 12 and 3
        Action: exponentiate
        Action Input: base: 3, exponent: 5
        Observation: 243
        Action: add
        Action Input: first_int: 12, second_int: 3
        Observation: 15
        Action: multiply
        Action Input: first_int: 243, second_int: 15
        Observation: 3645
        Thought: I now know the final answer
        Final Answer: 3645
    
    > Finished chain.




.. parsed-literal::

    {'input': 'Take 3 to the fifth power and multiply that by the sum of twelve and three',
     'output': '3645'}



Interactive Demo
----------------

`back to top ⬆️ <#Table-of-contents:>`__

Let’s create a interactive agent using
`Gradio <https://www.gradio.app/>`__.

Use built-in tool
~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

LangChain has provided a list of all `built-in
tools <https://python.langchain.com/docs/integrations/tools/>`__. In
this example, we will use ``Wikipedia`` python package to query key
words generated by agent.

.. code:: ipython3

    from langchain.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
    
    
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    print(f"description of `wikipedia` tool: {wikipedia.description}")
    
    tools = [wikipedia]
    
    agent = create_react_agent(ov_llm, tools, prompt, output_parser=output_parser)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


.. parsed-literal::

    description of `wikipedia` tool: A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects. Input should be a search query.


.. code:: ipython3

    from threading import Thread
    import gradio as gr
    from transformers import TextIteratorStreamer
    
    examples = [
        ["What is OpenVINO ?"],
        ["Who is 44th presedent of USA ?"],
        ["what is Obama's first name and who is him ?"],
        ["How many people live in Canada ?"],
        ["How tall is the Eiffel Tower ?"],
    ]
    
    
    def partial_text_processor(partial_text, new_text):
        """
        helper for updating partially generated answer, used by default
    
        Params:
          partial_text: text buffer for storing previosly generated text
          new_text: text update for the current step
        Returns:
          updated text string
    
        """
        new_text = new_text.replace("[INST]", "").replace("[/INST]", "")
        partial_text += new_text
        return partial_text
    
    
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
    
    
    def bot(history, temperature, top_p, top_k, repetition_penalty, return_intermediate_steps):
        """
        callback function for running chatbot on submit button click
    
        Params:
          history: conversation history
          temperature:  parameter for control the level of creativity in AI-generated text.
                        By adjusting the `temperature`, you can influence the AI model's probability distribution, making the text more focused or diverse.
          top_p: parameter for control the range of tokens considered by the AI model based on their cumulative probability.
          top_k: parameter for control the range of tokens considered by the AI model based on their cumulative probability, selecting number of tokens with highest probability.
          repetition_penalty: parameter for penalizing tokens based on how frequently they occur in the text.
          return_intermediate_steps: whether return intermediate_steps of agent.
    
        """
        streamer = TextIteratorStreamer(
            ov_llm.pipeline.tokenizer,
            timeout=60.0,
            skip_prompt=True,
            skip_special_tokens=True,
        )
    
        ov_llm.pipeline._forward_params = dict(
            max_new_tokens=512,
            temperature=temperature,
            do_sample=temperature > 0.0,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
        )
    
        t1 = Thread(target=agent_executor.invoke, args=({"input": history[-1][0]},))
        t1.start()
    
        # Initialize an empty string to store the generated text
        partial_text = ""
        final_answer = False
    
        for new_text in streamer:
            if "Answer" in new_text:
                final_answer = True
            if final_answer or return_intermediate_steps:
                partial_text = partial_text_processor(partial_text, new_text)
                history[-1][1] = partial_text
                yield history
    
    
    def request_cancel():
        ov_llm.pipeline.model.request.cancel()
    
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=".disclaimer {font-variant-caps: all-small-caps;}",
    ) as demo:
        gr.Markdown(f"""<h1><center>OpenVINO Agent for {wikipedia.name}</center></h1>""")
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
                    return_cot = gr.Checkbox(value=True, label="Return intermediate steps")
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
        gr.Examples(examples, inputs=msg, label="Click on any example and press the 'Submit' button")
    
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
                return_cot,
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
                return_cot,
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

    # please run this cell for stopping gradio interface
    demo.close()
