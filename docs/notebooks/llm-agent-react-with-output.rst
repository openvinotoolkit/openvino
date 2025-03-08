Create a native Agent with OpenVINO
===================================

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

This example will demonstrate how to create a native agent with
OpenVINO.


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Create LLM as agent <#create-llm-as-agent>`__

   -  `Download model <#select-model>`__
   -  `Select inference device for
      LLM <#select-inference-device-for-llm>`__
   -  `Instantiate LLM using Optimum
      Intel <#instantiate-llm-using-optimum-intel>`__
   -  `Create text generation method <#create-text-generation-method>`__

-  `Create prompt template <#create-prompt-template>`__
-  `Create parser <#create-parers>`__
-  `Create tools calling <#create-tool-calling>`__
-  `Run agent <#run-agent>`__
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
    import requests
    from pathlib import Path

    if not Path("notebook_utils.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
        )
        open("notebook_utils.py", "w").write(r.text)

    if not Path("pip_helper.py").exists():
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
        "transformers>=4.43.1",
        "gradio>=4.19",
    )
    pip_install(
        "-q",
        "git+https://github.com/huggingface/optimum-intel.git",
        "git+https://github.com/openvinotoolkit/nncf.git",
        "datasets",
        "accelerate",
        "huggingface-hub>=0.26.5",
    )
    pip_install("--pre", "-Uq", "openvino>=2024.4.0", "--extra-index-url", "https://storage.openvinotoolkit.org/simple/wheels/nightly")

    # Read more about telemetry collection at https://github.com/openvinotoolkit/openvino_notebooks?tab=readme-ov-file#-telemetry
    from notebook_utils import collect_telemetry

    collect_telemetry("llm-agent-react.ipynb")

Create LLM as agent
-------------------



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
``Qwen2.5`` as LLM in agent pipeline.

* **qwen2.5-3b-instruct/qwen2.5-7b-instruct/qwen2.5-14b-instruct** -
  Qwen2.5 is the latest series of Qwen large language models. Comparing
  with Qwen2, Qwen2.5 series brings significant improvements in coding,
  mathematics and general knowledge skills. Additionally, it brings
  long-context and multiple languages support including Chinese, English,
  French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean,
  Vietnamese, Thai, Arabic, and more. For more details, please refer to
  `model_card <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>`__,
  `blog <https://qwenlm.github.io/blog/qwen2.5/>`__,
  `GitHub <https://github.com/QwenLM/Qwen2.5>`__, and
  `Documentation <https://qwen.readthedocs.io/en/latest/>`__.

.. code:: ipython3

    import ipywidgets as widgets

    llm_model_ids = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen/qwen2.5-14b-instruct"]

    llm_model_id = widgets.Dropdown(
        options=llm_model_ids,
        value=llm_model_ids[0],
        description="Model:",
        disabled=False,
    )

    llm_model_id




.. parsed-literal::

    Dropdown(description='Model:', options=('Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/qwen2.5-…



.. code:: ipython3

    from pathlib import Path

    llm_model_path = llm_model_id.value.split("/")[-1]

    if not Path(llm_model_path).exists():
        !optimum-cli export openvino --model {llm_model_id.value} --task text-generation-with-past --trust-remote-code --weight-format int4 --group-size 128 --ratio 1.0 --sym {llm_model_path}

Select inference device for LLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    from notebook_utils import device_widget

    llm_device = device_widget("CPU", exclude=["NPU"])

    llm_device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'GPU', 'AUTO'), value='CPU')



Instantiate LLM using Optimum Intel
-----------------------------------



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
parameter ``export=True`` should be added (as we already converted model
before, we do not need to provide this parameter). We can save the
converted model for the next usage with the ``save_pretrained`` method.
Tokenizer class and pipelines API are compatible with Optimum models.

You can find more details about OpenVINO LLM inference using HuggingFace
Optimum API in `LLM inference
guide <https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html>`__.

.. code:: ipython3

    from optimum.intel.openvino import OVModelForCausalLM
    from transformers import AutoTokenizer, AutoConfig, TextStreamer
    from transformers.generation import (
        StoppingCriteriaList,
        StoppingCriteria,
    )
    import openvino.properties as props
    import openvino.properties.hint as hints
    import openvino.properties.streams as streams

    import json
    import json5
    import torch

    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)

    ov_config = {hints.performance_mode(): hints.PerformanceMode.LATENCY, streams.num(): "1", props.cache_dir(): ""}

    llm = OVModelForCausalLM.from_pretrained(
        llm_model_path,
        device=llm_device.value,
        ov_config=ov_config,
        config=AutoConfig.from_pretrained(llm_model_path, trust_remote_code=True),
        trust_remote_code=True,
    )

    llm.generation_config.top_k = 1
    llm.generation_config.max_length = 2000

Create text generation method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



In this example, we would like to stream the output text though
``TextStreamer``, and stop text generation before ``Observation``
received from tool calling..

.. code:: ipython3

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


    def text_completion(prompt: str, stop_words) -> str:
        im_end = "<|im_end|>"
        if im_end not in stop_words:
            stop_words = stop_words + [im_end]
        streamer = TextStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)

        stopping_criteria = StoppingCriteriaList([StopSequenceCriteria(stop_words, tokenizer)])
        input_ids = torch.tensor([tokenizer.encode(prompt)])
        generate_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            stopping_criteria=stopping_criteria,
        )
        output = llm.generate(**generate_kwargs)
        output = output.tolist()[0]
        output = tokenizer.decode(output, errors="ignore")
        assert output.startswith(prompt)
        output = output[len(prompt) :].replace("<|endoftext|>", "").replace(im_end, "")

        for stop_str in stop_words:
            idx = output.find(stop_str)
            if idx != -1:
                output = output[: idx + len(stop_str)]
        return output

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

.. figure:: https://github.com/user-attachments/assets/c26432c2-3cf1-4942-ae03-fd8e8ebb4509
   :alt: react

   react

A ReAct prompt consists of few-shot task-solving trajectories, with
human-written text reasoning traces and actions, as well as environment
observations in response to actions. ReAct prompting is intuitive and
flexible to design, and achieves state-of-the-art few-shot performances
across a variety of tasks, from question answering to online shopping!

In an prompt template for agent, ``query`` is user’s query and other
parameter should be a sequence of messages that contains the
``descriptions`` and ``parameters`` of agent tool.

.. code:: ipython3

    TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

    PROMPT_REACT = """Answer the following questions as best you can. You have access to the following APIs:

    {tools_text}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tools_name_text}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {query}"""

Meanwhile we have to create function for consolidate the tools
information and conversation history into the prompt template.

.. code:: ipython3

    def build_input_text(chat_history, list_of_tool_info) -> str:
        tools_text = []
        for tool_info in list_of_tool_info:
            tool = TOOL_DESC.format(
                name_for_model=tool_info["name_for_model"],
                name_for_human=tool_info["name_for_human"],
                description_for_model=tool_info["description_for_model"],
                parameters=json.dumps(tool_info["parameters"], ensure_ascii=False),
            )
            if tool_info.get("args_format", "json") == "json":
                tool += " Format the arguments as a JSON object."
            elif tool_info["args_format"] == "code":
                tool += " Enclose the code within triple backticks (`) at the beginning and end of the code."
            else:
                raise NotImplementedError
            tools_text.append(tool)
        tools_text = "\n\n".join(tools_text)

        tools_name_text = ", ".join([tool_info["name_for_model"] for tool_info in list_of_tool_info])

        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for i, (query, response) in enumerate(chat_history):
            if list_of_tool_info:
                if (len(chat_history) == 1) or (i == len(chat_history) - 2):
                    query = PROMPT_REACT.format(
                        tools_text=tools_text,
                        tools_name_text=tools_name_text,
                        query=query,
                    )
            if query:
                messages.append({"role": "user", "content": query})
            if response:
                messages.append({"role": "assistant", "content": response})

        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, return_tensors="pt")

        return prompt

Create parser
-------------



A Parser is used to convert raw output of LLM to the input arguments of
tools.

.. code:: ipython3

    def parse_latest_tool_call(text):
        tool_name, tool_args = "", ""
        i = text.rfind("\nAction:")
        j = text.rfind("\nAction Input:")
        k = text.rfind("\nObservation:")
        if 0 <= i < j:  # If the text has `Action` and `Action input`,
            if k < j:  # but does not contain `Observation`,
                # then it is likely that `Observation` is ommited by the LLM,
                # because the output text may have discarded the stop word.
                text = text.rstrip() + "\nObservation:"  # Add it back.
            k = text.rfind("\nObservation:")
            tool_name = text[i + len("\nAction:") : j].strip()
            tool_args = text[j + len("\nAction Input:") : k].strip()
            text = text[:k]
        return tool_name, tool_args, text

Create tools calling
--------------------



In this examples, we will create 2 customized tools for
``image generation`` and ``weather qurey``. A detailed description of
these tools should be defined in json format, which will be used as part
of prompt.

.. code:: ipython3

    tools = [
        {
            "name_for_human": "get weather",
            "name_for_model": "get_weather",
            "description_for_model": 'Get the current weather in a given city name."',
            "parameters": [
                {
                    "name": "city_name",
                    "description": "City name",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
        },
        {
            "name_for_human": "image generation",
            "name_for_model": "image_gen",
            "description_for_model": "AI painting (image generation) service, input text description, and return the image URL drawn based on text information.",
            "parameters": [
                {
                    "name": "prompt",
                    "description": "describe the image",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
        },
    ]

Then we should implement these tools with inputs and outputs, and
execute them according to the output of LLM.

.. code:: ipython3

    def call_tool(tool_name: str, tool_args: str) -> str:
        if tool_name == "get_weather":
            city_name = json5.loads(tool_args)["city_name"]
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
        elif tool_name == "image_gen":
            import urllib.parse

            tool_args = tool_args.replace("(", "").replace(")", "")
            prompt = json5.loads(tool_args)["prompt"]
            prompt = urllib.parse.quote(prompt)
            return json.dumps(
                {"image_url": f"https://image.pollinations.ai/prompt/{prompt}"},
                ensure_ascii=False,
            )
        else:
            raise NotImplementedError


    def llm_with_tool(prompt: str, history, list_of_tool_info=()):
        chat_history = [(x["user"], x["bot"]) for x in history] + [(prompt, "")]

        planning_prompt = build_input_text(chat_history, list_of_tool_info)
        text = ""
        while True:
            output = text_completion(planning_prompt + text, stop_words=["Observation:", "Observation:\n"])
            action, action_input, output = parse_latest_tool_call(output)
            if action:
                observation = call_tool(action, action_input)
                output += f"\nObservation: = {observation}\nThought:"
                observation = f"{observation}\nThought:"
                print(observation)
                text += output
            else:
                text += output
                break

        new_history = []
        new_history.extend(history)
        new_history.append({"user": prompt, "bot": text})
        return text, new_history

Run agent
---------



.. code:: ipython3

    history = []
    query = "get the weather in London, and create a picture of Big Ben based on the weather information"

    response, history = llm_with_tool(prompt=query, history=history, list_of_tool_info=tools)


.. parsed-literal::

    The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.


.. parsed-literal::

    Thought: First, I need to use the get_weather API to get the current weather in London.
    Action: get_weather
    Action Input: {"city_name": "London"}
    Observation:
    {'current_condition': {'temp_C': '6', 'FeelsLikeC': '3', 'humidity': '75', 'weatherDesc': [{'value': 'Overcast'}], 'observation_time': '12:32 AM'}}
    Thought:
     Now that I have the weather information, I will use the image_gen API to generate an image of Big Ben based on the weather conditions.
    Action: image_gen
    Action Input: {"prompt": "Big Ben under an overcast sky with a temperature of 6 degrees Celsius and humidity of 75%"}
    Observation:
    {"image_url": "https://image.pollinations.ai/prompt/Big%20Ben%20under%20an%20overcast%20sky%20with%20a%20temperature%20of%206%20degrees%20Celsius%20and%20humidity%20of%2075%25"}
    Thought:
     I now know the final answer.
    Final Answer: The current weather in London is overcast with a temperature of 6 degrees Celsius and humidity of 75%. Based on this information, here is the generated image of Big Ben under an overcast sky: ![](https://image.pollinations.ai/prompt/Big%20Ben%20under%20an%20overcast%20sky%20with%20a%20temperature%20of%206%20degrees%20Celsius%20and%20humidity%20of%2075%25).


Create AI agent demo with Gradio UI
-----------------------------------



.. code:: ipython3

    from transformers import TextIteratorStreamer
    from threading import Thread


    def run_chatbot(history):
        """
        callback function for running chatbot on submit button click

        Params:
          history: conversation history

        """
        chat_history = [(history[-1][0], "")]

        prompt = build_input_text(chat_history, tools)
        text = ""
        while True:
            planning_prompt = prompt + text
            im_end = "<|im_end|>"
            stop_words = ["Observation:", "Observation:\n"]
            if im_end not in stop_words:
                stop_words = stop_words + [im_end]
            streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)

            stopping_criteria = StoppingCriteriaList([StopSequenceCriteria(stop_words, tokenizer)])
            input_ids = torch.tensor([tokenizer.encode(planning_prompt)])
            generate_kwargs = dict(
                input_ids=input_ids,
                streamer=streamer,
                stopping_criteria=stopping_criteria,
            )

            thread = Thread(target=llm.generate, kwargs=generate_kwargs)
            thread.start()
            output = ""
            output_gui = ""
            show_response = False
            for new_text in streamer:
                output += new_text
                if "Final" in new_text:
                    show_response = True
                if show_response:
                    output_gui += new_text
                    history[-1][1] = output_gui
                    yield history

            # assert buffer.startswith(prompt)
            for stop_str in stop_words:
                idx = output.find(stop_str)
                if idx != -1:
                    output = output[: idx + len(stop_str)]
            print(output)
            action, action_input, output = parse_latest_tool_call(output)
            if action:
                observation = call_tool(action, action_input)
                output += f"\nObservation: = {observation}\nThought:"
                observation = f"{observation}\nThought:"
                print(observation)
                text += output
            else:
                text += output
                break


    def request_cancel():
        llm.request.cancel()

.. code:: ipython3

    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/llm-agent-react/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)

    from gradio_helper import make_demo

    examples = [
        ["Based on current weather in Beijing, show me a picture of Great Wall through its URL"],
        ["Create an image of pink cat and return its URL"],
        ["What is the weather like in New York now ?"],
    ]

    demo = make_demo(run_fn=run_chatbot, stop_fn=request_cancel, examples=examples)

    try:
        demo.launch()
    except Exception:
        demo.launch(share=True)
    # If you are launching remotely, specify server_name and server_port
    # EXAMPLE: `demo.launch(server_name='your server name', server_port='server port in int')`
    # To learn more please refer to the Gradio docs: https://gradio.app/docs/

.. code:: ipython3

    # please uncomment and run this cell for stopping gradio interface
    # demo.close()


.. parsed-literal::

    Closing server running on port: 5612

