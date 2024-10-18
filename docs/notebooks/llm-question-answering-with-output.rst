LLM Instruction-following pipeline with OpenVINO
================================================

LLM stands for “Large Language Model,” which refers to a type of
artificial intelligence model that is designed to understand and
generate human-like text based on the input it receives. LLMs are
trained on large datasets of text to learn patterns, grammar, and
semantic relationships, allowing them to generate coherent and
contextually relevant responses. One core capability of Large Language
Models (LLMs) is to follow natural language instructions.
Instruction-following models are capable of generating text in response
to prompts and are often used for tasks like writing assistance,
chatbots, and content generation.

In this tutorial, we consider how to run an instruction-following text
generation pipeline using popular LLMs and OpenVINO. We will use
pre-trained models from the `Hugging Face
Transformers <https://huggingface.co/docs/transformers/index>`__
library. The `Hugging Face Optimum
Intel <https://huggingface.co/docs/optimum/intel/index>`__ library
converts the models to OpenVINO™ IR format. To simplify the user
experience, we will use `OpenVINO Generate
API <https://github.com/openvinotoolkit/openvino.genai>`__ for
generation of instruction-following inference pipeline.

The tutorial consists of the following steps:

-  Install prerequisites
-  Download and convert the model from a public source using the
   `OpenVINO integration with Hugging Face
   Optimum <https://huggingface.co/blog/openvino>`__.
-  Compress model weights to INT8 and INT4 with `OpenVINO
   NNCF <https://github.com/openvinotoolkit/nncf>`__
-  Create an instruction-following inference pipeline with `Generate
   API <https://github.com/openvinotoolkit/openvino.genai>`__
-  Run instruction-following pipeline


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Select model for inference <#select-model-for-inference>`__
-  `Download and convert model to OpenVINO IR via Optimum Intel
   CLI <#download-and-convert-model-to-openvino-ir-via-optimum-intel-cli>`__
-  `Compress model weights <#compress-model-weights>`__

   -  `Weights Compression using Optimum Intel
      CLI <#weights-compression-using-optimum-intel-cli>`__
   -  `Weights Compression using
      NNCF <#weights-compression-using-nncf>`__

-  `Select device for inference and model
   variant <#select-device-for-inference-and-model-variant>`__
-  `Create an instruction-following inference
   pipeline <#create-an-instruction-following-inference-pipeline>`__

   -  `Setup imports <#setup-imports>`__
   -  `Prepare text streamer to get results
      runtime <#prepare-text-streamer-to-get-results-runtime>`__
   -  `Main generation function <#main-generation-function>`__
   -  `Helpers for application <#helpers-for-application>`__

-  `Run instruction-following
   pipeline <#run-instruction-following-pipeline>`__

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

    %pip uninstall -q -y optimum optimum-intel
    %pip install  -Uq "openvino>=2024.3.0" "openvino-genai"
    %pip install -q "torch>=2.1" "nncf>=2.7" "transformers>=4.40.0" "onnx<1.16.2" "optimum>=1.16.1" "accelerate" "datasets>=2.14.6" "gradio>=4.19" "git+https://github.com/huggingface/optimum-intel.git" --extra-index-url https://download.pytorch.org/whl/cpu

Select model for inference
--------------------------



The tutorial supports different models, you can select one from the
provided options to compare the quality of open source LLM solutions.
>\ **Note**: conversion of some models can require additional actions
from user side and at least 64GB RAM for conversion.

The available options are:

-  **tiny-llama-1b-chat** - This is the chat model finetuned on top of
   `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T <https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T>`__.
   The TinyLlama project aims to pretrain a 1.1B Llama model on 3
   trillion tokens with the adoption of the same architecture and
   tokenizer as Llama 2. This means TinyLlama can be plugged and played
   in many open-source projects built upon Llama. Besides, TinyLlama is
   compact with only 1.1B parameters. This compactness allows it to
   cater to a multitude of applications demanding a restricted
   computation and memory footprint. More details about model can be
   found in `model
   card <https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0>`__
-  **phi-2** - Phi-2 is a Transformer with 2.7 billion parameters. It
   was trained using the same data sources as
   `Phi-1.5 <https://huggingface.co/microsoft/phi-1_5>`__, augmented
   with a new data source that consists of various NLP synthetic texts
   and filtered websites (for safety and educational value). When
   assessed against benchmarks testing common sense, language
   understanding, and logical reasoning, Phi-2 showcased a nearly
   state-of-the-art performance among models with less than 13 billion
   parameters. More details about model can be found in `model
   card <https://huggingface.co/microsoft/phi-2#limitations-of-phi-2>`__.
-  **dolly-v2-3b** - Dolly 2.0 is an instruction-following large
   language model trained on the Databricks machine-learning platform
   that is licensed for commercial use. It is based on
   `Pythia <https://github.com/EleutherAI/pythia>`__ and is trained on
   ~15k instruction/response fine-tuning records generated by Databricks
   employees in various capability domains, including brainstorming,
   classification, closed QA, generation, information extraction, open
   QA, and summarization. Dolly 2.0 works by processing natural language
   instructions and generating responses that follow the given
   instructions. It can be used for a wide range of applications,
   including closed question-answering, summarization, and generation.
   More details about model can be found in `model
   card <https://huggingface.co/databricks/dolly-v2-3b>`__.
-  **red-pajama-3b-instruct** - A 2.8B parameter pre-trained language
   model based on GPT-NEOX architecture. The model was fine-tuned for
   few-shot applications on the data of
   `GPT-JT <https://huggingface.co/togethercomputer/GPT-JT-6B-v1>`__,
   with exclusion of tasks that overlap with the HELM core
   scenarios.More details about model can be found in `model
   card <https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1>`__.
-  **mistral-7b** - The Mistral-7B-v0.2 Large Language Model (LLM) is a
   pretrained generative text model with 7 billion parameters. You can
   find more details about model in the `model
   card <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2>`__,
   `paper <https://arxiv.org/abs/2310.06825>`__ and `release blog
   post <https://mistral.ai/news/announcing-mistral-7b/>`__.
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
   card <https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>`__,
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
    import requests
    
    # Fetch `notebook_utils` module
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)
    from notebook_utils import download_file, device_widget
    
    if not Path("./config.py").exists():
        download_file(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/llm-question-answering/config.py")
    from config import SUPPORTED_LLM_MODELS
    import ipywidgets as widgets

.. code:: ipython3

    model_ids = list(SUPPORTED_LLM_MODELS)
    
    model_id = widgets.Dropdown(
        options=model_ids,
        value=model_ids[1],
        description="Model:",
        disabled=False,
    )
    
    model_id




.. parsed-literal::

    Dropdown(description='Model:', index=1, options=('tiny-llama-1b', 'phi-2', 'dolly-v2-3b', 'red-pajama-instruct…



.. code:: ipython3

    model_configuration = SUPPORTED_LLM_MODELS[model_id.value]
    print(f"Selected model {model_id.value}")


.. parsed-literal::

    Selected model dolly-v2-3b


Download and convert model to OpenVINO IR via Optimum Intel CLI
---------------------------------------------------------------



Listed model are available for downloading via the `HuggingFace
hub <https://huggingface.co/models>`__. We will use optimum-cli
interface for exporting it into OpenVINO Intermediate Representation
(IR) format.

Optimum CLI interface for converting models supports export to OpenVINO
(supported starting optimum-intel 1.12 version). General command format:

.. code:: bash

   optimum-cli export openvino --model <model_id_or_path> --task <task> <output_dir>

where ``--model`` argument is model id from HuggingFace Hub or local
directory with model (saved using ``.save_pretrained`` method),
``--task`` is one of `supported
task <https://huggingface.co/docs/optimum/exporters/task_manager>`__
that exported model should solve. For LLMs it will be
``text-generation-with-past``. If model initialization requires to use
remote code, ``--trust-remote-code`` flag additionally should be passed.
Full list of supported arguments available via ``--help`` For more
details and examples of usage, please check `optimum
documentation <https://huggingface.co/docs/optimum/intel/inference#export>`__.

Compress model weights
----------------------



The Weights Compression algorithm is aimed at compressing the weights of
the models and can be used to optimize the model footprint and
performance of large models where the size of weights is relatively
larger than the size of activations, for example, Large Language Models
(LLM). Compared to INT8 compression, INT4 compression improves
performance even more but introduces a minor drop in prediction quality.

Weights Compression using Optimum Intel CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Optimum Intel supports weight compression via NNCF out of the box. For
8-bit compression we pass ``--weight-format int8`` to ``optimum-cli``
command line. For 4 bit compression we provide ``--weight-format int4``
and some other options containing number of bits and other compression
parameters. An example of this approach usage you can find in
`llm-chatbot notebook <llm-chatbot-with-output.html>`__

Weights Compression using NNCF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



You also can perform weights compression for OpenVINO models using NNCF
directly. ``nncf.compress_weights`` function accepts the OpenVINO model
instance and compresses its weights for Linear and Embedding layers. We
will consider this variant in this notebook for both int4 and int8
compression.

   **Note**: This tutorial involves conversion model for FP16 and
   INT4/INT8 weights compression scenarios. It may be memory and
   time-consuming in the first run. You can manually control the
   compression precision below. **Note**: There may be no speedup for
   INT4/INT8 compressed models on dGPU

.. code:: ipython3

    from IPython.display import display, Markdown
    
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

    from pathlib import Path
    import logging
    import openvino as ov
    import nncf
    
    nncf.set_log_level(logging.ERROR)
    
    pt_model_id = model_configuration["model_id"]
    fp16_model_dir = Path(model_id.value) / "FP16"
    int8_model_dir = Path(model_id.value) / "INT8_compressed_weights"
    int4_model_dir = Path(model_id.value) / "INT4_compressed_weights"
    
    core = ov.Core()
    
    
    def convert_to_fp16():
        if (fp16_model_dir / "openvino_model.xml").exists():
            return
        export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format fp16".format(pt_model_id)
        export_command = export_command_base + " " + str(fp16_model_dir)
        display(Markdown("**Export command:**"))
        display(Markdown(f"`{export_command}`"))
        ! $export_command
    
    
    def convert_to_int8():
        if (int8_model_dir / "openvino_model.xml").exists():
            return
        int8_model_dir.mkdir(parents=True, exist_ok=True)
        export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format int8".format(pt_model_id)
        export_command = export_command_base + " " + str(int8_model_dir)
        display(Markdown("**Export command:**"))
        display(Markdown(f"`{export_command}`"))
        ! $export_command
    
    
    def convert_to_int4():
        compression_configs = {
            "mistral-7b": {
                "sym": True,
                "group_size": 64,
                "ratio": 0.6,
            },
            "red-pajama-3b-instruct": {
                "sym": False,
                "group_size": 128,
                "ratio": 0.5,
            },
            "dolly-v2-3b": {"sym": False, "group_size": 32, "ratio": 0.5},
            "llama-3-8b-instruct": {"sym": True, "group_size": 128, "ratio": 1.0},
            "default": {
                "sym": False,
                "group_size": 128,
                "ratio": 0.8,
            },
        }
    
        model_compression_params = compression_configs.get(model_id.value, compression_configs["default"])
        if (int4_model_dir / "openvino_model.xml").exists():
            return
        export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format int4".format(pt_model_id)
        int4_compression_args = " --group-size {} --ratio {}".format(model_compression_params["group_size"], model_compression_params["ratio"])
        if model_compression_params["sym"]:
            int4_compression_args += " --sym"
        export_command_base += int4_compression_args
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


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino


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

    Size of FP16 model is 5297.21 MB
    Size of model with INT8 compressed weights is 2656.29 MB
    Compression rate for INT8 model: 1.994
    Size of model with INT4 compressed weights is 2154.54 MB
    Compression rate for INT4 model: 2.459


Select device for inference and model variant
---------------------------------------------



   **Note**: There may be no speedup for INT4/INT8 compressed models on
   dGPU.

.. code:: ipython3

    core = ov.Core()
    
    device = device_widget("CPU", exclude=["NPU"])
    
    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



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

    Dropdown(description='Model to run:', options=('INT4', 'INT8', 'FP16'), value='INT4')



.. code:: ipython3

    from transformers import AutoTokenizer
    from openvino_tokenizers import convert_tokenizer
    
    if model_to_run.value == "INT4":
        model_dir = int4_model_dir
    elif model_to_run.value == "INT8":
        model_dir = int8_model_dir
    else:
        model_dir = fp16_model_dir
    print(f"Loading model from {model_dir}")
    
    # optionally convert tokenizer if used cached model without it
    if not (model_dir / "openvino_tokenizer.xml").exists() or not (model_dir / "openvino_detokenizer.xml").exists():
        hf_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        ov_tokenizer, ov_detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
        ov.save_model(ov_tokenizer, model_dir / "openvino_tokenizer.xml")
        ov.save_model(ov_tokenizer, model_dir / "openvino_detokenizer.xml")


.. parsed-literal::

    Loading model from dolly-v2-3b/INT8_compressed_weights


Create an instruction-following inference pipeline
--------------------------------------------------



The ``run_generation`` function accepts user-provided text input,
tokenizes it, and runs the generation process. Text generation is an
iterative process, where each next token depends on previously generated
until a maximum number of tokens or stop generation condition is not
reached.

The diagram below illustrates how the instruction-following pipeline
works

.. figure:: https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/e881f4a4-fcc8-427a-afe1-7dd80aebd66e
   :alt: generation pipeline)

   generation pipeline)

As can be seen, on the first iteration, the user provided instructions.
Instructions is converted to token ids using a tokenizer, then prepared
input provided to the model. The model generates probabilities for all
tokens in logits format. The way the next token will be selected over
predicted probabilities is driven by the selected decoding methodology.
You can find more information about the most popular decoding methods in
this `blog <https://huggingface.co/blog/how-to-generate>`__.

To simplify user experience we will use `OpenVINO Generate
API <https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md>`__.
Firstly we will create pipeline with ``LLMPipeline``. ``LLMPipeline`` is
the main object used for decoding. You can construct it straight away
from the folder with the converted model. It will automatically load the
``main model``, ``tokenizer``, ``detokenizer`` and default
``generation configuration``. After that we will configure parameters
for decoding. We can get default config with
``get_generation_config()``, setup parameters and apply the updated
version with ``set_generation_config(config)`` or put config directly to
``generate()``. It’s also possible to specify the needed options just as
inputs in the ``generate()`` method, as shown below. Then we just run
``generate`` method and get the output in text format. We do not need to
encode input prompt according to model expected template or write
post-processing code for logits decoder, it will be done easily with
LLMPipeline.

To obtain intermediate generation results without waiting until when
generation is finished, we will write class-iterator based on
``StreamerBase`` class of ``openvino_genai``.

.. code:: ipython3

    from openvino_genai import LLMPipeline
    
    pipe = LLMPipeline(model_dir.as_posix(), device.value)
    print(pipe.generate("The Sun is yellow bacause", temperature=1.2, top_k=4, do_sample=True, max_new_tokens=150))


.. parsed-literal::

     of the presence of chlorophyll
    in its leaves. Chlorophyll absorbs all
    visible sunlight and this causes it to
    turn from a green to yellow colour.
    The Sun is yellow bacause of the presence of chlorophyll in its leaves. Chlorophyll absorbs all
    visible sunlight and this causes it to
    turn from a green to yellow colour.
    The yellow colour of the Sun is the
    colour we perceive as the colour of the
    sun. It also causes us to perceive the
    sun as yellow. This property is called
    the yellow colouration of the Sun and it
    is caused by the presence of chlorophyll
    in the leaves of plants.
    Chlorophyll is also responsible for the green colour of plants


There are several parameters that can control text generation quality:

-  | ``Temperature`` is a parameter used to control the level of
     creativity in AI-generated text. By adjusting the ``temperature``,
     you can influence the AI model’s probability distribution, making
     the text more focused or diverse.
   | Consider the following example: The AI model has to complete the
     sentence “The cat is \____.” with the following token
     probabilities:

   | playing: 0.5
   | sleeping: 0.25
   | eating: 0.15
   | driving: 0.05
   | flying: 0.05

   -  **Low temperature** (e.g., 0.2): The AI model becomes more focused
      and deterministic, choosing tokens with the highest probability,
      such as “playing.”
   -  **Medium temperature** (e.g., 1.0): The AI model maintains a
      balance between creativity and focus, selecting tokens based on
      their probabilities without significant bias, such as “playing,”
      “sleeping,” or “eating.”
   -  **High temperature** (e.g., 2.0): The AI model becomes more
      adventurous, increasing the chances of selecting less likely
      tokens, such as “driving” and “flying.”

-  ``Top-p``, also known as nucleus sampling, is a parameter used to
   control the range of tokens considered by the AI model based on their
   cumulative probability. By adjusting the ``top-p`` value, you can
   influence the AI model’s token selection, making it more focused or
   diverse. Using the same example with the cat, consider the following
   top_p settings:

   -  **Low top_p** (e.g., 0.5): The AI model considers only tokens with
      the highest cumulative probability, such as “playing.”
   -  **Medium top_p** (e.g., 0.8): The AI model considers tokens with a
      higher cumulative probability, such as “playing,” “sleeping,” and
      “eating.”
   -  **High top_p** (e.g., 1.0): The AI model considers all tokens,
      including those with lower probabilities, such as “driving” and
      “flying.”

-  ``Top-k`` is another popular sampling strategy. In comparison with
   Top-P, which chooses from the smallest possible set of words whose
   cumulative probability exceeds the probability P, in Top-K sampling K
   most likely next words are filtered and the probability mass is
   redistributed among only those K next words. In our example with cat,
   if k=3, then only “playing”, “sleeping” and “eating” will be taken
   into account as possible next word.

The generation cycle repeats until the end of the sequence token is
reached or it also can be interrupted when maximum tokens will be
generated. As already mentioned before, we can enable printing current
generated tokens without waiting until when the whole generation is
finished using Streaming API, it adds a new token to the output queue
and then prints them when they are ready.

Setup imports
~~~~~~~~~~~~~



.. code:: ipython3

    from threading import Thread
    from time import perf_counter
    from typing import List
    import numpy as np
    from openvino_genai import StreamerBase
    from queue import Queue
    import re

Prepare text streamer to get results runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Load the ``detokenizer``, use it to convert token_id to string output
format. We will collect print-ready text in a queue and give the text
when it is needed. It will help estimate performance.

.. code:: ipython3

    core = ov.Core()
    
    detokinizer_dir = Path(model_dir, "openvino_detokenizer.xml")
    
    
    class TextIteratorStreamer(StreamerBase):
        def __init__(self, tokenizer):
            super().__init__()
            self.tokenizer = tokenizer
            self.compiled_detokenizer = core.compile_model(detokinizer_dir.as_posix())
            self.text_queue = Queue()
            self.stop_signal = None
    
        def __iter__(self):
            return self
    
        def __next__(self):
            value = self.text_queue.get()
            if value == self.stop_signal:
                raise StopIteration()
            else:
                return value
    
        def put(self, token_id):
            openvino_output = self.compiled_detokenizer(np.array([[token_id]], dtype=int))
            text = str(openvino_output["string_output"][0])
            # remove labels/special symbols
            text = text.lstrip("!")
            text = re.sub("<.*>", "", text)
            self.text_queue.put(text)
    
        def end(self):
            self.text_queue.put(self.stop_signal)

Main generation function
~~~~~~~~~~~~~~~~~~~~~~~~



As it was discussed above, ``run_generation`` function is the entry
point for starting generation. It gets provided input instruction as
parameter and returns model response.

.. code:: ipython3

    def run_generation(
        user_text: str,
        top_p: float,
        temperature: float,
        top_k: int,
        max_new_tokens: int,
        perf_text: str,
    ):
        """
        Text generation function
    
        Parameters:
          user_text (str): User-provided instruction for a generation.
          top_p (float):  Nucleus sampling. If set to < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for a generation.
          temperature (float): The value used to module the logits distribution.
          top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
          max_new_tokens (int): Maximum length of generated sequence.
          perf_text (str): Content of text field for printing performance results.
        Returns:
          model_output (str) - model-generated text
          perf_text (str) - updated perf text filed content
        """
    
        # setup config for decoding stage
        config = pipe.get_generation_config()
        config.temperature = temperature
        if top_k > 0:
            config.top_k = top_k
        config.top_p = top_p
        config.do_sample = True
        config.max_new_tokens = max_new_tokens
    
        # Start generation on a separate thread, so that we don't block the UI. The text is pulled from the streamer
        # in the main thread.
        streamer = TextIteratorStreamer(pipe.get_tokenizer())
        t = Thread(target=pipe.generate, args=(user_text, config, streamer))
        t.start()
    
        model_output = ""
        per_token_time = []
        num_tokens = 0
        start = perf_counter()
        for new_text in streamer:
            current_time = perf_counter() - start
            model_output += new_text
            perf_text, num_tokens = estimate_latency(current_time, perf_text, per_token_time, num_tokens)
            yield model_output, perf_text
            start = perf_counter()
        return model_output, perf_text

Helpers for application
~~~~~~~~~~~~~~~~~~~~~~~



For making interactive user interface we will use Gradio library. The
code bellow provides useful functions used for communication with UI
elements.

.. code:: ipython3

    def estimate_latency(
        current_time: float,
        current_perf_text: str,
        per_token_time: List[float],
        num_tokens: int,
    ):
        """
        Helper function for performance estimation
    
        Parameters:
          current_time (float): This step time in seconds.
          current_perf_text (str): Current content of performance UI field.
          per_token_time (List[float]): history of performance from previous steps.
          num_tokens (int): Total number of generated tokens.
    
        Returns:
          update for performance text field
          update for a total number of tokens
        """
        num_tokens += 1
        per_token_time.append(1 / current_time)
        if len(per_token_time) > 10 and len(per_token_time) % 4 == 0:
            current_bucket = per_token_time[:-10]
            return (
                f"Average generation speed: {np.mean(current_bucket):.2f} tokens/s. Total generated tokens: {num_tokens}",
                num_tokens,
            )
        return current_perf_text, num_tokens

Run instruction-following pipeline
----------------------------------



Now, we are ready to explore model capabilities. This demo provides a
simple interface that allows communication with a model using text
instruction. Type your instruction into the ``User instruction`` field
or select one from predefined examples and click on the ``Submit``
button to start generation. Additionally, you can modify advanced
generation parameters:

-  ``Device`` - allows switching inference device. Please note, every
   time when new device is selected, model will be recompiled and this
   takes some time.
-  ``Max New Tokens`` - maximum size of generated text.
-  ``Top-p (nucleus sampling)`` - if set to < 1, only the smallest set
   of most probable tokens with probabilities that add up to top_p or
   higher are kept for a generation.
-  ``Top-k`` - the number of highest probability vocabulary tokens to
   keep for top-k-filtering.
-  ``Temperature`` - the value used to module the logits distribution.

.. code:: ipython3

    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/llm-question-answering/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)
    
    from gradio_helper import make_demo
    
    demo = make_demo(run_fn=run_generation, title=f"Question Answering with {model_id.value} and OpenVINO")
    
    try:
        demo.queue().launch(height=800)
    except Exception:
        demo.queue().launch(share=True, height=800)
    # If you are launching remotely, specify server_name and server_port
    # EXAMPLE: `demo.launch(server_name='your server name', server_port='server port in int')`
    # To learn more please refer to the Gradio docs: https://gradio.app/docs/

.. code:: ipython3

    # please uncomment and run this cell for stopping gradio interface
    # demo.close()
