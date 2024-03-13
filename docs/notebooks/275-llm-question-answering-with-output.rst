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
library. To simplify the user experience, the `Hugging Face Optimum
Intel <https://huggingface.co/docs/optimum/intel/index>`__ library
converts the models to OpenVINO™ IR format.

The tutorial consists of the following steps:

-  Install prerequisites
-  Download and convert the model from a public source using the
   `OpenVINO integration with Hugging Face
   Optimum <https://huggingface.co/blog/openvino>`__.
-  Compress model weights to INT8 and INT4 with `OpenVINO
   NNCF <https://github.com/openvinotoolkit/nncf>`__
-  Create an instruction-following inference pipeline
-  Run instruction-following pipeline

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prerequisites <#prerequisites>`__
-  `Select model for inference <#select-model-for-inference>`__
-  `Instantiate Model using Optimum
   Intel <#instantiate-model-using-optimum-intel>`__
-  `Compress model weights <#compress-model-weights>`__

   -  `Weights Compression using Optimum
      Intel <#weights-compression-using-optimum-intel>`__
   -  `Weights Compression using
      NNCF <#weights-compression-using-nncf>`__

-  `Select device for inference and model
   variant <#select-device-for-inference-and-model-variant>`__
-  `Create an instruction-following inference
   pipeline <#create-an-instruction-following-inference-pipeline>`__

   -  `Setup imports <#setup-imports>`__
   -  `Prepare template for user
      prompt <#prepare-template-for-user-prompt>`__
   -  `Main generation function <#main-generation-function>`__
   -  `Helpers for application <#helpers-for-application>`__

-  `Run instruction-following
   pipeline <#run-instruction-following-pipeline>`__

Prerequisites
-------------



.. code:: ipython3

    %pip uninstall -q -y openvino openvino-dev openvino-nightly optimum optimum-intel
    %pip install -q openvino-nightly "nncf>=2.7" "transformers>=4.36.0" onnx "optimum>=1.16.1" "accelerate" "datasets" gradio "git+https://github.com/huggingface/optimum-intel.git" --extra-index-url https://download.pytorch.org/whl/cpu

Select model for inference
--------------------------



The tutorial supports different models, you can select one from the
provided options to compare the quality of open source LLM solutions.

   **NOTE**: conversion of some models can require additional actions
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

.. code:: ipython3

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

    Selected model phi-2


Instantiate Model using Optimum Intel
-------------------------------------



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
method. When downloading and converting the Transformers model, the
parameter ``export=True`` should be added. We can save the converted
model for the next usage with the ``save_pretrained`` method. Tokenizer
class and pipelines API are compatible with Optimum models.

To optimize the generation process and use memory more efficiently, the
``use_cache=True`` option is enabled. Since the output side is
auto-regressive, an output token hidden state remains the same once
computed for every further generation step. Therefore, recomputing it
every time you want to generate a new token seems wasteful. With the
cache, the model saves the hidden state once it has been computed. The
model only computes the one for the most recently generated output token
at each time step, re-using the saved ones for hidden tokens. This
reduces the generation complexity from :math:`O(n^3)` to :math:`O(n^2)`
for a transformer model. More details about how it works can be found in
this
`article <https://scale.com/blog/pytorch-improvements#Text%20Translation>`__.
With this option, the model gets the previous step’s hidden states
(cached attention keys and values) as input and additionally provides
hidden states for the current step as output. It means for all next
iterations, it is enough to provide only a new token obtained from the
previous step and cached key values to get the next token prediction.

Compress model weights
----------------------

The Weights Compression
algorithm is aimed at compressing the weights of the models and can be
used to optimize the model footprint and performance of large models
where the size of weights is relatively larger than the size of
activations, for example, Large Language Models (LLM). Compared to INT8
compression, INT4 compression improves performance even more but
introduces a minor drop in prediction quality.

Weights Compression using Optimum Intel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



To enable weight compression via NNCF for models supported by Optimum
Intel ``OVQuantizer`` class should be used for ``OVModelForCausalLM``
model.
``OVQuantizer.quantize(save_directory=save_dir, weights_only=True)``
enables weights compression. An example of this approach usage you can
find in `llm-chatbot notebook <254-llm-chatbot-with-output.html>`__

Weights Compression using NNCF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



You also can perform weights compression for OpenVINO models using NNCF
directly. ``nncf.compress_weights`` function accepts the OpenVINO model
instance and compresses its weights for Linear and Embedding layers. We
will consider this variant in this notebook for both int4 and int8
compression.

   **NOTE**: This tutorial involves conversion model for FP16 and
   INT4/INT8 weights compression scenarios. It may be memory and
   time-consuming in the first run. You can manually control the
   compression precision below. **NOTE**: There may be no speedup for
   INT4/INT8 compressed models on dGPU

.. code:: ipython3

    from IPython.display import display

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
    import shutil
    import logging
    import openvino as ov
    import nncf
    from optimum.intel.openvino import OVModelForCausalLM
    from optimum.utils import NormalizedTextConfig, NormalizedConfigManager
    import gc

    NormalizedConfigManager._conf['phi'] = NormalizedTextConfig

    nncf.set_log_level(logging.ERROR)

    pt_model_id = model_configuration["model_id"]
    fp16_model_dir = Path(model_id.value) / "FP16"
    int8_model_dir = Path(model_id.value) / "INT8_compressed_weights"
    int4_model_dir = Path(model_id.value) / "INT4_compressed_weights"

    core = ov.Core()

    def convert_to_fp16():
        if (fp16_model_dir / "openvino_model.xml").exists():
            return
        ov_model = OVModelForCausalLM.from_pretrained(pt_model_id, export=True, compile=False, load_in_8bit=False)
        ov_model.half()
        ov_model.save_pretrained(fp16_model_dir)
        del ov_model
        gc.collect()


    def convert_to_int8():
        if (int8_model_dir / "openvino_model.xml").exists():
            return
        int8_model_dir.mkdir(parents=True, exist_ok=True)
        if fp16_model_dir.exists():
            model = core.read_model(fp16_model_dir / "openvino_model.xml")
            shutil.copy(fp16_model_dir / "config.json", int8_model_dir / "config.json")
        else:
            ov_model = OVModelForCausalLM.from_pretrained(pt_model_id, export=True, compile=False, load_in_8bit=False)
            ov_model.half()
            ov_model.config.save_pretrained(int8_model_dir)
            model = ov_model._original_model
            del ov_model
            gc.collect()

        compressed_model = nncf.compress_weights(model)
        ov.save_model(compressed_model, int8_model_dir / "openvino_model.xml")
        del ov_model
        del compressed_model
        gc.collect()


    def convert_to_int4():
        compression_configs = {
            "mistral-7b": {
                "mode": nncf.CompressWeightsMode.INT4_SYM,
                "group_size": 64,
                "ratio": 0.6,
            },
            'red-pajama-3b-instruct': {
                "mode": nncf.CompressWeightsMode.INT4_ASYM,
                "group_size": 128,
                "ratio": 0.5,
            },
            "dolly-v2-3b": {"mode": nncf.CompressWeightsMode.INT4_ASYM, "group_size": 32, "ratio": 0.5},
            "default": {
                "mode": nncf.CompressWeightsMode.INT4_ASYM,
                "group_size": 128,
                "ratio": 0.8,
            },
        }

        model_compression_params = compression_configs.get(
            model_id.value, compression_configs["default"]
        )
        if (int4_model_dir / "openvino_model.xml").exists():
            return
        int4_model_dir.mkdir(parents=True, exist_ok=True)
        if not fp16_model_dir.exists():
            model = OVModelForCausalLM.from_pretrained(pt_model_id, export=True, compile=False, load_in_8bit=False).half()
            model.config.save_pretrained(int4_model_dir)
            ov_model = model._original_model
            del model
            gc.collect()
        else:
            ov_model = core.read_model(fp16_model_dir / "openvino_model.xml")
            shutil.copy(fp16_model_dir / "config.json", int4_model_dir / "config.json")
        compressed_model = nncf.compress_weights(ov_model, **model_compression_params)
        ov.save_model(compressed_model, int4_model_dir / "openvino_model.xml")
        del ov_model
        del compressed_model
        gc.collect()


    if prepare_fp16_model.value:
        convert_to_fp16()
    if prepare_int8_model.value:
        convert_to_int8()
    if prepare_int4_model.value:
        convert_to_int4()


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino


.. parsed-literal::

    /home/ea/work/genai_env/lib/python3.8/site-packages/torch/cuda/__init__.py:138: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 11080). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.)
      return torch._C._cuda_getDeviceCount() > 0
    No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'


Let’s compare model size for different compression types

.. code:: ipython3

    fp16_weights = fp16_model_dir / "openvino_model.bin"
    int8_weights = int8_model_dir / "openvino_model.bin"
    int4_weights = int4_model_dir / "openvino_model.bin"

    if fp16_weights.exists():
        print(f"Size of FP16 model is {fp16_weights.stat().st_size / 1024 / 1024:.2f} MB")
    for precision, compressed_weights in zip([8, 4], [int8_weights, int4_weights]):
        if compressed_weights.exists():
            print(
                f"Size of model with INT{precision} compressed weights is {compressed_weights.stat().st_size / 1024 / 1024:.2f} MB"
            )
        if compressed_weights.exists() and fp16_weights.exists():
            print(
                f"Compression rate for INT{precision} model: {fp16_weights.stat().st_size / compressed_weights.stat().st_size:.3f}"
            )


.. parsed-literal::

    Size of model with INT4 compressed weights is 1734.02 MB


Select device for inference and model variant
---------------------------------------------



   **NOTE**: There may be no speedup for INT4/INT8 compressed models on
   dGPU.

.. code:: ipython3

    core = ov.Core()
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value="CPU",
        description="Device:",
        disabled=False,
    )

    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'GPU.0', 'GPU.1', 'AUTO'), value='CPU')



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



.. code:: ipython3

    from transformers import AutoTokenizer

    if model_to_run.value == "INT4":
        model_dir = int4_model_dir
    elif model_to_run.value == "INT8":
        model_dir = int8_model_dir
    else:
        model_dir = fp16_model_dir
    print(f"Loading model from {model_dir}")

    model_name = model_configuration["model_id"]
    ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}

    tok = AutoTokenizer.from_pretrained(model_name)

    ov_model = OVModelForCausalLM.from_pretrained(
        model_dir,
        device=device.value,
        ov_config=ov_config,
    )


.. parsed-literal::

    Loading model from phi-2/INT4_compressed_weights


.. parsed-literal::

    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    Compiling the model to CPU ...


Create an instruction-following inference pipeline
--------------------------------------------------



The ``run_generation`` function accepts user-provided text input,
tokenizes it, and runs the generation process. Text generation is an
iterative process, where each next token depends on previously generated
until a maximum number of tokens or stop generation condition is not
reached. To obtain intermediate generation results without waiting until
when generation is finished, we will use
`TextIteratorStreamer <https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.TextIteratorStreamer>`__,
provided as part of HuggingFace `Streaming
API <https://huggingface.co/docs/transformers/main/en/generation_strategies#streaming>`__.

The diagram below illustrates how the instruction-following pipeline
works

.. figure:: https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/e881f4a4-fcc8-427a-afe1-7dd80aebd66e
   :alt: generation pipeline)

   generation pipeline)

As can be seen, on the first iteration, the user provided instructions
converted to token ids using a tokenizer, then prepared input provided
to the model. The model generates probabilities for all tokens in logits
format The way the next token will be selected over predicted
probabilities is driven by the selected decoding methodology. You can
find more information about the most popular decoding methods in this
`blog <https://huggingface.co/blog/how-to-generate>`__.

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

To optimize the generation process and use memory more efficiently, the
``use_cache=True`` option is enabled. Since the output side is
auto-regressive, an output token hidden state remains the same once
computed for every further generation step. Therefore, recomputing it
every time you want to generate a new token seems wasteful. With the
cache, the model saves the hidden state once it has been computed. The
model only computes the one for the most recently generated output token
at each time step, re-using the saved ones for hidden tokens. This
reduces the generation complexity from O(n^3) to O(n^2) for a
transformer model. More details about how it works can be found in this
`article <https://scale.com/blog/pytorch-improvements#Text%20Translation>`__.
With this option, the model gets the previous step’s hidden states
(cached attention keys and values) as input and additionally provides
hidden states for the current step as output. It means for all next
iterations, it is enough to provide only a new token obtained from the
previous step and cached key values to get the next token prediction.

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
    import gradio as gr
    from transformers import AutoTokenizer, TextIteratorStreamer
    import numpy as np

Prepare template for user prompt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



For effective generation, model expects to have input in specific
format. The code below prepare template for passing user instruction
into model with providing additional context.

.. code:: ipython3

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_kwargs = model_configuration.get("toeknizer_kwargs", {})


    def get_special_token_id(tokenizer: AutoTokenizer, key: str) -> int:
        """
        Gets the token ID for a given string that has been added to the tokenizer as a special token.

        Args:
            tokenizer (PreTrainedTokenizer): the tokenizer
            key (str): the key to convert to a single token

        Raises:
            RuntimeError: if more than one ID was generated

        Returns:
            int: the token ID for the given key
        """
        token_ids = tokenizer.encode(key)
        if len(token_ids) > 1:
            raise ValueError(f"Expected only a single token for '{key}' but found {token_ids}")
        return token_ids[0]

    response_key = model_configuration.get("response_key")
    tokenizer_response_key = None

    if response_key is not None:
        tokenizer_response_key = next((token for token in tokenizer.additional_special_tokens if token.startswith(response_key)), None)

    end_key_token_id = None
    if tokenizer_response_key:
        try:
            end_key = model_configuration.get("end_key")
            if end_key:
                end_key_token_id = get_special_token_id(tokenizer, end_key)
            # Ensure generation stops once it generates "### End"
        except ValueError:
            pass

    prompt_template = model_configuration.get("prompt_template", "{instruction}")
    end_key_token_id = end_key_token_id or tokenizer.eos_token_id
    pad_token_id = end_key_token_id or tokenizer.pad_token_id


.. parsed-literal::

    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.


Main generation function
~~~~~~~~~~~~~~~~~~~~~~~~



As it was discussed above, ``run_generation`` function is the entry
point for starting generation. It gets provided input instruction as
parameter and returns model response.

.. code:: ipython3

    def run_generation(user_text:str, top_p:float, temperature:float, top_k:int, max_new_tokens:int, perf_text:str):
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

        # Prepare input prompt according to model expected template
        prompt_text = prompt_template.format(instruction=user_text)

        # Tokenize the user text.
        model_inputs = tokenizer(prompt_text, return_tensors="pt", **tokenizer_kwargs)

        # Start generation on a separate thread, so that we don't block the UI. The text is pulled from the streamer
        # in the main thread. Adds timeout to the streamer to handle exceptions in the generation thread.
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=float(temperature),
            top_k=top_k,
            eos_token_id=end_key_token_id,
            pad_token_id=pad_token_id
        )
        t = Thread(target=ov_model.generate, kwargs=generate_kwargs)
        t.start()

        # Pull the generated text from the streamer, and update the model output.
        model_output = ""
        per_token_time = []
        num_tokens = 0
        start = perf_counter()
        for new_text in streamer:
            current_time = perf_counter() - start
            model_output += new_text
            perf_text, num_tokens = estimate_latency(current_time, perf_text, new_text, per_token_time, num_tokens)
            yield model_output, perf_text
            start = perf_counter()
        return model_output, perf_text

Helpers for application
~~~~~~~~~~~~~~~~~~~~~~~



For making interactive user interface we will use Gradio library. The
code bellow provides useful functions used for communication with UI
elements.

.. code:: ipython3

    def estimate_latency(current_time:float, current_perf_text:str, new_gen_text:str, per_token_time:List[float], num_tokens:int):
        """
        Helper function for performance estimation

        Parameters:
          current_time (float): This step time in seconds.
          current_perf_text (str): Current content of performance UI field.
          new_gen_text (str): New generated text.
          per_token_time (List[float]): history of performance from previous steps.
          num_tokens (int): Total number of generated tokens.

        Returns:
          update for performance text field
          update for a total number of tokens
        """
        num_current_toks = len(tokenizer.encode(new_gen_text))
        num_tokens += num_current_toks
        per_token_time.append(num_current_toks / current_time)
        if len(per_token_time) > 10 and len(per_token_time) % 4 == 0:
            current_bucket = per_token_time[:-10]
            return f"Average generation speed: {np.mean(current_bucket):.2f} tokens/s. Total generated tokens: {num_tokens}", num_tokens
        return current_perf_text, num_tokens

    def reset_textbox(instruction:str, response:str, perf:str):
        """
        Helper function for resetting content of all text fields

        Parameters:
          instruction (str): Content of user instruction field.
          response (str): Content of model response field.
          perf (str): Content of performance info filed

        Returns:
          empty string for each placeholder
        """
        return "", "", ""

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

    examples = [
        "Give me a recipe for pizza with pineapple",
        "Write me a tweet about the new OpenVINO release",
        "Explain the difference between CPU and GPU",
        "Give five ideas for a great weekend with family",
        "Do Androids dream of Electric sheep?",
        "Who is Dolly?",
        "Please give me advice on how to write resume?",
        "Name 3 advantages to being a cat",
        "Write instructions on how to become a good AI engineer",
        "Write a love letter to my best friend",
    ]



    with gr.Blocks() as demo:
        gr.Markdown(
            "# Question Answering with " + model_id.value + " and OpenVINO.\n"
            "Provide instruction which describes a task below or select among predefined examples and model writes response that performs requested task."
        )

        with gr.Row():
            with gr.Column(scale=4):
                user_text = gr.Textbox(
                    placeholder="Write an email about an alpaca that likes flan",
                    label="User instruction"
                )
                model_output = gr.Textbox(label="Model response", interactive=False)
                performance = gr.Textbox(label="Performance", lines=1, interactive=False)
                with gr.Column(scale=1):
                    button_clear = gr.Button(value="Clear")
                    button_submit = gr.Button(value="Submit")
                gr.Examples(examples, user_text)
            with gr.Column(scale=1):
                max_new_tokens = gr.Slider(
                    minimum=1, maximum=1000, value=256, step=1, interactive=True, label="Max New Tokens",
                )
                top_p = gr.Slider(
                    minimum=0.05, maximum=1.0, value=0.92, step=0.05, interactive=True, label="Top-p (nucleus sampling)",
                )
                top_k = gr.Slider(
                    minimum=0, maximum=50, value=0, step=1, interactive=True, label="Top-k",
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=5.0, value=0.8, step=0.1, interactive=True, label="Temperature",
                )

        user_text.submit(run_generation, [user_text, top_p, temperature, top_k, max_new_tokens, performance], [model_output, performance])
        button_submit.click(run_generation, [user_text, top_p, temperature, top_k, max_new_tokens, performance], [model_output, performance])
        button_clear.click(reset_textbox, [user_text, model_output, performance], [user_text, model_output, performance])

    if __name__ == "__main__":
        demo.queue()
        try:
            demo.launch(height=800)
        except Exception:
            demo.launch(share=True, height=800)

    # If you are launching remotely, specify server_name and server_port
    # EXAMPLE: `demo.launch(server_name='your server name', server_port='server port in int')`
    # To learn more please refer to the Gradio docs: https://gradio.app/docs/
