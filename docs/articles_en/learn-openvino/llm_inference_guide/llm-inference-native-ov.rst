.. {#llm_inference_native_ov}

Inference with Native OpenVINO
===============================

To run Generative AI models using native OpenVINO APIs you need to follow regular **Convert -> Optimize -> Deploy** path with a few simplifications.

To convert a model from Hugging Face, you can use Optimum-Intel export feature that allows you to export model in the OpenVINO format without invoking conversion API and tools directly.
In this case, the conversion process is a bit more simplified. You can still use a regular conversion path if the model comes from outside of Hugging Face ecosystem, i.e., in source framework format (PyTorch, etc.)

Model optimization can be performed within Hugging Face or directly using NNCF as described in :doc:`Weight Compression <../../openvino-workflow/model-optimization-guide/weight-compression>`.

.. note::

   It is recommended to use models in 4-bit precision, as maintaining the model in its original precision may result in significantly decreased performance.

Inference code that uses native API cannot benefit from Hugging Face pipelines. You need to write your custom code or take it from the available examples. Below are some examples of popular Generative AI scenarios:

* In case of LLMs for text generation, you need to handle tokenization, inference and token selection loop, and de-tokenization. If token selection involves beam search, it also needs to be written.
* For image generation models, you need to make a pipeline that includes several model inferences: inference for source (e.g., text) encoder models, inference loop for diffusion process and inference for the decoding part. Scheduler code is also required.

To write such pipelines, you can follow the examples provided as part of OpenVINO:

* `Text generation C++ samples that support most popular models like LLaMA 2 <https://github.com/openvinotoolkit/openvino.genai/tree/master/text_generation/causal_lm/cpp>`__
* `OpenVINO Latent Consistency Model C++ image generation pipeline <https://github.com/openvinotoolkit/openvino.genai/tree/master/image_generation/lcm_dreamshaper_v7/cpp>`__
* `OpenVINO Stable Diffusion (with LoRA) C++ image generation pipeline <https://github.com/openvinotoolkit/openvino.genai/tree/master/image_generation/stable_diffusion_1_5/cpp>`__

To perform inference, models must be first converted to OpenVINO IR format using Hugging Face Optimum-Intel API.

An inference pipeline for a text generation LLM is set up in the following stages:

1.	Read and compile the model in OpenVINO IR.
2.	Pre-process text prompt with a tokenizer and set the result as model inputs.
3.	Run token generation loop.
4.	De-tokenize outputs.

Prerequisites
########################

Linux operating system (as of the current version).

**Installation**

1. Create a virtual environment

   .. code-block:: python

      python -m venv openvino_llm

   ``openvino_llm`` is an example name; you can choose any name for your environment.

2. Activate the virtual environment

   .. code-block:: python

      source openvino_llm/bin/activate

3. Install OpenVINO tokenizers and dependencies

   .. code-block:: python

      pip install optimum[openvino]


Convert Hugging Face tokenizer and model to OpenVINO IR format
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

**Convert Tokenizer**

`OpenVINO Tokenizers <https://github.com/openvinotoolkit/openvino_tokenizers>`__
come equipped with a CLI tool that facilitates the conversion of tokenizers
from either the Hugging Face Hub or those saved locally to the OpenVINO IR format:

.. code-block:: python

   convert_tokenizer microsoft/Llama2-7b-WhoIsHarryPotter --with-detokenizer -o openvino_tokenizer

In this example, the ``microsoft/Llama2-7b-WhoIsHarryPotter tokenizer`` is transformed from the Hugging
Face hub. You can substitute this tokenizer with one of your preference. You can also rename
the output directory (``openvino_tokenizer``).

**Convert Model**

The optimum-cli command can be used for converting a Hugging Face model to the OpenVINO IR model format.
Learn more in Loading an LLM with OpenVINO.

.. code-block:: python

   optimum-cli export openvino --convert-tokenizer --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 openvino_model

Full OpenVINO Text Generation Pipeline
######################################################################

1.	Import and Compile Models
+++++++++++++++++++++++++++++++++++++++

Use the model and tokenizer converted from the previous step:

.. code-block:: python

   import numpy as np
   from openvino import compile_model

   # Compile the tokenizer, model, and detokenizer using OpenVINO. These files are XML representations of the models optimized for OpenVINO
   compiled_tokenizer = compile_model("openvino_tokenizer.xml")
   compiled_model = compile_model("openvino_model.xml")
   compiled_detokenizer = compile_model("openvino_detokenizer.xml")

2.	Tokenize and Transform Input
+++++++++++++++++++++++++++++++++++++++

Tokenization is a mandatory step in the process of generating text using LLMs. Tokenization
converts the input text into a sequence of tokens, which are essentially the format that the
model can understand and process. The input text string must be tokenized and set up in the
structure expected by the model before running inference.

.. code-block:: python

   text_input = ["Quick brown fox was"]
   ov_input = compiled_tokenizer(text_input)

3.	Generate Tokens
+++++++++++++++++++++++++++++++++++++++

The core of text generation lies in the inference and token selection loop. In each iteration
of this loop, the model runs inference on the input sequence, generates and selects a new token,
and appends it to the existing sequence.

.. code-block:: python

   # Define the number of new tokens to generate
   new_tokens_size = 10

   # Determine the size of the existing prompt
   prompt_size = ov_input["input_ids"].shape[-1]

   # Prepare the input dictionary for the model
   # It combines existing tokens with additional space for new tokens
   input_dict = {
     output.any_name: np.hstack([tensor, np.zeros(shape=(1, new_tokens_size), dtype=np.int_)])
     for output, tensor in ov_input.items()
   }

   # Generate new tokens iteratively
   for idx in range(prompt_size, prompt_size + new_tokens_size):
       # Get output from the model
       output = compiled_model(input_dict)["token_ids"]
       # Update the input_ids with newly generated token
       input_dict["input_ids"][:, idx] = output[:, idx - 1]
       # Update the attention mask to include the new token
       input_dict["attention_mask"][:, idx] = 1

4.	Decode and Display Output
+++++++++++++++++++++++++++++++++++++++

The final step in the process is de-tokenization, where the sequence of token IDs generated by
the model is converted back into human-readable text.
This step is essential for interpreting the model's output.

.. code-block:: python

   # Extract token IDs for the final output
   ov_token_ids = input_dict["input_ids"]
   # Decode the model output back to string
   ov_output = compiled_detokenizer(ov_token_ids)["string_output"]
   print(f"OpenVINO output string: `{ov_output}`")

.. code-block:: python

   # Example output:
   ['<s> Quick brown fox was walking through the forest. He was looking for something']


Additional Resources
####################

* `Text generation C++ samples that support most popular models like LLaMA 2 <https://github.com/openvinotoolkit/openvino.genai/tree/master/text_generation/causal_lm/cpp>`__
* `OpenVINO GenAI Repo <https://github.com/openvinotoolkit/openvino.genai>`__
* `OpenVINO Tokenizers <https://github.com/openvinotoolkit/openvino_tokenizers>`__
* `Neural Network Compression Framework <https://github.com/openvinotoolkit/nncf>`__
* :doc:`Stateful Models Low-Level Details <../../openvino-workflow/running-inference/stateful-models>`
* :doc:`Working with Textual Data <../../openvino-workflow/running-inference/string-tensors>`


