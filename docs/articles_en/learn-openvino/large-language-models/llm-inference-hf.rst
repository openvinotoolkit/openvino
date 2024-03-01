.. {#llm_inference}

LLM Inference with Hugging Face and Optimum Intel
=====================================================

OpenVINO provides optimized inference for Large Language Models (LLMs). This page explains how
to perform LLM inference using either the Hugging Face Optimum Intel API or the native OpenVINO API.

Before performing inference, a model must be converted into OpenVINO IR format. This conversion
occurs automatically when loading an LLM from Hugging Face with the Optimum Intel library.
For more information on how to load LLMs in OpenVINO, see :doc:`Loading an LLM to OpenVINO <gen_ai_guide>`.

Loading and Optimizing LLMs with Optimum Intel
############################################################

The steps below show how to load LLMs from Hugging Face using Optimum Intel.
They also show how to convert models into OpenVINO IR format so they can be optimized
by NNCF and used with other OpenVINO tools.

Prerequisites
+++++++++++++++++++++++++++

* Create a Python environment by following the instructions on the :doc:`Install OpenVINO PIP <openvino_docs_install_guides_overview>` page.
* Install the necessary dependencies for Optimum Intel:

.. code-block:: console

    pip install optimum[openvino,nncf]

Loading a Hugging Face Model to Optimum Intel
++++++++++++++++++++++++++++++++++++++++++++++++++++++

To start using OpenVINO as a backend for Hugging Face, change the original Hugging Face code in two places:

.. code-block:: diff

    -from transformers import AutoModelForCausalLM
    +from optimum.intel import OVModelForCausalLM

    model_id = "meta-llama/Llama-2-7b-chat-hf"
    -model = AutoModelForCausalLM.from_pretrained(model_id)
    +model = OVModelForCausalLM.from_pretrained(model_id, export=True)


Instead of using ``AutoModelForCasualLM`` from the Hugging Face transformers library,
switch to ``OVModelForCasualLM`` from the optimum.intel library. This change enables
you to use OpenVINO's optimization features. You may also use other AutoModel types,
such as ``OVModelForSeq2SeqLM``, though this guide will focus on CausalLM.

By setting the parameter ``export=True``, the model is converted to OpenVINO IR format on the fly.

After that, you can call ``save_pretrained()`` method to save model to the folder in the OpenVINO
Intermediate Representation and use it further.

.. code-block:: python

    model.save_pretrained("ov_model")

This will create a new folder called `ov_model` with the LLM in OpenVINO IR format inside.
You can change the folder and provide another model directory instead of `ov_model`.

Once the model is saved, you can load it with the following command:

.. code-block:: python

    model = OVModelForCausalLM.from_pretrained("ov_model")

Converting a Hugging Face Model to OpenVINO IR
++++++++++++++++++++++++++++++++++++++++++++++++++++++

The optimum-cli tool allows you to convert models from Hugging Face to
the OpenVINO IR format:

.. code-block:: python

    optimum-cli export openvino --model <MODEL_NAME> <NEW_MODEL_NAME>

If you want to convert the `Llama 2` model from Hugging Face to an OpenVINO IR
model and name it `ov_llama_2`, the command would look like this:

.. code-block:: python

    optimum-cli export openvino --model meta-llama/Llama-2-7b-chat-hf ov_llama_2

In this case, you can load the converted model in OpenVINO representation directly from the disk:

.. code-block:: python

    model_id = "llama_openvino"
    model = OVModelForCausalLM.from_pretrained(model_id)


By default, inference will run on CPU. To select a different inference device, for example, GPU,
add ``device="GPU"`` to the ``from_pretrained()`` call. To switch to a different device after
the model has been loaded, use the ``.to()`` method. The device naming convention is the same
as in OpenVINO native API:

.. code-block:: python

    model.to("GPU")


Optimum-Intel API also provides out-of-the-box model optimization through weight compression
using NNCF which substantially reduces the model footprint and inference latency:

.. code-block:: python

    model = OVModelForCausalLM.from_pretrained(model_id, export=True, load_in_8bit=True)


Weight compression is applied by default to models larger than one billion parameters and is
also available for CLI interface as the ``--int8`` option.

.. note::

   8-bit weight compression is enabled by default for models larger than 1 billion parameters.

`Optimum Intel <https://huggingface.co/docs/optimum/intel/inference>`__ also provides 4-bit weight
compression with ``OVWeightQuantizationConfig`` class to control weight quantization parameters.


.. code-block:: python

    from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig
    import nncf

    model = OVModelForCausalLM.from_pretrained(
        model_id,
        export=True,
        quantization_config=OVWeightQuantizationConfig(bits=4, asym=True, ratio=0.8, dataset="ptb"),
    )


The optimized model can be saved as usual with a call to ``save_pretrained()``.
For more details on compression options, refer to the :doc:`weight compression guide <weight_compression>`.

.. note::

   OpenVINO also supports 4-bit models from Hugging Face `Transformers <https://github.com/huggingface/transformers>`__ library optimized
   with `GPTQ <https://github.com/PanQiWei/AutoGPTQ>`__. In this case, there is no need for an additional model optimization step because model conversion will automatically preserve the INT4 optimization results, allowing model inference to benefit from it.

Below are some examples of using Optimum-Intel for model conversion and inference:

* `Instruction following using Databricks Dolly 2.0 and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/240-dolly-2-instruction-following/240-dolly-2-instruction-following.ipynb>`__
* `Create an LLM-powered Chatbot using OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot/254-llm-chatbot.ipynb>`__

.. note::

  Optimum-Intel can be used for other generative AI models. See `Stable Diffusion v2.1 using Optimum-Intel OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/236-stable-diffusion-v2/236-stable-diffusion-v2-optimum-demo.ipynb>`__ and `Image generation with Stable Diffusion XL and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/248-stable-diffusion-xl/248-stable-diffusion-xl.ipynb>`__ for more examples.

Inference with Hugging Face
############################


Installation
+++++++++++++++++++++++++++

1. Create a virtual environment. ``openvino_llm`` is an example name; you can choose any name for your environment.

.. code-block:: python

  python -m venv openvino_llm

2. Activate the virtual environment

.. code-block:: python

  source openvino_llm/bin/activate

3. Install the libraries

.. code-block:: python

  pip install transformers optimum[openvino,nncf]

Inference Example
+++++++++++++++++++++++++++

For Hugging Face models, the ``AutoTokenizer`` and the ``pipeline`` function are used to create
an inference pipeline. This setup allows for easy text processing and model interaction:

.. code-block:: python

  from optimum.intel import OVModelForCausalLM
  # new imports for inference
  from transformers import AutoTokenizer

  # load the model
  model_id = "meta-llama/Llama-2-7b-chat-hf"
  model = OVModelForCausalLM.from_pretrained(model_id, export=True)

  # inference
  prompt = "The weather is:"
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  inputs = tokenizer(prompt, return_tensors="pt")

  outputs = model.generate(**inputs, max_new_tokens=50)
  print(tokenizer.decode(outputs[0], skip_special_tokens=True))

.. note::

  Converting LLMs on the fly every time to OpenVINO IR is a resource intensive task.
  It is a good practice to convert the model once, save it in a folder and load it for inference.

By default, inference will run on CPU. To switch to a different device, the ``device`` attribute
from the ``from_pretrained`` function can be used. The device naming convention is the
same as in OpenVINO native API:

.. code-block:: python

  model = OVModelForCausalLM.from_pretrained(model_id, export=True, device="GPU")

For more information on how to run text generation with Huggin Face APIs, see their documentation:

* `Optimum Intel documentation <https://huggingface.co/docs/optimum/intel/inference>`__
* :doc:`LLM Weight Compression <weight_compression>`
* `Hugging Face Transformers <https://huggingface.co/docs/transformers/index>`__
* `Generation with LLMs <https://huggingface.co/docs/transformers/llm_tutorial>`__
*	`Pipeline class <https://huggingface.co/docs/transformers/main_classes/pipelines>`__
* `GenAI Pipeline Repository <https://github.com/openvinotoolkit/openvino.genai>`__
* `OpenVINO Tokenizers <https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/custom_operations/user_ie_extensions/tokenizer/python>`__
