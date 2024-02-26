.. {#llm_inference}

LLM Inference with Hugging Face and Optimum Intel
=====================================================

OpenVINO provides optimized inference for Large Language Models (LLMs). This page explains how
to perform LLM inference using either the Hugging Face Optimum Intel API or the native OpenVINO API.

Before performing inference, a model must be converted into OpenVINO IR format. This conversion
occurs automatically when loading an LLM from Hugging Face with the Optimum Intel library.
For more information on how to load LLMs in OpenVINO, see :doc:`Loading an LLM to OpenVINO <gen_ai_guide>`.

Inference with Hugging Face
############################


**Installation**

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

* `Hugging Face Transformers <https://huggingface.co/docs/transformers/index>`__
* `Generation with LLMs <https://huggingface.co/docs/transformers/llm_tutorial>`__
*	`Pipeline class <https://huggingface.co/docs/transformers/main_classes/pipelines>`__