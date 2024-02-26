.. {#gen_ai_guide}

Loading and Optimizing LLMs with Optimum Intel
=================================================

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
##############################################################

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

Obtaining OpenVINO Model Object
##############################################################

When you use Intel Optimum for loading, the resulting model is a Hugging
Face model with additional functionalities provided by Optimum.
The model object created in the snippets above is not a native OpenVINO IR model
but rather a Hugging Face model adapted to work with OpenVINO's optimizations.

If you need to access the underlying OpenVINO model object directly, you
can do so through a specific attribute of the Optimum Intel model named ``model``.

To access this native OpenVINO model object, you can assign it to a new variable like this:

.. code-block:: python

    openvino_model = model.model

The first model refers to the Optimum Intel `model` you loaded, and the `.model`
accesses the native OpenVINO model object within it. Now, `openvino_model` holds
the native OpenVINO model, allowing you to interact with it directly,
as you would with a standard OpenVINO model. You can compress the model using `NNCF <https://github.com/openvinotoolkit/nncf>`__
and infer it with a custom OpenVINO pipeline. For more information, see the :doc:`LLM Weight Compression <weight_compression>` page.

If you want to work with Native OpenVINO after loading the model with Optimum Intel,
it is recommended to disable model compilation in the loading function.
Set the compile attribute to False while loading the model:

.. code-block:: python

    model = OVModelForCausalLM.from_pretrained(model_id, export=True, compile=False)

Converting a Hugging Face Model to OpenVINO IR
##############################################################

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

`NNCF <https://github.com/openvinotoolkit/nncf>`__ also provides 4-bit weight compression,
which is supported by OpenVINO. It can be applied to Optimum objects as follows:

.. code-block:: python

    from nncf import compress_weights, CompressWeightsMode

    model = OVModelForCausalLM.from_pretrained(model_id, export=True, load_in_8bit=False)
    model.model = compress_weights(model.model, mode=CompressWeightsMode.INT4_SYM, group_size=128, ratio=0.8)


The optimized model can be saved as usual with a call to ``save_pretrained()``.
For more details on compression options, refer to the :doc:`weight compression guide <weight_compression>`.

.. note::

   OpenVINO also supports 4-bit models from Hugging Face `Transformers <https://github.com/huggingface/transformers>`__ library optimized
   with `GPTQ <https://github.com/PanQiWei/AutoGPTQ>`__. In this case, there is no need for an additional model optimization step because model conversion will automatically preserve the INT4 optimization results, allowing model inference to benefit from it.

Below are some examples of using Optimum-Intel for model conversion and inference:

* `Stable Diffusion v2.1 using Optimum-Intel OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/236-stable-diffusion-v2/236-stable-diffusion-v2-optimum-demo.ipynb>`__
* `Image generation with Stable Diffusion XL and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/248-stable-diffusion-xl/248-stable-diffusion-xl.ipynb>`__
* `Instruction following using Databricks Dolly 2.0 and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/240-dolly-2-instruction-following/240-dolly-2-instruction-following.ipynb>`__
* `Create an LLM-powered Chatbot using OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot/254-llm-chatbot.ipynb>`__


Stateful Model Optimization
############################

When you use the ``OVModelForCausalLM`` class, the model is transformed into a stateful form by default for optimization.
This transformation improves inference performance and decreases runtime memory usage in long running text generation tasks.
It is achieved by hiding the model's inputs and outputs that represent past KV-cache tensors, and handling them inside the model in a more efficient way.
This feature is activated automatically for many supported text generation models, while unsupported models remain in a regular, stateless form.

Model usage remains the same for stateful and stateless models with the Optimum-Intel API, as KV-cache is handled internally by text-generation API of Transformers library.
The model's format matters when an OpenVINO IR model is exported from Optimum-Intel and used in an application with the native OpenVINO API.
This is because stateful and stateless models have a different number of inputs and outputs.
Learn more about the `native OpenVINO API <Running-Generative-AI-Models-using-Native-OpenVINO-APIs>`__.

Enabling OpenVINO Runtime Optimizations
#########################################

OpenVINO runtime provides a set of optimizations for more efficient LLM inference. This includes **Dynamic quantization** of activations of 4/8-bit quantized MatMuls and **KV-cache quantization**.

* **Dynamic quantization** enables quantization of activations of MatMul operations that have 4 or 8-bit quantized weights (see :doc:`LLM Weight Compression <weight_compression>`).
  It improves inference latency and throughput of LLMs, though it may cause insignificant deviation in generation accuracy.  Quantization is performed in a
  group-wise manner, with configurable group size. It means that values in a group share quantization parameters. Larger group sizes lead to faster inference but lower accuracy. Recommended group size values are: ``32``, ``64``, or ``128``. To enable Dynamic quantization, use the corresponding
  inference property as follows:


  .. code-block:: python

      model = OVModelForCausalLM.from_pretrained(
          model_path,
          ov_config={"DYNAMIC_QUANTIZATION_GROUP_SIZE": "32", "PERFORMANCE_HINT": "LATENCY"}
      )



* **KV-cache quantization** allows lowering the precision of Key and Value cache in LLMs. This helps reduce memory consumption during inference, improving latency and throughput. KV-cache can be quantized into the following precisions:
  ``u8``, ``bf16``, ``f16``.  If ``u8`` is used, KV-cache quantization is also applied in a group-wise manner. Thus, it can use ``DYNAMIC_QUANTIZATION_GROUP_SIZE`` value if defined.
  Otherwise, the group size ``32`` is used by default. KV-cache quantization can be enabled as follows:


  .. code-block:: python

      model = OVModelForCausalLM.from_pretrained(
          model_path,
          ov_config={"KV_CACHE_PRECISION": "u8", "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32", "PERFORMANCE_HINT": "LATENCY"}
      )


.. note::

  Currently, both Dynamic quantization and KV-cache quantization are available for CPU device.


Working with Models Tuned with LoRA
#########################################

Low-rank Adaptation (LoRA) is a popular method to tune Generative AI models to a downstream task
or custom data. However, it requires some extra steps to be done for efficient deployment using
the Hugging Face API. Namely, the trained adapters should be fused into the baseline model to
avoid extra computation. This is how it can be done for LLMs:

.. code-block:: python

    model_id = "meta-llama/Llama-2-7b-chat-hf"
    lora_adaptor = "./lora_adaptor"

    model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=True)
    model = PeftModelForCausalLM.from_pretrained(model, lora_adaptor)
    model.merge_and_unload()
    model.get_base_model().save_pretrained("fused_lora_model")


Now the model can be converted to OpenVINO using Optimum Intel Python API or CLI interfaces mentioned above.


Additional Resources
#####################

* `Optimum Intel documentation <https://huggingface.co/docs/optimum/intel/inference>`__
* :doc:`LLM Weight Compression <weight_compression>`
* `Neural Network Compression Framework <https://github.com/openvinotoolkit/nncf>`__
* `GenAI Pipeline Repository <https://github.com/openvinotoolkit/openvino.genai>`__
* `OpenVINO Tokenizers <https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/custom_operations/user_ie_extensions/tokenizer/python>`__
* :doc:`Stateful Models Low-Level Details <openvino_docs_OV_UG_stateful_models_intro>`
* :doc:`Working with Textual Data <openvino_docs_OV_UG_string_tensors>`
