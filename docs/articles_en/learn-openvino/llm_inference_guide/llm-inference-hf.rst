Run LLMs with Hugging Face and Optimum Intel
===============================================================================================

.. meta::
   :description: Learn how to use the native OpenVINO package to execute LLM models.


The steps below show how to load and infer LLMs from
`Hugging Face <https://huggingface.co/models>`__ using Optimum Intel. They also show how to
convert models into OpenVINO IR format so they can be optimized by NNCF and used with
other OpenVINO tools.

Prerequisites
############################################################

* Create a Python environment by following the instructions on the
  :doc:`Install OpenVINO PIP <../../get-started/install-openvino>` page.
* Install the necessary dependencies for Optimum Intel:

.. code-block:: console

   pip install optimum[openvino,nncf]

Loading a Hugging Face Model to Optimum Intel
############################################################

To start using OpenVINO as a backend for Hugging Face, change the original Hugging Face
code in two places:

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

It is recommended to save model to disk after conversion using ``save_pretrained()`` and
loading it from disk at deployment time via ``from_pretrained()`` for better efficiency.

.. code-block:: python

   model.save_pretrained("ov_model")

This will create a new folder called `ov_model` with the LLM in OpenVINO IR format inside.
You can change the folder and provide another model directory instead of `ov_model`.

Once the model is saved, you can load it with the following command:

.. code-block:: python

   model = OVModelForCausalLM.from_pretrained("ov_model")


Converting a Hugging Face Model to OpenVINO IR
############################################################

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

Optimum-Intel API also provides out-of-the-box model optimization through weight compression
using NNCF which substantially reduces the model footprint and inference latency:

.. tab-set::

   .. tab-item:: CLI
      :sync: CLI

      .. code-block:: sh

         optimum-cli export openvino --model meta-llama/Llama-2-7b-chat-hf --weight-format int8 ov_llama_2

   .. tab-item:: API
      :sync: API

      .. code-block:: python

         model = OVModelForCausalLM.from_pretrained(model_id, export=True, load_in_8bit=True)

         # or if the model has been already converted
         model = OVModelForCausalLM.from_pretrained(model_path, load_in_8bit=True)

         # save the model after optimization
         model.save_pretrained(optimized_model_path)


Weight compression is applied by default to models larger than one billion parameters and is
also available for CLI interface as the ``--int8`` option.

.. note::

   8-bit weight compression is enabled by default for models larger than 1 billion parameters.

`Optimum Intel <https://huggingface.co/docs/optimum/intel/inference>`__ also provides 4-bit
weight compression with ``OVWeightQuantizationConfig`` class to control weight quantization
parameters.

.. tab-set::

   .. tab-item:: CLI
      :sync: CLI

      .. code-block:: python

         optimum-cli export openvino --model meta-llama/Llama-2-7b-chat-hf --weight-format int4 ov_llama_2

   .. tab-item:: API
      :sync: API

      .. code-block:: python

         from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig
         import nncf

         model = OVModelForCausalLM.from_pretrained(
             model_id,
             export=True,
             quantization_config=OVWeightQuantizationConfig(bits=4),
         )

         # or if the model has been already converted
         model = OVModelForCausalLM.from_pretrained(
             model_path,
             quantization_config=OVWeightQuantizationConfig(bits=4),
         )

         # use custom parameters for weight quantization
         model = OVModelForCausalLM.from_pretrained(
             model_path,
             quantization_config=OVWeightQuantizationConfig(bits=4, asym=True, ratio=0.8, dataset="ptb"),
         )

         # save the model after optimization
         model.save_pretrained(optimized_model_path)


.. note::

   Optimum-Intel has a predefined set of weight quantization parameters for popular models,
   such as ``meta-llama/Llama-2-7b`` or ``Qwen/Qwen-7B-Chat``. These parameters are used by
   default only when ``bits=4`` is specified in the config.

   For more details on compression options, refer to the corresponding `Optimum documentation <https://huggingface.co/docs/optimum/en/intel/openvino/optimization#4-bit>`__.
   For native NNCF weight quantization options, refer to the :doc:`weight compression guide <../../openvino-workflow/model-optimization-guide/weight-compression>`.

   OpenVINO also supports 4-bit models from Hugging Face `Transformers <https://github.com/huggingface/transformers>`__
   library optimized with `GPTQ <https://github.com/PanQiWei/AutoGPTQ>`__. In this case, there
   is no need for an additional model optimization step because model conversion will
   automatically preserve the INT4 optimization results, allowing model inference to benefit
   from it.

Below are some examples of using Optimum-Intel for model conversion and inference:

* `Instruction following using Databricks Dolly 2.0 and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/dolly-2-instruction-following/dolly-2-instruction-following.ipynb>`__
* `Create an LLM-powered Chatbot using OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llm-chatbot/llm-chatbot.ipynb>`__

.. note::

   Optimum-Intel can be used for other generative AI models. See
   `Stable Diffusion v2.1 using Optimum-Intel OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/stable-diffusion-v2/stable-diffusion-v2-optimum-demo.ipynb>`__
   and
   `Image generation with Stable Diffusion XL and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/stable-diffusion-xl/stable-diffusion-xl.ipynb>`__
   for more examples.

Inference Example
############################################################

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
   It is a good practice to convert the model once, save it in a folder and load it for
   inference.

By default, inference will run on CPU. To select a different inference device, for example, GPU,
add ``device="GPU"`` to the ``from_pretrained()`` call. To switch to a different device after
the model has been loaded, use the ``.to()`` method. The device naming convention is the same
as in OpenVINO native API:

.. code-block:: python

   model.to("GPU")

.. _enabling-runtime-optimizations:

Enabling OpenVINO Runtime Optimizations
############################################################

OpenVINO runtime provides a set of optimizations for more efficient LLM inference. This
includes **Dynamic quantization** of activations of 4/8-bit quantized MatMuls and
**KV-cache quantization**.

* **Dynamic quantization** enables quantization of activations of MatMul operations
  that have 4 or 8-bit quantized weights (see
  :doc:`LLM Weight Compression <../../openvino-workflow/model-optimization-guide/weight-compression>`).
  It improves inference latency and throughput of LLMs, though it may cause
  insignificant deviation in generation accuracy.  Quantization is performed in a group-wise
  manner, with configurable group size. It means that values in a group share quantization
  parameters. Larger group sizes lead to faster inference but lower accuracy. Recommended
  group size values are ``0``, ``32``, ``64``, or ``128``. Dynamic quantization is enabled **by
  default** on the CPU device. To disable dynamic quantization you can either:

  * **(Primary Option)** Set ``DYNAMIC_QUANTIZATION_GROUP_SIZE`` to the ``0`` value.
  * Switch execution mode from the ``PERFORMANCE mode`` to the ``ACCURACY mode``. However, this
    option affects inference precision as well. You can learn more about both: ``PERFORMANCE``
    and ``ACCURACY`` modes by following the :ref:`Precision Control Guide <execution-mode>`.

  To change a group size value (e.g. to ``64``), you need to execute the following code:

  .. code-block:: python

     model = OVModelForCausalLM.from_pretrained(
         model_path,
         ov_config={"DYNAMIC_QUANTIZATION_GROUP_SIZE": "64"}
     )

  .. note::

     As of release 2024.3, dynamic quantization is not enabled for BF16 inference.


* **KV-cache quantization** allows lowering the precision of Key and Value cache in LLMs.
  This helps reduce memory consumption during inference, improving latency and throughput.
  KV-cache can be quantized into the following precisions: ``u8``, ``bf16``, ``f16``.
  If ``u8`` is used, KV-cache quantization is also applied in a group-wise manner. Thus,
  it can use ``DYNAMIC_QUANTIZATION_GROUP_SIZE`` value if defined. Otherwise, the group
  size ``32`` is used by default. KV-cache quantization can be enabled as follows:


  .. code-block:: python

     model = OVModelForCausalLM.from_pretrained(
         model_path,
         ov_config={"KV_CACHE_PRECISION": "u8", "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32", "PERFORMANCE_HINT": "LATENCY"}
     )

.. note::

   Currently, both Dynamic quantization and KV-cache quantization are available for CPU device.


Working with Models Tuned with LoRA
#########################################

Low-rank Adaptation (LoRA) is a popular method to tune Generative AI models to a downstream
task or custom data. However, it requires some extra steps to be done for efficient deployment
using the Hugging Face API. Namely, the trained adapters should be fused into the baseline
model to avoid extra computation. This is how it can be done for LLMs:

.. code-block:: python

   model_id = "meta-llama/Llama-2-7b-chat-hf"
   lora_adaptor = "./lora_adaptor"
   model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=True)
   model = PeftModelForCausalLM.from_pretrained(model, lora_adaptor)
   model.merge_and_unload()
   model.get_base_model().save_pretrained("fused_lora_model")

Now the model can be converted to OpenVINO using Optimum Intel Python API or CLI interfaces
mentioned above.

Execution on CPU device
##########################

As mentioned in the :ref:`Composability of different threading runtimes <Composability_of_different_threading_runtimes>` section, OpenVINO's default threading runtime,
oneTBB, keeps CPU cores active for a while after inference is done. When using Optimum Intel Python API,
it calls Torch (via HF transformers) for postprocessing, such as beam search or gready search.
Torch uses OpenMP for threading, OpenMP needs to wait for CPU cores that are kept active by
oneTBB. By default, OpenMP uses the `busy-wait <https://gcc.gnu.org/onlinedocs/libgomp/GOMP_005fSPINCOUNT.html>`__ which can delay the next OpenVINO inference as well.

It is recommended to:

* Limit the number of CPU threads used by Torch with `torch.set_num_threads <https://pytorch.org/docs/stable/generated/torch.set_num_threads.html>`__.
* Set the environment variable `OMP_WAIT_POLICY <https://gcc.gnu.org/onlinedocs/libgomp/OMP_005fWAIT_005fPOLICY.html>`__ to `PASSIVE`, which disables OpenMP `busy-wait <https://gcc.gnu.org/onlinedocs/libgomp/GOMP_005fSPINCOUNT.html>`__.

Additional Resources
#####################

* `Optimum Intel documentation <https://huggingface.co/docs/optimum/intel/inference>`__
* :doc:`LLM Weight Compression <../../openvino-workflow/model-optimization-guide/weight-compression>`
* `Neural Network Compression Framework <https://github.com/openvinotoolkit/nncf>`__
* `Hugging Face Transformers <https://huggingface.co/docs/transformers/index>`__
* `Generation with LLMs <https://huggingface.co/docs/transformers/llm_tutorial>`__
*	`Pipeline class <https://huggingface.co/docs/transformers/main_classes/pipelines>`__
* `GenAI Pipeline Repository <https://github.com/openvinotoolkit/openvino.genai>`__
* `OpenVINO Tokenizers <https://github.com/openvinotoolkit/openvino_tokenizers>`__