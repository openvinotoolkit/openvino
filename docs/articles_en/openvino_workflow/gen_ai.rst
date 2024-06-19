.. {#gen_ai_guide}

Optimize and Deploy Generative AI Models
========================================


Generative AI is an innovative technique that creates new data, such as text, images, video,
or audio, using neural networks. OpenVINO accelerates Generative AI use cases as they mostly
rely on model inference, allowing for faster development and better performance. When it
comes to generative models, OpenVINO supports:

* Conversion, optimization and inference for text, image and audio generative models, for
  example, Llama 2, MPT, OPT, Stable Diffusion, Stable Diffusion XL, etc.
* Int8 weight compression for text generation models.
* Storage format reduction (fp16 precision for non-compressed models and int8 for compressed
  models).
* Inference on CPU and GPU platforms, including integrated Intel® Processor Graphics,
  discrete Intel® Arc™ A-Series Graphics, and discrete Intel® Data Center GPU Flex Series.


OpenVINO offers two main paths for Generative AI use cases:

* Using OpenVINO as a backend for Hugging Face frameworks (transformers, diffusers) through
  the `Optimum Intel <https://huggingface.co/docs/optimum/intel/inference>`__ extension.
* Using OpenVINO native APIs (Python and C++) with custom pipeline code.


In both cases, OpenVINO runtime and tools are used, the difference is mostly in the preferred
API and the final solution's footprint. Native APIs enable the use of generative models in
C++ applications, ensure minimal runtime dependencies, and minimize application footprint.
The Native APIs approach requires the implementation of glue code (generation loop, text
tokenization, or scheduler functions), which is hidden within Hugging Face libraries for a
better developer experience.

It is recommended to start with Hugging Face frameworks. Experiment with different models and
scenarios to find your fit, and then consider converting to OpenVINO native APIs based on your
specific requirements.

Optimum Intel provides interfaces that enable model optimization (weight compression) using
`Neural Network Compression Framework (NNCF) <https://github.com/openvinotoolkit/nncf>`__,
and export models to the OpenVINO model format for use in native API applications.

The table below summarizes the differences between Hugging Face and Native APIs approaches.

.. list-table::
   :widths: 20 25 55
   :header-rows: 1

   * -
     - Hugging Face through OpenVINO
     - OpenVINO Native API
   * - Model support
     - Broad set of Models
     - Broad set of Models
   * - APIs
     - Python (Hugging Face API)
     - Python, C++ (OpenVINO API)
   * - Model Format
     - Source Framework / OpenVINO
     - OpenVINO
   * - Inference code
     - Hugging Face based
     - Custom inference pipelines
   * - Additional dependencies
     - Many Hugging Face dependencies
     - Lightweight (e.g. numpy, etc.)
   * - Application footprint
     - Large
     - Small
   * - Pre/post-processing and glue code
     - Available at Hugging Face out-of-the-box
     - OpenVINO samples and notebooks
   * - Performance
     - Good
     - Best


Running Generative AI Models using Hugging Face Optimum Intel
##############################################################

Prerequisites
+++++++++++++++++++++++++++

* Create a Python environment.
* Install Optimum Intel:

.. code-block:: console

    pip install optimum[openvino,nncf]


To start using OpenVINO as a backend for Hugging Face, change the original Hugging Face code in two places:

.. code-block:: diff

    -from transformers import AutoModelForCausalLM
    +from optimum.intel import OVModelForCausalLM

    model_id = "meta-llama/Llama-2-7b-chat-hf"
    -model = AutoModelForCausalLM.from_pretrained(model_id)
    +model = OVModelForCausalLM.from_pretrained(model_id, export=True)


After that, you can call ``save_pretrained()`` method to save model to the folder in the OpenVINO
Intermediate Representation and use it further.

.. code-block:: python

    model.save_pretrained(model_dir)


Alternatively, you can download and convert the model using CLI interface:
``optimum-cli export openvino --model meta-llama/Llama-2-7b-chat-hf llama_openvino``.
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

   OpenVINO also supports 4-bit models from Hugging Face `Transformers <https://github.com/huggingface/transformers>`__
   library optimized with `GPTQ <https://github.com/PanQiWei/AutoGPTQ>`__. In this case,
   there is no need for an additional model optimization step because model conversion
   will automatically preserve the INT4 optimization results, allowing model inference
   to benefit from it.

Below are some examples of using Optimum-Intel for model conversion and inference:

* `Stable Diffusion v2.1 using Optimum-Intel OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/236-stable-diffusion-v2/236-stable-diffusion-v2-optimum-demo.ipynb>`__
* `Image generation with Stable Diffusion XL and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/248-stable-diffusion-xl/248-stable-diffusion-xl.ipynb>`__
* `Instruction following using Databricks Dolly 2.0 and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/240-dolly-2-instruction-following/240-dolly-2-instruction-following.ipynb>`__
* `Create an LLM-powered Chatbot using OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot/254-llm-chatbot.ipynb>`__

Stateful Model Optimization
+++++++++++++++++++++++++++

When you use the ``OVModelForCausalLM`` class, the model is transformed into a stateful form by default for optimization.
This transformation improves inference performance and decreases runtime memory usage in long running text generation tasks.
It is achieved by hiding the model's inputs and outputs that represent past KV-cache tensors, and handling them inside the model in a more efficient way.
This feature is activated automatically for many supported text generation models, while unsupported models remain in a regular, stateless form.

Model usage remains the same for stateful and stateless models with the Optimum-Intel API, as KV-cache is handled internally by text-generation API of Transformers library.
The model's form matters when an OpenVINO IR model is exported from Optimum-Intel and used in an application with the native OpenVINO API.
This is because stateful and stateless models have a different number of inputs and outputs.
Learn more about the `native OpenVINO API <Running-Generative-AI-Models-using-Native-OpenVINO-APIs>`__.


Working with Models Tuned with LoRA
++++++++++++++++++++++++++++++++++++

Low-rank Adaptation (LoRA) is a popular method to tune Generative AI models to a downstream task or custom data. However, it requires some extra steps to be done for efficient deployment using the Hugging Face API. Namely, the trained adapters should be fused into the baseline model to avoid extra computation. This is how it can be done for Large Language Models (LLMs):

.. code-block:: python

    model_id = "meta-llama/Llama-2-7b-chat-hf"
    lora_adaptor = "./lora_adaptor"

    model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=True)
    model = PeftModelForCausalLM.from_pretrained(model, lora_adaptor)
    model.merge_and_unload()
    model.get_base_model().save_pretrained("fused_lora_model")


Now the model can be converted to OpenVINO using Optimum Intel Python API or CLI interfaces mentioned above.

Running Generative AI Models using Native OpenVINO APIs
########################################################

To run Generative AI models using native OpenVINO APIs, you need to follow regular **Convert -> Optimize -> Deploy** path with a few simplifications.

The recommended way for converting a Hugging Face model is to use the Optimum-Intel export feature. This feature enables model export in OpenVINO format without directly invoking conversion API and tools, as demonstrated above.
The conversion process is significantly simplified as Optimum-Intel provides the necessary conversion parameters. These parameters are often model-specific and require knowledge of various model input properties.

Moreover, Optimum-Intel applies several model optimizations, such as weight compression and using stateful form by default, that further simplify the model exporting flow.
You can still use the regular conversion path if the model comes from outside the Hugging Face ecosystem, such as in its source framework format (PyTorch, TensorFlow, etc.).

Model optimization can be performed within Hugging Face or directly using NNCF as described in the :doc:`weight compression guide <weight_compression>`.

Inference code that uses native API cannot benefit from Hugging Face pipelines. You need to write your custom code or take it from the available examples. Below are some examples of popular Generative AI scenarios:

* In case of LLMs for text generation, you need to handle tokenization, inference and token sampling, and de-tokenization. If token sampling involves beam search, you need to implement it as well. This is covered in details by `C++ Text Generation Samples <https://github.com/openvinotoolkit/openvino.genai/tree/master/text_generation/causal_lm/cpp>`__.
* For image generation models, you need to make a pipeline that includes several model inferences: inference for source (for example, text) encoder models, inference loop for diffusion process and inference for decoding part. Scheduler code is also required. `C++ Implementation of Stable Diffusion <https://github.com/openvinotoolkit/openvino.genai/tree/master/image_generation/stable_diffusion_1_5/cpp>`__ is a good reference point.


Additional Resources
#####################

* `Optimum Intel documentation <https://huggingface.co/docs/optimum/intel/inference>`__
* :doc:`LLM Weight Compression <weight_compression>`
* `Neural Network Compression Framework <https://github.com/openvinotoolkit/nncf>`__
* `GenAI Pipeline Repository <https://github.com/openvinotoolkit/openvino.genai>`__
* `OpenVINO Tokenizers <https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/custom_operations/user_ie_extensions/tokenizer/python>`__
* :doc:`Stateful Models Low-Level Details <openvino_docs_OV_UG_stateful_models_intro>`
* :doc:`Working with Textual Data <openvino_docs_OV_UG_string_tensors>`
