Generative AI workflow
========================================

.. meta::
   :description: learn how to use OpenVINO to run generative AI models.


.. toctree::
   :maxdepth: 1
   :hidden:

   Generative Model Preparation <llm_inference_guide/genai-model-preparation>
   Inference with OpenVINO GenAI <llm_inference_guide/genai-guide>
   Inference with Optimum Intel <llm_inference_guide/llm-inference-hf>
   Generative AI with Base OpenVINO (not recommended) <llm_inference_guide/llm-inference-native-ov>
   OpenVINO Tokenizers <llm_inference_guide/ov-tokenizers>



Generative AI is a specific area of Deep Learning models used for producing new and “original”
data, based on input in the form of image, sound, or natural language text. Due to their
complexity and size, generative AI pipelines are more difficult to deploy and run efficiently.
OpenVINO simplifies the process and ensures high-performance integrations, with the following
options:

.. tab-set::

   .. tab-item:: OpenVINO GenAI

      | - Suggested for production deployment for the supported use cases.
      | - Smaller footprint and fewer dependencies.
      | - More optimization and customization options.
      | - Available in both Python and C++.
      | - A limited set of supported use cases.

      :doc:`Install the OpenVINO GenAI package <../get-started/install-openvino/install-openvino-genai>`
      and run generative models out of the box. With custom
      API and tokenizers, among other components, it manages the essential tasks such as the
      text generation loop, tokenization, and scheduling, offering ease of use and high
      performance.

   .. tab-item:: Hugging Face integration

      | - Suggested for prototyping and, if the use case is not covered by OpenVINO GenAI, production.
      | - Bigger footprint and more dependencies.
      | - Limited customization due to Hugging Face dependency.
      | - Not usable for C++ applications.
      | - A very wide range of supported models.

      Using Optimum Intel is a great way to experiment with different models and scenarios,
      thanks to a simple interface for the popular API and infrastructure offered by Hugging Face.
      It also enables weight compression with
      `Neural Network Compression Framework (NNCF) <https://github.com/openvinotoolkit/nncf>`__,
      as well as conversion on the fly. For integration with the final product it may offer
      lower performance, though.

`Check out the GenAI Quick-start Guide [PDF] <https://docs.openvino.ai/2024/_static/download/GenAI_Quick_Start_Guide.pdf>`__

The advantages of using OpenVINO for LLM deployment:

.. dropdown:: Fewer dependencies and smaller footprint
   :animate: fade-in-slide-down
   :color: secondary

   Less bloated than frameworks such as Hugging Face and PyTorch, with a smaller binary size and reduced
   memory footprint, makes deployments easier and updates more manageable.

.. dropdown:: Compression and precision management
   :animate: fade-in-slide-down
   :color: secondary

   Techniques such as 8-bit and 4-bit weight compression, including embedding layers, and storage
   format reduction. This includes fp16 precision for non-compressed models and int8/int4 for
   compressed models, like GPTQ models from `Hugging Face <https://huggingface.co/models>`__.

.. dropdown:: Enhanced inference capabilities
   :animate: fade-in-slide-down
   :color: secondary

   Advanced features like in-place KV-cache, dynamic quantization, KV-cache quantization and
   encapsulation, dynamic beam size configuration, and speculative sampling, and more are
   available.

.. dropdown:: Stateful model optimization
   :animate: fade-in-slide-down
   :color: secondary

   Models from the Hugging Face Transformers are converted into a stateful form, optimizing
   inference performance and memory usage in long-running text generation tasks by managing past
   KV-cache tensors more efficiently internally. This feature is automatically activated for
   many supported models, while unsupported ones remain stateless. Learn more about the
   :doc:`Stateful models and State API <../openvino-workflow/running-inference/stateful-models>`.

.. dropdown:: Optimized LLM inference
   :animate: fade-in-slide-down
   :color: secondary

   Includes a Python API for rapid development and C++ for further optimization, offering
   better performance than Python-based runtimes.


Proceed to guides on:

* :doc:`OpenVINO GenAI Flavor <./llm_inference_guide/genai-guide>`
* :doc:`Hugging Face and Optimum Intel <./llm_inference_guide/llm-inference-hf>`


