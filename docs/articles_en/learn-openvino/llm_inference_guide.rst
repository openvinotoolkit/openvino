Large Language Model Inference Guide
========================================

.. meta::
   :description: Explore learning materials, including interactive
                 Python tutorials and sample console applications that explain
                 how to use OpenVINO features.


.. toctree::
   :maxdepth: 1
   :hidden:

   Run LLMs with Optimum Intel <llm_inference_guide/llm-inference-hf>
   Run LLMs on OpenVINO GenAI Flavor <llm_inference_guide/genai-guide>
   Run LLMs on Base OpenVINO <llm_inference_guide/llm-inference-native-ov>
   OpenVINO Tokenizers <llm_inference_guide/ov-tokenizers>

Large Language Models (LLMs) like GPT are transformative deep learning networks capable of a
broad range of natural language tasks, from text generation to language translation. OpenVINO
optimizes the deployment of these models, enhancing their performance and integration into
various applications. This guide shows how to use LLMs with OpenVINO, from model loading and
conversion to advanced use cases.

The advantages of using OpenVINO for LLM deployment:

* **OpenVINO offers optimized LLM inference**:
  provides a full C/C++ API, leading to faster operation than Python-based runtimes; includes a
  Python API for rapid development, with the option for further optimization in C++.
* **Compatible with diverse hardware**:
  supports CPUs, GPUs, and neural accelerators across ARM and x86/x64 architectures, integrated
  Intel® Processor Graphics, discrete Intel® Arc™ A-Series Graphics, and discrete Intel® Data
  Center GPU Flex Series; features automated optimization to maximize performance on target
  hardware.
* **Requires fewer dependencies**:
  than frameworks like Hugging Face and PyTorch, resulting in a smaller binary size and reduced
  memory footprint, making deployments easier and updates more manageable.
* **Provides compression and precision management techniques**:
  such as 8-bit and 4-bit weight compression, including embedding layers, and storage format
  reduction. This includes fp16 precision for non-compressed models and int8/int4 for compressed
  models, like GPTQ models from `Hugging Face <https://huggingface.co/models>`__.
* **Supports a wide range of deep learning models and architectures**:
  including text, image, and audio generative models like Llama 2, MPT, OPT, Stable Diffusion,
  Stable Diffusion XL. This enables the development of multimodal applications, allowing for
  write-once, deploy-anywhere capabilities.
* **Enhances inference capabilities**:
  fused inference primitives such as Scaled Dot Product Attention, Rotary Positional Embedding,
  Group Query Attention, and Mixture of Experts. It also offers advanced features like in-place
  KV-cache, dynamic quantization, KV-cache quantization and encapsulation, dynamic beam size
  configuration, and speculative sampling.
* **Provides stateful model optimization**:
  models from the Hugging Face Transformers are converted into a stateful form, optimizing
  inference performance and memory usage in long-running text generation tasks by managing past
  KV-cache tensors more efficiently internally. This feature is automatically activated for many
  supported models, while unsupported ones remain stateless. Learn more about the
  :doc:`Stateful models and State API <../openvino-workflow/running-inference/stateful-models>`.

OpenVINO offers three main paths for Generative AI use cases:

* **Hugging Face**: use OpenVINO as a backend for Hugging Face frameworks (transformers,
  diffusers) through the `Optimum Intel <https://huggingface.co/docs/optimum/intel/inference>`__
  extension.
* **OpenVINO GenAI Flavor**: use OpenVINO GenAI APIs (Python and C++).
* **Base OpenVINO**: use OpenVINO native APIs (Python and C++) with
  `custom pipeline code <https://github.com/openvinotoolkit/openvino.genai>`__.

In both cases, the OpenVINO runtime is used for inference, and OpenVINO tools are used for
optimization. The main differences are in footprint size, ease of use, and customizability.

The Hugging Face API is easy to learn, provides a simple interface and hides the complexity of
model initialization and text generation for a better developer experience. However, it has more
dependencies, less customization, and cannot be ported to C/C++.

The OpenVINO GenAI Flavor reduces the complexity of LLMs implementation by
automatically managing essential tasks like the text generation loop, tokenization,
and scheduling. The Native OpenVINO API provides a more hands-on experience,
requiring manual setup of these functions. Both methods are designed to minimize dependencies
and the overall application footprint and enable the use of generative models in C++ applications.

It is recommended to start with Hugging Face frameworks to experiment with different models and
scenarios. Then the model can be used with OpenVINO APIs if it needs to be optimized
further. Optimum Intel provides interfaces that enable model optimization (weight compression)
using `Neural Network Compression Framework (NNCF) <https://github.com/openvinotoolkit/nncf>`__,
and export models to the OpenVINO model format for use in native API applications.

Proceed to run LLMs with:

* :doc:`Hugging Face and Optimum Intel <./llm_inference_guide/llm-inference-hf>`
* :doc:`OpenVINO GenAI Flavor <./llm_inference_guide/genai-guide>`
* :doc:`Native OpenVINO API <./llm_inference_guide/llm-inference-native-ov>`

The table below summarizes the differences between Hugging Face and the native OpenVINO API
approaches.

.. dropdown:: Differences between Hugging Face and the native OpenVINO API

   .. list-table::
      :widths: 20 25 55
      :header-rows: 1

      * -
        - Hugging Face through OpenVINO
        - OpenVINO Native API
      * - Model support
        - Supports transformer-based models such as LLMs
        - Supports all model architectures from most frameworks
      * - APIs
        - Python (Hugging Face API)
        - Python, C++ (OpenVINO API)
      * - Model Format
        - Source Framework / OpenVINO
        - Source Framework / OpenVINO
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
        - Provided through high-level Hugging Face APIs
        - Must be custom implemented (see OpenVINO samples and notebooks)
      * - Performance
        - Good, but less efficient compared to native APIs
        - Inherent speed advantage with C++, but requires hands-on optimization
      * - Flexibility
        - Constrained to Hugging Face API
        - High flexibility with Python and C++; allows custom coding
      * - Learning Curve and Effort
        - Lower learning curve; quick to integrate
        - Higher learning curve; requires more effort in integration
      * - Ideal Use Case
        - Ideal for quick prototyping and Python-centric projects
        - Best suited for high-performance, resource-optimized production environments
      * - Model Serving
        - Paid service, based on CPU/GPU usage with Hugging Face
        - Free code solution, run script for own server; costs may incur for cloud services
          like AWS but generally cheaper than Hugging Face rates
