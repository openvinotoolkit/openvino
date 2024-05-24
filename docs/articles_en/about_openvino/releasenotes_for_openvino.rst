.. {#openvino_release_notes}

OpenVINO Release Notes
========================================


2023.3 (LTS) - 24.01.2024
###########################

Summary of major features and improvements
++++++++++++++++++++++++++++++++++++++++++++

* More Generative AI coverage and framework integrations to minimize code changes.

  * Introducing `OpenVINO Gen AI repository <https://github.com/openvinotoolkit/openvino.genai>`__
    on GitHub that demonstrates native C and C++ pipeline samples for Large Language Models
    (LLMs). String tensors are now supported as inputs and tokenizers natively to reduce
    overhead and ease production.
  * New and noteworthy models validated; Mistral, Zephyr, Qwen, ChatGLM3, and Baichuan
  * New Jupyter Notebooks for
    `Latent Consistency Models (LCM) <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/263-latent-consistency-models-image-generation>`__
    and `Distil-Whisper <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/267-distil-whisper-asr>`__.
    Updated `LLM Chatbot notebook <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/254-llm-chatbot>`__
    to include LangChain, Neural Chat, TinyLlama, ChatGLM3, Qwen, Notus and Youri models.
  * Torch.compile is now fully integrated with OpenVINO, which now includes a hardware
    'options' parameter allowing for seamless inference hardware selection by leveraging
    the plugin architecture in OpenVINO.

* Broader Large Language Model (LLM) support and more model compression techniques.

  * As part of the Neural Network Compression Framework (NNCF), INT4 weight compression model
    formats are now fully supported on Intel® Xeon® CPUs  in addition to Intel® Core™ and iGPU,
    adding more performance, lower memory usage, and accuracy opportunity when using LLMs.
  * Improved performance of transformer based LLM on CPU and GPU using stateful model technique
    to increase memory efficiency where internal states are shared among multiple iterations of
    inference.
  * Easier optimization and conversion of Hugging Face models - compress LLM models to INT8
    and INT4 with Hugging Face Optimum command line interface and export models to OpenVINO
    format. Note this is part of `Optimum-Intel <https://huggingface.co/docs/optimum/intel/index>`__
    which needs to be installed separately.
  * Tokenizer and TorchVision transform support is now available in the OpenVINO runtime
    (via new API) requiring less preprocessing code and enhancing performance by automatically
    handling this model setup. More details on Tokenizers support in Ecosystem section.

* More portability and performance to run AI at the edge, in the cloud, or locally.

  * Full support for 5th Gen Intel® Xeon® Scalable processors (codename Emerald Rapids).
  * Further optimized performance on Intel® Core™ Ultra (codename Meteor Lake) CPU with
    latency hint, by leveraging both P-core and E-cores.
  * Improved performance on ARM platforms using throughput hint, which increases efficiency
    in utilization of CPU cores and memory bandwidth.
  * Preview JavaScript API to enable node JS development to access JavaScript binding via
    source code. See details below.
  * Improved `model serving of LLMs <https://github.com/openvinotoolkit/model_server/tree/main/demos/python_demos/llm_text_generation>`__
    through OpenVINO Model Server. This not only enables LLM serving over KServe v2 gRPC
    and REST APIs for more flexibility, but also improves throughput by running processing
    like tokenization on the server side. More details in the Ecosystem section.


Support Change and Deprecation Notices
++++++++++++++++++++++++++++++++++++++++++

* The OpenVINO™ Development Tools package (pip install openvino-dev) is deprecated and will be
  removed from installation options and distribution channels beginning with 2025.0.
  For more details, refer to the :doc:`OpenVINO Legacy Features and Components <openvino_legacy_features>`
  page.
* Ubuntu 18.04 support will be discontinued in the 2023.3 LTS release. The recommended version
  of Ubuntu is 22.04.
* Starting in release 2023.3 OpenVINO will no longer support Python 3.7 due to the Python
  community discontinuing support. Update to a newer version (currently 3.8-3.11) to avoid
  interruptions.
*	All ONNX Frontend legacy API (known as ONNX_IMPORTER_API) will no longer be available in 2024.0 release.
* ``PerfomanceMode.UNDEFINED`` property as part of the OpenVINO Python API will be
  discontinued in the 2024.0 release.

* Tools:

  * :doc:`Deployment Manager <openvino_docs_install_guides_deployment_manager_tool>`
    is deprecated and will be supported for two years according to our LTS policy.
    Visit our :doc:`selector tool <openvino_docs_install_guides_overview>` to see
    package distribution options or our :doc:`deployment guide <openvino_deployment_guide>`
    documentation.
  * Accuracy Checker is deprecated and will be discontinued with 2024.0.
  * Post-Training Optimization Tool (POT)  has been deprecated and the 2023.3 LTS will be
    the last release that will support the tool.  Developers are encouraged to use the Neural
    Network Compression Framework (NNCF) for this feature.
  * Model Optimizer is deprecated and will be fully supported until the 2025.0 release.
    We encourage developers to perform model conversion through OpenVINO Model Converter
    (API call: OVC). Follow the
    :doc:`model conversion transition guide <openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition>`
    for more details.
  * Deprecated support for a `git patch <https://github.com/openvinotoolkit/nncf/tree/develop/third_party_integration/huggingface_transformers>`__
    for NNCF integration with `huggingface/transformers <https://github.com/huggingface/transformers>`__.
    The recommended approach is to use `huggingface/optimum-intel <https://github.com/huggingface/optimum-intel>`__
    for applying NNCF optimization on top of models from Hugging Face.
  * Support for Apache MXNet, Caffe, and Kaldi model formats is deprecated and will be
    discontinued with the 2024.0 release.

* Runtime:

  * Intel® Gaussian & Neural Accelerator (Intel® GNA) will be deprecated in a future release.
    We encourage developers to use the Neural Processing Unit (NPU) for low-powered systems
    like Intel® CoreTM Ultra or 14th generation and beyond.
  * OpenVINO C++/C/Python 1.0 APIs are deprecated and will be discontinued in the 2024.0 release.
    Please use API 2.0 in your applications going forward to avoid disruption.
  * OpenVINO property Affinity API will be deprecated from 2024.0 and will be discontinued in 2025.0.
    It will be replaced with CPU binding configurations (``ov::hint::enable_cpu_pinning``).


OpenVINO™ Development Tools
++++++++++++++++++++++++++++++++++++++++++

* `Neural Network Compression Framework (NNCF) <https://github.com/openvinotoolkit/nncf>`__

  * Weight compression API, ``nncf.compress_weights()``, has been extended by:

    * When using the 'all_layers' parameter, it compresses the model, including embeddings
      and final layers, to the 4-bit format. This helps make the model footprint smaller
      and improves performance, but it might impact the model accuracy. By default, this
      parameter is disabled, and the backup precision (INT8) is assigned for the embeddings
      and last layers.
    * When using INT8_SYM compression mode for better performance of the compressed model
      in case of 8-bit weight compression you might experience an impact on model accuracy
      therefore by default, we use INT8_ASYM mode to better balance performance and accuracy.
    * We implemented a 4-bit data-aware weight compression feature, introducing the 'dataset'
      optional parameter in ``nncf.compress_weights()``. This parameter can be utilized to
      mitigate accuracy loss in compressed models. It's important to note that enabling
      this option will extend the compression time.
    * Post-training Quantization with Accuracy Control, ``nncf.quantize_with_accuracy_control()``,
      has been extended by the  'restore_mode' optional parameter to revert weights to INT8
      instead of the original precision. This parameter helps to reduce the size of the
      quantized model and improves its performance. By default, it is disabled and model
      weights are reverted to the original precision in ``nncf.quantize_with_accuracy_control()``.

OpenVINO™ Runtime
++++++++++++++++++++++++

* Model Import Updates

  * TensorFlow Framework Support

    * Supported TF1 While Control flow construction w/o TensorArray operations
      (`PR #20800 <https://github.com/openvinotoolkit/openvino/pull/20800>`__).
    * Support for complex tensors has been added
      (`PR #20860 <https://github.com/openvinotoolkit/openvino/pull/20860>`__),
      (`PR #21477 <https://github.com/openvinotoolkit/openvino/pull/21477>`__).
    * Provided fixes for the following:

      * Accept any model file extension for frozen protobuf format
        (`PR #21508 <https://github.com/openvinotoolkit/openvino/pull/21508>`__).
      * Correct ArgMin/ArgMax translators for repeating elements case
        (`PR #21364 <https://github.com/openvinotoolkit/openvino/pull/21364>`__).
      * Correct PartitionedCall translator when numbers of external and internal
        body inputs mismatch
        (`PR #20825 <https://github.com/openvinotoolkit/openvino/pull/20825>`__).

  * PyTorch Framework Support

    * Added support of nested dictionaries and lists as example input.
    * Disabled ``torch.jit.freeze`` in default model tracing scenario and
      improved support for models without freezing, extending model
      coverage and improving accuracy for some models.

  * ONNX Framework Support

    * Switched to ONNX 1.15.0 as a supported version of original framework
      (`PR #20929 <https://github.com/openvinotoolkit/openvino/pull/20929>`__).

* CPU

  * Full support for 5th Gen Intel® Xeon® Scalable processors (codename Emerald Rapids)
    with sub-numa (SNC) and efficient core resource scheduling to improve performance.
  * Further optimized performance on Intel® Core™ Ultra (codename Meteor Lake) CPU with
    latency hint, by leveraging both P-core and E-cores.
  * Further improved performance of LLMs in INT4 weight compression, especially on 1st
    token latency and on 4th and 5th Gen of Intel Xeon platforms (codename Sapphire
    Rapids and Emerald Rapids) with AMX capabilities.
  * Improved performance of transformer-based LLM using stateful model technique to
    increase memory efficiency where internal states (KV cache) are shared among multiple
    iterations of inference. The stateful model implementation supports both greedy search
    and beam search (preview) for LLMs. This technique also reduces the memory footprint
    of LLMs, where Intel Core and Ultra platforms like Raptor Lake and Meteor Lake can
    run INT4 models, such as Llama v2 7B.
  * Improved performance on ARM platforms with throughput hint, by increasing
    efficiency in usage of the CPU cores and memory bandwidth.

* GPU

  * Full support for Intel® Core™ Ultra (codename Meteor Lake) integrated graphics.
  * For LLMs, the first inference latency for INT8 and INT4 weight-compressed models has
    been improved on iGPU thanks to more efficient context processing. Overall average
    token latency for INT8 and INT4 has also been enhanced on iGPU with graph compilation
    optimization, various host overhead optimization, and dynamic padding support for GEMM.
  * Stateful model is functionally supported for LLMs.
  * Model caching for dynamically shaped models is now supported. Model loading time is
    improved for these models, including LLMs.
  * API for switching between size mode (model caching) and speed mode (kernel caching)
    is introduced.
  * The model cache file name is changed to be independent of GPU driver versions.
    The GPU will not generate separate model cache files when the driver is updated.
  * Compilation time for Stable Diffusion models has been improved.

* NPU

  * NPU plugin is available as part of OpenVINO. With the Intel(R) Core Ultra NPU driver
    installed, inference can run on the NPU device.

* AUTO device plug-in (AUTO)

  * Introduced the round-robin policy to AUTO cumulative throughput hint, which dispatches
    inference requests to multiple devices (such as multiple GPU devices) in the round-robin
    sequence, instead of in the device priority sequence. The device priority sequence
    remains as the default configuration.
  * AUTO loads stateful models to GPU or CPU per device priority, since GPU now supports
    stateful model inference.

* OpenVINO Common

  * Enhanced support of String tensors has been implemented, enabling the use of operators
    and models that rely on string tensors.  This update also enhances the capability in
    the torchvision preprocessing (`PR #21244 <https://github.com/openvinotoolkit/openvino/pull/20929>`__).
  * A new feature has been added that enables the selection of P-Cores for model compilation
    on CPU device(s) with hybrid architecture (i.e. Intel® Core™ 12th Gen and beyond).
    This will reduce compilation time compared to previous implementation where P-cores
    and E-cores are used randomly by OS scheduling.

* OpenVINO JavaScript API (preview feature)

  * We've introduced a preview version of
    `JS API <https://github.com/openvinotoolkit/openvino/tree/master/src/bindings/js>`__
    for OpenVINO runtime in this release. We hope that you will try this feature and
    provide your feedback through GitHub `issues <https://github.com/openvinotoolkit/openvino/issues>`__.
  * Known limitations:

    * Only supported in manylinux and x86 (Windows, ARM, ARM64, and macOS have not been tested)
    * Node.js version >= 18.16
    * CMake version < 3.14 is not supported
    * gcc compiler version < 7 is not supported

* OpenVINO Python API

  * Introducing string tensor support for Python API.
  * Added support for the following:

    * Create ov.Tensor from Python lists
    * Create ov.Tensor from empty numpy arrays.
    * Constants from empty numpy arrays.
    * Autogenerated get/set methods for Node attributes.
    * Inference functions (``InferRequest.infer/start_async``, ``CompiledModel.__call__`` etc.) support OVDict as the input.
    * PILLOW interpolation modes bindings. (`PR #21188 <https://github.com/openvinotoolkit/openvino/pull/21188>`__ external contribution: @meetpatel0963)

  * Torchvision to :doc:`OpenVINO preprocessing <openvino_docs_OV_UG_string_tensors>`
    converter documentation has been added to OpenVINO docs.


OpenVINO Ecosystem
+++++++++++++++++++++++++++++++++++++++++++++

* OpenVINO Tokenizer (Preview feature)

  * OpenVINO Tokenizer adds text processing operations to OpenVINO:

    * Text PrePostprocessing without third-party dependencies
    * Convert a HuggingFace tokenizer into the OpenVINO model tokenizer and the
      detokenizer using a CLI tool or Python API
    * Connect a tokenizer and a model to get a single model with text input

  * OpenVINO Tokenizer models work only on the CPU device
  * Supported platforms: Linux (x86 and ARM), Windows and Mac (x86 and ARM)


* OpenVINO Model Server

  * Added support for serving pipelines with custom nodes implemented as a
    `python code <https://github.com/openvinotoolkit/model_server/blob/main/docs/python_support/quickstart.md>`__
    This greatly simplifies exposing GenAI algorithms based on Hugging Face
    and Optimum libraries. It can be also applied for arbitrary pre and
    post-processing in model serving pipelines.
  * Included a new set of model serving demos that use custom nodes with python
    code. These include LLM `text generation <https://github.com/openvinotoolkit/model_server/tree/main/demos/python_demos/llm_text_generation>`__,
    `stable diffusion <https://github.com/openvinotoolkit/model_server/tree/main/demos/python_demos/stable_diffusion>`__,
    and `seq2seq translation <https://github.com/openvinotoolkit/model_server/tree/main/demos/python_demos/seq2seq_translation>`__.
  * Improved video stream analysis `demo <https://github.com/openvinotoolkit/model_server/tree/main/demos/real_time_stream_analysis/python>`__.
    A simple client example can now process the
    video stream from a local camera, video file or RTSP stream.
  * Learn more about these changes on
    `GitHub <https://github.com/openvinotoolkit/model_server/releases>`__.


* Jupyter Notebook Tutorials

  * The following notebooks have been updated or newly added:

    * `Sound generation with AudioLDM2 and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/270-sound-generation-audioldm2>`__.
    * `Single-step image generation using SDXL-turbo and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/271-sdxl-turbo>`__.
    * `Paint by Example using Diffusion models and OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/272-paint-by-example>`__.
    * `LLM-powered chatbot using Stable-Zephyr-3b and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/273-stable-zephyr-3b-chatbot>`__.
    * `Object segmentations with EfficientSAM and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/274-efficient-sam>`__.
    * `Create an LLM-powered RAG system using OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot/254-rag-chatbot.ipynb>`__
      - Demonstrates an integration with LangChain.
    * `High-resolution image generation with Segmind-VegaRT and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/248-stable-diffusion-xl/248-segmind-vegart.ipynb>`__.
    * `Text-to-Image Generation with LCM LoRA and ControlNet Conditioning <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/263-latent-consistency-models-image-generation/263-lcm-lora-controlnet.ipynb>`__.
    * `LLM Instruction-following pipeline with OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/275-llm-question-answering>`__ -
      Demonstrates how to run an instruction-following text generation pipeline using
      tiny-llama-1b-chat, phi-2, dolly-v2-3b, red-pajama-3b-instruct and mistral-7b models.
    * `LLM chabot notebook <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot>`__
      updated with support for new LLMs and INT4/INT8 Weight Compression: TinyLlama-1b-chat,
      Mistral-7B, neural-chat-7b, notus-7b, ChatGLM3, youri-7b-chat (for Japanese language).

  * Added optimization support (8-bit quantization, weight compression) by NNCF for the following notebooks:

    * `Image generation with Würstchen and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/265-wuerstchen-image-generation>`__
    * `QR-code monster <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/264-qrcode-monster>`__
    * `INT4-compression support for LLaVA multimodal chatbot <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/257-llava-multimodal-chatbot>`__
    * `Distil-whisper quantization <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/267-distil-whisper-asr>`__





Known issues
++++++++++++++++++++++++++++++++++++++++++++

| **ID - 127202**
| *Component* - CPU Plugin
| *Description:*
|   Deeplabv3 model from TF framework shows lower performance than previous
    release. This is because the TopK layer in the model is now correctly
    conducting the stable sort as specified by the model, slower than the
    previous unstable sort.
| *Workaround:*
|   This release has the correct behavior. If performance is critical,
    please use the previous version of OpenVINO, or tune the model.

| **ID - 123101**
| *Component* - GPU plugin
| *Description:*
|   Hung up of GPU plugin on A770 Graphics (dGPU) in case of large
    batch size (1750).
| *Workaround:*
|   Decrease the batch size, and wait for the fixed driver released.


Included in This Release
+++++++++++++++++++++++++++++++++++++++++++++

The Intel® Distribution of OpenVINO™ toolkit is available for downloading in
three types of operating systems: Windows, Linux, and macOS.

+--------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------+
|| Component                                                         || License                                                  | Location                                        |
+================================+===================================+=================+=================+=======================+=================================================+
|| OpenVINO (Inference Engine) C++ Runtime                           || Dual licensing:                                          || <install_root>/runtime/*                       |
|| Unified API to integrate the inference with application logic     || Intel® OpenVINO™ Distribution License (Version May 2021) || <install_root>/runtime/include/*               |
|| OpenVINO (Inference Engine) Headers                               || Apache 2.0                                               ||                                                |
+--------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------+
|| OpenVINO (Inference Engine) Python API                            || Apache 2.0                                               || <install_root>/python/*                        |
+--------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------+
|| OpenVINO (Inference Engine) Samples                               || Apache 2.0                                               || <install_root>/samples/*                       |
|| Samples that illustrate OpenVINO C++/ Python API usage            ||                                                          ||                                                |
+--------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------+
|| [Deprecated] Deployment manager                                   || Apache 2.0                                               || <install_root>/tools/deployment_manager/*      |
|| The Deployment Manager is a Python command-line tool that         ||                                                          ||                                                |
|| creates a deployment package by assembling the model, IR files,   ||                                                          ||                                                |
|| your application, and associated dependencies into a runtime      ||                                                          ||                                                |
|| package for your target device.                                   ||                                                          ||                                                |
+--------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------+


Legal Information
+++++++++++++++++++++++++++++++++++++++++++++

You may not use or facilitate the use of this document in connection with any infringement
or other legal analysis concerning Intel products described herein.

You agree to grant Intel a non-exclusive, royalty-free license to any patent claim
thereafter drafted which includes subject matter disclosed herein.

No license (express or implied, by estoppel or otherwise) to any intellectual property
rights is granted by this document.

All information provided here is subject to change without notice. Contact your Intel
representative to obtain the latest Intel product specifications and roadmaps.

The products described may contain design defects or errors known as errata which may
cause the product to deviate from published specifications. Current characterized errata
are available on request.

Intel technologies' features and benefits depend on system configuration and may require
enabled hardware, software or service activation. Learn more at
`http://www.intel.com/ <http://www.intel.com/>`__
or from the OEM or retailer.

No computer system can be absolutely secure.

Intel, Atom, Arria, Core, Movidius, Xeon, OpenVINO, and the Intel logo are trademarks
of Intel Corporation in the U.S. and/or other countries.

OpenCL and the OpenCL logo are trademarks of Apple Inc. used by permission by Khronos

Other names and brands may be claimed as the property of others.

Copyright © 2023, Intel Corporation. All rights reserved.

For more complete information about compiler optimizations, see our Optimization Notice.

Performance varies by use, configuration and other factors. Learn more at
`www.Intel.com/PerformanceIndex <www.Intel.com/PerformanceIndex>`__.

Download
+++++++++++++++++++++++++++++++++++++++++++++

`The OpenVINO product selector tool <https://docs.openvino.ai/install>`__
provides easy access to the right packages that match your desired OS, version,
and distribution options.




