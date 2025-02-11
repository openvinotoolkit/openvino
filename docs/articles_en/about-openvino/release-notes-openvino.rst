=============================
OpenVINO Release Notes
=============================

.. meta::
   :description: See what has changed in OpenVINO with the latest release, as well as all
                 previous releases in this year's cycle.


.. toctree::
   :maxdepth: 1
   :hidden:

   release-notes-openvino/system-requirements
   release-notes-openvino/release-policy



2025.0 - 05 February 2025
#############################

:doc:`System Requirements <./release-notes-openvino/system-requirements>` | :doc:`Release policy <./release-notes-openvino/release-policy>` | :doc:`Installation Guides <./../get-started/install-openvino>`



What's new
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* More GenAI coverage and framework integrations to minimize code changes.

  * New models supported: Qwen 2.5, Deepseek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-7B,
    and DeepSeek-R1-Distill-Qwen-1.5B, FLUX.1 Schnell and FLUX.1 Dev.
  * Whisper Model: Improved performance on CPUs, built-in GPUs, and discrete GPUs with GenAI API.
  * Preview: Introducing NPU support for torch.compile, giving developers the ability to use the
    OpenVINO backend to run the PyTorch API on NPUs. 300+ deep learning models enabled from the
    TorchVision, Timm, and TorchBench repositories.

* Broader Large Language Model (LLM) support and more model compression techniques.

  * Preview: Addition of Prompt Lookup to GenAI API improves 2nd token latency for LLMs by
    effectively utilizing predefined prompts that match the intended use case.
  * Preview: The GenAI API now offers image-to-image inpainting functionality. This feature
    enables models to generate realistic content by inpainting specified modifications and
    seamlessly integrating them with the original image.
  * Asymmetric KV Cache compression  is now enabled  for INT8   on CPUs, resulting in lower
    memory consumption and improved 2nd token latency, especially when dealing with long prompts
    that require significant memory. The option should be explicitly specified by the user.

* More portability and performance to run AI at the edge, in the cloud, or locally.

  * Support for the latest Intel® Core™ Ultra 200H series processors (formerly codenamed
    Arrow Lake-H)
  * Integration of the OpenVINO ™ backend with the Triton Inference Server allows developers to
    utilize the Triton server for enhanced model serving performance when deploying on Intel
    CPUs.
  * Preview: A new OpenVINO ™ backend integration allows developers to leverage OpenVINO
    performance optimizations directly within Keras 3 workflows for faster AI inference on CPUs,
    built-in GPUs, discrete GPUs, and NPUs. This feature is available with the latest Keras 3.8
    release.
  * The OpenVINO Model Server now supports native Windows Server deployments, allowing
    developers to leverage better performance by eliminating container overhead and simplifying
    GPU deployment.



Now Deprecated
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* Legacy prefixes `l_`, `w_`, and `m_` have been removed from OpenVINO archive names.
* The `runtime` namespace for Python API has been marked as deprecated and designated to be
  removed for 2026.0. The new namespace structure has been delivered, and migration is possible
  immediately. Details will be communicated through warnings and via documentation.
* NNCF create_compressed_model() method is deprecated. nncf.quantize() method is now
  recommended for Quantization-Aware Training of PyTorch and TensorFlow models.

OpenVINO™ Runtime
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Common
-----------------------------------------------------------------------------------------------

* Support for Python 3.13 has been enabled for OpenVINO Runtime. Tools, like NNCF will follow
  based on their dependency's readiness.


AUTO Inference Mode
-----------------------------------------------------------------------------------------------

* The issue where AUTO failed to load models to NPU, found on Intel® Core™ Ultra 200V processors
  platform only, has been fixed.
* When ov::CompiledModel, ov::InferRequest, ov::Model are defined as static variables, the APP
  crash issue during exiting has been fixed.


CPU Device Plugin
-----------------------------------------------------------------------------------------------

* Intel® Core™ Ultra 200H processors (formerly code named Arrow Lake-H) are now fully supported.
* Asymmetric 8bit KV Cache compression is now enabled on CPU by default, reducing memory
  usage and memory bandwidth consumption for large language models and improving performance
  for 2nd token generation. Asymmetric 4bit KV Cache compression on CPU is now supported
  as an option to further reduce memory consumption.
* Performance of models running in FP16 on 6th generation of Intel® Xeon® processors with P-core
  has been enhanced by improving utilization of the underlying AMX FP16 capabilities.
* LLM performance has been improved on CPU when using OpenVINO GenAI APIs with the continuous
  batching feature.
* Performance of depth-wise convolution neural networks has been improved.
* CPU platforms where some CPU cores are disabled in the system, which is used in some
  virtualization or real-time system configurations, are now supported.


GPU Device Plugin
-----------------------------------------------------------------------------------------------

* Intel® Core™ Ultra 200H processors (formerly code named Arrow Lake-H) are now fully supported.
* ScaledDotProductAttention (SDPA) operator has been enhanced, improving LLM performance for
  OpenVINO GenAI APIs with continuous batching and SDPA-based LLMs with long prompts (>4k).
* Stateful models are now enabled, significantly improving performance of Whisper models on all
  GPU platforms.
* Stable Diffusion 3 and FLUX.1 performance has been improved.
* The issue of a black image output for image generation models, including SDXL, SD3, and
  FLUX.1, with FP16 precision has been solved.


NPU Device Plugin
-----------------------------------------------------------------------------------------------

* Performance has been improved for Channel-Wise symmetrically quantized LLMs, including Llama2-7B-chat,
  Llama3-8B-instruct, Qwen-2-7B, Mistral-0.2-7B-Instruct, Phi-3-Mini-4K-Instruct, MiniCPM-1B
  models. The best performance is achieved using symmetrically-quantized 4-bit (INT4) quantized
  models.
* Preview: Introducing NPU support for torch.compile, giving developers the ability to use the
  OpenVINO backend to run the PyTorch API on NPUs. 300+ deep learning models enabled from
  the TorchVision, Timm, and TorchBench repositories.

OpenVINO Python API
-----------------------------------------------------------------------------------------------

* Ov:OpExtension feature has been completed for Python API. It will enable users to experiment
  with models and operators that are not officially supported, directly with python. It's
  equivalent to the well-known add_extension option for C++.
* Constant class has been extended with get_tensor_view and get_strides methods that will allow
  advanced users to easily manipulate Constant and Tensor objects, to experiment with data flow
  and processing.

OpenVINO Node.js API
-----------------------------------------------------------------------------------------------

* OpenVINO tokenizer bindings for JavaScript are now available via the
  `npm package <https://www.npmjs.com/package/openvino-tokenizers-node>`__.
  This is another OpenVINO tool available for JavaScript developers in a way that is most
  natural and easy to use and extends capabilities we are delivering to that ecosystem.


TensorFlow Framework Support
-----------------------------------------------------------------------------------------------

* The following has been fixed:

  * Output of TensorListLength to be a scalar.
  * Support of corner cases for ToBool op such as scalar input.
  * Correct output type for UniqueWithCounts.

PyTorch Framework Support
-----------------------------------------------------------------------------------------------

* Preview: Introducing NPU support for torch.compile, giving developers the ability to use
  the OpenVINO backend to run the PyTorch API on NPUs. 300+ deep learning models enabled from
  the TorchVision, Timm, and TorchBench repositories.
* Preview: Support conversion of PyTorch models with AWQ weights compression, enabling models
  like SauerkrautLM-Mixtral-8x7B-AWQ and similar.


OpenVINO Python API
-----------------------------------------------------------------------------------------------

* JAX 0.4.38 is now supported.


Keras 3 Multi-backend Framework Support
-----------------------------------------------------------------------------------------------

* Preview: with Keras 3.8, inference-only OpenVINO backend is introduced, for running model
  predictions using OpenVINO in Keras 3 workflow. To switch to the OpenVINO backend, set the
  KERAS_BACKEND environment variable to "openvino". It supports base operations to infer
  convolutional and transformer models such as MobileNet and Bert from Keras Hub.

  Note: The OpenVINO backend may currently lack support for some operations. This will be
  addressed in upcoming Keras releases as operation coverage is being expanded


ONNX Framework Support
-----------------------------------------------------------------------------------------------

* Runtime memory consumption for models with quantized weight has been reduced.
* Workflow which affected reading of 2 bytes data types has been fixed.




OpenVINO Model Server
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* New feature: Windows native server deployment

  * Model server deployment is now available as a binary application on Windows operating
    systems.
  * Generative endpoints are fully supported, including text generation and embeddings based on
    the OpenAI API, and reranking based on the Cohere API.
  * Functional parity with the Linux version is available with minor differences.
  * The feature is targeted at client machines with Windows 11 and data center environment
    with Windows 2022 Server OS.
  * Demos have been updated to work on both Linux and Windows. Check the
    `installation guide <https://docs.openvino.ai/2025/openvino-workflow/model-server/ovms_docs_deploying_server_baremetal.html>`__

* The following is now officially supported:

  * Intel® Arc™ B-Series Graphics
  * Intel® Core™ Ultra 200V and 200S Processors CPU, iGPU, and NPU.

* Image base OSes have been updated:
  dropped Ubuntu20 and Red Hat UBI 8, added Ubuntu24 and Red Hat UBI9.

* The following has been added:

  * Truncate option in the embeddings endpoint. It is now possible to export the embeddings
    model and automatically truncate the input to match the embeddings context length.
    By default, an error is raised if the input is too long.
  * Speculative decoding algorithm in text generation. Check
    `the demo <https://docs.openvino.ai/nightly/openvino-workflow/model-server/ovms_demos_continuous_batching_speculative_decoding.html>`__.
  * Direct support for models without named outputs. For models without named outputs, generic
    names are assigned during model initialization using the pattern ``out_<index>``.
  * Chat/completions have been extended to support max_completion_tokens parameter and message
    content as an array, ensuring API compatibility with OpenAI API.
  * Histogram metric for tracking pipeline processing duration.
  * Security and stability improvements.

* The following has been fixed:

  * Cancelling text generation for disconnected clients.
  * Detecting of the model context length for embeddings endpoint.


Neural Network Compression Framework
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* Post-training quantization time with the Fast Bias Correction algorithm has been reduced.
* Model compression time with nncf.compress_weights() has been reduced significantly.
* Added a new method quantize_pt2e() for accurate quantization of Torch FX models with NNCF
  algorithms for different non-OpenVINO torch.compile() backends.
* Introduced OpenVINOQuantizer class inherited from PyTorch 2 Quantizer for more accurate and
  efficient quantized PyTorch models for deployments with OpenVINO.
* Added support for nncf.quantize() method as the initialization step for Quantization-Aware
  Training for TensorFlow models.
* NNCF create_compressed_model() method is deprecated. nncf.quantize() method is now
  recommended for Quantization-Aware Training of PyTorch and TensorFlow models.


OpenVINO Tokenizers
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* WordLevel tokenizer/detokenizer and WordPiece detokenizer models are now supported.
* UTF-8 (UCS Transformation Format 8) validation with replacement is now enabled by default in
  detokenizer.
* New models are supported: GLM Edge, ModernBERT, BART-G2P.


OpenVINO.GenAI
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The following has been added:

* Samples

  * Restructured the samples folder, grouping the samples by use case.
  * ChunkStreamer for multinomial_causal_lm.py increasing performance for smaller LLMs.
  * Imageimage and inpainting image generation samples.
  * Progress bar for cpp/image_generation samples.

* Python API specific

  * PYI file describing Python API.
  * TorchGenerator which wraps torch.Generator for random generation.

* WhisperPipeline

  * Stateful decoder for WhisperPipeline. Whisper decoder models with past are deprecated.
  * Export a model with new optimum-intel to obtain stateful version.
  * Performance metrics for WhisperPipeline.
  * initial_prompt and hotwords parameters for WhisperPipeline allowing to guide generation.

* LLMPipeline

  * LoRA support for speculative decoding and continuous batching backend.
  * Prompt lookup decoding with LoRA support.

* Image generation

  * Image2Image and Inpainting pipelines which currently support only Unet-based pipelines.
  * rng_seed parameter to ImageGenerationConfig.
  * Callback for image generation pipelines allowing to track generation progress and obtain
    intermediate results.
  * EulerAncestralDiscreteScheduler for SDXL turbo.
  * PNDMScheduler for Stable Diffusion 1.x and 2.x.
  * Models: FLUX.1-Schnell, Flux.1-Lite-8B-Alpha, FLUX.1-Dev, and Shuttle-3-Diffusion.
  * T5 encoder for SD3 Pipeline.

* VLMPipeline

  * Qwen2VL support.
  * Performance metrics.

* Enabled streaming with non-empty stop_strings.


Other Changes and Known Issues
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Jupyter Notebooks
-----------------------------

* `Janus Pro <https://openvinotoolkit.github.io/openvino_notebooks/?search=Multimodal+understanding+and+generation+with+Janus-Pro+and+OpenVINO>`__
* `Running LLMs with OpenVINO and LocalAI <https://openvinotoolkit.github.io/openvino_notebooks/?search=LocalAI+and+OpenVINO>`__
* `GLM-V-Edge <https://openvinotoolkit.github.io/openvino_notebooks/?search=Visual-language+assistant+with+GLM-Edge-V+and+OpenVINO>`__
* `Multimodal RAG with Llamaindex <https://openvinotoolkit.github.io/openvino_notebooks/?search=Multimodal+RAG+for+video+analytics+with+LlamaIndex>`__
* `OmniGen <https://openvinotoolkit.github.io/openvino_notebooks/?search=Unified+image+generation+using+OmniGen+and+OpenVINO>`__
* `Sana <https://openvinotoolkit.github.io/openvino_notebooks/?search=Image+generation+with+Sana+and+OpenVINO>`__
* `LTX Video <https://openvinotoolkit.github.io/openvino_notebooks/?search=LTX+Video+and+OpenVINO%E2%84%A2>`__
* `Image-to-Image generation using OpenVINO GenAI <https://openvinotoolkit.github.io/openvino_notebooks/?search=Image-to-image+generation+using+OpenVINO+GenAI>`__
* `Inpainting using OpenVINO GenAI <https://openvinotoolkit.github.io/openvino_notebooks/?search=Inpainting+with+OpenVINO+GenAI>`__
* `RAG using OpenVINO GenAI and LangChain <https://openvinotoolkit.github.io/openvino_notebooks/?search=Create+a+RAG+system+using+OpenVINO+GenAI+and+LangChain>`__
* `LLM chatbot <https://openvinotoolkit.github.io/openvino_notebooks/?search=Create+an+LLM-powered+Chatbot+using+OpenVINO+Generate+API>`__
  extended with GLM-Edge, Phi4, and Deepseek-R1 distilled models
* `LLM reasoning with DeepSeek-R1 distilled models <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/deepseek-r1>`__


Known Issues
-----------------------------

| **Component: OVC**
| ID: 160167
| Description:
|   TensorFlow Object Detection models converted to the IR through the OVC tool gives poor
    performance on CPU, GPU, and NPU devices. As a workaround, please use the MO tool from
    2024.6 or earlier to generate IRs.

| **Component: Tokenizers**
| ID: 159392
| Description:
|   ONNX model fails to convert when openvino-tokenizers is installed. As a workaround please
    uninstall openvino-tokenizers to convert ONNX model to the IR.

| **Component: CPU Plugin**
| ID: 161336
| Description:
|   Compilation of an openvino model performing weight quantization fails with Segmentation
    Fault on Intel® Core™ Ultra 200V processors. The following workaround can be applied to
    make it work with existing OV versions (including 25.0 RCs) before application run:
    export DNNL_MAX_CPU_ISA=AVX2_VNNI.

| **Component: GPU Plugin**
| ID: 160802
| Description:
|   mllama model crashes on Intel® Core™ Ultra 200V processors. Please use OpenVINO 2024.6 or
    earlier to run the model.

| **Component: GPU Plugin**
| ID: 160948
| Description:
|   Several models have accuracy degradation on Intel® Core™ Ultra 200V processors,
    Intel® Arc™ A-Series Graphics, and Intel® Arc™ B-Series Graphics. Please use OpenVINO 2024.6
    to run the models. Model list: fastseg-small, hbonet-0.5,
    modnet_photographic_portrait_matting, modnet_webcam_portrait_matting,
    mobilenet-v3-small-1.0-224, nasnet-a-mobile-224, yolo_v4, yolo_v5m, yolo_v5s, yolo_v8n,
    yolox-tiny, yolact-resnet50-fpn-pytorch.




.. Previous 2025 releases
.. +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
.. ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++








Deprecation And Support
+++++++++++++++++++++++++++++

Using deprecated features and components is not advised. They are available to enable a smooth
transition to new solutions and will be discontinued in the future. To keep using discontinued
features, you will have to revert to the last LTS OpenVINO version supporting them.
For more details, refer to:
`OpenVINO Legacy Features and Components <https://docs.openvino.ai/2025/documentation/legacy-features.html>`__.



Discontinued in 2025
-----------------------------

* Runtime components:

  * The OpenVINO property of Affinity API will is no longer available. It has been replaced with CPU
    binding configurations (``ov::hint::enable_cpu_pinning``).

* Tools:

  * The OpenVINO™ Development Tools package (pip install openvino-dev) is no longer available
    for OpenVINO releases in 2025.
  * Model Optimizer is no longer available. Consider using the
    :doc:`new conversion methods <../openvino-workflow/model-preparation/convert-model-to-ir>`
    instead. For more details, see the
    `model conversion transition guide <https://docs.openvino.ai/2024/documentation/legacy-features/transition-legacy-conversion-api.html>`__.
  * Intel® Streaming SIMD Extensions (Intel® SSE) are currently not enabled in the binary
    package by default. They are still supported in the source code form.


Deprecated and to be removed in the future
--------------------------------------------

* Ubuntu 20.04 support will be deprecated in future OpenVINO releases due to the end of
  standard support.
* The openvino-nightly PyPI module will soon be discontinued. End-users should proceed with the
  Simple PyPI nightly repo instead. More information in
  `Release Policy <https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/release-policy.html#nightly-releases>`__.
* “auto shape” and “auto batch size” (reshaping a model in runtime) will be removed in the
  future. OpenVINO's dynamic shape models are recommended instead.
* MacOS x86 is no longer recommended for use due to the discontinuation of validation.
  Full support will be removed later in 2025.
* The `openvino` namespace of the OpenVINO Python API has been redesigned, removing the nested
  `openvino.runtime` module. The old namespace is now considered deprecated and will be
  discontinued in 2026.0.








Legal Information
+++++++++++++++++++++++++++++++++++++++++++++

You may not use or facilitate the use of this document in connection with any infringement
or other legal analysis concerning Intel products described herein. All information provided
here is subject to change without notice. Contact your Intel representative to obtain the
latest Intel product specifications and roadmaps.

No license (express or implied, by estoppel or otherwise) to any intellectual property
rights is granted by this document.

The products described may contain design defects or errors known as errata which may
cause the product to deviate from published specifications. Current characterized errata
are available on request.

Intel technologies' features and benefits depend on system configuration and may require
enabled hardware, software or service activation. Learn more at
`www.intel.com <https://www.intel.com/>`__
or from the OEM or retailer.

No computer system can be absolutely secure.

Intel, Atom, Core, Xeon, OpenVINO, and the Intel logo are trademarks of Intel Corporation in
the U.S. and/or other countries. Other names and brands may be claimed as the property of
others.

Copyright © 2025, Intel Corporation. All rights reserved.

For more complete information about compiler optimizations, see our Optimization Notice.

Performance varies by use, configuration and other factors.