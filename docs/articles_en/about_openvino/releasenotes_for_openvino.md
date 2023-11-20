# OpenVINO Release Notes {#openvino_release_notes}

@sphinxdirective

The Intel® Distribution of OpenVINO™ toolkit is an open-source solution for optimizing
and deploying AI inference in domains such as computer vision,automatic speech
recognition, natural language processing, recommendation systems, and generative AI.
With its plug-in architecture, OpenVINO enables developers to write once and deploy
anywhere. We are proud to announce the release of OpenVINO 2023.2 introducing a range
of new features, improvements, and deprecations aimed at enhancing the developer
experience.

New and changed in 2023.2 
###########################

Summary of major features and improvements
++++++++++++++++++++++++++++++++++++++++++++

* More Generative AI coverage and framework integrations to minimize code changes. 

  * **Expanded model support for direct PyTorch model conversion** - automatically convert
    additional models directly from PyTorch or execute via ``torch.compile`` with OpenVINO
    as the backend.
  * **New and noteworthy models supported** - we have enabled models used for chatbots,
    instruction following, code generation, and many more, including prominent models 
    like Llava, chatGLM, Bark (text to audio) and LCM (Latent Consistency Models, an
    optimized version of Stable Diffusion).
  * **Easier optimization and conversion of Hugging Face models** - compress LLM models
    to Int8 with the Hugging Face Optimum command line interface and export models to
    the OpenVINO IR format.
  * **OpenVINO is now available on Conan** - a package manager which allows more seamless
    package management for large scale projects for C and C++ developers.

* Broader Large Language Model (LLM) support and more model compression techniques. 

  * Accelerate inference for LLM models on Intel® CoreTM  CPU and iGPU with the
    use of Int8 model weight compression.  
  * Expanded model support for dynamic shapes for improved performance on GPU.
  * Preview support for Int4 model format is now included. Int4 optimized model
    weights are now available to try on Intel® Core™ CPU and iGPU, to accelerate
    models like Llama 2 and chatGLM2.
  * The following Int4 model compression formats are supported for inference
    in runtime:
    
    * Generative Pre-training Transformer Quantization (GPTQ); with GPTQ-compressed
      models, you can access them through the Hugging Face repositories.
    * Native Int4 compression through Neural Network Compression Framework (NNCF).

* More portability and performance to run AI at the edge, in the cloud, or locally.
  
  * **In 2023.1 we announced full support for ARM** architecture, now we have improved
    performance by enabling FP16 model formats for LLMs and integrating additional 
    acceleration libraries to improve latency.
 
Support Change and Deprecation Notices
++++++++++++++++++++++++++++++++++++++++++

* The OpenVINO™ Development Tools package (pip install openvino-dev) is deprecated
  and will be removed from installation options and distribution channels with 
  2025.0. To learn more, refer to the 
  :doc:`OpenVINO Legacy Features and Components page <openvino_legacy_features>`.
  To ensure optimal performance, install the OpenVINO package (pip install openvino),
  which includes essential components such as OpenVINO Runtime, OpenVINO Converter,
  and Benchmark Tool.

* Tools:

  * :doc:`Deployment Manager <openvino_docs_install_guides_deployment_manager_tool>`
    is deprecated and will be removed in the 2024.0 release. 
  * Accuracy Checker is deprecated and will be discontinued with 2024.0.
  * Post-Training Optimization Tool (POT) is deprecated and will be 
    discontinued with 2024.0. 
  * Model Optimizer is deprecated and will be fully supported up until the 2025.0
    release. Model conversion to the OpenVINO format should be performed through
    OpenVINO Model Converter, which is part of the PyPI package. Follow the 
    :doc:`Model Optimizer to OpenVINO Model Converter transition <openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition>`
    guide for smoother transition. Known limitations are TensorFlow model with 
    TF1 Control flow and object detection models. These limitations relate to 
    the gap in TensorFlow direct conversion capabilities which will be addressed
    in upcoming releases.
  * PyTorch 1.13 support is deprecated in Neural Network Compression Framework (NNCF)

* Runtime:

  * Intel® Gaussian & Neural Accelerator (Intel® GNA) will be deprecated in a future
    release. We encourage developers to use the Neural Processing Unit (NPU) for
    low powered systems like Intel® Core™ Ultra or 14th  generation and beyond.
  * OpenVINO C++/C/Python 1.0 APIs will be discontinued with 2024.0. 
  * Python 3.7 support has been discontinued. 

OpenVINO™ Development Tools
++++++++++++++++++++++++++++++++++++++++++

List of components and their changes:
------------------------------------------

* :doc:`OpenVINO Model Converter tool <openvino_docs_model_processing_introduction>`
  now supports the original framework shape format.
* `Neural Network Compression Framework (NNCF) <https://github.com/openvinotoolkit/nncf>`__
  
  * Added data-free Int4 weight compression support for LLMs in OpenVINO IR with
    ``nncf.compress_weights()``.
  * Improved quantization time of LLMs with NNCF PTQ API for ``nncf.quantize()``
    and ``nncf.quantize_with_accuracy_control()``.
  * Added support for SmoothQuant and ChannelAlighnment algorithms in NNCF HyperParameter
    Tuner for automatic optimization of their hyperparameters during quantization.  
  * Added quantization support for the ``IF`` operation of models in OpenVINO format
    to speed up such models.
  * NNCF Post-training Quantization for PyTorch backend is now supported with
    ``nncf.quantize()`` and the common implementation of quantization algorithms. 
  * Added support for PyTorch 2.1. PyTorch 1.13 support has been deprecated. 

OpenVINO™ Runtime (previously known as Inference Engine) 
---------------------------------------------------------

* OpenVINO Common 

  * Operations for reference implementations updated from legacy API to API 2.0.
  * Symbolic transformation introduced the ability to remove Reshape operations 
    surrounding MatMul operations.

* OpenVINO Python API 

  * Better support for the ``openvino.properties`` submodule, which now allows the use
    of properties directly, without additional parenthesis. Example use-case: 
    ``{openvino.properties.cache_dir: “./some_path/”}``.
  * Added missing properties: ``execution_devices`` and ``loaded_from_cache``.
  * Improved error propagation on imports from OpenVINO package.

* AUTO device plug-in (AUTO) 

  * o	Provided additional option to improve performance of cumulative throughput
    (or MULTI), where part of CPU resources can be reserved for GPU inference 
    when GPU and CPU are both used for inference (using ``ov::hint::enable_cpu_pinning(true)``).
    This avoids the performance issue of CPU resource contention where there
    is not enough CPU resources to schedule tasks for GPU  
    (`PR #19214 <https://github.com/openvinotoolkit/openvino/pull/19214>`__).

* CPU

  * Introduced support of GPTQ quantized Int4 models, with improved performance
    compared to Int8 weight-compressed or FP16 models. In the CPU plugin, 
    the gain in performance is achieved by FullyConnected acceleration with
    4bit weight decompression
    (`PR #20607 <https://github.com/openvinotoolkit/openvino/pull/20607>`__).
  * Improved performance of Int8 weight-compressed large language models on
    some platforms, such as 13th Gen Intel Core
    (`PR #20607 <https://github.com/openvinotoolkit/openvino/pull/20607>`__). 
  * Further reduced memory consumption of select large language models on
    CPU platforms with AMX and AVX512 ISA, by eliminating extra memory copy 
    with a unified weight layout 
    (`PR #19575 <https://github.com/openvinotoolkit/openvino/pull/19575>`__). 

  * Fixed performance issue observed in 2023.1 release on select Xeon CPU
    platform with improved thread workload partitioning matching L2 cache 
    utilization 
    (`PR #20436 <https://github.com/openvinotoolkit/openvino/pull/20436>`__).
  * Extended support of configuration (enable_cpu_pinning) on Windows
    platforms to allow fine-grain control on CPU resource used for inference
    workload, by binding inference thread to CPU cores
    (`PR #19418 <https://github.com/openvinotoolkit/openvino/pull/19418>`__).
  * Optimized YoloV8n and YoloV8s model performance for BF16/FP32 precision.
  * Optimized Falcon model on 4th Gen Intel® Xeon® Scalable Processors.
  * Enabled support for FP16 inference precision on ARM.

* GPU

  * Enhanced inference performance for Large Language Models.
  * Introduced int8 weight compression to boost LLM performance. 
    (`PR #19548 <https://github.com/openvinotoolkit/openvino/pull/19548>`__).
  * Implemented Int4 GPTQ weight compression for improved LLM performance.
  * Optimized constant weights for LLMs, resulting in better memory usage
    and faster model loading.
  * Optimized gemm (general matrix multiply) and fc (fully connected) for
    enhanced performance on iGPU. 
    (`PR #19780 <https://github.com/openvinotoolkit/openvino/pull/19780>`__).
  * Completed GPU plugin migration to API 2.0.
  * Added support for oneDNN 3.3 version.

* Model Import Updates

  * TensorFlow Framework Support 

    * Supported conversion of models from memory in keras.Model and tf.function formats.
      `PR #19903 <https://github.com/openvinotoolkit/openvino/pull/19903>`__
    * Supported TF 2.14.
      `PR #20385 <https://github.com/openvinotoolkit/openvino/pull/20385>`__

  * PyTorch Framework Support 

    * Supported Int4 GPTQ models.
    * New operations supported. 

  * ONNX Framework Support 

    * Added support for ONNX version 1.14.1 
      (`PR #18359 <https://github.com/openvinotoolkit/openvino/pull/18359>`__)


OpenVINO Ecosystem
+++++++++++++++++++++++++++++++++++++++++++++

OpenVINO Model Server
--------------------------

Introduced an extension of the KServe gRPC API, enabling streaming input and
output for servables with Mediapipe graphs. This extension ensures the persistence
of Mediapipe graphs within a user session, improving processing performance.
This enhancement supports stateful graphs, such as tracking algorithms, and
enables the use of source calculators. 
(`see additional documentation <https://github.com/openvinotoolkit/model_server/blob/main/docs/streaming_endpoints.md>`__)

* Mediapipe framework has been updated to the version 0.10.3.
* model_api used in the openvino inference Mediapipe calculator has been updated
  and included with all its features. 
* Added a demo showcasing gRPC streaming with Mediapipe graph. 
  (`see here <https://github.com/openvinotoolkit/model_server/tree/main/demos/mediapipe/holistic_tracking>`__)
* Added parameters for gRPC quota configuration and changed default gRPC channel
  arguments to add rate limits. It will minimize the risks of impact of the service
  from uncontrolled flow of requests. 
* Updated python clients requirements to match wide range of python versions from 3.6 to 3.11

Learn more about the changes in https://github.com/openvinotoolkit/model_server/releases

Jupyter Notebook Tutorials
-----------------------------

* The following notebooks have been updated or newly added:

  * `LaBSE <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/220-cross-lingual-books-alignment>`__
    Cross-lingual Books Alignment With Transformers
  * `LLM chatbot <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot>`__
    Create LLM-powered Chatbot
    
    * Updated to include Int4 weight compression and Zephyr 7B model

  * `Bark Text-to-Speech <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/256-bark-text-to-audio>`__
    Text-to-Speech generation using Bark
  * `LLaVA Multimodal Chatbot <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/257-llava-multimodal-chatbot>`__
    Visual-language assistant with LLaVA
  * `BLIP-Diffusion - Subject-Driven Generation <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/258-blip-diffusion-subject-generation>`__
    Subject-driven image generation and editing using BLIP Diffusion
  * `DeciDiffusion <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/259-decidiffusion-image-generation>`__
    Image generation with DeciDiffusion
  * `Fast Segment Anything <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/261-fast-segment-anything>`__
    Object segmentations with FastSAM
  * `SoftVC VITS Singing Voice Conversion <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/262-softvc-voice-conversion>`__
  * `QR Code Monster <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/264-qrcode-monster>`__
    Generate creative QR codes with ControlNet QR Code Monster
  * `Würstchen <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/265-wuerstchen-image-generation>`__
    Text-to-image generation with Würstchen
  * `Distil-Whisper <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/267-distil-whisper-asr>`__
    Automatic speech recognition using Distil-Whisper and OpenVINO™


* Added optimization support (8-bit quantization, weight compression)
  by NNCF for the following notebooks:

  * `Image generation with DeepFloyd IF <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/238-deepfloyd-if>`__
  * `Instruction following using Databricks Dolly 2.0 <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/240-dolly-2-instruction-following>`__
  * `Visual Question Answering and Image Captioning using BLIP <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/233-blip-visual-language-processing>`__
  * `Grammatical Error Correction <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/214-grammar-correction>`__
  * `Universal segmentation with OneFormer <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/249-oneformer-segmentation>`__
  * `Visual-language assistant with LLaVA and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/257-llava-multimodal-chatbot>`__
  * `Image editing with InstructPix2Pix <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/231-instruct-pix2pix-image-editing>`__
  * `MMS: Scaling Speech Technology to 1000+ languages <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/255-mms-massively-multilingual-speech>`__
  * `Image generation with Latent Consistency Model <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/263-latent-consistency-models-image-generation>`__
  * `Object segmentations with FastSAM <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/261-fast-segment-anything>`__
  * `Automatic speech recognition using Distil-Whisper <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/267-distil-whisper-asr>`__



Known issues
++++++++++++++++++++++++++++++++++++++++++++

| **ID - 118179**
| *Component* - Python API, Plugins
| *Description:*
|   When input byte sizes are matching, inference methods accept incorrect inputs
    in copy mode (share_inputs=False). Example: [1, 4, 512, 512] is allowed when
    [1, 512, 512, 4] is required by the model.
| *Workaround:*
|   Pass inputs which shape and layout match model ones.

| **ID - 124181**
| *Component* - CPU plugin
| *Description:*
|   On CPU platform with L2 cache size less than 256KB, such as i3 series of 8th
    Gen Intel CORE platforms, some models may hang during model loading.
| *Workaround:*
|   Rebuild the software from OpenVINO master or use the next OpenVINO release.

| **ID - 121959**
| *Component* - CPU plugin
| *Description:*
|   During inference using latency hint on selected hybrid CPU platforms 
    (such as 12th or 13th Gen Intel CORE), there is a sporadic occurrence of 
    increased latency caused by the operating system scheduling of P-cores or 
    E-cores during OpenVINO initialization.
| *Workaround:*
|   This will be fixed in the next OpenVINO release. 

| **ID - 123101**
| *Component* - GPU plugin 
| *Description:*
|   Hung up of GPU plugin on A770 Graphics (dGPU) in case of
    large batch size (1750).   
| *Workaround:*
|   Decrease the batch size, wait for fixed driver released.

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
|| OpenVINO (Inference Engine) Pythion API                           || Apache 2.0                                               || <install_root>/python/*                        |
+--------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------+
|| OpenVINO (Inference Engine) Samples                               || Apache 2.0                                               || <install_root>/samples/*                       |
|| Samples that illustrate OpenVINO C++/ Python API usage            ||                                                          ||                                                |
+--------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------+
|| [Deprecated] Deployment manager                                   || Apache 2.0                                               || <install_root>/tools/deployment_manager/*      | 
|| The Deployment Manager is a Python* command-line tool that        ||                                                          ||                                                | 
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


 

@endsphinxdirective