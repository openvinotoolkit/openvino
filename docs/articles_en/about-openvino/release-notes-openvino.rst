.. {#openvino_release_notes}

OpenVINO Release Notes
=============================


2024.0 - 06 March 2024
#############################

:doc:`System Requirements <./system-requirements>`  |  :doc:`Installation Guides <./../get-started/install-openvino>`


What's new
+++++++++++++++++++++++++++++

* More Generative AI coverage and framework integrations to minimize code changes.

  * Improved out-of-the-box experience for TensorFlow sentence encoding models through the
    installation of OpenVINO™ toolkit Tokenizers.
  * New and noteworthy models validated:
    Mistral, StableLM-tuned-alpha-3b, and StableLM-Epoch-3B.
  * OpenVINO™ toolkit now supports Mixture of Experts (MoE), a new architecture that helps
    process more efficient generative models through the pipeline.
  * JavaScript developers now have seamless access to OpenVINO API. This new binding enables a
    smooth integration with JavaScript API.

* Broader Large Language Model (LLM) support and more model compression techniques.

  * Broader Large Language Model (LLM) support and more model compression techniques.
  * Improved quality on INT4 weight compression for LLMs by adding the popular technique,
    Activation-aware Weight Quantization, to the Neural Network Compression Framework (NNCF).
    This addition reduces memory requirements and helps speed up token generation.
  * Experience enhanced LLM performance on Intel® CPUs, with internal memory state enhancement,
    and INT8 precision for KV-cache. Specifically tailored for multi-query LLMs like ChatGLM.
  * The OpenVINO™ 2024.0 release makes it easier for developers, by integrating more OpenVINO™
    features with the Hugging Face ecosystem. Store quantization configurations for popular
    models directly in Hugging Face to compress models into INT4 format while preserving
    accuracy and performance.

* More portability and performance to run AI at the edge, in the cloud, or locally.

  * A preview plugin architecture of the integrated Neural Processor Unit (NPU) as part of
    Intel® Core™ Ultra processor (codename Meteor Lake) is now included in the main OpenVINO™
    package on PyPI.
  * Improved performance on ARM by enabling the ARM threading library. In addition, we now
    support multi-core ARM platforms and enabled FP16 precision by default on MacOS.
  * New and improved LLM serving samples from OpenVINO Model Server for multi-batch inputs and
    Retrieval Augmented Generation (RAG).


OpenVINO™ Runtime
+++++++++++++++++++++++++++++

Common
-----------------------------

* The legacy API for CPP and Python bindings has been removed.
* StringTensor support has been extended by operators such as ``Gather``, ``Reshape``, and
  ``Concat``, as a foundation to improve support for tokenizer operators and compliance with
  the TensorFlow Hub.
* oneDNN has been updated to v3.3.
  (`see oneDNN release notes <https://github.com/oneapi-src/oneDNN/releases>`__).


CPU Device Plugin
-----------------------------

* LLM performance on Intel® CPU platforms has been improved for systems based on AVX2 and
  AVX512, using dynamic quantization and internal memory state optimization, such as INT8
  precision for KV-cache. 13th and 14th generations of Intel® Core™ processors and Intel® Core™
  Ultra processors use AVX2 for CPU execution, and these platforms will benefit from speedup.
  Enable these features by setting ``"DYNAMIC_QUANTIZATION_GROUP_SIZE":"32"`` and
  ``"KV_CACHE_PRECISION":"u8"`` in the configuration file.
* The ``ov::affinity`` API configuration is now deprecated and will be removed in release
  2025.0.
* The following have been improved and optimized:

  * Multi-query structure LLMs (such as ChatGLM 2/3) for BF16 on the 4th and 5th generation
    Intel® Xeon® Scalable processors.
  * `Mixtral <https://huggingface.co/docs/transformers/model_doc/mixtral>`__ model performance.
  * 8-bit compressed LLM compilation time and memory usage, valuable for models with large
    embeddings like `Qwen <https://github.com/QwenLM/Qwen>`__.
  * Convolutional networks in FP16 precision on ARM platforms.

GPU Device Plugin
-----------------------------

* The following have been improved and optimized:

  * Average token latency for LLMs on integrated GPU (iGPU) platforms, using INT4-compressed
    models with large context size on Intel® Core™ Ultra processors.
  * LLM beam search performance on iGPU. Both average and first-token latency decrease may be
    expected for larger context sizes.
  * Multi-batch performance of YOLOv5 on iGPU platforms.

* Memory usage for LLMs has been optimized, enabling '7B' models with larger context on
  16Gb platforms.

NPU Device Plugin (preview feature)
-----------------------------------

* The NPU plugin for OpenVINO™ is now available through PyPI (run “pip install openvino”).

OpenVINO Python API
-----------------------------

* ``.add_extension`` method signatures have been aligned, improving API behavior for better
  user experience.

OpenVINO C API
-----------------------------

* ov_property_key_cache_mode (C++ ov::cache_mode) now enables the ``optimize_size`` and
  ``optimize_speed`` modes to set/get model cache.
* The VA surface on Windows exception has been fixed.

OpenVINO Node.js API
-----------------------------

* OpenVINO - `JS bindings <https://docs.openvino.ai/2024/api/nodejs_api/nodejs_api.html>`__
  are consistent with the OpenVINO C++ API.
* A new distribution channel is now available: Node Package Manager (npm) software registry
  (:doc:`check the installation guide <../get-started/install-openvino/install-openvino-npm>`).
* JavaScript API is now available for Windows users, as some limitations for platforms other
  than Linux have been removed.

TensorFlow Framework Support
-----------------------------

* String tensors are now natively supported, handled on input, output, and intermediate layers
  (`PR #22024 <https://github.com/openvinotoolkit/openvino/pull/22024>`__).

  * TensorFlow Hub universal-sentence-encoder-multilingual inferred out of the box
  * string tensors supported for ``Gather``, ``Concat``, and ``Reshape`` operations
  * integration with openvino-tokenizers module - importing openvino-tokenizers automatically
    patches TensorFlow FE with the required translators for models with tokenization

* Fallback for Model Optimizer by operation to the legacy Frontend is no longer available.
  Fallback by .json config will remain until Model Optimizer is discontinued
  (`PR #21523 <https://github.com/openvinotoolkit/openvino/pull/21523>`__).
* Support for the following has been added:

  * Mutable variables and resources such as HashTable*, Variable, VariableV2
    (`PR #22270 <https://github.com/openvinotoolkit/openvino/pull/22270>`__).
  * New tensor types: tf.u16, tf.u32, and tf.u64
    (`PR #21864 <https://github.com/openvinotoolkit/openvino/pull/21864>`__).
  * 14 NEW Ops*.
    `Check the list here (marked as NEW) <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/frontends/tensorflow/docs/supported_ops.md>`__.
  * TensorFlow 2.15
    (`PR #22180 <https://github.com/openvinotoolkit/openvino/pull/22180>`__).

* The following issues have been fixed:

  * UpSampling2D conversion crashed when input type as int16
    (`PR #20838 <https://github.com/openvinotoolkit/openvino/pull/20838>`__).
  * IndexError list index for Squeeze
    (`PR #22326 <https://github.com/openvinotoolkit/openvino/pull/22326>`__).
  * Correct FloorDiv computation for signed integers
    (`PR #22684 <https://github.com/openvinotoolkit/openvino/pull/22684>`__).
  * Fixed bad cast error for tf.TensorShape to ov.PartialShape
    (`PR #22813 <https://github.com/openvinotoolkit/openvino/pull/22813>`__).
  * Fixed reading tf.string attributes for models in memory
    (`PR #22752 <https://github.com/openvinotoolkit/openvino/pull/22752>`__).


ONNX Framework Support
-----------------------------

* ONNX Frontend now uses the OpenVINO API 2.0.

PyTorch Framework Support
-----------------------------

* Names for outputs unpacked from dict or tuple are now clearer
  (`PR #22821 <https://github.com/openvinotoolkit/openvino/pull/22821>`__).
* FX Graph (torch.compile) now supports kwarg inputs, improving data type coverage.
  (`PR #22397 <https://github.com/openvinotoolkit/openvino/pull/22397>`__).


OpenVINO Model Server
+++++++++++++++++++++++++++++

* OpenVINO™ Runtime backend used is now 2024.0.
* Text generation demo now supports multi batch size, with streaming and unary clients.
* The REST client now supports servables based on mediapipe graphs, including python pipeline
  nodes.
* Included dependencies have received security-related updates.
* Reshaping a model in runtime based on the incoming requests (auto shape and auto batch size)
  is deprecated and will be removed in the future. Using OpenVINO's dynamic shape models is
  recommended instead.


Neural Network Compression Framework (NNCF)
+++++++++++++++++++++++++++++++++++++++++++

* The `Activation-aware Weight Quantization (AWQ) <https://arxiv.org/abs/2306.00978>`__
  algorithm for data-aware 4-bit weights compression is now available. It facilitates better
  accuracy for compressed LLMs with high ratio of 4-bit weights. To enable it, use the
  dedicated ``awq`` optional parameter of ``the nncf.compress_weights()`` API.
* ONNX models are now supported in Post-training Quantization with Accuracy Control, through
  the ``nncf.quantize_with_accuracy_control()``, method. It may be used for models in the
  OpenVINO IR and ONNX formats.
* A `weight compression example tutorial <https://github.com/openvinotoolkit/nncf/tree/develop/examples/llm_compression/openvino/tiny_llama_find_hyperparams>`__
  is now available, demonstrating how to find the appropriate hyperparameters for the TinyLLama
  model from the Hugging Face Transformers, as well as other LLMs, with some modifications.


OpenVINO Tokenizer
+++++++++++++++++++++++++++++

* Regex support has been improved.
* Model coverage has been improved.
* Tokenizer metadata has been added to rt_info.
* Limited support for Tensorflow Text models has been added: convert MUSE for TF Hub with
  string inputs.
* OpenVINO Tokenizers have their own repository now:
  `/openvino_tokenizers <https://github.com/openvinotoolkit/openvino_tokenizers>`__


Other Changes and Known Issues
+++++++++++++++++++++++++++++++

Jupyter Notebooks
-----------------------------

The following notebooks have been updated or newly added:

* `Mobile language assistant with MobileVLM <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/279-mobilevlm-language-assistant>`__
* `Depth estimation with DepthAnything <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/280-depth-anything>`__
* `Kosmos-2 <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/281-kosmos2-multimodal-large-language-model>`__
* `Zero-shot Image Classification with SigLIP <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/282-siglip-zero-shot-image-classification>`__
* `Personalized image generation with PhotMaker <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/283-photo-maker>`__
* `Voice tone cloning with OpenVoice <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/284-openvoice>`__
* `Line-level text detection with Surya <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/285-surya-line-level-text-detection>`__
* `InstantID: Zero-shot Identity-Preserving Generation using OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/286-instant-id>`__
* `Tutorial for Big Image Transfer  (BIT) model quantization using NNCF <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/127-big-transfer-quantization>`__
* `Tutorial for OpenVINO Tokenizers integration into inference pipelines <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/128-openvino-tokenizers>`__
* `LLM chatbot <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot/254-llm-chatbot.ipynb>`__ and
  `LLM RAG pipeline <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot/254-rag-chatbot.ipynb>`__
  have received integration with new models: minicpm-2b-dpo, gemma-7b-it, qwen1.5-7b-chat, baichuan2-7b-chat


Known issues
-----------------------------

| **Component - CPU Plugin**
| *ID* - N/A
| *Description:*
|   Starting with 24.0, model inputs and outputs will no longer have tensor names, unless
    explicitly set to align with the PyTorch framework behavior.

| **Component - GPU runtime**
| *ID* - 132376
| *Description:*
|   First-inference latency slow down for LLMs on Intel® Core™ Ultra processors. Up to 10-20%
    drop may occur due to radical memory optimization for processing ling sequences
    (about 1.5-2 GB reduced memory usage).

| **Component - CPU runtime**
| *ID* - N/A
| *Description:*
|   Performance results (first token latency) may vary from those offered by the previous OpenVINO version, for
    “latency” hint inference of LLMs with long prompts on Xeon platforms with 2 or more
    sockets. The reason is that all CPU cores of just the single socket running the application
    are employed, lowering the memory overhead for LLMs when numa control is not used.
| *Workaround:*
|   The behavior is expected but stream and thread configuration may be used to include cores
    from all sockets.


Deprecation And Support
+++++++++++++++++++++++++++++
Using deprecated features and components is not advised. They are available to enable a smooth
transition to new solutions and will be discontinued in the future. To keep using discontinued
features, you will have to revert to the last LTS OpenVINO version supporting them.
For more details, refer to the :doc:`OpenVINO Legacy Features and Components <../documentation/legacy-features>`
page.

Discontinued in 2024
-----------------------------

* Runtime components:

  * Intel® Gaussian & Neural Accelerator (Intel® GNA). Consider using the Neural Processing
    Unit (NPU) for low-powered systems like Intel® Core™ Ultra or 14th generation and beyond.
  * OpenVINO C++/C/Python 1.0 APIs (see
    `2023.3 API transition guide <https://docs.openvino.ai/2023.3/openvino_2_0_transition_guide.html>`__
    for reference).
  * All ONNX Frontend legacy API (known as ONNX_IMPORTER_API)
  * ``PerfomanceMode.UNDEFINED`` property as part of the OpenVINO Python API

* Tools:

  * Deployment Manager. See :doc:`installation <../get-started/install-openvino>` and
    :doc:`deployment <../get-started/install-openvino>` guides for current distribution
    options.
  * `Accuracy Checker <https://docs.openvino.ai/2023.3/omz_tools_accuracy_checker.html>`__.
  * `Post-Training Optimization Tool <https://docs.openvino.ai/2023.3/pot_introduction.html>`__
    (POT). Neural Network Compression Framework (NNCF) should be used instead.
  * A `Git patch <https://github.com/openvinotoolkit/nncf/tree/develop/third_party_integration/huggingface_transformers>`__
    for NNCF integration with `huggingface/transformers <https://github.com/huggingface/transformers>`__.
    The recommended approach is to use `huggingface/optimum-intel <https://github.com/huggingface/optimum-intel>`__
    for applying NNCF optimization on top of models from Hugging Face.
  * Support for Apache MXNet, Caffe, and Kaldi model formats. Conversion to ONNX may be used
    as a solution.

Deprecated and to be removed in the future
--------------------------------------------

* The OpenVINO™ Development Tools package (pip install openvino-dev) will be removed from
  installation options and distribution channels beginning with OpenVINO 2025.
* Model Optimizer will be discontinued with OpenVINO 2025.0. Consider using the
  :doc:`new conversion methods <../openvino-workflow/model-preparation/convert-model-to-ir>`
  instead. For more details, see the
  :doc:`model conversion transition guide <../documentation/legacy-features/transition-legacy-conversion-api>`.
* OpenVINO property Affinity API will be discontinued with OpenVINO 2025.0.
  It will be replaced with CPU binding configurations (``ov::hint::enable_cpu_pinning``).



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






