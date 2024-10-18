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



2024.4 - 19 September 2024
#############################

:doc:`System Requirements <./release-notes-openvino/system-requirements>` | :doc:`Release policy <./release-notes-openvino/release-policy>` | :doc:`Installation Guides <./../get-started/install-openvino>`



What's new
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* More Gen AI coverage and framework integrations to minimize code changes.

  * Support for GLM-4-9B Chat, MiniCPM-1B, Llama 3 and 3.1, Phi-3-Mini, Phi-3-Medium and
    YOLOX-s models.
  * Noteworthy notebooks added: Florence-2, NuExtract-tiny Structure Extraction, Flux.1 Image
    Generation, PixArt-α: Photorealistic Text-to-Image Synthesis, and Phi-3-Vision Visual
    Language Assistant.

* Broader Large Language Model (LLM) support and more model compression techniques.

  * OpenVINO™ runtime optimized for Intel® Xe Matrix Extensions (Intel® XMX) systolic arrays on
    built-in GPUs for efficient matrix multiplication resulting in significant LLM performance
    boost with improved 1st and 2nd token latency, as well as a smaller memory footprint on
    Intel® Core™ Ultra Processors (Series 2).
  * Memory sharing enabled for NPUs on Intel® Core™ Ultra Processors (Series 2) for efficient
    pipeline integration without memory copy overhead.
  * Addition of the PagedAttention feature for discrete GPUs* enables a significant boost in
    throughput for parallel inferencing when serving LLMs on Intel® Arc™ Graphics or Intel®
    Data Center GPU Flex Series.

* More portability and performance to run AI at the edge, in the cloud, or locally.

  * Support for Intel® Core Ultra Processors Series 2 (formerly codenamed Lunar Lake) on Windows.
  * OpenVINO™ Model Server now comes with production-quality support for OpenAI-compatible API
    which enables significantly higher throughput for parallel inferencing on Intel® Xeon®
    processors when serving LLMs to many concurrent users.
  * Improved performance and memory consumption with prefix caching, KV cache compression, and
    other optimizations for serving LLMs using OpenVINO™ Model Server.
  * Support for Python 3.12.
  * Support for Red Hat Enterprise Linux (RHEL) version 9.3 - 9.4.

Now deprecated
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* The following will not be available beyond the 2024.4 OpenVINO version:

  * The macOS x86_64 debug bins
  * Python 3.8
  * Discrete Keem Bay support

* Intel® Streaming SIMD Extensions (Intel® SSE) will be supported in source code form, but not
  enabled in the binary package by default, starting with OpenVINO 2025.0.

|    Check the `deprecation section <#deprecation-and-support>`__ for more information.



Common
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* Encryption and decryption of topology in model cache is now supported with callback functions
  provided by the user (CPU only for now; ov::cache_encryption_callbacks).
* The Ubuntu20 and Ubuntu22 Docker images now include the tokenizers and GenAI CPP modules,
  including pre-installed Python modules, in development versions of these images.
* Python 3.12 is now supported.

CPU Device Plugin
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* The following is now supported:

  * Tensor parallel feature for multi-socket CPU inference, with performance improvement for
    LLMs with 6B+ parameters (enabled through model_distribution_policy hint configurations).
  * RMSNorm operator, optimized with JIT kernel to improve both the 1st and 2nd token
    performance of LLMs.

* The following has been improved:

  * vLLM support, with PagedAttention exposing attention score as the second output. It can now
    be used in the cache eviction algorithm to improve LLM serving performance.
  * 1st token performance with Llama series of models, with additional CPU operator optimization
    (such as MLP, SDPA) on BF16 precision.
  * Default oneTBB version on Linux is now 2021.13.0, improving overall performance on latest
    Intel XEON platforms.
  * MXFP4 weight compression models (compressing weights to 4-bit with the e2m1 data type
    without a zero point and with 8-bit e8m0 scales) have been optimized for Xeon platforms
    thanks to fullyconnected compressed weight LLM support.

* The following has been fixed:

  * Memory leak when ov::num_streams value is 0.
  * CPU affinity mask is changed after OpenVINO execution when OpenVINO is compiled
    with -DTHREADING=SEQ.


GPU Device Plugin
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* Dynamic quantization for LLMs is now supported on discrete GPU platforms.
* Stable Diffusion 3 is now supported with good accuracy on Intel GPU platforms.
* Both first and second token latency for LLMs have been improved on Intel GPU platforms.
* The issue of model cache not regenerating with the value changes of
  ``ov::hint::performance_mode`` or ``ov::hint::dynamic_quantization_group_size`` has been
  fixed.


NPU Device Plugin
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* `Remote Tensor API <https://docs.openvino.ai/nightly/openvino-workflow/running-inference/inference-devices-and-modes/npu-device/remote-tensor-api-npu-plugin.html>`__
  is now supported.
* You can now query the available number of tiles (ov::intel_npu::max_tiles) and force a
  specific number of tiles to be used by the model, per inference request
  (ov::intel_npu::tiles). **Note:** ov::intel_npu::tiles overrides the default number of tiles
  selected by the compiler based on performance hints (ov::hint::performance_mode). Any tile
  number other than 1 may be a problem for cross platform compatibility, if not tested
  explicitly versus the max_tiles value.
* You can now bypass the model caching mechanism in the driver
  (ov::intel_npu::bypass_umd_caching). Read more about driver and OpenVINO caching.
* Memory footprint at model execution has been reduced by one blob (compiled model) size.
  For execution, the plugin no longer retrieves the compiled model from the driver, it uses the
  level zero graph handle directly, instead. The compiled model is now retrieved from the driver
  only during the export method.


OpenVINO Python API
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* Openvino.Tensor, when created in the shared memory mode, now prevents “garbage collection” of
  numpy memory.
* The ``openvino.experimental`` submodule is now available, providing access to experimental
  functionalities under development.
* New python-exclusive openvino.Model constructors have been added.
* Image padding in PreProcessor is now available.
* OpenVINO Runtime is now compatible with numpy 2.0.


OpenVINO Node.js API
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* The following has been improved

  * Unit tests for increased efficiency and stability
  * Security updates applied to dependencies

* `Electron <https://www.electronjs.org/>`__
  compatibility is now confirmed with new end-to-end tests.
* `New API methods <https://docs.openvino.ai/2024/api/nodejs_api/nodejs_api.html>`__ added.


TensorFlow Framework Support
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* TensorFlow 2.17.0 is now supported.
* JAX 0.4.31 is now supported via a path of jax2tf with native_serialization=False
* `8 NEW* operations <https://github.com/openvinotoolkit/openvino/blob/releases/2024/2/src/frontends/tensorflow/docs/supported_ops.md>`__
  have been added.
* Tensor lists with multiple undefined dimensions in element_shape are now supported, enabling
  support for TF Hub lite0-detection/versions/1 model.


PyTorch Framework Support
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* Torch 2.4 is now supported.
* Inplace ops are now supported automatically if the regular version is supported.
* Symmetric GPTQ model from Hugging Face will now be automatically converted to the signed type
  (INT4) and zero-points will be removed.


ONNX Framework Support
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* ONNX 1.16.0 is now supported
* models with constants/inputs of uint4/int4 types are now supported.
* 4 NEW operations have been added.


OpenVINO Model Server
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* OpenAI API for text generation is now officially supported and recommended for production
  usage. It comes with the following new features:

  * Prefix caching feature, caching the prompt evaluation to speed up text generation.
  * Ability to compress the KV Cache to a lower precision, reducing memory consumption without
    a significant loss of accuracy.
  * ``stop`` sampling parameters, to define a sequence that stops text generation.
  * ``logprobs`` sampling parameter, returning the probabilities to returned tokens.
  * Generic metrics related to execution of the MediaPipe graph that can be used for autoscaling
    based on the current load and the level of concurrency.
  * `Demo of text generation horizontal scalability <https://github.com/openvinotoolkit/model_server/tree/main/demos/continuous_batching/scaling>`__
    using basic docker containers and Kubernetes.
  * Automatic cancelling of text generation for disconnected clients.
  * Non-UTF-8 responses from the model can be now automatically changed to Unicode replacement
    characters, due to their configurable handling.
  * Intel GPU with paged attention is now supported.
  * Support for Llama3.1 models.

* The following has been improved:

  * Handling of model templates without bos_token is now fixed.
  * Performance of the multinomial sampling algorithm.
  * ``finish_reason`` in the response correctly determines reaching max_tokens (length) and
    completing the sequence (stop).
  * Security and stability.



Neural Network Compression Framework
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* The LoRA Correction algorithm is now included in the Weight Compression method, improving the
  accuracy of INT4-compressed models on top of other data-aware algorithms, such as AWQ and
  Scale Estimation. To enable it, set the lora_correction option to True in
  nncf.compress_weights().
* The GPTQ compression algorithm can now be combined with the Scale Estimation algorithm,
  making it possible to run GPTQ, AWQ, and Scale Estimation together, for the optimum-accuracy
  INT4-compressed models.
* INT8 quantization of LSTMSequence and Convolution operations for constant inputs is now
  enabled, resulting in better performance and reduced model size.


OpenVINO Tokenizers
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* Split and BPE tokenization operations have been reimplemented, resulting in improved
  tokenization accuracy and performance.
* New building options are now available, offering up to a 12x reduction in binary size.
* An operation is now available to validate and skip/replace model-generated non-Unicode
  bytecode sequences during detokenization.

OpenVINO.GenAI
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* New samples and pipelines are now available:

  * An example IterableStreamer implementation in
    `multinomial_causal_lm/python sample <https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/python/multinomial_causal_lm>`__

* GenAI compilation is now available as part of OpenVINO via the –DOPENVINO_EXTRA_MODULES CMake
  option.



Other Changes and Known Issues
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Jupyter Notebooks
-----------------------------

* `Florence-2 <https://openvinotoolkit.github.io/openvino_notebooks/?search=Florence-2%3A+Open+Source+Vision+Foundation+Model>`__
* `NuExtract: Structure Extraction <https://openvinotoolkit.github.io/openvino_notebooks/?search=Structure+Extraction+with+NuExtract+and+OpenVINO>`__
* `Flux.1 Image Generation <https://openvinotoolkit.github.io/openvino_notebooks/?search=Image+generation+with+Flux.1+and+OpenVINO>`__
* `PixArt-α: Photorealistic Text-to-Image Synthesis <https://openvinotoolkit.github.io/openvino_notebooks/?search=PixArt-%CE%B1%3A+Fast+Training+of+Diffusion+Transformer+for+Photorealistic+Text-to-Image+Synthesis+with+OpenVINO>`__
* `Phi-3-Vision Visual Language Assistant <https://openvinotoolkit.github.io/openvino_notebooks/?search=Visual-language+assistant+with+Phi3-Vision+and+OpenVINO>`__
* `MiniCPMV2.6 <https://openvinotoolkit.github.io/openvino_notebooks/?search=Visual-language+assistant+with+MiniCPM-V2+and+OpenVINO>`__
* `InternVL2 <https://openvinotoolkit.github.io/openvino_notebooks/?search=Visual-language+assistant+with+InternVL2+and+OpenVINO>`__
* The list of supported models in
  `LLM chatbot <https://openvinotoolkit.github.io/openvino_notebooks/?search=Create+an+LLM-powered+Chatbot+using+OpenVINO+Generate+API>`__
  now includes Phi3.5, Gemma2 support

Known Issues
-----------------------------

| **Component: CPU**
| ID: CVS-150542, CVS-145996
| Description:
|   The upgrade of default oneTBB on Linux platforms to 2021.13.0 improves overall
    performance on latest Intel XEON platform but causes regression in some cases. Limit the
    threads usage of postprocessing done by Torch can mitigate the regression (For example:
    torch.set_num_threads(n), n can be 1, beam search number, prompt batch size or other
    numbers).

| **Component: OpenVINO.Genai**
| ID: 149694
| Description:
|   Passing openvino.Tensor instance to LLMPipleine triggers incompatible arguments error if
    OpenVINO and GenAI are installed from PyPI on Windows.

| **Component: OpenVINO.Genai**
| ID: 148308
| Description:
|   OpenVINO.GenAI archive doesn't have debug libraries for OpenVINO Tokenizers and
    OpenVINO.GenAI.

| **Component: ONNX for ARM**
| ID: n/a
| Description:
|   For ARM binaries, the `1.16 ONNX library <https://vcpkg.link/ports/onnx/versions>`__
    is not yet available. The ONNX library for ARM, version 1.15, does not include the latest
    functional and security updates. Users should update to the latest version as it becomes
    available.
|   Currently, if an unverified AI model is supplied to the ONNX frontend, it could lead to a
    directory traversal issue. Ensure that the file name and file path that a model contains
    are verified and correct. To learn more about the vulnerability, see:
    `CVE-2024-27318 <https://nvd.nist.gov/vuln/detail/CVE-2024-27318>`__ and
    `CVE-2024-27319 <https://nvd.nist.gov/vuln/detail/CVE-2024-27319>`__.

| **Component: Kaldi**
| ID: n/a
| Description:
|   There is a known issue with the Kaldi DL framework support on the Python version 3.12 due
    to the numpy version incompatibilities. As Kaldi support in OpenVINO is currently deprecated
    and will be discontinued with version 2025.0, the issue will not be addressed.


Previous 2024 releases
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
.. ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. dropdown:: 2024.3 - 31 July 2024
   :animate: fade-in-slide-down
   :color: secondary

   **What's new**

   * More Gen AI coverage and framework integrations to minimize code changes.

     * OpenVINO pre-optimized models are now available in Hugging Face making it easier for developers
       to get started with these models.

   * Broader Large Language Model (LLM) support and more model compression techniques.

     * Significant improvement in LLM performance on Intel discrete GPUs with the addition of
       Multi-Head Attention (MHA) and OneDNN enhancements.

   * More portability and performance to run AI at the edge, in the cloud, or locally.

     * Improved CPU performance when serving LLMs with the inclusion of vLLM and continuous batching
       in the OpenVINO Model Server (OVMS). vLLM is an easy-to-use open-source library that supports
       efficient LLM inferencing and model serving.
     * Ubuntu 24.04 is now officially supported.

   **OpenVINO™ Runtime**

   *Common*

   * OpenVINO may now be used as a backend for vLLM, offering better CPU performance due to
     fully-connected layer optimization, fusing multiple fully-connected layers (MLP), U8 KV cache,
     and dynamic split fuse.
   * Ubuntu 24.04 is now officially supported, which means OpenVINO is now validated on this
     system (preview support).
   * The following have been improved:

     * Increasing support for models like YoloV10 or PixArt-XL-2, thanks to enabling Squeeze and
       Concat layers.
     * Performance of precision conversion FP16/BF16 -> FP32.

   *AUTO Inference Mode*

   * Model cache is now disabled for CPU acceleration even when cache_dir is set, because CPU
     acceleration is skipped when the cached model is ready for the target device in the 2nd run.

   *Heterogeneous Inference Mode*

   * PIPELINE_PARALLEL policy is now available, to inference large models on multiple devices per
     available memory size, being especially useful for large language models that don't fit into
     one discrete GPU (a preview feature).

   *CPU Device Plugin*

   * Fully Connected layers have been optimized together with RoPE optimization with JIT kernel to
     improve performance for LLM serving workloads on Intel AMX platforms.
   * Dynamic quantization of Fully Connected layers is now enabled by default on Intel AVX2 and
     AVX512 platforms, improving out-of-the-box performance for 8bit/4bit weight-compressed LLMs.
   * Performance has been improved for:

     * ARM server configuration, due to migration to Intel® oneAPI Threading Building Blocks 2021.13.
     * ARM for FP32 and FP16.

   *GPU Device Plugin*

   * Performance has been improved for:

     * LLMs and Stable Diffusion on discrete GPUs, due to latency decrease, through optimizations
       such as Multi-Head Attention (MHA) and oneDNN improvements.
     * Whisper models on discrete GPU.


   *NPU Device Plugin*

   * NPU inference of LLMs is now supported with GenAI API (preview feature). To support LLMs on
     NPU (requires the most recent version of the NPU driver), additional relevant features are
     also part of the NPU plugin now.
   * Models bigger than 2GB are now supported on both NPU driver
     (Intel® NPU Driver - Windows* 32.0.100.2540) and NPU plugin side (both Linux and Windows).
   * Memory optimizations have been implemented:

     * Weights are no longer copied from NPU compiler adapter.
     * Improved memory and first-ever inference latency for inference on NPU.

   *OpenVINO Python API*

   * visit_attributes is now available for custom operation implemented in Python, enabling
     serialization of operation attributes.
   * Python API is now extended with new methods for Model class, e.g. Model.get_sink_index, new
     overloads for Model.get_result_index.

   *OpenVINO Node.js API*

   * Tokenizers and StringTensor are now supported for LLM inference.
   * Compatibility with electron.js is now restored for desktop application developers.
   * Async version of Core.import_model and enhancements for Core.read_model methods are now
     available, for more efficient model reading, especially for LLMs.

   *TensorFlow Framework Support*

   * Models with keras.LSTM operations are now more performant in CPU inference.
   * The tensor list initialized with an undefined element shape value is now supported.

   *TensorFlow Lite Framework Support*

   * Constants containing spare tensors are now supported.

   *PyTorch Framework Support*

   * Setting types/shapes for nested structures (e.g., dictionaries and tuples) is now supported.
   * The aten::layer_norm has been updated to support dynamic shape normalization.
   * Dynamic shapes support in the FX graph has been improved, benefiting torch.compile and
     torch.export based applications, improving performance for gemma and chatglm model
     families.

   *ONNX Framework Support*

   * More models are now supported:

     * Models using the new version of the ReduceMean operation (introduced in ONNX opset 18).
     * Models using the Multinomial operation (introduced in ONNX opset 7).


   **OpenVINO Model Server**

   * The following has been improved in OpenAI API text generation:

     * Performance results, due to OpenVINO Runtime and sampling algorithms.
     * Reporting generation engine metrics in the logs.
     * Extra sampling parameters added.
     * Request parameters affecting memory consumption now have value restrictions, within a
       configurable range.

   * The following has been fixed in OpenAI API text generation:

     * Generating streamer responses impacting incomplete utf-8 sequences.
     * A sporadic generation hang.
     * Incompatibility of the last response from the ``completions`` endpoint stream with the vLLM
       benchmarking script.

   **Neural Network Compression Framework**

   * The `MXFP4 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`__
     data format is now supported in the Weight Compression method, compressing weights to 4-bit
     with the e2m1 data type without a zero point and with 8-bit e8m0 scales. This feature
     is enabled by setting ``mode=CompressWeightsMode.E2M1`` in nncf.compress_weights().
   * The AWQ algorithm in the Weight Compression method has been extended for patterns:
     Act->MatMul and Act->MUltiply->MatMul to cover the Phi family models.
   * The representation of symmetrically quantized weights has been updated to a signed data type
     with no zero point. This allows NPU to support compressed LLMs with the symmetric mode.
   * BF16 models in Post-Training Quantization are now supported; nncf.quantize().
   * `Activation Sparsity <https://arxiv.org/abs/2310.17157>`__ (Contextual Sparsity) algorithm in
     the Weight Compression method is now supported (preview), speeding up LLM inference.
     The algorithm is enabled by setting the ``target_sparsity_by_scope`` option in
     nncf.compress_weights() and supports Torch models only.


   **OpenVINO Tokenizers**

   * The following is now supported:

     * Full Regex syntax with the PCRE2 library for text normalization and splitting.
     * Left padding side for all tokenizer types.

   * GLM-4 tokenizer support, as well as detokenization support for Phi-3 and Gemma have been
     improved.





   **Other Changes and Known Issues**

   *Jupyter Notebooks*

   * `Stable Diffusion V3 <https://openvinotoolkit.github.io/openvino_notebooks/?search=Image+generation+with+Stable+Diffusion+v3+and+OpenVINO>`__
   * `Depth Anything V2 <https://openvinotoolkit.github.io/openvino_notebooks/?search=Depth+estimation+with+DepthAnythingV2+and+OpenVINO>`__
   * `RAG System with LLamaIndex <https://openvinotoolkit.github.io/openvino_notebooks/?search=Create+a+RAG+system+using+OpenVINO+and+LlamaIndex>`__
   * `Image Synthesis with Pixart <https://openvinotoolkit.github.io/openvino_notebooks/?search=PixArt-%CE%B1%3A+Fast+Training+of+Diffusion+Transformer+for+Photorealistic+Text-to-Image+Synthesis+with+OpenVINO>`__
   * `Function calling LLM agent with Qwen-Agent <https://openvinotoolkit.github.io/openvino_notebooks/?search=Create+Function-calling+Agent+using+OpenVINO+and+Qwen-Agent>`__
   * `Jina-CLIP <https://openvinotoolkit.github.io/openvino_notebooks/?search=CLIP+model+with+Jina+CLIP+and+OpenVINO>`__
   * `MiniCPM -V2 Visual Language Assistant <https://openvinotoolkit.github.io/openvino_notebooks/?search=Visual-language+assistant+with+MiniCPM-V2+and+OpenVINO>`__
   * `OpenVINO XAI: first steps <https://openvinotoolkit.github.io/openvino_notebooks/?search=eXplainable+AI+%28XAI%29+for+OpenVINO%E2%84%A2+IR+Models>`__
   * `OpenVINO XAI: deep dive <https://openvinotoolkit.github.io/openvino_notebooks/?search=OpenVINO%E2%84%A2+Explainable+AI+Toolkit%3A+Deep+Dive+notebook>`__
   * `LLM Agent with LLamaIndex <https://openvinotoolkit.github.io/openvino_notebooks/?search=Create+an+Agentic+RAG+using+OpenVINO+and+LlamaIndex>`__
   * `Stable Audio <https://openvinotoolkit.github.io/openvino_notebooks/?search=stable+audio>`__
   * `Phi-3-vision <https://openvinotoolkit.github.io/openvino_notebooks/?search=Phi3-Vision>`__

   *OpenVINO.GenAI*

   * Performance counters have been added.
   * Preview support for NPU is now available.

   *Hugging Face*

   OpenVINO pre-optimized models are now available on Hugging Face:

   * Phi-3-mini-128k-instruct (
     `INT4 <https://huggingface.co/OpenVINO/Phi-3-mini-128k-instruct-int4-ov>`__,
     `INT8 <https://huggingface.co/OpenVINO/Phi-3-mini-128k-instruct-int8-ov>`__,
     `FP16 <https://huggingface.co/OpenVINO/Phi-3-mini-128k-instruct-fp16-ov>`__)
   * Mistral-7B-Instruct-v0.2 (
     `INT4 <https://huggingface.co/OpenVINO/Mistral-7B-Instruct-v0.2-int4-ov>`__,
     `INT8 <https://huggingface.co/OpenVINO/Mistral-7B-Instruct-v0.2-int8-ov>`__,
     `FP16 <https://huggingface.co/OpenVINO/Mistral-7B-Instruct-v0.2-fp16-ov>`__)
   * Mixtral-8x7b-Instruct-v0.1 (
     `INT4 <https://huggingface.co/OpenVINO/mixtral-8x7b-instruct-v0.1-int4-ov>`__,
     `INT8 <https://huggingface.co/OpenVINO/Mixtral-8x7B-Instruct-v0.1-int8-ov>`__)
   * LCM_Dreamshaper_v7 (
     `INT8 <https://huggingface.co/OpenVINO/LCM_Dreamshaper_v7-int8-ov>`__,
     `FP16 <https://huggingface.co/OpenVINO/LCM_Dreamshaper_v7-fp16-ov>`__)
   * starcoder2-7b (
     `INT4 <https://huggingface.co/OpenVINO/starcoder2-7b-int4-ov>`__,
     `INT8 <https://huggingface.co/OpenVINO/starcoder2-7b-int8-ov>`__,
     `FP16 <https://huggingface.co/OpenVINO/starcoder2-7b-fp16-ov>`__)
   * For all the models see `HuggingFace <https://huggingface.co/OpenVINO>`__




   *Known Issues*

   | **Component: OpenVINO.GenAI**
   | ID: 148308
   | Description:
   |   The OpenVINO.GenAI archive distribution doesn't include debug libraries for OpenVINO
       Tokenizers and OpenVINO.GenAI.

   | **Component: GPU**
   | ID: 146283
   | Description:
   |   For some LLM models, longer prompts, such as several thousand tokens, may result in
       decreased accuracy on the GPU plugin.
   | Workaround:
   |   It is recommended to run the model in the FP32 precision to avoid the issue.





.. ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
.. ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. dropdown:: 2024.2 - 17 June 2024
   :animate: fade-in-slide-down
   :color: secondary

   **What's new**

   * More :doc:`Gen AI <../learn-openvino/llm_inference_guide/genai-guide>` coverage and framework
     integrations to minimize code changes.

     * Llama 3 optimizations for CPUs, built-in GPUs, and discrete GPUs for improved performance
       and efficient memory usage.
     * Support for Phi-3-mini, a family of AI models that leverages the power of small language
       models for faster, more accurate and cost-effective text processing.
     * Python Custom Operation is now enabled in OpenVINO making it easier for Python developers
       to code their custom operations instead of using C++ custom operations (also supported).
       Python Custom Operation empowers users to implement their own specialized operations into
       any model.
     * Notebooks expansion to ensure better coverage for new models. Noteworthy notebooks added:
       DynamiCrafter, YOLOv10, Chatbot notebook with Phi-3, and QWEN2.


   * Broader Large Language Model (LLM) support and more model compression techniques.

     * GPTQ method for 4-bit weight compression added to NNCF for more efficient inference and
       improved performance of compressed LLMs.
     * Significant LLM performance improvements and reduced latency for both built-in GPUs and
       discrete GPUs.
     * Significant improvement in 2nd token latency and memory footprint of FP16 weight LLMs on
       AVX2 (13th Gen Intel® Core™ processors) and AVX512 (3rd Gen Intel® Xeon® Scalable Processors)
       based CPU platforms, particularly for small batch sizes.

   * More portability and performance to run AI at the edge, in the cloud, or locally.

     * Model Serving Enhancements:

       * Preview: OpenVINO Model Server (OVMS) now supports OpenAI-compatible API along with Continuous
         Batching and PagedAttention, enabling significantly higher throughput for parallel
         inferencing, especially on Intel® Xeon® processors, when serving LLMs to many concurrent
         users.
       * OpenVINO backend for Triton Server now supports dynamic input shapes.
       * Integration of TorchServe through torch.compile OpenVINO backend for easy model deployment,
         provisioning to multiple instances, model versioning, and maintenance.

     * Preview: addition of the :doc:`Generate API <../learn-openvino/llm_inference_guide/genai-guide>`,
       a simplified API for text generation using large language models with only a few lines of
       code. The API is available through the newly launched OpenVINO GenAI package.
     * Support for Intel Atom® Processor X Series. For more details, see :doc:`System Requirements <./release-notes-openvino/system-requirements>`.
     * Preview: Support for Intel® Xeon® 6 processor.

   **OpenVINO™ Runtime**

   *Common*

   * Operations and data types using UINT2, UINT3, and UINT6 are now supported, to allow for a more
     efficient LLM weight compression.
   * Common OV headers have been optimized, improving binary compilation time and reducing binary
     size.

   *AUTO Inference Mode*

   * AUTO takes model caching into account when choosing the device for fast first-inference latency.
     If model cache is already in place, AUTO will directly use the selected device instead of
     temporarily leveraging CPU as first-inference device.
   * Dynamic models are now loaded to the selected device, instead of loading to CPU without
     considering device priority.
   * Fixed the exceptions when use AUTO with stateful models having dynamic input or output.

   *CPU Device Plugin*

   * Performance when using latency mode in FP32 precision has been improved on Intel client
     platforms, including Core Ultra (codename Meteor Lake) and 13th Gen Core processors
     (codename Raptor Lake).
   * 2nd token latency and memory footprint for FP16 LLMs have been improved significantly on AVX2
     and AVX512 based CPU platforms, particularly for small batch sizes.
   * PagedAttention has been optimized on AVX2, AVX512 and AMX platforms together with INT8 KV cache
     support to improve the performance when serving LLM workloads on Intel CPUs.
   * LLMs with shared embeddings have been optimized to improve performance and memory consumption
     on several models including Gemma.
   * Performance on ARM-based servers is significantly improved with upgrade to TBB 2021.2.5.
   * Improved FP32 and FP16 performance on ARM CPU.

   *GPU Device Plugin*

   * Both first token and average token latency of LLMs is improved on all GPU platforms, most
     significantly on discrete GPUs. Memory usage of LLMs has been reduced as well.
   * Stable Diffusion FP16 performance improved on Core Ultra platforms, with significant pipeline
     improvement for models with dynamic-shaped input. Memory usage of the pipeline has been reduced,
     as well.
   * Optimized permute_f_y kernel performance has been improved.

   *NPU Device Plugin*

   * A new set of configuration options is now available.
   * Performance increase has been unlocked, with the new `2408 NPU driver <https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html>`__.

   *OpenVINO Python API*

   * Writing custom Python operators is now supported for basic scenarios (alignment with OpenVINO
     C++ API.) This empowers users to implement their own specialized operations into any model.
     Full support with more advanced features is within the scope of upcoming releases.

   *OpenVINO C API*

   * More element types are now supported to algin with the OpenVINO C++ API.

   *OpenVINO Node.js API*

   * OpenVINO node.js packages now support the electron.js framework.
   * Extended and improved JS API documentation for more complete usage guidelines.
   * Better JS API alignment with OpenVINO C++ API, delivering more advanced features to JS users.

   *TensorFlow Framework Support*

   * 3 new operations are now supported. See operations marked as `NEW here <https://github.com/openvinotoolkit/openvino/blob/releases/2024/2/src/frontends/tensorflow/docs/supported_ops.md>`__.
   * LookupTableImport has received better support, required for 2 models from TF Hub:

     * mil-nce
     * openimages-v4-ssd-mobilenet-v2

   *TensorFlow Lite Framework Support*

   * The GELU operation required for customer model is now supported.

   *PyTorch Framework Support*

   * 9 new operations are now supported.
   * aten::set_item now supports negative indices.
   * Issue with adaptive pool when shape is list has been fixed (PR `#24586 <https://github.com/openvinotoolkit/openvino/pull/24586>`__).

   *ONNX Support*

   * The InputModel interface should be used from now on, instead of a number of deprecated APIs
     and class symbols
   * Translation for ReduceMin-18 and ReduceSumSquare-18 operators has been added, to address
     customer model requests
   * Behavior of the Gelu-20 operator has been fixed for the case when “none” is set as the
     default value.

   **OpenVINO Model Server**

   * OpenVINO Model server can be now used for text generation use cases using OpenAI compatible API.
   * Added support for continuous batching and PagedAttention algorithms for text generation with
     fast and efficient in high concurrency load especially on Intel Xeon processors.
     `Learn more about it <https://github.com/openvinotoolkit/model_server/tree/releases/2024/2/demos/continuous_batching>`__.

   **Neural Network Compression Framework**

   * GPTQ method is now supported in nncf.compress_weights() for data-aware 4-bit weight
     compression of LLMs. Enabled by `gptq=True`` in nncf.compress_weights().
   * Scale Estimation algorithm for more accurate 4-bit compressed LLMs. Enabled by
     `scale_estimation=True`` in nncf.compress_weights().
   * Added support for models with BF16 weights in nncf.compress_weights().
   * nncf.quantize() method is now the recommended path for quantization initialization of
     PyTorch models in Quantization-Aware Training. See example for more details.
   * compressed_model.nncf.get_config() and nncf.torch.load_from_config() API have been added to
     save and restore quantized PyTorch models. See example for more details.
   * Automatic support for int8 quantization of PyTorch models with custom modules has been added.
     Now it is not needed to register such modules before quantization.

   **Other Changes and Known Issues**

   *Jupyter Notebooks*

   * Latest notebooks along with the GitHub validation status can be found in the
     `OpenVINO notebook section <https://openvinotoolkit.github.io/openvino_notebooks/>`__
   * The following notebooks have been updated or newly added:

     * `Image to Video Generation with Stable Video Diffusion <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/stable-video-diffusion/stable-video-diffusion.ipynb>`__
     * `Image generation with Stable Cascade <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/stable-cascade-image-generation/stable-cascade-image-generation.ipynb>`__
     * `One Step Sketch to Image translation with pix2pix-turbo and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/sketch-to-image-pix2pix-turbo/sketch-to-image-pix2pix-turbo.ipynb>`__
     * `Animating Open-domain Images with DynamiCrafter and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/dynamicrafter-animating-images/dynamicrafter-animating-images.ipynb>`__
     * `Text-to-Video retrieval with S3D MIL-NCE and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/s3d-mil-nce-text-to-video-retrieval/s3d-mil-nce-text-to-video-retrieval.ipynb>`__
     * `Convert and Optimize YOLOv10 with OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/yolov10-optimization/yolov10-optimization.ipynb>`__
     * `Visual-language assistant with nanoLLaVA and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/nano-llava-multimodal-chatbot/nano-llava-multimodal-chatbot.ipynb>`__
     * `Person Counting System using YOLOV8 and OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/person-counting-webcam/person-counting.ipynb>`__
     * `Quantization-Sparsity Aware Training with NNCF, using PyTorch framework <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/pytorch-quantization-sparsity-aware-training/pytorch-quantization-sparsity-aware-training.ipynb>`__
     * `Create an LLM-powered Chatbot using OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llm-chatbot/llm-chatbot.ipynb>`__

   *Known Issues*

   | **Component: TBB**
   | ID: TBB-1400/ TBB-1401
   | Description:
   |   In 2024.2, oneTBB 2021.2.x is used for Intel Distribution of OpenVINO Ubuntu and Red Hat
       archives, instead of system TBB/oneTBB. This improves performance on the new generation of
       Xeon platforms but may increase latency of some models on the previous generation. You can
       build OpenVINO with **-DSYSTEM_TBB=ON** to get better latency performance for these models.

   | **Component: python API**
   | ID: CVS-141744
   | Description:
   |   During post commit tests we found problem related with custom operations. Fix is ready and
       will be delivered with 2024.3 release.
   |   - Initial problem: test_custom_op hanged on destruction because it was waiting for a
       thread which tried to acquire GIL.
   |   - The second problem is that pybind11 doesn't allow to work with GIL besides of current
       scope and it's impossible to release GIL for destructors. Blocking destructors and the
       GIL pybind/pybind11#1446
   |   - Current solution allows to release GIL for InferRequest and all called by chain destructors.

   | **Component: CPU runtime**
   | *ID:* MFDNN-11428
   | *Description:*
   |   Due to adopting a new OneDNN library, improving performance for most use cases,
       particularly for AVX2 BRGEMM kernels with the latency hint, the following regressions may
       be noticed:
   |   a. latency regression on certain models, such as unet-camvid-onnx-0001 and mask_rcnn_resnet50_atrous_coco on MTL Windows latency mode
   |   b. performance regression on Intel client platforms if the throughput hint is used
   |   The issue is being investigated and planned to be resolved in the following releases.

   | **Component: Hardware Configuration**
   | *ID:* N/A
   | *Description:*
   |   Reduced performance for LLMs may be observed on newer CPUs. To mitigate, modify the default settings in BIOS to change the system into 2 NUMA node system:
   |    1. Enter the BIOS configuration menu.
   |    2. Select EDKII Menu -> Socket Configuration -> Uncore Configuration -> Uncore General Configuration -> SNC.
   |    3. The SNC setting is set to *AUTO* by default. Change the SNC setting to *disabled* to configure one NUMA node per processor socket upon boot.
   |    4. After system reboot, confirm the NUMA node setting using: `numatcl -H`. Expect to see only nodes 0 and 1 on a 2-socket system with the following mapping:
   |       Node - 0   -  1
   |       0    - 10  -  21
   |       1    - 21  -  10









.. ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
.. ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. dropdown:: 2024.1 - 24 April 2024
   :animate: fade-in-slide-down
   :color: secondary

   **What's new**

   * More Gen AI coverage and framework integrations to minimize code changes.

     * Mixtral and URLNet models optimized for performance improvements on Intel® Xeon® processors.
     * Stable Diffusion 1.5, ChatGLM3-6B, and Qwen-7B models optimized for improved inference speed
       on Intel® Core™ Ultra processors with integrated GPU.
     * Support for Falcon-7B-Instruct, a GenAI Large Language Model (LLM) ready-to-use chat/instruct
       model with superior performance metrics.
     * New Jupyter Notebooks added: YOLO V9, YOLO V8 Oriented Bounding Boxes Detection (OOB), Stable
       Diffusion in Keras, MobileCLIP, RMBG-v1.4 Background Removal, Magika, TripoSR, AnimateAnyone,
       LLaVA-Next, and RAG system with OpenVINO and LangChain.

   * Broader LLM model support and more model compression techniques.

     * LLM compilation time reduced through additional optimizations with compressed embedding.
       Improved 1st token performance of LLMs on 4th and 5th generations of Intel® Xeon® processors
       with Intel® Advanced Matrix Extensions (Intel® AMX).
     * Better LLM compression and improved performance with oneDNN, INT4, and INT8 support for
       Intel® Arc™ GPUs.
     * Significant memory reduction for select smaller GenAI models on Intel® Core™ Ultra processors
       with integrated GPU.

   * More portability and performance to run AI at the edge, in the cloud, or locally.

     * The preview NPU plugin for Intel® Core™ Ultra processors is now available in the OpenVINO
       open-source GitHub repository, in addition to the main OpenVINO package on PyPI.
     * The JavaScript API is now more easily accessible through the npm repository, enabling
       JavaScript developers' seamless access to the OpenVINO API.
     * FP16 inference on ARM processors now enabled for the Convolutional Neural Network (CNN) by
       default.

   **OpenVINO™ Runtime**

   *Common*

   * Unicode file paths for cached models are now supported on Windows.
   * Pad pre-processing API to extend input tensor on edges with constants.
   * A fix for inference failures of certain image generation models has been implemented
     (fused I/O port names after transformation).
   * Compiler's warnings-as-errors option is now on, improving the coding criteria and quality.
     Build warnings will not be allowed for new OpenVINO code and the existing warnings have been
     fixed.

   *AUTO Inference Mode*

   * Returning the ov::enable_profiling value from ov::CompiledModel is now supported.

   *CPU Device Plugin*

   * 1st token performance of LLMs has been improved on the 4th and 5th generations of Intel® Xeon®
     processors with Intel® Advanced Matrix Extensions (Intel® AMX).
   * LLM compilation time and memory footprint have been improved through additional optimizations
     with compressed embeddings.
   * Performance of MoE (e.g. Mixtral), Gemma, and GPT-J has been improved further.
   * Performance has been improved significantly for a wide set of models on ARM devices.
   * FP16 inference precision is now the default for all types of models on ARM devices.
   * CPU architecture-agnostic build has been implemented, to enable unified binary distribution
     on different ARM devices.

   *GPU Device Plugin*

   * LLM first token latency has been improved on both integrated and discrete GPU platforms.
   * For the ChatGLM3-6B model, average token latency has been improved on integrated GPU platforms.
   * For Stable Diffusion 1.5 FP16 precision, performance has been improved on Intel® Core™ Ultra
     processors.

   *NPU Device Plugin*

   * NPU Plugin is now part of the OpenVINO GitHub repository. All the most recent plugin changes
     will be immediately available in the repo. Note that NPU is part of Intel® Core™ Ultra
     processors.
   * New OpenVINO™ notebook “Hello, NPU!” introducing NPU usage with OpenVINO has been added.
   * Version 22H2 or later is required for Microsoft Windows® 11 64-bit to run inference on NPU.

   *OpenVINO Python API*

   * GIL-free creation of RemoteTensors is now used - holding GIL means that the process is not suited
     for multithreading and removing the GIL lock will increase performance which is critical for
     the concept of Remote Tensors.
   * Packed data type BF16 on the Python API level has been added, opening a new way of supporting
     data types not handled by numpy.
   * 'pad' operator support for ov::preprocess::PrePostProcessorItem has been added.
   * ov.PartialShape.dynamic(int) definition has been provided.

   *OpenVINO C API*

   * Two new pre-processing APIs for scale and mean have been added.

   *OpenVINO Node.js API*

   * New methods to align JavaScript API with CPP API have been added, such as
     CompiledModel.exportModel(), core.import_model(), Core set/get property and Tensor.get_size(),
     and Model.is_dynamic().
   * Documentation has been extended to help developers start integrating JavaScript applications
     with OpenVINO™.

   *TensorFlow Framework Support*

   * `tf.keras.layers.TextVectorization tokenizer <https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization>`__
     is now supported.
   * Conversion of models with Variable and HashTable (dictionary) resources has been improved.
   * 8 NEW operations have been added
     (`see the list here, marked as NEW <https://github.com/openvinotoolkit/openvino/blob/releases/2024/1/src/frontends/tensorflow/docs/supported_ops.md>`__).
   * 10 operations have received complex tensor support.
   * Input tensor names for TF1 models have been adjusted to have a single name per input.
   * Hugging Face model support coverage has increased significantly, due to:

     * extraction of input signature of a model in memory has been fixed,
     * reading of variable values for a model in memory has been fixed.

   *PyTorch Framework Support*

   * ModuleExtension, a new type of extension for PyTorch models is now supported
     (`PR #23536 <https://github.com/openvinotoolkit/openvino/pull/23536>`__).
   * 22 NEW operations have been added.
   * Experimental support for models produced by torch.export (FX graph) has been added
     (`PR #23815 <https://github.com/openvinotoolkit/openvino/pull/23815>`__).

   *ONNX Framework Support*

   * 8 new operations have been added.

   **OpenVINO Model Server**

   * OpenVINO™ Runtime backend used is now 2024.1
   * OpenVINO™ models with String data type on output are supported. Now, OpenVINO™ Model Server
     can support models with input and output of the String type, so developers can take advantage
     of the tokenization built into the model as the first layer. Developers can also rely on any
     postprocessing embedded into the model which returns text only. Check the
     `demo on string input data with the universal-sentence-encoder model <https://docs.openvino.ai/2024/ovms_demo_universal-sentence-encoder.html>`__
     and the
     `String output model demo <https://github.com/openvinotoolkit/model_server/tree/main/demos/image_classification_with_string_output>`__.
   * MediaPipe Python calculators have been updated to support relative paths for all related
     configuration and Python code files. Now, the complete graph configuration folder can be
     deployed in an arbitrary path without any code changes.
   * KServe REST API support has been extended to properly handle the string format in JSON body,
     just like the binary format compatible with NVIDIA Triton™.
   * `A demo showcasing a full RAG algorithm <https://github.com/openvinotoolkit/model_server/tree/main/demos/python_demos/rag_chatbot>`__
     fully delegated to the model server has been added.

   **Neural Network Compression Framework**

   * Model subgraphs can now be defined in the ignored scope for INT8 Post-training Quantization,
     nncf.quantize(), which simplifies excluding accuracy-sensitive layers from quantization.
   * A batch size of more than 1 is now partially supported for INT8 Post-training Quantization,
     speeding up the process. Note that it is not recommended for transformer-based models as it
     may impact accuracy. Here is an
     `example demo <https://github.com/openvinotoolkit/nncf/blob/develop/examples/quantization_aware_training/torch/resnet18/README.md>`__.
   * Now it is possible to apply fine-tuning on INT8 models after Post-training Quantization to
     improve model accuracy and make it easier to move from post-training to training-aware
     quantization. Here is an
     `example demo <https://github.com/openvinotoolkit/nncf/blob/develop/examples/quantization_aware_training/torch/resnet18/README.md>`__.

   **OpenVINO Tokenizers**

   * TensorFlow support has been extended - TextVectorization layer translation:

     * Aligned existing ops with TF ops and added a translator for them.
     * Added new ragged tensor ops and string ops.

   * A new tokenizer type, RWKV is now supported:

     * Added Trie tokenizer and Fuse op for ragged tensors.
     * A new way to get OV Tokenizers: build a vocab from file.

   * Tokenizer caching has been redesigned to work with the OpenVINO™ model caching mechanism.

   **Other Changes and Known Issues**

   *Jupyter Notebooks*

   The default branch for the OpenVINO™ Notebooks repository has been changed from 'main' to
   'latest'. The 'main' branch of the notebooks repository is now deprecated and will be maintained
   until September 30, 2024.

   The new branch, 'latest', offers a better user experience and simplifies maintenance due to
   significant refactoring and an improved directory naming structure.

   Use the local
   `README.md <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/README.md>`__
   file and OpenVINO™ Notebooks at
   `GitHub Pages <https://openvinotoolkit.github.io/openvino_notebooks/>`__
   to navigate through the content.

   The following notebooks have been updated or newly added:

   * `Grounded Segment Anything <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/grounded-segment-anything/grounded-segment-anything.ipynb>`__
   * `Visual Content Search with MobileCLIP <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/mobileclip-video-search/mobileclip-video-search.ipynb>`__
   * `YOLO V8 Oriented Bounding Box Detection Optimization <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/yolov8-optimization/yolov8-obb.ipynb>`__
   * `Magika: AI-powered fast and efficient file type identification <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/magika-content-type-recognition/magika-content-type-recognition.ipynb>`__
   * `Keras Stable Diffusion <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/stable-diffusion-keras-cv/stable-diffusion-keras-cv.ipynb>`__
   * `RMBG background removal <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/rmbg-background-removal/rmbg-background-removal.ipynb>`__
   * `AnimateAnyone: pose guided image to video generation <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/animate-anyone/animate-anyone.ipynb>`__
   * `LLaVA-Next visual-language assistant <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llava-next-multimodal-chatbot/llava-next-multimodal-chatbot.ipynb>`__
   * `TripoSR: single image 3d reconstruction <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/triposr-3d-reconstruction/triposr-3d-reconstruction.ipynb>`__
   * `RAG system with OpenVINO and LangChain <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llm-rag-langchain/llm-rag-langchain.ipynb>`__

   *Known Issues*

   | **Component: CPU Plugin**
   | *ID:* N/A
   | *Description:*
   |   Default CPU pinning policy on Windows has been changed to follow Windows' policy
       instead of controlling the CPU pinning in the OpenVINO plugin. This brings certain dynamic or
       performance variance on Windows. Developers can use ov::hint::enable_cpu_pinning to enable
       or disable CPU pinning explicitly.

   | **Component: Hardware Configuration**
   | *ID:* N/A
   | *Description:*
   |   Reduced performance for LLMs may be observed on newer CPUs. To mitigate, modify the default settings in BIOS to
   |   change the system into 2 NUMA node system:
   |    1. Enter the BIOS configuration menu.
   |    2. Select EDKII Menu -> Socket Configuration -> Uncore Configuration -> Uncore General Configuration ->  SNC.
   |    3. The SNC setting is set to *AUTO* by default. Change the SNC setting to *disabled* to configure one NUMA node per processor socket upon boot.
   |    4. After system reboot, confirm the NUMA node setting using: `numatcl -H`. Expect to see only nodes 0 and 1 on a
   |    2-socket system with the following mapping:
   |     Node - 0  -  1
   |      0  - 10  -  21
   |      1 -  21  -  10










.. dropdown:: 2024.0 - 06 March 2024
   :animate: fade-in-slide-down
   :color: secondary

   **What's new**

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
       support multi-core ARM processors and enabled FP16 precision by default on MacOS.
     * New and improved LLM serving samples from OpenVINO Model Server for multi-batch inputs and
       Retrieval Augmented Generation (RAG).

   **OpenVINO™ Runtime**

   *Common*

   * The legacy API for CPP and Python bindings has been removed.
   * StringTensor support has been extended by operators such as ``Gather``, ``Reshape``, and
     ``Concat``, as a foundation to improve support for tokenizer operators and compliance with
     the TensorFlow Hub.
   * oneDNN has been updated to v3.3.
     (`see oneDNN release notes <https://github.com/oneapi-src/oneDNN/releases>`__).

   *CPU Device Plugin*

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
     * Convolutional networks in FP16 precision on ARM processors.

   *GPU Device Plugin*

   * The following have been improved and optimized:

     * Average token latency for LLMs on integrated GPU (iGPU) platforms, using INT4-compressed
       models with large context size on Intel® Core™ Ultra processors.
     * LLM beam search performance on iGPU. Both average and first-token latency decrease may be
       expected for larger context sizes.
     * Multi-batch performance of YOLOv5 on iGPU platforms.

   * Memory usage for LLMs has been optimized, enabling '7B' models with larger context on
     16Gb platforms.

   *NPU Device Plugin (preview feature)*

   * The NPU plugin for OpenVINO™ is now available through PyPI (run “pip install openvino”).

   *OpenVINO Python API*

   * ``.add_extension`` method signatures have been aligned, improving API behavior for better
     user experience.

   *OpenVINO C API*

   * ov_property_key_cache_mode (C++ ov::cache_mode) now enables the ``optimize_size`` and
     ``optimize_speed`` modes to set/get model cache.
   * The VA surface on Windows exception has been fixed.

   *OpenVINO Node.js API*

   * OpenVINO - `JS bindings <https://docs.openvino.ai/2024/api/nodejs_api/nodejs_api.html>`__
     are consistent with the OpenVINO C++ API.
   * A new distribution channel is now available: Node Package Manager (npm) software registry
     (:doc:`check the installation guide <../get-started/install-openvino/install-openvino-npm>`).
   * JavaScript API is now available for Windows users, as some limitations for platforms other
     than Linux have been removed.

   *TensorFlow Framework Support*

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

   *ONNX Framework Support*

   * ONNX Frontend now uses the OpenVINO API 2.0.

   *PyTorch Framework Support*

   * Names for outputs unpacked from dict or tuple are now clearer
     (`PR #22821 <https://github.com/openvinotoolkit/openvino/pull/22821>`__).
   * FX Graph (torch.compile) now supports kwarg inputs, improving data type coverage.
     (`PR #22397 <https://github.com/openvinotoolkit/openvino/pull/22397>`__).

   **OpenVINO Model Server**

   * OpenVINO™ Runtime backend used is now 2024.0.
   * Text generation demo now supports multi batch size, with streaming and unary clients.
   * The REST client now supports servables based on mediapipe graphs, including python pipeline
     nodes.
   * Included dependencies have received security-related updates.
   * Reshaping a model in runtime based on the incoming requests (auto shape and auto batch size)
     is deprecated and will be removed in the future. Using OpenVINO's dynamic shape models is
     recommended instead.

   **Neural Network Compression Framework (NNCF)**

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

   **OpenVINO Tokenizer**

   * Regex support has been improved.
   * Model coverage has been improved.
   * Tokenizer metadata has been added to rt_info.
   * Limited support for Tensorflow Text models has been added: convert MUSE for TF Hub with
     string inputs.
   * OpenVINO Tokenizers have their own repository now:
     `/openvino_tokenizers <https://github.com/openvinotoolkit/openvino_tokenizers>`__

   **Other Changes and Known Issues**

   *Jupyter Notebooks*

   The following notebooks have been updated or newly added:

   * `Mobile language assistant with MobileVLM <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/279-mobilevlm-language-assistant>`__
   * `Depth estimation with DepthAnything <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/280-depth-anything>`__
   * `Kosmos-2 <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/281-kosmos2-multimodal-large-language-model>`__
   * `Zero-shot Image Classification with SigLIP <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/282-siglip-zero-shot-image-classification>`__
   * `Personalized image generation with PhotoMaker <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/283-photo-maker>`__
   * `Voice tone cloning with OpenVoice <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/284-openvoice>`__
   * `Line-level text detection with Surya <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/285-surya-line-level-text-detection>`__
   * `InstantID: Zero-shot Identity-Preserving Generation using OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/286-instant-id>`__
   * `Tutorial for Big Image Transfer  (BIT) model quantization using NNCF <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/127-big-transfer-quantization>`__
   * `Tutorial for OpenVINO Tokenizers integration into inference pipelines <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/128-openvino-tokenizers>`__
   * `LLM chatbot <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot/254-llm-chatbot.ipynb>`__ and
     `LLM RAG pipeline <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot/254-rag-chatbot.ipynb>`__
     have received integration with new models: minicpm-2b-dpo, gemma-7b-it, qwen1.5-7b-chat, baichuan2-7b-chat

   *Known issues*

   | **Component: CPU Plugin**
   | *ID:* N/A
   | *Description:*
   |   Starting with 24.0, model inputs and outputs will no longer have tensor names, unless
       explicitly set to align with the PyTorch framework behavior.

   | **Component: GPU runtime**
   | *ID:* 132376
   | *Description:*
   |   First-inference latency slow down for LLMs on Intel® Core™ Ultra processors. Up to 10-20%
       drop may occur due to radical memory optimization for processing long sequences
       (about 1.5-2 GB reduced memory usage).

   | **Component: CPU runtime**
   | *ID:* N/A
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
  * All ONNX Frontend legacy API (known as ONNX_IMPORTER_API).
  * ``PerfomanceMode.UNDEFINED`` property as part of the OpenVINO Python API.

* Tools:

  * Deployment Manager. See :doc:`installation <../get-started/install-openvino>` and
    :doc:`deployment <../get-started/install-openvino>` guides for current distribution
    options.
  * `Accuracy Checker <https://github.com/openvinotoolkit/open_model_zoo/blob/master/tools/accuracy_checker/README.md>`__.
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

* The macOS x86_64 debug bins will no longer be provided with the OpenVINO toolkit, starting
  with OpenVINO 2024.5.
* Python 3.8 is now considered deprecated, and it will not be available beyond the 2024.4
  OpenVINO version.

  * As MxNet doesn't support Python version higher than 3.8, according to the
    `MxNet PyPI project <https://pypi.org/project/mxnet/>`__,
    it will no longer be supported in future versions, either.

* Discrete Keem Bay support is now considered deprecated and will be fully removed with OpenVINO 2024.5
* Intel® Streaming SIMD Extensions (Intel® SSE) will be supported in source code form, but not
  enabled in the binary package by default, starting with OpenVINO 2025.0
* The openvino-nightly PyPI module will soon be discontinued. End-users should proceed with the
  Simple PyPI nightly repo instead. More information in
  `Release Policy <https://docs.openvino.ai/2024/about-openvino/release-notes-openvino/release-policy.html#nightly-releases>`__.
* The OpenVINO™ Development Tools package (pip install openvino-dev) will be removed from
  installation options and distribution channels beginning with OpenVINO 2025.0.
* Model Optimizer will be discontinued with OpenVINO 2025.0. Consider using the
  :doc:`new conversion methods <../openvino-workflow/model-preparation/convert-model-to-ir>`
  instead. For more details, see the
  :doc:`model conversion transition guide <../documentation/legacy-features/transition-legacy-conversion-api>`.
* OpenVINO property Affinity API will be discontinued with OpenVINO 2025.0.
  It will be replaced with CPU binding configurations (``ov::hint::enable_cpu_pinning``).
* OpenVINO Model Server components:

  * “auto shape” and “auto batch size” (reshaping a model in runtime) will be removed in the
    future. OpenVINO's dynamic shape models are recommended instead.

* A number of notebooks have been deprecated. For an up-to-date listing of available notebooks,
  refer to the `OpenVINO™ Notebook index (openvinotoolkit.github.io) <https://openvinotoolkit.github.io/openvino_notebooks/>`__.

  .. dropdown:: See the deprecated notebook list
     :animate: fade-in-slide-down
     :color: muted

     * `Handwritten OCR with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/handwritten-ocr>`__

       * See alternative: `Optical Character Recognition (OCR) with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/optical-character-recognition>`__,
       * See alternative: `PaddleOCR with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/paddle-ocr-webcam>`__,
       * See alternative: `Handwritten Text Recognition Demo <https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/handwritten_text_recognition_demo/python/README.md>`__

     * `Image In-painting with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/image-inpainting>`__

       * See alternative: `Image Inpainting Python Demo <https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/image_inpainting_demo/python/README.md>`__

     * `Interactive Machine Translation with OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/machine-translation>`__

       * See alternative: `Machine Translation Python* Demo <https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/machine_translation_demo/python/README.md>`__

     * `Open Model Zoo Tools Tutorial <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/model-tools>`__

       * No alternatives, demonstrates deprecated tools.

     * `Super Resolution with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vision-superresolution>`__

       * See alternative: `Super Resolution with PaddleGAN and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vision-paddlegan-superresolution>`__
       * See alternative:  `Image Processing C++ Demo <https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/image_processing_demo/cpp/README.md>`__

     * `Image Colorization with OpenVINO Tutorial <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vision-image-colorization>`__
     * `Interactive Question Answering with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/question-answering>`__

       * See alternative: `BERT Question Answering Embedding Python* Demo <https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/bert_question_answering_embedding_demo/python/README.md>`__
       * See alternative:  `BERT Question Answering Python* Demo <https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/bert_question_answering_demo/python/README.md>`__

     * `Vehicle Detection And Recognition with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vehicle-detection-and-recognition>`__

       * See alternative: `Security Barrier Camera C++ Demo  <https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/security_barrier_camera_demo/cpp/README.md>`__

     * `The attention center model with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/attention-center>`_
     * `Image Generation with DeciDiffusion <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/decidiffusion-image-generation>`_
     * `Image generation with DeepFloyd IF and OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/deepfloyd-if>`_
     * `Depth estimation using VI-depth with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/depth-estimation-videpth>`_
     * `Instruction following using Databricks Dolly 2.0 and OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/dolly-2-instruction-following>`_

       * See alternative: `LLM Instruction-following pipeline with OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-question-answering>`__

     * `Image generation with FastComposer and OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/fastcomposer-image-generation>`__
     * `Video Subtitle Generation with OpenAI Whisper  <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/whisper-subtitles-generation>`__

       * See alternative: `Automatic speech recognition using Distil-Whisper and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/distil-whisper-asr/distil-whisper-asr.ipynb>`__

     * `Introduction to Performance Tricks in OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/performance-tricks>`__
     * `Speaker Diarization with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/pyannote-speaker-diarization>`__
     * `Subject-driven image generation and editing using BLIP Diffusion and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/blip-diffusion-subject-generation>`__
     * `Text Prediction with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/text-prediction>`__
     * `Training to Deployment with TensorFlow and OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/tensorflow-training-openvino>`__
     * `Speech to Text with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/speech-to-text>`__
     * `Convert and Optimize YOLOv7 with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov7-optimization>`__
     * `Quantize Data2Vec Speech Recognition Model using NNCF PTQ API <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/speech-recognition-quantization/speech-recognition-quantization-data2vec.ipynb>`__

       * See alternative: `Quantize Speech Recognition Models with accuracy control using NNCF PTQ API <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/quantizing-model-with-accuracy-control/speech-recognition-quantization-wav2vec2.ipynb>`__

     * `Semantic segmentation with LRASPP MobileNet v3 and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/torchvision-zoo-to-openvino/lraspp-segmentation.ipynb>`__
     * `Video Recognition using SlowFast and OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/slowfast-video-recognition>`__

       * See alternative: `Live Action Recognition with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/action-recognition-webcam>`__

     * `Semantic Segmentation with OpenVINO™ using Segmenter <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/segmenter-semantic-segmentation>`__
     * `Programming Language Classification with OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/code-language-id>`__
     * `Stable Diffusion Text-to-Image Demo <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/stable-diffusion-v2/stable-diffusion-v2-text-to-image-demo.ipynb>`__

       * See alternative: `Stable Diffusion v2.1 using Optimum-Intel OpenVINO and multiple Intel Hardware <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/stable-diffusion-v2/stable-diffusion-v2-optimum-demo.ipynb>`__

     * `Text-to-Image Generation with Stable Diffusion v2 and OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/stable-diffusion-v2/stable-diffusion-v2-text-to-image.ipynb>`__

       * See alternative: `Stable Diffusion v2.1 using Optimum-Intel OpenVINO and multiple Intel Hardware <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/stable-diffusion-v2/stable-diffusion-v2-optimum-demo.ipynb>`__

     * `Image generation with Segmind Stable Diffusion 1B (SSD-1B) model and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/stable-diffusion-xl/ssd-b1.ipynb>`__
     * `Data Preparation for 2D Medical Imaging <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/ct-segmentation-quantize/data-preparation-ct-scan.ipynb>`__
     * `Train a Kidney Segmentation Model with MONAI and PyTorch Lightning <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/ct-segmentation-quantize/pytorch-monai-training.ipynb>`__
     * `Live Inference and Benchmark CT-scan Data with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/ct-segmentation-quantize/ct-scan-live-inference.ipynb>`__

       * See alternative: `Quantize a Segmentation Model and Show Live Inference <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/ct-segmentation-quantize/ct-segmentation-quantize-nncf.ipynb>`__

     * `Live Style Transfer with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/style-transfer-webcam>`__



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
`www.intel.com <http://www.intel.com/>`__
or from the OEM or retailer.

No computer system can be absolutely secure.

Intel, Atom, Core, Xeon, OpenVINO, and the Intel logo are trademarks
of Intel Corporation in the U.S. and/or other countries.

Other names and brands may be claimed as the property of others.

Copyright © 2024, Intel Corporation. All rights reserved.

For more complete information about compiler optimizations, see our Optimization Notice.

Performance varies by use, configuration and other factors.


