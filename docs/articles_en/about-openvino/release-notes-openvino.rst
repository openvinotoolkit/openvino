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



2025.2 - 18 June 2025
#############################################################################################

:doc:`System Requirements <./release-notes-openvino/system-requirements>` | :doc:`Release policy <./release-notes-openvino/release-policy>` | :doc:`Installation Guides <./../get-started/install-openvino>`


What's new
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* More Gen AI coverage and frameworks integrations to minimize code changes

  * New models supported on CPUs & GPUs: Phi-4, Mistral-7B-Instruct-v0.3, SD-XL Inpainting 0.1, 
    Stable Diffusion 3.5 Large Turbo, Phi-4-reasoning, Qwen3, and Qwen2.5-VL-3B-Instruct. Mistral 
    7B Instruct v0.3 is also supported on NPUs.​
  * Preview: OpenVINO ™ GenAI introduces a text-to-speech pipeline for the SpeechT5 TTS model, 
    while the new RAG backend offers developers a simplified API that delivers reduced memory usage 
    and improved performance.​
  * Preview: OpenVINO™ GenAI offers a GGUF Reader for seamless integration of llama.cpp based LLMs, 
    with Python and C++ pipelines that load GGUF models, build OpenVINO graphs, and run GPU inference 
    on-the-fly. Validated for popular models: DeepSeek-R1-Distill-Qwen (1.5B, 7B), Qwen2.5 Instruct 
    (1.5B, 3B, 7B) & llama-3.2 Instruct (1B, 3B, 8B).

* Broader LLM model support and more model compression techniques

  * Further optimization of LoRA adapters in OpenVINO GenAI for improved LLM, VLM, and text-to-image 
    model performance on built-in GPUs. Developers can use LoRA adapters to quickly customize models 
    for specialized tasks. ​
  * KV cache compression for CPUs is enabled by default for INT8, providing a reduced memory footprint
    while maintaining accuracy compared to FP16. Additionally, it delivers substantial memory savings 
    for LLMs with INT4 support compared to INT8.​
  * Optimizations for Intel® Core™ Ultra Processor Series 2 built-in GPUs and Intel® Arc™ B Series 
    Graphics with the Intel® XMX systolic platform to enhance the performance of VLM models and hybrid 
    quantized image generation models, as well as improve first-token latency for LLMs through dynamic 
    quantization.

* More portability and performance to run AI at the edge, in the cloud or locally

  * Enhanced Linux* support with the latest GPU driver for built-in GPUs on Intel® Core™ Ultra Processor 
    Series 2 (formerly codenamed Arrow Lake H). ​
  * OpenVINO™ Model Server now offers a streamlined C++ version for Windows and enables improved performance 
    for long-context models through prefix caching, and a smaller Windows package that eliminates the Python 
    dependency. Support for Hugging Face models is now included.​
  * Support for INT4 data-free weights compression for ONNX models implemented in the Neural Network 
    Compression Framework (NNCF)​.
  * NPU support for FP16-NF4 precision on Intel® Core™ 200V Series processors for models with up to 8B 
    parameters is enabled through symmetrical and channel-wise quantization, improving accuracy while 
    maintaining performance efficiency.


OpenVINO™ Runtime 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Common 
---------------------------------------------------------------------------------------------

* Better developer experience with shorter build times, due to optimizations and source code 
  refactoring. Code readability has been improved, helping developers understand the 
  components included between different C++ files.
* Memory consumption has been optimized by expanding the usage of mmap for the GenAI
  component and introducing the delayed constant weights mechanism.
* Support for ISTFT operator for GPU has been expanded, improving support of text-to-speech,
  speech-to-text, and speech-to-speech models, like AudioShake and Kokoro.
* Models like Behavior Sequence Transformer are now supported, thanks to SparseFillEmptyRows
  and SegmentMax operators. 
* google/fnet-base, tf/InstaNet, and more models are now enabled, thanks to DFT operators
  (discrete Fourier transform) supporting dynamism.
* "COMPILED_BLOB" hint property is now available to speed up model compilation.
  The "COMPILED_BLOB" can be a regular or weightless model. For weightless models,
  the "WEIGHT_PATH" hint provides location of the model weights. 
* Reading tensor data from file as copy or using mmap feature is now available. 


AUTO Inference Mode
---------------------------------------------------------------------------------------------

* Memory footprint in model caching has been reduced by loading the model only for the selected 
  plugin, avoiding duplicate model objects.

CPU Device Plugin
---------------------------------------------------------------------------------------------

* Per-channel INT8 KV cache compression is now enabled by default, helping LLMs
  maintain accuracy while reducing memory consumption.
* Per-channel INT4 KV cache compression is supported and can be enabled using the properties
  `KEY_CACHE_PRECISION` and `KEY_CACHE_QUANT_MODE`.
  Some models may be sensitive to INT4 KV cache compression.
* Performance of encoder-based LLMs has been improved through additional graph-level optimizations,
  including QKV (Query, Key, and Value) projection and Multi-Head Attention (MHA).
* SnapKV support has been implemented in the CPU plugin to reduce KV cache size while
  maintaining comparable performance. It calculates attention scores in PagedAttention
  for both prefill and decode stages. This feature is enabled by default in OpenVINO GenAI when
  KV cache eviction is used.

GPU Device Plugin
---------------------------------------------------------------------------------------------

* Performance of generative models (e.g. large language models, visual language models, image
  generation models) has been improved on XMX-based platforms (Intel® Core™ Ultra Processor
  Series 2 built-in GPUs and Intel® Arc™ B Series Graphics) with dynamic quantization and
  optimization in GEMM and Convolution.
* 2nd token latency of INT4 generative models has been improved on Intel® Core™ Processors,
  Series 1.
* LoRa support has been optimized for Intel® Core™ Processor GPUs and its memory footprint
  improved, by optimizing the OPS nodes dependency.
* SnapKV cache rotation now supports accurate token eviction through re-rotation of cache
  segments that change position after token eviction.
* KV cache compression is now available for systolic platforms with an update to micro kernel
  implementation.
* Improvements to Paged Attention performance and functionality have been made, with support
  of different head sizes for Key and Value in KV-Cache inputs.

NPU Device Plugin
---------------------------------------------------------------------------------------------

* The NPU Plugin can now retrieve options from the compiler and mark only the corresponding
  OpenVINO properties as supported.
* The model import path now supports passing precompiled models directly to the plugin using the
  `ov::compiled_blob` property (Tensor), removing the need for stream access.
* The `ov::intel_npu::turbo` property is now forwarded both to the compiler and the driver
  when supported. Using NPU_TURBO may result in longer compile time, increased memory footprint,
  changes in workload latency, and compatibility issues with older NPU drivers.
* The same Level Zero context is now used across OpenVINO Cores, enabling remote tensors created
  through one Core object to be used with inference requests created with another Core object.
* BlobContainer has been replaced with regular OpenVINO tensors, simplifying the underlying container
  for a compiled blob.
* Weightless caching and compilation for LLMs are now available when used with OpenVINO GenAI.
* LLM accuracy issues with BF16 models have been resolved.
* The NPU driver is now included in OpenVINO Docker images for Ubuntu, enabling out-of-the-box NPU 
  support without manual driver installation. For instructions, refer to the
  `OpenVINO Docker documentation <https://github.com/openvinotoolkit/docker_ci/blob/master/docs/npu_accelerator.md>`__.
* NPU support for FP16-NF4 precision on Intel® Core™ 200V Series processors for models with up to 8B parameters is 
  enabled through symmetrical and channel-wise quantization, improving accuracy while maintaining performance 
  efficiency. FP16-NF4 is not supported on CPUs and GPUs.


OpenVINO Python API
---------------------------------------------------------------------------------------------

* Wheel package and source code now include type hinting support (.pyi files), to help
  Python developers work in IDE. By default, pyi files will be generated automatically but
  can be triggered manually by developers themselves.
* The `compiled_blob` property has been added to improve work with compiled blobs for NPU.

OpenVINO C API
---------------------------------------------------------------------------------------------

* A new API function is now available, to read IR models directly from memory.

OpenVINO Node.js API
---------------------------------------------------------------------------------------------

* OpenVINO GenAI has been expanded for JS package API compliance, to address future LangChain.js
  user requirements (defined by the LangChain adapter definition). 
* A new sample has been added, demonstrating OpenVINO GenAI in JS. 

PyTorch Framework Support 
---------------------------------------------------------------------------------------------
* Complex numbers in the RoPE pattern, used in Wan2.1 model, are now supported. 


OpenVINO™ Model Server
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* Major new features:

  * Image generation endpoint - this preview feature enables image generation based on text
    prompts. The endpoint is compatible with OpenAI API making it easy to integrate with the
    existing ecosystem.
  * Agentic AI enablement via support for tools in LLM models. This preview feature allows
    easy integration of OpenVINO serving with AI Agents.
  * Model management via OVMS CLI now includes automatic download of OpenVINO models from
    Hugging Face Hub. This makes it possible to deploy generative pipelines with just a
    single command and manage the models without extra scripts or manual steps.

* Other improvements:

  * VLM models with chat/completion endpoint can now support passing the images as URL or as
    path to a local file system.
  * Option to use C++ only server version with support for LLM models. This smaller deployment
    package can be used both for completion and chat/completions.

* The following issues have been fixed:

  * Correct error status now reported in streaming mode.

* Known limitations:

  * VLM models QuenVL2, QwenVL2.5 and Phi3_VL have low accuracy when deployed in a text
    generation pipeline with continuous batching. It is recommended to deploy these models
    in a stateful pipeline which processes the requests serially.


Neural Network Compression Framework
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* Data-free AWQ (Activation-aware Weight Quantization) method for 4-bit weight compression,
  nncf.compress_weights(), is now available for OpenVINO models. Now it is possible to
  compress weights to 4-bit with AWQ even without the dataset.
* 8-bit and 4-bit data-free weight compression, nncf.compress_weights(), is now available
  for models in ONNX format.
  `See example <https://github.com/openvinotoolkit/nncf/tree/develop/examples/llm_compression/onnx/tiny_llama>`__.
* 4-bit data-aware AWQ (Activation-aware Weight Quantization) and Scale Estimation methods
  are now available for models in the TorchFX format.
* TorchFunctionMode-based model tracing is now enabled by default for PyTorch models in
  nncf.quantize() and nncf.compress_weights().
* Neural Low-Rank Adapter Search (NLS) Quantization-Aware Training (QAT) for more
  accurate 4-bit compression of LLMs on downstream tasks is now available.
  `See example <https://github.com/openvinotoolkit/nncf/tree/develop/examples/llm_compression/torch/downstream_qat_with_nls>`__.
* Weight compression time for NF4 data type has been reduced.


OpenVINO Tokenizers
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* Regex-based normalization and split operations have been optimized, resulting in significant 
  speed improvements, especially for long input strings.
* Two-string inputs are now supported, enabling various tasks, including RAG reranking.
* Sentencepiece char-level tokenizers are now supported to enhance the SpeechT5 TTS model.
* The tokenization node factory has been exposed to enable OpenVINO GenAI GGUF support.

OpenVINO GenAI
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* New preview pipelines with C++ and Python samples have been added:

  * Text2SpeechPipeline,
  * TextEmbeddingPipeline covering RAG scenario.

* Visual language modeling (VLMPipeline):

  * VLM prompt can now refer to specific images. For example, 
    ``<ov_genai_image_0>What’s in the image?`` will prepend the corresponding image to the prompt 
    while ignoring other images. See VLMPipeline’s docstrings for more details.
  * VLM uses continuous batching by default, improving performance.
  * VLMPipeline can now be constructed from in-memory `ov::Model`.
  * Qwen2.5-VL support has been added.

* JavaScript: 

  * JavaScript samples have been added: beam_search_causal_lm and multinomial_causal_lm.
  * An interruption option for LLMPipeline streaming has been introduced.
  
* The following has been added:

  * cache encryption samples demonstrating how to encode OpenVINO’s cached compiled model,
  * LLM ReAct Agent sample capable of calling external functions during text generation,
  * SD3 LoRA Adapter support for Text2ImagePipeline,
  * `ov::genai::Tokenizer::get_vocab()` method for C++ and Python,
  * `ov::Property` as arguments to the `ov_genai_llm_pipeline_create` function for the C API,
  * support for the SnapKV method for more accurate KV cache eviction, enabled by default when 
    KV cache eviction is used,
  * preview support for `GGUF models (GGML Unified Format) <https://huggingface.co/models?library=gguf>`__.
    See the `OpenVINO blog <https://blog.openvino.ai/blog-posts/openvino-genai-supports-gguf-models>`__ for details. 
  

Other Changes and Known Issues
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



Jupyter Notebooks
-----------------------------
* `Wan2.1 text to video  <https://openvinotoolkit.github.io/openvino_notebooks/?search=Text+to+Video+generation+with+Wan2.1+and+OpenVINO>`__
* `Flex2  <https://openvinotoolkit.github.io/openvino_notebooks/?search=Image+generation+with+universal+control+using+Flex.2+and+OpenVINO>`__
* `DarkIR <https://openvinotoolkit.github.io/openvino_notebooks/?search=Low-Light+Image+Restoration+with+DarkIR+model+using+OpenVINO%E2%84%A2>`__
* `OpenVoice2 and MeloTTS  <https://openvinotoolkit.github.io/openvino_notebooks/?search=Voice+tone+cloning+with+OpenVoice2+and+MeloTTS+for+Text-to-Speech+by+OpenVINO>`__
* `InternVideo2 text to video retrieval  <https://openvinotoolkit.github.io/openvino_notebooks/?search=Video+Classification+with+InternVideo2+and+OpenVINO>`__
* `Kokoro <https://openvinotoolkit.github.io/openvino_notebooks/?search=Text-to-Speech+synthesis+using+Kokoro+and+OpenVINO>`__
* `Qwen2.5-Omni  <https://openvinotoolkit.github.io/openvino_notebooks/?search=Omnimodal+assistant+with+Qwen2.5-Omni+and+OpenVINO>`__
* `InternVL3  <https://openvinotoolkit.github.io/openvino_notebooks/?search=Visual-language+assistant+with+InternVL2+and+OpenVINO>`__

Known Issues
-----------------------------

| **Component: GPU**
| ID: 168284
| Description:
|   Using the phi-3 or phi-3.5 model for speculative decoding with large input sequences on GPU 
    may cause an `OpenCL out of resources` error.

| **Component: GPU**
| ID: 168637
| Description:
|   Quantizing the Qwen3-8b model to int4 using the AWQ method results in accuracy issues on GPU.

| **Component: GPU**
| ID: 168889
| Description:
|    Running multiple `benchmark_app` processes simultaneously on Intel® Flex 170 or Intel® Arc™ A770 
    may lead to a system crash. This is due to a device driver issue but appears when using 
    `benchmark_app`.

| **Component: OpenVINO GenAI**
| ID: 167065, 168564, 168360, 168339, 168361
| Description:
|   Models such as Qwen-7B-Chat, Phi4-Reasoning, Llama-3.2-1B-Instruct, Qwen3-8B, and DeepSeek-R1-Distill-* 
    show reduced accuracy in chat scenarios compared to regular generation requests. Currently 
    no workaround is available; a fix is planned for future releases.

| **Component: OpenVINO GenAI**
| ID: 168957
| Description:
|   The stable-diffusion-v1-5 model in FP16 precision shows up to a 10% degradation in the 2nd token
    latency on Intel® Xeon® Platinum 8580. Currently no workaround is available;
    a fix is planned for future releases.

| **Component: ARM**
| ID: 166178
| Description:
|   Performance regression of models on ARM due to an upgrade to the latest ACL. A corresponding issue has 
    been created in the ACL and oneDNN repositories. 

| **Component: OpenVINO Homebrew Installation**
| ID: NA
| Description:
|    Recent OpenVINO releases are currently unavailable via Homebrew due to an issue with
     `protobuf` in the internal Homebrew CI infrastructure. This affects the ability
     to build and publish formula updates. The upstream issue is actively monitored,
     and availability will be restored as soon as it is resolved. Until then, use the
     S3 archive as an alternative to Homebrew on Linux and macOS.
     For more information, see :doc:`Linux Installation Guide <./../get-started/install-openvino/install-openvino-archive-linux>`
     and :doc:`macOS Installation Guide <./../get-started/install-openvino/install-openvino-archive-macos>`.


.. Previous 2025 releases
.. ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
.. ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. dropdown:: 2025.1 - 09 April 2025
   :animate: fade-in-slide-down
   :color: secondary

   **OpenVINO™ Runtime**

   *Common*

   * Delayed weight compression is now available - compressed weights are not stored in memory
     but saved to a file immediately after compression to control memory consumption.
   * Register extensions per frontend (update for extension API)
   * mmaped tensors havve been added, to read ov::Tensor from file on disk using mmap and
     help reduce memory consumption in some scenarios, for example, when using LoRa adapters
     in GenAI.

   *CPU Device Plugin*

   * Dynamic quantization of Fully Connected layers with asymmetric weights is now enabled on
     Intel AVX2 platforms, improving out-of-the-box performance for 8bit/4bit asymmetric
     weight-compressed LLMs.
   * Performance of weight compressed LLMs for long prompts has been optimized on Intel client
     and Xeon platforms, especially on 1st token latency.
   * Optimization of QKV (Query, Key, and Value) projection and MLP (Multilayer Perceptrons)
     fusion for LLMs has been extended to support BF16 on Windows OS for performance
     improvements on AMX platforms.
   * GEMM kernel has been removed from the OpenVINO CPU library, reducing its size.
   * FP8 (alias for f8e4m3 and f8e5m2) model support has been enhanced with optimized FakeConvert
     operator. Compilation time for FP8 LLMs has also been improved.

   *GPU Device Plugin*

   * Second token latency of large language models has been improved on all GPU platforms
     with optimization of translation lookaside buffer (TLB) scenario and
     Group Query Attention (GQA).
   * First token latency of large language models has been improved on
     Intel Core Ultra Processors Series 2 with Paged Attention optimization.
   * Int8 compressed KV-cache is enabled for LLMs by default on all GPU platforms.
   * Performance of VLM (visual language models) has been improved on GPU platforms
     with XMX (Xe Matrix eXtensions).

   *NPU Device Plugin*

   * Support for LLM weightless caching and encryption of LLM blobs.
   * When a model is imported from cache, you can now use ``ov::internal::cached_model_buffer``
     to reduce memory footprint.
   * NF4 (4-bit NormalFloat) inputs/outputs are now supported. E2E support depends on the
     driver version.
   * The following issues have been fixed:

     * for stateful models: update level zero command list when tensor is relocated.
     * for zeContextDestroy error that occurred when applications were using static ov::Cores.

   *OpenVINO Python API*

   * Ability to create a Tensor directly from a Pillow image, eliminating the need for
     casting it to a NumPy array first.
   * Optimization of memory consumption for export_model, read_model, and compile_model methods.

   *OpenVINO Node.js API*

   * Node.js bindings for OpenVINO GenAI are now available in the genai-node npm package
     and bring the simplicity of OpenVINO GenAI API to Node.js applications.

   *PyTorch Framework Support*

   * PyTorch version 2.6 is now supported.
   * Common translators have been implemented to unify decompositions for operations of multiple
     frameworks (PyTorch, TensorFlow, ONNX, JAX) and to support complex tensors.
   * FP8 model conversion is now supported.
   * Conversion of TTS models containing STFT/ISTFT operators has been enabled.

   *JAX Framework Support*

   * JAX 0.5.2 and Flax 0.10.4 have been added to validation.

   *Keras 3 Multi-backend Framework Support*

   * Keras 3.9.0 is now supported.
   * Provided more granular test exclusion mechanism for convenient enabling per operation.

   *TensorFlow Lite Framework Support*

   * Enabled support for models which use quantized tensors between layers in runtime.

   **OpenVINO Model Server**

   * Major new features:

     * VLM support with continuous batching - the endpoint `chat/completion` has been extended
       to support vision models. Now it is possible to send images in the context of chat.
       Vision models can be deployed like the LLM models.
     * NPU acceleration for text generation - now it is possible to deploy LLM and VLM models
       on NPU accelerator. Text generation will be exposed over completions and chat/completions
       endpoints. From the client perspective it works the same way as in GPU and CPU deployment,
       however it doesn't use the continuous batching algorithm, and target is AI PC use cases
       with low concurrency.

   * Other improvements

     * Model management improvements - mediapipe graphs and generative endpoints can be now
       started just using command line parameters without the configuration file. Configuration
       file Json structure for models and graphs has been unified under the
       `models_config_list` section.
     * Updated scalability demonstration using multiple instances, see
       `the demo <https://github.com/openvinotoolkit/model_server/tree/releases/2025/1/demos/continuous_batching/scaling>`__.
     * Increased allowed number of stop words in a request from 4 to 16.
     * Integration with the Visual Studio Code extension of Continue has been enabled making
       it possible to use the assistance of local AI service while writing code.
     * Performance improvements - enhancements in OpenVINO Runtime and also text sampling
       generation algorithm which should increase the throughput in high concurrency load
       scenarios.

   * Breaking changes

     * gRPC server is now optional. There is no default gRPC port set. The ``--port`` parameter
       is mandatory to start the gRPC server. It is possible to start REST API server only with
       the ``--rest_port`` parameter. At least one port number needs to be defined to start
       OVMS server from CLI (--port or --rest_port). Starting OVMS server via C API calls does
       not require any port to be defined.

   * The following issues have been fixed:

     * Handling of the LLM context length - OVMS will now stop generating the text when model
       context is exceeded. An error will be raised when the prompt is longer from the context
       or when the max_tokens plus the input tokens exceeds the model context.
       In addition, it is possible to constrain the max number of generated tokens for all
       users of the model.
     * Security and stability improvements.
     * Cancellation of LLM generation without streaming.

   * Known limitations

     * `Chat/completions` accepts images encoded to base64 format but not as URL links.

   **Neural Network Compression Framework**

   * Preview support for the Quantization-Aware Training (QAT) with LoRA adapters for more
     accurate 4-bit weight compression of LLMs in PyTorch. The ``nncf.compress_weight`` API has
     been extended by a new ``compression_format`` option: ``CompressionFormat.FQ_LORA``, for this
     QAT method. To see how it works, see
     `the sample <https://github.com/openvinotoolkit/nncf/tree/develop/examples/llm_compression/torch/qat_with_lora>`__.
   * Added Activation-aware Weight Quantization and Scale Estimation data-aware 4-bit compression
     methods for PyTorch backend. Now the compression of LLMs can directly be applied to PyTorch
     models to speed up the process.
   * Reduced Generative Pre-trained Transformers Quantization (GPTQ) compression time and peak
     memory usage.
   * Reduced compression time and peak memory usage of data-free mixed precision weight
     compression.
   * New tracing for PyTorch models based on TorchFunctionMode for ``nncf.quantize`` and
     ``nncf.compress_weights``, which does not require torch namespace fixes.
     Disabled by default, it can be enabled by the environment variable ``"NNCF_EXPERIMENTAL_TORCH_TRACING=1”``.
   * Multiple improvements in TorchFX backend to comply with the Torch AO guidelines:

     * The constant folding pass is removed from the OpenVINO Quantizer and the  ``quantize_pt2e``
       function.
     * Support for dynamic shape TorchFX models.

   * Initial steps to adopt custom quantizers in quantize_pt2e within NNCF:

     * The hardware configuration is generalized with the narrow_range parameter.
     * The quantizer parameter calculation code is refactored to explicitly depend on narrow_range.

   * Preview support of the OpenVINO backend in `ExecuTorch <https://github.com/pytorch/executorch>`__
     has been introduced, model quantization is implemented via the function:
     `nncf.experimental.torch.fx.quantize_pt2e <https://openvinotoolkit.github.io/nncf/autoapi/nncf/experimental/torch/fx/index.html#nncf.experimental.torch.fx.quantize_pt2e>`__.
   * PyTorch version 2.6 is now supported.

   **OpenVINO Tokenizers**

   * Support for Unigram tokenization models.
   * Build OpenVINO Tokenizers with installed ICU (International Components for Unicode)
     plugin for reduced binary size.
   * max_length and padding rule parameters can be dynamically adjusted with Tokenizer class
     from OpenVINO GenAI.
   * Remove fast_tokenizer dependency, no core_tokenizers binary in the OpenVINO Tokenizers
     distribution anymore.

   **OpenVINO.GenAI**

   * The following has been added:

     * Preview support for the Token Eviction mechanism for more efficient KVCache memory
       management of LLMs during text generation. Disabled by default.
       `See the sample <https://github.com/openvinotoolkit/openvino.genai/blob/master/site/docs/concepts/optimization-techniques/kvcache-eviction-algorithm.md>`__.
     * LLMPipeline C bindings and JavaScript bindings.
     * StreamerBase::write(int64_t token) and
       StreamerBase::write(const std::vector<int64_t>& tokens).
     * Phi-3-vision-128k-instruct and Phi-3.5-vision-instruct support for VLMPipeline.
     * Added Image2image and inpainting pipelines that support FLUX and Stable-Diffusion-3.

   * LLMPipeline now uses Paged Attention backend by default.
   * Streaming is now performed in a separate thread while the next token is being inferred by LLM.
   * Chat template is applied even with disabled chat mode. Use the ``apply_chat_template`` flag
     to disable chat template in GenerationConfig.
   * Time consuming methods now release Global Interpreter Lock (GIL).

   **Other Changes and Known Issues**

   *Windows PDB Archives*:

   |  Archives containing PDB files for Windows packages are now available.
   |  You can find them right next to the regular archives, in the same folder.

   *Jupyter Notebooks*

   * `Qwen2.5VL <https://openvinotoolkit.github.io/openvino_notebooks/?search=Visual-language+assistant+with+Qwen2.5VL+and+OpenVINO>`__
   * `Phi4-multimodal <https://openvinotoolkit.github.io/openvino_notebooks/?search=Multimodal+assistant+with+Phi-4-multimodal+and+OpenVINO>`__
   * `Gemma3 <https://openvinotoolkit.github.io/openvino_notebooks/?search=Visual-language+assistant+with+Gemma3+and+OpenVINO>`__
   * `SigLIP2 <https://openvinotoolkit.github.io/openvino_notebooks/?search=Zero-shot+Image+Classification+with+SigLIP2>`__
   * `YOLO v12 <https://openvinotoolkit.github.io/openvino_notebooks/?search=Convert+and+Optimize+YOLOv12+real-time+object+detection+with+OpenVINO%E2%84%A2>`__
   * `DeepSeek-VL2 <https://openvinotoolkit.github.io/openvino_notebooks/?search=Visual-language+assistant+using+DeepSeek-VL2+and+OpenVINO>`__
   * `LLasa <https://openvinotoolkit.github.io/openvino_notebooks/?search=Text-to-Speech+synthesis+using+Llasa+and+OpenVINO>`__
   * `GLM4-V <https://openvinotoolkit.github.io/openvino_notebooks/?search=Visual-language+assistant+with+GLM4-V+and+OpenVINO>`__
   * `GOT-OCR 2.0 <https://openvinotoolkit.github.io/openvino_notebooks/?search=Optical+Character+Recognition+with+GOT-OCR+2.0+and+OpenVINO>`__
   * `OmniParser V2 <https://openvinotoolkit.github.io/openvino_notebooks/?search=Screen+Parsing+with+OmniParser-v2.0+and+OpenVINO>`__
   * `Keras3 with OpenVINO backend <https://openvinotoolkit.github.io/openvino_notebooks/?search=Run+inference+in+Keras+3+with+the+OpenVINO%E2%84%A2+IR+backend>`__


   *Known Issues*

   | **Component: NPU**
   | ID: n/a
   | Description:
   |   For LLM runs with prompts longer than the user may set through the MAX_PROMPT_LEN parameter,
       an exception occurs, with a note providing the reason. In the current version of OpenVINO,
       the message is not correct. in future releases, the explanation will be fixed.

   | **Component: NPU**
   | ID: 164469
   | Description:
   |   With the NPU Linux driver release v1.13.0, a new behavior for NPU recovery in kernel
       has been introduced. Corresponding changes in Ubuntu kernels are pending, targeting
       new kernel releases.
   | Workaround:
   |   If inference on NPU crashes, a manual reload of the driver is a recommended option
       ``sudo rmmod intel_vpu`` ``sudo modprobe intel_vpu``.
       A rollback to an earlier version of Linux NPU driver will also work.

   | **Component: GPU**
   | ID: 164331
   | Description:
   |   Qwen2-VL model crashes on some Intel platforms when large inputs are used.
   | Workaround:
   |   Build OpenVINO GenAI from source.

   | **Component: OpenVINO GenAI**
   | ID: 165686
   | Description:
   |   In the VLM ContinuousBatching pipeline, when multiple requests are processed
       using ``add_request()`` and ``step()`` API in multiple threads, the resulting
       text is not correct.
   | Workaround:
   |   Build OpenVINO GenAI from source.

.. dropdown:: 2025.0 - 05 February 2025
   :animate: fade-in-slide-down
   :color: secondary

   **OpenVINO™ Runtime**

   *Common*

   * Support for Python 3.13 has been enabled for OpenVINO Runtime. Tools, like NNCF will follow
     based on their dependency's readiness.

   *AUTO Inference Mode*

   * The issue where AUTO failed to load models to NPU, found on Intel® Core™ Ultra 200V processors
     platform only, has been fixed.
   * When ov::CompiledModel, ov::InferRequest, ov::Model are defined as static variables, the APP
     crash issue during exiting has been fixed.

   *CPU Device Plugin*

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


   *GPU Device Plugin*

   * Intel® Core™ Ultra 200H processors (formerly code named Arrow Lake-H) are now fully supported.
   * ScaledDotProductAttention (SDPA) operator has been enhanced, improving LLM performance for
     OpenVINO GenAI APIs with continuous batching and SDPA-based LLMs with long prompts (>4k).
   * Stateful models are now enabled, significantly improving performance of Whisper models on all
     GPU platforms.
   * Stable Diffusion 3 and FLUX.1 performance has been improved.
   * The issue of a black image output for image generation models, including SDXL, SD3, and
     FLUX.1, with FP16 precision has been solved.


   *NPU Device Plugin*

   * Performance has been improved for Channel-Wise symmetrically quantized LLMs, including Llama2-7B-chat,
     Llama3-8B-instruct, Qwen-2-7B, Mistral-0.2-7B-Instruct, Phi-3-Mini-4K-Instruct, MiniCPM-1B
     models. The best performance is achieved using symmetrically-quantized 4-bit (INT4) quantized
     models.
   * Preview: Introducing NPU support for torch.compile, giving developers the ability to use the
     OpenVINO backend to run the PyTorch API on NPUs. 300+ deep learning models enabled from
     the TorchVision, Timm, and TorchBench repositories.

   *OpenVINO Python API*

   * Ov:OpExtension feature has been completed for Python API. It will enable users to experiment
     with models and operators that are not officially supported, directly with python. It's
     equivalent to the well-known add_extension option for C++.
   * Constant class has been extended with get_tensor_view and get_strides methods that will allow
     advanced users to easily manipulate Constant and Tensor objects, to experiment with data flow
     and processing.

   *OpenVINO Node.js API*

   * OpenVINO tokenizer bindings for JavaScript are now available via the
     `npm package <https://www.npmjs.com/package/openvino-tokenizers-node>`__.
     This is another OpenVINO tool available for JavaScript developers in a way that is most
     natural and easy to use and extends capabilities we are delivering to that ecosystem.

   *TensorFlow Framework Support*

   * The following has been fixed:

     * Output of TensorListLength to be a scalar.
     * Support of corner cases for ToBool op such as scalar input.
     * Correct output type for UniqueWithCounts.

   *PyTorch Framework Support*

   * Preview: Introducing NPU support for torch.compile, giving developers the ability to use
     the OpenVINO backend to run the PyTorch API on NPUs. 300+ deep learning models enabled from
     the TorchVision, Timm, and TorchBench repositories.
   * Preview: Support conversion of PyTorch models with AWQ weights compression, enabling models
     like SauerkrautLM-Mixtral-8x7B-AWQ and similar.


   *OpenVINO Python API*

   * JAX 0.4.38 is now supported.


   *Keras 3 Multi-backend Framework Support*

   * Preview: with Keras 3.8, inference-only OpenVINO backend is introduced, for running model
     predictions using OpenVINO in Keras 3 workflow. To switch to the OpenVINO backend, set the
     KERAS_BACKEND environment variable to "openvino". It supports base operations to infer
     convolutional and transformer models such as MobileNet and Bert from Keras Hub.

     Note: The OpenVINO backend may currently lack support for some operations. This will be
     addressed in upcoming Keras releases as operation coverage is being expanded


   *ONNX Framework Support*

   * Runtime memory consumption for models with quantized weight has been reduced.
   * Workflow which affected reading of 2 bytes data types has been fixed.




   **OpenVINO Model Server**

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


   **Neural Network Compression Framework**

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


   **OpenVINO Tokenizers**

   * WordLevel tokenizer/detokenizer and WordPiece detokenizer models are now supported.
   * UTF-8 (UCS Transformation Format 8) validation with replacement is now enabled by default in
     detokenizer.
   * New models are supported: GLM Edge, ModernBERT, BART-G2P.


   **OpenVINO.GenAI**

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


   **Other Changes and Known Issues**

   *Jupyter Notebooks*

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


   *Known Issues*

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

































Deprecation And Support
+++++++++++++++++++++++++++++

Using deprecated features and components is not advised. They are available to enable a smooth
transition to new solutions and will be discontinued in the future.
For more details, refer to:
`OpenVINO Legacy Features and Components <https://docs.openvino.ai/2025/documentation/legacy-features.html>`__.



Discontinued in 2025
-----------------------------

* Runtime components:

  * The OpenVINO property of Affinity API is no longer available. It has been replaced with CPU
    binding configurations (``ov::hint::enable_cpu_pinning``).
  * The openvino-nightly PyPI module has been discontinued. End-users should proceed with the
    Simple PyPI nightly repo instead. More information in
    `Release Policy <https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/release-policy.html#nightly-releases>`__.

* Tools:

  * The OpenVINO™ Development Tools package (pip install openvino-dev) is no longer available
    for OpenVINO releases in 2025.
  * Model Optimizer is no longer available. Consider using the
    :doc:`new conversion methods <../openvino-workflow/model-preparation/convert-model-to-ir>`
    instead. For more details, see the
    `model conversion transition guide <https://docs.openvino.ai/2024/documentation/legacy-features/transition-legacy-conversion-api.html>`__.
  * Intel® Streaming SIMD Extensions (Intel® SSE) are currently not enabled in the binary
    package by default. They are still supported in the source code form.
  * Legacy prefixes: ``l_``, ``w_``, and ``m_`` have been removed from OpenVINO archive names.

* OpenVINO GenAI:

  * StreamerBase::put(int64_t token)
  * The ``Bool`` value for Callback streamer is no longer accepted. It must now return one of
    three values of StreamingStatus enum.
  * ChunkStreamerBase is deprecated. Use StreamerBase instead.

* NNCF ``create_compressed_model()`` method is now deprecated. ``nncf.quantize()`` method is
  recommended for Quantization-Aware Training of PyTorch and TensorFlow models.

* OpenVINO Model Server (OVMS) benchmark client in C++ using TensorFlow Serving API.







Deprecated and to be removed in the future
--------------------------------------------
* Python 3.9 is now deprecated and will be unavailable after OpenVINO version 2025.4.
* ``openvino.Type.undefined`` is now deprecated and will be removed with version 2026.0.
  ``openvino.Type.dynamic`` should be used instead.
* APT & YUM Repositories Restructure:
  Starting with release 2025.1, users can switch to the new repository structure for APT and YUM,
  which no longer uses year-based subdirectories (like “2025”). The old (legacy) structure will
  still be available until 2026, when the change will be finalized.
  Detailed instructions are available on the relevant documentation pages:

  * `Installation guide - yum <https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-yum.html>`__
  * `Installation guide - apt <https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-apt.html>`__

* OpenCV binaries will be removed from Docker images in 2026.
* Ubuntu 20.04 support will be deprecated in future OpenVINO releases due to the end of
  standard support.
* “auto shape” and “auto batch size” (reshaping a model in runtime) will be removed in the
  future. OpenVINO's dynamic shape models are recommended instead.
* MacOS x86 is no longer recommended for use due to the discontinuation of validation.
  Full support will be removed later in 2025.
* The `openvino` namespace of the OpenVINO Python API has been redesigned, removing the nested
  `openvino.runtime` module. The old namespace is now considered deprecated and will be
  discontinued in 2026.0. A new namespace structure is available for immediate migration.
  Details will be provided through warnings and documentation.




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
