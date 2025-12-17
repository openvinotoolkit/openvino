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



2025.4 - 1 December 2025
#############################################################################################

:doc:`System Requirements <./release-notes-openvino/system-requirements>` | :doc:`Release policy <./release-notes-openvino/release-policy>` | :doc:`Installation Guides <./../get-started/install-openvino>`


What's new
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* More Gen AI coverage and frameworks integrations to minimize code changes

  * New models supported:
  
    * On CPUs & GPUs: Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B, Mistral-Small-24B-Instruct-2501.
    * On NPUs: Gemma-3-4b-it and Qwen2.5-VL-3B-Instruct.
  
  * Preview: Mixture of Experts (MoE) models optimized for CPUs and GPUs, validated for Qwen3-30B-A3B.
  * GenAI pipeline integrations: Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B for enhanced retrieval/ranking, and Qwen2.5VL-7B for video pipeline. 
 
* Broader LLM model support and more model compression techniques

  * Gold support for Windows ML* enables developers to deploy AI models and applications effortlessly across CPUs, GPUs, and NPUs on Intel® Core™ Ultra processor-powered AI PCs.
  * The Neural Network Compression Framework (NNCF) ONNX backend now supports INT8 static post-training quantization (PTQ) and INT8/INT4 weight-only compression to ensure accuracy parity with OpenVINO IR format models. 
    SmoothQuant algorithm support added for INT8 quantization. 
  * Accelerated multi-token generation for GenAI, leveraging optimized GPU kernels to deliver faster inference, smarter KV-cache reuse, and scalable LLM performance.  
  * GPU plugin updates include improved performance with prefix caching for chat history scenarios and enhanced LLM accuracy with dynamic quantization support for INT8. 

* More portability and performance to run AI at the edge, in the cloud or locally

  * Announcing support for Intel® Core™ Ultra Processor Series 3.
  * Encrypted blob format support added for secure model deployment with OpenVINO™ GenAI. 
    Model weights and artifacts are stored and transmitted in an encrypted format, reducing risks of IP theft during deployment.
    Developers can deploy with minimal code changes using OpenVINO GenAI pipelines. 
  * OpenVINO™ Model Server and OpenVINO™ GenAI now extend support for Agentic AI scenarios with new features such as output parsing and improved chat templates for reliable multi-turn interactions, and preview functionality for the Qwen3-30B-A3B model. OVMS also introduces a preview for audio endpoints.
  * NPU deployment is simplified with batch support, enabling seamless model execution across 
    Intel® Core™ Ultra processors while eliminating driver dependencies. Models are reshaped to batch_size=1 before compilation. 
  * The improved NVIDIA Triton Server* integration with OpenVINO backend now enables developers to utilize Intel GPUs or NPUs for deployment. 

OpenVINO™ Runtime
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


CPU Device Plugin
---------------------------------------------------------------------------------------------

* Qwen3-MoE is now supported, with improved performance for Mixture-of-Experts subgraphs.
* Model inference on Intel® Core™ Ultra Series 3 processors has been optimized with AI workload 
  scheduling among P-cores, E-cores and LP E-cores. 
* BitNet model is supported and optimized on both Intel® Xeon® processors and Client processors 
  for 2-bit weight compression support.
* Qwen2/2.5-VL performance and memory footprint has been optimized with 3D Rotary Position Embedding fusion support.
* FP16 model performance on Intel® Xeon® 6 processors with P-cores
  has been enhanced by improving utilization of the underlying AMX FP16 capabilities and graph-level optimizations.
* Inference support for AI workloads is now available on Intel® Xeon® 6 processors with E-cores 
  for Windows 11 and Windows Server.


GPU Device Plugin
---------------------------------------------------------------------------------------------

* Intel® Core™ Ultra Series 3 is fully supported with optimized performance.
* Initial optimization for MoE (Mixture of Experts) has been introduced on Intel® XMX based platforms. 
  Qwen3-30B-A3B model has been enabled.
* Prefix caching performance has been significantly improved on Intel® XMX based platforms.  
* Per-group dynamic quantization is now supported and configurable, providing an alternative 
  when accuracy is insufficient with the default per-token dynamic quantization.
* Performance of Qwen3-Embedding and Qwen3-Reranker models has been optimized on Intel® XMX based platforms. 
* Multiple primitives have been optimized for non-Gen AI models, improving performance of vision 
  embedding models, RNN-based models, and models in the GeekBench AI benchmark tool.
* Runtime memory footprint has been reduced for dynamic shape models.
* 4.2 GB memory allocation limit has been removed, with large allocations now allowed using the ``GPU_ENABLE_LARGE_ALLOCATIONS`` property.
* Querying of discrete and integrated GPUs upon plugin creation has been optimized, extending
  battery life and reducing power consumption.
* Qwen2/2.5-VL performance and memory footprint has been optimized by accelerating image processing 
  on GPU and supporting 3D Rotary Position Embedding fusion.
* XAttention (Block Sparse Attention with Antidiagonal Scoring) is now initially supported on Xe2 architecture to improve time-to-first-token.

NPU Device Plugin
---------------------------------------------------------------------------------------------

* Gemma-3-4B-it and Qwen2.5-VL-3B-Instruct models are now enabled.
* Sliding window mask support for Phi-3 on NPU has been fixed.
* Asynchronous weight processing has been introduced to provide a slight speed-up in importing pre-compiled LLMs.
* LLM prefix caching is introduced to reduce TTFT in long chat scenarios, enabled via property ``NPUW_LLM_ENABLE_PREFIX_CACHING:YES``.
* OpenVINO™ cached models are now memory-mapped and imported within the current Level Zero context, 
  reducing peak memory consumption during imports by eliminating an additional in-memory copy of the compiled model.
* Implicit I/O tensor import is now supported. A shadow-copy tensor is created only when the 
  user-provided tensor address or size is not aligned to the 4K page size.
* NPU deployment is simplified through batch support, which automatically reshapes models to batch 
  size = 1 for compatibility with older driver versions. This enables seamless model execution 
  across all Intel® Core™ Ultra processors regardless of driver version. 
* I/O layout information is now preserved after an export/import cycle. Information is stored inside the blob metadata.

OpenVINO Python API
---------------------------------------------------------------------------------------------

* Python* 3.14 is now supported, including experimental free-threaded mode (3.14t) on Linux and 
  macOS for improved parallel processing performance.
* The issue with the ``PERFORMANCE_HINT`` property not being properly applied in benchmark_app when using custom configurations was fixed. Benchmark results now accurately reflect intended performance settings.
* Precompiled model import from tensor: precompiled models can now be imported directly from ``ov.Tensor`` 
  objects in Python, matching C++ API capabilities and enabling more flexible model deployment workflows.
* The issue that prevented ``import_model()`` from working with large models on Windows was fixed. 
  Models of any size can now be imported on all supported platforms.

OpenVINO C API
---------------------------------------------------------------------------------------------

* Log callback setup has been added to C API: ``ov_util_set_log_callback`` and ``ov_util_reset_log_callback``.

OpenVINO Node.js API
---------------------------------------------------------------------------------------------

* ``Tensor.setShape`` has been added to Node.js API, enabling in-place tensor shape updates from JavaScript/TypeScript 
  without recreating the tensor object. This is useful for dynamic input sizes and batched inference workflows.
* Full 64‑bit integer (BigInt64/Uint64) support is added for tensors and inference I/O, enabling models 
  and pipelines that require 64‑bit index/ID types or high‑range counters.


PyTorch Framework Support 
---------------------------------------------------------------------------------------------

* Support for the padding operations family has been improved by adding new operations 
  (``aten::reflection_padnd`` and ``aten::replication_padnd``) and resolving issues in existing implementations.
* Complex data type support has been added for ``aten::unsqueeze`` and ``aten::cat`` operations.
*  An issue in ``aten::index`` operation with applying boolean masks on specified axes has been resolved.


ONNX Framework Support 
---------------------------------------------------------------------------------------------

* Initial support for sequence data types has been added, beginning with the implementation of ``SequenceAt`` and ``SplitToSequence`` operations.
* The incorrect output shape calculation in the ``ConvTranspose`` operation when using automatic padding (``auto_pad``) has been fixed. 
* The ``LayerNormalization`` operation has been corrected to properly handle scale and bias inputs with flattened shapes by automatically reshaping them to match the input tensor dimensions. 
* A regression causing FP16 model conversion failures due to node not found errors has been fixed.



OpenVINO™ Model Server
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* Agentic use case improvements:
  
  * Tool parsers for the new models Qwen3-Coder-30B and Qwen3-30B-A3B-Instruct have been enabled.
    These models are supported in OpenVINO Runtime as a preview feature and can be evaluated with “tool calling” capabilities.
  * Streaming with “tool calling” for phi-4-mini-instruct and mistral-7B-v0.4 models is now supported.
  * Tool parsers for mistral and hermes have been improved, resolving multiple issues related 
    to complex generated JSON objects and increasing overall response reliability.
  * Guided generation now supports all rules from XGrammar integration. The `response_format` parameter
    can now accept `XGrammar structural tags format <https://github.com/mlc-ai/xgrammar/blob/main/docs/tutorials/structural_tag.md#format-types>`__ (not part of the OpenAI API). Example: `{ "type": "const_string", "value": "Hello World!" }`.

* New capabilities and demos:
  
  * Integration with OpenWebU
  * Integration with Visual Studio Code using the Continue extension
  * Agentic client demo
  * `Audio endpoints <https://docs.openvino.ai/2025/model-server/ovms_demos_audio.html>`__
  * Windows service usage
  * GGUF model pulling

* Deployment improvements:
  
  * GGUF model format can now be deployed directly from Hugging Face Hub for several LLM architectures. 
    Architectures such as Qwen2, Qwen2.5, Qwen3 and Llama3 can be deployed with a single command. See `Loading GGUF models in OVMS demo <https://docs.openvino.ai/2025/model-server/ovms_demos_gguf.html>`__ for details.
  * OpenVINO Model Server can be deployed as a service in the Windows operating system. It can be 
    managed by service configuration management, shared by all running applications, and controlled using a simplified CLI to pull, configure, enable, and disable models. Check the `OVMS documentation <https://docs.openvino.ai/2025/model-server/ovms_docs_deploying_server_service.html>`__  for more details. 
  * Pulling the model in IR format has been extended beyond the OpenVINO™ organization in HuggingFace* Hub. While OpenVINO org models are validated by Intel, a rapidly growing ecosystem of IR-format models from other publishers can now also be pulled and deployed via the OVMS CLI.  Note: The repository needs to be populated by ``optimum-cli export openvino`` command and must include tokenizer model in IR format to be successfully loaded by OpenVINO Model Server. 

*  CLI simplifications for easier deployment:
  
   *  ``--plugin_config`` parameter can now be applied not only to classic models but also to generative pipelines.
   *  ``cache_dir`` now enables compilation caching for both classic models and generative pipelines.
   * ``enable_prefix_caching`` can be used the same way for all target devices.

*  ``--add_to_config`` and ``--remove_from_config``, like ``--list_models``, are now OVMS CLI directives and no longer 
   expect a value. Configuration values should be passed through the following parameters: ``--config_path``, ``--model_repository_path``, ``--model_name``, or ``--model_path``.
* When a service is deployed, the CLI can be simplified by setting the environment variable ``OVMS_MODEL_REPOSITORY_PATH`` to point to the models folder. This automatically applies the default parameters for model management, ensuring that ``config_path`` and ``model_repository_path`` are set correctly.
	
.. dropdown:: Check the CLI example below

	.. code-block:: bash

		ovms -pull -task text_generation OpenVINO/Qwen3-8B-int4 
		ovms -list_models 
		ovms -add_to_models -model_name OpenVINO/Qwen3-8B-int4 
		ovms -remove_from_models -model_name OpenVINO/Qwen3-8B-int4 

* The ``--api_key`` parameter is now available, enabling client authorization using an API key.
* Binding parameters are added for both IPv6 and IPv4 addresses for gRPC and REST interfaces. 
* The metrics endpoint is now compatible with Prometheus v3. The output header type has been updated from JSON to plain text.

Performance improvements: 

* First-token generation performance has been significantly improved for LLM models with GPU 
  acceleration and prefix caching. This is particularly beneficial for agentic use cases, 
  where repeated chat history creates very long contexts that can now be processed much faster. 
  Prefix caching can be enabled with the OVMS CLI parameter ``–enable_prefix_caching true``.
* A new parameter is introduced to increase the allowed prompt length for LLM and VLLM models deployed on NPU.
  The context length can now be extended by adding the CLI parameter ``--max_prompt_length``. 
  The default is 1024 tokens and can be increased up to 10k tokens. Set it to the required value to avoid unnecessary memory usage.
  
  For VLM models running on both NPU and CPU, use a device-specific configuration to apply the setting only to the NPU device
  ``--plugin_config '{device:NPU,{...}}'``. 
* Model loading time has been reduced through compilation cache, with significant improvements on GPU and NPU devices. 
  Enable caching using the ``--cache_dir`` parameter.
* Improved guided generation performance, including support for tool-call guiding.


Audio endpoints added: 

* text to speech endpoint compatible with the OpenAI API - /audio/speech 
* speech to text endpoints compatible with the OpenAI API - /audio/speech_to_text
* /audio/translation - converts provided audio content to English text
* /audio/transcription - converts provided audio content to text in the original language.

Embeddings endpoints improvements: 

* A tokenize endpoint has been added to get tokens before sending the input text to embeddings calculation. This helps assess input length to avoid exceeding the model context.
* Embeddings Model now supports three pooling options: ``CLS``, ``LAST``, and ``MEAN``, improving model compatibility. See `Text Embeddings Models list <https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#text-embeddings-models>`__ for details.

Breaking changes: 

* The old ``embeddings`` and ``reranking`` calculators were removed and replaced by ``embeddings_ov`` and ``reranking_ov``. These new calculators follow the optimum-cli / Hugging Face model structure and support more features. If you use the old calculators, re-export your models and pull the updated versions from Hugging Face.  

Bug fixes: 

* Fixed the model phi-4-mini-instruct generating incorrect responses when context exceeded 4k tokens.  

Neural Network Compression Framework
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* SmoothQuant algorithm support has been added to the int8 post-training quantization method, 
  ``nncf.quantize()`` for the ONNX backend in NNCF, improving the accuracy of transformer-based int8 ONNX models.
* Saving ONNX models with int8 after int8 post-training quantization is now enabled, significantly reducing the model size.  
* Histogram Observer support has been added to the int8 post-training quantization method ``nncf.quantize()`` for more accurate quantization results. 
* MXFP8 precision support has been included in the weight-only compression method ``nncf.compress_weights()``. 

OpenVINO Tokenizers
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* ``BPE`` and ``SplitSpecialTokens`` operations have been optimized, resulting in faster processing 
  of large input strings.
* ``Metaspace`` operation has been improved to support the LLaVA-NeXT-Video model.
* Pair-input support has been added for the Qwen3 tokenizer.

OpenVINO GenAI
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* Tokenizers:

  * ``JsonContainer`` has been added to represent arbitrary string containers.
  * The apply_chat_template() function has been extended with tools and arbitrary values wrapped with ``JsonContainer`` to be used by the chat template 
  * ``get_original_chat_template()`` has been added.
  * ``TextStreamer`` constructor is extended with ``detokenization_params`` to pass to detokenizer.

* LLM Pipeline enhancements: 

  * Preview: Mixture of Experts (MoE) models optimized for CPUs and GPUs, validated for Qwen3-30B-A3B.
  * Parsers have been added for C++, Python, and JavaScript. They structure the generated output 
    splitting into arbitrary sections. For example, thinking and tool calling. 
  * Structured Output grammar compilation time has been improved, and reworked structural tags,
    providing new grammar building blocks for imposing complex output constraints. 
  * ChatHistory (C++, Python, and JavaScript) API is added which stores conversation messages 
    and optional metadata for chat templates. This is a recommended way to manage history instead of ``start/finish_chat()`` for LLMs. See updated `C++ and Python chat_sample <https://github.com/openvinotoolkit/openvino.genai/blob/releases/2025/4/samples/cpp/text_generation/chat_sample.cpp>`__
  * Automatic memory allocation for ContinuousBatching has been improved: it now allocates a 
    fixed number of extra tokens instead of exponential growth, aligning with the GPU plugin.
  * SDPA-based Speculative Decoding has been implemented (used for NPU). 
  * GGUF Q4_1 gibberish has been fixed. 

* VLM Pipeline enhancements: 

  * LLaVA-NeXT-Video, Qwen2-VL, and Qwen2.5-VL now support video input alongside images (samples TBA). 
  * nanoLLaVA, MiniCPM-o-2.6 are now supported, and optimizations for Qwen-VL have been added. 
  * On NPU, absent images and start/finish_chat() are now supported.
  * C API covering VLMPipeline class is added.

* Image generation improvements: 

  * Import/export to .blob is now supported. See stable_diffusion_export_import samples in `C++ <https://github.com/openvinotoolkit/openvino.genai/blob/releases/2025/4/samples/cpp/image_generation/stable_diffusion_export_import.cpp>`__ and `Python <https://github.com/openvinotoolkit/openvino.genai/blob/releases/2025/4/samples/python/image_generation/stable_diffusion_export_import.py>`__.
  * Callbacks now execute in a separate thread, allowing to postprocess intermediate results in parallel with denoising.

* RAG: 

  * ``pad_to_max_length`` and ``batch_size`` config fields have been added, along with the ``LAST_TOKEN`` PoolingType for EmbeddingPipeline.
  * Qwen3 support is added for EmbeddingPipeline and TextRerankPipeline.

* ``-DENABLE_GIL_PYTHON_API=OFF`` cmake option is added to build GenAI with free threaded Python

* Node.js bindings: 
 
  * Issues with launching on NodeJS version 22 and later on Linux have been resolved.
  * ``StructuredOutputConfig`` class enhanced to support structured output generation, including concatenation and tagging features. 
  * ChatHistory is now implemented to manage conversational context, improving tools usage and providing extra context for more accurate prompts.
  * SchedulerConfig entity is introduced, enabling advanced scheduling configurations for pipelines. 
  * Getters for PerfMetrics grammar have been added, allowing developers to retrieve detailed performance data for analysis.
  * Configuration parameter issues in the TextEmbeddingPipeline have been resolved, expanding the list of supported parameters. 
  * Sample list for text generation using JavaScript has been expanded, including new examples for structured output and benchmarking. 

Other Changes and Known Issues
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


Jupyter Notebooks
-----------------------------

New models and use cases: 

* `AFM-4.5B  <https://openvinotoolkit.github.io/openvino_notebooks/?search=chatbot>`__
* `SmolLM2-135M-GGUF and Qwen2.5-0.5B-Instruct-GGUF  <https://openvinotoolkit.github.io/openvino_notebooks/?search=LLM+Instruction-following+pipeline>`__
* `Mistral-Small-24B-Instruct-2501  <https://openvinotoolkit.github.io/openvino_notebooks/?search=chatbot>`__
* `TextRerankPipeline  <https://openvinotoolkit.github.io/openvino_notebooks/?search=Create+a+RAG+system+using+OpenVINO+GenAI+and+LangChain>`__
* `Gemma3 and OpenVINO GenAI  <https://openvinotoolkit.github.io/openvino_notebooks/?search=Gemma3>`__
* `BitNet  <https://openvinotoolkit.github.io/openvino_notebooks/?search=chatbot>`__
* `Qwen3-VL  <https://openvinotoolkit.github.io/openvino_notebooks/?search=Visual-language+assistant+with+Qwen3-VL+and+OpenVINO>`__

.. dropdown:: Deleted notebooks (still available in 2025.3 branch)

	* `Frame interpolation with FILM and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.3/notebooks/film-slowmo>`__
	* `Sound Generation with Stable Audio Open and OpenVINO™  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.3/notebooks/stable-audio>`__
	* `Kosmos-2: Multimodal Large Language Model and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.3/notebooks/kosmos2-multimodal-large-language-model>`__
	* `One Step Sketch to Image translation with pix2pix-turbo and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.3/notebooks/sketch-to-image-pix2pix-turbo>`__
	* `Visual-language assistant with LLaVA and Optimum Intel OpenVINO integration  <https://github.com/openvinotoolkit/openvino_notebooks/blob/2025.3/notebooks/llava-multimodal-chatbot/llava-multimodal-chatbot-optimum.ipynb>`__
	* `PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis with OpenVINO™  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.3/notebooks/pixart>`__
	* `Table Question Answering using TAPAS and OpenVINO™  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.3/notebooks/table-question-answering>`__
	* `Knowledge graphs model optimization using the Intel OpenVINO toolkit <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.3/notebooks/knowledge-graphs-conve>`__
	* `SpeechBrain Emotion Recognition with OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.3/notebooks/speechbrain-emotion-recognition>`__
	* `Accelerate Inference of Sparse Transformer Models with OpenVINO™ and 4th Gen Intel® Xeon® Scalable Processors  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.3/notebooks/sparsity-optimization>`__
	* `Text-to-Image Generation with LCM LoRA and ControlNet Conditioning  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.3/notebooks/lcm-lora-controlnet>`__
	* `Image Generation with Stable Diffusion using OpenVINO TorchDynamo backend <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.3/notebooks/stable-diffusion-torchdynamo-backend>`__
	* `Image Generation with Stable Diffusion and IP-Adapter  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.3/notebooks/stable-diffusion-ip-adapter>`__
	* `TensorFlow Hub models + OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.3/notebooks/tensorflow-hub>`__
	* `Generate creative QR codes with ControlNet QR Code Monster and OpenVINO™  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.3/notebooks/qrcode-monster>`__
	* `Image editing with InstructPix2Pix  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.3/notebooks/instruct-pix2pix-image-editing>`__
	* `Audio compression with EnCodec and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.3/notebooks/encodec-audio-compression>`__


Known Issues
-----------------------------

| **Component: OpenVINO Tokenizers**
| ID: 174531 
| Description:
| Accuracy regression of Mistral-7b-instruct-v0.2 and Mistral-7b-instruct-v0.3 on all devices when executed with OpenVINO GenAI. As a workaround, use the IR converted with OpenVINO 2025.3. The accuracy will be improved with the next release. 

| **Component: OpenVINO GenAI**
| ID: 176777 
| Description:
| Using the ``callback`` parameter with the Python API call generate() in Text2ImagePipeline, Image2ImagePipeline, InpaintingPipeline may cause the process to hang. As a workaround, do not use the ``callback`` parameter. The issue will be resolved in the next release. C++ implementations are not affected. 

Previous 2025 releases
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. dropdown:: 2025.3 - 3 September 2025

	**OpenVINO™ Runtime**

	*Common*

	* Public API has been added to set and reset the log message handling callback. It allows injecting an external log handler to read OpenVINO messages in the user's infrastructure, rather than from log files left by OpenVINO.
	* Build-time optimizations have been introduced to improve developer experience in project compilation.
	* Ability to import a precompiled model from an `ov::Tensor` has been added. Using `ov::Tensor`, which also supports memory-mapped files, to store precompiled models benefits both the OpenVINO caching mechanism and applications using `core.import_model()`. 
	* Several fixes for conversion between different precisions, such as u2 and f32->f4e2,1, have been implemented to improve compatibility with quantized models. 
	* Support for negative indices in GatherElements and GatherND operators has been added to ensure compliance with ONNX standards. 
	* vLLM-OpenVINO integration now supports vLLM API v1. 

	*CPU Device Plugin*
	
	* Sage Attention is now supported. This feature is turned on with the ``ENABLE_SAGE_ATTN`` property, providing a performance boost for 1st token generation in LLMs with long prompts, while maintaining accuracy. 
    * FP16 model performance on 6th generation Intel® Xeon® processors has been enhanced by improving utilization of the underlying AMX FP16 capabilities and graph-level optimizations.
	
	*GPU Device Plugin*
	
	* LLM accuracy has been improved with by-channel key cache compression. Default KV-cache compression has also been switched from by-token to by-channel compression.
    * Gemma3-4b and Qwen-VL VLM performance has been improved on XMX-supporting platforms. 
    * Basic functionalities for dynamic shape custom operations in GPU extension have been enabled. 
    * LoRA performance has been improved for systolic platforms. 

	 
	*NPU Device Plugin*

	* Models compressed as NF4-FP16 are now enabled on NPU. This is the recommended precision for the following models: deepseek-r1-distill-qwen-7b, deepseek-r1-distill-qwen-14b, and qwen2-7b-instruct, providing a reasonable balance between accuracy and performance. This quantization is not supported on Intel® Core™ Ultra Series 1, where only symmetrically quantized channel-wise or group-wise INT4-FP16 models are supported.
	* Peak memory consumption of LLMs on NPU has been significantly reduced when using ahead-of-time compilation.
	* Optimizations for LLM vocabularies (LM Heads) compressed in INT8 asymmetric have been introduced, available with NPU driver 32.0.100.4181 or later. 
	* Accuracy of LLMs with RoPE on longer contexts has been improved. 
	* The NPU plug-in now supports dynamic batch sizes by reshaping the model to a batch size of 1 and concurrently managing multiple inference requests, enhancing performance and optimizing memory utilization. This requires driver 32.0.202.298 or later. 
	* The remote tensor interface has been extended to support tensor creation from files; recent NPU drivers now support memory-mapped inputs/outputs.

	*OpenVINO Python API*

	* TensorVector binding has been enabled to avoid extra copies and speed up PostponedConstant usage. 
	* Support for building experimental free-threaded 3.13t Python API has been added; prebuilt wheels are not distributed yet.
	* Free-threaded Python performance has been improved. 
	* `set_rt_info()` method has been added to Node, Output, and Input to align with `Model.set_rt_info()`.

	*OpenVINO Node.js API*

	* AsyncInferQueue class has been added to support easier implementation of asynchronous inference. The change comes with a benchmark tool to evaluate performance.
	* `Model.reshape` method has been exposed, including type conversion ability and type validation helpers, useful for reshaping LLMs.
	* Support for ov-node types in TypeScript part of bindings has been extended, enabling  direct integration with the JavaScript API. 
	* Wrapping of `compileModel()` method has been fixed to allow checking type of returned objects.  
	* The version of LLMPipeline.generate() that returns strings is now deprecated. Starting with 2026.0.0 LLMPipeline.generate() will return DecodedResults by default. To use the new behavior with current release, set ``["return_decoded_results": true]`` in GenerationConfig.
	
	*PyTorch Framework Support*

	* Tensor concatenation inside loops is now supported, enabling the Paraformer model family.

	**OpenVINO Model Server**

	* Major new features:
  
		* Tool-guided generation has been implemented with the `enable_tool_guided_generation` parameter and `–tool_parser` option to enable model-specific XGrammar configuration for following expected response syntax. It uses dynamic rules based on the generated sequence, increasing model accuracy and minimizing invalid response formats for tools.
		* Tool parser has been added for Mistral-7B-Instruct-v0.3, extending the list of supported models with tool handling.
		* Stream response has been implemented for Qwen3, Hermes3 and Llama3 models, enabling more interactive use with tools.
		* BREAKING CHANGE: Separation of tool parser and reasoning parser has been implemented. Instead of the `response_parser` parameter, use separate parameters: `tool_parser` and `reasoning_parser`, allowing more flexible implementation and configuration on the server. Parsers can now be shared independently between models. Currently, Qwen3 is the only reasoning parser implemented. 
		* Reading of the chat template has been changed from `template.jinja` to `chat_template.jinja` if the chat template is not included in `tokenizer_config.json`.
		* Structured output is now supported with the addition of JSON schema-guided generation using the OpenAI `response_format` field. This parameter enables generation of JSON responses for automation purposes and improvements in response accuracy. See `Structured response in LLM models <https://docs.openvino.ai/2025/model-server/ovms_structured_output.html>`__ article for more details. A script testing the accuracy gain is also included.
		* Enforcement of tool call generation has been implemented using the `tool_call=required` in chat/completions field. This feature forces the model to generate at least one tool response, increasing response reliability while not guaranteeing response validity.
		* `MCP server demo <https://docs.openvino.ai/2025/model-server/ovms_demos_continuous_batching_agent.html>`__ has been updated to include available features.

	* New models and use cases supported:
   
		* Qwen3-embedding and cross-encoders embedding models,
		* Qwen3-reranker,
		* Gemma3 VLM models.
  
	* Deployment improvements:
  
		* Progress bar display has been implemented for model downloads from Hugging Face. For models from the OpenVINO organization, download status is now shown in the logs.
		* `Documentation <https://github.com/openvinotoolkit/model_server/blob/main/docs/pull_optimum_cli.md>`__  on how to build a docker image with optimum-cli is now available, enabling the image to pull any model from Hugging Face and convert it to IR online in one step.
		* Models endpoint for OpenAI has been implemented, returning a list of available models in the expected OpenAI JSON schema for easier integration with existing applications.
		* The package size has been reduced by removing git and gitlfs dependencies, reducing the image by ~15MB. Model files are now pulled from Hugging Face using libgit2 and curl libraries.
		* UTF-8 chat template is now supported out of the box, no additional installation steps on Windows required.
		* Preview functionality for GGUF models has been added for LLM architectures including Qwen2, Qwen2.5, and Llama3. Models can now be deployed directly from HuggingFace Hub by passing `model_id` and file name. Note that accuracy and performance may be lower than with IR format models.
	* Bug fixes:
  
		* Truncation of prompts exceeding model length in embeddings has been implemented.

	**Neural Network Compression Framework**

	* 4-bit data-aware Scale Estimation and AWQ compression methods have been introduced for ONNX models, providing more accurate compression results.
	* NF4 data type is now supported as an FP8 look-up table for faster inference. 
	* New parameter has been added to support a fallback group size in 4-bit weight compression methods. This helps when the specified group size can not be applied, for example, in models with an unusual number of channels in matrix multiplication (matmuls). When enabled with `nncf.AdvancedCompressionParameters(group_size_fallback_mode=ADJUST)`, NNCF automatically adjusts the group size. By default, `nncf.AdvancedCompressionParameters(group_size_fallback_mode=IGNORE)` is used, meaning that NNCF will not compress nodes when the specified group size can not be applied. 
	* Initialization for 4-bit QAT with absorbable LoRA has been enhanced using advanced compression methods (AWQ + Scale Estimation). This replaces the previous basic data-free compression approach, enabling QAT to start with a more accurate model baseline and achieve better final accuracy.
	* External quantizers in the `quantize_pt2e` API have been enabled, including `XNNPACKQuantizer  <https://docs.pytorch.org/executorch/stable/backends-xnnpack.html>`__ and `CoreMLQuantizer <https://docs.pytorch.org/executorch/stable/backends-coreml.html>`__.
	* PyTorch 2.8 is now supported. 

	**OpenVINO Tokenizers**

	* OpenVINO GenAI integration: 

		* Padding side can now be set dynamically during runtime.
		* Tokenizer loading now supports a second input for relevant GenAI pipelines, for example TextRerankPipeline.

	* Two inputs are now supported to accommodate a wider range of tokenizer types. 

	**OpenVINO.GenAI**

	* New OpenVINO GenAI docs homepage: https://openvinotoolkit.github.io/openvino.genai/
	* Transitioned from Jinja2Cpp to Minja, improving chat_template coverage support.
	* Cache eviction algorithms added:

		* KVCrush algorithm
		* Sparse attention prefill

	* Support for Structured Output for flexible and efficient structured generation with XGrammar:

		* C++ and Python samples
		* Constraint sampling with Regex, JSONSchema, EBNF Grammar and Structural tags
		* Compound grammar to combine multiple grammar types (Regex, JSONSchema, EBNF) using Union (|) and Concat (+) operations for more flexible and complex output constraints.

	* GGUF 

		* Qwen3 architecture is now supported 
		* `enable_save_ov_model` property to serialize generated ov::Model as IR for faster LLMPipeline construction next time 

	* LoRA  

		* Dynamic LoRA for NPU has been enabled.
		* Model weights can now be overridden from .safetensors 

	* Tokenizer

		* `padding_side property` has been added to specify padding direction (left or right) 
		* `add_second_input` property to transform Tokenizer from one input to two inputs, used for TextRerankPipeline

	* JavaScript bindings:

		* New pipeline: TextEmbeddingPipeline 
		* PerfMetrics for the LLMPipeline 
		* Implemented getTokenizer into LLMPipeline

	* Other changes:

		* C API for WhisperPipeline has been added 
		* gemma3-4b-it model is now supported in VLM Pipeline
		* Performance metrics for speculative decoding have been extended
		* Qwen2-VL and Qwen2.5-VL have been optimized for GPU 
		* Exporting stateful Whisper models is now supported on NPU out of the box, using `--disable-stateful` is no longer required.
  
	* Dynamic prompts are now enabled by default on NPU: 
  	  
		* Longer contexts are available as preview feature on 32GB Intel® Core™ Ultra Series 2 (with prompt size up to 8..12K tokens). 
		* The default chunk size is 1024 and can be controlled via property `NPUW_LLM_PREFILL_CHUNK_SIZE`. For example, set it to 256 to see the effect on shorter prompts. 
		* `PREFILL_HINT` can be set to STATIC to bring back the old behavior. 

	**Other Changes and Known Issues**

	*Jupyter Notebooks*

	* `FLUX.1-Kontext-dev  <https://openvinotoolkit.github.io/openvino_notebooks/?search=Image-to-image+generation+with+Flux.1+Kontext+and+OpenVINO>`__
	* `GLM-4.1V-9B-Thinking  <https://openvinotoolkit.github.io/openvino_notebooks/?search=GLM-4.1V-9B-Thinking>`__
	* `FLUX.1-Krea-dev  <https://openvinotoolkit.github.io/openvino_notebooks/?search=Image+generation+with+Flux.1+and+OpenVINO>`__
	* `MiniCPM-V-4  <https://openvinotoolkit.github.io/openvino_notebooks/?search=MiniCPM-V>`__
	* `qwen3-embedding  <https://openvinotoolkit.github.io/openvino_notebooks/?search=Text+Embedding+with+Qwen3+and+OpenVINO>`__
	* `qwen3-reranker  <https://openvinotoolkit.github.io/openvino_notebooks/?search=Text+Rerank+with+Qwen3+and+OpenVINO>`__
	* `ACE-Step  <https://openvinotoolkit.github.io/openvino_notebooks/?search=ACE+Step>`__


	*Known Issues*
	
	| **Component: NPU compiler**
	| ID: 169077
	| Description:
	|   miniCPM3-4B model is inaccurate with OV 2025.3 and NPU driver 32.0.100.4239 (latest one available 
		as of OV 2025.3 release). The accuracy will be improved with the next driver release.

	| **Component: NPU plugin**
	| ID: 171934
	| Description:
	|   Whisper model is not functional with transformer v.4.53. Recommended workaround is to use 
	|   transformers=4.52 and optimum-intel=1.25.2 for Whisper model conversion.

	| **Component: NPU plugin**
	| ID: 169074
	| Description:
	|   phi-4-multimodal-instruct is not functional on NPU. Planned to be fixed in future releases.

	| **Component: NPU plugin**
	| ID: 173053
	| Description:
	|   Transformers v4.53 introduce a performance regression for prompts smaller than 1K tokens. For optimal performance, it is recommended to use v4.51.

	| **Component: CPU, GPU plugins**
	| ID: 171208
	| Description:
	|   ChatGLM-3-6B is inaccurate on CPU and GPU.

	| **Component: GPU plugin**
	| ID: 172726
	| Description:
	|   Flux.1-schnell or Flux.1-dev model can functionally fail on Intel® Core™ Ultra Series 1. As a workaround, the model can be converted using OpenVINO 2025.2.

	| **Component: GPU plugin**
	| ID: 171017
	| Description:
	|   If `OV_GPU_DYNAMIC_QUANTIZATION_THRESHOLD` config is explicitly set to less than 64 on XMX-supporting platforms, functional failure can be observed with several GenAI models. 64 is the default value, and setting it to less than 64 is not normally recommended due to performance degradation.

	| **Component: CPU plugin**
	| ID: 172548
	| Description:
	|   Performance regression has been observed on Atom x7835RE with Ubuntu 22.04 OS. This is planned to be fixed in the next release.

	| **Component: CPU plugin**
	| ID: 172518
	| Description:
	|   Qwen2-VL-7b-instruct is inaccurate on with 4th and 6th Gen Intel® Xeon® Scalable processors. As a workaround, the model can be converted using OpenVINO 2025.2. 

.. dropdown:: 2025.2 - 18 June 2025

	**OpenVINO™ Runtime**

	*Common*

	* Better developer experience with shorter build times, due to optimizations and source code refactoring. Code readability has been improved, helping developers understand the components included between different C++ files.
	* Memory consumption has been optimized by expanding the usage of mmap for the GenAI component and introducing the delayed constant weights mechanism.
	* Support for ISTFT operator for GPU has been expanded, improving support of text-to-speech,speech-to-text, and speech-to-speech models, like AudioShake and Kokoro.
	* Models like Behavior Sequence Transformer are now supported, thanks to SparseFillEmptyRows and SegmentMax operators. 
	* google/fnet-base, tf/InstaNet, and more models are now enabled, thanks to DFT operators (discrete Fourier transform) supporting dynamism.
	* "COMPILED_BLOB" hint property is now available to speed up model compilation. The "COMPILED_BLOB" can be a regular or weightless model. For weightless models, the "WEIGHT_PATH" hint provides location of the model weights. 
	* Reading tensor data from file as copy or using mmap feature is now available. 

	*AUTO Inference Mode*

	* Memory footprint in model caching has been reduced by loading the model only for the selected plugin, avoiding duplicate model objects.

	*CPU Device Plugin*
	
	* Per-channel INT8 KV cache compression is now enabled by default, helping LLMs maintain accuracy while reducing memory consumption.
	* Per-channel INT4 KV cache compression is supported and can be enabled using the properties `KEY_CACHE_PRECISION` and `KEY_CACHE_QUANT_MODE`. Some models may be sensitive to INT4 KV cache compression.
	* Performance of encoder-based LLMs has been improved through additional graph-level optimizations, including QKV (Query, Key, and Value) projection and Multi-Head Attention (MHA).
	* SnapKV support has been implemented in the CPU plugin to reduce KV cache size while maintaining comparable performance. It calculates attention scores in PagedAttention for both prefill and decode stages. This feature is enabled by default in OpenVINO GenAI when KV cache eviction is used.

	*GPU Device Plugin*
	
	* Performance of generative models (e.g. large language models, visual language models, image generation models) has been improved on XMX-based platforms (Intel® Core™ Ultra Processor Series 2 built-in GPUs and Intel® Arc™ B Series Graphics) with dynamic quantization and optimization in GEMM and Convolution.
	* 2nd token latency of INT4 generative models has been improved on Intel® Core™ Processors, Series 1.
	* LoRa support has been optimized for Intel® Core™ Processor GPUs and its memory footprint improved, by optimizing the OPS nodes dependency.
	* SnapKV cache rotation now supports accurate token eviction through re-rotation of cache segments that change position after token eviction.
	* KV cache compression is now available for systolic platforms with an update to micro kernel implementation.
	* Improvements to Paged Attention performance and functionality have been made, with support of different head sizes for Key and Value in KV-Cache inputs.
	 
	*NPU Device Plugin*

	* The NPU Plugin can now retrieve options from the compiler and mark only the corresponding OpenVINO properties as supported.
	* The model import path now supports passing precompiled models directly to the plugin using the `ov::compiled_blob` property (Tensor), removing the need for stream access.
	* The `ov::intel_npu::turbo` property is now forwarded both to the compiler and the driver when supported. Using NPU_TURBO may result in longer compile time, increased memory footprint, changes in workload latency, and compatibility issues with older NPU drivers.
	* The same Level Zero context is now used across OpenVINO Cores, enabling remote tensors created through one Core object to be used with inference requests created with another Core object.
	* BlobContainer has been replaced with regular OpenVINO tensors, simplifying the underlying container for a compiled blob.
	* Weightless caching and compilation for LLMs are now available when used with OpenVINO GenAI.
	* LLM accuracy issues with BF16 models have been resolved.
	* The NPU driver is now included in OpenVINO Docker images for Ubuntu, enabling out-of-the-box NPU support without manual driver installation. For instructions, refer to the `OpenVINO Docker documentation <https://github.com/openvinotoolkit/docker_ci/blob/master/docs/npu_accelerator.md>`__.
	* NPU support for FP16-NF4 precision on Intel® Core™ 200V Series processors for models with up to 8B parameters is enabled through symmetrical and channel-wise quantization, improving accuracy while maintaining performance efficiency. FP16-NF4 is not supported on CPUs and GPUs.

	*OpenVINO Python API*

	* Wheel package and source code now include type hinting support (.pyi files), to help Python developers work in IDE. By default, pyi files will be generated automatically but can be triggered manually by developers themselves.
	* The `compiled_blob` property has been added to improve work with compiled blobs for NPU.

	*OpenVINO C API*

	* A new API function is now available, to read IR models directly from memory.

	*OpenVINO Node.js API*

	* OpenVINO GenAI has been expanded for JS package API compliance, to address future LangChain.js user requirements (defined by the LangChain adapter definition). 
	* A new sample has been added, demonstrating OpenVINO GenAI in JS. 

	*PyTorch Framework Support*

	* Complex numbers in the RoPE pattern, used in Wan2.1 model, are now supported. 

	**OpenVINO Model Server**

	* Major new features:

	  * Image generation endpoint - this preview feature enables image generation based on text prompts. The endpoint is compatible with OpenAI API making it easy to integrate with the existing ecosystem.
	  * Agentic AI enablement via support for tools in LLM models. This preview feature allows easy integration of OpenVINO serving with AI Agents.
	  * Model management via OVMS CLI now includes automatic download of OpenVINO models from Hugging Face Hub. This makes it possible to deploy generative pipelines with just a single command and manage the models without extra scripts or manual steps. 

	* Other improvements

	  * VLM models with chat/completion endpoint can now support passing the images as URL or as path to a local file system.
	  * Option to use C++ only server version with support for LLM models. This smaller deployment package can be used both for completion and chat/completions.

	* The following issues have been fixed:

	  * Correct error status now reported in streaming mode.

	* Known limitations

	  * VLM models QuenVL2, QwenVL2.5 and Phi3_VL have low accuracy when deployed in a text generation pipeline with continuous batching. It is recommended to deploy these models in a stateful pipeline which processes the requests serially.

	**Neural Network Compression Framework**

	* Data-free AWQ (Activation-aware Weight Quantization) method for 4-bit weight compression, nncf.compress_weights(), is now available for OpenVINO models. Now it is possible to compress weights to 4-bit with AWQ even without the dataset.
	* 8-bit and 4-bit data-free weight compression, nncf.compress_weights(), is now available  for models in ONNX format. `See example <https://github.com/openvinotoolkit/nncf/tree/develop/examples/llm_compression/onnx/tiny_llama>`__.
	* 4-bit data-aware AWQ (Activation-aware Weight Quantization) and Scale Estimation methods are now available for models in the TorchFX format.
	* TorchFunctionMode-based model tracing is now enabled by default for PyTorch models in nncf.quantize() and nncf.compress_weights().
	* Neural Low-Rank Adapter Search (NLS) Quantization-Aware Training (QAT) for more accurate 4-bit compression of LLMs on downstream tasks is now available. `See example <https://github.com/openvinotoolkit/nncf/tree/develop/examples/llm_compression/torch/downstream_qat_with_nls>`__.
	* Weight compression time for NF4 data type has been reduced.

 **OpenVINO Tokenizers**

	* Regex-based normalization and split operations have been optimized, resulting in significant speed improvements, especially for long input strings.
	* Two-string inputs are now supported, enabling various tasks, including RAG reranking.
	* Sentencepiece char-level tokenizers are now supported to enhance the SpeechT5 TTS model.
	* The tokenization node factory has been exposed to enable OpenVINO GenAI GGUF support.

	**OpenVINO.GenAI**

	* New preview pipelines with C++ and Python samples have been added:

	 * Text2SpeechPipeline,
	 * TextEmbeddingPipeline covering RAG scenario.

	* Visual language modeling (VLMPipeline):

	 * VLM prompt can now refer to specific images. For example, ``<ov_genai_image_0>What’s in the image?`` will prepend the corresponding image to the prompt 
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
	 * support for the SnapKV method for more accurate KV cache eviction, enabled by default when KV cache eviction is used,
	 * preview support for `GGUF models (GGML Unified Format) <https://huggingface.co/models?library=gguf>`__. See the `OpenVINO blog <https://blog.openvino.ai/blog-posts/openvino-genai-supports-gguf-models>`__ for details. 

	**Other Changes and Known Issues**

	*Jupyter Notebooks*

	* `Wan2.1 text to video  <https://openvinotoolkit.github.io/openvino_notebooks/?search=Text+to+Video+generation+with+Wan2.1+and+OpenVINO>`__
	* `Flex2  <https://openvinotoolkit.github.io/openvino_notebooks/?search=Image+generation+with+universal+control+using+Flex.2+and+OpenVINO>`__
	* `DarkIR <https://openvinotoolkit.github.io/openvino_notebooks/?search=Low-Light+Image+Restoration+with+DarkIR+model+using+OpenVINO%E2%84%A2>`__
	* `OpenVoice2 and MeloTTS  <https://openvinotoolkit.github.io/openvino_notebooks/?search=Voice+tone+cloning+with+OpenVoice2+and+MeloTTS+for+Text-to-Speech+by+OpenVINO>`__
	* `InternVideo2 text to video retrieval  <https://openvinotoolkit.github.io/openvino_notebooks/?search=Video+Classification+with+InternVideo2+and+OpenVINO>`__
   * `Kokoro <https://openvinotoolkit.github.io/openvino_notebooks/?search=Text-to-Speech+synthesis+using+Kokoro+and+OpenVINO>`__
   * `Qwen2.5-Omni  <https://openvinotoolkit.github.io/openvino_notebooks/?search=Omnimodal+assistant+with+Qwen2.5-Omni+and+OpenVINO>`__
   * `InternVL3  <https://openvinotoolkit.github.io/openvino_notebooks/?search=Visual-language+assistant+with+InternVL2+and+OpenVINO>`__


	*Known Issues*

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
		 `the demo <https://docs.openvino.ai/2025/openvino-workflow/model-server/ovms_demos_continuous_batching_speculative_decoding.html>`__.
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

  * The OpenVINO property of Affinity API is no longer available. It has been replaced with CPU binding configurations (``ov::hint::enable_cpu_pinning``).
  * The runtime namespace for Python API has been marked as deprecated and designated to be removed for 2026.0. The new namespace structure has been delivered, and migration is possible immediately. Details will be communicated through warnings and via documentation.  
  * Binary operations Node API has been removed from Python API after previous deprecation. 
  * PostponedConstant Python API Update: The PostponedConstant constructor signature is changing for better usability. Update maker from Callable[[Tensor], None] to Callable[[], Tensor]. The old signature will be removed in version 2026.0. 

* Tools:

  * The OpenVINO™ Development Tools package (pip install openvino-dev) is no longer available for OpenVINO releases in 2025.
  * Model Optimizer is no longer available. Consider using the :doc:`new conversion methods <../openvino-workflow/model-preparation/convert-model-to-ir>` instead. For more details, see the `model conversion transition guide <https://docs.openvino.ai/2024/documentation/legacy-features/transition-legacy-conversion-api.html>`__.
  * Intel® Streaming SIMD Extensions (Intel® SSE) are currently not enabled in the binary package by default. They are still supported in the source code form.
  * Legacy prefixes: ``l_``, ``w_``, and ``m_`` have been removed from OpenVINO archive names.

* OpenVINO GenAI:

  * StreamerBase::put(int64_t token)
  * The ``Bool`` value for Callback streamer is no longer accepted. It must now return one of three values of StreamingStatus enum.
  * ChunkStreamerBase is deprecated. Use StreamerBase instead.

* Deprecated OpenVINO Model Server (OVMS) benchmark client in C++ using TensorFlow Serving API.

* NPU Device Plugin: 

  * Removed logic to detect and handle Intel® Core™ Ultra Processors (Series 1) drivers older than v1688. Since v1688 is the earliest officially supported driver, older versions (e.g., v1477) are no longer recommended or supported. 

* Python 3.9 support will be discontinued starting with the OpenVINO 2025.4 and Neural Network Compression Framework (NNCF) 2.19.0. 




Deprecated and to be removed in the future
--------------------------------------------
* ``openvino.Type.undefined`` is now deprecated and will be removed with version 2026.0.
  ``openvino.Type.dynamic`` should be used instead.
* Support for Ubuntu 20.04 has been discontinued due to the end of its standard support.
* The openvino-nightly PyPI module will soon be discontinued. End-users should proceed with the  Simple PyPI nightly repo instead. Find more information in the :doc:`Release policy <./release-notes-openvino/release-policy>`.   
* ``auto shape`` and ``auto batch size`` (reshaping a model in runtime) will be removed in the future. OpenVINO's dynamic shape models are recommended instead.   
* MacOS x86 is no longer recommended for use due to the discontinuation of validation. Full support will be removed later in 2025.   
* The ``openvino`` namespace of the OpenVINO Python API has been redesigned, removing the nested  ``openvino.runtime`` module. The old namespace is now considered deprecated and will be discontinued in 2026.0.   
* Starting with OpenVINO release 2026.0, the CPU plugin will require support for the AVX2 instruction set as a minimum system requirement. The SSE instruction set will no longer be supported. 
* APT & YUM Repositories Restructure:
  Starting with release 2025.1, users can switch to the new repository structure for APT and YUM,
  which no longer uses year-based subdirectories (like “2025”). The old (legacy) structure will
  still be available until 2026, when the change will be finalized.
  Detailed instructions are available on the relevant documentation pages:

  * `Installation guide - yum <https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-yum.html>`__
  * `Installation guide - apt <https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-apt.html>`__

* OpenCV binaries will be removed from Docker images in 2026.
* Starting with the 2026.0 release, OpenVINO will migrate builds based on RHEL 8 to RHEL 9. 
* NNCF ``create_compressed_model()`` method is now deprecated and will be removed in 2026. ``nncf.quantize()`` method is recommended for Quantization-Aware Training of PyTorch models.  
* NNCF optimization methods for TensorFlow models and TensorFlow backend in NNCF are deprecated and will be removed in 2026. 
  It is recommended to use PyTorch analogous models for training-aware optimization methods and OpenVINO IR, PyTorch, and ONNX models for post-training optimization methods from NNCF. 
* The following experimental NNCF methods are deprecated and will be removed in 2026: NAS, Structural Pruning, AutoML, Knowledge Distillation, Mixed-Precision Quantization, Movement Sparsity. 
* Starting with the 2026.0 release, manylinux2014 will be upgraded to manylinux_2_28. 
  This aligns with modern toolchain requirements but also means that CentOS 7 will no longer be supported due to glibc incompatibility.
* With the release of Node.js v22, updated Node.js bindings are now available and compatible with the latest LTS version. These bindings do not support CentOS 7, as they rely on newer system libraries unavailable on legacy systems.

* OpenVINO Model Server: 

  * The dedicated OpenVINO operator for Kubernetes and OpenShift is now deprecated in favor of the recommended KServe operator.
    The OpenVINO operator will remain functional in upcoming OpenVINO Model Server releases but will no longer be actively developed.
    Since KServe provides broader capabilities, no loss of functionality is expected. On the contrary, more functionalities will be accessible and migration between other serving solutions and OpenVINO Model Server will be much easier.
  * TensorFlow Serving (TFS) API support is planned for deprecation. With increasing adoption of the KServe API for classic models 
    and the OpenAI API for generative workloads, usage of the TFS API has significantly declined. Dropping date is to be determined based on the feedback, with a tentative target of mid-2026. 
  * Support for `Stateful models  <https://docs.openvino.ai/2025/model-server/ovms_docs_stateful_models.html>`__  will be deprecated.
    These capabilities were originally introduced for Kaldi audio models which is no longer relevant. Current audio models support relies on the OpenAI API, and pipelines implemented via OpenVINO GenAI library. 
  * `Directed Acyclic Graph Scheduler <https://docs.openvino.ai/2025/model-server/ovms_docs_dag.html>`__ will be deprecated in favor of pipelines managed by MediaPipe scheduler and will be removed in 2026.3. That approach gives more flexibility, includes wider range of calculators and has support for using processing accelerators. 


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
