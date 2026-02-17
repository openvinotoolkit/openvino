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



2026.0 - 18 February 2026
#############################################################################################

:doc:`System Requirements <./release-notes-openvino/system-requirements>` | :doc:`Release policy <./release-notes-openvino/release-policy>` | :doc:`Installation Guides <./../get-started/install-openvino>`


What's new
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* More Gen AI coverage and frameworks integrations to minimize code changes

  * New models supported on CPUs & GPUs: GPT-OSS-20B, MiniCPM-V-4_5-8B,  and MiniCPM-o-2.6​
  * New models supported on NPUs: MiniCPM-o-2.6. In addition, NPU support is now available on Qwen2.5-1B-Instruct, Qwen3-Embedding-0.6B, Qwen-2.5-coder-0.5B.​
  * OpenVINO™ GenAI now adds word-level timestamp functionality to the Whisper Pipeline on CPUs, GPUs, and NPUs, enabling more accurate transcriptions and subtitling in line with OpenAI and FasterWhisper implementations.​
  * Phi-3-mini FastDraft model is now available on Hugging Face to accelerate LLM inference on NPUs. FastDraft optimizes speculative decoding for LLMs.
 
* Broader LLM model support and more model compression techniques


  * With the new int4 data-aware weight compression for 3D MatMuls, the Neural Network Compression Framework enables MoE LLMs to run with reduced memory, bandwidth, and improved accuracy compared to data-free schemes-delivering faster, more efficient deployment on resource-constrained devices.​
  * The Neural Network Compression Framework now supports per-layer and per-group Look-Up Tables (LUT) for FP8-4BLUT quantization. This enables fine-grained, codebook-based compression that reduces model size and bandwidth while improving inference speed and accuracy for LLMs and transformer workloads.

* More portability and performance to run AI at the edge, in the cloud or locally

  * OpenVINO™ GenAI adds VLM pipeline support to enhance Agentic AI framework integration.​
  * OpenVINO GenAI now supports speculative decoding for NPUs, delivering improved performance and efficient text generation through a small draft model that is periodically validated by the full-size model.​
  * Preview: NPU compiler integration with the NPU plugin enables ahead-of-time and on-device compilation without relying on OEM driver updates. Developers can enable this feature for a single, ready-to-ship package that reduces integration friction and accelerates time-to-value.​
  * OpenVINO™ Model Server adds enhanced support for audio endpoint plus agentic continuous batching and concurrent runs for improved LLM performance in agentic workflows on Intel CPUs and GPUs.

OpenVINO™ Runtime
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


Common Plugin
---------------------------------------------------------------------------------------------

* API methods that accept filesystem paths as input are now standardized to accept ``std::filesystem::path``. This makes path handling more consistent across OpenVINO™ and simplifies integration in modern C++ codebases that already rely on ``std::filesystem``. Existing ``std::string`` and ``std::wstring`` overloads are still available. 

CPU Device Plugin
---------------------------------------------------------------------------------------------

* GPT-OSS-20B model is now supported, with improved performance for Mixture-of-Experts subgraphs as well as Paged Attention with sink input. 
* Rotary Position Embedding fusion and kernel optimization have been expanded to cover more LLMs, including GLM4, to enhance overall performance. 
* The accuracy issue with Boolean causal masks in ScaledDotProduct Attention when using BF16/FP16 precision has been resolved, addressing accuracy problems in LFM2. 
* XAttention (Block Sparse Attention with Antidiagonal Scoring) is now available as a preview feature to improve Time-To-First-Token (TTFT) performance when processing long context inputs. 
* OneTBB library in OpenVINO™ Windows release has been upgraded from 2021.2.1 to 2021.13.1 
* Linux docker support for offline cores on platforms with multiple numa nodes. 

GPU Device Plugin
---------------------------------------------------------------------------------------------

* Preview support for XAttention on Intel's Xe2/Xe3 architecture to improve TTFT performance. 
* 2nd token latency has been improved for GPT-OSS-20B INT4 model on Intel® Core™ Ultra Series 2, Intel® Core™ Ultra Series 3, and Intel® Arc™ B-Series Graphics. 
* TTFT has been improved for vision language models including Phi-3.5-vision, Phi-4-multimodal, and LLaVa-NeXT-Video. 

NPU Device Plugin
---------------------------------------------------------------------------------------------

* NPU compiler is now included in the OpenVINO™ distribution package as a separate library. This is a preview feature and can be enabled by setting ``ov::intel_npu::compiler_type`` property to ``PREFER_PLUGIN`` to utilize compiler-in-plugin with fallback to compiler-in-driver in case of compatibility or support issues. By default, the NPU will continue using compiler-in-driver. 
* A new model marshaling and serialization mechanism has been implemented to avoid weight copying during compilation, reducing peak memory consumption by up to 1x the original weights size. This mechanism is currently available only when compiler-in-plugin option is enabled.  
* Added support for importing CPU virtual addresses into level zero memory through Remote Tensor APIs. 
* Fixed various issues related to sliding window context handling in models like Gemma and Phi, improved compatibility with the recent transformers packages.
* Introduced new methods to handle attention, ``NPUW_LLM_PREFILL_ATTENTION_HINT`` can be set to ``PYRAMID`` to significantly improve TTFT. The default value is ``STATIC`` (no change to the existing behavior).
* Reduced KV-cache memory consumption, reaching up to 2.5 GB saving for select models on longer contexts (8..12K).

OpenVINO Python API
---------------------------------------------------------------------------------------------

* OpenVINO™ now supports u2, u3, and u6 unsigned integer data types, enabling more efficient memory usage for quantized models. The u3 and u6 types include optimized packing that writes values into three INT8 containers using a concurrency-friendly pattern, ensuring safe concurrent read/write operations without data spanning across byte boundaries. 
* Introduced ``release_gil_before_calling_cpp_dtor`` feature in Python bindings, which optimizes Global Interpreter Lock (GIL) handling during C++ destructor calls. This improves both stability and performance in multi-threaded Python applications. 
* Improved PyThreadState management in the Python API for increased stability and crash prevention in complex threading scenarios. 
* OpenVINO Python package now requires only NumPy as a runtime dependency. The other packaging dependencies have been removed, resulting in a lighter installation footprint and fewer potential dependency conflicts. 
* Added instructions for debugging the Python API on Linux, helping developers troubleshoot and diagnose issues more effectively. 

OpenVINO Node.js API
---------------------------------------------------------------------------------------------

* The Node.js API has been improved with GenAI features: 

  * New parsers have been added to the LLMPipeline to extract structured outputs, reasoning steps, and tool calls from model responses. The parsing layer is fully extensible, enabling developers to plug in their own parsers to tailor how model outputs are interpreted and consumed in downstream applications. 
  * Added support for running Visual-Language Models, enabling richer multimodal applications that combine image, video, and text understanding in a single VLMPipeline. 
  * Introduced a dedicated TextRerankPipeline for re-ranking documents, providing a straightforward way to improve retrieval quality and increase relevance in search and RAG scenarios. 
  * Removed the legacy behaviour whereby ``LLMPipeline.generate()`` could return a string. It now always returns ``DecodedResults``, which provides consistent access to comprehensive information about the generation result, including the output text, scores, performance metrics, and parsed values. 

PyTorch Framework Support 
---------------------------------------------------------------------------------------------

* The ``axis=None`` parameter is now supported for mean reduction operations, allowing for more flexible tensor averaging. 
* Enhanced support for complex data types has been implemented to improve compatibility with vision-language models, such as Qwen. 


ONNX Framework Support 
---------------------------------------------------------------------------------------------

* Major internal refactoring of the graph iteration mechanism has been implemented for improved performance and maintainability. The legacy path can be enabled by setting the ``ONNX_ITERATOR=0`` environment variable. This legacy path is deprecated and will be removed in future releases. 



OpenVINO™ Model Server
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* Improvements in performance and accuracy for GPT-OSS and Qwen3-MOE models. 

  * Improvements in execution performance especially on Intel® Core™ Ultra Series 3 built-in GPUs 
  * Improved chat template examples to fix handling agentic use cases 
  * Improvements in tool parsers to be less restrictive for the generated content and improve response reliability 
  * Better accuracy with INT4 precisions especially with long prompts 
* Improvements in text2speech endpoint  

  * Added voice parameter to choose speaker based on provided embeddings vector 
  * Corrected handling of compilation cache to speed up model loading 
* Improvements in speech2text endpoint: 

  * Added handling for temperature sampling parameter 
  * Support for timestamps in the output 
  
* New parameters have been added to VLM pipelines to control domain name restrictions for image URLs in requests, with optional URL redirection support. By default, all URLs are blocked. 
* NPU execution for text embeddings endpoint (experimental) 
* Exposed tokenizer endpoint for reranker and LLM pipelines 
* Added configurable preprocessing for classic models. Deployed models can include extra preprocessing layers added in at runtime. This can simplify client implementations and enable sending encoded images to models, which are accepted as an array of input. Possible options include:

  * Color format change 
  * Layout change 
  * Scale changes 
  * Mean changes 
* Added support for tool parser compatible with devstral model - take advantage of unsloth/Devstral-Small-2507 model or similar for coding tasks. 
* Updated numerous demos 

  * Audio endpoints 
  * VLM endpoints usage 
  * Agentic demo 
  * Visual Studio Code integration for code assistant 
  * Image classification 
* Optimized file handle usage to reduce the number of open files during high-load operations on Linux deployments.   

Neural Network Compression Framework
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* Extended 4-bit compression data-aware methods (AWQ, Scale Estimation, GPTQ) to support 3D matmuls for more accurate compression of such models as GPT-OSS-20B. 
* Preview support for per-layer and per-block codebooks has been introduced for 4-bit weight compression using the CB4 data type, which helps reduce quantization errors. 
* Added NNCF Profiler for layer-by-layer profiling of OpenVINO™ model activations. This is useful for debugging quantization and compression issues, comparing model variants, and understanding activation distributions. See more details in `Readme <https://github.com/openvinotoolkit/nncf/blob/develop/tools/activation_profiler/README.md>`__ and `Jupyter notebook <https://github.com/openvinotoolkit/nncf/blob/develop/tools/activation_profiler/nncf_profiler_example.ipynb>`__. 
* Added new API method, ``nncf.prune()``, for unstructured pruning of PyTorch models previously supported with the deprecated and removed ``nncf.create_compressed_model()`` method.   
* NNCF optimization methods for TensorFlow models and TensorFlow backend in NNCF are deprecated and removed in 2026. It is recommended to use PyTorch analogous models for training-aware optimization methods and OpenVINO IR, PyTorch, and ONNX models for post-training optimization methods from NNCF. 
* The following experimental NNCF methods are deprecated and removed: NAS, Structural Pruning, AutoML, Knowledge Distillation, Mixed-Precision Quantization, Movement Sparsity. 

OpenVINO Tokenizers
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* Added support for Qwen3 Reranker and LFM2 models. 
* The ``UTF8Validate`` operation has been made available for use in the GGUF GenAI converter. 
* Improved tokenization accuracy through improved metaspace handling when processing special tokens. 

OpenVINO GenAI
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 
* `Conditional Diversity Visual Token Pruning <https://arxiv.org/pdf/2506.10967t>`__ to minimize TTFT of Qwen2/2.5 VL models, this feature is disabled by default and must be turned on. 
* Added word-level timestamp generation for detailed transcriptions with WhisperPipeline. 
* Added ChatHistory API support for VLMPipeline with images and video.  
* Added VLLMParser wrapper. 
* Added universal video tags ``<ov_genai_video_i>`` for VLM models with video support (Qwen2-VL, Qwen2.5-VL, LLaVa-NeXT-Video) 
* Introduced NPU support for text embedding pipelines (for Qwen3-Embeddings-0.6B and similar models).

Other Changes and Known Issues
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


Jupyter Notebooks
-----------------------------

New models and use cases: 

* `LFM2  <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llm-chatbot/llm-chatbot.ipynb>`__
* `Visual-language assistant with Qwen3-VL and OpenVINO <https://openvinotoolkit.github.io/openvino_notebooks/?search=Visual-language+assistant+with+Qwen3-VL>`__
* `Text-to-image generation with Qwen-Image and OpenVINO <https://openvinotoolkit.github.io/openvino_notebooks/?search=qwen-image>`__ (experimental) 
* `Multi-speaker dialogue generation with FireRedTTS-2 and OpenVINO <https://openvinotoolkit.github.io/openvino_notebooks/?search=Fireredtts>`__ (experimental) 
* `Document Parsing using DeepSeek-OCR and OpenVINO <https://openvinotoolkit.github.io/openvino_notebooks/?search=DeepSeek-OCR>`__ (experimental) 
* `Text-to-image generation with Z-Image-Turbo and OpenVINO <https://openvinotoolkit.github.io/openvino_notebooks/?search=Z-Image-Turbo>`__ (experimental) 
* `Text-Image to Video generation with Wan2.2 and OpenVINO <https://openvinotoolkit.github.io/openvino_notebooks/?search=wan2.2>`__ (experimental) 
* `End-to-End Speech Recognition with Fun-ASR-Nano and OpenVINO <https://openvinotoolkit.github.io/openvino_notebooks/?search=Fun-ASR-Nano>`__ (experimental) 
* `Text-to-Speech (TTS) system with Fun-CosyVoice 3.0 and OpenVINO <https://openvinotoolkit.github.io/openvino_notebooks/?search=CosyVoice>`__ (experimental) 
* `PointPillar for 3D object detection <https://openvinotoolkit.github.io/openvino_notebooks/?search=PointPillar>`__

.. dropdown:: Deleted notebooks (still available in 2025.4 branch)

	* `Visual-language assistant with GLM-Edge-V and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/glm-edge-v>`__
	* `Run inference in Keras 3 with the OpenVINO™ IR backend  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/keras-with-openvino-backend>`__
	* `Unified image generation using OmniGen and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/omnigen>`__
	* `Convert a JAX Model to OpenVINO™ IR  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/jax-to-openvino>`__
	* `Text-to-Speech synthesis using Llasa and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/llasa-speech-synthesis>`__
	* `Visual-language assistant with LLaVA Next and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/nano-llava-multimodal-chatbot>`__
	* `Sound Generation with AudioLDM2 and OpenVINO™  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/sound-generation-audioldm2>`__
	* `Text Generation via Prompt Lookup Decoding using OpenVINO™  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/prompt-lookup-decoding>`__
	* `Stable Diffusion with KerasCV and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/stable-diffusion-keras-cv>`__
	* `Text-to-image generation using PhotoMaker and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/photo-maker>`__
	* `Image generation with Sana and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/sana-image-generation>`__
	* `Document Visual Question Answering Using Pix2Struct and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/pix2struct-docvqa>`__
	* `Running OpenCLIP models using OpenVINO™  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/open-clip>`__
	* `Magika: AI powered fast and efficient file type identification using OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/magika-content-type-recognition>`__
	* `Text-to-Video retrieval with S3D MIL-NCE and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/s3d-mil-nce-text-to-video-retrieval>`__
	* `Named entity recognition with OpenVINO™  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/named-entity-recognition>`__
	* `Big Transfer Image Classification Model Quantization with NNCF in OpenVINO™  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/big-transfer-quantization>`__
	* `Image generation with StableCascade and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/stable-cascade-image-generation>`__
	* `Optical Character Recognition with GOT-OCR 2.0 and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/got-ocr2>`__
	* `Optimizing PyTorch models with Neural Network Compression Framework of OpenVINO™ by 8-bit quantization.  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/pytorch-quantization-aware-training>`__
	* `Language-Visual Saliency with CLIP and OpenVINO™  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/clip-language-saliency-map>`__
	* `LocalAI and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/localai>`__
	* `Grammatical Error Correction with OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/grammar-correction>`__
	* `Stable Fast 3D Mesh Reconstruction and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/stable-fast-3d>`__
	* `Colorize grayscale images using DDColor and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/ddcolor-image-colorization>`__
	* `Image Generation with Tiny-SD  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/tiny-sd-image-generation>`__
	* `Text-to-Speech synthesis using OuteTTS and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/outetts-text-to-speech>`__
	* `Text-to-Music generation using Riffusion and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/riffusion-text-to-music>`__
	* `Video Classification with InternVideo2 and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/intern-video2-classiciation>`__
	* `Single-step image generation using SDXL-turbo and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/sdxl-turbo>`__
	* `Optimizing TensorFlow models with Neural Network Compression Framework of OpenVINO™ by 8-bit quantization  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/tensorflow-quantization-aware-training>`__
	* `Animating Open-domain Images with DynamiCrafter and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/dynamicrafter-animating-images>`__
	* `Create a native Agent with OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/llm-native-agent-react>`__
	* `Visual-language assistant with GLM4-V and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.4/notebooks/glm4-v>`__


Known Issues
-----------------------------

| **Component: Optimum**
| ID: 179936  
| Description:
| phi-4-multimodal instruct model isn't functional when converted using optimum-cli as channel-wise one (with -group-size -1) with OpenVINO 2026.0. It's recommended to use for the conversion OV 2025.4/OV 2025.4.1 

| **Component: OpenVINO Runtime**
| ID: 180693 
| Description:
| Qwen3-30B-A3B converted with newer transformers doesn't work, recommend using transformers 4.55.4 for model conversion which was verified and worked.

| **Component: OpenVINO GenAI**
| ID: 179973  
| Description:
| Qwen2-vl, Qwen-2.5VL, Qwen3-VL dense models may not work through GenAI API with GPU, due internal issue on model transformation level

| **Component: OpenVINO Runtime**
| ID: 180696 
| Description:
| 2nd (and further) latency degradation for Qwen3-MOE family, including lack of ability to fit a model on iGPU, due high memory consumption and potential graph corruption. Problem affects only IRs generated with 2026.0, former IRs generated with 2025.4 will work properly. 

| **Component: OpenVINO Runtime**
| ID: 179009 
| Description:
| Memory leak for static builds with HybridCRT enabled; impacts Windows only



Deprecation And Support
+++++++++++++++++++++++++++++

Using deprecated features and components is not advised. They are available to enable a smooth
transition to new solutions and will be discontinued in the future.
For more details, refer to:
`OpenVINO Legacy Features and Components <https://docs.openvino.ai/2026/documentation/legacy-features.html>`__.



Discontinued in 2026.0
-----------------------------

* The deprecated ``openvino.runtime`` namespace has been removed. Please use the ``openvino`` namespace directly. 
* The deprecated ``openvino.Type.undefined`` has been removed. Please use ``openvino.Type.dynamic`` instead. 
* The PostponedConstant constructor signature has been updated for improved usability: 

  * Old (removed): ``Callable[[Tensor], None]``
  * New: ``Callable[[], Tensor]``
* The deprecated OpenVINO™ GenAI predefined generation configs were removed. 
* The deprecated OpenVINO GenAI support for whisper stateless decoder model has been removed. Please use a stateful model. 
* The deprecated OpenVINO GenAI StreamerBase ``put`` method, ``bool`` return type for callbacks, and ``ChunkStreamer`` class has been removed. 
* NNCF ``create_compressed_model()`` method is now deprecated and removed in 2026. Please use ``nncf.prune()`` method for unstructured pruning and ``nncf.quantize()`` for INT8 quantization. 
* NNCF optimization methods for TensorFlow models and TensorFlow backend in NNCF are deprecated and removed in 2026. It is recommended to use PyTorch analogous models for training-aware optimization methods and OpenVINO™ IR, PyTorch, and ONNX models for post-training optimization methods from NNCF.  
* The following experimental NNCF methods are deprecated and removed: NAS, Structural Pruning, AutoML, Knowledge Distillation, Mixed-Precision Quantization, Movement Sparsity. 
* CPU plugin now requires support for the AVX2 instruction set as a minimum system requirement. The SSE instruction set will no longer be supported. 
* OpenVINO™ migrated builds based on RHEL 8 to RHEL 9. 
* manylinux2014 upgraded to manylinux_2_28. This aligns with modern toolchain requirements but also means that CentOS 7 will no longer be supported due to glibc incompatibility. 




Deprecated and to be removed in the future
--------------------------------------------
* Support for Ubuntu 20.04 has been discontinued due to the end of its standard support. 
* The openvino-nightly PyPI module will soon be discontinued. End-users should proceed with the Simple PyPI nightly repo instead. Find more information in the `Release policy <https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/release-policy.html>`__. 
* ``auto shape`` and ``auto batch`` size (reshaping a model in runtime) will be removed in the future. OpenVINO™'s dynamic shape models are recommended instead. 
* MacOS x86 is no longer recommended for use due to the discontinuation of support. 
* APT & YUM Repositories Restructure:
  Starting with release 2025.1, users can switch to the new repository structure for APT and YUM,
  which no longer uses year-based subdirectories (like “2025”). The old (legacy) structure will
  still be available until 2026, when the change will be finalized.
  Detailed instructions are available on the relevant documentation pages:

  * `Installation guide - yum <https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-yum.html>`__
  * `Installation guide - apt <https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-apt.html>`__

* OpenCV binaries will be removed from Docker images in 2026.
* OpenCV binaries will be removed from Docker images in 2026. 
* With the release of Node.js v22, updated Node.js bindings are now available and compatible with the latest LTS version. These bindings do not support CentOS 7, as they rely on newer system libraries unavailable on legacy systems. 
* Starting with 2026.0 release major internal refactoring of the graph iteration mechanism has been implemented for improved performance and maintainability. The legacy path can be enabled by setting the ONNX_ITERATOR=0 environment variable. This legacy path is deprecated and will be removed in future releases. 

* OpenVINO Model Server: 

  * The dedicated OpenVINO operator for Kubernetes and OpenShift is now deprecated in favor of the recommended KServe operator.
    The OpenVINO operator will remain functional in upcoming OpenVINO Model Server releases but will no longer be actively developed.
    Since KServe provides broader capabilities, no loss of functionality is expected. On the contrary, more functionalities will be accessible and migration between other serving solutions and OpenVINO Model Server will be much easier.
  * TensorFlow Serving (TFS) API support is planned for deprecation. With increasing adoption of the KServe API for classic models 
    and the OpenAI API for generative workloads, usage of the TFS API has significantly declined. Dropping date is to be determined based on the feedback, with a tentative target of mid-2026. 
  * Support for `Stateful models  <https://docs.openvino.ai/2026/model-server/ovms_docs_stateful_models.html>`__  will be deprecated.
    These capabilities were originally introduced for Kaldi audio models which is no longer relevant. Current audio models support relies on the OpenAI API, and pipelines implemented via OpenVINO GenAI library. 
  * `Directed Acyclic Graph Scheduler <https://docs.openvino.ai/2026/model-server/ovms_docs_dag.html>`__ will be deprecated in favor of pipelines managed by MediaPipe scheduler and will be removed in 2026.3. That approach gives more flexibility, includes wider range of calculators and has support for using processing accelerators. 


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

Copyright © 2026, Intel Corporation. All rights reserved.

For more complete information about compiler optimizations, see our Optimization Notice.

Performance varies by use, configuration and other factors.
