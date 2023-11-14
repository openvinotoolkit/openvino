# OpenVINO Releease Notes {#openvino_release_notes}

@sphinxdirective

The Intel® Distribution of OpenVINO™ toolkit is an open-source solution for
optimizing and deploying AI inference in domains such as computer vision,
automatic speech recognition, natural language processing, recommendation
systems, and generative AI. With its plug-in architecture, OpenVINO enables
developers to write once and deploy anywhere.  We are proud to announce the
release of OpenVINO 2023.2 introducing a range of new features, improvements,
and deprecations aimed at enhancing the developer experience.   

2023.2
########## 

Summary of major features and improvements
++++++++++++++++++++++++++++++++++++++++++

* More Generative AI coverage and framework integrations to minimize code changes 

  * **Expanded model support for direct PyTorch* model conversion** - easily convert 
    additional models directly from PyTorch or execute via torch.compile with
    OpenVINO as the backend.
  * **New and noteworthy models supported** - we have enabled models used for chatbots,
    instruction following, code generation, and many more, including prominent models 
    like Llava, chatGLM, Bark (text to audio) and LCM (Latent Consistency Models, 
    an optimized version of Stable Diffusion).
  * **Easier optimization and conversion of Hugging Face models** - compress LLM models
    to int8 with the Hugging Face Optimum command line interface and export models 
    to the OpenVINO IR format.
  * **OpenVINO is now available on Conan**, a package manager which allows more seamless
    package management for large scale projects for C and C++ developers.

* Broader Large Language Model (LLM) support and more model compression techniques 

  * Accelerate inference for LLM models on CPU and iGPU with the use of int8 and int4 
    model weight compression. 
  * Expanded model support for dynamic shapes for improved performance on GPU.
  * Int4 model format is now supported. Int4 optimized model weights are now 
    supported on Intel® Core™ CPU and iGPU, to accelerate models like
    Llama 2 and chatGLM2.
  * The following int4 model compression formats are supported for inference
    in runtime:
    
    * Generative Pre-training Transformer Quantization (GPTQ); with GPTQ-compressed
      models, you can access them through the Hugging Face repositories.
    * Native int4 compression through Neural Network Compression Framework (NNCF).

* More portability and performance to run AI at the edge, in the cloud, or locally
  
  * In 2023.1 we announced full support for ARM architecture, now we have improved
    performance by enabling FP16 model formats for LLMs and integrating additional 
    acceleration libraries to improve latency.
 
Support Change and Deprecation Notices
++++++++++++++++++++++++++++++++++++++++++

* The OpenVINO™ Development Tools package (pip install openvino-dev) is currently being 
  deprecated and will be removed from installation options and distribution channels
  with 2025.0. 

* Tools:  

  * :doc:`Deployment Manager <openvino_docs_install_guides_deployment_manager_tool>` is
    currently being deprecated and will be removed in the 2024.0 release.
  * Model Optimizer is being deprecated and will be fully supported until the 2025.0
    release. Model conversion to the OpenVINO IR format should be performed through
    OpenVINO Model Converter, which is part of the PyPI package. Follow the 
    :doc:`Model Optimizer to OpenVINO Model Converter transition <openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition>`
    guide for smoother transition. Known limitations are TensorFlow model with
    TF1 Control flow and object detection models. These limitations relate to the
    gap in TensorFlow direct conversion capabilities which will be addressed in
    upcoming releases.
  * Deprecated support for PyTorch 1.13 in Neural Network Compression Framework (NNCF)

* Runtime:

  * Python 3.7 support has been discontinued. 

OpenVINO™ Development Tools
++++++++++++++++++++++++++++++++++++++++++

List of components and their changes:
------------------------------------------

* :doc:`OpenVINO Model Converter tool <openvino_docs_model_processing_introduction>`
  supports the original framework shape format 
* `Neural Network Compression Framework (NNCF) <https://github.com/openvinotoolkit/nncf>`__
  
  * Added data-free INT4 weights compression support for LLMs in OpenVINO format
    with nncf.compress_weights().
  * Preview feature was added to compress model weights to NF4 of LLMs in OpenVINO
    format with nncf.compress_weights().
  * Improved quantization time of LLMs with NNCF PTQ API for nncf.quantize() and 
    nncf.quantize_with_accuracy_control().
  * Added support for SmoothQuant and ChannelAlighnment algorithms in NNCF
    HyperParameter Tuner for automatic optimization of their hyperparameters
    during quantization.  
  * Added quantization support for IF operation of models in OpenVINO format to
    speed up such models.
  * NNCF Post-training Quantization for PyTorch backend is now supported with
    nncf.quantize() and the common implementation of quantization algorithms. 
  * Added support for PyTorch 2.1. PyTorch 1.13 support has been deprecated. 


OpenVINO™ Runtime (previously known as Inference Engine) 
---------------------------------------------------------

* OpenVINO Core 

  * Operations for reference implementations updated from legacy API to API 2.0.
  * Symbolic transformation introduced the ability to remove Reshape operations surrounding MatMul operations.

* OpenVINO Python API 

  * Better support for the ``openvino.properties`` submodule, which now allows the use
    of properties directly, without additional parenthesis. Example use-case: 
    ``{openvino.properties.cache_dir: “./some_path/”}``.
  * Added missing properties: ``execution_devices`` and ``loaded_from_cache``.
  * Improved error propagation on imports from OpenVINO package.

* AUTO device plug-in (AUTO) 

  * Provided additional option to improve performance of cumulative throughput
    (or MULTI), where part of CPU resources can be reserved for GPU inference
    when GPU and CPU are both used for inference. This avoids the performance
    issue of CPU resource contention where there is not enough CPU resources to
    schedule tasks for GPU (`PR#19214 <https://github.com/openvinotoolkit/openvino/pull/19214>`__).
  * Improved support of the NPU plugin: Get optimal_number_of_infer_requests from
    NPU plugin instead of estimating by AUTO in CPU acceleration scenario (when
    running inference with AUTO:NPU,CPU and without other configurations).

* Intel® CPU

  * Introduced support of GPTQ quantized INT4 models, with improved performance
    compared to INT8 weight compressed or FP16 models. In the CPU plugin, the
    gain in performance is achieved by FullyConnected acceleration with 4bit
    weight decompression (`PR #20607 <https://github.com/openvinotoolkit/openvino/pull/20607>`__).
  * Improved performance of INT8 weight-compressed large language models on
    memory constraint platforms, such as 13th Gen Intel Core (`PR #20607 <https://github.com/openvinotoolkit/openvino/pull/20607>`__). 
  * Further reduced memory consumption of select large language models on CPU
    platforms with AMX and AVX512 ISA, by eliminating extra memory copy with
    unified weight layout in matrix multiplication operator (`PR #19575 <https://github.com/openvinotoolkit/openvino/pull/19575>`__). 
  * Fixed performance issue observed in 2023.1 release on selected Xeon CPU
    platform with improved thread workload partitioning matching L2 cache
    utilization for operator like inner_product (`PR #20436 <https://github.com/openvinotoolkit/openvino/pull/20436>`__).
  * Extended support of configuration (enable_cpu_pinning) on Windows platforms
    to allow fine-grain control on CPU resource used for inference workload,
    by binding inference thread to CPU cores (`PR #19418 <https://github.com/openvinotoolkit/openvino/pull/19418>`__).
  * Optimized YoloV8n and YoloV8s model performance for BF16/FP32 precision.
  * Optimized Falcon model on 4th Gen Intel® Xeon® Scalable Processors.
  * Enabled support for FP16 inference precision on ARM.







* Intel® GPU

  * int8 weight compression further improves LLM performance. PR #19548 
  * int4 GPTQ weight compression for better LLM performance 
  * Constant weight optimization has been done for LLMs. Memory consumption 
    and models loading is improved. 
  * Inference performance optimization for LLMs 
  * Optimization for gemm & fc in iGPU. PR #19780 
  * GPU plugin is finally migrated to API 2.0 
  * Performance optimization for PVC PR #19767 
  * Dynamic model support with loop operator 
  * oneDNN v3.3 version support 

* Model Import Updates

  * TensorFlow Framework Support 

    * Supported conversion of models from memory in keras.Model and tf.function formats. PR#19903 
    * Supported TF 2.14. PR#20385 
    * New operations supported: 

      * BatchMatMulV3. PR#20528
      * BitwiseAnd. PR#20340
      * BitwiseNot. PR#20340
      * BitwiseOr. PR#20340
      * BitwiseXor. PR#20340
      * Inv. PR#20720 
      * OnesLike. PR#20385 
      * Selu. PR #19528 
      * TensorArrayCloseV3. PR#20270 
      * TensorArrayConcatV3. PR#20270 
      * TensorArrayGatherV3. PR#20270 
      * TensorArrayReadV3. PR#20270 
      * TensorArrayScatterV3. PR#20270 
      * TensorArrayV3. PR#20270 
      * TensorArrayWriteV3. PR#20270 
      * TensorListLength. PR #19390 
      * TensorListResize. PR #19390 
      * ToBool. PR#20511 
      * TruncateDiv. PR#20615 
      * TruncateMod. PR#20468 
      * XlaConvV2. PR #19466 
      * XlaDotV2. PR#19269 
      * Xlogy. PR#20467 
      * Xlog1py. PR#20500 

    * Fixes:
  
      * Attributes handling for CTCLoss operation. PR#20775 
      * Attributes handling for CumSum operation. PR#20680 
      * PartitionedCall fix for number of external and internal inputs mismatch. PR#20680 
      * Preserving input and output tensor names for conversion of models from memory. PR#19690 
      * 5D case for FusedBatchNorm. PR#19904 

  * PyTorch Framework Support 

    * Supported INT4 GPTQ models 
    * New operations supported: 

      * aten::minimum aten::maximum. PR #19996 
      * aten::broadcast_tensors. PR #19994 
      * added support aten::logical_and, aten::logical_or, aten::logical_not, aten::logical_xor. PR #19981 
      * aten::scatter_reduce and extend aten::scatter. PR #19980 
      * prim::TupleIndex operation. PR #19978 
      * mixed precision in aten::min/max. PR #19936 
      * aten::tile op PR #19645 
      * aten::one_hot PR #19779 
      * aten::prelu. PR #19515 
      * aten::swapaxes. PR #19483 
      * non-boolean inputs for or and and operations. PR #19268 
      * aten::max_poolnd_with_indices 
      * aten::channel_shuffle 
      * aten::amax 
      * aten::amin 
      * aten::clip 
      * aten::clamp_ 
      * aten::pixel_unshuffle 
      * aten::erf 
      * aten::adaptive_avg_poolNd, aten::adaptive_max_poolNd 
      * aten::fill_diagonal_ 
      * aten::fill 
      * aten::as_strided 
      * aten::log1p 
      * aten::numpy_T 
      * aten::feature_dropout 
      * aten::pixel_shuffle 

  * ONNX Framework Support 

    * Supported ONNX version 1.14.1 #18359 
    * New operations supported: 
  
      * GroupNormalization #20694  
      * BlackmanWindow #19428  
      * HammingWindow #19428 
      * HannWindow #19428 

Distribution (where to download the release)
+++++++++++++++++++++++++++++++++++++++++++++

The OpenVINO product selector tool (available at docs.openvino.ai/install) provides easy
access to the right packages that match your desired needs; OS, version, and distribution options.

The 2023.2 release is available via the following distribution channels:   

* pypi.org: https://pypi.org/project/openvino/   
* DockerHub* https://hub.docker.com/u/openvino   
* Release Archives specifically for C++ users can be found here: https://storage.openvinotoolkit.org/repositories/openvino/packages/   
* APT & YUM  
* Homebrew https://formulae.brew.sh/formula/openvino   
* A new distribution channel has been introduced for C++ developers: Conda Forge.  
* Conan C/C++ package manager: https://conan.io/center/recipes/openvino  
* Runtime can now be installed and used via vcpkg C++ package manager (vcpkg install openvino).


OpenVINO Ecosystem
+++++++++++++++++++++++++++++++++++++++++++++

OpenVINO Model Server
--------------------------

* Updated OpenVINO backend to version 2023.2 
* Introduced extension of KServe gRPC API with a stream on input and output. 
  That extension is enabled for the servables with Mediapipe graphs. Mediapipe
  graph is persistent in the scope of the user session. That improves processing
  performance and supports stateful graphs like for tracking algorithms. It also
  enables the use of source calculators. 
* Mediapipe framework has been updated to the version 0.10.3.
* model_api used in the openvino inference mediapipe calculator has been updated
  and included with all its features. 
* Added a demo showcasing gRPC streaming with Mediapipe graph. 
* Added parameters for gRPC quota configuration and changed default gRPC channel
  arguments to add rate limits. It will minimize the risks of impact of the service
  from uncontrolled flow of requests. 
* Updated python clients requirements to match wide range of python versions from 3.6 to 3.11 

Learn more about the changes in https://github.com/openvinotoolkit/model_server/releases  

Jupyter Notebook Tutorials
-----------------------------

* Since the 2023.1 release, the following new notebooks have been added: 

  * LaBSE Cross-lingual Books Alignment With Transformers and OpenVINO™ 
  * LLM chatbot Create LLM-powered Chatbot using OpenVINO™ 
  * Bark Text-to-Speech Text-to-Speech generation using Bark and OpenVINO™ 
  * LLaVA Multimodal Chatbot Visual-language assistant with LLaVA and OpenVINO™ 
  * BLIP-Diffusion - Subject-Driven Generation Subject-driven image generation and
    editing using BLIP Diffusion and OpenVINO™ 
  * DeciDiffusion Image generation with DeciDiffusion and OpenVINO™ 
  * Fast Segment Anything Object segmentations with FastSAM and OpenVINO™ 
  * SoftVC VITS Singing Voice Conversion SoftVC VITS Singing Voice Conversion and OpenVINO™ 
  * QR Code Monster Generate creative QR codes with ControlNet QR Code Monster and OpenVINO™ 
  * Würstchen Text-to-image generation with Würstchen and OpenVINO™ 
  * Distil-Whisper Automatic speech recognition using Distil-Whisper and OpenVINO™ 

* Added optimization support (8-bit quantization, weights compression)
  by NNCF for the following notebooks:

  * Image generation with DeepFloyd IF 
  * Instruction following using Databricks Dolly 2.0 
  * Visual Question Answering and Image Captioning using BLIP 
  * Grammatical Error Correction 
  * Universal segmentation with OneFormer 
  * Visual-language assistant with LLaVA and OpenVINO 
  * Image editing with InstructPix2Pix 
  * MMS: Scaling Speech Technology to 1000+ languages 
  * Image generation with Latent Consistency Model 






@endsphinxdirective