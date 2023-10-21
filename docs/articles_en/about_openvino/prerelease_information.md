# Pre-release Information {#prerelease_information}

@sphinxdirective

.. meta::
   :description: Check the pre-release information that includes a general 
                 changelog for each version of OpenVINO Toolkit published under 
                 the current cycle.

To ensure you can test OpenVINO's upcoming features even before they are officially released, 
OpenVINO developers continue to roll out pre-release software. On this page you can find
a general changelog for each version published under the current cycle.

Your feedback on these new features is critical for us to make the best possible production quality version.
Please file a github Issue on these with the label “pre-release” so we can give it immediate attention. Thank you.

.. note:: 

   These versions are pre-release software and have not undergone full validation or qualification. OpenVINO™ toolkit pre-release is:

   * NOT to be incorporated into production software/solutions.
   * NOT subject to official support.
   * Subject to change in the future.
   * Introduced to allow early testing and get early feedback from the community.

   .. button-link:: https://github.com/openvinotoolkit/openvino/issues/new?assignees=octocat&labels=Pre-release%2Csupport_request&projects=&template=pre_release_feedback.yml&title=%5BPre-Release+Feedback%5D%3A
      :color: primary
      :outline:

      :material-regular:`feedback;1.4em` Share your feedback






.. dropdown:: OpenVINO Toolkit 2023.2 Dev 22.09.2023
   :animate: fade-in-slide-down
   :color: primary
   :open:

   **What's Changed:**
   
   * CPU runtime: 

     * Optimized Yolov8n and YoloV8s models on BF16/FP32. 
     * Optimized Falcon model on 4th Generation Intel® Xeon® Scalable Processors. 

   * GPU runtime:  

     * int8 weight compression further improves LLM performance. PR #19548 
     * Optimization for gemm & fc in iGPU. PR #19780 

   * TensorFlow FE: 

     * Added support for Selu operation. PR #19528 
     * Added support for XlaConvV2 operation. PR #19466 
     * Added support for TensorListLength and TensorListResize operations. PR #19390 

   * PyTorch FE: 

     * New operations supported 
  
       * aten::minimum aten::maximum. PR #19996 
       * aten::broadcast_tensors. PR #19994 
       * added support aten::logical_and, aten::logical_or, aten::logical_not, aten::logical_xor. PR #19981 
       * aten::scatter_reduce and extend aten::scatter. PR #19980 
       * prim::TupleIndex operation. PR #19978 
       * mixed precision in aten::min/max. PR #19936 
       * aten::tile op PR #19645 
       * aten::one_hot PR #19779 
       * PReLU. PR #19515 
       * aten::swapaxes. PR #19483 
       * non-boolean inputs for __or__ and __and__ operations. PR #19268 

   * Torchvision NMS can accept negative scores. PR #19826 
   * New openvino_notebooks: 

     * Visual Question Answering and Image Captioning using BLIP 

   **Fixed GitHub issues**

   * Fixed #19784 “[Bug]: Cannot install libprotobuf-dev along with libopenvino-2023.0.2 on Ubuntu 22.04” with PR #19788 
   * Fixed #19617 “Add a clear error message when creating an empty Constant” with PR #19674 
   * Fixed #19616 “Align openvino.compile_model and openvino.Core.compile_model functions” with PR #19778 
   * Fixed #19469 “[Feature Request]: Add SeLu activation in the OpenVino IR (TensorFlow Conversion)” with PR #19528 
   * Fixed #19019 “[Bug]: Low performance of the TF quantized model.” With PR #19735 
   * Fixed #19018 “[Feature Request]: Support aarch64 python wheel for Linux” with PR #19594 
   * Fixed #18831 “Question: openvino support for Nvidia Jetson Xavier ?” with PR #19594 
   * Fixed #18786 “OpenVINO Wheel does not install Debug libraries when CMAKE_BUILD_TYPE is Debug #18786” with PR #19197 
   * Fixed #18731 “[Bug] Wrong output shapes of MaxPool” with PR #18965 
   * Fixed #18091 “[Bug] 2023.0 Version crashes on Jetson Nano - L4T - Ubuntu 18.04” with PR #19717 
   * Fixed #7194 “Conan for simplifying dependency management” with PR #17580 

 
   **Acknowledgements:**

   Thanks for contributions from the OpenVINO developer community: 
   
   * @siddhant-0707, 
   * @PRATHAM-SPS, 
   * @okhovan 


.. dropdown:: OpenVINO Toolkit 2023.1.0.dev20230728
   :animate: fade-in-slide-down
   :color: secondary

   `Check on GitHub <https://github.com/openvinotoolkit/openvino/releases/tag/2023.1.0.dev20230811>`__ 

   **New features:**
   
   * CPU runtime: 

     * Enabled weights decompression support for Large Language models (LLMs). The implementation 
       supports avx2 and avx512 HW targets for Intel® Core™ processors for improved 
       latency mode (FP32 VS FP32+INT8 weights comparison). For 4th Generation Intel® Xeon® 
       Scalable Processors (formerly Sapphire Rapids) this INT8 decompression feature provides 
       performance improvement, compared to pure BF16 inference.
     * Reduced memory consumption of compile model stage by moving constant folding of Transpose 
       nodes to the CPU Runtime side.  
     * Set FP16 inference precision by default for non-convolution networks on ARM. Convolution 
       network will be executed in FP32.  

   * GPU runtime: Added paddings for dynamic convolutions to improve performance for models like 
     Stable-Diffusion v2.1. 

   * Python API: 

     * Added the ``torchvision.transforms`` object to OpenVINO preprocessing.  
     * Moved all python tools related to OpenVINO into a single namespace, 
       improving user experience with better API readability. 

   * TensorFlow FE: 

     * Added support for the TensorFlow 1 Checkpoint format. All native TensorFlow formats are now enabled. 
     * Added support for 8 new operations: 

       * MaxPoolWithArgmax 
       * UnravelIndex 
       * AdjustContrastv2 
       * InvertPermutation 
       * CheckNumerics 
       * DivNoNan 
       * EnsureShape 
       * ShapeN 

   * PyTorch FE: 

     * Added support for 6 new operations. To know how to enjoy PyTorch models conversion follow 
       this `Link <https://docs.openvino.ai/2023.1/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_PyTorch.html#experimental-converting-a-pytorch-model-with-pytorch-frontend>`__ 

       * aten::concat 
       * aten::masked_scatter 
       * aten::linspace 
       * aten::view_as 
       * aten::std 
       * aten::outer 
       * aten::broadcast_to 

   **New openvino_notebooks:**

   * `245-typo-detector <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/245-typo-detector>`__
     : English Typo Detection in sentences with OpenVINO™ 

   * `247-code-language-id <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/247-code-language-id/247-code-language-id.ipynb>`__
     : Identify the programming language used in an arbitrary code snippet 

   * `121-convert-to-openvino <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/121-convert-to-openvino>`__
     : Learn OpenVINO model conversion API 

   * `244-named-entity-recognition <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/244-named-entity-recognition>`__
     : Named entity recognition with OpenVINO™ 

   * `246-depth-estimation-videpth <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/246-depth-estimation-videpth>`__
     : Monocular Visual-Inertial Depth Estimation with OpenVINO™ 

   * `248-stable-diffusion-xl <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/248-stable-diffusion-xl>`__
     : Image generation with Stable Diffusion XL 

   * `249-oneformer-segmentation <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/249-oneformer-segmentation>`__
     : Universal segmentation with OneFormer 


.. dropdown:: OpenVINO Toolkit 2023.1.0.dev20230728
   :animate: fade-in-slide-down
   :color: secondary
   
   `Check on GitHub <https://github.com/openvinotoolkit/openvino/releases/tag/2023.1.0.dev20230728>`__ 
   
   **New features:**
   
   * Common:
   
     - Proxy & hetero plugins have been migrated to API 2.0, providing enhanced compatibility and stability. 
     - Symbolic shape inference preview is now available, leading to improved performance for Large Language models (LLMs).

   * CPU Plugin: Memory efficiency for output data between CPU plugin and the inference request has been significantly improved, 
     resulting in better performance for LLMs.  
   * GPU Plugin: 

     - Enabled support for dynamic shapes in more models, leading to improved performance. 
     - Introduced the 'if' and DetectionOutput operator to enhance model capabilities. 
     - Various performance improvements for StableDiffusion, SegmentAnything, U-Net, and Large Language models. 
     - Optimized dGPU performance through the integration of oneDNN 3.2 and fusion optimizations for MVN, Crop+Concat, permute, etc. 

   * Frameworks:

     - PyTorch Updates: OpenVINO now supports originally quantized PyTorch models, including models produced with the Neural Network Compression Framework (NNCF).
     - TensorFlow FE: Now supports Switch/Merge operations, bringing TensorFlow 1.x control flow support closer to full compatibility and enabling more models.
     - Python API: Python Conversion API is now the primary conversion path, making it easier for Python developers to work with OpenVINO.

   * NNCF: Enabled SmoothQuant method for Post-training Quantization, offering more techniques for quantizing models.

   **Distribution:**

   * Added conda-forge pre-release channel, simplifying OpenVINO pre-release installation with "conda install -c "conda-forge/label/openvino_dev" openvino" command.
   * Python API is now distributed as a part of conda-forge distribution, allowing users to access it using the command above.
   * Runtime can now be installed and used via vcpkg C++ package manager, providing more flexibility in integrating OpenVINO into projects.

   **New models:**

   * Enabled Large Language models such as open-llama, bloom, dolly-v2, GPT-J, llama-2, and more. We encourage users to try running their custom LLMs and share their feedback with us! 
   * Optimized performance for Stable Diffusion v2.1 (FP16 and INT8 for GPU) and Clip (CPU, INT8) models, improving their overall efficiency and accuracy. 
   
   **New openvino_notebooks:**

   * `242-freevc-voice-conversion <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/242-freevc-voice-conversion>`__ - High-Quality Text-Free One-Shot Voice Conversion with FreeVC
   * `241-riffusion-text-to-music <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/241-riffusion-text-to-music>`__ - Text-to-Music generation using Riffusion
   * `220-books-alignment-labse <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/220-cross-lingual-books-alignment>`__ - Cross-lingual Books Alignment With Transformers
   * `243-tflite-selfie-segmentation <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/243-tflite-selfie-segmentation>`__ - Selfie Segmentation using TFLite


.. dropdown:: OpenVINO Toolkit 2023.1.0.dev20230623
   :animate: fade-in-slide-down
   :color: secondary

   The first pre-release for OpenVINO 2023.1, focused on fixing bugs and performance issues.

   `Check on GitHub <https://github.com/openvinotoolkit/openvino/releases/tag/2023.1.0.dev20230623>`__ 
   

.. dropdown:: OpenVINO Toolkit 2023.0.0.dev20230407
   :animate: fade-in-slide-down
   :color: secondary

   Note that a new distribution channel has been introduced for C++ developers: `Conda Forge <https://anaconda.org/conda-forge/openvino>`__ 
   (the 2022.3.0 release is available there now).

   * ARM device support is improved:

     * increased model coverage up to the scope of x86, 
     * dynamic shapes enabled, 
     * performance boosted for many models including BERT,
     * validated for Raspberry Pi 4 and Apple® Mac M1/M2.

   * Performance for NLP scenarios is improved, especially for int8 models.
   * The CPU device is enabled with BF16 data types, such that quantized models (INT8) can be run with BF16 plus INT8 mixed 
     precision, taking full advantage of the AMX capability of 4th Generation Intel® Xeon® Scalable Processors
     (formerly Sapphire Rapids). The customer sees BF16/INT8 advantage, by default.
   * Performance is improved on modern, hybrid Intel® Xeon® and Intel® Core® platforms, 
     where threads can be reliably and correctly mapped to the E-cores, P-cores, or both CPU core types. 
     It is now possible to optimize for performance or for power savings as needed.
   * Neural Network Compression Framework (NNCF) becomes the quantization tool of choice. It now enables you to perform
     post-training optimization, as well as quantization-aware training. Try it out: ``pip install nncf``. 
     Post-training Optimization Tool (POT) has been deprecated and will be removed in the future 
     (`MR16758 <https://github.com/openvinotoolkit/openvino/pull/16758/files>`__).
   * New models are enabled, such as:
   
     * Stable Diffusion 2.0, 
     * Paddle Slim, 
     * Segment Anything Model (SAM),
     * Whisper,
     * YOLOv8.  
 
   * Bug fixes:  
 
     * Fixes the problem of OpenVINO-dev wheel not containing the benchmark_app package.
     * Rolls back the default of model saving with the FP16 precision - FP32 is the default again.  
   
   * Known issues:   
  
     * PyTorch model conversion via convert_model Python API fails if “silent=false” is specified explicitly. 
       By default, this parameter is set to true and there should be no issues.


.. dropdown:: OpenVINO Toolkit 2023.0.0.dev20230407
   :animate: fade-in-slide-down
   :color: secondary

   * Enabled remote tensor in C API 2.0 (accepting tensor located in graph memory)
   * Introduced model caching on GPU. Model Caching, which reduces First Inference Latency (FIL), is 
     extended to work as a single method on both CPU and GPU plug-ins.
   * Added the post-training Accuracy-Aware Quantization mechanism for OpenVINO IR. By using this mechanism 
     the user can define the accuracy drop criteria and NNCF will consider it during the quantization.
   * Migrated the CPU plugin to OneDNN 3.1.
   * Enabled CPU fall-back for the AUTO plugin - in case of run-time failure of networks on accelerator devices, CPU is used.
   * Now, AUTO supports the option to disable CPU as the initial acceleration device to speed up first-inference latency.
   * Implemented ov::hint::inference_precision, which enables running network inference independently of the IR precision. 
     The default mode is FP16, it is possible to infer in FP32 to increase accuracy. 
   * Optimized performance on dGPU with Intel oneDNN v3.1, especially for transformer models.
   * Enabled dynamic shapes on iGPU and dGPU for Transformer(NLP) models. Not all dynamic models are enabled but model coverage will be expanded in following releases.
   * Improved performance for Transformer models for NLP pipelines on CPU. 
   * Extended support to the following models:

     * Enabled MLPerf RNN-T model.
     * Enabled Detectron2 MaskRCNN.
     * Enabled OpenSeeFace models.
     * Enabled Clip model.
     * Optimized WeNet model.


   Known issues:

   * OpenVINO-dev wheel does not contain the benchmark_app package



.. dropdown:: OpenVINO Toolkit 2023.0.0.dev20230217
   :animate: fade-in-slide-down
   :color: secondary

   OpenVINO™ repository tag: `2023.0.0.dev20230217 <https://github.com/openvinotoolkit/openvino/releases/tag/2023.0.0.dev20230217>`__

   * Enabled PaddlePaddle Framework 2.4
   * Preview of TensorFlow Lite Frontend – Load models directly via “read_model” into OpenVINO Runtime and export OpenVINO IR format using model conversion API or “convert_model”
   * PyTorch Frontend is available as an experimental feature which will allow you to convert PyTorch models, using convert_model Python API directly from your code without the need to export to the ONNX format. Model coverage is continuously increasing. Feel free to start using the option and give us feedback.
   * Model conversion API now uses the TensorFlow Frontend as the default path for conversion to IR. Known limitations compared to the legacy approach are: TF1 Loop, Complex types, models requiring config files and old python extensions. The solution detects unsupported functionalities and provides fallback. To force using the legacy frontend ``use_legacy_fronted`` can be specified.
   * Model conversion API now supports out-of-the-box conversion of TF2 Object Detection models. At this point, same performance experience is guaranteed only on CPU devices. Feel free to start enjoying TF2 Object Detection models without config files!
   * Introduced new option ov::auto::enable_startup_fallback / ENABLE_STARTUP_FALLBACK to control whether to use CPU to accelerate first inference latency for accelerator HW devices like GPU.
   * New FrontEndManager register_front_end(name, lib_path) interface added, to remove “OV_FRONTEND_PATH” env var (a way to load non-default frontends).


@endsphinxdirective
