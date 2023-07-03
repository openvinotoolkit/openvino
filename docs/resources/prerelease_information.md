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



.. dropdown:: OpenVINO Toolkit 2023.1.0.dev20230623
   :open:
   :animate: fade-in-slide-down
   :color: primary

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
