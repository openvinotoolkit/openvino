# Pre-release Information {#prerelease_information}

@sphinxdirective

To ensure you do not have to wait long to test OpenVINO's upcoming features, 
OpenVINO developers continue to roll out prerelease versions. In this page you can find
a general changelog and the schedule for all versions for the current year.

.. note:: 
   These versions are pre-release software and have not undergone full validation or qualification. OpenVINO™ toolkit pre-release is:

   * NOT to be incorporated into production software/solutions.
   * NOT subject to official support.
   * Subject to change in the future.
   * Introduced to allow early testing and get early feedback from the community.
 


.. dropdown:: OpenVINO Toolkit 2023.0.0.dev20230229
   :open:
   :animate: fade-in-slide-down
   :color: primary

   * Added remote tensor support (accepting tensor located in graph memory) in C API 2.0
   * Introduced model caching on GPU. Model Caching which reduces First Inference Latency (FIL) is 
     extended to work as a single method on both CPU and GPU plug-ins.
   * Added the post-training Accuracy-Aware Quantization mechanism for OpenVINO IR. By using this mechanism 
     the user can define the accuracy drop criteria and NNCF will consider it during the quantization.
   * CPU plugin migrated to OneDNN 3.1.
   * AUTO supports fall-back to CPU device in case of run-time failure of networks on accelerator devices. 
   * AUTO supports the option to disable CPU as the initial acceleration device to speed up first-inference latency.
   * Implementing a parameter enabling network inference independently of IR precision. The default mode will be FP16 
     with FP32 as an option to increase accuracy. General hints (performance vs accuracy) to be provided.
   * Adding support for new model use cases or optimizing the existing support (better accuracy or performance).
   * Performance on dGPU optimized with Intel oneDNN v3.1, especially for transformer models. 
   * Dynamic shapes supported on iGPU and dGPU for Transformer(NLP) models. Not all dynamic models are supported but model coverage will be expanded in future releases.
   * Performance improvement for Transformer models for NLP pipelines on CPU. 
   * Models:

     * Enabled MLPerf RNN-T model.
     * Enabled Detectron2 MaskRCNN.
     * Enabled OpenSeeFace models.
     * Enabled Clip model.
     * Optimized WeNet model.



.. dropdown:: OpenVINO Toolkit 2023.0.0.dev20230217
   :animate: fade-in-slide-down
   :color: secondary

   OpenVINO™ repository tag: `2023.0.0.dev20230217 <https://github.com/openvinotoolkit/openvino/releases/tag/2023.0.0.dev20230217>`__

   * Enabled PaddlePaddle Framework 2.4
   * Preview of TensorFlow Lite Front End – Load models directly via “read_model” into OpenVINO Runtime and export OpenVINO IR format using Model Optimizer or “convert_model”
   * PyTorch Frontend is available as an experimental feature which will allow you to convert PyTorch models, using convert_model Python API directly from your code without the need to export to ONNX. Model coverage is continuously increasing. Feel free to start using the option and give us feedback.
   * Model Optimizer now uses the TensorFlow Frontend as the default path for conversion to IR. Known limitations compared to the legacy approach are: TF1 Loop, Complex types, models requiring config files and old python extensions. The solution detects unsupported functionalities and provides fallback. To force using the legacy frontend ``--use_legacy_fronted`` can be specified.
   * Model Optimizer now supports out-of-the-box conversion of TF2 Object Detection models. At this point, same performance experience is guaranteed only on CPU devices. Feel free to start enjoying TF2 Object Detection models without config files!
   * Introduced new option ov::auto::enable_startup_fallback / ENABLE_STARTUP_FALLBACK to control whether to use CPU to accelerate first inference latency for accelerator HW devices like GPU.
   * New FrontEndManager register_front_end(name, lib_path) interface added, to remove “OV_FRONTEND_PATH” env var (a way to load non-default frontends).


@endsphinxdirective
