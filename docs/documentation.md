# Documentation {#documentation}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :caption: Converting and Preparing Models
   :hidden:

   openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide
   omz_tools_downloader


.. toctree::
   :maxdepth: 1
   :caption: Deploying Inference
   :hidden:

   openvino_docs_OV_UG_OV_Runtime_User_Guide
   openvino_2_0_transition_guide
   openvino_deployment_guide
   openvino_inference_engine_tools_compile_tool_README


.. toctree::
   :maxdepth: 1
   :caption: Tuning for Performance
   :hidden:

   openvino_docs_optimization_guide_dldt_optimization_guide
   openvino_docs_MO_DG_Getting_Performance_Numbers
   openvino_docs_model_optimization_guide
   openvino_docs_deployment_optimization_guide_dldt_optimization_guide
   openvino_docs_tuning_utilities
   openvino_docs_performance_benchmarks


.. toctree::
   :maxdepth: 1
   :caption: Graphical Web Interface for OpenVINO™ toolkit  
   :hidden:

   workbench_docs_Workbench_DG_Introduction
   workbench_docs_Workbench_DG_Install
   workbench_docs_Workbench_DG_Work_with_Models_and_Sample_Datasets
   Tutorials <workbench_docs_Workbench_DG_Tutorials>
   User Guide <workbench_docs_Workbench_DG_User_Guide>
   workbench_docs_Workbench_DG_Troubleshooting

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Media Processing and Computer Vision Libraries

   Intel® Deep Learning Streamer <openvino_docs_dlstreamer>
   openvino_docs_gapi_gapi_intro
   OpenCV* Developer Guide <https://docs.opencv.org/master/>
   OpenCL™ Developer Guide <https://software.intel.com/en-us/openclsdk-devguide>   

.. toctree::
   :maxdepth: 1
   :caption: Add-Ons
   :hidden:

   ovms_what_is_openvino_model_server
   ote_documentation
   ovsa_get_started

.. toctree::
   :maxdepth: 1
   :caption: OpenVINO Extensibility
   :hidden:

   openvino_docs_Extensibility_UG_Intro
   openvino_docs_transformations
   OpenVINO Plugin Developer Guide <openvino_docs_ie_plugin_dg_overview>
   
.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Use OpenVINO™ Toolkit Securely
   
   openvino_docs_security_guide_introduction
   openvino_docs_security_guide_workbench
   openvino_docs_OV_UG_protecting_model_guide
   ovsa_get_started

@endsphinxdirective

This section provides reference documents that guide you through developing your own deep learning applications with the OpenVINO™ toolkit. These documents will most helpful if you have first gone through the [Get Started](get_started.md) guide.

## Converting and Preparing Models
With the [Model Downloader](@ref omz_tools_downloader) and [Model Optimizer](MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) guides, you will learn to download pre-trained models and convert them for use with the OpenVINO™ toolkit. You can provide your own model or choose a public or Intel model from a broad selection provided in the [Open Model Zoo](model_zoo.md).

## Deploying Inference
The [OpenVINO™ Runtime User Guide](./OV_Runtime_UG/openvino_intro.md) explains the process of creating your own application that runs inference with the OpenVINO™ toolkit. The [API Reference](./api_references.html) defines the OpenVINO Runtime API for Python, C++, and C. The OpenVINO Runtime API is what you'll use to create an OpenVINO™ inference application, use enhanced operations sets and other features. After writing your application, you can use the [Deployment with OpenVINO](./OV_Runtime_UG/deployment/deployment_intro.md) for deploying to target devices.

## Tuning for Performance
The toolkit provides a [Performance Optimization Guide](optimization_guide/dldt_optimization_guide.md) and utilities for squeezing the best performance out of your application, including [Accuracy Checker](@ref omz_tools_accuracy_checker), [Post-Training Optimization Tool](@ref pot_introduction), and other tools for measuring accuracy, benchmarking performance, and tuning your application.

## Graphical Web Interface for OpenVINO™ Toolkit
You can choose to use the [OpenVINO™ Deep Learning Workbench](@ref workbench_docs_Workbench_DG_Introduction), a web-based tool that guides you through the process of converting, measuring, optimizing, and deploying models. This tool also serves as a low-effort introduction to the toolkit and provides a variety of useful interactive charts for understanding performance.

## Media Processing and Computer Vision Libraries

The OpenVINO™ toolkit also works with the following media processing frameworks and libraries:

* [Intel® Deep Learning Streamer (Intel® DL Streamer)](@ref openvino_docs_dlstreamer) — A streaming media analytics framework based on GStreamer, for creating complex media analytics pipelines optimized for Intel hardware platforms. Go to the Intel® DL Streamer [documentation](https://dlstreamer.github.io/) website to learn more.
* [Intel® oneAPI Video Processing Library (oneVPL)](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/api-based-programming/intel-oneapi-video-processing-library-onevpl.html) — A programming interface for video decoding, encoding, and processing to build portable media pipelines on CPUs, GPUs, and other accelerators.

You can also add computer vision capabilities to your application using optimized versions of [OpenCV](https://opencv.org/).

