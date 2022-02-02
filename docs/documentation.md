# Documentation {#documentation}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :caption: Converting and Preparing Models
   :hidden:

   openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide
   openvino_docs_HOWTO_Custom_Layers_Guide
   omz_tools_downloader


.. toctree::
   :maxdepth: 1
   :caption: Deploying Inference
   :hidden:

   openvino_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide
   openvino_docs_install_guides_deployment_manager_tool
   openvino_inference_engine_tools_compile_tool_README


.. toctree::
   :maxdepth: 1
   :caption: Tuning for Performance
   :hidden:

   openvino_docs_performance_benchmarks
   openvino_docs_optimization_guide_dldt_optimization_guide
   openvino_docs_MO_DG_Getting_Performance_Numbers
   pot_README
   openvino_docs_tuning_utilities


.. toctree::
   :maxdepth: 1
   :caption: Graphical Web Interface for OpenVINO™ toolkit  
   :hidden:

   workbench_docs_Workbench_DG_Introduction
   workbench_docs_Workbench_DG_Install
   workbench_docs_Workbench_DG_Work_with_Models_and_Sample_Datasets
   workbench_docs_Workbench_DG_User_Guide
   workbench_docs_security_Workbench
   workbench_docs_Workbench_DG_Troubleshooting

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Media Processing

   DL Streamer API Reference <https://openvinotoolkit.github.io/dlstreamer_gst/>
   gst_samples_README
   openvino_docs_gapi_gapi_intro
   OpenVX Developer Guide <https://software.intel.com/en-us/openvino-ovx-guide>
   OpenVX API Reference <https://khronos.org/openvx>
   OpenCV* Developer Guide <https://docs.opencv.org/master/>
   OpenCL™ Developer Guide <https://software.intel.com/en-us/openclsdk-devguide>   

.. toctree::
   :maxdepth: 1
   :caption: Add-Ons
   :hidden:

   openvino_docs_ovms
   ovsa_get_started

.. toctree::
   :maxdepth: 1
   :caption: Developing Inference Engine Plugins 
   :hidden:

   Inference Engine Plugin Developer Guide <openvino_docs_ie_plugin_dg_overview>
   groupie_dev_api
   Plugin Transformation Pipeline <openvino_docs_IE_DG_plugin_transformation_pipeline>
   
.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Use OpenVINO™ Toolkit Securely
   
   openvino_docs_security_guide_introduction
   openvino_docs_security_guide_workbench
   openvino_docs_IE_DG_protecting_model_guide
   ovsa_get_started

@endsphinxdirective

This section provides reference documents that guide you through developing your own deep learning applications with the OpenVINO™ toolkit. These documents will most helpful if you have first gone through the [Get Started](get_started.md) guide.

## Converting and Preparing Models
With the [Model Downloader](@ref omz_tools_downloader) and [Model Optimizer](MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) guides, you will learn to download pre-trained models and convert them for use with the OpenVINO™ toolkit. You can provide your own model or choose a public or Intel model from a broad selection provided in the [Open Model Zoo](model_zoo.md).

## Deploying Inference
The [OpenVINO™ Runtime User Guide](OV_Runtime_UG/Deep_Learning_Inference_Engine_DevGuide.md) explains the process of creating your own application that runs inference with the OpenVINO™ toolkit. The [API Reference](./api_references.html) defines the Inference Engine API for Python, C++, and C and the nGraph API for Python and C++. The Inference Engine API is what you'll use to create an OpenVINO™ application, while the nGraph API is available for using enhanced operations sets and other features. After writing your application, you can use the [Deployment Manager](install_guides/deployment-manager-tool.md) for deploying to target devices.

## Tuning for Performance
The toolkit provides a [Performance Optimization Guide](optimization_guide/dldt_optimization_guide.md) and utilities for squeezing the best performance out of your application, including [Accuracy Checker](@ref omz_tools_accuracy_checker), [Post-Training Optimization Tool](@ref pot_README), and other tools for measuring accuracy, benchmarking performance, and tuning your application.

## Graphical Web Interface for OpenVINO™ Toolkit
You can choose to use the [OpenVINO™ Deep Learning Workbench](@ref workbench_docs_Workbench_DG_Introduction), a web-based tool that guides you through the process of converting, measuring, optimizing, and deploying models. This tool also serves as a low-effort introduction to the toolkit and provides a variety of useful interactive charts for understanding performance.

## Media Processing
The OpenVINO™ toolkit comes with several sets of libraries and tools that add capability and flexibility to the toolkit. These include [DL Streamer](@ref gst_samples_README), a utility that eases creation of pipelines via command line or API, and optimized versions of OpenCV and OpenCL.
