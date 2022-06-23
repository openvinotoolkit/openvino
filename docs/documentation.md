# Documentation {#documentation}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :caption: API 2.0
   :hidden:

   openvino_2_0_transition_guide


.. toctree::
   :maxdepth: 1
   :caption: Converting and Preparing Models
   :hidden:

   openvino_docs_model_processing_introduction
   openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide
   omz_tools_downloader


.. toctree::
   :maxdepth: 1
   :caption: Optimization and Performance
   :hidden:

   openvino_docs_optimization_guide_dldt_optimization_guide
   openvino_docs_MO_DG_Getting_Performance_Numbers
   openvino_docs_model_optimization_guide
   openvino_docs_deployment_optimization_guide_dldt_optimization_guide
   openvino_docs_tuning_utilities
   openvino_docs_performance_benchmarks


.. toctree::
   :maxdepth: 1
   :caption: Deploying Inference
   :hidden:

   openvino_docs_deployment_guide_introduction
   openvino_docs_OV_UG_OV_Runtime_User_Guide
   openvino_deployment_guide
   openvino_inference_engine_tools_compile_tool_README


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: THE Ecosystem

   openvino_ecosystem
   ovms_what_is_openvino_model_server
   ovsa_get_started
   ovtf_integration
   ote_documentation
   workbench_docs_Workbench_DG_Introduction


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

This section provides reference documents that guide you through the OpenVINO toolkit workflow, from obtaining models, optimizing them, to deploying them in your own deep learning applications.

## Converting and Preparing Models
With [Model Downloader](@ref omz_tools_downloader) and [Model Optimizer](MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) guides, you will learn to download pre-trained models and convert them for use with OpenVINO™. You can use your own models or choose some from a broad selection provided in the [Open Model Zoo](./model_zoo.md).

## Optimization and Performance
In this section you will find resources on [how to test inference performance](MO_DG/prepare_model/Getting_performance_numbers.md) and [how to increase it](optimization_guide/dldt_optimization_guide.md). It can be achieved by [optimizing the model](optimization_guide/model_optimization_guide.md) or [optimizing inference at runtime](optimization_guide/dldt_deployment_optimization_guide.md). 

## Deploying Inference
This section explains the process of creating your own inference application using [OpenVINO™ Runtime](./OV_Runtime_UG/openvino_intro.md) and documents the [OpenVINO Runtime API](./api_references.html) for both Python and C++.
It also provides a [guide on deploying applications with OpenVINO](./OV_Runtime_UG/deployment/deployment_intro.md) and directs you to other sources on this topic.

## OpenVINO Ecosystem
Apart from the core components, OpenVINO offers tools, plugins, and expansions revolving around it, even if not constituting necessary parts of its workflow. This section will give you an overview of [what makes up OpenVINO Toolkit](./Documentation/openvino_ecosystem.md).

## Media Processing and Computer Vision Libraries

The OpenVINO™ toolkit also works with the following media processing frameworks and libraries:

* [Intel® Deep Learning Streamer (Intel® DL Streamer)](@ref openvino_docs_dlstreamer) — A streaming media analytics framework based on GStreamer, for creating complex media analytics pipelines optimized for Intel hardware platforms. Go to the Intel® DL Streamer [documentation](https://dlstreamer.github.io/) website to learn more.
* [Intel® oneAPI Video Processing Library (oneVPL)](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/api-based-programming/intel-oneapi-video-processing-library-onevpl.html) — A programming interface for video decoding, encoding, and processing to build portable media pipelines on CPUs, GPUs, and other accelerators.

You can also add computer vision capabilities to your application using optimized versions of [OpenCV](https://opencv.org/).

