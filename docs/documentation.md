# Documentation {#documentation}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :caption: Converting and Preparing Models
   :hidden:

   openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide
   openvino_docs_HOWTO_Custom_Layers_Guide
   openvino_docs_IE_DG_Tools_Model_Downloader


.. toctree::
   :maxdepth: 1
   :caption: Intermediate Representation and Operations Sets
   :hidden:
    
   openvino_docs_ops_opset
   openvino_docs_ops_broadcast_rules
   openvino_docs_operations_specifications


 
.. toctree::
   :maxdepth: 1
   :caption: Deploying Inference
   :hidden:

   openvino_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide
   openvino_docs_nGraph_DG_DevGuide
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
   tuning_utilities


.. toctree::
   :maxdepth: 1
   :caption: OpenVINO™ Deep Learning Workbench 
   :hidden:

   workbench_docs_Workbench_DG_Introduction

@endsphinxdirective

This section provides documentation that guides you through developing your deep learning applications with the OpenVINO™ toolkit.

With the [Model Optimizer](MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) and [Model Downloader](@ref omz_tools_downloader) guides, you will learn how to download or prepare deep learning models to use with OpenVINO™ toolkit.

The [Inference Engine Developer Guide](IE_DG/Deep_Learning_Inference_Engine_DevGuide.md) and API References provides information on how to integrate and utilize the inference API to run inference on your models.

Read the [Accuracy Checker](@ref omz_tools_accuracy_checker), [Performance Optimization](optimization_guide/dldt_optimization_guide.md) and [Post-Training Optimization Tool](@ref pot_README) guides to know how to check the accuracy of you model, tune it in development and deploy a more efficient application.
