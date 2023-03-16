# Overview of Inference Engine Plugin Library {#openvino_docs_ie_plugin_dg_overview}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :caption: Converting and Preparing Models
   :hidden:

   Implement Plugin Functionality <openvino_docs_ie_plugin_dg_plugin>
   Implement Executable Network Functionality <openvino_docs_ie_plugin_dg_executable_network>
   Implement Synchronous Inference Request <openvino_docs_ie_plugin_dg_infer_request>
   Implement Asynchronous Inference Request <openvino_docs_ie_plugin_dg_async_infer_request>
   openvino_docs_ie_plugin_dg_plugin_build
   openvino_docs_ie_plugin_dg_plugin_testing
   openvino_docs_ie_plugin_detailed_guides
   openvino_docs_ie_plugin_api_references


The plugin architecture of the Inference Engine allows to develop and plug independent inference 
solutions dedicated to different devices. Physically, a plugin is represented as a dynamic library 
exporting the single ``CreatePluginEngine`` function that allows to create a new plugin instance.

Inference Engine Plugin Library
###############################

Inference Engine plugin dynamic library consists of several main components:

1. :doc:`Plugin class <openvino_docs_ie_plugin_dg_plugin>`:
	- Provides information about devices of a specific type.
	- Can create an :doc:`executable network <openvino_docs_ie_plugin_dg_executable_network>` instance which represents a Neural 
	Network backend specific graph structure for a particular device in opposite to the :ref:`InferenceEngine::ICNNNetwork <doxid-class_inference_engine_1_1_i_c_n_n_network>` 
	interface which is backend-independent.
	- Can import an already compiled graph structure from an input stream to an 
	:doc:`executable network <openvino_docs_ie_plugin_dg_executable_network>` object.
2. :doc:`Executable Network class <openvino_docs_ie_plugin_dg_executable_network>`:
	- Is an execution configuration compiled for a particular device and takes into account its capabilities.
	- Holds a reference to a particular device and a task executor for this device.
	- Can create several instances of :doc:`Inference Request <openvino_docs_ie_plugin_dg_infer_request>`.
	- Can export an internal backend specific graph structure to an output stream.
3. :doc:`Inference Request class <openvino_docs_ie_plugin_dg_infer_request>`:
    - Runs an inference pipeline serially.
    - Can extract performance counters for an inference pipeline execution profiling.
4. :doc:`Asynchronous Inference Request class <openvino_docs_ie_plugin_dg_async_infer_request>`:
    - Wraps the :doc:`Inference Request <openvino_docs_ie_plugin_dg_infer_request>` class and runs pipeline stages in parallel 
	on several task executors based on a device-specific pipeline structure.

.. note::  
   This documentation is written based on the ``Template`` plugin, which demonstrates plugin development details. Find the complete code of the ``Template``, which is fully compilable and up-to-date, at ``<openvino source dir>/src/plugins/template``.


Detailed Guides
###############

* :doc:`Build <openvino_docs_ie_plugin_dg_plugin_build>` a plugin library using CMake
* Plugin and its components :ref:`testing <openvino_docs_ie_plugin_dg_plugin_testing>`
* :doc:`Quantized networks <openvino_docs_ie_plugin_dg_quantized_networks>`
* :doc:`Low precision transformations <openvino_docs_OV_UG_lpt>` guide
* :doc:`Writing OpenVINO™ transformations <openvino_docs_transformations>` guide

API References
##############

* :doc:`Inference Engine Plugin API <ie_dev_api>`
* :doc:`Inference Engine Transformation API <ie_transformation_api>`

@endsphinxdirective
