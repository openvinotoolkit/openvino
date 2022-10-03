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

@endsphinxdirective

The plugin architecture of the Inference Engine allows to develop and plug independent inference 
solutions dedicated to different devices. Physically, a plugin is represented as a dynamic library 
exporting the single `CreatePluginEngine` function that allows to create a new plugin instance.

Inference Engine Plugin Library
-----------------------

Inference Engine plugin dynamic library consists of several main components:

1. [Plugin class](@ref openvino_docs_ie_plugin_dg_plugin):
	- Provides information about devices of a specific type.
	- Can create an [executable network](@ref openvino_docs_ie_plugin_dg_executable_network) instance which represents a Neural 
	Network backend specific graph structure for a particular device in opposite to the InferenceEngine::ICNNNetwork 
	interface which is backend-independent.
	- Can import an already compiled graph structure from an input stream to an 
	[executable network](@ref openvino_docs_ie_plugin_dg_executable_network) object.
2. [Executable Network class](@ref openvino_docs_ie_plugin_dg_executable_network):
	- Is an execution configuration compiled for a particular device and takes into account its capabilities.
	- Holds a reference to a particular device and a task executor for this device.
	- Can create several instances of [Inference Request](@ref openvino_docs_ie_plugin_dg_infer_request).
	- Can export an internal backend specific graph structure to an output stream.
3. [Inference Request class](@ref openvino_docs_ie_plugin_dg_infer_request):
    - Runs an inference pipeline serially.
    - Can extract performance counters for an inference pipeline execution profiling.
4. [Asynchronous Inference Request class](@ref openvino_docs_ie_plugin_dg_async_infer_request):
    - Wraps the [Inference Request](@ref openvino_docs_ie_plugin_dg_infer_request) class and runs pipeline stages in parallel 
	on several task executors based on a device-specific pipeline structure.

> **NOTE**: This documentation is written based on the `Template` plugin, which demonstrates plugin 
development details. Find the complete code of the `Template`, which is fully compilable and up-to-date,
at `<openvino source dir>/src/plugins/template`.

Detailed guides
-----------------------

* [Build](@ref openvino_docs_ie_plugin_dg_plugin_build) a plugin library using CMake\*
* Plugin and its components [testing](@ref openvino_docs_ie_plugin_dg_plugin_testing)
* [Quantized networks](@ref openvino_docs_ie_plugin_dg_quantized_networks)
* [Low precision transformations](@ref openvino_docs_OV_UG_lpt) guide
* [Writing OpenVINO™ transformations](@ref openvino_docs_transformations) guide

API References
-----------------------

* [Inference Engine Plugin API](@ref ie_dev_api)
* [Inference Engine Transformation API](@ref ie_transformation_api)
