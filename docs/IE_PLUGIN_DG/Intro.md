@mainpage Overview of Inference Engine Plugin Library

The plugin architecture of the Inference Engine allows to develop and plug independent inference 
solutions dedicated to different devices. Physically, a plugin is represented as a dynamic library 
exporting the single `CreatePluginEngine` function that allows to create a new plugin instance.

Inference Engine Plugin Library
-----------------------

Inference Engine plugin dynamic library consists of several main components:

1. [Plugin class](@ref plugin):
	- Provides information about devices of a specific type.
	- Can create an [executable network](@ref executable_network) instance which represents a Neural 
	Network backend specific graph structure for a particular device in opposite to the InferenceEngine::ICNNNetwork 
	interface which is backend-independent.
	- Can import an already compiled graph structure from an input stream to an 
	[executable network](@ref executable_network) object.
2. [Executable Network class](@ref executable_network):
	- Is an execution configuration compiled for a particular device and takes into account its capabilities.
	- Holds a reference to a particular device and a task executor for this device.
	- Can create several instances of [Inference Request](@ref infer_request).
	- Can export an internal backend specific graph structure to an output stream.
3. [Inference Request class](@ref infer_request):
    - Runs an inference pipeline serially.
    - Can extract performance counters for an inference pipeline execution profiling.
4. [Asynchronous Inference Request class](@ref async_infer_request):
    - Wraps the [Inference Request](@ref infer_request) class and runs pipeline stages in parallel 
	on several task executors based on a device-specific pipeline structure.

> **NOTE**: This documentation is written based on the `Template` plugin, which demonstrates plugin 
development details. Find the complete code of the `Template`, which is fully compilable and up-to-date,
at `<dldt source dir>/docs/template_plugin`.

Detailed guides
-----------------------

* [Build](@ref plugin_build) a plugin library using CMake\*
* Plugin and its components [testing](@ref plugin_testing)
* [Quantized networks](@ref quantized_networks)
* [Writing ngraph transformations](@ref new_ngraph_transformation) guide

API References
-----------------------

* [Inference Engine Plugin API](group__ie__dev__api.html)
* [Inference Engine Transformation API](group__ie__transformation__api.html)
