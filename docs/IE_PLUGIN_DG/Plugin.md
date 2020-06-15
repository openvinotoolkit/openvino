# Plugin {#plugin}

In addition to the Inference Engine Public API, the Inference Engine provides the Plugin API, which is a set of functions and helper classes that simplify new plugin development:

- header files in the `inference_engine/src/plugin_api` directory
- implementations in the `inference_engine/src/inference_engine` directory
- symbols in the Inference Engine Core shared library

To build an Inference Engine plugin with the Plugin API, see the [Inference Engine Plugin Building](@ref plugin_build) guide.  

Plugin Class
------------------------

Inference Engine Plugin API provides the helper InferenceEngine::InferencePluginInternal class recommended to use as a base class for a plugin.
Based on that, declaration of a plugin class can look as follows:

@snippet src/template_plugin.hpp plugin:header

#### Class Fields

The provided plugin class also has a single field:

* `_cfg` of type `Configuration`:

@snippet src/template_config.hpp configuration:header

As an example, a plugin configuration has three value parameters:

- `deviceId` - particular device ID to work with. Applicable if a plugin supports more than one `Template` device. In this case, some plugin methods, like `SetConfig`, `QueryNetwork`, and `LoadNetwork`, must support the CONFIG_KEY(KEY_DEVICE_ID) parameter. 
- `perfCounts` - boolean value to identify whether to collect performance counters during [Inference Request](@ref infer_request) execution.

### Engine Constructor

A plugin constructor must contain code that checks the ability to work with a device of the `Template` 
type. For example, if some drivers are required, the code must check 
driver availability. If a driver is not available (for example, OpenCL runtime is not installed in 
case of a GPU device or there is an improper version of a driver is on a host machine), an exception 
must be thrown from a plugin constructor.

A plugin must define a device name enabled via the `_pluginName` field of a base class:

@snippet src/template_plugin.cpp plugin:ctor

### `LoadExeNetworkImpl()`

**Implementation details:** The base InferenceEngine::InferencePluginInternal class provides a common implementation 
of the public InferenceEngine::InferencePluginInternal::LoadNetwork method that calls plugin-specific `LoadExeNetworkImpl`, which is defined in a derived class.

This is the most important function of the `Plugin` class and creates an instance of compiled `ExecutableNetwork`,
which holds a hardware-dependent compiled graph in an internal representation:

@snippet src/template_plugin.cpp plugin:load_exe_network_impl

Before a creation of an `ExecutableNetwork` instance via a constructor, a plugin may check if a provided 
InferenceEngine::ICNNNetwork object is supported by a device. In the example above, the plugin checks precision information.

Actual graph compilation is done in the `ExecutableNetwork` constructor. Refer to the [ExecutableNetwork Implementation Guide](@ref executable_network) for details.

> **NOTE**: Actual configuration map used in `ExecutableNetwork` is constructed as a base plugin 
> configuration set via `Plugin::SetConfig`, where some values are overwritten with `config` passed to `Plugin::LoadExeNetworkImpl`. 
> Therefore, the config of  `Plugin::LoadExeNetworkImpl` has a higher priority.

### `QueryNetwork()`

Use the method with the `HETERO` mode, which allows to distribute network execution between different 
devices based on the `ngraph::Node::get_rt_info()` map, which can contain the `"affinity"` key.
The `QueryNetwork` method analyzes operations of provided `network` and returns a list of supported
operations via the InferenceEngine::QueryNetworkResult structure:

@snippet src/template_plugin.cpp plugin:query_network

### `AddExtension()`

Adds an extension of the InferenceEngine::IExtensionPtr type to a plugin. If a plugin does not 
support extensions, the method must throw an exception:

@snippet src/template_plugin.cpp plugin:add_extension

### `SetConfig()`

Sets new values for plugin configuration keys:

@snippet src/template_plugin.cpp plugin:set_config

In the snippet above, the `Configuration` class overrides previous configuration values with the new 
ones. All these values are used during hardware-specific graph compilation and execution of inference requests.

> **NOTE**: The function must throw an exception if it receives an unsupported configuration key.

### `GetConfig()`

Returns a current value for a specified configuration key:

@snippet src/template_plugin.cpp plugin:get_config

The function is implemented with the `Configuration::Get` method, which wraps an actual configuration 
key value to the InferenceEngine::Parameter and returns it.

> **NOTE**: The function must throw an exception if it receives an unsupported configuration key.

### `GetMetric()`

Returns a metric value for a metric with the name `name`. A device metric is a static type of information 
from a plugin about its devices or device capabilities. 

Examples of metrics:

- METRIC_KEY(AVAILABLE_DEVICES) - list of available devices that are required to implement. In this case, you can use 
all devices of the same `Template` type with automatic logic of the `MULTI` device plugin.
- METRIC_KEY(FULL_DEVICE_NAME) - full device name. In this case, a particular device ID is specified 
in the `option` parameter as `{ CONFIG_KEY(KEY_DEVICE_ID), "deviceID" }`.
- METRIC_KEY(SUPPORTED_METRICS) - list of metrics supported by a plugin
- METRIC_KEY(SUPPORTED_CONFIG_KEYS) - list of configuration keys supported by a plugin that
affects their behavior during a hardware-specific graph compilation or an inference requests execution
- METRIC_KEY(OPTIMIZATION_CAPABILITIES) - list of optimization capabilities of a device.
For example, supported data types and special optimizations for them.
- Any other device-specific metrics. In this case, place metrics declaration and possible values to 
a plugin-specific public header file, for example, `template/template_config.hpp`. The example below 
demonstrates the definition of a new optimization capability value specific for a device:

@snippet template/template_config.hpp public_header:metrics 

The snippet below provides an example of the implementation for `GetMetric`:

@snippet src/template_plugin.cpp plugin:get_metric

> **NOTE**: If an unsupported metric key is passed to the function, it must throw an exception.

### `ImportNetworkImpl()`

The importing network mechanism allows to import a previously exported hardware-specific graph and wrap it 
using an [ExecutableNetwork](@ref executable_network) object. This functionality is useful if 
hardware-specific graph compilation takes significant time and/or cannot be done on a target host 
device due to other reasons.

**Implementation details:** The base plugin class InferenceEngine::InferencePluginInternal implements InferenceEngine::InferencePluginInternal::ImportNetwork 
as follows: exports a device type (InferenceEngine::InferencePluginInternal::_pluginName) and then calls `ImportNetworkImpl`, 
which is implemented in a derived class. 
If a plugin cannot use the base implementation InferenceEngine::InferencePluginInternal::ImportNetwork, it can override base 
implementation and define an output blob structure up to its needs. This 
can be useful if a plugin exports a blob in a special format for integration with other frameworks 
where a common Inference Engine header from a base class implementation is not appropriate. 

During export of hardware-specific graph using `ExecutableNetwork::Export`, a plugin may export any 
type of information it needs to import a compiled graph properly and check its correctness. 
For example, the export information may include:

- Compilation options (state of `Plugin::_cfg` structure)
- Information about a plugin and a device type to check this information later during the import and 
throw an exception if the `model` stream contains wrong data. For example, if devices have different 
capabilities and a graph compiled for a particular device cannot be used for another, such type of 
information must be stored and checked during the import. 
- Compiled hardware-specific graph itself
- Information about precisions and shapes set by the user

@snippet src/template_plugin.cpp plugin:import_network_impl

Create Instance of Plugin Class
------------------------

Inference Engine plugin library must export only one function creating a plugin instance:

@snippet src/template_plugin.cpp plugin:create_plugin_engine

Next step in a plugin library implementation is the [ExecutableNetwork](@ref executable_network) class.
