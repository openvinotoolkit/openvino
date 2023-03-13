# Plugin {#openvino_docs_ie_plugin_dg_plugin}

@sphinxdirective

Inference Engine Plugin usually represents a wrapper around a backend. Backends can be:
* OpenCL-like backend (e.g. clDNN library) for GPU devices.
* oneDNN backend for Intel CPU devices.
* NVIDIA cuDNN for NVIDIA GPUs.

The responsibility of Inference Engine Plugin:
* Initializes a backend and throw exception in ``Engine`` constructor if backend cannot be initialized.
* Provides information about devices enabled by a particular backend, e.g. how many devices, their properties and so on.
* Loads or imports :doc:`executable network <openvino_docs_ie_plugin_dg_executable_network>` objects.

In addition to the Inference Engine Public API, the Inference Engine provides the Plugin API, which is a set of functions and helper classes that simplify new plugin development:

* header files in the ``inference_engine/src/plugin_api`` directory
* implementations in the ``inference_engine/src/inference_engine`` directory
* symbols in the Inference Engine Core shared library

To build an Inference Engine plugin with the Plugin API, see the :doc:`Inference Engine Plugin Building <openvino_docs_ie_plugin_dg_plugin_build>` guide.  

Plugin Class
############

Inference Engine Plugin API provides the helper :ref:`InferenceEngine::IInferencePlugin <doxid-class_inference_engine_1_1_i_inference_plugin>` class recommended to use as a base class for a plugin.
Based on that, declaration of a plugin class can look as follows:

.. doxygensnippet:: template/src/plugin.hpp
   :language: hpp
   :fragment: [plugin:header]

Class Fields
++++++++++++

The provided plugin class also has several fields:

* ``_backend`` - a backend engine that is used to perform actual computations for network inference. For ``Template`` plugin ``ngraph::runtime::Backend`` is used which performs computations using OpenVINO™ reference implementations.
* ``_waitExecutor`` - a task executor that waits for a response from a device about device tasks completion.
* ``_cfg`` of type ``Configuration``:

.. doxygensnippet:: template/src/plugin.hpp
   :language: hpp
   :fragment: [configuration:header]

As an example, a plugin configuration has three value parameters:

- ``deviceId`` - particular device ID to work with. Applicable if a plugin supports more than one ``Template`` device. In this case, some plugin methods, like ``SetConfig``, ``QueryNetwork``, and `LoadNetwork`, must support the :ref:`CONFIG_KEY(KEY_DEVICE_ID) <doxid-ie__plugin__config_8hpp_1aad09cfba062e8ec9fb7ab9383f656ec7>` parameter. 
- ``perfCounts`` - boolean value to identify whether to collect performance counters during :doc:`Inference Request <openvino_docs_ie_plugin_dg_infer_request>`execution.
- ``_streamsExecutorConfig`` - configuration of ``:ref:`InferenceEngine::IStreamsExecutor <doxid-class_inference_engine_1_1_i_streams_executor>``` to handle settings of multi-threaded context.

Engine Constructor
------------------

A plugin constructor must contain code that checks the ability to work with a device of the ``Template`` 
type. For example, if some drivers are required, the code must check 
driver availability. If a driver is not available (for example, OpenCL runtime is not installed in 
case of a GPU device or there is an improper version of a driver is on a host machine), an exception 
must be thrown from a plugin constructor.

A plugin must define a device name enabled via the ``_pluginName`` field of a base class:

.. doxygensnippet:: template/src/plugin.hpp
   :language: hpp
   :fragment: [plugin:ctor]

`LoadExeNetworkImpl()`
----------------------

**Implementation details:** The base :ref:`InferenceEngine::IInferencePlugin <doxid-class_inference_engine_1_1_i_inference_plugin>` class provides a common implementation 
of the public :ref:`InferenceEngine::IInferencePlugin::LoadNetwork <doxid-class_inference_engine_1_1_i_inference_plugin_1addf67bb7bae8f00cad65545d5a5a0d51>` method that calls plugin-specific ``LoadExeNetworkImpl``, which is defined in a derived class.

This is the most important function of the `Plugin` class and creates an instance of compiled ``ExecutableNetwork``,
which holds a backend-dependent compiled graph in an internal representation:

.. doxygensnippet:: template/src/plugin.hpp
   :language: hpp
   :fragment: [plugin:load_exe_network_impl]

Before a creation of an ``ExecutableNetwork`` instance via a constructor, a plugin may check if a provided 
:ref:`InferenceEngine::ICNNNetwork <doxid-class_inference_engine_1_1_i_c_n_n_network>` object is supported by a device. In the example above, the plugin checks precision information.

The very important part before creation of ``ExecutableNetwork`` instance is to call ``TransformNetwork`` method which applies OpenVINO™ transformation passes.

Actual graph compilation is done in the ``ExecutableNetwork`` constructor. Refer to the :doc:`ExecutableNetwork Implementation Guide <openvino_docs_ie_plugin_dg_executable_network>` for details.

.. note::
 Actual configuration map used in ``ExecutableNetwork`` is constructed as a base plugin configuration set via ``Plugin::SetConfig``, where some values are overwritten with `config` passed to ``Plugin::LoadExeNetworkImpl``. Therefore, the config of  ``Plugin::LoadExeNetworkImpl`` has a higher priority.

`TransformNetwork()`
--------------------

The function accepts a const shared pointer to ``:ref:`ov::Model <doxid-classov_1_1_model>``` object and performs the following steps:

1. Deep copies a const object to a local object, which can later be modified.
2. Applies common and plugin-specific transformations on a copied graph to make the graph more friendly to hardware operations. For details how to write custom plugin-specific transformation, please, refer to [Writing OpenVINO™ transformations](@ref openvino_docs_transformations) guide. See detailed topics about network representation:
    * `Intermediate Representation and Operation Sets <openvino_docs_MO_DG_IR_and_opsets>`
    * [Quantized networks](@ref openvino_docs_ie_plugin_dg_quantized_networks).

.. doxygensnippet:: template/src/plugin.hpp
   :language: hpp
   :fragment: [plugin:transform_network]

.. note:: 
   After all these transformations, a ``:ref:`ov::Model <doxid-classov_1_1_model>``` object contains operations which can be perfectly mapped to backend kernels. E.g. if backend has kernel computing ``A + B`` operations at once, the ``TransformNetwork`` function should contain a pass which fuses operations ``A`` and ``B`` into a single custom operation ``A + B`` which fits backend kernels set.

`QueryNetwork()`
----------------

Use the method with the ``HETERO`` mode, which allows to distribute network execution between different 
devices based on the ``:ref:`ov::Node::get_rt_info() <doxid-classov_1_1_node_1a6941c753af92828d842297b74df1c45a>``` map, which can contain the ``"affinity"`` key.
The ``QueryNetwork`` method analyzes operations of provided ``network`` and returns a list of supported
operations via the InferenceEngine::QueryNetworkResult structure. The ``QueryNetwork`` firstly applies ``TransformNetwork`` passes to input ``:ref:`ov::Model <doxid-classov_1_1_model>``` argument. After this, the transformed network in ideal case contains only operations are 1:1 mapped to kernels in computational backend. In this case, it's very easy to analyze which operations is supposed (``_backend`` has a kernel for such operation or extensions for the operation is provided) and not supported (kernel is missed in ``_backend``):

1. Store original names of all operations in input ``:ref:`ov::Model <doxid-classov_1_1_model>```.
2. Apply ``TransformNetwork`` passes. Note, the names of operations in a transformed network can be different and we need to restore the mapping in the steps below.
3. Construct ``supported`` and `unsupported` maps which contains names of original operations. Note, that since the inference is performed using OpenVINO™ reference backend, the decision whether the operation is supported or not depends on whether the latest OpenVINO opset contains such operation.
4. ``QueryNetworkResult.supportedLayersMap`` contains only operations which are fully supported by ``_backend``.

.. doxygensnippet:: template/src/plugin.hpp
   :language: hpp
   :fragment: [plugin:query_network]

`SetConfig()`
-------------

Sets new values for plugin configuration keys:

.. doxygensnippet:: template/src/plugin.hpp
   :language: hpp
   :fragment: [plugin:set_config]

In the snippet above, the ``Configuration`` class overrides previous configuration values with the new 
ones. All these values are used during backend specific graph compilation and execution of inference requests.

.. note::
   The function must throw an exception if it receives an unsupported configuration key.

`GetConfig()`
-------------

Returns a current value for a specified configuration key:

.. doxygensnippet:: template/src/plugin.hpp
   :language: hpp
   :fragment: [plugin:get_config]

The function is implemented with the ``Configuration::Get`` method, which wraps an actual configuration 
key value to the InferenceEngine::Parameter and returns it.

.. note::
   The function must throw an exception if it receives an unsupported configuration key.

`GetMetric()`
-------------

Returns a metric value for a metric with the name ``name``. A device metric is a static type of information 
from a plugin about its devices or device capabilities. 

Examples of metrics:

* :ref:`METRIC_KEY(AVAILABLE_DEVICES) <doxid-ie__plugin__config_8hpp_1a69d0efa20c5b2bec020a706279f0c7be>` - list of available devices that are required to implement. In this case, you can use 
all devices of the same ``Template`` type with automatic logic of the `MULTI` device plugin.
* :ref:`METRIC_KEY(FULL_DEVICE_NAME) <doxid-ie__plugin__config_8hpp_1a69d0efa20c5b2bec020a706279f0c7be>` - full device name. In this case, a particular device ID is specified 
in the ``option`` parameter as ``{ :ref:`CONFIG_KEY(KEY_DEVICE_ID) <doxid-ie__plugin__config_8hpp_1aad09cfba062e8ec9fb7ab9383f656ec7>`, "deviceID" }``.
* :ref:`METRIC_KEY(SUPPORTED_METRICS) <doxid-ie__plugin__config_8hpp_1a69d0efa20c5b2bec020a706279f0c7be>` - list of metrics supported by a plugin
* :ref:`METRIC_KEY(SUPPORTED_CONFIG_KEYS) <doxid-ie__plugin__config_8hpp_1a69d0efa20c5b2bec020a706279f0c7be>` - list of configuration keys supported by a plugin that
affects their behavior during a backend specific graph compilation or an inference requests execution
* :ref:`METRIC_KEY(OPTIMIZATION_CAPABILITIES) <doxid-ie__plugin__config_8hpp_1a69d0efa20c5b2bec020a706279f0c7be>` - list of optimization capabilities of a device.
For example, supported data types and special optimizations for them.
* Any other device-specific metrics. In this case, place metrics declaration and possible values to 
a plugin-specific public header file, for example, ``template/config.hpp``. The example below 
demonstrates the definition of a new optimization capability value specific for a device:

.. doxygensnippet:: template/config.hpp
   :language: hpp
   :fragment: [public_header:properties]

The snippet below provides an example of the implementation for `GetMetric`:

.. note:: 
   If an unsupported metric key is passed to the function, it must throw an exception.

`ImportNetwork()`
-----------------

The importing network mechanism allows to import a previously exported backend specific graph and wrap it 
using an :ref:`ExecutableNetwork <openvino_docs_ie_plugin_dg_executable_network>` object. This functionality is useful if 
backend specific graph compilation takes significant time and/or cannot be done on a target host 
device due to other reasons.

During export of backend specific graph using ``ExecutableNetwork::Export``, a plugin may export any 
type of information it needs to import a compiled graph properly and check its correctness. 
For example, the export information may include:

* Compilation options (state of ``Plugin::_cfg`` structure)
* Information about a plugin and a device type to check this information later during the import and 
throw an exception if the ``model`` stream contains wrong data. For example, if devices have different 
capabilities and a graph compiled for a particular device cannot be used for another, such type of 
information must be stored and checked during the import. 
* Compiled backend specific graph itself
* Information about precisions and shapes set by the user

.. doxygensnippet:: template/src/plugin.cpp
   :language: hpp
   :fragment: [plugin:import_network]

Create Instance of Plugin Class
+++++++++++++++++++++++++++++++

Inference Engine plugin library must export only one function creating a plugin instance using IE_DEFINE_PLUGIN_CREATE_FUNCTION macro:

.. doxygensnippet:: template/src/plugin.cpp
   :language: hpp
   :fragment: [plugin:create_plugin_engine]

Next step in a plugin library implementation is the :ref:`ExecutableNetwork <openvino_docs_ie_plugin_dg_executable_network>` class.

@endsphinxdirective


