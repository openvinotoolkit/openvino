Plugin
======


.. meta::
   :description: Explore OpenVINO Plugin API, which includes functions and
                 helper classes that simplify the development of new plugins.


OpenVINO Plugin usually represents a wrapper around a backend. Backends can be:

* OpenCL-like backend (e.g. clDNN library) for GPU devices.
* oneDNN backend for Intel CPU devices.
* NVIDIA cuDNN for NVIDIA GPUs.

The responsibility of OpenVINO Plugin:

* Initializes a backend and throw exception in ``Engine`` constructor if backend cannot be initialized.
* Provides information about devices enabled by a particular backend, e.g. how many devices, their properties and so on.
* Loads or imports :doc:`compiled model <compiled-model>` objects.

In addition to the OpenVINO Public API, the OpenVINO provides the Plugin API, which is a set of functions and helper classes that simplify new plugin development:

* header files in the ``src/inference/dev_api/openvino`` directory
* implementations in the ``src/inference/src/dev/`` directory
* symbols in the OpenVINO shared library

To build an OpenVINO plugin with the Plugin API, see the :doc:`OpenVINO Plugin Building <build-plugin-using-cmake>` guide.

Plugin Class
############

OpenVINO Plugin API provides the helper ov::IPlugin class recommended to use as a base class for a plugin.
Based on that, declaration of a plugin class can look as follows:

.. doxygensnippet:: src/plugins/template/src/plugin.hpp
   :language: cpp
   :fragment: [plugin:header]


Class Fields
++++++++++++

The provided plugin class also has several fields:

* ``m_backend`` - a backend engine that is used to perform actual computations for model inference. For ``Template`` plugin ``ov::runtime::Backend`` is used which performs computations using OpenVINO™ reference implementations.
* ``m_waitExecutor`` - a task executor that waits for a response from a device about device tasks completion.
* ``m_cfg`` of type ``Configuration``:

.. doxygensnippet:: src/plugins/template/src/config.hpp
   :language: cpp
   :fragment: [configuration:header]

As an example, a plugin configuration has three value parameters:

* ``device_id`` - particular device ID to work with. Applicable if a plugin supports more than one ``Template`` device. In this case, some plugin methods, like ``set_property``, ``query_model``, and ``compile_model``, must support the ov::device::id property.
* ``perf_counts`` - boolean value to identify whether to collect performance counters during :doc:`Inference Request <synch-inference-request>` execution.
* ``streams_executor_config`` - configuration of ``ov::threading::IStreamsExecutor`` to handle settings of multi-threaded context.
* ``performance_mode`` - configuration of ``ov::hint::PerformanceMode`` to set the performance mode.
* ``disable_transformations`` - allows to disable transformations which are applied in the process of model compilation.
* ``exclusive_async_requests`` - allows to use exclusive task executor for asynchronous infer requests.

Plugin Constructor
++++++++++++++++++

A plugin constructor must contain code that checks the ability to work with a device of the ``Template``
type. For example, if some drivers are required, the code must check
driver availability. If a driver is not available (for example, OpenCL runtime is not installed in
case of a GPU device or there is an improper version of a driver is on a host machine), an exception
must be thrown from a plugin constructor.

A plugin must define a device name enabled via the ``set_device_name()`` method of a base class:

.. doxygensnippet:: src/plugins/template/src/plugin.cpp
   :language: cpp
   :fragment: [plugin:ctor]

Plugin Destructor
+++++++++++++++++

A plugin destructor must stop all plugins activities, and clean all allocated resources.


.. doxygensnippet:: src/plugins/template/src/plugin.cpp
   :language: cpp
   :fragment: [plugin:dtor]

compile_model()
+++++++++++++++

The plugin should implement two ``compile_model()`` methods: the first one compiles model without remote context, the second one with remote context if plugin supports.

This is the most important function of the ``Plugin`` class is to create an instance of compiled ``CompiledModel``,
which holds a backend-dependent compiled model in an internal representation:

.. doxygensnippet:: src/plugins/template/src/plugin.cpp
   :language: cpp
   :fragment: [plugin:compile_model]

.. doxygensnippet:: src/plugins/template/src/plugin.cpp
   :language: cpp
   :fragment: [plugin:compile_model_with_remote]

Before a creation of an ``CompiledModel`` instance via a constructor, a plugin may check if a provided
ov::Model object is supported by a device if it is needed.

Actual model compilation is done in the ``CompiledModel`` constructor. Refer to the :doc:`CompiledModel Implementation Guide <compiled-model>` for details.

.. note::

   Actual configuration map used in ``CompiledModel`` is constructed as a base plugin configuration set via ``Plugin::set_property``, where some values are overwritten with ``config`` passed to ``Plugin::compile_model``. Therefore, the config of  ``Plugin::compile_model`` has a higher priority.

transform_model()
+++++++++++++++++

The function accepts a const shared pointer to `ov::Model` object and applies common and device-specific transformations on a copied model to make it more friendly to hardware operations. For details how to write custom device-specific transformation, refer to :doc:`Writing OpenVINO™ transformations <../transformation-api>` guide. See detailed topics about model representation:

* :doc:`Intermediate Representation and Operation Sets <../../openvino-ir-format/operation-sets>`
* :doc:`Quantized models <advanced-guides/quantized-models>`.


.. doxygensnippet:: src/plugins/template/src/plugin.cpp
   :language: cpp
   :fragment: [plugin:transform_model]

.. note::

   After all these transformations, an ``ov::Model`` object contains operations which can be perfectly mapped to backend kernels. E.g. if backend has kernel computing ``A + B`` operations at once, the ``transform_model`` function should contain a pass which fuses operations ``A`` and ``B`` into a single custom operation `A + B` which fits backend kernels set.

query_model()
+++++++++++++

Use the method with the ``HETERO`` mode, which allows to distribute model execution between different
devices based on the ``ov::Node::get_rt_info()`` map, which can contain the ``affinity`` key.
The ``query_model`` method analyzes operations of provided ``model`` and returns a list of supported
operations via the ov::SupportedOpsMap structure. The ``query_model`` firstly applies ``transform_model`` passes to input ``ov::Model`` argument. After this, the transformed model in ideal case contains only operations are 1:1 mapped to kernels in computational backend. In this case, it's very easy to analyze which operations is supposed (``m_backend`` has a kernel for such operation or extensions for the operation is provided) and not supported (kernel is missed in ``m_backend``):

1. Store original names of all operations in input ``ov::Model``.
2. Apply ``transform_model`` passes. Note, the names of operations in a transformed model can be different and we need to restore the mapping in the steps below.
3. Construct ``supported`` map which contains names of original operations. Note that since the inference is performed using OpenVINO™ reference backend, the decision whether the operation is supported or not depends on whether the latest OpenVINO opset contains such operation.
4. ``ov.SupportedOpsMap`` contains only operations which are fully supported by ``m_backend``.

.. doxygensnippet:: src/plugins/template/src/plugin.cpp
   :language: cpp
   :fragment: [plugin:query_model]

set_property()
++++++++++++++

Sets new values for plugin property keys:

.. doxygensnippet:: src/plugins/template/src/plugin.cpp
   :language: cpp
   :fragment: [plugin:set_property]

In the snippet above, the ``Configuration`` class overrides previous configuration values with the new
ones. All these values are used during backend specific model compilation and execution of inference requests.

.. note::

   The function must throw an exception if it receives an unsupported configuration key.

get_property()
++++++++++++++

Returns a current value for a specified property key:

.. doxygensnippet:: src/plugins/template/src/plugin.cpp
   :language: cpp
   :fragment: [plugin:get_property]

The function is implemented with the ``Configuration::Get`` method, which wraps an actual configuration
key value to the ov::Any and returns it.

.. note::

   The function must throw an exception if it receives an unsupported configuration key.

import_model()
++++++++++++++

The importing of compiled model mechanism allows to import a previously exported backend specific model and wrap it
using an :doc:`CompiledModel <compiled-model>` object. This functionality is useful if
backend specific model compilation takes significant time and/or cannot be done on a target host
device due to other reasons.

During export of backend specific model using ``CompiledModel::export_model``, a plugin may export any
type of information it needs to import a compiled model properly and check its correctness.
For example, the export information may include:

* Compilation options (state of ``Plugin::m_cfg`` structure).
* Information about a plugin and a device type to check this information later during the import and throw an exception if the ``model`` stream contains wrong data. For example, if devices have different capabilities and a model compiled for a particular device cannot be used for another, such type of information must be stored and checked during the import.
* Compiled backend specific model itself.

.. doxygensnippet:: src/plugins/template/src/plugin.cpp
   :language: cpp
   :fragment: [plugin:import_model]

.. doxygensnippet:: src/plugins/template/src/plugin.cpp
   :language: cpp
   :fragment: [plugin:import_model_with_remote]


create_context()
++++++++++++++++

The Plugin should implement ``Plugin::create_context()`` method which returns ``ov::RemoteContext`` in case if plugin supports remote context, in other case the plugin can throw an exception that this method is not implemented.

.. doxygensnippet:: src/plugins/template/src/plugin.cpp
   :language: cpp
   :fragment: [plugin:create_context]


get_default_context()
+++++++++++++++++++++

``Plugin::get_default_context()`` also needed in case if plugin supports remote context, if the plugin doesn't support it, this method can throw an exception that functionality is not implemented.

.. doxygensnippet:: src/plugins/template/src/plugin.cpp
   :language: cpp
   :fragment: [plugin:get_default_context]

Create Instance of Plugin Class
###############################

OpenVINO plugin library must export only one function creating a plugin instance using OV_DEFINE_PLUGIN_CREATE_FUNCTION macro:

.. doxygensnippet:: src/plugins/template/src/plugin.cpp
   :language: cpp
   :fragment: [plugin:create_plugin_engine]


Next step in a plugin library implementation is the :doc:`CompiledModel <compiled-model>` class.

