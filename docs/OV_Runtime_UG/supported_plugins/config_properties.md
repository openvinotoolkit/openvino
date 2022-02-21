# Query device properties, configuration {#openvino_docs_OV_UG_query_api}

## Query device properties and devices configuration

The OpenVINO™ toolkit supports inferencing with several types of devices (processors or accelerators).
This section provides a high-level description of the process of querying of different device properties and configuration values at runtime.

OpenVINO runtime has two types of properties:
- Read only properties which provides information about the devices (such device name, termal, execution capabilities, etc) and information about ov::CompiledModel to understand what configuration values were used to compile the model with.
- Mutable properties which are primarily used to configure ov::Core::compile_model process and affect final inference on the specific set of devices. Such properties can be set globally per device via ov::Core::set_property or locally for particular model in ov::Core::compile_model and ov::Core::query_model calls.

OpenVINO property is represented as a named constexpr variable with a given string name and type (see ). Example:
```c++
static constexpr Property<std::vector<std::string>, PropertyMutability::RO> available_devices{"AVAILABLE_DEVICES"};
```
represents a read-only property with C++ name `ov::available_devices`, string name `AVAILABLE_DEVICES` and type `std::vector<std::string>`.

Refer to the [Hello Query Device С++ Sample](../../../samples/cpp/hello_query_device/README.md) sources and the [Multi-Device execution](../multi_device.md) documentation for examples of using setting and getting properties in user applications.

### Get a set of available devices

Based on read-only property `ov::available_devices`, OpenVINO Core collects information about currently available devices enabled by OpenVINO plugins and returns information using the `ov::Core::get_available_devices` method:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_properties_api.cpp
       :language: cpp
       :fragment: [get_available_devices]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_properties_api.py
       :language: python
       :fragment: [get_available_devices]

@endsphinxdirective

The function returns a list of available devices, for example:

```
MYRIAD.1.2-ma2480
MYRIAD.1.4-ma2480
CPU
GPU.0
GPU.1
```

If there are more than one instance of a specific device, the devices are enumerated with `.suffix` where `suffix` is a unique string identifier. Each device name can then be passed to:

* `ov::Core::compile_model` to load the model to a specific device with specific configuration properties.
* `ov::Core::get_property` to get common or device specific properties.
* All other methods of the `ov::Core` class that accept `deviceName`.

### Working with properties in Your Code

The `ov::Core` class provides the following method to query device information, set or get different device configuration properties:

* `ov::Core::get_property` - Gets the current value of a specific property.
* `ov::Core::set_property` - Sets a new value for the property globally for specified `device_name`.

The `ov::CompiledModel` class is also extended to support the properties:

* `ov::CompiledModel::get_property`
* `ov::CompiledModel::set_property`

For documentation about OpenVINO common device-independent properties, refer to `openvino/runtime/properties.hpp`. Device specific configuration keys can be found in corresponding device folders (for example, `openvino/runtime/intel_gpu/properties.hpp`).

### Working with properties via ov::Core

#### Getting device properties

The code below demonstrates how to query `HETERO` device priority of devices which will be used to infer the model:

@snippet snippets/ov_properties_api.cpp hetero_priorities

> **NOTE**: All properties have a type, which is specified during property declaration. Based on this, actual type under `auto` is automatically deduced by C++ compiler.

To extract device properties such as available devices (`ov::available_devices`), device name (`ov::device::full_name`), supported properties (`ov::supported_properties`), and others, use the `ov::Core::get_property` method:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_properties_api.cpp
       :language: cpp
       :fragment: [cpu_device_name]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_properties_api.py
       :language: python
       :fragment: [cpu_device_name]

@endsphinxdirective

A returned value appears as follows: `Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz`.

#### Configure process of working with model

`ov::Core` methods like:

* `ov::Core::compile_model`
* `ov::Core::import_model`
* `ov::Core::query_model`

accept variadic list of properties as last arguments. Each property in such parameters lists should be used as function call to pass property value with specified property type.

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_properties_api.cpp
       :language: cpp
       :fragment: [compile_model_with_property]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_properties_api.py
       :language: python
       :fragment: [compile_model_with_property]

@endsphinxdirective

The example below specifies hints that a model should be compiled to be inferenced with multiple inference requests in parallel to achive best throughput while inference should be performed without accuracy loss with FP32 precision.

#### Setting properties globally

`ov::Core::set_property` with a given device name should be used to set global configuration properties which are the same accross multiple `ov::Core::compile_model`, `ov::Core::query_model`, etc calls, while setting property on the specific `ov::Core::compile_model` call applies properties only for current call:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_properties_api.cpp
       :language: cpp
       :fragment: [core_set_property_then_compile]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_properties_api.py
       :language: python
       :fragment: [core_set_property_then_compile]

@endsphinxdirective

### Properties on CompiledModel level

#### Getting property

The `ov::CompiledModel::get_property` method is used to get property values the compiled model has been created with or compiled model specific property such as `ov::optimal_number_of_infer_requests`:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_properties_api.cpp
       :language: cpp
       :fragment: [optimal_number_of_infer_requests]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_properties_api.py
       :language: python
       :fragment: [optimal_number_of_infer_requests]

@endsphinxdirective

Or the current temperature of the `MYRIAD` device:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_properties_api.cpp
       :language: cpp
       :fragment: [device_thermal]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_properties_api.py
       :language: python
       :fragment: [device_thermal]

@endsphinxdirective

Or the number of threads that would be used for inference on `CPU` device:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_properties_api.cpp
       :language: cpp
       :fragment: [inference_num_threads]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_properties_api.py
       :language: python
       :fragment: [inference_num_threads]

@endsphinxdirective

#### Setting properties for compiled model

The only device that supports this method is [Multi-Device execution](../multi_device.md):

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_properties_api.cpp
       :language: cpp
       :fragment: [multi_device]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_properties_api.py
       :language: python
       :fragment: [multi_device]

@endsphinxdirective
