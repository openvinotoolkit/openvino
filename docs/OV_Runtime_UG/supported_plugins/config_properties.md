# Query Device Properties - Configuration {#openvino_docs_OV_UG_query_api}

OpenVINO™ toolkit supports inference with several types of processor and accelerator devices. This section provides a high-level explanation of how to query various device and model properties and configure them at runtime.

There are two types of properties that are queryable in OpenVINO Runtime:
* Device properties - these provide information about the hardware, such as device name, supported data types, and execution capabilities. Devices have read only properties that describe intrinsic characteristics of the device (such as number of execution units for a GPU) and mutable properties that can be used to configure how models are compiled on the device.
* Compiled model properties - Once a model has been compiled using the compile_model method, it has properties showing the model’s configuration parameters, such as the model’s priority, performance hint, and cache directory.

Device and model properties are represented as a named variable with a given string name and a type. Each property has a key name and a value associated with it.

The following sections show how to query device and model properties and configure them. Refer to the [Hello Query Device C++](../../../samples/cpp/hello_query_device/README.md) and [Hello Query Device Python](../../../samples/python/hello_query_device/README.md) samples to see more example code showing how to get and set properties in user applications.

### Querying Available Devices

To find all the available processing devices and accelerators in the system, use the `ov::Core::get_available_devices` (C++) or [Core.available_devices](api/ie_python_api/_autosummary/openvino.runtime.Core.html#openvino.runtime.Core.available_devices) (Python) method:

Based on the `ov::available_devices` read-only property, OpenVINO Core collects information about currently available devices enabled by OpenVINO plugins and returns information, using the `ov::Core::get_available_devices` method:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_properties_api.cpp get_available_devices

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_properties_api.py get_available_devices

@endsphinxtab

@endsphinxtabset


This returns a list of available device names, for example:

```
CPU
GPU.0
GPU.1
```

If there are multiple instances of a specific device, each device is assigned a unique DEVICE_ID that is appended to the device name (e.g. `GPU.0` or `GPU.1`). These device names can be used to configure each device as described in the following sections. 

Note: Devices that have not been configured to work with OpenVINO will not be listed by get_available_devices. To configure hardware to work with OpenVINO, follow the instructions on the [Additional Configurations for Hardware](../../install_guides/configurations-header.md) page.

### Device Properties

Each hardware device has a set of read-only properties that describe characteristics of the device and a set of writeable properties that can be used to configure how models are compiled on the device. To see a full list of properties and definitions for each type of hardware device, see the [Device Properties API specification](https://docs.openvino.ai/latest/groupov_runtime_cpp_prop_api.html). 

OpenVINO provides two methods to query device information or set configuration parameters:
* `ov::Core::get_property` (C++) or [Core.get_property](api/ie_python_api/_autosummary/openvino.runtime.Core.html#openvino.runtime.Core.get_property) (Python) - Gets the current value of a specific property for a device.
* `ov::Core::set_property` (C++) or [Core.set_property](api/ie_python_api/_autosummary/openvino.runtime.Core.html#openvino.runtime.Core.set_property) (Python) - Sets a new value for the specific property for a device (writeable properties only).

The properties can also be listed using the get_property method with the “SUPPORTED_PROPERTIES” key as described in the next section. Each dedicated device page gives more information on device-specific properties:
* [CPU Device](CPU.md)
* [GPU Device](GPU.md)
* [VPU Devices](VPU.md)
* [GNA Device](GNA.md)
* [Arm® CPU Device](ARM_CPU)

#### Getting Device Properties
The get_property method is used to query properties of each device. It takes the name of the device and the specific property key as arguments, and returns the value for that property on that device. 

To list all of the property keys of a certain device, query the `SUPPORTED_PROPERTIES` property:

@sphinxdirective

.. tab:: C++
   .. code-block::
   
     auto property_keys = core.get_property("CPU", ov::supported_properties);
     
   It will return a vector of property keynames. Properties which can be changed will have the `ov::PropertyName::is_mutable` value set as `true`.
     
.. tab:: Python
   .. code-block
   
     property_keys = core.get_property("CPU", "SUPPORTED_PROPERTIES")
     
   It will return a dictionary of property key names and whether the property is read-only (`"RO"`) or writable (`"RW"`).
   
@endsphinxdirective

Each individual device property can be queried using the device name and the property name. For example, to query the “FULL_DEVICE_NAME” property of the “CPU” device, use:

@sphinxdirective

.. tab:: C++
   .. code-block::
   
     auto cpu_device_name = core.get_property(“CPU”, ov::device::full_name);
     
.. tab:: Python
   .. code-block
   
     cpu_device_name = core.get_property(“CPU”, “FULL_DEVICE_NAME”)
   
@endsphinxdirective

This will return a value similar to: `12th Gen Intel(R) Core(TM) i7-12700`

#### Setting Device Properties
Devices have writable properties that globally configure how a model is compiled on the device. Once the configuration property is set, every model compiled on that device will use that configuration (unless it is specifically overwritten when calling the compile_model method as described below). These global properties are set using the set_property method, which takes the device name, property name, and desired property value as input arguments.

For example, setting the CPU’s “PERFORMANCE_HINT” property to “LATENCY” will cause every model that is compiled on the CPU to use LATENCY mode by default. However, if a different value is specified when calling compile_model, it will override the default value:

@sphinxdirective

.. tab:: C++
   .. code-block::
   
   // Set LATENCY hint as a default for all models compiled on CPU
   core.set_proprety(“CPU”, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
   
   // Now, when model is compiled on CPU, it will use LATENCY mode by default
   auto compiled_model_latency = core.compile_model(model, “CPU”);
   
   // If a different performance hint is called out by compile_model, it will override the default value
   auto compiled_model_thrp = core.compile_model(model, “CPU”, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
   
.. tab:: Python
   .. code-block::
   
   # Set LATENCY hint as a default for all models compiled on CPU
   core.set_proprety(“CPU”, {“PERFORMANCE_HINT”: “LATENCY”})
   
   # Now, when model is compiled on CPU, it will use LATENCY mode by default
   compiled_model_latency = core.compile_model(model, “CPU”)
   
   # If a different performance hint is called out by compile_model, it will override the default value
   compiled_model_thrp = core.compile_model(model, “CPU”, {“PERFORMANCE_HINT”:”THROUGHPUT”})
   
@endsphinxdirective
   
### Compiled Model Properties




To extract device properties such as available devices (`ov::available_devices`), device name (`ov::device::full_name`), supported properties (`ov::supported_properties`), and others, use the `ov::Core::get_property` method:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_properties_api.cpp cpu_device_name

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_properties_api.py cpu_device_name

@endsphinxtab

@endsphinxtabset

A returned value appears as follows: `Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz`.

> **NOTE**: In order to understand a list of supported properties on `ov::Core` or `ov::CompiledModel` levels, use `ov::supported_properties` which contains a vector of supported property names. Properties which can be changed, has `ov::PropertyName::is_mutable` returning the `true` value. Most of the properites which are changable on `ov::Core` level, cannot be changed once the model is compiled, so it becomes immutable read-only property.

#### Configure a Work with a Model

The `ov::Core` methods like:

* `ov::Core::compile_model`
* `ov::Core::import_model`
* `ov::Core::query_model`

accept a selection of properties as last arguments. Each of the properties should be used as a function call to pass a property value with a specified property type.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_properties_api.cpp compile_model_with_property

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_properties_api.py compile_model_with_property

@endsphinxtab

@endsphinxtabset

The example below specifies hints that a model should be compiled to be inferred with multiple inference requests in parallel to achieve best throughput, while inference should be performed without accuracy loss with FP32 precision.

#### Setting Properties Globally

`ov::Core::set_property` with a given device name should be used to set global configuration properties, which are the same across multiple `ov::Core::compile_model`, `ov::Core::query_model`, and other calls. However, setting properties on a specific `ov::Core::compile_model` call applies properties only for the current call:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_properties_api.cpp core_set_property_then_compile

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_properties_api.py core_set_property_then_compile

@endsphinxtab

@endsphinxtabset

### Properties on CompiledModel Level

#### Getting Property

The `ov::CompiledModel::get_property` method is used to get property values the compiled model has been created with or a compiled model level property such as `ov::optimal_number_of_infer_requests`:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_properties_api.cpp optimal_number_of_infer_requests

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_properties_api.py optimal_number_of_infer_requests

@endsphinxtab

@endsphinxtabset

Or the current temperature of the `MYRIAD` device:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_properties_api.cpp device_thermal

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_properties_api.py device_thermal

@endsphinxtab

@endsphinxtabset


Or the number of threads that would be used for inference on `CPU` device:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_properties_api.cpp inference_num_threads

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_properties_api.py inference_num_threads

@endsphinxtab

@endsphinxtabset

#### Setting Properties for Compiled Model

The only mode that supports this method is [Multi-Device execution](../multi_device.md):

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_properties_api.cpp multi_device

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_properties_api.py multi_device

@endsphinxtab

@endsphinxtabset
