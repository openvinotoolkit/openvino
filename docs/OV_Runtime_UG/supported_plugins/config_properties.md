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
   .. code-block:: sh
   
     auto property_keys = core.get_property("CPU", ov::supported_properties);
     
   It will return a vector of property keynames. Properties which can be changed will have the `ov::PropertyName::is_mutable` value set as `true`.
     
.. tab:: Python
   .. code-block:: sh
   
     property_keys = core.get_property("CPU", "SUPPORTED_PROPERTIES")
     
   It will return a dictionary of property key names and whether the property is read-only (`"RO"`) or writable (`"RW"`).
   
@endsphinxdirective

Each individual device property can be queried using the device name and the property name. For example, to query the “FULL_DEVICE_NAME” property of the “CPU” device, use:

@sphinxdirective

.. tab:: C++
   .. code-block:: sh
   
     auto cpu_device_name = core.get_property(“CPU”, ov::device::full_name);
     
.. tab:: Python
   .. code-block:: sh
   
     cpu_device_name = core.get_property(“CPU”, “FULL_DEVICE_NAME”)
   
@endsphinxdirective

This will return a value similar to: `12th Gen Intel(R) Core(TM) i7-12700`

#### Setting Device Properties
Devices have writable properties that globally configure how a model is compiled on the device. Once the configuration property is set, every model compiled on that device will use that configuration (unless it is specifically overwritten when calling the compile_model method as described below). These global properties are set using the set_property method, which takes the device name, property name, and desired property value as input arguments.

For example, setting the CPU’s “PERFORMANCE_HINT” property to “LATENCY” will cause every model that is compiled on the CPU to use LATENCY mode by default. However, if a different value is specified when calling compile_model, it will override the default value:

@sphinxdirective

.. tab:: C++
   .. code-block:: sh
   
   //Set LATENCY hint as a default for all models compiled on CPU
   core.set_proprety(“CPU”, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
   
   //Now, when model is compiled on CPU, it will use LATENCY mode by default
   auto compiled_model_latency = core.compile_model(model, “CPU”);
   
   //If a different performance hint is called out by compile_model, it will override the default value
   auto compiled_model_thrp = core.compile_model(model, “CPU”, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
   
.. tab:: Python
   .. code-block:: sh
   
   #Set LATENCY hint as a default for all models compiled on CPU
   core.set_proprety(“CPU”, {“PERFORMANCE_HINT”: “LATENCY”})
   
   #Now, when model is compiled on CPU, it will use LATENCY mode by default
   compiled_model_latency = core.compile_model(model, “CPU”)
   
   #If a different performance hint is called out by compile_model, it will override the default value
   compiled_model_thrp = core.compile_model(model, “CPU”, {“PERFORMANCE_HINT”:”THROUGHPUT”})
   
@endsphinxdirective
   
### Compiled Model Properties

When models are compiled in OpenVINO using the `ov::Core::compile_model` (C++) or [Core.compile_model](api/ie_python_api/_autosummary/openvino.runtime.Core.html#openvino.runtime.Core.compile_model) (Python) method, a CompiledModel object is created that represents the model. The compiled model has its own properties that show its configuration (such as how many processing streams the model will use) and give optimal parameters for running inference (such as optimal batch size on GPUs).

Similar to device properties, OpenVINO provides two methods to query device information or set configuration parameters:
* `ov::CompiledModel::get_property` (C++) or [CompiledModel.get_property](api/ie_python_api/_autosummary/openvino.runtime.CompiledModel.html#openvino.runtime.CompiledModel.get_property) (Python) - Gets the current value of a specific property for a compiled model.
* `ov::CompiledModel::set_property` (C++) or [CompiledModel.set_property](api/ie_python_api/_autosummary/openvino.runtime.CompiledModel.html#openvino.runtime.CompiledModel.set_property) (Python) - Sets a new value for the specific property for a compiled model (writeable properties only).

#### Getting Compiled Model Properties
The CompiledModel get_property method is used to query properties of compiled models of a CompiledModel object. It takes the specific property key as an argument, and returns the value of that property.

To list all the property keys of a CompiledModel object, query the “SUPPORTED_PROPERTIES” key:

@sphinxdirective

.. tab:: C++
   .. code_block:: sh
   
   auto compiled_model = core.compile_model(model, "CPU");
   auto model_property_keys = compiled_model.get_property(ov::supported_properties);
   
.. tab:: Python
   .. code_block:: sh
   
   compiled_model = core.compile_model(model, “CPU”)
   model_property_keys = compiled_model.get_property(“SUPPORTED_PROPERTIES”)
   
@endsphinxdirective

#### Setting Compiled Model Properties
Model properties can be set using the CompiledModel set_property method. The only mode that supports setting compiled model properties is [Multi-Device execution](../multi_device.md) mode. It allows you to set the device priorities for a model that has been compiled with “MULTI”:

@sphinxdirective

.. tab:: C++
   .. code_block:: sh
   auto compiled_model = core.compile_model(model, "MULTI", ov::device::priorities("CPU", "GPU"));
   
   //Change the order of priorities
   compiled_model.set_property(ov::device::priorities("GPU", "CPU"));
   
.. tab:: Python
   .. code_block:: sh
   config = {"MULTI_DEVICE_PRIORITIES": "CPU,GPU"}
   compiled_model = core.compile_model(model, "MULTI", config)
   
   #Change the order of priorities
   compiled_model.set_property({"MULTI_DEVICE_PRIORITIES": "GPU,CPU"})
   
@endsphinxdirective

### Example: Hello Query Device
To see a full example of how to list all devices and their properties in an OpenVINO application, try the following samples:
* [Hello Query Device C++](../../../samples/cpp/hello_query_device/README.md)
* [Hello Query Device Python](../../../samples/python/hello_query_device/README.md)
