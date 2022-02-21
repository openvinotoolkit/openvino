# Introduction to OpenVINO™ Device Properties API {#openvino_docs_IE_DG_InferenceEngine_QueryAPI}

## OpenVINO™ Properties API (C++)

@sphinxdirective
.. raw:: html

    <div id="switcher-cpp" class="switcher-anchor">C++</div>
@endsphinxdirective

The OpenVINO™ toolkit supports inferencing with several types of devices (processors or accelerators).
This section provides a high-level description of the process of querying of different device properties and configuration values at runtime. Refer to the [Hello Query Device С++ Sample](../../samples/cpp/hello_query_device/README.md) sources and the [Multi-Device Plugin documentation](supported_plugins/MULTI.md) for examples of using the OpenVINO™ Properties API in user applications.

### Using the OpenVINO™ Properties API in Your Code

The `ov::Core` class provides the following API to query device information, set or get different device configuration properties:

* `ov::Core::get_available_devices` - Provides a list of available devices. If there are more than one instance of a specific device, the devices are enumerated with `.suffix` where `suffix` is a unique string identifier. The device name can be passed to all methods of the `ov::Core` class that work with devices, for example `ov::Core::compile_model`.
* `ov::Core::get_property` - Gets the current value of a specific property.
* `ov::Core::set_property` - Sets a new value for the property.

The `ov::CompiledModel` class is also extended to support the Properties API:

* `ov::CompiledModel::get_property`
* `ov::CompiledModel::set_property`

### Properties API in the Core Class

#### get_available_devices

@snippet snippets/ov_properties_api.cpp part0

The function returns a list of available devices, for example:

```
MYRIAD.1.2-ma2480
MYRIAD.1.4-ma2480
CPU
GPU.0
GPU.1
```

Each device name can then be passed to:

* `ov::Core::compile_model` to load the network to a specific device.
* `ov::Core::get_property` to get common or device specific properties.
* All other methods of the `ov::Core` class that accept `deviceName`.

#### ov::Core methods

`ov::Core` methods like:

* `ov::Core::compile_model`
* `ov::Core::import_model`
* `ov::Core::query_model`
* `ov::Core::query_model`
* `ov::Core::create_context`

accept variadic list of properties as last arguments. Each property in such parameters lists should be used as function call to pass property value with specified property type

@snippet snippets/ov_properties_api.cpp part3

#### get_property()

For documentation about common configuration keys, refer to `openvino/runtime/properties.hpp`. Device specific configuration keys can be found in corresponding plugin folders.

* The code below demonstrates how to query `HETERO` device priority of devices which will be used to infer the model:

@snippet snippets/ov_properties_api.cpp part1

* To extract device properties such as available device, device name, supported configuration keys, and others, use the `ov::Core::get_property` method:

@snippet snippets/ov_properties_api.cpp part2

A returned value appears as follows: `Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz`.

> **NOTE**: All properties have a type, which is specified during property declaration. The list of common device-agnostic properties can be found in `openvino/runtime/properties.hpp`. Device specific properties (for example, for HDDL or MYRIAD devices) can be found in corresponding plugin folders.

### Properties API in the CompiledModel Class

#### get_property()

The method is used to get configuration values the compiled model has been created with or compiled model specific property such as `ov::optimal_number_of_infer_requests`:

@snippet snippets/ov_properties_api.cpp part4

Or the current temperature of the `MYRIAD` device:

@snippet snippets/ov_properties_api.cpp part5

Or the number of threads that would be used for inference in `CPU` device:

@snippet snippets/ov_properties_api.cpp part6

#### set_property()

The only device that supports this method is [Multi-Device](supported_plugins/MULTI.md).

## OpenVINO™ Properties API (Python)

@sphinxdirective
.. raw:: html

    <div id="switcher-python" class="switcher-anchor">Python</div>
@endsphinxdirective

This section provides a high-level description of the process of querying of different device properties and configuration values. Refer to the [Hello Properties Device Python Sample](../../samples/python/hello_query_device/README.md) sources and the [Multi-Device Plugin documentation](supported_plugins/MULTI.md) for examples of using the OpenVINO™ Properties API in user applications.

### Using the OpenVINO™ Properties API in Your Code

The OpenVINO™ [Core](api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino-inference-engine-iecore) class provides the following API to query device information, set or get different device configuration properties:

* [ie_api.IECore.available_devices](api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.available_devices) - Provides a list of available devices. If there are more than one instance of a specific device, the devices are enumerated with .suffix where suffix is a unique string identifier. The device name can be passed to all methods of the IECore class that work with devices, for example [ie_api.IECore.load_network](api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.load_network).
* [ie_api.ieCore.get_property](api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.get_property) - Provides information about specific device.
* [ie_api.IECore.get_config](api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.get_config) - Gets the current value of a specific configuration key.
* [ie_api.IECore.set_config](api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.set_config)  - Sets a new value for the configuration key.

The [ie_api.CompiledModel](api/ie_python_api/_autosummary/openvino.inference_engine.CompiledModel.html) class is also extended to support the Properties API:
* [ie_api.CompiledModel.get_property](api/ie_python_api/_autosummary/openvino.inference_engine.CompiledModel.html#openvino.inference_engine.CompiledModel.get_property)
* [ie_api.CompiledModel.get_config](latest/api/ie_python_api/_autosummary/openvino.inference_engine.CompiledModel.html#openvino.inference_engine.CompiledModel.get_config)
* There is no method to call for set_config, but the equivalent action is described below.

### Properties API in the IECore Class

#### Get Available Devices

```python
from openvino.inference_engine import IECore

ie = IECore()
print(ie.available_devices)
```

This code prints a list of available devices, for example:

```
MYRIAD.1.2-ma2480
MYRIAD.1.4-ma2480
FPGA.0
FPGA.1
CPU
GPU.0
GPU.1
```

Each device name can then be passed to:

* `IECore.load_network` to load the network to a specific device.
* `IECore.get_property` to get common or device specific properties.
* All other methods of the `IECore` class that accept a device name.

#### Get Metric

To extract device properties such as available device, device name, supported configuration keys, and others, use the [IECore.get_property](api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.get_property) method:

```python
from openvino.inference_engine import IECore

ie = IECore()
ie.get_property(device_name="CPU", property_name="FULL_DEVICE_NAME")
```

A returned value appears as follows: `Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz`.

To list all supported properties for a device:

```python
from openvino.inference_engine import IECore

ie = IECore()
ie.get_property(device_name="GPU", property_name="SUPPORTED_METRICS")
```

#### Get Configuration

The code below uses the [IECore.get_config](api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.get_config) method and demonstrates how to understand whether the HETERO device dumps .dot files with split graphs during the split stage:

```python
from openvino.inference_engine import IECore

ie = IECore()
ie.get_config(device_name="HETERO", config_name="HETERO_DUMP_GRAPH_DOT")
```

To list all supported configuration keys for a device:

```python
from openvino.inference_engine import IECore

ie = IECore()
ie.get_property(device_name=device, property_name="SUPPORTED_CONFIG_KEYS")
```

For documentation about common configuration keys, refer to `ie_plugin_config.hpp`. Device specific configuration keys can be found in corresponding plugin folders.


### Properties API in the CompiledModel Class

#### Get Metric

To get the name of the loaded network:

```python
from openvino.inference_engine import IECore

ie = IECore()
net = ie.read_network(model=path_to_xml_file)
exec_net = ie.load_network(network=net, device_name=device)
exec_net.get_property("NETWORK_NAME")
```

Use `exec_net.get_property("SUPPORTED_METRICS")` to list all supported properties for an CompiledModel instance.


#### Get Configuration

The [IECore.get_config](api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.get_config) method is used to get information about configuration values the compiled model has been created with:

```python
from openvino.inference_engine import IECore

ie = IECore()
net = ie.read_network(model=path_to_xml_file)
exec_net = ie.load_network(network=net, device_name="CPU")
exec_net.get_config("CPU_THREADS_NUM")
```

Or the current temperature of MYRIAD device:

```python
from openvino.inference_engine import IECore

ie = IECore()
net = ie.read_network(model=path_to_xml_file)
exec_net = ie.load_network(network=net, device_name="MYRIAD")
exec_net.get_config("DEVICE_THERMAL")
```

Use `exec_net.get_property("SUPPORTED_CONFIG_KEYS")`  to list all supported configuration keys.

#### Set Configuration

The only device that supports this method in the CompiledModel class is the [Multi-Device](supported_plugins/MULTI.md), where you can change the priorities of the devices for the Multi plugin in real time: `exec_net.set_config({{"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}})`. See the Multi-Device documentation for more details.