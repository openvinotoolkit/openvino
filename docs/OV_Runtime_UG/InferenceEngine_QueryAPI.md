# Introduction to Inference Engine Device Query API {#openvino_docs_IE_DG_InferenceEngine_QueryAPI}

## Inference Engine Query API (C++)

@sphinxdirective
.. raw:: html

    <div id="switcher-cpp" class="switcher-anchor">C++</div>
@endsphinxdirective

The OpenVINO™ toolkit supports inferencing with several types of devices (processors or accelerators). 
This section provides a high-level description of the process of querying of different device properties and configuration values at runtime. Refer to the [Hello Query Device С++ Sample](../../samples/cpp/hello_query_device/README.md) sources and the [Multi-Device Plugin documentation](supported_plugins/MULTI.md) for examples of using the Inference Engine Query API in user applications.

### Using the Inference Engine Query API in Your Code

The `InferenceEngine::Core` class provides the following API to query device information, set or get different device configuration properties:

* `InferenceEngine::Core::GetAvailableDevices` - Provides a list of available devices. If there are more than one instance of a specific device, the devices are enumerated with `.suffix` where `suffix` is a unique string identifier. The device name can be passed to all methods of the `InferenceEngine::Core` class that work with devices, for example `InferenceEngine::Core::LoadNetwork`.
* `InferenceEngine::Core::GetMetric` - Provides information about specific device.
  `InferenceEngine::Core::GetConfig` - Gets the current value of a specific configuration key.
* `InferenceEngine::Core::SetConfig` - Sets a new value for the configuration key.

The `InferenceEngine::ExecutableNetwork` class is also extended to support the Query API:

* `InferenceEngine::ExecutableNetwork::GetMetric`
* `InferenceEngine::ExecutableNetwork::GetConfig`
* `InferenceEngine::ExecutableNetwork::SetConfig`

### Query API in the Core Class

#### GetAvailableDevices

@snippet snippets/InferenceEngine_QueryAPI0.cpp part0

The function returns a list of available devices, for example:

```
MYRIAD.1.2-ma2480
MYRIAD.1.4-ma2480
CPU
GPU.0
GPU.1
```

Each device name can then be passed to:

* `InferenceEngine::Core::LoadNetwork` to load the network to a specific device.
* `InferenceEngine::Core::GetMetric` to get common or device specific metrics.
* All other methods of the `InferenceEngine::Core` class that accept `deviceName`.

#### GetConfig()

The code below demonstrates how to understand whether the `HETERO` device dumps GraphViz `.dot` files with split graphs during the split stage:

@snippet snippets/InferenceEngine_QueryAPI1.cpp part1

For documentation about common configuration keys, refer to `ie_plugin_config.hpp`. Device specific configuration keys can be found in corresponding plugin folders.

#### GetMetric()

* To extract device properties such as available device, device name, supported configuration keys, and others, use the `InferenceEngine::Core::GetMetric` method:

@snippet snippets/InferenceEngine_QueryAPI2.cpp part2

A returned value appears as follows: `Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz`.

> **NOTE**: All metrics have a type, which is specified during metric instantiation. The list of common device-agnostic metrics can be found in `ie_plugin_config.hpp`. Device specific metrics (for example, for HDDL or MYRIAD devices) can be found in corresponding plugin folders.

### Query API in the ExecutableNetwork Class

#### GetMetric()

The method is used to get an executable network specific metric such as `METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)`:

@snippet snippets/InferenceEngine_QueryAPI3.cpp part3

Or the current temperature of the `MYRIAD` device:

@snippet snippets/InferenceEngine_QueryAPI4.cpp part4

#### GetConfig()

The method is used to get information about configuration values the executable network has been created with:

@snippet snippets/InferenceEngine_QueryAPI5.cpp part5

#### SetConfig()

The only device that supports this method is [Multi-Device](supported_plugins/MULTI.md).

## Inference Engine Query API (Python)

@sphinxdirective
.. raw:: html

    <div id="switcher-python" class="switcher-anchor">Python</div>
@endsphinxdirective

This section provides a high-level description of the process of querying of different device properties and configuration values. Refer to the [Hello Query Device Python Sample](../../samples/python/hello_query_device/README.md) sources and the [Multi-Device Plugin documentation](supported_plugins/MULTI.md) for examples of using the Inference Engine Query API in user applications.

### Using the Inference Engine Query API in Your Code

The Inference Engine [Core](api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino-inference-engine-iecore) class provides the following API to query device information, set or get different device configuration properties:

* [ie_api.IECore.available_devices](api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.available_devices) - Provides a list of available devices. If there are more than one instance of a specific device, the devices are enumerated with .suffix where suffix is a unique string identifier. The device name can be passed to all methods of the IECore class that work with devices, for example [ie_api.IECore.load_network](api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.load_network).
* [ie_api.ieCore.get_metric](api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.get_metric) - Provides information about specific device.
* [ie_api.IECore.get_config](api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.get_config) - Gets the current value of a specific configuration key.
* [ie_api.IECore.set_config](api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.set_config)  - Sets a new value for the configuration key.

The [ie_api.ExecutableNetwork](api/ie_python_api/_autosummary/openvino.inference_engine.ExecutableNetwork.html) class is also extended to support the Query API:
* [ie_api.ExecutableNetwork.get_metric](api/ie_python_api/_autosummary/openvino.inference_engine.ExecutableNetwork.html#openvino.inference_engine.ExecutableNetwork.get_metric)
* [ie_api.ExecutableNetwork.get_config](latest/api/ie_python_api/_autosummary/openvino.inference_engine.ExecutableNetwork.html#openvino.inference_engine.ExecutableNetwork.get_config)
* There is no method to call for set_config, but the equivalent action is described below.

### Query API in the IECore Class

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
* `IECore.get_metric` to get common or device specific metrics.
* All other methods of the `IECore` class that accept a device name.

#### Get Metric

To extract device properties such as available device, device name, supported configuration keys, and others, use the [IECore.get_metric](api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.get_metric) method:

```python
from openvino.inference_engine import IECore

ie = IECore()
ie.get_metric(device_name="CPU", metric_name="FULL_DEVICE_NAME")
```

A returned value appears as follows: `Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz`.

To list all supported metrics for a device:

```python
from openvino.inference_engine import IECore

ie = IECore()
ie.get_metric(device_name="GPU", metric_name="SUPPORTED_METRICS")
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
ie.get_metric(device_name=device, metric_name="SUPPORTED_CONFIG_KEYS")
```

For documentation about common configuration keys, refer to `ie_plugin_config.hpp`. Device specific configuration keys can be found in corresponding plugin folders.


### Query API in the ExecutableNetwork Class

#### Get Metric

To get the name of the loaded network:

```python
from openvino.inference_engine import IECore

ie = IECore()
net = ie.read_network(model=path_to_xml_file)
exec_net = ie.load_network(network=net, device_name=device)
exec_net.get_metric("NETWORK_NAME")
```

Use `exec_net.get_metric("SUPPORTED_METRICS")` to list all supported metrics for an ExecutableNetwork instance.


#### Get Configuration

The [IECore.get_config](api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.get_config) method is used to get information about configuration values the executable network has been created with:

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

Use `exec_net.get_metric("SUPPORTED_CONFIG_KEYS")`  to list all supported configuration keys.

#### Set Configuration

The only device that supports this method in the ExecutableNetwork class is the [Multi-Device](supported_plugins/MULTI.md), where you can change the priorities of the devices for the Multi plugin in real time: `exec_net.set_config({{"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}})`. See the Multi-Device documentation for more details.