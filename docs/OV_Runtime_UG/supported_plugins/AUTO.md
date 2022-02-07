# Auto-Device Plugin {#openvino_docs_IE_DG_supported_plugins_AUTO}

## Auto-Device Plugin Execution (C++)

@sphinxdirective
.. raw:: html

    <div id="switcher-cpp" class="switcher-anchor">C++</div>
@endsphinxdirective

The AUTO device is a new, special "virtual" or "proxy" device in the OpenVINO™ toolkit.

Use "AUTO" as the device name to delegate selection of an actual accelerator to OpenVINO. The Auto-device plugin internally recognizes and selects devices from among CPU, integrated GPU and discrete Intel GPUs (when available) depending on the device capabilities and the characteristics of CNN models (for example, precision). Then the Auto-device assigns inference requests to the selected device.

From the application's point of view, this is just another device that handles all accelerators in the full system.

With the 2021.4 release, Auto-device setup is done in three major steps:
1. Configure each device as usual (for example, via the conventional `SetConfig()` method)
2. Load a network to the Auto-device plugin. This is the only change needed in your application.
3. As with any other executable network resulting from `LoadNetwork()`, create as many requests as needed to saturate the devices. 

These steps are covered below in detail.

### Defining and Configuring the Auto-Device Plugin
Following the OpenVINO convention for devices names, the Auto-device uses the label "AUTO". The only configuration option for Auto-device is a limited device list:

| Parameter name     | Parameter values      | Default            |             Description                                                      |
| :---               | :---                  | :---               |:-----------------------------------------------------------------------------|
| "MULTI_DEVICE_PRIORITIES" | comma-separated device names <span style="color:red">with no spaces</span>| N/A | Device candidate list to be selected    |

You can use the configuration name directly as a string or use `InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES` from `multi-device/multi_device_config.hpp`, which defines the same string.

There are two ways to use Auto-device:
1. Directly indicate device by "AUTO" or an empty string:
@snippet snippets/AUTO0.cpp part0

2. Use the Auto-device configuration:
@snippet snippets/AUTO1.cpp part1

Both methods allow limiting the list of device candidates for the AUTO plugin.

> **NOTE**: The Inference Engine lets you use "GPU" as an alias for "GPU.0" in function calls. 

The Auto-device plugin supports query device optimization capabilities in metric.

| Parameter name                 | Parameter values         |
| :---                           | :---                     |
| "OPTIMIZATION_CAPABILITIES"    | Auto-Device capabilities |

### Enumerating Devices and Selection Logic

The Inference Engine now features a dedicated API to enumerate devices and their capabilities. 
See [Hello Query Device C++ Sample](../../../samples/cpp/hello_query_device/README.md).
This is the example output from the sample (truncated to device names only):

```sh
./hello_query_device
Available devices: 
    Device: CPU
...
    Device: GPU.0
...
    Device: GPU.1
```

### Default Auto-Device Selection Logic

With the 2021.4 release, the Auto-Device selects the most suitable device using the following default logic:

1. Check if dGPU (discrete), iGPU (integrated) and CPU devices are available
2. Get the precision of the input model, such as FP32
3. According to the priority of dGPU, iGPU, and CPU (in this order), if the device supports the precision of the input network, select it as the most suitable device

For example, CPU, dGPU and iGPU can support the following precision and optimization capabilities:

| Device   | OPTIMIZATION_CAPABILITIES       |
| :---     | :---                            |
| CPU      | WINOGRAD FP32 FP16 INT8 BIN     |
| dGPU     | FP32 BIN BATCHED_BLOB FP16 INT8 |
| iGPU     | FP32 BIN BATCHED_BLOB FP16 INT8 |

* When the application uses the Auto-device to run FP16 IR on a system with CPU, dGPU and iGPU, Auto-device will offload this workload to dGPU.
* When the application uses the Auto-device to run FP16 IR on a system with CPU and iGPU, Auto-device will offload this workload to iGPU.
* When the application uses the Auto-device to run WINOGRAD-enabled IR on a system with CPU, dGPU and iGPU, Auto-device will offload this workload to CPU.

In cases when loading the network to dGPU or iGPU fails, CPU is the fall-back choice.

According to the Auto-device selection logic from the previous section, tell the Inference Engine 
to use the most suitable device from available devices as follows:

@snippet snippets/AUTO2.cpp part2

You can also use the Auto-device plugin to choose a device from a limited choice of devices, in this example CPU and GPU:

@snippet snippets/AUTO3.cpp part3

### Configuring the Individual Devices and Creating the Auto-Device on Top

It is possible to configure each individual device as usual and create the "AUTO" device on top:

@snippet snippets/AUTO4.cpp part4

Alternatively, you can combine all the individual device settings into single config file and load it, allowing the Auto-device plugin to parse and apply it to the right devices. See the code example here:

@snippet snippets/AUTO5.cpp part5

### Using the Auto-Device with OpenVINO Samples and Benchmark App

Note that every OpenVINO sample or application that supports the "-d" (which stands for "device") command-line option transparently accepts the Auto-device. The Benchmark Application is the best example of the optimal usage of the Auto-device. You do not need to set the number of requests and CPU threads, as the application provides optimal out-of-the-box performance. Below is the example command-line to evaluate AUTO performance with that:

@sphinxdirective
.. tab:: Package, Docker, open-source installation

   .. code-block:: sh

      ./benchmark_app.py –d AUTO –m <model>

.. tab:: pip installation

    .. code-block:: sh

      benchmark_app –d AUTO –m <model>

@endsphinxdirective


You can also use the auto-device with limit device choice:

@sphinxdirective
.. tab:: Package, Docker, open-source installation

   .. code-block:: sh

      ./benchmark_app.py –d AUTO:CPU,GPU –m <model>

.. tab:: pip installation

    .. code-block:: sh

      benchmark_app –d AUTO:CPU,GPU –m <model>

@endsphinxdirective

**NOTES:**
* The default CPU stream is 1 if using `-d AUTO`. 
* You can use the FP16 IR to work with Auto-device.
* No demos are fully optimized for Auto-device yet to select the most suitable device, 
use GPU streams/throttling, and so on.

## Auto-Device Plugin Execution (Python)

@sphinxdirective
.. raw:: html

    <div id="switcher-python" class="switcher-anchor">Python</div>
@endsphinxdirective

The AUTO device is a new, special "virtual" or "proxy" device in the OpenVINO™ toolkit.

Use "AUTO" as the device name to delegate selection of an actual accelerator to OpenVINO. The Auto-device plugin internally recognizes and selects devices from among CPU, integrated GPU and discrete Intel GPUs (when available) depending on the device capabilities and the characteristics of CNN models (for example, precision). Then the Auto-device assigns inference requests to the selected device.

From the application's point of view, this is just another device that handles all accelerators in the full system.

With the 2021.4 release, Auto-device setup is done in three major steps:

1. Configure each device as usual (for example, via the conventional [IECore.set_config](https://docs.openvino.ai/latest/ie_python_api/classie__api_1_1IECore.html#a2c738cee90fca27146e629825c039a05) method).
2. Load a network to the Auto-device plugin. This is the only change needed in your application.
3. As with any other executable network resulting from [IECore.load_network](https://docs.openvino.ai/latest/ie_python_api/classie__api_1_1IECore.html#ac9a2e043d14ccfa9c6bbf626cfd69fcc), create as many requests as needed to saturate the devices. 

These steps are covered below in detail.

### Defining and Configuring the Auto-Device Plugin
Following the OpenVINO convention for devices names, the Auto-device uses the label "AUTO". The only configuration option for Auto-device is a limited device list:

| Parameter name | Parameter values | Default | Description |
| -------------- | ---------------- | ------- | ----------- |
| "AUTO_DEVICE_LIST" | comma-separated device names with no spaces | N/A | Device candidate list to be selected

There are two ways to use the Auto-device plugin:

1. Directly indicate device by "AUTO" or an empty string.
2. Use the Auto-device configuration

Both methods allow limiting the list of device candidates for the AUTO plugin.

```python
from openvino.inference_engine import IECore

ie = IECore()
# Read a network in IR or ONNX format
net = ie.read_network(model=path_to_model)

# Load a network on the "AUTO" device
exec_net = ie.load_network(network=net, device_name="AUTO")

# Optionally specify the list of device candidates for the AUTO plugin
# The following two lines are equivalent
exec_net = ie.load_network(network=net, device_name="AUTO:CPU,GPU")
exec_net = ie.load_network(network=net, device_name="AUTO",
                           config={"AUTO_DEVICE_LIST": "CPU,GPU"})
```

The Auto-device plugin supports query device optimization capabilities in metric.

| Parameter name | Parameter values |
| --- | --- |
| "OPTIMIZATION_CAPABILITIES" | Auto-Device capabilities |

### Enumerating Devices and Selection Logic

The Inference Engine now features a dedicated API to enumerate devices and their capabilities. See the [Hello Query Device Python Sample](../../../inference_engine/ie_bridges/python/sample_hello_query_device_README.html) for code.

This is the example output from the sample (truncated to device names only):

```python
./hello_query_device

Available devices:
    Device: CPU
...
    Device: GPU.0
...
    Device: GPU.1
```

### Default Auto-Device Selection Logic

With the 2021.4 release, the Auto-Device selects the most suitable device using the following default logic:

1. Check if dGPU (discrete), iGPU (integrated) and CPU devices are available
2. Get the precision of the input model, such as FP32
3. According to the priority of dGPU, iGPU, and CPU (in this order), if the device supports the precision of the input network, select it as the most suitable device

For example, CPU, dGPU and iGPU can support the following precision and optimization capabilities:

| Device | OPTIMIZATION_CAPABILITIES |
| --- | --- |
| CPU | WINOGRAD FP32 FP16 INT8 BIN |
| dGPU | FP32 BIN BATCHED_BLOB FP16 INT8 |
| iGPU | FP32 BIN BATCHED_BLOB FP16 INT8 |

* When the application uses the Auto-device to run FP16 IR on a system with CPU, dGPU and iGPU, Auto-device will offload this workload to dGPU.
* When the application uses the Auto-device to run FP16 IR on a system with CPU and iGPU, Auto-device will offload this workload to iGPU.
* When the application uses the Auto-device to run WINOGRAD-enabled IR on a system with CPU, dGPU and iGPU, Auto-device will offload this workload to CPU.

In cases when loading the network to dGPU or iGPU fails, CPU is the fall-back choice.

To show the capabilities for a specific device, query the OPTIMIZATION_CAPABILITIES metric:


```python
from openvino.inference_engine import IECore

ie = IECore()
ie.get_metric(device_name=device,
              metric_name="OPTIMIZATION_CAPABILITIES")
```

### Configuring the Individual Devices and Creating the Auto-Device on Top

It is possible to configure each individual device as usual and create the "AUTO" device on top:

```python
from openvino.inference_engine import IECore

ie = IECore()
net = ie.read_network(model=path_to_model)

cpu_config = {}
gpu_config = {}

ie.set_config(config=cpu_config, device_name="CPU")
ie.set_config(config=gpu_config, device_name="GPU")

# Load the network to the AUTO device
exec_net = ie.load_network(network=net, device_name="AUTO")
```

Alternatively, you can combine all the individual device settings into single config file and load it, allowing the Auto-device plugin to parse and apply it to the right devices. See the code example here:

```python
from openvino.inference_engine import IECore

# Init the Inference Engine Core
ie = IECore()

# Read a network in IR or ONNX format
net = ie.read_network(model=path_to_model)

full_config = {}

# Load the network to the AUTO device
exec_net = ie.load_network(network=net, device_name="AUTO", config=full_config)
```

### Using the Auto-Device with OpenVINO Samples and Benchmark App

Note that every OpenVINO sample or application that supports the "-d" (which stands for "device") command-line option transparently accepts the Auto-device. The Benchmark Application is the best example of the optimal usage of the Auto-device. You do not need to set the number of requests and CPU threads, as the application provides optimal out-of-the-box performance. Below is the example command-line to evaluate AUTO performance with that:

@sphinxdirective
.. tab:: Package, Docker, open-source installation

   .. code-block:: sh

      ./benchmark_app.py –d AUTO –m <model>

.. tab:: pip installation

    .. code-block:: sh

      benchmark_app –d AUTO –m <model>

@endsphinxdirective

You can also use the auto-device with limit device choice:

@sphinxdirective
.. tab:: Package, Docker, open-source installation

   .. code-block:: sh

      ./benchmark_app.py –d AUTO:CPU,GPU –m <model>

.. tab:: pip installation

    .. code-block:: sh

      benchmark_app –d AUTO:CPU,GPU –m <model>

@endsphinxdirective

> **NOTE**: If you installed OpenVINO with pip, use `benchmark_app -d AUTO:CPU,GPU -m <model>`
