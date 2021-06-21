# Auto-Device Plugin {#openvino_docs_IE_DG_supported_plugins_AUTO}

## Auto-Device Plugin Execution

Auto-device is a new special "virtual" or "proxy" device in the OpenVINO™ toolkit. 

Use "AUTO" as the device name to delegate selection of an actual accelerator to OpenVINO. 
With the 2021.4 release, Auto-device internally recognizes and selects devices from CPU, 
integrated GPU and discrete Intel GPUs (when available) depending on the device capabilities and the characteristic of CNN models, 
for example, precisions. Then Auto-device assigns inference requests to the selected device.

From the application point of view, this is just another device that handles all accelerators in full system. 

With the 2021.4 release, Auto-device setup is done in three major steps:
* Step 1: Configure each device as usual (for example, via the conventional <code>SetConfig</code> method)
* Step 2: Load a network to the Auto-device plugin. This is the only change needed in your application
* Step 3: Just like with any other executable network (resulted from <code>LoadNetwork</code>), create as many requests as needed to saturate the devices. 
These steps are covered below in details.


## Defining and Configuring the Auto-Device Plugin
Following the OpenVINO notions of “devices”, the Auto-device has “AUTO” name. The only configuration option for Auto-device is a limited device list:

| Parameter name     | Parameter values      | Default            |             Description                                                      |
| :---               | :---                  | :---               |:-----------------------------------------------------------------------------|
| "AUTO_DEVICE_LIST" | comma-separated device names <span style="color:red">with no spaces</span>| N/A | Device candidate list to be selected    |

You can use the configuration name directly as a string or use <code>IE::KEY_AUTO_DEVICE_LIST</code> from <code>ie_plugin_config.hpp</code>,
which defines the same string.

There are two ways to use Auto-device:
1. Directly indicate device by “AUTO” or empty string:

@snippet snippets/AUTO0.cpp part0

2. Use Auto-device configuration to limit the device candidates list to be selected:

@snippet snippets/AUTO1.cpp part1

Auto-device supports query device optimization capabilities in metric;

| Parameter name                 | Parameter values         |
| :---                           | :---                     |
| "OPTIMIZATION_CAPABILITIES"    | Auto-Device capabilities |

## Enumerating Available Devices and Auto-Device Selecting Logic

### Enumerating Available Devices

Inference Engine now features a dedicated API to enumerate devices and their capabilities. 
See [Hello Query Device C++ Sample](../../../inference-engine/samples/hello_query_device/README.md).
This is the example output from the sample (truncated to the devices' names only):

```sh
./hello_query_device
Available devices: 
    Device: CPU
...
    Device: GPU.0
...
    Device: GPU.1
```

###	Default Auto-Device selecting logic

With the 2021.4 release, Auto-Device selects the most suitable device with following default logic:
1.	Check if dGPU, iGPU and CPU device are available
2.	Get the precision of the input model, such as FP32
3.	According to the priority of dGPU, iGPU and CPU (in this order), if the device supports the precision of input network, select it as the most suitable device

For example, CPU, dGPU and iGPU can support below precision and optimization capabilities:

| Device   | OPTIMIZATION_CAPABILITIES       |
| :---     | :---                            |
| CPU      | WINOGRAD FP32 FP16 INT8 BIN     |
| dGPU     | FP32 BIN BATCHED_BLOB FP16 INT8 |
| iGPU     | FP32 BIN BATCHED_BLOB FP16 INT8 |

When application use Auto-device to run FP16 IR on system with CPU, dGPU and iGPU, Auto-device will offload this workload to dGPU.

When application use Auto-device to run FP16 IR on system with CPU and iGPU, Auto-device will offload this workload to iGPU.

When application use Auto-device to run WINOGRAD-enabled IR on system with CPU, dGPU and iGPU, Auto-device will offload this workload to CPU.

In any case, when loading the network to dGPU or iGPU fails, the networks falls back to CPU as the last choice.

### Limit Auto Target Devices Logic

According to the Auto-device selection logic from the previous section, 
the most suitable device from available devices to load mode as follows:

@snippet snippets/AUTO2.cpp part2

Another way to load mode to device from limited choice of devices is with Auto-device:

@snippet snippets/AUTO3.cpp part3

## Configuring the Individual Devices and Creating the Auto-Device on Top

As described in the first section, configure each individual device as usual and then just create the "AUTO" device on top:

@snippet snippets/AUTO4.cpp part4

Alternatively, you can combine all the individual device settings into single config and load it, 
allowing the Auto-device plugin to parse and apply it to the right devices. See the code example here:

@snippet snippets/AUTO5.cpp part5

## Using the Auto-Device with OpenVINO Samples and Benchmark App

Note that every OpenVINO sample that supports "-d" (which stands for "device") command-line option transparently accepts the Auto-device. 
The Benchmark Application is the best example of the optimal usage of the Auto-device. 
You do not need to set the number of requests and CPU threads, as the application provides optimal out-of-the-box performance. 
Below is the example command-line to evaluate AUTO performance with that:

```sh
./benchmark_app –d AUTO –m <model> -i <input> -niter 1000
```
You can also use the auto-device with limit device choice:

```sh
./benchmark_app –d AUTO:CPU,GPU –m <model> -i <input> -niter 1000
```
Note that the default CPU stream is 1 if using “-d AUTO”.

Note that you can use the FP16 IR to work with auto-device.
Also note that no demos are (yet) fully optimized for the auto-device, by means of selecting the most suitable device, 
using the GPU streams/throttling, and so on.
