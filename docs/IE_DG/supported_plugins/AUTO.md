# Auto-Device Plugin {#openvino_docs_IE_DG_supported_plugins_AUTO}

## Auto-Device Plugin Execution

Auto-Device is a new special "virtual" or "proxy" device in the OpenVINO. 

User can use "AUTO" as the device name, to delegate an actual accelerator selection to the OpenVINO. 
With the 2021.4 release, the Auto-Device internally recognizes and selects from CPU, 
integrated GPU and discrete Intel GPUs (when available) depending on the device capabilities and the characteristic of CNN models, 
for example, precisions. Then Auto-Device will assign inference requests to selected device.

From the application point of view, this is just another device that handles all accelerators in full system. 

With the 2021.4 release, the "setup" of Auto-device can be described in three major steps:
* First is configuration of each device as usual (e.g. via conventional SetConfig method)
* Second is loading of a network to the AUTO-Device plugin. This is the only change that you need in your application
* Finally, just like with any other ExecutableNetwork (resulted from LoadNetwork) you just create as many requests as needed to saturate the devices. 
These steps are covered below in details


## Defining and Configuring the Auto-Device plugin
Following the OpenVINO notions of “devices”, the Auto-Device has a “AUTO” name. The only configuration option for AUTO-Device is a limited device list:

| Parameter name     | Parameter values      | Default            |             Description                                                      |
| :---               | :---                  | :---               |:-----------------------------------------------------------------------------|
| "AUTO_DEVICE_LIST" | comma-separated device names <span style="color:red">with no spaces</span>| N/A | Device candidate list to be selected    |

You can use name of the configuration directly as a string, or use `IE::KEY_AUTO_DEVICE_LIST` from the `ie_plugin_config.hpp`, 
which defines the same string.

Basically, there are two ways to use Auto-Device, directly indicate device by “AUTO” or empty string:

@snippet snippets/AUTO0.cpp part0

Also, you can use Auto-Device configuration to limit the device candidate list to be selected:

@snippet snippets/AUTO1.cpp part1

Besides, Auto-Device supports query device optimization capabilities in metric; 

| Parameter name                 | Parameter values         |
| :---                           | :---                     |
| "OPTIMIZATION_CAPABILITIES"    | Auto-Device capabilities |

## Enumerating Available Devices and Limit Auto Target Devices

Inference Engine now features a dedicated API to enumerate devices and their capabilities. 
See [Hello Query Device C++ Sample](../../../inference-engine/samples/hello_query_device/README.md).
This is example output from the sample (truncated to the devices' names only):

```sh
./hello_query_device
Available devices: 
    Device: CPU
...
    Device: GPU.0
...
    Device: GPU.1
```
Auto-device will select the most suitable device from available or limit devices to load, 
and simple programmatic way to load mode to device from available devices with the auto-device is as follows:

@snippet snippets/AUTO2.cpp part2

Another way to load mode to device from limit device choice with Auto-device::

@snippet snippets/AUTO3.cpp part3

## Configuring the Individual Devices and Creating the Auto-Device On Top

As discussed in the first section, you shall configure each individual device as usual and then just create the "AUTO" device on top:

@snippet snippets/AUTO4.cpp part4

Alternatively, you can combine all the individual device settings into single config and load that, 
allowing the Auto-device plugin to parse and apply that to the right devices. See code example here:

@snippet snippets/AUTO5.cpp part5

## Using the Auto-Device with OpenVINO Samples and Benchmark App

Notice that every OpenVINO sample that supports "-d" (which stands for "device") command-line option transparently accepts the auto-device. 
The Benchmark Application is the best reference to the optimal usage of the auto-device. 
You don't need to setup number of requests, CPU threads as the application provides optimal out of the box performance. 
Below is example command-line to evaluate AUTO performance with that:

```sh
./benchmark_app –d AUTO –m <model> -i <input> -niter 1000
```
You can also use the auto-device with limit device choice:

```sh
./benchmark_app –d AUTO:CPU,GPU –m <model> -i <input> -niter 1000
```
Note that default CPU stream is 1 if using “-d AUTO”.

Notice that you can use the FP16 IR to work with auto-device. Also notice that no demos are (yet) fully optimized for the auto-device, by means of selecting the most suitable device , using the GPU streams/throttling, and so on.