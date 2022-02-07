# Model Caching Overview {#openvino_docs_IE_DG_Model_caching_overview}

## Introduction (C++)

@sphinxdirective
.. raw:: html

    <div id="switcher-cpp" class="switcher-anchor">C++</div>
@endsphinxdirective

As described in the [Inference Engine Developer Guide](Deep_Learning_Inference_Engine_DevGuide.md), a common application flow consists of the following steps:

1. **Create an Inference Engine Core object**: First step to manage available devices and read network objects

2. **Read the Intermediate Representation**: Read an Intermediate Representation file into an object of the `InferenceEngine::CNNNetwork`

3. **Prepare inputs and outputs**: If needed, manipulate precision, memory layout, size or color format

4. **Set configuration**: Pass device-specific loading configurations to the device

5. **Compile and Load Network to device**: Use the `InferenceEngine::Core::LoadNetwork()` method with a specific device

6. **Set input data**: Specify input blob

7. **Execute**: Carry out inference and process results

Step 5 can potentially perform several time-consuming device-specific optimizations and network compilations,
and such delays can lead to a bad user experience on application startup. To avoid this, some devices offer
import/export network capability, and it is possible to either use the [Compile tool](../../tools/compile_tool/README.md)
or enable model caching to export compiled network automatically. Reusing cached networks can significantly reduce load network time.

### Set "CACHE_DIR" config option to enable model caching

To enable model caching, the application must specify a folder to store cached blobs, which is done like this:

@snippet snippets/InferenceEngine_Caching0.cpp part0

With this code, if the device specified by `LoadNetwork` supports import/export network capability, a cached blob is automatically created inside the `myCacheFolder` folder.
CACHE_DIR config is set to the Core object. If the device does not support import/export capability, cache is not created and no error is thrown.

Depending on your device, total time for loading network on application startup can be significantly reduced.
Also note that the very first LoadNetwork (when cache is not yet created) takes slightly longer time to "export" the compiled blob into a cache file:

![caching_enabled]

### Even faster: use LoadNetwork(modelPath)

In some cases, applications do not need to customize inputs and outputs every time. Such an application always
call `cnnNet = ie.ReadNetwork(...)`, then `ie.LoadNetwork(cnnNet, ..)` and it can be further optimized.
For these cases, the 2021.4 release introduces a more convenient API to load the network in a single call, skipping the export step:

@snippet snippets/InferenceEngine_Caching1.cpp part1

With model caching enabled, total load time is even smaller, if ReadNetwork is optimized as well.

@snippet snippets/InferenceEngine_Caching2.cpp part2

![caching_times]

### Advanced Examples

Not every device supports network import/export capability. For those that don't, enabling caching has no effect.
To check in advance if a particular device supports model caching, your application can use the following code:

@snippet snippets/InferenceEngine_Caching3.cpp part3

## Introduction (Python)

@sphinxdirective
.. raw:: html

    <div id="switcher-python" class="switcher-anchor">Python</div>
@endsphinxdirective

As described in Inference Engine Developer Guide, a common application flow consists of the following steps:

1. **Create an Inference Engine Core Object**
2. **Read the Intermediate Representation** - Read an Intermediate Representation file into an object of the [ie_api.IENetwork](api/ie_python_api/_autosummary/openvino.inference_engine.IENetwork.html)
3. **Prepare inputs and outputs**
4. **Set configuration** - Pass device-specific loading configurations to the device
5. **Compile and Load Network to device** - Use the `IECore.load_network()` method and specify the target device
6. **Set input data**
7. **Execute the model** - Run inference

Step #5 can potentially perform several time-consuming device-specific optimizations and network compilations, and such delays can lead to bad user experience on application startup. To avoid this, some devices offer Import/Export network capability, and it is possible to either use the [Compile tool](../../tools/compile_tool/README.md) or enable model caching to export the compiled network automatically. Reusing cached networks can significantly reduce load network time.

### Set the “CACHE_DIR” config option to enable model caching

To enable model caching, the application must specify the folder where to store cached blobs. It can be done using [IECore.set_config](api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.set_config).

``` python
from openvino.inference_engine import IECore

ie = IECore()
ie.set_config(config={"CACHE_DIR": path_to_cache}, device_name=device)
net = ie.read_network(model=path_to_xml_file)
exec_net = ie.load_network(network=net, device_name=device)
```

With this code, if a device supports the Import/Export network capability, a cached blob is automatically created inside the path_to_cache directory `CACHE_DIR` config is set to the Core object. If device does not support Import/Export capability, cache is just not created and no error is thrown

Depending on your device, total time for loading network on application startup can be significantly reduced. Please also note that very first [IECore.load_network](api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.load_network) (when the cache is not yet created) takes slightly longer time to ‘export’ the compiled blob into a cache file.

![caching_enabled]


### Even Faster: Use IECore.load_network(path_to_xml_file)

In some cases, applications do not need to customize inputs and outputs every time. These applications always call [IECore.read_network](api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.read_network), then `IECore.load_network(model=path_to_xml_file)` and may be further optimized. For such cases, it's more convenient to load the network in a single call to `ie.load_network()`
A model can be loaded directly to the device, with model caching enabled:

``` python
from openvino.inference_engine import IECore

ie = IECore()
ie.set_config(config={"CACHE_DIR" : path_to_cache}, device_name=device)
ie.load_network(network=path_to_xml_file, device_name=device)
```

![caching_times]

### Advanced Examples

Not every device supports network import/export capability, enabling of caching for such devices does not have any effect. To check in advance if a particular device supports model caching, your application can use the following code:

```python
all_metrics = ie.get_metric(device_name=device, metric_name="SUPPORTED_METRICS")
# Find the 'IMPORT_EXPORT_SUPPORT' metric in supported metrics
allows_caching = "IMPORT_EXPORT_SUPPORT" in all_metrics
```

> **NOTE**: The GPU plugin does not have the IMPORT_EXPORT_SUPPORT capability, and does not support model caching yet. However, the GPU plugin supports caching kernels (see the [GPU plugin documentation](supported_plugins/GPU.md)). Kernel caching for the GPU plugin can be accessed the same way as model caching: by setting the `CACHE_DIR` configuration key to a folder where the cache should be stored.


[caching_enabled]: ../img/caching_enabled.png
[caching_times]: ../img/caching_times.png
