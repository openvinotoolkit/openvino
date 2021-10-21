# Model Caching Overview {#openvino_docs_IE_DG_Model_caching_overview}

## Introduction (C++)

@sphinxdirective
.. raw:: html

    <div id="switcher-cpp" class="switcher-anchor">C++</div>
@endsphinxdirective

As described in the [Inference Engine Developer Guide](Deep_Learning_Inference_Engine_DevGuide.md), common application flow consists of the following steps:

1. **Create Inference Engine Core object**: First step to manage available devices and read network objects

2. **Read the Intermediate Representation**: Read an Intermediate Representation file into an object of the `InferenceEngine::CNNNetwork`

3. **Prepare inputs and outputs**: If needed, manipulate precision, memory layout, size or color format

4. **Set configuration**: Pass device-specific loading configurations to the device

5. **Compile and Load Network to device**: Use the `InferenceEngine::Core::LoadNetwork()` method with a specific device

6. **Set input data**: Specify input blob

7. **Execute**: Carry out inference and process results

Step 5 can potentially perform several time-consuming device-specific optimizations and network compilations,
and such delays can lead to a bad user experience on application startup. To avoid this, some devices offer
import/export network capability, and it is possible to either use the [Compile tool](../../inference-engine/tools/compile_tool/README.md)
or enable model caching to export compiled network automatically. Reusing cached networks can significantly reduce load network time.

## Set "CACHE_DIR" config option to enable model caching

To enable model caching, the application must specify a folder to store cached blobs, which is done like this:

@snippet snippets/InferenceEngine_Caching0.cpp part0

With this code, if the device specified by `LoadNetwork` supports import/export network capability, a cached blob is automatically created inside the `myCacheFolder` folder.
CACHE_DIR config is set to the Core object. If the device does not support import/export capability, cache is not created and no error is thrown.

Depending on your device, total time for loading network on application startup can be significantly reduced.
Also note that the very first LoadNetwork (when cache is not yet created) takes slightly longer time to "export" the compiled blob into a cache file:

![caching_enabled]

## Even faster: use LoadNetwork(modelPath)

In some cases, applications do not need to customize inputs and outputs every time. Such an application always
call `cnnNet = ie.ReadNetwork(...)`, then `ie.LoadNetwork(cnnNet, ..)` and it can be further optimized.
For these cases, the 2021.4 release introduces a more convenient API to load the network in a single call, skipping the export step:

@snippet snippets/InferenceEngine_Caching1.cpp part1

With model caching enabled, total load time is even smaller, if ReadNetwork is optimized as well.

@snippet snippets/InferenceEngine_Caching2.cpp part2

![caching_times]


## Advanced examples

Not every device supports network import/export capability. For those that don't, enabling caching has no effect.
To check in advance if a particular device supports model caching, your application can use the following code:

@snippet snippets/InferenceEngine_Caching3.cpp part3

[caching_enabled]: ../img/caching_enabled.png
[caching_times]: ../img/caching_times.png
