# Model Caching Overview {#openvino_docs_IE_DG_Model_caching_overview}

## Introduction

As described in [Inference Engine Introduction](inference_engine_intro.md), common application flow consists of the following steps:

1. **Create Inference Engine Core object**

2. **Read the Intermediate Representation** - Read an Intermediate Representation file into an object of the `InferenceEngine::CNNNetwork`

3. **Prepare inputs and outputs**

4. **Set configuration** Pass device-specific loading configurations to the device

5. **Compile and Load Network to device** - Use the `InferenceEngine::Core::LoadNetwork()` method with specific device

6. **Set input data**

7. **Execute**

Step #5 can potentially perform several time-consuming device-specific optimizations and network compilations,
and such delays can lead to bad user experience on application startup. To avoid this, some devices offer
Import/Export network capability, and it is possible to either use [Compile tool](../../inference-engine/tools/compile_tool/README.md)
or enable model caching to export compiled network automatically. Reusing cached networks can significantly reduce load network time.


## Set "CACHE_DIR" config option to enable model caching

To enable model caching, the application must specify the folder where to store cached blobs. It can be done like this


@snippet snippets/InferenceEngine_Caching0.cpp part0

With this code, if device supports Import/Export network capability, cached blob is automatically created inside the `myCacheFolder` folder
CACHE_DIR config is set to the Core object. If device does not support Import/Export capability, cache is just not created and no error is thrown

Depending on your device, total time for loading network on application startup can be significantly reduced.
Please also note that very first LoadNetwork (when cache is not yet created) takes slightly longer time to 'export' compiled blob into a cache file
![caching_enabled]

## Even faster: use LoadNetwork(modelPath)

In some cases, applications do not need to customize inputs and outputs every time. Such applications always
call `cnnNet = ie.ReadNetwork(...)`, then `ie.LoadNetwork(cnnNet, ..)` and it can be further optimized.
For such cases, more convenient API to load network in one call is introduced in the 2021.4 release.

@snippet snippets/InferenceEngine_Caching1.cpp part1

With enabled model caching, total load time is even smaller - in case that ReadNetwork is optimized as well

@snippet snippets/InferenceEngine_Caching2.cpp part2

![caching_times]


## Advanced examples

Not every device supports network import/export capability, enabling of caching for such devices do not have any effect.
To check in advance if a particular device supports model caching, your application can use the following code:

@snippet snippets/InferenceEngine_Caching3.cpp part3


[caching_enabled]: ../img/caching_enabled.png
[caching_times]: ../img/caching_times.png
