# Model Caching Overview {#openvino_docs_IE_DG_Model_caching_overview}

## Introduction

As described in [Inference Engine Introduction](inference_engine_intro.md) the common application flow contains the following steps:

1. **Create Inference Engine Core object**

2. **Read the Intermediate Representation** - Read an Intermediate Representation file into an object of the `InferenceEngine::CNNNetwork`

3. **Prepare inputs and outputs**

4. **Set configuration** Pass device specific loading configurations to the device

5. **Compile and Load Network to device** - Use the `InferenceEngine::Core::LoadNetwork()` method with specific device

6. **Set input data**

7. **Execute**

Step #5 can potentially perform several time-consuming device-specific optimizations and network compilations,
and such delays can lead to bad user experience on application startup. To avoid this, some devices offer
Import/Export network capability, and it is possible to either use [Compile tool](../../inference-engine/tools/compile_tool/README.md)
or enable model caching to export compiled network automatically. Reusing cached networks can significantly reduce load network time


## Set "CACHE_DIR" config option to enable model caching

To enable model caching application shall specify folder where to store cached blobs. It can be done like this

```cpp
InferenceEngine::Core ie;                                 // Step 1: create Inference engine object
ie.SetConfig({{CONFIG_KEY(CACHE_DIR), "myCacheFolder"}}); // Step 1b: Enable caching
auto cnnNet = ie.ReadNetwork(<modelFile>);                // Step 2: ReadNetwork
...                                                       // Step 3: Prepare inputs/outputs
...                                                       // Step 4: Set device configuration
ie.LoadNetwork(cnnNet, <device>, <deviceConfig>);         // Step 5: LoadNetwork
```
With this code, if device supports Import/Export network capability, cached blob will be automatically created inside `myCacheFolder` folder
CACHE_DIR config is set to Core object, so if device doesn't support Import/Export capability, cache will just not be created and no error will be thrown

Depending on your device, total time for loading network on application startup can be significantly reduced.
Please also note that very first LoadNetwork (when cache is not yet created) will take slightly longer time to 'export' compiled blob into a cache file
![caching_enabled]

## Even faster: use LoadNetwork(\<modelName\>)

In some cases applications do not need to customize inputs and outputs every time. In this case such applications always
call `cnnNet = ie.ReadNetwork(...)`, then `ie.LoadNetwork(cnnNet, ..)` and it can be further optimized.
For such cases more convenient API to load network in one call is introduced in release 2021.4

```cpp
InferenceEngine::Core ie;                                 // Step 1: create Inference engine object
ie.LoadNetwork(<modelPath>, <device>, <deviceConfig>);    // Step 2: LoadNetwork by model file path
```

And with enabled model caching total load time will be even smaller - in case that ReadNetwork will be optimized as well
```cpp
InferenceEngine::Core ie;                                 // Step 1: create Inference engine object
ie.SetConfig({{CONFIG_KEY(CACHE_DIR), "myCacheFolder"}}); // Step 1b: Enable caching
ie.LoadNetwork(<modelPath>, <device>, <deviceConfig>);    // Step 2: LoadNetwork by model file path
```
![caching_times]


## Advanced examples

Not any device supports network import/export capability, enabling of caching for such devices don't have any effect.
To check in advance if particular device supports model caching, application can use the following code:
```cpp
// Get list of supported metrics
std::vector<std::string> keys = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_METRICS));

// Find 'IMPORT_EXPORT_SUPPORT' metric in supported metrics
auto it = std::find(keys.begin(), keys.end(), METRIC_KEY(IMPORT_EXPORT_SUPPORT));

// If metric 'IMPORT_EXPORT_SUPPORT' exists, check it's value
bool cachingSupported = (it != keys.end()) && ie.GetMetric(deviceName, METRIC_KEY(IMPORT_EXPORT_SUPPORT));
```

[caching_enabled]: ../img/caching_enabled.png
[caching_times]: ../img/caching_times.png
