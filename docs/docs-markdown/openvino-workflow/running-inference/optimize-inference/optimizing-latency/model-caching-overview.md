---
sidebar_position: 1
sidebar_label: "Model Caching Overview"
description: "Enabling model caching to export compiled model automatically and reusing it can significantly reduce duration of model compilation on application startup."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Model Caching Overview

As described in [Integrate OpenVINO™ with Your Application](../../index.md),
a common workflow consists of the following steps:

1. **Create a Core object**:  
   First step to manage available devices and read model objects
2. **Read the Intermediate Representation**:  
   Read an Intermediate Representation file into the [ov::Model](https://docs.openvino.ai/2026/api/c_cpp_api/classov_1_1_model.html) object
3. **Prepare inputs and outputs**:  
   If needed, manipulate precision, memory layout, size or color format
4. **Set configuration**:  
   Add device-specific loading configurations to the device
5. **Compile and Load Network to device**:  
   Use the [ov::Core::compile\_model()](https://docs.openvino.ai/2026/api/c_cpp_api/classov_1_1_core.html) method with a specific device
6. **Set input data**:  
   Specify input tensor
7. **Execute**:  
   Carry out inference and process results

Step 5 can potentially perform several time-consuming device-specific optimizations and network compilations.
To reduce the resulting delays at application startup, you can use Model Caching. It exports the compiled model
automatically and reuses it to significantly reduce the model compilation time.

:::info[Important]
Not all devices support import/export of models. They will perform normally but will not
enable the compilation stage speed-up.
:::

## Set configuration options

Use the `device_name` option to specify the inference device.  
Specify `cache_dir` to enable model caching.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  from utils import get_path_to_model, get_temp_dir
  import openvino as ov

  import openvino.properties as props

  # For example: "CPU", "GPU", "NPU".
  device_name = 'CPU'
  model_path = get_path_to_model()
  path_to_cache_dir = get_temp_dir()

  core = ov.Core()
  core.set_property({props.cache_dir: path_to_cache_dir})
  model = core.read_model(model=model_path)
  compiled_model = core.compile_model(model=model, device_name=device_name)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  void part0() {
      std::string modelPath = "/tmp/myModel.xml";
      std::string device = "GPU";                             // For example: "CPU", "GPU", "NPU".
      ov::AnyMap config;
  ov::Core core;                                              // Step 1: create ov::Core object
  core.set_property(ov::cache_dir("/path/to/cache/dir"));     // Step 1b: Enable caching
  auto model = core.read_model(modelPath);                    // Step 2: Read Model
  //...                                                       // Step 3: Prepare inputs/outputs
  //...                                                       // Step 4: Set device configuration
  auto compiled = core.compile_model(model, device, config);  // Step 5: LoadNetwork
  ```
  </TabItem>
</Tabs>

If the specified device supports import/export of models,
a cached blob file: `.cl_cache` (GPU) or  `.blob` (CPU) is automatically
created inside the `/path/to/cache/dir` folder.
If the device does not support import/export of models, the cache is not
created and no error is thrown.

Note that the first `compile_model` operation takes slightly more time,
as the cache needs to be created - the compiled blob is saved into a file:

![](caching_enabled.svg)

## Use optimized methods

Applications do not always require an initial customization of inputs and
outputs, as they can call `model = core.read_model(...)`, then `core.compile_model(model, ..)`,
which can be further optimized. Thus, the model can be compiled conveniently in a single call,
skipping the read step:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  core = ov.Core()
  compiled_model = core.compile_model(model=model_path, device_name=device_name)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  ov::Core core;                                                  // Step 1: create ov::Core object
  auto compiled = core.compile_model(modelPath, device, config);  // Step 2: Compile model by file path
  ```
  </TabItem>
</Tabs>

The total load time is even shorter, when model caching is enabled and `read_model` is optimized as well.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  core = ov.Core()
  core.set_property({props.cache_dir: path_to_cache_dir})
  compiled_model = core.compile_model(model=model_path, device_name=device_name)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  ov::Core core;                                                  // Step 1: create ov::Core object
  core.set_property(ov::cache_dir("/path/to/cache/dir"));         // Step 1b: Enable caching
  auto compiled = core.compile_model(modelPath, device, config);  // Step 2: Compile model by file path
  ```
  </TabItem>
</Tabs>

![](caching_times.svg)

## Advanced Examples

Enabling model caching has no effect when the specified device does not support
import/export of models. To check in advance if a particular device supports
model caching, use the following code in your application:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  import openvino.properties.device as device

  # Find 'EXPORT_IMPORT' capability in supported capabilities
  caching_supported = 'EXPORT_IMPORT' in core.get_property(device_name, device.capabilities)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  // Get list of supported device capabilities
  std::vector<std::string> caps = core.get_property(deviceName, ov::device::capabilities);

  // Find 'EXPORT_IMPORT' capability in supported capabilities
  bool cachingSupported = std::find(caps.begin(), caps.end(), ov::device::capability::EXPORT_IMPORT) != caps.end();
  ```
  </TabItem>
</Tabs>

## Set `CacheMode` property to `OPTIMIZE_SIZE` to enable weightless caching

Weightless caching is a feature that allows you to create a cache file which doesn't contain the weights of the model. Instead, the weights are loaded from the original model file. This helps to reduce the size of the cache file.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  core = ov.Core()
  if "GPU" in core.available_devices:
      core.set_property({props.cache_dir: path_to_cache_dir})
      config_cache = {}
      config_cache["CACHE_MODE"] = "OPTIMIZE_SIZE"
      # Note: model path needs to point to the *.xml file, not *.bin when using the IR model format.
      compiled_model = core.compile_model(model=model_path, device_name='GPU', config=config_cache)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  // Note: model path needs to point to the *.xml file, not *.bin when using the IR model format.
  auto compiled = core.compile_model(modelPath,
                                     device,
                                     ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
  ```
  </TabItem>
</Tabs>

:::info[Important]
Currently, this property is supported only by the GPU Plugin and IR model format.
:::

:::info[Important]
Some weights which undergo transformations during model compilation may not be eligible for weightless caching. In such cases, the cache file will contain these weights while still using the weightless caching mechanism for the rest. The feature supports some of the common transformations and replicates them after loading the model from the cache.
:::

## Enable cache encryption

If model caching is enabled in the CPU Plugin, set the "cache\_encryption\_callbacks"
config option to encrypt the model while caching it and decrypt it when
loading it from the cache. Currently, this property can be set only in `compile_model`.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  import base64

  def encrypt_base64(src: bytes):
      return base64.b64encode(src)

  def decrypt_base64(src: bytes):
      return base64.b64decode(src)

  core = ov.Core()
  core.set_property({props.cache_dir: path_to_cache_dir})
  config_cache = {}
  config_cache["CACHE_ENCRYPTION_CALLBACKS"] = [encrypt_base64, decrypt_base64]
  model = core.read_model(model=model_path)
  compiled_model = core.compile_model(model=model, device_name=device_name, config=config_cache)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  ov::AnyMap config;
  ov::EncryptionCallbacks encryption_callbacks;
  static const char codec_key[] = {0x30, 0x60, 0x70, 0x02, 0x04, 0x08, 0x3F, 0x6F, 0x72, 0x74, 0x78, 0x7F};
  auto codec_xor = [&](const std::string& source_str) {
      auto key_size = sizeof(codec_key);
      int key_idx = 0;
      std::string dst_str = source_str;
      for (char& c : dst_str) {
          c ^= codec_key[key_idx % key_size];
          key_idx++;
      }
      return dst_str;
  };
  encryption_callbacks.encrypt = codec_xor;
  encryption_callbacks.decrypt = codec_xor;
  config.insert(ov::cache_encryption_callbacks(encryption_callbacks));  // Step 4: Set device configuration
  auto compiled = core.compile_model(model, device, config);            // Step 5: LoadNetwork
  ```
  </TabItem>
</Tabs>

If model caching is enabled in the GPU Plugin, the model topology is encrypted when saved to the cache and decrypted when loaded from the cache if the `CacheMode` property is set to `OPTIMIZE_SIZE`. The weights are encrypted only when `CacheMode` is set to `OPTIMIZE_SPEED`. Weight encryption requires extra disk space equal to the size of the weights and may introduce runtime memory overhead for decryption, depending on the encryption algorithm.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  import base64

  def encrypt_base64(src: bytes):
      return base64.b64encode(src)

  def decrypt_base64(src: bytes):
      return base64.b64decode(src)

  core = ov.Core()
  if any("GPU" in device for device in core.available_devices):
      core.set_property({props.cache_dir: path_to_cache_dir})
      config_cache = {}
      config_cache["CACHE_ENCRYPTION_CALLBACKS"] = [encrypt_base64, decrypt_base64]
      config_cache["CACHE_MODE"] = "OPTIMIZE_SIZE"
      compiled_model = core.compile_model(model=model_path, device_name='GPU', config=config_cache)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  static const char codec_key[] = {0x30, 0x60, 0x70, 0x02, 0x04, 0x08, 0x3F, 0x6F, 0x72, 0x74, 0x78, 0x7F};
  auto codec_xor = [&](const std::string& source_str) {
      auto key_size = sizeof(codec_key);
      int key_idx = 0;
      std::string dst_str = source_str;
      for (char& c : dst_str) {
          c ^= codec_key[key_idx % key_size];
          key_idx++;
      }
      return dst_str;
  };
  auto compiled = core.compile_model(modelPath,
                                     device,
                                     ov::cache_encryption_callbacks(ov::EncryptionCallbacks{codec_xor, codec_xor}),
                                     ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));  // Step 5: Compile model
  ```
  </TabItem>
</Tabs>

:::info[Important]
Currently, encryption is supported only by the CPU and GPU plugins. Enabling this
feature for other HW plugins will not encrypt/decrypt model topology in the
cache and will not affect performance.
:::
