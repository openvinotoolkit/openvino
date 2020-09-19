[DEPRECATED] Migration from Inference Engine Plugin API to Core API {#openvino_docs_IE_DG_Migration_CoreAPI}
===============================

For 2019 R2 Release, the new Inference Engine Core API is introduced. This guide is updated to reflect the new API approach. The Inference Engine Plugin API is still supported, but is going to be deprecated in future releases.

This section provides common steps to migrate your application written using the Inference Engine Plugin API (`InferenceEngine::InferencePlugin`) to the Inference Engine Core API (`InferenceEngine::Core`). 

To learn how to write a new application using the Inference Engine, refer to [Integrate the Inference Engine Request API with Your Application](Integrate_with_customer_application_new_API.md) and [Inference Engine Samples Overview](Samples_Overview.md).

## Inference Engine Core Class

The Inference Engine Core class is implemented on top existing Inference Engine Plugin API and handles plugins internally. 
The main responsibility of the `InferenceEngine::Core` class is to hide plugin specifics inside and provide a new layer of abstraction that works with devices (`InferenceEngine::Core::GetAvailableDevices`). Almost all methods of this class accept `deviceName` as an additional parameter that denotes an actual device you are working with. Plugins are listed in the `plugins.xml` file, which is loaded during constructing `InferenceEngine::Core` objects:

```bash
<ie>
    <plugins>
        <plugin name="CPU" location="libMKLDNNPlugin.so">
        </plugin>
        ...
</ie>
```

## Migration Steps

Common migration process includes the following steps:

1. Migrate from the `InferenceEngine::InferencePlugin` initialization:
```cpp
InferenceEngine::InferencePlugin plugin = InferenceEngine::PluginDispatcher({ FLAGS_pp }).getPluginByDevice(FLAGS_d);
```
to the `InferenceEngine::Core` class initialization:
```cpp
InferenceEngine::Core core;
```

2. Instead of using `InferenceEngine::CNNNetReader` to read IR:
```cpp
CNNNetReader network_reader;
network_reader.ReadNetwork(fileNameToString(input_model));
network_reader.ReadWeights(fileNameToString(input_model).substr(0, input_model.size() - 4) + ".bin");
CNNNetwork network = network_reader.getNetwork();
```
read networks using the Core class:
```cpp
CNNNetwork network = core.ReadNetwork(input_model);
```
The Core class also allows reading models from the ONNX format (more information is [here](./ONNX_Support.md)):
```cpp
CNNNetwork network = core.ReadNetwork("model.onnx");
```

3. Instead of adding CPU device extensions to the plugin:
```cpp
plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
```
add extensions to CPU device using the Core class:
```cpp
core.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
```

4. Instead of setting configuration keys to a particular plugin, set (key, value) pairs via `InferenceEngine::Core::SetConfig`
```cpp
core.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "GPU");
```
> **NOTE**: If `deviceName` is omitted as the last argument, configuration is set for all Inference Engine devices.

5. Migrate from loading the network to a particular plugin:
```cpp
auto execNetwork = plugin.LoadNetwork(network, { });
```
to `InferenceEngine::Core::LoadNetwork` to a particular device:
```cpp
auto execNetwork = core.LoadNetwork(network, deviceName, { });
```

After you have an instance of `InferenceEngine::ExecutableNetwork`, all other steps are as usual.
