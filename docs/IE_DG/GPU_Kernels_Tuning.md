Using GPU Kernels Tuning {#openvino_docs_IE_DG_GPU_Kernels_Tuning}
======================

GPU Kernels Tuning allows you to tune models, so the heavy computational layers are configured to fit better into
hardware, which the tuning was done on. It is required to achieve best performance on GPU.
> **NOTE** Currently only convolution and fully connected layers undergo tuning process. It means that the performance boost depends on the amount of that layers in the model.

OpenVINO™ releases include the `<INSTALL_DIR>/inference_engine/bin/intel64/Release/cache.json` file with pretuned data for current state of the art models. It is highly recommended to do the
tuning for new kind of models, hardwares or drivers.

## Tuned data

GPU tuning data is saved in JSON format.
File's content is composed of 2 types of attributes and 1 type of value:
1. Execution units number - this attribute splits the content into different EU sections.
2. Hash - hashed tuned kernel data.
Key: Array with kernel name and kernel's mode index.

## Usage

---

You can activate Kernels Tuning process by setting `KEY_TUNING_MODE` flag to `TUNING_CREATE` and `KEY_TUNING_FILE` to `<"filename">` in a configuration map that is
passed to the plugin while loading a network.
This configuration modifies the behavior of the `ExecutableNetwork` object. Instead of standard network compilation, it will run the tuning process.
Please keep in mind that the tuning can be very time consuming. The bigger the network, the longer it will take.
File with tuned data is the result of this step.

> **NOTE** If a filename passed to `KEY_TUNING_FILE` points to existing tuned data and you are tuning a new model, then this file will be extended by new data. This allows you to extend existing `cache.json` provided in the OpenVINO™ release package. 

The example below shows how to set and use the key files:
```cpp
Core ie;          
  ie.SetConfig({{ CONFIG_KEY(TUNING_MODE), CONFIG_VALUE(TUNING_CREATE) }}, "GPU");
  ie.SetConfig({{ CONFIG_KEY(TUNING_FILE), "/path/to/tuning/file.json" }}, "GPU");
  // Further LoadNetwork calls will use the specified tuning parameters
```
---

You can activate the inference with tuned data by setting `KEY_TUNING_MODE` flag to `TUNING_USE_EXISTING` and
`KEY_TUNING_FILE` flag to `<"filename">`. 

GPU backend will process the content of the file during network compilation to configure the OpenCL kernels for the best performance.
