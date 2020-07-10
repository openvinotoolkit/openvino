Using GPU Kernels Tuning  {#openvino_docs_IE_DG_GPU_Kernels_Tuning}
======================

GPU Kernels Tuning allows you to tune models, so the heavy computational layers* are configured to fit better into
hardware, which the tuning was done on. It is required to achieve best performance on GPU.
DLDT releases includes pretuned data (`cache.json` - it is located in IE's binaries folder) for current state of the art models. It is highly recommended to do the
tuning for new kind of models, hardwares or drivers.

* Currently only convolution and fully connected layers undergo tuning process. It means that, the performance boost depends on the amount of that layers in the model.

## Tuned data

GPU tuning data is saved in JSON format.
File's content is composed of 2 types of attributes and 1 type of value:
1. Execution units number - this attribute splits the content into different EU sections.
2. Hash - hashed tuned kernel data.
Key: Array with kernel name and kernel's mode index.

## Usage

---

You can activate Kernels Tuning process by setting <code>KEY_TUNING_MODE</code> flag to <code>TUNING_CREATE</code> and <code>KEY_TUNING_FILE</code> to <code><"filename"></code> in a configuration map that is
passed to the plugin while loading a network.
This configuration modifies the behavior of the <code>ExecutableNetwork</code> object. Instead of standard network compilation, it will run the tuning process.
Please keep in mind that the tuning can be very time consuming. The bigger the network, the longer it will take.
File with tuned data is the result of this step*.

* If the filenames passed to <code>KEY_TUNING_FILE</code> points to existing tuned data and user is tuning new model, then this file will be extended by new data. This allows users to extened existing `cache.json` provided in DLDT release package. 

---

You can activate the inference with tuned data by setting <code>KEY_TUNING_MODE</code> flag to <code>TUNING_USE_EXISTING</code> and
<code>KEY_TUNING_FILE</code> flag to <code><"filename"></code>. 

GPU backend will process the content of the file during network compilation to configure the OpenCL kernels for the best performance.
