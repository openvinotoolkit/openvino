# Using Dynamic Batching {#openvino_docs_IE_DG_DynamicBatching}

## Using Dynamic Batching (C++)

@sphinxdirective
.. raw:: html

    <div id="switcher-cpp" class="switcher-anchor">C++</div>
@endsphinxdirective

The Dynamic Batching feature allows you to dynamically change batch size for inference calls
within a preset batch size limit. This feature might be useful when batch size is unknown beforehand and using an extra-large batch size is undesirable or impossible due to resource limitations. For example, applying face detection and then mood labeling to a video, you won't know in advance how many frames will contain a face when you pass inferencing results to a secondary model.


You can activate Dynamic Batching by setting `KEY_DYN_BATCH_ENABLED` flag to `YES` in a configuration map that is
passed to the plugin while loading a network.
This configuration creates an `ExecutableNetwork` object that will allow setting batch size
dynamically in all of its infer requests using `SetBatch()` method.
The batch size that was set in the passed `CNNNetwork` object will be used as a maximum batch size limit.

Here is a code example:

@snippet snippets/DynamicBatching.cpp part0


### Limitations

Currently, there are certain limitations for the use of Dynamic Batching exist:

* Use Dynamic Batching with CPU and GPU plugins only.
* Use Dynamic Batching on topologies that consist of certain layers only:
   * Convolution
   * Deconvolution
   * Activation
   * LRN
   * Pooling
   * FullyConnected
   * SoftMax
   * Split
   * Concatenation
   * Power
   * Eltwise
   * Crop
   * BatchNormalization
   * Copy

The following types of layers are not supported:

* Layers that might arbitrary change tensor shape (such as Flatten, Permute, Reshape)
* Layers specific to object detection topologies (ROIPooling, ProirBox, DetectionOutput)
* Custom layers

Topology analysis is performed during the process of loading a network into plugin, and if the topology is not supported, an exception is generated.

## Using Dynamic Batching (Python)

@sphinxdirective
.. raw:: html

    <div id="switcher-python" class="switcher-anchor">Python</div>
@endsphinxdirective

Dynamic Batching is a feature that allows you to dynamically change batch size for inference calls within a preset batch size limit. This feature might be useful when batch size is unknown beforehand, and using extra large batch size is not desired or impossible due to resource limitations. For example, face detection with person age, gender, or mood recognition is a typical usage scenario.

You can activate Dynamic Batching by setting the "DYN_BATCH_ENABLED" flag to "YES" in a configuration map that is passed to the plugin while loading a network. This configuration creates an `ExecutableNetwork` object that will allow setting batch size dynamically in all of its infer requests using the  [ie_api.batch_size](api/ie_python_api/_autosummary/openvino.inference_engine.IENetwork.html#openvino.inference_engine.IENetwork.batch_size) method. The batch size that was set in the passed CNNNetwork object will be used as a maximum batch size limit.

```python
from openvino.inference_engine import IECore

ie = IECore()
dyn_config = {"DYN_BATCH_ENABLED": "YES"}
ie.set_config(config=dyn_config, device_name=device)
# Read a network in IR or ONNX format
net = ie.read_network(path_to_model)
net.batch_size = 32  # set the maximum batch size to 32
exec_net = ie.load_network(network=net, device_name=device)
```

### Limitations

Currently, certain limitations for the use of Dynamic Batching exist:

* Use Dynamic Batching with CPU and GPU plugins only.
* Use Dynamic Batching on topologies that consist of certain layers only:
   * Convolution
   * Deconvolution
   * Activation
   * LRN
   * Pooling
   * FullyConnected
   * SoftMax
   * Split
   * Concatenation
   * Power
   * Eltwise
   * Crop
   * BatchNormalization
   * Copy

The following types of layers are not supported:

* Layers that might arbitrary change tensor shape (such as Flatten, Permute, Reshape)
* Layers specific to object detection topologies (ROIPooling, ProirBox, DetectionOutput)
* Custom layers

Topology analysis is performed during the process of loading a network into plugin, and if the topology is not supported, an exception is generated.