# Using Dynamic Batching {#openvino_docs_IE_DG_DynamicBatching}

## Dynamic Batching Introduction (C++)

@sphinxdirective
.. raw:: html

    <div id="switcher-cpp" class="switcher-anchor">C++</div>
@endsphinxdirective

The Dynamic Batching feature allows you to dynamically change batch size for inference calls
within a preset batch size limit.
This feature might be useful when batch size is unknown beforehand and using an extra-large batch size is
undesirable or impossible due to resource limitations.
For example, applying face detection and then mood labeling to a video, you won't know in advance how many frames will contain a face when you pass inferencing results to a secondary model.

## Usage

You can activate Dynamic Batching by setting <code>KEY_DYN_BATCH_ENABLED</code> flag to <code>YES</code> in a configuration map that is
passed to the plugin while loading a network.
This configuration creates an <code>ExecutableNetwork</code> object that will allow setting batch size
dynamically in all of its infer requests using <code>SetBatch()</code> method.
The batch size that was set in the passed <code>CNNNetwork</code> object will be used as a maximum batch size limit.

Here is a code example:

@snippet snippets/DynamicBatching.cpp part0


## Limitations

Currently, there certain limitations for using Dynamic Batching:

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
	
Do not use layers that might arbitrary change tensor shape (such as Flatten, Permute, Reshape),
layers specific to object detection topologies (ROIPooling, ProirBox, DetectionOutput), and
custom layers.
Topology analysis is performed during the process of loading a network into plugin, and if topology is
not applicable, an exception is generated.

