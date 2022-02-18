# Using the Reshape Inference Feature {#openvino_docs_IE_DG_ShapeInference}

## Introduction (C++)

@sphinxdirective
.. raw:: html

    <div id="switcher-cpp" class="switcher-anchor">C++</div>
@endsphinxdirective

OpenVINO™ provides two methods for runtime model reshaping: setting a new input shape and setting a new batch dimension value.

### Set a new input shape with the reshape() method

The `InferenceEngine::CNNNetwork::reshape` method updates input shapes and propagates them down to the outputs of the model through all intermediate layers. 
   
> **NOTES**:
> - Starting with the 2021.1 release, the Model Optimizer converts topologies keeping shape-calculating sub-graphs by default, which enables correct shape propagation during reshaping in most cases.
> - Older versions of IRs are not guaranteed to reshape successfully. Please regenerate them with the Model Optimizer of the latest version of OpenVINO™.<br>
> - If an ONNX model does not have a fully defined input shape and the model was imported with the ONNX importer, reshape the model before loading it to the plugin.

### Set a new batch dimension value with the setBatchSize() method

The meaning of a model batch may vary depending on the model design.
This method does not deduce batch placement for inputs from the model architecture.
It assumes that the batch is placed at the zero index in the shape for all inputs and uses the `InferenceEngine::CNNNetwork::reshape` method to propagate updated shapes through the model.

The method transforms the model before a new shape propagation to relax a hard-coded batch dimension in the model, if any.

Use `InferenceEngine::CNNNetwork::reshape` instead of `InferenceEngine::CNNNetwork::setBatchSize` to set new input shapes for the model if the model has one of the following:

* Multiple inputs with different zero-index dimension meanings
* Input without a batch dimension
* 0D, 1D, or 3D shape

   The `InferenceEngine::CNNNetwork::setBatchSize` method is a high-level API method that wraps the `InferenceEngine::CNNNetwork::reshape` method call and works for trivial models from the batch placement standpoint.
   Use `InferenceEngine::CNNNetwork::reshape` for other models.

   Using the `InferenceEngine::CNNNetwork::setBatchSize` method for models with a non-zero index batch placement or for models with inputs that do not have a batch dimension may lead to undefined behaviour.
    
You can change input shapes multiple times using the `InferenceEngine::CNNNetwork::reshape` and `InferenceEngine::CNNNetwork::setBatchSize` methods in any order.
If a model has a hard-coded batch dimension, use `InferenceEngine::CNNNetwork::setBatchSize` first to change the batch, then call `InferenceEngine::CNNNetwork::reshape` to update other dimensions, if needed.

Inference Engine takes three kinds of a model description as an input, which are converted into an `InferenceEngine::CNNNetwork` object:
1. [Intermediate Representation (IR)](../MO_DG/IR_and_opsets.md) through `InferenceEngine::Core::ReadNetwork`
2. [ONNX model](../OV_Runtime_UG/ONNX_Support.md) through `InferenceEngine::Core::ReadNetwork`
3. [OpenVINO Model](../OV_Runtime_UG/model_representation.md) through the constructor of `InferenceEngine::CNNNetwork`

`InferenceEngine::CNNNetwork` keeps an `ngraph::Function` object with the model description internally.
The object should have fully-defined input shapes to be successfully loaded to Inference Engine plugins.
To resolve undefined input dimensions of a model, call the `CNNNetwork::reshape` method to provide new input shapes before loading to the Inference Engine plugin.

Run the following code right after `InferenceEngine::CNNNetwork` creation to explicitly check for model input names and shapes:

```cpp
CNNNetwork network = ... // read IR / ONNX model or create from nGraph::Function explicitly
const auto parameters = network.getFunction()->get_parameters();
for (const auto & parameter : parameters) {
    std::cout << "name: " << parameter->get_friendly_name() << " shape: " << parameter->get_partial_shape() << std::endl;
    if (parameter->get_partial_shape().is_dynamic())
        std::cout << "ATTENTION: Input shape is not fully defined. Use the CNNNetwork::reshape method to resolve it." << std::endl;
}
```

To feed input data of a shape that is different from the model input shape, reshape the model first.

Once the input shape of `InferenceEngine::CNNNetwork` is set, call the `InferenceEngine::Core::LoadNetwork` method to get an `InferenceEngine::ExecutableNetwork` object for inference with updated shapes.

There are other approaches to reshape the model during the stage of <a href="_docs_MO_DG_prepare_model_convert_model_Converting_Model.html#when_to_specify_input_shapes">IR generation</a> or [ov::Model creation](../OV_Runtime_UG/model_representation.md).

Practically, some models are not ready to be reshaped. In this case, a new input shape cannot be set with the Model Optimizer or the `InferenceEngine::CNNNetwork::reshape` method.

### Usage of Reshape Method <a name="usage_of_reshape_method"></a>

The primary method of the feature is `InferenceEngine::CNNNetwork::reshape`. It gets new input shapes and propagates it from input to output for all intermediates layers of the given network.
The method takes `InferenceEngine::ICNNNetwork::InputShapes` - a map of pairs: name of input data and its dimension.

The algorithm for resizing network is the following:

1) **Collect the map of input names and shapes from Intermediate Representation (IR)** using helper method `InferenceEngine::CNNNetwork::getInputShapes`

2) **Set new input shapes**

3) **Call reshape**

Here is a code example:

@snippet snippets/ShapeInference.cpp part0

The Shape Inference feature is used in [Smart Classroom Demo](@ref omz_demos_smart_classroom_demo_cpp).

### Troubleshooting Reshape Errors

Operation semantics may impose restrictions on input shapes of the operation. 
Shape collision during shape propagation may be a sign that a new shape does not satisfy the restrictions. 
Changing the model input shape may result in intermediate operations shape collision.

Examples of such operations:
* [Reshape](../ops/shape/Reshape_1.md) operation with a hard-coded output shape value
* [MatMul](../ops/matrix/MatMul_1.md) operation with the `Const` second input cannot be resized by spatial dimensions due to operation semantics

Model structure and logic should not change significantly after model reshaping.
- The Global Pooling operation is commonly used to reduce output feature map of classification models output.
Having the input of the shape [N, C, H, W], Global Pooling returns the output of the shape [N, C, 1, 1].
Model architects usually express Global Pooling with the help of the `Pooling` operation with the fixed kernel size [H, W].
During spatial reshape, having the input of the shape [N, C, H1, W1], Pooling with the fixed kernel size [H, W] returns the output of the shape [N, C, H2, W2], where H2 and W2 are commonly not equal to `1`.
It breaks the classification model structure.
For example, [publicly available Inception family models from TensorFlow*](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models) have this issue.

- Changing the model input shape may significantly affect its accuracy.
For example, Object Detection models from TensorFlow have resizing restrictions by design. 
To keep the model valid after the reshape, choose a new input shape that satisfies conditions listed in the `pipeline.config` file. 
For details, refer to the <a href="_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html#tf_od_custom_input_shape">Tensorflow Object Detection API models resizing techniques</a>.

### Extensibility
The Inference Engine provides a special mechanism that allows adding support of shape inference for custom operations. This mechanism is described in the [Extensibility documentation](Extensibility_DG/Intro.md)

## Introduction (Python)

@sphinxdirective
.. raw:: html

    <div id="switcher-python" class="switcher-anchor">Python</div>
@endsphinxdirective

OpenVINO™ provides the following methods for runtime model reshaping:

* Set a new input shape with the [IENetwork.reshape](api/ie_python_api/_autosummary/openvino.inference_engine.IENetwork.html#openvino.inference_engine.IENetwork.reshape) method.

  The [IENetwork.reshape](api/ie_python_api/_autosummary/openvino.inference_engine.IENetwork.html#openvino.inference_engine.IENetwork.reshape) method updates input shapes and propagates them down to the outputs of the model through all intermediate layers.

  **NOTES**:
  * Model Optimizer converts topologies keeping shape-calculating sub-graphs by default, which enables correct shape propagation during reshaping in most cases.
  * Older versions of IRs are not guaranteed to reshape successfully. Please regenerate them with the Model Optimizer of the latest version of OpenVINO™.
  * If an ONNX model does not have a fully defined input shape and the model was imported with the ONNX importer, reshape the model before loading it to the plugin.


* Set a new batch dimension value with the [IENetwork.batch_size](api/ie_python_api/_autosummary/openvino.inference_engine.IENetwork.html#openvino.inference_engine.IENetwork.batch_size) method.

   The meaning of a model batch may vary depending on the model design. This method does not deduce batch placement for inputs from the model architecture. It assumes that the batch is placed at the zero index in the shape for all inputs and uses the [IENetwork.reshape](api/ie_python_api/_autosummary/openvino.inference_engine.IENetwork.html#openvino.inference_engine.IENetwork.reshape) method to propagate updated shapes through the model.

The method transforms the model before a new shape propagation to relax a hard-coded batch dimension in the model, if any.

Use [IENetwork.reshape](api/ie_python_api/_autosummary/openvino.inference_engine.IENetwork.html#openvino.inference_engine.IENetwork.reshape) rather than [IENetwork.batch_size](api/ie_python_api/_autosummary/openvino.inference_engine.IENetwork.html#openvino.inference_engine.IENetwork.batch_size) to set new input shapes for the model if the model has:

  * Multiple inputs with different zero-index dimension meanings
  * Input without a batch dimension
  * 0D, 1D, or 3D shape

The [IENetwork.batch_size](api/ie_python_api/_autosummary/openvino.inference_engine.IENetwork.html#openvino.inference_engine.IENetwork.batch_size) method is a high-level API method that wraps the [IENetwork.reshape](api/ie_python_api/_autosummary/openvino.inference_engine.IENetwork.html#openvino.inference_engine.IENetwork.reshape)  method call and works for trivial models from the batch placement standpoint. Use [IENetwork.reshape](api/ie_python_api/_autosummary/openvino.inference_engine.IENetwork.html#openvino.inference_engine.IENetwork.reshape) for other models.

Using the [IENetwork.batch_size](api/ie_python_api/_autosummary/openvino.inference_engine.IENetwork.html#openvino.inference_engine.IENetwork.batch_size) method for models with a non-zero index batch placement or for models with inputs that do not have a batch dimension may lead to undefined behaviour.

You can change input shapes multiple times using the `IENetwork.reshape` and `IENetwork.batch_size` methods in any order. If a model has a hard-coded batch dimension, use `IENetwork.batch_size` first to change the batch, then call `IENetwork.reshape` to update other dimensions, if needed.

Inference Engine takes three kinds of a model description as an input, which are converted into an IENetwork object:

1. Intermediate Representation (IR) through `IECore.read_network`
2. ONNX model through `IECore.read_network`
3. nGraph function through the constructor of IENetwork

IENetwork keeps an `ngraph::Function` object with the model description internally. The object should have fully defined input shapes to be successfully loaded to the Inference Engine plugins. To resolve undefined input dimensions of a model, call the `IENetwork.reshape` method providing new input shapes before loading to the Inference Engine plugin.

Run the following code right after IENetwork creation to explicitly check for model input names and shapes:

To feed input data of a shape that is different from the model input shape, reshape the model first.

Once the input shape of IENetwork is set, call the `IECore.load_network` method to get an ExecutableNetwork object for inference with updated shapes.

There are other approaches to reshape the model during the stage of IR generation or [nGraph function](https://docs.openvino.ai/latest/openvino_docs_nGraph_DG_PythonAPI.html#create_an_ngraph_function_from_a_graph) creation.

Practically, some models are not ready to be reshaped. In this case, a new input shape cannot be set with the Model Optimizer or the `IENetwork.reshape` method.

### Troubleshooting Reshape Errors
Operation semantics may impose restrictions on input shapes of the operation. Shape collision during shape propagation may be a sign that a new shape does not satisfy the restrictions. Changing the model input shape may result in intermediate operations shape collision.

Examples of such operations:

* Reshape operation with a hard-coded output shape value
* MatMul operation with the Const second input cannot be resized by spatial dimensions due to operation semantics

A model's structure and logic should not significantly change after model reshaping.

* The Global Pooling operation is commonly used to reduce output feature map of classification models output. Having the input of the shape [N, C, H, W], Global Pooling returns the output of the shape [N, C, 1, 1]. Model architects usually express Global Pooling with the help of the Pooling operation with the fixed kernel size [H, W]. During spatial reshape, having the input of the shape [N, C, H1, W1], Pooling with the fixed kernel size [H, W] returns the output of the shape [N, C, H2, W2], where H2 and W2 are commonly not equal to 1. It breaks the classification model structure. For example, publicly available Inception family models from TensorFlow* have this issue.

* Changing the model input shape may significantly affect its accuracy. For example, Object Detection models from TensorFlow have resizing restrictions by design. To keep the model valid after the reshape, choose a new input shape that satisfies conditions listed in the pipeline.config file. For details, refer to the Tensorflow Object Detection API models resizing techniques.


### Usage of the Reshape Method

The primary method of the feature is `IENetwork.reshape`. It gets new input shapes and propagates it from input to output for all intermediates layers of the given network. Use `IENetwork.input_info` to get names of input_layers and `.tensor_desc.dims` to get the current network input shape.

The following code example shows how to reshape a model to the size of an input image.

```python
import cv2
import numpy as np
from openvino.inference_engine import IECore

ie = IECore()

# Read an input image and transpose input to NCWH
image = cv2.imread(path_to_image_file)
input_image = image.transpose((2, 0, 1))
input_image = np.expand_dims(input_image, axis=0)

# Load the model and get input info
# Note that this model must support arbitrary input shapes
net = ie.read_network(model=path_to_xml_file)
input_layer = next(iter(net.input_info))
print(f"Input shape: {net.input_info[input_blob].tensor_desc.dims}")

# Call reshape
net.reshape({input_layer: input_image.shape})
print(f"New input shape: {net.input_info[input_blob].tensor_desc.dims}")

# Load the model to the device and proceed with inference
exec_net = ie.load_network(network=net, device_name="CPU")
```

### Extensibility
The Inference Engine provides a special mechanism that allows adding support of shape inference for custom operations. This mechanism is described in the [Extensibility documentation](Extensibility_DG/Intro.md)

### See Also:

[Hello Reshape Python Sample](../../samples/python/hello_reshape_ssd/README.html)
