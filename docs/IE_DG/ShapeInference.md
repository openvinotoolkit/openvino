Using Shape Inference {#openvino_docs_IE_DG_ShapeInference}
==========================================

OpenVINO™ provides the following methods for runtime model reshaping:

* **Set a new input shape** with the `InferenceEngine::CNNNetwork::reshape` method.<br>
   The `InferenceEngine::CNNNetwork::reshape` method updates input shapes and propagates them down to the outputs of the model through all intermediate layers. 
   
> **NOTES**:
> - Starting with the 2021.1 release, the Model Optimizer converts topologies keeping shape-calculating sub-graphs by default, which enables correct shape propagation during reshaping in most cases.
> - Older versions of IRs are not guaranteed to reshape successfully. Please regenerate them with the Model Optimizer of the latest version of OpenVINO™.<br>
> - If an ONNX model does not have a fully defined input shape and the model was imported with the ONNX importer, reshape the model before loading it to the plugin.

* **Set a new batch dimension value** with the `InferenceEngine::CNNNetwork::setBatchSize` method.<br>     
   The meaning of a model batch may vary depending on the model design.
   This method does not deduce batch placement for inputs from the model architecture.
   It assumes that the batch is placed at the zero index in the shape for all inputs and uses the `InferenceEngine::CNNNetwork::reshape` method to propagate updated shapes through the model.

   The method transforms the model before a new shape propagation to relax a hard-coded batch dimension in the model, if any.

   Use `InferenceEngine::CNNNetwork::reshape` instead of `InferenceEngine::CNNNetwork::setBatchSize` to set new input shapes for the model in case the model has:
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
2. [ONNX model](../IE_DG/OnnxImporterTutorial.md) through `InferenceEngine::Core::ReadNetwork`
3. [nGraph function](../nGraph_DG/nGraph_dg.md) through the constructor of `InferenceEngine::CNNNetwork`

`InferenceEngine::CNNNetwork` keeps an `ngraph::Function` object with the model description internally.
The object should have fully defined input shapes to be successfully loaded to the Inference Engine plugins.
To resolve undefined input dimensions of a model, call the `CNNNetwork::reshape` method providing new input shapes before loading to the Inference Engine plugin.

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

There are other approaches to reshape the model during the stage of <a href="_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html#when_to_specify_input_shapes">IR generation</a> or [nGraph::Function creation](../nGraph_DG/build_function.md).

Practically, some models are not ready to be reshaped. In this case, a new input shape cannot be set with the Model Optimizer or the `InferenceEngine::CNNNetwork::reshape` method.

## Troubleshooting Reshape Errors

Operation semantics may impose restrictions on input shapes of the operation. 
Shape collision during shape propagation may be a sign that a new shape does not satisfy the restrictions. 
Changing the model input shape may result in intermediate operations shape collision.

Examples of such operations:
- [Reshape](../ops/shape/Reshape_1.md) operation with a hard-coded output shape value
- [MatMul](../ops/matrix/MatMul_1.md) operation with the `Const` second input cannot be resized by spatial dimensions due to operation semantics

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

## Usage of Reshape Method <a name="usage_of_reshape_method"></a>

The primary method of the feature is `InferenceEngine::CNNNetwork::reshape`.
It gets new input shapes and propagates it from input to output for all intermediates layers of the given network.
The method takes `InferenceEngine::ICNNNetwork::InputShapes` - a map of pairs: name of input data and its dimension.

The algorithm for resizing network is the following:

1) **Collect the map of input names and shapes from Intermediate Representation (IR)** using helper method `InferenceEngine::CNNNetwork::getInputShapes`

2) **Set new input shapes**

3) **Call reshape**

Here is a code example:

@snippet snippets/ShapeInference.cpp part0

Shape Inference feature is used in [Smart Classroom Demo](@ref omz_demos_smart_classroom_demo_cpp).

## Extensibility

Inference Engine provides a special mechanism that allows to add the support of shape inference for custom operations. 
This mechanism is described in the [Extensibility documentation](Extensibility_DG/Intro.md)
