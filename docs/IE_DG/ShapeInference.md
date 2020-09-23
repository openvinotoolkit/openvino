Using Shape Inference {#openvino_docs_IE_DG_ShapeInference}
==========================================

Inference Engine takes three kinds of a model description as an input, which are converted into an `InferenceEngine::CNNNetwork` object:
1. [Intermediate Representation (IR)](../MO_DG/IR_and_opsets.md) through `InferenceEngine::Core::ReadNetwork`
2. [ONNX model](../IE_DG/OnnxImporterTutorial.md) through `InferenceEngine::Core::ReadNetwork`
3. [nGraph::Function](../IE_DG/nGraph_Flow.md) through the constructor of `InferenceEngine::CNNNetwork`

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

OpenVINO™ provides the following methods for runtime model reshaping:

* **Set a new input shape** with the `InferenceEngine::CNNNetwork::reshape` method.<br>
   The `InferenceEngine::CNNNetwork::reshape` method updates input shapes and propagates them down to the outputs of the model through all intermediate layers. 
   You can reshape a model multiple times like in this application scheme:
   ```
   ReadNetwork -> reshape(input_1_shape) -> LoadNetwork -> infer(input_1)
              \
               -> reshape(input_2_shape) -> LoadNetwork -> infer(input_2)
   ```
   > **NOTES**:
   > - Starting with the 2021.1 release, the Model Optimizer converts topologies keeping shape-calculating sub-graphs by default, which enables correct shape propagation during reshaping.
   > - Older versions of IRs are not guaranteed to reshape successfully. Please regenerate them with the Model Optimizer of the latest version of OpenVINO™.<br>
   > - If an ONNX model does not have a fully defined input shape and the model was imported with the ONNX importer, reshape the model before loading it to the plugin.
* **Set a new batch dimension value** with the `InferenceEngine::CNNNetwork::setBatchSize` method.<br>     
   The meaning of a model batch may vary depending on the model design.
   The `InferenceEngine::CNNNetwork::setBatchSize` method deduces the index of a batch dimension based only on the input rank. 
   This method does not work for models with a non-zero index batch placement or models with inputs without a batch dimension. 
   The batch-setting algorithm does not involve the shape inference mechanism.
   Batch of input and output shapes for all layers is set to a new batch value without layer validation.
   It may cause both positive and negative side effects.
   Due to the limitations described above, the current method is not recommended to use.
   If you need to set a new batch size for the model, use the `CNNNetwork::reshape` method instead.

Do not use runtime reshaping methods simultaneously, especially do not call the `CNNNetwork::reshape` method after you use `InferenceEngine::CNNNetwork::setBatchSize`.
The `InferenceEngine::CNNNetwork::setBatchSize` method causes irreversible conversion of the internal model representation into the legacy model representation.
The method does not use nGraph for shape inference which leads to reduced reshape opportunities and may affect the performance of the model.

There are other approaches to reshape the model during the stage of <a href="_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html#when_to_specify_input_shapes">IR generation</a> or [nGraph::Function creation](../IE_DG/nGraphTutorial.md).

Practically, some models are not ready to be reshaped. In this case, a new input shape cannot be set with the Model Optimizer or the `InferenceEngine::CNNNetwork::reshape` method.

## Troubleshooting Reshape Errors

Operation semantics may impose restrictions on input shapes of the operation. 
Shape collision during shape propagation may be a sign that a new shape does not satisfy the restrictions. 
Changing the model input shape may result in intermediate operations shape collision.

Examples of such operations:
- <a href="_docs_MO_DG_prepare_model_convert_model_IR_V10_opset1.html#Reshape">`Reshape` operation</a> with a hard-coded output shape value
- <a href="_docs_MO_DG_prepare_model_convert_model_IR_V10_opset1.html#MatMul">`MatMul` operation</a> with the `Const` second input cannot be resized by spatial dimensions due to operation semantics

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

@snippet openvino/docs/snippets/ShapeInference.cpp part0

Shape Inference feature is used in [Smart classroom sample](@ref omz_demos_smart_classroom_demo_README).

## Extensibility

Inference Engine provides a special mechanism that allows to add the support of shape inference for custom operations. 
This mechanism is described in the [Extensibility documentation](Extensibility_DG/Intro.md)

## Deprecation Notice

<table>
  <tr>
    <td><strong>Deprecation Begins</strong></td>
    <td>June 1, 2020</td>
  </tr>
  <tr>
    <td><strong>Removal Date</strong></td>
    <td>December 1, 2020</td>
  </tr>
</table> 

*Starting with the OpenVINO™ toolkit 2020.2 release, all of the features previously available through nGraph have been merged into the OpenVINO™ toolkit. As a result, all the features previously available through ONNX RT Execution Provider for nGraph have been merged with ONNX RT Execution Provider for OpenVINO™ toolkit.*

*Therefore, ONNX RT Execution Provider for nGraph will be deprecated starting June 1, 2020 and will be completely removed on December 1, 2020. Users are recommended to migrate to the ONNX RT Execution Provider for OpenVINO™ toolkit as the unified solution for all AI inferencing on Intel® hardware.*
