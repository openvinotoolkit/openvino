Using Shape Inference {#openvino_docs_IE_DG_ShapeInference}
==========================================

Inference Engine takes two kinds of model description as an input: [Intermediate Representation (IR)](../MO_DG/IR_and_opsets.md) and [nGraph::Function](nGraph_Flow.md) objects. 
Both should have fixed input shapes to be successfully loaded to the Inference Engine.
To feed input data of a shape that is different from the model input shape, resize the model first.

Model resizing on the stage of <a href="_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html#when_to_specify_input_shapes">IR generation</a> or [nGraph::Function creation](nGraphTutorial.md) is the recommended approach. 
OpenVINO™ provides the following experimental methods for runtime model reshaping:

1.  Setting a new input shape with the `InferenceEngine::CNNNetwork::reshape` method
 
	`InferenceEngine::CNNNetwork::reshape` method updates input shapes and propagates them down to the outputs of the model through all intermediate layers.
    
    Shape propagation for `InferenceEngine::CNNNetwork` objects created from `nGraph::Function` or IR of the version 10 works through the `nGraph` shape inference mechanism. 
    `InferenceEngine::CNNNetwork` objects created from lower IR versions are considered deprecated and may be reshaped incorrectly or give unexpected results.
 
	To keep the v10 IR resizable by the `InferenceEngine::CNNNetwork::reshape` method, convert the model with the additional Model Optimizer key `--keep_shape_ops`.
 
2.  Setting a new batch dimension value with the `InferenceEngine::CNNNetwork::setBatchSize` method
    
    The meaning of a model batch may vary depending on choices you made during the model designing. 
    The `InferenceEngine::CNNNetwork::setBatchSize` method deduces index of batch dimension relying only on the input rank. 
    This method does not work for models with a non-zero index batch placement or models with inputs without a batch dimension. 

    Batch-setting algorithm does not involve shape inference mechanism.
    Batch of input and output shapes for all layers is set to a new batch value without layer validation.
    It may cause both positive and negative side effects.
 
    Due to the limitations described above, the current method is recommended for simple image processing models only.


Practically, some models are not ready to be resized. In this case, a new input shape cannot be set with the Model Optimizer or the `InferenceEngine::CNNNetwork::reshape` method.

## Troubleshooting Resize Errors

Operation semantics may impose restrictions on input shapes of the operation. 
Shape collision during shape propagation may be a sign that a new shape does not satisfy the restrictions. 
Changing the model input shape may result in intermediate operations shape collision.

Examples of such operations:
- <a href="_docs_MO_DG_prepare_model_convert_model_IR_V10_opset1.html#Reshape">`Reshape` operation</a> with a hard-coded output shape value
- <a href="_docs_MO_DG_prepare_model_convert_model_IR_V10_opset1.html#MatMul">`MatMul` operation</a> with the `Const` second input cannot be resized by spatial dimensions due to operation semantics

Model structure and logic should not change significantly after resizing.
- The Global Pooling operation is commonly used to reduce output feature map of classification models output.
Having the input of the shape [N, C, H, W], Global Pooling returns the output of the shape [N, C, 1, 1].
Model architects usually express Global Pooling with the help of the `Pooling` operation with the fixed kernel size [H, W].
During spatial reshape, having the input of the shape [N, C, H1, W1], Pooling with the fixed kernel size [H, W] returns the output of the shape [N, C, H2, W2], where H2 and W2 are commonly not equal to `1`.
It breaks the classification model structure.
For example, [publicly available Inception family models from TensorFlow*](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models) have this issue.

- Resizing the model input shape may significantly affect its accuracy.
For example, Object Detection models from TensorFlow have resizing restrictions by design. 
To keep the model valid after the reshape, choose a new input shape that satisfies conditions listed in the `pipeline.config` file. 
For details, refer to the <a href="_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html#tf_od_custom_input_shape">Tensorflow Object Detection API models resizing techniques</a>.

## Usage of Reshape Method

The primary method of the feature is `InferenceEngine::CNNNetwork::reshape`.
It gets new input shapes and propagates it from input to output for all intermediates layers of the given network.
The method takes `InferenceEngine::ICNNNetwork::InputShapes` - a map of pairs: name of input data and its dimension.

The algorithm for resizing network is the following:

1) **Collect the map of input names and shapes from Intermediate Representation (IR)** using helper method `InferenceEngine::CNNNetwork::getInputShapes`

2) **Set new input shapes**

3) **Call reshape**

Here is a code example:
```cpp
    InferenceEngine::Core core;
    // ------------- 0. Read IR and image ----------------------------------------------
    CNNNetwork network = core.ReadNetwork("path/to/IR/xml");
    cv::Mat image = cv::imread("path/to/image");
    // ---------------------------------------------------------------------------------

    // ------------- 1. Collect the map of input names and shapes from IR---------------
    auto input_shapes = network.getInputShapes();
    // ---------------------------------------------------------------------------------

    // ------------- 2. Set new input shapes -------------------------------------------
    std::string input_name;
    SizeVector input_shape;
    std::tie(input_name, input_shape) = *input_shapes.begin(); // let's consider first input only
    input_shape[0] = batch_size; // set batch size to the first input dimension
    input_shape[2] = image.rows; // changes input height to the image one
    input_shape[3] = image.cols; // changes input width to the image one
    input_shapes[input_name] = input_shape;
    // ---------------------------------------------------------------------------------

    // ------------- 3. Call reshape ---------------------------------------------------
    network.reshape(input_shapes);
    // ---------------------------------------------------------------------------------

    ...

    // ------------- 4. Loading model to the device ------------------------------------
    std::string device = "CPU";
    ExecutableNetwork executable_network = core.LoadNetwork(network, device);
    // ---------------------------------------------------------------------------------


```
Shape Inference feature is used in [Smart classroom sample](@ref omz_demos_smart_classroom_demo_README).

## Extensibility

Inference Engine provides a special mechanism that allows to add the support of shape inference for custom operations. 
This mechanism is described in the [Extensibility documentation](Extensibility_DG/Intro.md)
