# Custom Layers in the Model Optimizer  {#openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer}

Model Optimizer searches for each layer of the input model in the list of known layers before building the model's internal representation, optimizing the model, and producing the Intermediate Representation.

The list of known layers is different for each of supported frameworks. To see the layers supported by your framework, refer to the [corresponding section](../Supported_Frameworks_Layers.md).

Custom layers are layers that are not included into a list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.

## Caffe\* Models with Custom Layers <a name="caffe-models-with-custom-layers"></a>

You have two options if your Caffe\* model has custom layers:

*   **Register the custom layers as extensions to the Model Optimizer**. For instructions, see [Extending Model Optimizer with New Primitives](Extending_Model_Optimizer_with_New_Primitives.md). When your custom layers are registered as extensions, the Model Optimizer generates a valid and optimized Intermediate Representation. You only need to write a small chunk of Python\* code that lets the Model Optimizer:

    *   Generate a valid Intermediate Representation according to the rules you specified
    *   Be independent from the availability of Caffe on your computer
	
*   **Register the custom layers as Custom and use the system Caffe to calculate the output shape of each Custom Layer**, which is required by the Intermediate Representation format. For this method, the Model Optimizer requires the Caffe Python interface on your system. When registering the custom layer in the `CustomLayersMapping.xml` file, you can specify if layer parameters should appear in Intermediate Representation or if they should be skipped. To read more about the expected format and general structure of this file, see [Legacy Mode for Caffe* Custom Layers](Legacy_Mode_for_Caffe_Custom_Layers.md). This approach has several limitations:

    *   If your layer output shape depends on dynamic parameters, input data or previous layers parameters, calculation of output shape of the layer via Caffe can be incorrect. In this case, you need to patch Caffe on your own.
	
    *   If the calculation of output shape of the layer via Caffe fails inside the framework, Model Optimizer is unable to produce any correct Intermediate Representation and you also need to investigate the issue in the implementation of layers in the Caffe and patch it.
	
    *   You are not able to produce Intermediate Representation on any machine that does not have Caffe installed. If you want to use Model Optimizer on multiple machines, your topology contains Custom Layers and you use `CustomLayersMapping.xml` to fallback on Caffe, you need to configure Caffe on each new machine. 
	
	For these reasons, it is best to use the Model Optimizer extensions for Custom Layers: you do not depend on the framework and fully control the workflow.

If your model contains Custom Layers, it is important to understand the internal workflow of Model Optimizer. Consider the following example.

**Example**:

The network has:

*   One input layer (#1)
*   One output Layer (#5)
*   Three internal layers (#2, 3, 4)

The custom and standard layer types are:

*   Layers #2 and #5 are implemented as Model Optimizer extensions.
*   Layers #1 and #4 are supported in Model Optimizer out-of-the box.
*   Layer #3 is neither in the list of supported layers nor in extensions, but is specified in CustomLayersMapping.xml.

> **NOTE**: If any of the layers are not in one of three categories described above, the Model Optimizer fails with an appropriate message and a link to the corresponding question in [Model Optimizer FAQ](../Model_Optimizer_FAQ.md).

The general process is as shown:

![Example custom layer network](../../img/mo_caffe_priorities.png)

1.  The example model is fed to the Model Optimizer that **loads the model** with the special parser, built on top of `caffe.proto` file. In case of failure, Model Optimizer asks you to prepare the parser that can read the model. For more information, refer to Model Optimizer, <a href="MO_FAQ.html#FAQ1">FAQ #1</a>.

2.  Model Optimizer **extracts the attributes of all layers**. In particular, it goes through the list of layers and attempts to find the appropriate extractor. In order of priority, Model Optimizer checks if the layer is:
    
    *   Registered in `CustomLayersMapping.xml`
    *   Registered as a Model Optimizer extension
    *   Registered as a standard Model Optimizer layer
    
    When the Model Optimizer finds a satisfying condition from the list above, it extracts the attributes according to the following rules:
    
    *   For bullet #1 - either takes all parameters or no parameters, according to the content of `CustomLayersMapping.xml`
    *   For bullet #2 - takes only the parameters specified in the extension
    *   For bullet #3 - takes only the parameters specified in the standard extractor
	
3.  Model Optimizer **calculates the output shape of all layers**. The logic is the same as it is for the priorities. **Important:** the Model Optimizer always takes the first available option.

4.  Model Optimizer **optimizes the original model and produces the Intermediate Representation**.

## TensorFlow\* Models with Custom Layers <a name="Tensorflow-models-with-custom-layers"></a>

You have two options for TensorFlow\* models with custom layers:

*   **Register those layers as extensions to the Model Optimizer.** In this case, the Model Optimizer generates a valid and optimized Intermediate Representation.
*   **If you have sub-graphs that should not be expressed with the analogous sub-graph in the Intermediate Representation, but another sub-graph should appear in the model, the Model Optimizer provides such an option.** This feature is helpful for many TensorFlow models. To read more, see [Sub-graph Replacement in the Model Optimizer](Subgraph_Replacement_Model_Optimizer.md).
	
## MXNet\* Models with Custom Layers <a name="mxnet-models-with-custom-layers"></a>

There are two options to convert your MXNet* model that contains custom layers:

1.  Register the custom layers as extensions to the Model Optimizer. For instructions, see [Extending MXNet Model Optimizer with New Primitives](Extending_MXNet_Model_Optimizer_with_New_Primitives.md). When your custom layers are registered as extensions, the Model Optimizer generates a valid and optimized Intermediate Representation. You can create Model Optimizer extensions for both MXNet layers with op `Custom` and layers which are not standard MXNet layers.

2.  If you have sub-graphs that should not be expressed with the analogous sub-graph in the Intermediate Representation, but another sub-graph should appear in the model, the Model Optimizer provides such an option. In MXNet the function is actively used for ssd models provides an opportunity to  for the necessary subgraph sequences and replace them. To read more, see [Sub-graph Replacement in the Model Optimizer](Subgraph_Replacement_Model_Optimizer.md).

