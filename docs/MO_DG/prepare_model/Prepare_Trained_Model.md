# Preparing and Optimizing Your Trained Model {#openvino_docs_MO_DG_prepare_model_Prepare_Trained_Model}

## How the Model Optimizer Works

Model Optimizer loads a model into memory, reads it, builds the internal representation of the model, optimizes it, and produces the Intermediate Representation. Intermediate Representation is the only format the Inference Engine accepts.

Model Optimizer has two main purposes:

*   **Produce a valid Intermediate Representation**. If this main conversion artifact is not valid, the Inference Engine cannot run. The primary responsibility of the Model Optimizer is to produce the two files (`.xml` and `.bin`) that form the Intermediate Representation.
*   **Produce an optimized Intermediate Representation**. Pre-trained models contain layers that are important for training, such as the `Dropout` layer. These layers are useless during inference and might increase the inference time. In many cases, these operations can be automatically removed from the resulting Intermediate Representation. However, if a group of operations can be represented as a single mathematical operation, and thus as a single operation node in a model graph, the Model Optimizer recognizes such patterns and replaces this group of operation nodes with the only one operation. The result is an Intermediate Representation that has fewer operation nodes than the original model. This decreases the inference time.

To produce a valid Intermediate Representation, the Model Optimizer must be able to read the original model operations, handle their properties and represent them in Intermediate Representation format, while maintaining validity of the resulting Intermediate Representation. The resulting model consists of operations described in the [Operations Specification](../../ops/opset.md).

## What You Need to Know about Your Model

Many common layers exist across known frameworks and neural network topologies. Examples of these layers are `Convolution`, `Pooling`, and `Activation`. To read the original model and produce the Intermediate Representation of a model, the Model Optimizer must be able to work with these layers.

The full list of them depends on the framework and can be found in the [Supported Framework Layers](Supported_Frameworks_Layers.md) section. If your topology contains only layers from the list of layers, as is the case for the topologies used by most users, the Model Optimizer easily creates the Intermediate Representation. After that you can proceed to work with the Inference Engine.

However, if you use a topology with layers that are not recognized by the Model Optimizer out of the box, see [Custom Layers in the Model Optimizer](customize_model_optimizer/Customize_Model_Optimizer.md) to learn how to work with custom layers.

The following sections provide the information about how to use the Model Optimizer, from configuring the tool and generating an IR for a given model to customizing the tool for your needs:

* [Configuring Model Optimizer](Config_Model_Optimizer.md)
* [Converting a Model to Intermediate Representation](convert_model/Converting_Model.md)
* [Custom Layers in Model Optimizer](customize_model_optimizer/Customize_Model_Optimizer.md)
* [Model Optimization Techniques](Model_Optimization_Techniques.md)
* [Model Optimizer Frequently Asked Questions](Model_Optimizer_FAQ.md)
