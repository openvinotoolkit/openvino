# Preparing and Optimizing Your Trained Model {#openvino_docs_MO_DG_prepare_model_Prepare_Trained_Model}

Inference Engine enables _deploying_ your network model trained with any of supported deep learning frameworks: Caffe\*, TensorFlow\*, Kaldi\*, MXNet\* or converted to the ONNX\* format. To perform the inference, the Inference Engine does not operate with the original model, but with its Intermediate Representation (IR), which is optimized for execution on end-point target devices. To generate an IR for your trained model, the Model Optimizer tool is used.

## How the Model Optimizer Works

Model Optimizer loads a model into memory, reads it, builds the internal representation of the model, optimizes it, and produces the Intermediate Representation. Intermediate Representation is the only format the Inference Engine accepts.

> **NOTE**: Model Optimizer does not infer models. Model Optimizer is an offline tool that runs before the inference takes place.

Model Optimizer has two main purposes:

*   **Produce a valid Intermediate Representation**. If this main conversion artifact is not valid, the Inference Engine cannot run. The primary responsibility of the Model Optimizer is to produce the two files (`.xml` and `.bin`) that form the Intermediate Representation.
*   **Produce an optimized Intermediate Representation**. Pre-trained models contain layers that are important for training, such as the `Dropout` layer. These layers are useless during inference and might increase the inference time. In many cases, these operations can be automatically removed from the resulting Intermediate Representation. However, if a group of operations can be represented as a single mathematical operation, and thus as a single operation node in a model graph, the Model Optimizer recognizes such patterns and replaces this group of operation nodes with the only one operation. The result is an Intermediate Representation that has fewer operation nodes than the original model. This decreases the inference time.

To produce a valid Intermediate Representation, the Model Optimizer must be able to read the original model operations, handle their properties and represent them in Intermediate Representation format, while maintaining validity of the resulting Intermediate Representation. The resulting model consists of operations described in the [Operations Specification](../../ops/opset.md).

## What You Need to Know about Your Model

Many common layers exist across known frameworks and neural network topologies. Examples of these layers are `Convolution`, `Pooling`, and `Activation`. To read the original model and produce the Intermediate Representation of a model, the Model Optimizer must be able to work with these layers.

The full list of them depends on the framework and can be found in the [Supported Framework Layers](Supported_Frameworks_Layers.md) section. If your topology contains only layers from the list of layers, as is the case for the topologies used by most users, the Model Optimizer easily creates the Intermediate Representation. After that you can proceed to work with the Inference Engine.

However, if you use a topology with layers that are not recognized by the Model Optimizer out of the box, see [Custom Layers in the Model Optimizer](customize_model_optimizer/Customize_Model_Optimizer.md) to learn how to work with custom layers.

## Model Optimizer Directory Structure

After installation with OpenVINO&trade; toolkit or Intel&reg; Deep Learning Deployment Toolkit, the Model Optimizer folder has the following structure:
```
|-- model_optimizer
    |-- extensions
        |-- front - Front-End framework agnostic transformations (operations output shapes are not defined yet). 
            |-- caffe - Front-End Caffe-specific transformations and Caffe layers extractors
                |-- CustomLayersMapping.xml.example - example of file for registering custom Caffe layers (compatible with the 2017R3 release)
            |-- kaldi - Front-End Kaldi-specific transformations and Kaldi operations extractors
            |-- mxnet - Front-End MxNet-specific transformations and MxNet symbols extractors
            |-- onnx - Front-End ONNX-specific transformations and ONNX operators extractors            
            |-- tf - Front-End TensorFlow-specific transformations, TensorFlow operations extractors, sub-graph replacements configuration files. 
        |-- middle - Middle-End framework agnostic transformations (layers output shapes are defined).
        |-- back - Back-End framework agnostic transformations (preparation for IR generation).        
    |-- mo
        |-- back - Back-End logic: contains IR emitting logic
        |-- front - Front-End logic: contains matching between Framework-specific layers and IR specific, calculation of output shapes for each registered layer
        |-- graph - Graph utilities to work with internal IR representation
        |-- middle - Graph transformations - optimizations of the model
        |-- pipeline - Sequence of steps required to create IR for each framework
        |-- utils - Utility functions
    |-- tf_call_ie_layer - Source code that enables TensorFlow fallback in Inference Engine during model inference
    |-- mo.py - Centralized entry point that can be used for any supported framework
    |-- mo_caffe.py - Entry point particularly for Caffe
    |-- mo_kaldi.py - Entry point particularly for Kaldi
    |-- mo_mxnet.py - Entry point particularly for MXNet
    |-- mo_onnx.py - Entry point particularly for ONNX
    |-- mo_tf.py - Entry point particularly for TensorFlow
```

The following sections provide the information about how to use the Model Optimizer, from configuring the tool and generating an IR for a given model to customizing the tool for your needs:

* [Configuring Model Optimizer](Config_Model_Optimizer.md)
* [Converting a Model to Intermediate Representation](convert_model/Converting_Model.md)
* [Custom Layers in Model Optimizer](customize_model_optimizer/Customize_Model_Optimizer.md)
* [Model Optimization Techniques](Model_Optimization_Techniques.md)
* [Model Optimizer Frequently Asked Questions](Model_Optimizer_FAQ.md)
