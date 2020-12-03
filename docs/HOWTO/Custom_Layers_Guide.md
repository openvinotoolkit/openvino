# Custom Operations Guide {#openvino_docs_HOWTO_Custom_Layers_Guide}

The Intel® Distribution of OpenVINO™ toolkit supports neural network models trained with multiple frameworks including
TensorFlow*, Caffe*, MXNet*, Kaldi* and ONNX* file format. The list of supported operations (layers) is different for
each of the supported frameworks. To see the operations supported by your framework, refer to
[Supported Framework Layers](../MO_DG/prepare_model/Supported_Frameworks_Layers.md).

Custom operations are operations that are not included in the list of known operations. If your model contains any
operation that is not in the list of known operations, the Model Optimizer is not able to generate an Intermediate
Representation (IR) for this model.

This guide illustrates the workflow for running inference on topologies featuring custom operations, allowing you to
plug in your own implementation for existing or completely new operation.

> **NOTE:** *Layer* — The legacy term for an *operation* which came from Caffe\* framework. Currently it is not used.
> Refer to the [Deep Learning Network Intermediate Representation and Operation Sets in OpenVINO™](../MO_DG/IR_and_opsets.md)
> for more information on the topic.

## Terms Used in This Guide

- *Intermediate Representation (IR)* — Neural Network used only by the Inference Engine in OpenVINO abstracting the
  different frameworks and describing the model topology, operations parameters and weights.

- *Operation* — The abstract concept of a math function that is selected for a specific purpose. Operations supported by
  OpenVINO™ are listed in the supported operation set provided in the [Available Operations Sets](../ops/opset.md).
  Examples of the operations are: [ReLU](../ops/activation/ReLU_1.md), [Convolution](../ops/convolution/Convolution_1.md),
  [Add](../ops/arithmetic/Add_1.md), etc.

- *Kernel* — The implementation of a operation function in the OpenVINO™ plugin, in this case, the math programmed (in
  C++ and OpenCL) to perform the operation for a target hardware (CPU or GPU).

- *Inference Engine Extension* — Device-specific module implementing custom operations (a set of kernels).

## Custom Operation Support Overview

There are three steps to support inference of a model with custom operation(s):
1. Add support for a custom operation in the [Model Optimizer](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) so
the Model Optimizer can generate the IR with the operation.
2. Create an operation set and implement a custom nGraph operation in it as described in the
[Custom nGraph Operation](../IE_DG/Extensibility_DG/AddingNGraphOps.md).
3. Implement a customer operation in one of the [Inference Engine](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md)
plugins to support inference of this operation using a particular target hardware (CPU, GPU or VPU).

To see the operations that are supported by each device plugin for the Inference Engine, refer to the
[Supported Devices](../IE_DG/supported_plugins/Supported_Devices.md).

> **NOTE:** If a device doesn't support a particular operation, an alternative to creating a new operation is to target
> an additional device using the HETERO plugin. The [Heterogeneous Plugin](../IE_DG/supported_plugins/HETERO.md) may be
> used to run an inference model on multiple devices allowing the unsupported operations on one device to "fallback" to
> run on another device (e.g., CPU) that does support those operations.

### Custom Operation Support for the Model Optimizer

Model Optimizer model conversion pipeline is described in details in
[Model Conversion Pipeline](../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md#model-conversion-pipeline).
It is recommended to read that article first for a better understanding of the following material.

Model Optimizer provides extensions mechanism to support new operations and implement custom model transformations to
generate optimized IR. This mechanism is described in the
[Model Optimizer Extensions](../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md#extensions).

Two types of the Model Optimizer extensions should be implemented to support custom operation at minimum:
1. Operation class for a new operation. This class stores information about the operation, its attributes, shape
inference function, attributes to be saved to an IR and some others internally used attributes. Refer to the
[Model Optimizer Operation](../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md#extension-operation)
for the detailed instruction on how to implement it.
2. Operation attributes extractor. The extractor is responsible for parsing framework-specific representation of the
operation and uses corresponding operation class to update graph node attributes with necessary attributes of the
operation. Refer to the
[Operation Extractor](../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md#operation-extractor)
for the detailed instruction on how to implement it.

> **NOTE:** In some cases you may need to implement some transformation to support the operation. This topic is covered
> in the [Graph Transformation Extensions](../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer#graph-transformations).

## Custom Operations Extensions for the Inference Engine

Inference Engine provides extensions mechanism to support new operations. This mechanism is described in the
[Inference Engine Extensibility Mechanism](../IE_DG/Extensibility_DG/Intro.md).

Each device plugin includes a library of optimized implementations to execute known operations which must be extended to
execute a custom operation. The custom operation extension is implemented according to the target device:

- Custom Layer CPU Extension
   - A compiled shared library (.so, .dylib or .dll binary) needed by the CPU Plugin for executing the custom operation
   on a CPU. Refer to the [How to Implement Custom CPU Operations](../IE_DG/Extensibility_DG/CPU_Kernel.md) for more
   details.
- Custom Layer GPU Extension
   - OpenCL source code (.cl) for the custom operation kernel that will be compiled to execute on the GPU along with a
   operation description file (.xml) needed by the GPU Plugin for the custom operation kernel. Refer to the
   [How to Implement Custom GPU Operations](../IE_DG/Extensibility_DG/GPU_Kernel.md) for more details.
- Custom Layer VPU Extension
   - OpenCL source code (.cl) for the custom operation kernel that will be compiled to execute on the VPU along with a
   operation description file (.xml) needed by the VPU Plugin for the custom operation kernel. Refer to the
   [How to Implement Custom Operations for VPU](../IE_DG/Extensibility_DG/VPU_Kernel.md) for more details.

## Additional Resources

- Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)
- OpenVINO™ toolkit online documentation: [https://docs.openvinotoolkit.org](https://docs.openvinotoolkit.org)
- [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
- [Model Optimizer Customization](../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md)
- [Inference Engine Extensibility Mechanism](../IE_DG/Extensibility_DG/Intro.md)
- [Inference Engine Samples Overview](../IE_DG/Samples_Overview.md)
- [Overview of OpenVINO™ Toolkit Pre-Trained Models](@ref omz_models_intel_index)
- [Inference Engine Tutorials](https://github.com/intel-iot-devkit/inference-tutorials-generic)
- For IoT Libraries and Code Samples see the [Intel® IoT Developer Kit](https://github.com/intel-iot-devkit).

## Converting Models:

- [Convert Your Caffe* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_Caffe.md)
- [Convert Your Kaldi* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_Kaldi.md)
- [Convert Your TensorFlow* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_TensorFlow.md)
- [Convert Your MXNet* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_MxNet.md)
- [Convert Your ONNX* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_ONNX.md)



