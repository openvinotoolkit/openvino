# Introduction to Intel® Deep Learning Deployment Toolkit {#openvino_docs_IE_DG_Introduction}

## Deployment Challenges

Deploying deep learning networks from the training environment to embedded platforms for inference
might be a complex task that introduces a number of technical challenges that must be addressed:

* There are a number of deep learning frameworks widely used in the industry, such as Caffe*, TensorFlow*, MXNet*, Kaldi* etc.

* Typically the training of the deep learning networks is performed in data centers or server farms while the inference
might take place on embedded platforms, optimized for performance and power consumption. Such platforms are typically
limited both from software perspective (programming languages, third party dependencies, memory consumption,
supported operating systems), and from hardware perspective (different data types, limited power envelope),
so usually it is not recommended (and sometimes just impossible) to use original training framework for inference.
An alternative solution would be to use dedicated inference APIs that are well optimized for specific hardware platforms.

* Additional complications of the deployment process include supporting various layer types and networks that are getting
more and more complex. Obviously, ensuring the accuracy of the transforms networks is not trivial.

## Deployment Workflow
The process assumes that you have a network model trained using one of the [supported frameworks](#SupportedFW).
The scheme below illustrates the typical workflow for deploying a trained deep learning model:
![scheme]

The steps are:

1. [Configure Model Optimizer](../MO_DG/prepare_model/Config_Model_Optimizer.md) for the specific framework (used to train your model).

2. Run [Model Optimizer](#MO) to produce an optimized [Intermediate Representation (IR)](../MO_DG/IR_and_opsets.md)
of the model based on the trained network topology, weights and biases values, and other optional parameters.

3. Test the model in the IR format using the [Inference Engine](#IE) in the target environment with provided 
[Inference Engine sample applications](Samples_Overview.md).

4. [Integrate Inference Engine](Integrate_with_customer_application_new_API.md) in your application to deploy the model in the target environment.


## Model Optimizer <a name = "MO"></a>

Model Optimizer is a cross-platform command line tool that facilitates the transition between the training and
deployment environment, performs static model analysis and automatically adjusts deep learning
models for optimal execution on end-point target devices.

Model Optimizer is designed to support multiple deep learning [supported frameworks and formats](#SupportedFW).

While running Model Optimizer you do not need to consider what target device you wish to use, the same output of the MO can be used in all targets.

### Model Optimizer Workflow

The process assumes that you have a network model trained using one of the [supported frameworks](#SupportedFW).
The Model Optimizer workflow can be described as following:

* [Configure Model Optimizer](../MO_DG/prepare_model/Config_Model_Optimizer.md) for one of the supported deep learning framework that was used to train the model.
* Provide as input a trained network that contains a certain network topology, and the adjusted weights and
biases (with some optional parameters).
* [Run Model Optimizer](../MO_DG/prepare_model/convert_model/Converting_Model.md) to perform specific model optimizations (for example, horizontal fusion of certain network layers). Exact optimizations
are framework-specific, refer to appropriate documentation pages: [Converting a Caffe Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_Caffe.md),
[Converting a TensorFlow Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_TensorFlow.md), [Converting a MXNet Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_MxNet.md), [Converting a Kaldi Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_Kaldi.md),
[Converting an ONNX Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_ONNX.md).
* Model Optimizer produces as output an [Intermediate Representation (IR)](../MO_DG/IR_and_opsets.md) of the network which is used as an input for the Inference Engine on all targets.


### Supported Frameworks and Formats <a name = "SupportedFW"></a>
* Caffe* (most public branches)
* TensorFlow*
* MXNet*
* Kaldi*
* ONNX*

### Supported Models
For the list of supported models refer to the framework or format specific page:
* [Supported Caffe* models](../MO_DG/prepare_model/convert_model/Convert_Model_From_Caffe.md)
* [Supported TensorFlow* models](../MO_DG/prepare_model/convert_model/Convert_Model_From_TensorFlow.md)
* [Supported MXNet* models](../MO_DG/prepare_model/convert_model/Convert_Model_From_MxNet.md)
* [Supported ONNX* models](../MO_DG/prepare_model/convert_model/Convert_Model_From_ONNX.md)
* [Supported Kaldi* models](../MO_DG/prepare_model/convert_model/Convert_Model_From_Kaldi.md)


## Intermediate Representation

Intermediate representation describing a deep learning model plays an important role connecting the OpenVINO&trade; toolkit components.
The IR is a pair of files:
    * `.xml`: The topology file - an XML file that describes the network topology
    * `.bin`: The trained data file - a .bin file that contains the weights and biases binary data

Intermediate Representation (IR) files can be read, loaded and inferred with the [Inference Engine](#IE).
Inference Engine API offers a unified API across a number of [supported Intel® platforms](#SupportedTargets).
IR is also consumed, modified and written by Post-Training Optimization Tool which provides quantization capabilities.

Refer to a dedicated description about [Intermediate Representation and Operation Sets](../MO_DG/IR_and_opsets.md) for further details.

## nGraph Integration

OpenVINO toolkit is powered by nGraph capabilities for Graph construction API, Graph transformation engine and Reshape.
nGraph Function is used as an intermediate representation for a model in the run-time underneath the CNNNetwork API.
The conventional representation for CNNNetwork is still available if requested for backward compatibility when some conventional API methods are used.
Please refer to the [Overview of nGraph Flow](nGraph_Flow.md) describing the details of nGraph integration into the Inference Engine and co-existence with the conventional representation.

## Inference Engine <a name = "IE"></a>

Inference Engine is a runtime that delivers a unified API to integrate the inference with application logic:

* Takes a model as an input. The model can be presented in [the native ONNX format](./ONNX_Support.md) or in the specific form of [Intermediate Representation (IR)](../MO_DG/IR_and_opsets.md)
produced by Model Optimizer.
* Optimizes inference execution for target hardware.
* Delivers inference solution with reduced footprint on embedded inference platforms.

The Inference Engine supports inference of multiple image classification networks,
including AlexNet, GoogLeNet, VGG and ResNet families of networks, fully convolutional networks like FCN8 used for image
 segmentation, and object detection networks like Faster R-CNN.

For the full list of supported hardware, refer to the
[Supported Devices](supported_plugins/Supported_Devices.md) section.

For Intel® Distribution of OpenVINO™ toolkit, the Inference Engine package contains [headers](files.html), runtime libraries, and
[sample console applications](Samples_Overview.md) demonstrating how you can use
the Inference Engine in your applications.

The open source version is available in the [OpenVINO™ toolkit GitHub repository](https://github.com/openvinotoolkit/openvino) and can be built for supported platforms using the <a href="https://github.com/openvinotoolkit/openvino/blob/master/build-instruction.md">Inference Engine Build Instructions</a>.
## See Also
- [Inference Engine Samples](Samples_Overview.md)
- [Intel&reg; Deep Learning Deployment Toolkit Web Page](https://software.intel.com/en-us/computer-vision-sdk)


[scheme]: img/workflow_steps.png

#### Optimization Notice
<sup>For complete information about compiler optimizations, see our [Optimization Notice](https://software.intel.com/en-us/articles/optimization-notice#opt-en).</sup>
