# Glossary {#openvino_docs_OV_Glossary}

## Acronyms and Abbreviations

| Abbreviation      | Description     |
| :---              | :--- |
| API               | Application Programming Interface |
| AVX               | Advanced Vector Extensions |
| clDNN             | Compute Library for Deep Neural Networks |
| CLI               | Command Line Interface |
| CNN               | Convolutional Neural Network |
| CPU               | Central Processing Unit |
| CV                | Computer Vision |
| DL                | Deep Learning |
| DLL               | Dynamic Link Library |
| DNN               | Deep Neural Networks |
| ELU               | Exponential Linear rectification Unit |
| FCN               | Fully Convolutional Network |
| FP                | Floating Point |
| GCC               | GNU Compiler Collection |
| GPU               | Graphics Processing Unit |
| HD                | High Definition |
| IR                | Intermediate Representation |
| JIT               | Just In Time |
| JTAG              | Joint Test Action Group |
| LPR               | License-Plate Recognition |
| LRN               | Local Response Normalization |
| mAP               | Mean Average Precision |
| Intel(R) MKL-DNN  | Intel(R) Math Kernel Library Deep Neural Networks |
| MO                | Model Optimizer |
| MVN               | Mean Variance Normalization |
| NCDHW             | Number of images, Channels, Depth, Height, Width |
| NCHW              | Number of images, Channels, Height, Width |
| NHWC              | Number of images, Height, Width, Channels |
| NMS               | Non-Maximum Suppression |
| NN                | Neural Network |
| NST               | Neural Style Transfer |
| OD                | Object Detection |
| OS                | Operating System |
| PCI               | Peripheral Component Interconnect |
| PReLU             | Parametric Rectified Linear Unit |
| PSROI             | Position Sensitive Region Of Interest |
| RCNN, R-CNN       | Region-based Convolutional Neural Network |
| ReLU              | Rectified Linear Unit |
| ROI               | Region Of Interest |
| SDK               | Software Development Kit |
| SSD               | Single Shot multibox Detector |
| SSE               | Streaming SIMD Extensions |
| USB               | Universal Serial Bus |
| VGG               | Visual Geometry Group |
| VOC               | Visual Object Classes |
| WINAPI            | Windows Application Programming Interface |

## Terms

Glossary of terms used in the OpenVINO™


| Term                        | Description         |
| :---                        | :---                |
| Batch | Number of images to analyze during one call of infer. Maximum batch size is a property of the network and it is set before loading of the network to the plugin. In NHWC, NCHW and NCDHW image data layout representation, the N refers to the number of images in the batch |
| Tensor | Memory container used for storing inputs, outputs of the network, weights and biases of the layers |
| Device (Affinitity) | A preferred Intel(R) hardware device to run the inference (CPU, GPU, etc.) |
| Extensibility mechanism, Custom layers | The mechanism that provides you with capabilities to extend the OpenVINO™ Runtime and Model Optimizer so that they can work with topologies containing layers that are not yet supported |
| <code>ov::Model</code> | A class of the Model that OpenVINO™ Runtime reads from IR. Consists of topology, weights and biases |
| <code>ov::CompiledModel</code> | An instance of the loaded network which allows the OpenVINO™ Runtime to request (several) infer requests and perform inference synchronously or asynchronously |
| <code>InferRequest</code> | A class that represents the end point of inference on the model loaded to the plugin and represented by executable network. Inputs are set here, outputs should be requested from this interface as well |
| <code>ov::ProfileInfo</code> | Represents basic inference profiling information per layer |
| OpenVINO™ Runtime | A C++ library with a set of classes that you can use in your application to infer input data (images) and get the result |
| OpenVINO™ API | The basic default API for all supported devices, which allows you to load a model from Intermediate Representation, set input and output formats and execute the model on various devices |
| OpenVINO™ <code>Core</code> | OpenVINO™ Core is a software component that manages inference on certain Intel(R) hardware devices: CPU, GPU, MYRIAD, GNA, etc. |
| <code>ov::Layout</code> | Image data layout refers to the representation of images batch. Layout shows a sequence of 4D or 5D tensor data in memory. A typical NCHW format represents pixel in horizontal direction, rows by vertical dimension, planes by channel and images into batch |
| <code>ov::element::Type</code> | Represents data element type. For example, f32 is 32-bit floating point, f16 is 16-bit floating point. Element type can be changed before loading the network to the plugin |


## See Also
* [Available Operations Sets](ops/opset.md)
* [Terminology](OV_Runtime_UG/supported_plugins/Supported_Devices.md)
