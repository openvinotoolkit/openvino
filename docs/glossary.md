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
| Intel® OneDNN     | Intel® OneAPI Deep Neural Network Library |
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

Glossary of terms used in OpenVINO™

@sphinxdirective

| Batch 
|   Number of images to analyze during one call of infer. Maximum batch size is a property of the model set before its compilation. In NHWC, NCHW, and NCDHW image data layout representations, the 'N' refers to the number of images in the batch.

| Device Affinitity 
|   A preferred hardware device to run inference (CPU, GPU, GNA, etc.).

| Extensibility mechanism, Custom layers 
|   The mechanism that provides you with capabilities to extend the OpenVINO™ Runtime and Model Optimizer so that they can work with models containing operations that are not yet supported.

| layer / operation
|   In OpenVINO, both terms are treated synonymously. To avoid confusion, "layer" is being pushed out and "operation" is the currently accepted term. 

| OpenVINO™ <code>Core</code> 
|   OpenVINO™ Core is a software component that manages inference on certain Intel(R) hardware devices: CPU, GPU, GNA, etc.    

| OpenVINO™ API 
|   The basic default API for all supported devices, which allows you to load a model from Intermediate Representation or convert from ONNX, PaddlePaddle, TensorFlow file formats, set input and output formats and execute the model on various devices.

| OpenVINO™ Runtime 
|   A C++ library with a set of classes that you can use in your application to infer input tensors and get the results.

| <code>ov::Model</code> 
|   A class of the Model that OpenVINO™ Runtime reads from IR or converts from ONNX, PaddlePaddle, TensorFlow formats. Consists of model structure, weights and biases.

| <code>ov::CompiledModel</code> 
|   An instance of the compiled model which allows the OpenVINO™ Runtime to request (several) infer requests and perform inference synchronously or asynchronously.

| <code>ov::InferRequest</code> 
|   A class that represents the end point of inference on the model compiled by the device and represented by a compiled model. Inputs are set here, outputs should be requested from this interface as well.

| <code>ov::ProfilingInfo</code> 
|   Represents basic inference profiling information per operation.

| <code>ov::Layout</code> 
|   Image data layout refers to the representation of images batch. Layout shows a sequence of 4D or 5D tensor data in memory. A typical NCHW format represents pixel in horizontal direction, rows by vertical dimension, planes by channel and images into batch. See also [Layout API Overview](./OV_Runtime_UG/layout_overview.md).

| <code>ov::element::Type</code> 
|   Represents data element type. For example, f32 is 32-bit floating point, f16 is 16-bit floating point.

| plugin / Inference Device / Inference Mode
|   OpenVINO makes hardware available for inference based on several core components. They used to be called "plugins" in earlier versions of documentation and you may still find this term in some articles. Because of their role in the software, they are now referred to as Devices and Modes ("virtual" devices). For a detailed description of the concept, refer to [Inference Modes](@ref openvino_docs_Runtime_Inference_Modes_Overview) and [Inference Devices](@ref openvino_docs_OV_UG_Working_with_devices).

| Tensor
|   A memory container used for storing inputs and outputs of the model, as well as weights and biases of the operations.

@endsphinxdirective


## See Also
* [Available Operations Sets](ops/opset.md)
* [Terminology](OV_Runtime_UG/supported_plugins/Supported_Devices.md)
