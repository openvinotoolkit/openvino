# Low-Precision 8-bit Integer Inference {#openvino_docs_IE_DG_Int8Inference}

## Disclaimer

Inference Engine with low-precision 8-bit integer inference requires the following prerequisites to be satisfied:
- Inference Engine [CPU Plugin](supported_plugins/CPU.md) must be built with the Intel® Math Kernel Library (Intel® MKL) dependency. In the Intel® Distribution of OpenVINO™ it is 
  satisfied by default, this is mostly the requirement if you are using OpenVINO™ available in open source, because [open source version of OpenVINO™](https://github.com/openvinotoolkit/openvino) can be built with OpenBLAS* that is unacceptable if you want to use 8-bit integer inference.
- Intel® platforms that support at least one extension to x86 instruction set from the following list:
  - Intel® Advanced Vector Extensions 512 (Intel® AVX-512)
  - Intel® Advanced Vector Extensions 2.0 (Intel® AVX2)
  - Intel® Streaming SIMD Extensions 4.2 (Intel® SSE4.2)
- A model must be quantized. To quantize the model, you can use the [Post-Training Optimization Toolkit](@ref pot_README) delivered with the Intel® Distribution of OpenVINO™ toolkit release package.

The 8-bit inference feature was validated on the following topologies:
* **Classification models:**
	* Caffe\* DenseNet-121, DenseNet-161, DenseNet-169, DenseNet-201
    * Caffe Inception v1, Inception v2, Inception v3, Inception v4
    * Caffe YOLO v1 tiny, YOLO v3
	* Caffe ResNet-50 v1, ResNet-101 v1, ResNet-152 v1, ResNet-269 v1
    * Caffe ResNet-18
	* Caffe MobileNet, MobileNet v2
    * Caffe SE ResNeXt-50
	* Caffe SqueezeNet v1.0, SqueezeNet v1.1
	* Caffe VGG16, VGG19
    * TensorFlow\* DenseNet-121, DenseNet-169
    * TensorFlow Inception v1, Inception v2, Inception v3, Inception v4, Inception ResNet v2
    * TensorFlow Lite Inception v1, Inception v2, Inception v3, Inception v4, Inception ResNet v2
    * TensorFlow Lite MobileNet v1, MobileNet v2
    * TensorFlow MobileNet v1, MobileNet v2
    * TensorFlow ResNet-50 v1.5, ResNet-50 v1, ResNet-101 v1, ResNet-152 v1, ResNet-50 v2, ResNet-101 v2, ResNet-152 v2
    * TensorFlow VGG16, VGG19
    * TensorFlow YOLO v3
    * MXNet\* CaffeNet
    * MXNet DenseNet-121, DenseNet-161, DenseNet-169, DenseNet-201
    * MXNet Inception v3,  inception_v4
    * MXNet Mobilenet, Mobilenet v2
    * MXNet ResNet-101 v1, ResNet-152 v1, ResNet-101 v2, ResNet-152 v2
    * MXNet ResNeXt-101
    * MXNet SqueezeNet v1.1
    * MXNet VGG16, VGG19
    

* **Object detection models:**
	* Caffe SSD GoogLeNet 
    * Caffe SSD MobileNet
    * Caffe SSD SqueezeNet
	* Caffe SSD VGG16 300, SSD VGG16 512
    * TensorFlow SSD MobileNet v1, SSD MobileNet v2
    * MXNet SSD Inception v3 512
    * MXNet SSD MobileNet 512
    * MXNet SSD ResNet-50 512
    * MXNet SSD VGG16 300
    * ONNX\* SSD ResNet 34

* **Semantic segmentation models:**
    * Unet2D

* **Recommendation system models:**
    * NCF

## Introduction

A lot of investigation was made in the field of deep learning with the idea of using low precision computations during inference in order to boost deep learning pipelines and gather higher performance. For example, one of the popular approaches is to shrink the precision of activations and weights values from `fp32` precision to smaller ones, for example, to `fp11` or `int8`. For more information about this approach, refer to 
**Brief History of Lower Precision in Deep Learning** section in [this whitepaper](https://software.intel.com/en-us/articles/lower-numerical-precision-deep-learning-inference-and-training).

8-bit computations (referred to as `int8`) offer better performance compared to the results of inference in higher precision (for example, `fp32`), because they allow loading more data into a single processor instruction. Usually the cost for significant boost is a reduced accuracy. However, it is proved that an accuracy drop can be negligible and depends on task requirements, so that the application engineer can set up the maximum accuracy drop that is acceptable.

Current Inference Engine solution for low-precision inference uses Intel MKL-DNN and supports inference of the following layers in 8-bit integer computation mode:
* Convolution
* FullyConnected
* ReLU
* ReLU6
* Reshape
* Permute
* Pooling
* Squeeze
* Eltwise
* Concat
* Resample
* MVN

This means that 8-bit inference can only be performed with the CPU plugin on the layers listed above. All other layers are executed in the format supported by the CPU plugin: 32-bit floating point format (`fp32`).

## Low-Precision 8-bit Integer Inference Workflow

For 8-bit integer computations, a model must be quantized. If the model is not quantized then you can use Intel&reg; `Post-Training Optimization Toolkit` tool to quantize the model. The quantization process adds `FakeQuantize` layers on activations and weights for most layers. Read more about mathematical computations under the hood in the [white paper](https://intel.github.io/mkl-dnn/ex_int8_simplenet.html).

8-bit inference pipeline includes two stages (also refer to the figure below):
1. *Offline stage*, or *model quantization*. During this stage, `FakeQuantize` layers are added before most layers to have quantized tensors before layers in a way that low-precision accuracy drop for 8-bit integer inference satisfies the specified threshold. The output of this stage is a quantized model. Quantized model precision is not changed, quantized tensors are in original precision range (`fp32`). `FakeQuantize` layer has `Quantization Levels` attribute whic defines quants count. Quants count defines precision which is used during inference. For `int8` range `Quantization Levels` attribute value has to be 255 or 256.

2. *Run-time stage*. This stage is an internal procedure of the [CPU Plugin](supported_plugins/CPU.md). During this stage, the quantized model is loaded to the plugin. The plugin updates each `FakeQuantize` layer on activations and weights to have `FakeQuantize` output tensor values in low precision range. 


### Offline Stage: Model Quantization

To infer a layer in low precision and get maximum performance, the input tensor for the layer has to be quantized and each value has to be in the target low precision range. For this purpose, `FakeQuantize` layer is used in the OpenVINO™ intermediate representation file (IR). To quantize the model, you can use the [Post-Training Optimization Toolkit](@ref pot_README) delivered with the Intel® Distribution of OpenVINO™ toolkit release package.

When you pass the calibrated IR to the [CPU plugin](supported_plugins/CPU.md), the plugin automatically recognizes it as a quantized model and performs 8-bit inference. Note, if you pass a quantized model to another plugin that does not support 8-bit inference, the model is inferred in precision that this plugin supports.

### Run-Time Stage: Quantization

This is the second stage of the 8-bit integer inference. After you load the quantized model IR to a plugin, the pluing uses the `Low Precision Transformation` component to update the model to infer it in low precision:
* Updates `FakeQuantize` layers to have quantized output tensors in low precision range and add dequantization layers to compensate the update. Dequantization layers are pushed through as many layers as possible to have more layers in low precision. After that, most layers have quantized input tensors in low precision range and can be inferred in low precision. Ideally, dequantization layers should be fused in next `FakeQuantize` or `ScaleShift` layers.
* Weights are quantized and stored in `Const` layers.
* Biases are updated to avoid shifts in dequantization layers.

## Performance Counters

Information about layer precision is stored in the performance counters that are
available from the Inference Engine API. The layers have the following marks:
* Suffix `I8` for layers that had 8-bit data type input and were computed in 8-bit precision
* Suffix `FP32` for layers computed in 32-bit precision

For example, the performance counters table for the Inception model can look as follows:

```
inception_5b/5x5_reduce       EXECUTED       layerType: Convolution        realTime: 417        cpu: 417            execType: gemm_blas_I8
inception_5b/output           EXECUTED       layerType: Concat             realTime: 34         cpu: 34             execType: ref_I8
inception_5b/output_U8_nhw... EXECUTED       layerType: Reorder            realTime: 33092      cpu: 33092          execType: reorder_I8
inception_5b/output_oScale... EXECUTED       layerType: ScaleShift         realTime: 1390       cpu: 1390           execType: jit_avx2_FP32
inception_5b/output_oScale... EXECUTED       layerType: Reorder            realTime: 143        cpu: 143            execType: reorder_FP32
inception_5b/pool             EXECUTED       layerType: Pooling            realTime: 59301      cpu: 59301          execType: ref_any_I8
```

The `execType` column of the table includes inference primitives with specific suffixes.
