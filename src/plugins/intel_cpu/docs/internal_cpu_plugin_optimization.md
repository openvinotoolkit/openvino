# Internal CPU Plugin Optimizations

The CPU plugin supports several graph optimization algorithms, such as fusing or removing layers.
Refer to the sections below for details.

> **NOTE**: For layer descriptions, see the [IR Notation Reference](https://docs.openvino.ai/latest/openvino_docs_ops_opset.html).


## Fusing Convolution and Simple Layers

Merge of a convolution layer and any of the simple layers listed below:
- Activation: ReLU, ELU, Sigmoid, Clamp
- Depthwise: ScaleShift, PReLU
- FakeQuantize

> **NOTE**: You can have any number and order of simple layers.

A combination of a convolution layer and simple layers results in a single fused layer called 
*Convolution*:

![conv_simple_01](https://user-images.githubusercontent.com/26419192/159540184-62d3b3f6-74d8-4135-ba8e-c84da42922d3.png)


## Fusing Pooling and FakeQuantize Layers

A combination of Pooling and FakeQuantize layers results in a single fused layer called *Pooling*:  

![pooling_fakequant_01](https://user-images.githubusercontent.com/26419192/159540321-41e7988a-f8a2-41f8-a2fe-17a3a55a00f0.png)

## Fusing FullyConnected and Activation Layers

A combination of FullyConnected and Activation layers results in a single fused layer called 
*FullyConnected*:

![fullyconnected_activation_01](https://user-images.githubusercontent.com/26419192/159540492-fd1aa3fc-ebb6-41d0-b1e0-3ec1d73da414.png)

## Fusing Convolution and Depthwise Convolution Layers Grouped with Simple Layers

> **NOTE**: This pattern is possible only on CPUs with support of Streaming SIMD Extensions 4.2 
> (SSE 4.2) and Intel AVX2 Instruction Set Architecture (ISA).

A combination of a group of a Convolution (or Binary Convolution) layer and simple layers and a group of a Depthwise Convolution
layer and simple layers results in a single layer called *Convolution* (or *Binary Convolution*):
> **NOTE**: Depthwise convolution layers should have the same values for the `group`, input channels, and output channels parameters.

![conv_depth_01](https://user-images.githubusercontent.com/26419192/159540640-d386ceea-30a8-4b43-9a0f-cfae03ba8dcf.png)

## Fusing Convolution and Sum Layers

A combination of convolution, simple, and Eltwise layers with the sum operation results in a single layer called *Convolution*:  

![conv_sum_relu_01](https://user-images.githubusercontent.com/26419192/159540705-7ff914c4-5097-454f-8231-da8623a1a607.png)

## Fusing a Group of Convolutions

If a topology contains the following pipeline, a CPU plugin merges split, convolution, and concatenation layers into a single convolution layer with the group parameter:   

![group_convolutions_01](https://user-images.githubusercontent.com/26419192/159540783-85e15dd2-f656-4287-9c1c-083cfb176903.png)

> **NOTE**: Parameters of the convolution layers must coincide.


## Removing a Power Layer

CPU plugin removes a Power layer from a topology if it has the following parameters:
  - <b>power</b> = 1
  - <b>scale</b> = 1
  - <b>offset</b> = 0

## See also
 * [OpenVINO™ README](../../../README.md)
 * [OpenVINO Core Components](../../README.md)
 * [OPENVINO CPU Plugin](../README.md)
 * [Developer documentation](../../../docs/dev/index.md)