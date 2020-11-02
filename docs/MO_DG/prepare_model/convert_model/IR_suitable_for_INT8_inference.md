# Intermediate Representation Suitable for INT8 Inference {#openvino_docs_MO_DG_prepare_model_convert_model_IR_suitable_for_INT8_inference}

## Introduction

Inference Engine CPU plugin can infer models in the 8-bit integer (INT8) precision. 
For details, refer to [INT8 inference on the CPU](../../../IE_DG/Int8Inference.md).

Intermediate Representation (IR) should be specifically formed to be suitable for the INT8 inference. 
Such an IR is called an INT8 IR and you can generate it in two ways:
- [Quantize model with the Post-Training Optimization tool](@ref pot_README)
- Use the Model Optimizer for TensorFlow\* pre-TFLite models (`.pb` model file with `FakeQuantize*` operations)

For an operation to be executed in INT8, it must have `FakeQuantize` operations as inputs with the `levels` attribute set to `255` or `256`. 
See the [specification of `FakeQuantize` operation](../../../ops/quantization/FakeQuantize_1.md) for details. 
To see the list of supported INT8 layers, refer to [INT8 inference on the CPU](../../../IE_DG/Int8Inference.md).

To execute the `Convolution` operation in INT8 on CPU, both data and weight inputs should have `FakeQuantize` as an input operation:
![](../../img/expanded_int8_Convolution_weights.png)

INT8 IR is also suitable for FP32 and FP16 inference if a chosen plugin supports all operations of the IR, because the only difference between an INT8 IR and FP16 or FP32 IR is the existence of `FakeQuantize` in the INT8 IR. 
Plugins with the INT8 inference support recognize these sub-graphs and quantize them during the inference time. 
Plugins without the INT8 support execute all operations, including `FakeQuantize`, as is in the FP32 or FP16 precision.   

Accordingly, the presence of FakeQuantize operations in the IR is a recommendation for a plugin on how to quantize particular operations in the model. 
If capable, a plugin accepts the recommendation and performs the INT8 inference, otherwise the plugin ignores the recommendation and executes a model in the floating-point precision. 

## Compressed INT8 Weights

Weighted operations, like `Convolution`, `MatMul`, and others, store weights as floating-point `Constant` in the graph followed by the `FakeQuantize` operation. 
`Constant` followed by the `FakeQuantize` operation could be optimized memory-wise due to the `FakeQuantize` operation semantics. 
The resulting weights sub-graph stores weights in INT8 `Constant`, which gets unpacked back to floating point with the `Convert` operation. 
Weights compression leaves `FakeQuantize` output arithmetically the same and weights storing takes four times less memory.

See the visualization of `Convolution` with the compressed weights:
![](../../img/compressed_int8_Convolution_weights.png)

Both Model Optimizer and Post-Training Optimization tool generate a compressed IR by default. To generate an expanded INT8 IR, use `--disable_weights_compression`.