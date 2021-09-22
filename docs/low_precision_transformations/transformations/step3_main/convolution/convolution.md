# OpenVINO™ Low Precision Transformations: ConvolutionTransformation {#openvino_docs_IE_DG_lpt_ConvolutionTransformation}

ngraph::pass::low_precision::ConvolutionTransformation class represents the `Convolution` operation transformation.

The transformation propagate dequantization operations on activations and on weights through `Convolution` operation. The transformation supports several weights quantization approaches:
* quantized weights in low precision with dequantization operations,
* weights in original precision with `FakeQuantize` operation.

Result dequantization `Multiply` constant value *result* is calculated as multiplication for dequantization `Multiply` constant value on activations *a* and dequantization `Multiply` constant value on weights *b* :

\f[
result_{i} = a_{i} \cdot b_{i}
\f]

## Limitations
* `Subtract` dequantization operations on activations and weights are not propagated and .
* Dequantization on activations have to be per-tensor. It means that dequantization `Multiply` constant value on activations has to be scalar.

## Subgraph before transformation

### Quantized weights in low precision with dequantization operations
The subgraph with quantized `Convolution` before transformation with quantized weights in low precision constant and dequantization operations:

![Convolution before](img/fq_and_convolution.common.png)

### Weights in original precision with FakeQuantize operation
The subgraph with quantized `Convolution` before transformation with weights in original precision and `FakeQuantize` operation:

![Convolution before](img/fq_fq_and_convolution.common.png)

## Subgraph after transformation
The subgraph with `Convolution` operation after the transformation:

![Convolution after](img/fq_and_convolution.transformed.png)