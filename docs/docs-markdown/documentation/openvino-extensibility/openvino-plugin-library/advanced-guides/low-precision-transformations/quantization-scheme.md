---
sidebar_position: 1
sidebar_label: "Quantization Scheme"
description: "Learn about quantization scheme."
---

# Quantization Scheme


Key steps in the quantization scheme:

- Low Precision Transformations: `FakeQuantize` decomposition to Quantize with a low precision output and Dequantize. For more details, refer to the [Quantize decomposition](index.md) section.
- Low Precision Transformations: move Dequantize through operations. For more details, refer to the [Main transformations](step3-main/index.md) section.
- Plugin: fuse operations with Quantize and inference in low precision.

Quantization scheme features:

- Quantization operation is expressed through the `FakeQuantize` operation, which involves more than scale and shift. For more details, see: [FakeQuantize-1](../../../../openvino-ir-format/operation-sets/operation-specs/quantization/fake-quantize-1.md). If the `FakeQuantize` input and output intervals are the same, `FakeQuantize` degenerates to `Multiply`, `Subtract` and `Convert` (scale & shift).
- Dequantization operation is expressed through element-wise `Convert`, `Subtract` and `Multiply` operations. `Convert` and `Subtract` are optional. These operations can be handled as typical element-wise operations, for example, fused or transformed to another.
- OpenVINO plugins fuse `Dequantize` and `Quantize` operations after a low precision operation and do not fuse `Quantize` before it.

Here is a quantization scheme example for int8 quantization applied to a part of a model with two `Convolution` operations in CPU plugin.

![Quantization scheme](quantization_scheme.svg)
