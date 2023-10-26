# Low-precision model representation

## Introduction
The goal of this document is to describe how optimized models are represented in OpenVINO Intermediate Representation (IR) and provide guidance on interpretation rules for such models at runtime.
Currently, there are two groups of optimization methods that can change the IR after applying them to the full-precision model:
- **Sparsity**. It is represented by zeros inside the weights and this is up to the hardware plugin how to interpret these zeros (use weights as is or apply special compression algorithms and sparse arithmetic). No additional mask is provided with the model.
- **Quantization**. The rest of this document is dedicated to the representation of quantized models.

## Representation of quantized models

The OpenVINO Toolkit represents all the quantized models using the so-called [FakeQuantize](https://docs.openvino.ai/2021.4/openvino_docs_MO_DG_prepare_model_convert_model_Legacy_IR_Layers_Catalog_Spec.html#fakequantize-layer) operation. This operation is very expressive and allows mapping values from arbitrary input and output ranges. We project (discretize) the input values to the low-precision data type using affine transformation (with clamp and rounding) and then re-project discrete values back to the original range and data type. It can be considered as an emulation of the quantization/dequantization process which happens at runtime. The figure below shows a part of the DL model, namely the Convolutional layer, that undergoes various transformations, from being a floating-point model to an integer model executed in the OpenVINO runtime. Column 2 of this figure below shows a model quantized with [Neural Network Compression Framework (NNCF)](https://github.com/openvinotoolkit/nncf).
![](images/model_flow.png) 

To reduce memory footprint weights of quantized models are transformed to a target data type, e.g. in the case of 8-bit quantization, this is int8. During this transformation, the floating-point weights tensor and one of the FakeQuantize operations that correspond to it are replaced with 8-bit weight tensor and the sequence of Convert, Subtract, Multiply operations that represent the typecast and dequantization parameters (scale and zero-point) as it is shown in column 3 of the figure.

## Interpreting FakeQuantize at runtime
At inference time, the quantized model undergoes the second set of transformations that allows interpreting floating-point operations with quantization rules as integer operations. OpenVINO Toolkit has Low-Precision Transformations (LPT) component for that purpose.
At runtime each FakeQuantize can be split into two independent operations: **Quantize** and **Dequantize** (column 4). **Quantize** transforms the input data into the target precision while **Dequantize** transforms the resulting values back to the original range. *Dequantize* operations can be propagated forward through the linear layers, such as *Convolution* or *Fully-Connected*, and, in some cases, fused with the following *Quantize* operation for the next layer into the so-called *Requantize* operation (column 5).

From the computation standpoint, the FakeQuantize formula is split into two parts:  
`output = round((x - input_low) / (input_high - input_low) * (levels-1)) / (levels-1) * (output_high - output_low) + output_low`  
The first part of this fomula represents *Quantize* operation:  
`q = round((x - input_low) / (input_high - input_low) * (levels-1))`  
The second is responsible for the dequantization:  
`r = q / (levels-1) * (output_high - output_low) + output_low`  
From the scale/zero-point notation standpoint the latter formula can be written as follows:  
`r = (output_high - output_low) / (levels-1) * (q + output_low / (output_high - output_low) * (levels-1))`  

Thus we can define:
- **Scale** as `(output_high - output_low) / (levels-1)`
- **Zero-point** as `-output_low / (output_high - output_low) * (levels-1)`

**Note**: During the quantization process the values `input_low`, `input_high`, `output_low`, `output_high` are selected so that to map a floating-point zero exactly to an integer value (zero-point) and vice versa.