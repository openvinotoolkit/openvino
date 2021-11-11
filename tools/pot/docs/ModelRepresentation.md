# Representation of low-precision models {#pot_docs_ModelRepresentation}
The goal of this document is to describe how optimized models are represented in OpenVINO Intermediate Representation (IR) and provide guidance on interpretation rules for such models at runtime. 
Currently, there are two groups of optimization methods that can influence on the IR after applying them to the full-precision model:
- **Sparsity**. It is represented by zeros inside the weights and this is up to the hardware plugin how to interpret these zeros (use weights as is or apply special compression algorithms and sparse arithmetic). No additional mask is provided with the model.
- **Quantization**. The rest of this document is dedicated to the representation of quantized models.

## Representation of quantized models
The OpenVINO Toolkit represents all the quantized models using the so-called [FakeQuantize](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Legacy_IR_Layers_Catalog_Spec.html#FakeQuantize) operation. This operation is very expressive and allows mapping values from arbitrary input and output ranges. The whole idea behind that is quite simple: we project (discretize) the input values to the low-precision data type using affine transformation (with clamp and rounding) and then reproject discrete values back to the original range and data type. It can be considered as an emulation of the quantization process which happens at runtime.
In order to be able to execute a particular DL operation in low-precision all its inputs should be quantized i.e. should have FakeQuantize between operation and data blobs.  The figure below shows an example of quantized Convolution which contains two FakeQuantize nodes: one for weights and one for activations (bias is quantized using the same parameters).
<div align="center"><img src="./images/quantized_convolution.png" alt="This browser does not support PNG" width=70% height=70%></div>  
<div align="center">Figure 1. Example of quantized Convolution operation.</div><br/>

Starting from OpenVINO 2020.2 release all the quantized models are represented in the compressed form. It means that the weights of low-precision operations are converted into the target precision (e.g. INT8). It helps to substantially reduce the model size. The rest of the parameters can be represented by FLOAT32 or FLOAT16 precision depending on the input full-precision model used in the quantization process. Fig. 2 below shows an example of the part of the compressed IR.
<div align="center"><img src="./images/quantized_model_example.png" alt="This browser does not support PNG" width=70% height=70%></div>  
<div align="center">Figure 2. Example of compressed quantized model.</div>  

### Interpreting FakeQuantize at runtime
One important question that arises at inference time is how to correctly interpret quantized models and specifically FakeQuantize operations. OpenVINO Deep Learning Deployment Toolkit has a special component which is called Low-Precision Transformations (LPT). It is responsible for the translation of "fake-quantized" models into the models with low-precision operations. For more information about low-precision flow please refer to the following [document](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Int8Inference.html). Here we provide only a high-level overview of the interpretation rules of FakeQuantize operation. 
At runtime each FakeQuantize can be split into two independent operations: **Quantize** and **Dequantize**. The former one is aimed to transform the input data into the target precision while the latter transforms the resulting values back to the original range and precision. In practice *Dequantize* operations can be propagated forward through the linear low-precision layers, such as *Convolution* or *Fully-Connected*, and in some cases fused with the following *Quantize* operation for the next layer into the so-called *Requantize* operation (see Fig. 3).
<div align="center"><img src="./images/qdq_propagation.png" alt="This browser does not support PNG" width=70% height=70%></div>
<div align="center">Figure 3. Quantization operations propagation at runtime. Q, DQ, RQ stand for Quantize, Dequantize, and Requantize correspondingly.</div><br/>

From the calculation standpoint, the FakeQuantize formula also is split into two parts accordingly:  
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