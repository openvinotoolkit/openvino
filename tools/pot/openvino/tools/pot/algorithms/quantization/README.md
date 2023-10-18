# Quantization

## Introduction

The primary optimization feature of the Post-training Optimization Tool (POT) is the uniform integer quantization which allows substantially increasing inference performance and reducing the model size. Different HW platforms can support different integer precisions and POT is designed to support all of them, for example, 8-bit for CPU, GPU, NPU, 16-bit for GNA. Moreover, POT makes the specification of HW settings transparent for the user by introducing a concept of the `target_device` parameter.

> **NOTE**: There is a special `target_device: "ANY"` which leads to portable quantized models compatible with CPU, GPU, and NPU devices. GNA-quantized models are compatible only with CPU.

During the quantization process, the POT tool runs inference of the optimizing model to estimate quantization parameters for input activations of the quantizable operation. It means that a calibration dataset is required to perform quantization. This dataset may have or not have annotation depending on the quantization algorithm that is used.

## Quantization Algorithms

Currently, the POT provides two algorithms for 8-bit quantization, which are verified and guarantee stable results on a
wide range of DNN models:
*  [**DefaultQuantization**](@ref pot_compression_algorithms_quantization_default_README) is a default method that provides fast and in most cases accurate results. It requires only an unannotated dataset for quantization. For details, see the [DefaultQuantization Algorithm](@ref pot_compression_algorithms_quantization_default_README) documentation.

*  [**AccuracyAwareQuantization**](@ref accuracy_aware_README) enables remaining at a predefined range of accuracy drop after quantization at the cost
   of performance improvement. The method requires annotated representative dataset and may require more time for quantization. For details, see the
   [AccuracyAwareQuantization Algorithm](@ref accuracy_aware_README) documentation.

For more details about the representation of the low-precision model please refer to this [document](@ref pot_docs_model_representation).

## See also
* [Optimization with Simplified mode](@ref pot_docs_simplified_mode)
* [Use POT Command-line for Model Zoo models](@ref pot_compression_cli_README)
* [POT API](@ref pot_compression_api_README)
* [Post-Training Optimization Best Practices](@ref pot_docs_BestPractices)




