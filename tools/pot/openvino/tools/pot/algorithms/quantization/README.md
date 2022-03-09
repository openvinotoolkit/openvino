# Quantization {#pot_compression_algorithms_quantization_README}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   
   DefaultQuantization Algorithm <pot_compression_algorithms_quantization_default_README>
   AccuracyAwareQuantization Algorithm <pot_compression_algorithms_quantization_accuracy_aware_README>
   TunableQuantization Algorithm <pot_compression_algorithms_quantization_tunable_quantization_README>
   Saturation Issue Workaround <pot_saturation_issue>
   Low-precision Model Representation <pot_docs_model_representation>


@endsphinxdirective

## Introduction

The primary optimization feature of the Post-training Optimization Tool (POT) is 8-bit uniform quantization which allows substantially increasing inference performance on all the platforms that have 8-bit instructions, for example, modern generations of CPU and GPU. Another benefit of quantization is a significant reduction of model footprint which in most cases achieves 4x. 

During the quantization process, the POT tool runs inference of the optimizing model to estimate quantization parameters for input activations of the quantizable operation. It means that a calibration dataset is required to perform quantization. This dataset may have or not have annotation depending on the quantization algorithm that is used.

## Quantization Algorithms

Currently, the POT provides two algorithms for 8-bit quantization, which are verified and guarantee stable results on a
wide range of DNN models:
*  [**DefaultQuantization**](@ref pot_compression_algorithms_quantization_default_README) is a default method that provides fast and in most cases accurate results for 8-bit
   quantization. It requires only a unannotated dataset for quantization. For details, see the [DefaultQuantization Algorithm](@ref pot_compression_algorithms_quantization_default_README) documentation.

*  [**AccuracyAwareQuantization**](@ref pot_compression_algorithms_quantization_accuracy_aware_README) enables remaining at a predefined range of accuracy drop after quantization at the cost
   of performance improvement. The method requires annotated representative dataset and may require more time for quantization. For details, see the
   [AccuracyAwareQuantization Algorithm](@ref pot_compression_algorithms_quantization_accuracy_aware_README) documentation.

For more details about the representation of the low-precision model please refer to this [document](@ref pot_docs_model_representation).

## See also
* [Optimization with Simplified mode](@ref pot_docs_simplified_mode)
* [Use POT Command-line for Model Zoo models](@ref pot_compression_cli_README)
* [POT API](@ref pot_compression_api_README)
* [Post-Training Optimization Best Practices](@ref pot_docs_BestPractices)




