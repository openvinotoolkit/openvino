# Optimizing models post-training {#pot_introduction}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   Quantizing Model <pot_default_quantization_usage>
   Quantizing Model with Accuracy Control <pot_accuracyaware_usage>
   Quantization Best Practices <pot_docs_BestPractices>
   API Reference <pot_compression_api_README>
   Command-line Interface <pot_compression_cli_README>
   Examples <pot_examples_description>
   pot_docs_FrequentlyAskedQuestions

@endsphinxdirective

## Introduction

Post-training model optimization is the process of applying special methods without model retraining or fine-tuning, for example, post-training 8-bit quantization. Therefore, this process does not require a training dataset or a training pipeline in the source DL framework. To apply post-training methods in OpenVINO&trade;, you need:
* A floating-point precision model, FP32 or FP16, converted into the OpenVINO&trade; Intermediate Representation (IR) format
and run on CPU with the OpenVINO&trade;.
* A representative calibration dataset representing a use case scenario, for example, 300 samples.
* In case of accuracy constraints, a validation dataset and accuracy metrics should be available.

For the needs of post-training optimization, OpenVINO&trade; provides a Post-training Optimization Tool (POT) which supports the uniform integer quantization method. This method allows substantially increasing inference performance and reducing the model size.

Figure below shows the optimization workflow with POT:
![](./images/workflow_simple.png)


## Quantizing models with POT

POT provides two main quantization methods that can be used depending on the user's needs and requirements:

*  [Default Quantization](@ref pot_default_quantization_usage) is a recommended method that provides fast and accurate results in most cases. It requires only a unannotated dataset for quantization. For details, see the [Default Quantization algorithm](@ref pot_compression_algorithms_quantization_default_README) documentation.

*  [Accuracy-aware Quantization](@ref pot_accuracyaware_usage) is an advanced method that allows keeping accuracy at a predefined range at the cost of performance improvement in case when `Default Quantization` cannot guarantee it. The method requires annotated representative dataset and may require more time for quantization. For details, see the
[Accuracy-aware Quantization algorithm](@ref accuracy_aware_README) documentation.

HW platforms support different integer precisions and quantization parameters, for example 8-bit in CPU, GPU, VPU, 16-bit for GNA. POT abstracts this complexity by introducing a concept of "target device" that is used to set quantization settings specific to the device. The `target_device` parameter is used for this purpose.

> **NOTE**: There is a special `target_device: "ANY"` which leads to portable quantized models compatible with CPU, GPU, and VPU devices. GNA-quantized models are compatible only with CPU.

For benchmarking results collected for the models optimized with the POT tool, refer to [INT8 vs FP32 Comparison on Select Networks and Platforms](@ref openvino_docs_performance_int8_vs_fp32).

## See Also

* [Performance Benchmarks](https://docs.openvino.ai/latest/openvino_docs_performance_benchmarks_openvino.html)
* [INT8 Quantization by Using Web-Based Interface of the DL Workbench](https://docs.openvino.ai/latest/workbench_docs_Workbench_DG_Int_8_Quantization.html)
