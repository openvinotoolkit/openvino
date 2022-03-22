# Post-Training Optimization Tool {#pot_introduction}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   Quantizing your model <pot_default_quantization_usage>
   Quantizing model with accuracy control <pot_accuracyaware_usage>
   Quantization Best Practices <pot_docs_BestPractices>
   Protecting your model <pot_ranger_README>
   API description <pot_compression_api_README>
   Command-line Interface <pot_compression_cli_README>
   Examples <pot_examples_description>
   pot_docs_FrequentlyAskedQuestions

@endsphinxdirective

## Introduction

Post-training Optimization Tool (POT) is designed to accelerate the inference of Deep Learning models by applying
special methods without model retraining or fine-tuning, for example, post-training 8-bit quantization. Therefore, the tool does not
require a training dataset or a pipeline. To apply post-training algorithms from the POT, you need:
* A floating-point precision model, FP32 or FP16, converted into the OpenVINO&trade; Intermediate Representation (IR) format
and run on CPU with the OpenVINO&trade;.
* A representative calibration dataset representing a use case scenario, for example, 300 images.
* In case of accuracy constraints, a validation dataset and accuracy metrics should be available.

Figure below shows the optimization workflow:
![](./images/workflow_simple.png)

### Features

* Two post-training [quantization](@ref pot_compression_algorithms_quantization_README) algorithms: fast [DefaultQuantization](openvino/tools/pot/algorithms/quantization/default/README.md) and precise [AccuracyAwareQuantization](openvino/tools/pot/algorithms/quantization/accuracy_aware/README.md).
* (Experimental) [Ranger algorithm](@ref pot_ranger_README) for the model protection in safety-critical cases.

For benchmarking results collected for the models optimized with the POT tool, refer to [INT8 vs FP32 Comparison on Select Networks and Platforms](@ref openvino_docs_performance_int8_vs_fp32).

POT is also integrated into [Deep Learning Workbench](@ref workbench_docs_Workbench_DG_Introduction) (DL Workbench), a web-based graphical environment 
that enables you to  to import, optimize, benchmark, visualize, and compare performance of deep learning models. 

## Quantizing models with POT
The primary feature of the Post-training Optimization Tool is the uniform integer quantization which allows substantially increasing inference performance and reduciing the model size. Different HW platforms can support different integer precisions and POT is designed to support all of them, for example 8-bit for CPU, GPU, VPU, 16-bit for GNA. Moreover, POT makes specification of HW settings transparent for the user by introducing a concept of `target_device` parameter.

> **NOTE**: There is a special `target_device: "ANY"` which leads to portable quantized models compatible with CPU, GPU, and VPU devices. GNA-quantized models are compatible only with CPU.

During the quantization process, the POT tool runs inference of the optimizing model to estimate quantization parameters for input activations of the quantizable operation. It means that a calibration dataset is required to perform quantization. This dataset may have or not have annotation depending on the quantization algorithm that is used and here there are two possible options:

*  [DefaultQuantization](@ref pot_default_quantization_usage) is a default method that provides fast and accurate results in most cases. It requires only a unannotated dataset for quantization. For details, see the [DefaultQuantization Algorithm](@ref pot_compression_algorithms_quantization_default_README) documentation.

*  [AccuracyAwareQuantization](@ref pot_accuracyaware_usage) enables remaining at a predefined range of accuracy drop after quantization at the cost
   of performance improvement. The method requires annotated representative dataset and may require more time for quantization. For details, see the
   [AccuracyAwareQuantization Algorithm](@ref pot_compression_algorithms_quantization_accuracy_aware_README) documentation.

### Examples

* Tutorials:
  * [Quantization of Image Classification model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/301-tensorflow-training-openvino)
  * [Quantization of Object Detection model from Model Zoo](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/111-detection-quantization)
  * [Quantization of Segmentation model for medical data](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/110-ct-segmentation-quantize)
  * [Quantization of BERT for Text Classification](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/105-language-quantize-bert)
* Samples:
  * [Quantization of 3D segmentation model](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/3d_segmentation)
  * [Quantization of Face Detection model](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/face_detection)
  * [Quantization of Object Detection model with controable accuracy](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/object_detection)
  * [Quantizatin of speech model for GNA device](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/speech)


## See Also

* [Performance Benchmarks](https://docs.openvino.ai/latest/openvino_docs_performance_benchmarks_openvino.html)
* [Post-Training Optimization Best Practices](BestPractices.md)
* [Using POT Command-line Interface](CLI.md)
* [POT Frequently Asked Questions](FrequentlyAskedQuestions.md)
* [INT8 Quantization by Using Web-Based Interface of the DL Workbench](https://docs.openvino.ai/latest/workbench_docs_Workbench_DG_Int_8_Quantization.html)
