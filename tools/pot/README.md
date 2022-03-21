# Post-Training Optimization Tool {#pot_README}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   pot_InstallationGuide
   pot_compression_algorithms_quantization_README
   Best Practices <pot_docs_BestPractices>
   Command-line Interface <pot_compression_cli_README>
   pot_compression_api_README
   pot_configs_README
   Deep Neural Network Protection <pot_ranger_README>
   pot_docs_FrequentlyAskedQuestions

@endsphinxdirective

## Introduction

Post-training Optimization Tool (POT) is designed to accelerate the inference of deep learning models by applying
special methods without model retraining or fine-tuning, for example, post-training 8-bit quantization. Therefore, the tool does not
require a training dataset or a pipeline. To apply post-training algorithms from the POT, you need:
* A floating-point precision model, FP32 or FP16, converted into the OpenVINO&trade; Intermediate Representation (IR) format
and run on CPU with the OpenVINO&trade;.
* A representative calibration dataset representing a use case scenario, for example, 300 images.

Figure below shows the optimization workflow:
![](docs/images/workflow_simple.png)

### Features

* Two post-training [quantization](@ref pot_compression_algorithms_quantization_README) algorithms: fast [DefaultQuantization](openvino/tools/pot/algorithms/quantization/default/README.md) and precise [AccuracyAwareQuantization](openvino/tools/pot/algorithms/quantization/accuracy_aware/README.md).
* Quantization for various types of hardware: CPU, GPU, VPU, GNA. The first three devices can use the same quantized model while GNA-quantized models are compatible with CPU only.
* [Python* API](openvino/tools/pot/api/README.md) and [Command-line tool](docs/CLI.md) interfaces.
* (Experimental) [Ranger algorithm](@ref pot_ranger_README) for the model protection in safety-critical cases.

For benchmarking results collected for the models optimized with the POT tool, refer to [INT8 vs FP32 Comparison on Select Networks and Platforms](@ref openvino_docs_performance_int8_vs_fp32).


Further documentation presumes that you are familiar with basic Deep Learning concepts, such as model inference, dataset preparation, model optimization, as well as with the OpenVINO&trade; toolkit and its components, such as  [Model Optimizer](@ref openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide) and [Accuracy Checker Tool](@ref omz_tools_accuracy_checker).

## Get started

### Installation
To install POT, follow the [Installation Guide](docs/InstallationGuide.md).

### Usage options

![](docs/images/use_cases.png)

The POT provides three basic usage options:
* [**Python\* API**](@ref pot_compression_api_README): allows creating Python* scripts that result to optimized models. This option is prefeble is the user has dataset, and a Python* code in any DL framework that does data loading and pre-processing.
* **Command-line interface (CLI)**:
  * [**Simplified mode**](@ref pot_docs_simplified_mode):  use this option if the model belongs to the **Computer Vision** domain and you have an **unannotated dataset** for optimization. This optimization method does not allow measuring model accuracy and might cause its deviation.
  * [**Model Zoo flow**](@ref pot_compression_cli_README): this option is recommended if the model is similar to the model from OpenVINO&trade; [Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) or there is a valid [Accuracy Checker Tool](@ref omz_tools_accuracy_checker)
configuration file for the model that allows validating model accuracy using [Accuracy Checker Tool](@ref omz_tools_accuracy_checker).

POT is also integrated into [Deep Learning Workbench](@ref workbench_docs_Workbench_DG_Introduction) (DL Workbench), a web-based graphical environment 
that enables you to  to import, optimize, benchmark, visualize, and compare performance of deep learning models. 

### Examples

OpenVINO provides several examples to demonstrate the POT optimization workflow:

* Command-line example:
  * [Quantization of Image Classification model](https://docs.openvino.ai/latest/pot_configs_examples_README.html)
* API tutorials:
  * [Quantization of Image Classification model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/301-tensorflow-training-openvino)
  * [Quantization of Object Detection model from Model Zoo](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/111-detection-quantization)
  * [Quantization of Segmentation model for medical data](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/110-ct-segmentation-quantize)
  * [Quantization of BERT for Text Classification](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/105-language-quantize-bert)
* API examples:
  * [Quantization of 3D segmentation model](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/3d_segmentation)
  * [Quantization of Face Detection model](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/face_detection)
  * [Quantization of Object Detection model with controable accuracy](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/object_detection)
  * [Quantizatin of speech model for GNA device](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/speech)


## See Also

* [Performance Benchmarks](https://docs.openvino.ai/latest/openvino_docs_performance_benchmarks_openvino.html)
* [Low Precision Optimization Guide](docs/LowPrecisionOptimizationGuide.md)
* [Post-Training Optimization Best Practices](docs/BestPractices.md)
* [POT Frequently Asked Questions](docs/FrequentlyAskedQuestions.md)
* [INT8 Quantization by Using Web-Based Interface of the DL Workbench](https://docs.openvino.ai/latest/workbench_docs_Workbench_DG_Int_8_Quantization.html)
