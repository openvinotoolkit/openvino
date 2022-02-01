# Post-Training Optimization Tool {#pot_README}

## Introduction

Post-training Optimization Tool (POT) is designed to accelerate the inference of deep learning models by applying
special methods without model retraining or fine-tuning, e.g. post-training 8-bit quantization. Therefore, the tool does not
require a training dataset or a pipeline. To apply post-training algorithms from the POT, you need:
* A floating-point precision model, FP32 or FP16, converted into the OpenVINO&trade; Intermediate Representation (IR) format
and run on CPU with the OpenVINO&trade;.
* A representative calibration dataset representing a use case scenario, e.g. 300 images. 

Figure below shows the optimization workflow:
![](docs/images/workflow_simple.png) 

Post-training Optimization Tool provides the following key
features:

* Two post-training 8-bit quantization algorithms: fast [DefaultQuantization](openvino/tools/pot/algorithms/quantization/default/README.md) and precise [AccuracyAwareQuantization](openvino/tools/pot/algorithms/quantization/accuracy_aware/README.md).
* Compression for different hardware targets such as CPU and GPU.
* Multiple domains: Computer Vision, Natural Language Processing, Recommendation Systems, Speech Recognition.
* [Command-line tool](docs/CLI.md) that provides a simple .
* [API](openvino/tools/pot/api/README.md) that helps to apply optimization methods within a custom inference script written with OpenVINO Python* API.
* (Experimental) Ranger algorithm for model prodection in safity critical cases.

The tool is aimed to fully automate the model transformation process without a need to change the model on the user's side. For details about 
the low-precision flow in OpenVINO&trade;, see the [Low Precision Optimization Guide](docs/LowPrecisionOptimizationGuide.md).

For benchmarking results collected for the models optimized with POT tool, see [INT8 vs FP32 Comparison on Select Networks and Platforms](@ref openvino_docs_performance_int8_vs_fp32).

POT is opensourced on GitHub as a part of OpenVINO and available at https://github.com/openvinotoolkit/openvino/tools/pot.

Further documentation presumes that you are familiar with the basic Deep Learning concepts, such as model inference,
dataset preparation, model optimization, as well as with the OpenVINO&trade; toolkit and its components such 
as  [Model Optimizer](@ref openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide) 
and [Accuracy Checker Tool](@ref omz_tools_accuracy_checker_README).

## Usage options
![](docs/images/use_cases.png) 

The POT provides four basic usage options:
* **Command-line interface (CLI)**:
  * [**Data-free mode**](@ref pot_docs_data_free): this option can be used in case when only a model is available and there is no access to data. Currentlu, The data-free flow is available only for Computer Vision models. Please note that there can be significant deviation of model accuracy after optimization using this method.
  * [**Simplified mode**](@ref pot_docs_simplified_mode): this option can be used if the model from Computer Vision domain and there is an unannotated dataset that can be used for optimization. Please note that there can be deviation of accuracy after optimization using this method.
  * [**Model Zoo flow**](@ref pot_compression_cli_README): this option is recommended if the model from OpenVINO&trade; 
[Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) or there is a valid [Accuracy Checker Tool](@ref omz_tools_accuracy_checker_README)
configuration file for the model that allows validating model accuracy using [Accuracy Checker Tool](@ref omz_tools_accuracy_checker_README).
* [**Python\* API**](@ref pot_compression_api_README): it allows integrating optimization methods implemented in POT into
a Python* inference script that uses [OpenVINO Python* API](@ref openvino_inference_engine_ie_bridges_python_docs_api_overview). 


POT is also integrated into [Deep Learning Workbench](@ref workbench_docs_Workbench_DG_Introduction) (DL Workbench), a web-based graphical environment 
that enables you to optimize, tune, analyze, visualize, and compare performance of deep learning models. 

## Getting started

To install POT, follow the [Installation Guide](docs/InstallationGuide.md).

OpenVINO provides several examples that demonstrates usage of POT optimization workflow:

* Command-line example:
  * [Quantization of Image Classification model](https://docs.openvino.ai/latest/pot_configs_examples_README.html) 
* API tutorials:
  * [Quantization of Image Classification model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/301-tensorflow-training-openvino)
  * [Quantization of Object Detection model from Model Zoo](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/111-detection-quantization)
  * [Quantization of BERT for Text Classification](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/105-language-quantize-bert)
* API examples:
  * [Quantization of 3D segmentation model](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/3d_segmentation)
  * [Quantization of Face Detection model](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/face_detection)
  * [Speech example for GNA device](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/speech)


## See Also

* [Low Precision Optimization Guide](docs/LowPrecisionOptimizationGuide.md)
* [Post-Training Optimization Best Practices](docs/BestPractices.md)
* [POT Frequently Asked Questions](docs/FrequentlyAskedQuestions.md) 
* [INT8 Quantization by Using Web-Based Interface of the DL Workbench](https://docs.openvinotoolkit.org/latest/workbench_docs_Workbench_DG_Int_8_Quantization.html)
