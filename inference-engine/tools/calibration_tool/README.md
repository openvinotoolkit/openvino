# Python* Calibration Tool

## Introduction

The Calibration Tool quantizes a given FP16 or FP32 model and produces a low-precision 8-bit integer (INT8) model with keeping model inputs in the original precision. To learn more about benefits of inference in INT8 precision, refer to [Using Low-Precision 8-bit Integer Inference](./docs/IE_DG/Int8Inference.md).

> **NOTE**: INT8 models are currently supported only by the CPU plugin. For the full list of supported configurations, see the [Supported Devices](./docs/IE_DG/supported_plugins/Supported_Devices.md) topic.

You can run the Calibration Tool in two modes: 

* The **standard mode** performs quantization with the minimal accuracy drop within specified threshold compared to accuracy of the original model. This mode utilizes the [Accuracy Checker tool](./tools/accuracy_checker/README.md) to measure accuracy during the calibration process. Use this mode to obtain an INT8 IR that can be directly used in your application

* The **simplified mode** produces the IR that contains plain statistics for each layer which is collected without any accuracy check, meaning that the accuracy of the new IR with statistics might be dramatically low. Therefore, all layers are considered to be executed in INT8. Use this mode to understand a potential performance gain of model conversion to INT8 precision and make a conclusion about running the standard mode routine.

The Calibration Tool is a Python\* command-line tool, which imports Python types from the `openvino.tools.calibration` package.

## System Requirements
Hardware requirements depend on a model. Typically for public models RAM memory size has to be not less then 16GB, drive has to have not less then 30 GB free space independently on operation system. Temporary directory is used to cache layers output during calibration.

## Usage
You can run the Calibration Tool in either standard or simplified mode with an appropriate set of configuration parameters. 

### Standard Mode
In the standard mode, the Calibration Tool is configured in the same way as the Accuracy Checker.

> **NOTE**: For consistency reasons, a part of arguments have the same name and meaning as in the Accuracy Checker and can be reused for running the Accuracy Checker. 

For configuring the tool, you can use the following command-line arguments:

**Command-Line arguments common for the Calibration Tool and Accuracy Checker**

| Argument                                     | Type   | Description                                             |
| -------------------------------------------- | ------ | ------------------------------------------------------- |
| `-c`, `--config`                                 | string | Required. Path to the YML file with local configuration. |
| `-d`, `--definitions`                            | string | Optional. Path to the YML file with definitions.         |
| `-m`, `--models`                                 | string | Optional. Prefix path to the models and weights.   |
| `-s`, `--source`                                 | string | Optional. Prefix path to the data source.    |
| `-a`, `--annotations`                            | string | Optional. Prefix path to the converted annotations and datasets meta data. |
| `-e`, `--extensions`                             | string | Optional. Prefix path to extensions folder.              |
| `--cpu_extensions_mode`, `--cpu-extensions-mode` | string | Optional. Preferable set of processor instruction for automatic searching the CPU extension lib: `avx2` or `sse4`. |
| `-C`, `--converted_models`, `--converted-models`   | string | Optional. Directory to store Model Optimizer converted models.|
| `-M`, `--model_optimizer`, `--model-optimizer`     | string | Optional. Path to model optimizer Caffe* directory.       |
| `--tf_custom_op_config_dir`, `--tf-custom-op-config-dir` | string | Optional. Path to directory with TensorFlow* custom operation configuration files for model optimizer. |
| `--tf_obj_detection_api_pipeline_config_path`, `--tf-obj-detection-api-pipeline-config-path` | string | Optional. Path to directory with TensorFlow object detection API pipeline configuration files for the Model Optimizer. |
| `--progress`                                   | string | Optional. Progress reporter: `bar`, `print` or `None`   |
| `-td`, `--target_devices`, `--target-devices`      | string | Optional. Space-separated list of devices for infer     |
| `-tt`, `--target_tags`, `--target-tags`            | string | Optional. Space-separated list of launcher tags for infer        |

**Command Line Arguments specific for Calibration Tool**

| Argument                          | Type   | Description                                               |
| --------------------------------- | ------ | --------------------------------------------------------- |
| `-p`, `--precision`                   | string | Optional. Precision to calibrate. Default value is INT8. In the simplified mode, determines output IR precision.   |
| `--ignore_layer_types`, `--ignore-layer-types` | string | Optional. Layer types list which will be skipped during quantization. |
| `--ignore_layer_types_path`, `--ignore-layer-types-path` | string | Optional. Ignore layer types file path. |
| `--ignore_layer_names`, `--ignore-layer-names` | string | Optional. Layer names list which will be skipped during quantization. |
| `--ignore_layer_names_path`, `--ignore-layer-names-path` | string | Optional. Ignore layer names file path. |
| `--batch_size`, `--batch-size`        | integer| Optional. Batch size value. If not specified, the batch size value is determined from IR. |
| `-th`, `--threshold`                  | float | Optional. Accuracy drop of quantized model should not exceed this threshold. Should be pointer in percents without percent sign. (1% is default). |
| `-ic`, `--benchmark_iterations_count`, `--benchmark-iterations-count` | integer | Optional. Benchmark iterations count (1 is default). |
| `-mn`, `--metric_name`, `--metric-name` | string | Optional. Metric name used during calibration. |
| `-mt`, `--metric_type`, `--metric-type` | string | Optional. Metric type used during calibration. |
| `-o`, `--output_dir`, `--output-dir`    | string | Optional. Directory to store converted models. Original model directory is used if not defined. |

### Simplified Mode

The tool in this mode does not use the Accuracy Checker, configuration and annotation files, but you are required to specify paths to an IR .xml file and a dataset folder. Optionally, you can specify a prefix path to an extensions folder and the number of images from the dataset folder:

| Argument                          | Type   | Description                                               |
| --------------------------------- | ------ | --------------------------------------------------------- |
| `-sm`, `--simplified_mode`, `--simplified-mode` |   | Required. If specified, the Calibration Tool runs in the simplified mode to collects statistics without searching for optimal data thresholds. |
| `-m`                                 | string | Required. Path to the IR .xml file.   |
| `-s`, `--source`                      | string | Optional. Path to a folder with images.  | 
| `-ss`, `--subset`                     | integer | Optional. This option is used only with `--simplified_mode`. Specifies a number of images from a folder that is set using `-s` option. |
| `-e`, `--extensions`                             | string | Optional. Prefix path to extensions folder.              |
| `-td`, `--target_devices`, `--target-devices`      | string | Optional. Space-separated list of devices for infer.     |
| `-p`, `--precision`                   | string | Optional. Precision to calibrate. Default value is INT8. In the simplified mode, determines output IR precision.   |
| `-o`, `--output_dir`, `--output-dir`    | string | Optional. Directory to store converted models. Original model directory is used if not defined. |

## Typical Workflow Samples (Standard Mode)

### Introduction
The calibration tool reads original FP16 or FP32 models, calibration dataset and creates a low precision model. The low precision model has two differences from the original model:
1. Per channel statistics are defined. Statistics have minimum and maximum values for each layer and each channel. Model statistics are stored in Inference Engine intermediate representation file (IR) in XML format.
2. `quantization_level` layer attribute is defined. The attribute defines precision which is used during inference.

### Prerequisites
* Model: TensorFlow* Inception v1. You can download the model from here: https://github.com/tensorflow/models/tree/master/research/slim
* Dataset: ImageNet. You can download ImageNet from here: http://www.image-net.org/download.php
* YML configuration files: you can find YML configuration files and YML definition file which are used below in `configs` directory:
  - `definitions.yml` - definition file
  - `inception_v1.yml` - configuration file for TensorFlow* Inception v1 model
  - `ncf_config.yml` - configuration file for NCF model in OpenVINO Inference Engine Intermediate Representation format
  - `ssd_mobilenet_v1_coco.yml` - configuration file for TensorFlow* SSD Mobilenet v1 model
  - `unet2d.yml` - configuration file for Unet2D mode in in OpenVINO Inference Engine Intermediate Representation format

If your custom topology does not support accuracy metric or a custom dataset, add some components implementation in `openvino.tools.accuracy_checker` Python\* package yourself. For more information about metric implementation and dataset support, go to the [Accuracy Checker documentation](./tools/accuracy_checker/README.md).

There are steps to calibrate and evaluate result model:
1. Convert data annotation files.
2. (Optional) Estimate low precision model performance.
3. Calibrate the model.
4. Evaluate the resulting model.

Additional optional step before calibration is available to rough estimate possible INT8 performance.

### Convert Data Annotation Files
Calibration dataset is subset of training dataset. Use Convert Annotation Tool to convert ImageNet\* dataset to Calibration Tool readable data annotation files. Data annotation files describe subset of images which are used during calibration. Command line:
```sh
python convert_annotation.py imagenet --annotation_file /datasets/ImageNet/val.txt --labels_file /datasets/ImageNet/synset_words.txt -ss 2000 -o ~/annotations -a imagenet.pickle -m imagenet.json
```

> **NOTE:** For simplicity, all command line tools in the steps below use the same command line arguments. In practice [Collect Statistics Tool](./inference-engine/tools/collect_statistics_tool/README.md) uses calibration dataset, but [Accuracy Checker Tool](./tools/accuracy_checker/README.md) has to use the whole validation dataset.


| Argument           | Type   | Description                                                                       |
| -------------------| ------ | --------------------------------------------------------------------------------- |
| --config           | string | Path to the YML file with local configuration                                     |
| -d                 | string | Path to the YML file with definitions                                             |
| -M                 | string | Path to model optimizer directory                                                 |
| --models           | string | Prefix path to the models and weights                                             |
| --source           | string | Prefix path to the data source                                                    |
| --annotations      | string | Prefix path to the converted annotations and datasets meta data                    |
| --converted_models | string | Directory to store Model Optimizer converted models |


### (Optional) Estimate Low-Precision Model Performance

Before calibration, you can roughly estimate low precision performance with [Collect Statistics Tool](./inference-engine/tools/collect_statistics_tool/README.md).

[Collect Statistics Tool](./inference-engine/tools/collect_statistics_tool/README.md) ignores metric in YML configuration file but you can use the same command line arguments.

Command line:

```sh
python collect_statistics.py --config ~/inception_v1.yml -d ~/defenitions.yml -M /home/user/intel/openvino/deployment_tools/model_optimizer --models ~/models --source /media/user/calibration/datasets --annotations ~/annotations --converted_models ~/models
```

Result model has statistics which allow you to infer this model in INT8 precision. To measure performance, you can use the [Benchmark App](./inference-engine/tools/benchmark_tool/README.md).

### Calibrate the Model
During calibration process, the model is adjusted for efficient quantization and minimization of accuracy drop on calibration dataset. Calibration tool produces calibrated model which will be executed in low precision 8-bit quantized mode after loading into CPU plugin.

[Calibration Tool](./inference-engine/tools/calibration_tool/README.md) has flexible and extensible mechanism of enabling new data set and metrics. Each network has its own dedicated network metric and dataset where network was trained. Dataset description and network metrics can be reused for different network.

To plug new dataset you need to develop YML file. To develop new metric you need to develop Python\* module implementing metric and describe in YML. Please, refer to [Accuracy Checker Tool](./tools/accuracy_checker/README.md) for details.


Command line example:
```sh
python calibrate.py --config ~/inception_v1.yml --definition ~/defenitions.yml -M /home/user/intel/openvino/deployment_tools/model_optimizer --tf_custom_op_config_dir ~/tf_custom_op_configs --models ~/models --source /media/user/calibration/datasets --annotations ~/annotations
```

### Evaluate the Resulting Model 
After calibration of the model it worse to evaluate network accuracy on whole validation set using [Accuracy Checker Tool](./tools/accuracy_checker/README.md).

#### Check accuracy
Command line:
```sh
python accuracy_check.py --config ~/inception_v1.yml -d ~/defenitions.yml -M /home/user/intel/openvino/deployment_tools/model_optimizer --tf_custom_op_config_dir ~/tf_custom_op_configs --models ~/models --source /media/user/calibration/datasets --annotations ~/annotations -tf dlsdk -td CPU
```

#### Check performance
Use the [Benchmark App](./inference-engine/samples/benchmark_app/README.md) command line tool to measure latency and throughput for synchronous and asynchronous modes. Note, the Benchmark App command line tool uses converted OpenVINO* Intermediate Representation model.

Command line for synchronous mode:

```sh
./benchmark_app -i <path_to_image>/inputImage.bmp -m <path_to_model>/inception_v1.xml -d CPU -api sync
```

Command line for the asynchronous mode:
```sh
./benchmark_app -i <path_to_image>/inputImage.bmp -m <path_to_model>/inception_v1.xml -d CPU -api async
```

## Typical Workflow Samples (Simplified Mode)

To run the Calibration Tool in the simplified mode, use the following command:
```sh
python3 calibrate.py -sm -m <path-to-ir.xml> -s <path-to-dataset> -ss <images-number> -e <path-to-extensions-folder> -td <target-device> -precision <output-ir-precision> --output-dir <output-directory-path>
```
Input:
- FP32 and FP16 models
- image files as a dataset
