# Python* Calibration Tool

The Python* Calibration Tool calibrates a given FP32 model so that you can run calibrated model in low-precision 8-bit integer mode while keeping the input data of this model in the original precision.
The Calibration Tool is a Python\* command-line tool, which imports Python types from the `openvino.tools.calibration` package.

> **NOTE**: INT8 models are currently supported only by the CPU plugin. For the full list of supported configurations, see the [Supported Devices](./docs/IE_DG/supported_plugins/Supported_Devices.md) topic.

## Hardware requirements
Hardware requirements depend on a model. Typically for public models RAM memory size has to be not less then 16Gb, drive has to have not less then 30 GB free space independently on operation system. Temporary directory is used to cache layers output during calibration.

## Usage
The Calibration Tool is configured in the same way as the Accuracy Checker. You can also use additional command-line arguments to define calibration-specific parameters.

### Command-Line Arguments for the Accuracy Checker Tool reused in Calibration Tool
| Argument                                     | Type   | Description                                             |
| -------------------------------------------- | ------ | ------------------------------------------------------- |
| -c, --config                                 | string | Required. Path to the YML file with local configuration |
| -d, --definitions                            | string | Optional. Path to the YML file with definitions         |
| -m, --models                                 | string | Optional. Prefix path to the models and weights         |
| -s, --source                                 | string | Optional. Prefix path to the data source                |
| -a, --annotations                            | string | Optional. Prefix path to the converted annotations and datasets meta data |
| -e, --extensions                             | string | Optional. Prefix path to extensions folder              |
| --cpu_extensions_mode, --cpu-extensions-mode | string | Optional. specified preferable set of processor instruction for automatic searching the CPU extension lib: `avx2` or `sse4` |
| -C, --converted_models, --converted-models   | string | Optional. Directory to store Model Optimizer converted models. Used for DLSDK launcher only |
| -M, --model_optimizer, --model-optimizer     | string | Optional. Path to model optimizer Caffe* directory       |
| --tf_custom_op_config_dir, --tf-custom-op-config-dir | string | Optional. Path to directory with TensorFlow* custom operation configuration files for model optimizer |
| --tf_obj_detection_api_pipeline_config_path, --tf-obj-detection-api-pipeline-config-path | string | Optional. Path to directory with TensorFlow object detection API pipeline configuration files for the Model Optimizer |
| --progress                                   | string | Optional. Progress reporter: `bar`, `print` or `None`   |
| -td, --target_devices, --target-devices      | string | Optional. Space-separated list of devices for infer     |
| -tt, --target_tags, --target-tags | string   | Optional. Space-separated list of launcher tags for infer        |

### Specific Command Line Arguments for Calibration Tool
| Argument                          | Type   | Description                                               |
| --------------------------------- | ------ | --------------------------------------------------------- |
| -p, --precision                   | string | Optional. Precision to calibrate. Default value is INT8   |
| --ignore_layer_types, --ignore-layer-types | string | Optional. Layer types list which will be skipped during quantization |
| --ignore_layer_types_path, --ignore-layer-types-path | string | Optional. Ignore layer types file path |
| --ignore_layer_names, --ignore-layer-names | string | Optional. Layer names list which will be skipped during quantization |
| --ignore_layer_names_path, --ignore-layer-names-path | string | Optional. Ignore layer names file path |
| --batch_size, --batch-size        | integer| Optional. Batch size value. If not specified, the batch size value is determined from IR |
| -th, --threshold                  | float | Optional. Accuracy drop of quantized model should not exceed this threshold. Should be pointer in percents without percent sign. (1% is default) |
| -ic, --benchmark_iterations_count, --benchmark-iterations-count | integer | Optional. Benchmark iterations count (1000 is default). |
| -mn, --metric_name, --metric-name | string | Optional. Metric name used during calibration |
| -mt, --metric_type, --metric-type | string | Optional. Metric type used during calibration |
| -o, --output_dir, --output-dir    | string | Optional. Directory to store converted models. Original model directory is used if not defined |

## Model Calibration Flow

### Introduction
The calibration tool read original FP32 model, calibration dataset and create low precision model. Low precision model has two differences from original model:
1. Per channel statistics are defined. Statistics have minimum and maximum values for each layer and each channel. Model statistics are stored in Inference Engine intermediate representation file (IR) in XML format.
2. `quantization_level` layer attribute is defined. The attribute defines precision which is used during inference.

### Prerequisites
* Model: Tensorflow\* Inception v1. You can download the model from here: https://github.com/tensorflow/models/tree/master/research/slim
* Dataset: ImageNet. You can download ImageNet from here: http://www.image-net.org/download.php
* YML configuration files: you can find YML configuration files and YML definition file which are used below in `configs` directory:
  - `definitions.yml` - definition file
  - `inception_v1.yml` - configuration file for Tensorflow\* Inception v1 model
  - `ncf_config.yml` - configuration file for NCF model in OpenVINO\* Inference Engine Intermediate Representation format
  - `ssd_mobilenet_v1_coco.yml` - configuration file for Tensorflow\* SSD Mobilenet v1 model
  - `unet2d.yml` - configuration file for Unet2D mode in in OpenVINO\* Inference Engine Intermediate Representation format

If you have custom topology with not supported accuracy metric or not suported custom dataset then you should add some components implementation in `openvino.tools.accuracy_checker` Python\* package yourself. Refer to `openvino.tools.accuracy_checker` documentation how to implement metric and dataset support. 

There are steps to calibrate and evaluate result model:
- Step #1. Convert data annotation files
- Optional step for low precision model performance estimation.
- Step #2. Calibration
- Step #3. Result model evaluation

Additional optional step before calibration is available to rough estimate possible INT8 performance.

### Step #1. Convert Data Annotation Files
Calibration dataset is subset of training dataset. Use Convert Annotation Tool to convert ImageNet\* dataset to Calibration Tool readable data annotation files. Data annotation files describe subset of images which are used during calibration. Command line:
```sh
python convert_annotation.py imagenet --annotation_file /datasets/ImageNet/val.txt --labels_file /datasets/ImageNet/synset_words.txt -ss 2000 -o ~/annotations -a imagenet.pickle -m imagenet.json
```

> **NOTE:** For simplicity all command line tools in below steps use the same command line arguments. In practice [Collect Statistics Tool](./inference-engine/tools/collect_statistics_tool/README.md) uses calibration dataset, but [Accuracy Checker Tool](./inference-engine/tools/accuracy_checker_tool/README.md) has to use whole validation dataset.


| Argument           | Type   | Description                                                                       |
| -------------------| ------ | --------------------------------------------------------------------------------- |
| --config           | string | Path to the YML file with local configuration                                     |
| -d                 | string | Path to the YML file with definitions                                             |
| -M                 | string | Path to model optimizer directory                                                 |
| --models           | string | Prefix path to the models and weights                                             |
| --source           | string | Prefix path to the data source                                                    |
| --annotations      | string | Pefix path to the converted annotations and datasets meta data                    |
| --converted_models | string | Directory to store Model Optimizer converted models. Used for DLSDK launcher only |


### Optional Step for Low Precision Model Performance Estimation

Before calibration, you can roughly estimate low precision performance with [Collect Statistics Tool](./inference-engine/tools/collect_statistics_tool/README.md).

[Collect Statistics Tool](./inference-engine/tools/collect_statistics_tool/README.md) ignores metric in YML configuration file but you can use the same command line arguments.

Command line:

```sh
python collect_statistics.py --config ~/inception_v1.yml -d ~/defenitions.yml -M /home/user/intel/openvino/deployment_tools/model_optimizer --models ~/models --source /media/user/calibration/datasets --annotations ~/annotations --converted_models ~/models
```

Result model has statistics which allow you to infer this model in INT8 precision. To measure performance you can use [Benchmark Tool](./inference-engine/tools/benchmark_tool/README.md).

### Step #2. Calibration
During calibration process, the model is ajusted for efficient quantization and minimization of accuracy drop on calibration dataset. Calibration tool produces calibrated model which will be executed in low precision 8 bit quantzed mode after loading into CPU plugin.

[Calibration Tool](./inference-engine/tools/calibration_tool/README.md) has flexible and extensible mechanism of enabling new data set and metrics. Each network has its own dedicated network metric and dataset where network was trained. Dataset description and network metrics can be reused for different network.

To plug new dataset you need to develop YML file. To develop new metric you need to develop Python\* module implementing metric and describe in YML. Please, refer to [Accuracy Checker Tool](./inference-engine/tools/accuracy_checker_tool/README.md) for details.


Command line example:
```sh
python calibrate.py --config ~/inception_v1.yml --definition ~/defenitions.yml -M /home/user/intel/openvino/deployment_tools/model_optimizer --tf_custom_op_config_dir ~/tf_custom_op_configs --models ~/models --source /media/user/calibration/datasets --annotations ~/annotations
```

### Step #3. Result model evaluation
After calibration of the model it worse to evaluate network accuracy on whole validation set using [Accuracy Checker Tool](./inference-engine/tools/accuracy_checker_tool/README.md).

#### Step #3.1 Check accuracy
Command line:
```sh
python accuracy_check.py --config ~/inception_v1.yml -d ~/defenitions.yml -M /home/user/intel/openvino/deployment_tools/model_optimizer --tf_custom_op_config_dir ~/tf_custom_op_configs --models ~/models --source /media/user/calibration/datasets --annotations ~/annotations -tf dlsdk -td CPU
```

#### Step #3.2 Check performance
Use `benchmark_app` command line tool to measure latency and throughput for synchronous and asynchronous modes. Note, please, `benchmark_app` command line tool uses converted OpenVINO\* Intermediate Representation model.

Command line for synchronous mode:

```sh
./benchmark_app -i <path_to_image>/inputImage.bmp -m <path_to_model>/inception_v1.xml -d CPU -api sync
```

Command line for the asynchronous mode:
```sh
./benchmark_app -i <path_to_image>/inputImage.bmp -m <path_to_model>/inception_v1.xml -d CPU -api async
```

#### Optional step to check performance
You can use Python\* [Benchmark Tool](./inference-engine/tools/benchmark_tool/README.md) command line tool to quickly check performance with the same command line arguments and configuration YML files as for [Calibration Tool](./inference-engine/tools/calibration_tool/README.md).

Command line:
```sh
python benchmark.py --config ~/inception_v1.yml -d ~/defenitions.yml -M /home/user/intel/openvino/deployment_tools/model_optimizer --tf_custom_op_config_dir ~/tf_custom_op_configs --models ~/models --source /media/user/calibration/datasets --annotations ~/annotations --converted_models ~/models
```

