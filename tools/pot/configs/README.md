# Configuration File Description {#pot_configs_README}

In the instructions below, the Post-training Optimization Tool directory `<INSTALL_DIR>/deployment_tools/tools/post_training_optimization_toolkit` is referred to as `<POT_DIR>`. `<INSTALL_DIR>` is the directory where Intel&reg; Distribution of OpenVINO&trade; toolkit is installed.
> **NOTE**: Installation directory is different in the case of PyPI installation and does not contain examples of 
> configuration files.   

The tool is designed to work with the configuration file where all the parameters required for the optimization are specified. These parameters are organized as a dictionary and stored in
a JSON file. JSON file allows using comments that are supported by the `jstyleson` Python* package.
Logically all parameters are divided into three groups:
- **Model parameters** that are related to the model definition (e.g. model name, model path, etc.)
- **Engine parameters** that define parameters of the engine which is responsible for the model inference and data preparation used for optimization and evaluation (e.g. preprocessing parameters, dataset path, etc.)
- **Compression parameters** that are related to the optimization algorithm (e.g. algorithm name and specific parameters)

## Model Parameters

```json
"model": {
        "model_name": "model_name",
        "model": "<MODEL_PATH>",
        "weights": "<PATH_TO_WEIGHTS>"
    }
```

This section contains only three parameters:
- `"model_name"` - string parameter that defines a model name, e.g. `"MobileNetV2"`
- `"model"` - string parameter that defines the path to an input model topology (.xml)
- `"weights"` - string parameter that defines the path to an input model weights (.bin)

## Engine Parameters

```json
"engine": {
        "type": "accuracy_checker",
        "config": "./configs/examples/accuracy_checker/mobilenet_v2.yaml"
    }
```
The main parameter is `"type"` which can take two possible options: `"accuracy_checher"` (default) and `"simplified"`,
which specify the engine that is used for model inference and validation (if supported):
- **Simplified mode** engine. This engine can be used only with `DefaultQuantization` algorithm to get fully quantized model 
using a subset of images. It does not use the Accuracy Checker tool and annotation. To measure accuracy, you should implement 
your own validation pipeline with OpenVINO API.  
  - To run the simplified mode, define engine section similar to the example `mobilenetV2_tf_int8_simple_mode.json` file from the `<POT_DIR>/configs/examples/quantization/classification/` directory.
- **Accuracy Checker** engine. It relies on the [Deep Learning Accuracy Validation Framework](@ref omz_tools_accuracy_checker) (Accuracy Checker) when inferencing DL models and working with datasets.
The benefit of this mode is you can compute accuracy in case you have annotations. It is possible to use accuracy aware
algorithms family when this mode is selected.
There are two options to define engine parameters in that mode:
  - Refer to the existing Accuracy Checker configuration file which is represented by the YAML file. It can be a file used for full-precision model validation. In this case, you should define only the `"config"` parameter containing a path to the AccuracyChecker configuration file.
  - Define all the [required Accuracy Checker parameters](@ref omz_tools_accuracy_checker_dlsdk_launcher)
    directly in the JSON file. In this case, POT just passes the corresponding dictionary of parameters to the Accuracy Checker when instantiating it.
    For more details, refer to the corresponding Accuracy Checker information and examples of configuration files provided with the tool:
    - For the SSD-MobileNet model:<br>`<POT_DIR>/configs/examples/quantization/object_detection/ssd_mobilenetv1_int8.json`

## Compression Parameters

This section defines optimization algorithms and their parameters. For more details about parameters of the concrete optimization algorithm, please refer to the corresponding
[documentation](@ref pot_compression_algorithms_quantization_README).

## Examples of the Configuration File

For a quick start, many examples of configuration files are provided and placed to the `<POT_DIR>/configs/examples`
 folder. There you can find ready-to-use configurations for the models from various domains: Computer Vision (Image 
 Classification, Object Detection, Segmentation), Natural Language Processing, Recommendation Systems. We basically 
 put configuration files for the models which require non-default configuration settings in order to get accurate results.
For details on how to run the Post-Training Optimization Tool with a sample configuration file, see the [instructions](@ref pot_configs_examples_README).
