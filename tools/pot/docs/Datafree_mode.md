# Optimization with Data-free mode {#pot_docs_data_free_mode}

## Introduction
Data-free mode is designed to apply optimization when there is only a model and no data is available. This assumes the generation of a synthetic dataset that can take time in some cases. Once generated, the dataset is cached and can be reused in the following runs of optimization methods. The mode is represented by an implementation of Engine interface from the POT API. For more details about POT API please refer to the corresponding [description](pot_compression_api_README). Currently, Data-free mode is available only for Computer Vision models.

Note: there can be a significant deviation of model accuracy after optimization using Data-free mode. Nevertheless, this mode can be helpful to estimate performance benefits when using optimization.

## Usage
There are two options to run POT in the Data-free mode:
* Using command-line options only. Here is an example for 8-bit quantization:
  
  `pot -q default -m <path_to_xml> -w <path_to_bin> --engine data_free`
* To provide more options you can use the corresponding `"engine"` section in the POT configuration file as follows:
    ```json
    "engine": {
        "type": "data_free",                  // Engine type​
        "generate_data": "True",              // (Optional) If True, generate synthetic data and store to `data_source`​
                                              // Otherwise, the dataset from `--data-source` will be used'​
        "layout": "NCHW",                     // (Optional) Layout of input data. Supported: ["NCHW", "NHWC", "CHW", "CWH"]​
        "shape": "[None, None, None, None]",  // (Optional) if model has dynamic shapes, input shapes must be provided​
        "data_type": "image",                 // (Optional) You can specify the type of data to be generated.​
                                              // Currently only `image` is supported.​
                                              // It is planned to add 'text` and 'audio' cases​
        "data_source": "PATH_TO_SOURCE"       // (Optional) You can specify path to directory​
                                              // where synthetic dataset is located or will be generated and saved​
    },
    ```


A template of configuration file for 8-bit quantization using Data-free mode can be found [here](https://github.com/openvinotoolkit/openvino/blob/master/tools/pot/configs/data_free_mode_template.json).

For more details about how to use POT via CLI please refer to this [document](@ref pot_compression_cli_README).

## See Also
 * [Configuration File Description](@ref pot_configs_README)