# Optimization with Simplified Mode {#pot_docs_simplified_mode}

## Introduction

Simplified mode is designed to make data preparation for the model optimization process easier. The mode is represented by an implementation of Engine interface from the POT API. It allows reading the data from an arbitrary folder specified by the user. For more details about POT API, refer to the corresponding [description](@ref pot_compression_api_README). Currently, Simplified mode is available only for image data in PNG or JPEG formats, stored in a single folder. It supports Computer Vision models with a single input or two inputs where the second is "image_info" (Faster R-CNN, Mask R-CNN, etc.).

> **NOTE**: This mode cannot be used with accuracy-aware methods. There is no way to control accuracy after optimization. Nevertheless, this mode can be helpful to estimate performance benefits when using model optimizations.

## Usage

To use the Simplified mode, prepare the data and place it in a separate folder. No other files should be present in this folder. There are two options to run POT in the Simplified mode:

* Using command-line options only. Here is an example for 8-bit quantization:
  
  `pot -q default -m <path_to_xml> -w <path_to_bin> --engine simplified --data-source <path_to_data>`
* To provide more options, use the corresponding `"engine"` section in the POT configuration file as follows:
    ```json
    "engine": {
        "type": "simplified",
        "layout": "NCHW",               // Layout of input data. Supported ["NCHW",
                                        // "NHWC", "CHW", "CWH"] layout
        "data_source": "PATH_TO_SOURCE" // You can specify path to the directory with images 
                                        // Also you can specify template for file names to filter images to load.
                                        // Templates are unix style (this option is valid only in Simplified mode)
    }
    ```


A template of configuration file for 8-bit quantization using Simplified mode can be found [at the following link](https://github.com/openvinotoolkit/openvino/blob/master/tools/pot/configs/simplified_mode_template.json).

For more details about POT usage via CLI, refer to this [CLI document](@ref pot_compression_cli_README).

## Additional Resources

 * [Configuration File Description](@ref pot_configs_README)