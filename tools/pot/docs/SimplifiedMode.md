# Optimization with Simplified mode {#pot_docs_simplified_mode}

## Introduction
Simplified mode is designed to simplify data preparation for the model optimization process. The mode is represented by an implementation of Engine interface from the POT API that allows reading data from an arbitrary folder specified by the user. For more details about POT API please refer to the corresponding [description](pot_compression_api_README). Currently, Simplified mode is available only for image data stored in a single folder in PNG or JPEG formats.

Note: This mode cannot be used with accuracy-aware methods, i.e. there is no way to control accuracy after optimization. Nevertheless, this mode can be helpful to estimate performance benefits when using model optimizations.

## Usage
To use Simplified mode you should prepare data and place them in a separate folder. No other files should be presented in this folder. There are two options to run POT in the Simplified mode:
* Using command-line options only. Here is an example for 8-bit quantization:
  
  `pot -q default -m <path_to_xml> -w <path_to_bin> --engine simplified --data-source <path_to_data>`
* To provide more options you can use the corresponding `"engine"` section in the POT configuration file as follows:
    ```json
    "engine": {
        "type": "simplified",
        "layout": "NCHW",               // Layout of input data. Supported ["NCHW",
                                        // "NHWC", "CHW", "CWH"] layout
        "data_source": "PATH_TO_SOURCE" // You can specify path to directory with images 
                                        // Also you can specify template for file names to filter images to load.
                                        // Templates are unix style (This option valid only in simplified mode)
    }
    ```


A template of configuration file for 8-bit quantization using Simplified mode can be found [here](https://github.com/openvinotoolkit/openvino/blob/master/tools/pot/configs/simplified_mode_template.json).

For more details about how to use POT via CLI please refer to this [document](@ref pot_compression_cli_README).

## See Also
 * [Configuration File Description](@ref pot_configs_README)