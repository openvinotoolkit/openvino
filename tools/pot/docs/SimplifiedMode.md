# Optimization using Simplified mode {#pot_docs_simplified_mode}

## Introduction
Simplified mode is designed to simplify data preparation for model optimization process. The mode is represented by an implementation of Engine interface from the POT API that allows reading data from an arbitrary folder specified by user. For more details about POT API please refer to the corresponding [description](pot_compression_api_README). Currently, Simplified mode is available only for image data stored in a single folder in PGN or JPEG formats.

## Usage
To use Simplified mode you should prepare data and place them in a separate folder. No other files should be presented in this folder. To activate the mode in POT you should specify the corresponding Engine in the `"engine"` section as follows:
```json
"engine": {
        "type": "simplified",
        "data_source": "PATH_TO_FOLDER"
    }
```


A template of configuration file for 8-bit quantization using Simplified mode can be found [here](https://github.com/openvinotoolkit/openvino/blob/master/tools/pot/configs/simplified_mode_template.json).

## See Also
 * [Configuration File Description](@ref pot_configs_README)