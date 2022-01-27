# Use Post-Training Optimization Tool Command-Line Interface {#pot_compression_cli_README}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   
   End-to-end CLI example <pot_configs_examples_README>

@endsphinxdirective

POT command-line interface (CLI) is designed to optimize models that are supported by the [Accuracy Checker Tool](@ref omz_tools_accuracy_checker) used for accuracy measurement. 
If your model is exactly from the OpenVINO&trade; [Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) or it is similar to one of 
its models then you can employ POT CLI to optimize your model.
In other cases, you should consider using POT [API](@ref pot_compression_api_README). To start with POT CLI please refer to the
following [example](@ref pot_configs_examples_README).

Note: There is also the so-called [**Simplified mode**](@ref pot_configs_README) that is basically aimed at INT8 quantization if the model is from the Computer Vision domain and has a simple dataset preprocessing, like image resize and crop. In this case, you can also use POT CLI for 
optimization. However, the accuracy results are not guaranteed in this case. Moreover, you are also limited in the 
optimization methods choice since the accuracy measurement is not available.
 


## Prerequisites
1. Install POT following the [Installation Guide](@ref pot_InstallationGuide).
2. Convert your model from the framework representation into the OpenVINO&trade; IR format with the 
[Model Optimizer](@ref openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide).
3. Prepare the Accuracy Checker configuration file and make sure that the model can be successfully inferred and achieves 
similar accuracy numbers as the reference model from the original framework. 
4. Activate the Python environment in the command-line shell where the POT and the Accuracy Checker were installed.
5. (Optional). Set up the OpenVINO&trade; environment in the command-line shell with the following script if you 
installed it from form the distribution file:
   ```sh
   source <INSTALL_DIR>/bin/setupvars.sh
   ```
   > **NOTE**: This step is not required if you use PyPI distribution.

## Run POT CLI 
There are two ways how to run POT via command line:

- **Basic usage**. In this case you can run POT with basic setting just specifying all the options via command line:
   ```sh
   pot -q default -m <path_to_xml> -w <path_to_bin> --ac-config <path_to_AC_config_yml>
   ```
- **Advanced usage**. In this case you should prepare a configuration file for the POT where you can specify advanced options for the optimization
methods available. See [POT configuration file description](@ref pot_configs_README) for more details. 
To launch the command-line tool with the configuration file run:
   ```sh
   pot -c <path_to_config_file>
   ```
For all available usage options, use the `-h`, `--help` arguments or refer to the Command-Line Arguments section below.  

By default, the results are dumped into the separate output subfolder inside the `./results` folder that is created 
in the same directory where the tool is run from. Use the `-e` option to evaluate the accuracy directly from the tool.

See also the [End-to-end example](@ref pot_configs_examples_README) about how to run a particular example of 8-bit
quantization with the POT.

### Command-Line Arguments

The following command-line options are available to run the tool: 

| Argument                                          | Description                                             |
| ------------------------------------------------- | ------------------------------------------------------- |
| `-h`, `--help`                                    | Optional. Show help message and exit. |
| `-q`, `--quantize`                                | Quantize model to 8 bits with specified quantization method: `default` or `accuracy_aware`. |
| `--preset`                                        | Use `performance` for fully symmetric quantization or `mixed` preset for symmetric quantization of weight and asymmetric quantization of activations. Applicable only when `-q` option is used.|
| `-m`, `--model`                                   | Path to the optimizing model file (.xml). Applicable only when `-q` option is used. |
| `-w`, `--weights`                                 | Path to the weights file of the optimizing model (.bin). Applicable only when `-q` option is used. |
| `-n`, `--name`                                    | Model name. Applicable only when `-q` option is used. |
| `--ac-config`                                     | Path to the Accuracy Checker configuration file. Applicable only when `-q` option is used. |
| `--max-drop`                                      | Optional. Maximum accuracy drop. Valid only for accuracy-aware quantization. Applicable only when `-q` option is used and `accuracy_aware` method is selected. |
| `-c CONFIG`, `--config CONFIG`                    | Path to a config file with task- or model-specific parameters.         |
| `-e`, `--evaluate`                                | Optional. Evaluate model on the whole dataset after optimization.  |
| `--output-dir OUTPUT_DIR`                         | Optional. A directory where results are saved. Default: `./results`. |
| `-sm`, `--save-model`                             | Optional. Save the original full-precision model. |
| `-d`, `--direct-dump`                             | Optional. Save results to the "optimized" subfolder within the specified output directory with no additional subpaths added at the end. |
| `--log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}` | Optional. Log level to print. Default: INFO. |
| `--progress-bar`                                  | Optional. Disable CL logging and enable progress bar. |
| `--stream-output`                                 | Optional. Switch model quantization progress display to a multiline mode. Use with third-party components. |
| `--keep-uncompressed-weights`                     | Optional. Keep Convolution, Deconvolution and FullyConnected weights uncompressed. Use with third-party components.|


## See Also

* [Installation Guide](@ref pot_InstallationGuide)
* [Post-Training Optimization Best Practices](@ref pot_docs_BestPractices)
