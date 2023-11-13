# Use Post-Training Optimization Tool Command-Line Interface (Model Zoo flow){#pot_compression_cli_README}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   Simplified Mode <pot_docs_simplified_mode>
   pot_configs_README



Introduction
####################

POT command-line interface (CLI) is aimed at optimizing models that are similar to the models from OpenVINO `Model Zoo <https://github.com/openvinotoolkit/open_model_zoo>`__ or if there is a valid :doc:`AccuracyChecker Tool <omz_tools_accuracy_checker>` configuration file for the model. Examples of AccuracyChecker configuration files can be found on `GitHub <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public>`__. Each model folder contains a YAML configuration file that can be used with POT as is.

.. note::

   There is also a :doc:`Simplified mode <pot_docs_simplified_mode>` aimed at the optimization of models from the Computer Vision domain and has a simple dataset preprocessing like image resize and crop. In this case, you can also use POT CLI for optimization. However, the accuracy results are not guaranteed in this case. Moreover, you are also limited in the optimization methods choice since the accuracy measurement is not available.


Run POT CLI
####################

There are two ways how to run POT via the command line:

- **Basic usage for DefaultQuantization**. In this case, you can run POT with basic settings just specifying all the options via the command line. ``-q default`` stands for :doc:`DefaultQuantization <pot_compression_algorithms_quantization_default_README>` algorithm:

  .. code-block:: sh

     pot -q default -m <path_to_xml> -w <path_to_bin> --ac-config <path_to_AC_config_yml>

- **Basic usage for AccuracyAwareQuantization**. You can also run :doc:`AccuracyAwareQuantization <accuracy_aware_README>` method with basic options. ``--max-drop 0.01`` option defines maximum accuracy deviation to 1 absolute percent from the original model:

  .. code-block:: sh

     pot -q accuracy_aware -m <path_to_xml> -w <path_to_bin> --ac-config <path_to_AC_config_yml> --max-drop 0.01


- **Advanced usage**. In this case, you should prepare a configuration file for the POT where you can specify advanced options for the optimization methods available. See :doc:`POT configuration file description <pot_configs_README>` for more details.

  To launch the command-line tool with the configuration file run:

  .. code-block:: sh

     pot -c <path_to_config_file>


For all available usage options, use the ``-h``, ``--help`` arguments or refer to the Command-Line Arguments section below.

By default, the results are dumped into the separate output subfolder inside the ``./results`` folder that is created 
in the same directory where the tool is run from. Use the ``-e`` option to evaluate the accuracy directly from the tool.

See also the :doc:`End-to-end example <pot_configs_examples_README>` about how to run a particular example of 8-bit
quantization with the POT.

Command-Line Arguments
++++++++++++++++++++++

The following command-line options are available to run the tool:

+-----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Argument                                            | Description                                                                                                                                                                                           |
+=====================================================+=======================================================================================================================================================================================================+
| ``-h``, ``--help``                                  | Optional. Show help message and exit.                                                                                                                                                                 |
+-----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``-q``, ``--quantize``                              | Quantize model to 8 bits with specified quantization method: ``default`` or ``accuracy_aware``.                                                                                                       |
+-----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``--preset``                                        | Use ``performance`` for fully symmetric quantization or ``mixed`` preset for symmetric quantization of weight and asymmetric quantization of activations. Applicable only when ``-q`` option is used. |
+-----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``-m``, ``--model``                                 | Path to the optimizing model file (.xml). Applicable only when ``-q`` option is used.                                                                                                                 |
+-----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``-w``, ``--weights``                               | Path to the weights file of the optimizing model (.bin). Applicable only when ``-q`` option is used.                                                                                                  |
+-----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``-n``, ``--name``                                  | Optional. Model name. Applicable only when ``-q`` option is used.                                                                                                                                     |
+-----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``--engine {accuracy_checker, simplified}``         | Engine type used to specify CLI mode. Default: ``accuracy_checker``.                                                                                                                                  |
+-----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``--data-source DATA_DIR``                          | Optional. Valid and required for Simplified mode only. Specifies the path to calibration data.                                                                                                        |
+-----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``--ac-config``                                     | Path to the Accuracy Checker configuration file. Applicable only when ``-q`` option is used.                                                                                                          |
+-----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``--max-drop``                                      | Optional. Maximum accuracy drop. Valid only for accuracy-aware quantization. Applicable only when ``-q`` option is used and the ``accuracy_aware`` method is selected.                                |
+-----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``-c CONFIG``, ``--config CONFIG``                  | Path to a config file with task- or model-specific parameters.                                                                                                                                        |
+-----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``-e``, ``--evaluate``                              | Optional. Evaluate the model on the whole dataset after optimization.                                                                                                                                 |
+-----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``--output-dir OUTPUT_DIR``                         | Optional. A directory where results are saved. Default: ``./results``.                                                                                                                                |
+-----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``-sm``, ``--save-model``                           | Optional. Save the original full-precision model.                                                                                                                                                     |
+-----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``-d``, ``--direct-dump``                           | Optional. Save results to the "optimized" subfolder within the specified output directory with no additional subpaths added at the end.                                                               |
+-----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``--log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}`` | Optional. Log level to print. Default: INFO.                                                                                                                                                          |
+-----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``--progress-bar``                                  | Optional. Disable CL logging and enable the progress bar.                                                                                                                                             |
+-----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``--stream-output``                                 | Optional. Switch model quantization progress display to a multiline mode. Use with third-party components.                                                                                            |
+-----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``--keep-uncompressed-weights``                     | Optional. Keep Convolution, Deconvolution and FullyConnected weights uncompressed. Use with third-party components.                                                                                   |
+-----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


See Also
####################

* :doc:`Optimization with Simplified mode <pot_docs_simplified_mode>`
* :doc:`Post-Training Optimization Best Practices <pot_docs_BestPractices>`

@endsphinxdirective
