# DefaultQuantization Parameters {#pot_compression_algorithms_quantization_default_README}

@sphinxdirective

The DefaultQuantization Algorithm is designed to perform fast and accurate quantization. It does not offer direct control over the accuracy metric itself but provides many options that can be used to improve it.

Parameters
####################

Default Quantization algorithm has mandatory and optional parameters. For more details on how to use these parameters, refer to :doc:`Best Practices <pot_docs_BestPractices>` document. Below is an example of the definition of Default Quantization method and its parameters:

.. code-block:: py
   :force:

   {
       "name": "DefaultQuantization", # the name of optimization algorithm
       "params": {
           ...
       }
   }


Mandatory parameters
++++++++++++++++++++

- ``"preset"`` - a preset that controls the quantization mode (symmetric and asymmetric). It can take two values:

  - ``"performance"`` (default) - stands for symmetric quantization of weights and activations. This is the most efficient across all the HW.
  - ``"mixed"`` - symmetric quantization of weights and asymmetric quantization of activations. This mode can be useful for the quantization of NN, which has both negative and positive input values in quantizing operations, for example, non-ReLU based CNN.

- ``"stat_subset_size"`` - size of a subset to calculate activations statistics used for quantization. The whole dataset is used if no parameter is specified. It is recommended to use not less than 300 samples.
- ``"stat_batch_size"`` - size of a batch to calculate activations statistics used for quantization. It has a value of 1 if no parameter is specified.

Optional parameters
+++++++++++++++++++

All other options should be considered as an advanced mode and require deep knowledge of the quantization process. Below
is an overall description of all possible parameters:

- ``"model type"`` - required for accurate optimization of some model architectures. Now, only ``"transformer"`` type is supported for Transformer-based models (BERT, etc.). Default value is `None`.
- ``"inplace_statistics"`` - used to change a method of statistics collection from in-place (in-graph operations) to external collectors that require more memory but can increase optimization time. Default value is `True`.
- ``"ignored"`` - NN subgraphs which should be excluded from the optimization process

  - ``"scope"`` - list of particular nodes to exclude
  - ``"operations"`` - list of operation types to exclude (expressed in OpenVINO IR notation). This list consists of the following tuples:

    - ``"type"`` - a type of ignored operation.
    - ``"attributes"`` - if attributes are defined, they will be considered during inference. They are defined by a dictionary of ``"<NAME>": "<VALUE>"`` pairs.

- ``"weights"`` - this section describes the quantization scheme for weights and the way to estimate the quantization range for that. It is worth noting that changing the quantization scheme may lead to the inability to infer such mode on the existing HW.

  - ``"bits"`` - bit-width, the default value is "8".
  - ``"mode"`` - a quantization mode (symmetric or asymmetric).
  - ``"level_low"`` - the minimum level in the integer range to quantize. The default is "0" for an unsigned range, and "-2^(bit-1)" for a signed one.
  - ``"level_high"`` - the maximum level in the integer range to quantize. The default is "2^bits-1" for an unsigned range, and "2^(bit-1)-1" for a signed one.
  - ``"granularity"`` - quantization scale granularity. It can take the following values:

    - ``"pertensor"`` (default) - per-tensor quantization with one scale factor and zero-point.
    - ``"perchannel"`` - per-channel quantization with per-channel scale factor and zero-point.

  - ``"range_estimator"`` - this section describes the parameters of the range estimator that is used in the MinMaxQuantization method to get the quantization ranges and filter outliers based on the collected statistics. Below are the parameters that can be modified to get better accuracy results:

    - ``"max"`` - parameters to estimate top border of quantizing floating-point range:

      - ``"type"`` - a type of the estimator:

        - ``"max"`` (default) - estimates the maximum in the quantizing set of value.
        - ``"quantile"`` - estimates the quantile in the quantizing set of value.

      - ``"outlier_prob"`` - outlier probability used in the "quantile" estimator.

    - ``"min"`` - parameters to estimate the bottom border of quantizing floating-point range:

      - ``"type"`` - a type of the estimator:

        - ``"min"`` (default) - estimates the minimum in the quantizing set of value.
        - ``"quantile"`` - estimates the quantile in the quantizing set of value.

      - ``"outlier_prob"`` - outlier probability used in the "quantile" estimator.

- ``"activations"`` - this section describes the quantization scheme for activations and the way to estimate the quantization range for that. As before, changing the quantization scheme may lead to the inability to infer such mode on the existing HW:

  - ``"bits"`` - bit-width, the default value is "8".
  - ``"mode"`` - a quantization mode (symmetric or asymmetric).
  - ``"level_low"`` - the minimum level in the integer range to quantize. The default is "0" for an unsigned range, and "-2^(bit-1)" for a signed one.
  - ``"level_high"`` - the maximum level in the integer range to quantize. The default is "2^bits-1" for an unsigned range, and "2^(bit-1)-1" for a signed one.
  - ``"granularity"`` - quantization scale granularity. It can take the following values:

    - ``"pertensor"`` (default) - per-tensor quantization with one scale factor and zero-point.
    - ``"perchannel"`` - per-channel quantization with per-channel scale factor and zero-point.

  - ``"range_estimator"`` - this section describes the parameters of the range estimator that is used in the MinMaxQuantization method to get the quantization ranges and filter outliers based on the collected statistics. These are the parameters that can be modified to get better accuracy results:

    - ``"preset"`` - preset that defines the same estimator for both top and bottom borders of quantizing floating-point range. Possible value is ``"quantile"``.
    - ``"max"`` - parameters to estimate top border of quantizing floating-point range:

      - ``"aggregator"`` - a type of function used to aggregate statistics obtained with the estimator over the calibration dataset to get a value of the top border:

        - ``"mean"`` (default) - aggregates mean value.
        - ``"max"`` - aggregates max value.
        - ``"min"`` - aggregates min value.
        - ``"median"`` - aggregates median value.
        - ``"mean_no_outliers"`` - aggregates mean value after removal of extreme quantiles.
        - ``"median_no_outliers"`` - aggregates median value after removal of extreme quantiles.
        - ``"hl_estimator"`` - Hodges-Lehmann filter based aggregator.

      - ``"type"`` - a type of the estimator:

        - ``"max"`` (default) - estimates the maximum in the quantizing set of value.
        - ``"quantile"`` - estimates the quantile in the quantizing set of value.

      - ``"outlier_prob"`` - outlier probability used in the "quantile" estimator.

    - ``"min"`` - parameters to estimate the bottom border of quantizing floating-point range:

      - ``"type"`` - a type of the estimator:

        - ``"max"`` (default) - estimates the maximum in the quantizing set of value.
        - ``"quantile"`` - estimates the quantile in the quantizing set of value.

      - ``"outlier_prob"`` - outlier probability used in the "quantile" estimator.

- ``"use_layerwise_tuning"`` - enables layer-wise fine-tuning of model parameters (biases, Convolution/MatMul weights, and FakeQuantize scales) by minimizing the mean squared error between original and quantized layer outputs. Enabling this option may increase compressed model accuracy, but will result in increased execution time and memory consumption.

Additional Resources
####################

Tutorials:

* `Quantization of Image Classification model <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/301-tensorflow-training-openvino>`__
* `Quantization of Object Detection model from Model Zoo <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/111-yolov5-quantization-migration>`__
* `Quantization of Segmentation model for medical data <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/110-ct-segmentation-quantize>`__
* `Quantization of BERT for Text Classification <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/105-language-quantize-bert>`__

Examples:

* :doc:`Quantization of 3D segmentation model <pot_example_3d_segmentation_README>`
* :doc:`Quantization of Face Detection model <pot_example_face_detection_README>`
* :doc:`Quantization of speech model for GNA device <pot_example_speech_README>`

Command-line example:

* :doc:`Quantization of Image Classification model <pot_configs_examples_README>`

A template and full specification for DefaultQuantization algorithm for POT command-line interface:

* `Template <https://github.com/openvinotoolkit/openvino/blob/master/tools/pot/openvino/tools/pot/configs/templates/default_quantization_template.json>`__
* `Full specification <https://github.com/openvinotoolkit/openvino/blob/master/tools/pot/configs/default_quantization_spec.json>`__


.. dropdown:: Template

   .. code-block:: javascript

        /* This configuration file is the fastest way to get started with the default
        quantization algorithm. It contains only mandatory options with commonly used
        values. All other options can be considered as an advanced mode and require
        deep knowledge of the quantization process. An overall description of all possible
        parameters can be found in the default_quantization_spec.json */

        {
            /* Model parameters */

            "model": {
                "model_name": "model_name", // Model name
                "model": "<MODEL_PATH>", // Path to model (.xml format)
                "weights": "<PATH_TO_WEIGHTS>" // Path to weights (.bin format)
            },

            /* Parameters of the engine used for model inference */

            "engine": {
                "config": "<CONFIG_PATH>" // Path to Accuracy Checker config
            },

            /* Optimization hyperparameters */

            "compression": {
                "target_device": "ANY", // Target device, the specificity of which will be taken
                                        // into account during optimization
                "algorithms": [
                    {
                        "name": "DefaultQuantization", // Optimization algorithm name
                        "params": {
                            "preset": "performance", // Preset [performance, mixed, accuracy] which control the quantization
                                                    // mode (symmetric, mixed (weights symmetric and activations asymmetric)
                                                    // and fully asymmetric respectively)

                            "stat_subset_size": 300  // Size of the subset to calculate activations statistics that can be used
                                                    // for quantization parameters calculation
                        }
                    }
                ]
            }
        }


@endsphinxdirective
