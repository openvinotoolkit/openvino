# AccuracyAwareQuantization Parameters {#accuracy_aware_README}

## Introduction
Accuracy-aware Quantization algorithm is aimed at accurate quantization and allows the model's accuracy to stay within the 
pre-defined range. This may cause a degradation in performance in comparison to [Default Quantization](../default/README.md) algorithm because some layers can be reverted back to the original precision.

## Parameters
Since the [Default Quantization](../default/README.md) algorithm is used as an initialization, all its parameters are also valid and can be specified. Here is an example of the definition of the Accuracy-aware Quantization method and its parameters:
```json
{
    "name": "AccuracyAwareQuantization", // the name of optimization algorithm 
    "params": {
        ...
    }
}
```

Below are the descriptions of AccuracyAwareQuantization-specific parameters:
- `"ranking_subset_size"` - size of a subset that is used to rank layers by their contribution to the accuracy drop. 
Default value is `300`, and more samples it has the better ranking, potentially.
- `"max_iter_num"` - the maximum number of iterations of the algorithm. In other words, the maximum number of layers that may
 be reverted back to floating-point precision. By default, it is limited by the overall number of quantized layers.
- `"maximal_drop"` - the maximum accuracy drop which has to be achieved after the quantization. The default value is `0.01` (1%).
- `"drop_type"` - a drop type of the accuracy metric: 
    - `"absolute"` - the (default) absolute drop with respect to the results of the full-precision model.
    - `"relative"` - relative to the results of the full-precision model.
- `"use_prev_if_drop_increase"` - the use of network snapshot from the previous iteration when a drop 
increases. The default value is `True`.
- `"base_algorithm"` - name of the algorithm that is used to quantize a model at the beginning. The default value is 
    "DefaultQuantization".
- `"convert_to_mixed_preset"` - set to convert the model to "mixed" mode if the accuracy criteria for the model
 quantized with "performance" preset are not satisfied. This option can help to reduce number of layers that are reverted
 to floating-point precision. Keep in mind that this is an **experimental** feature.
- `"metrics"` - an optional list of metrics that are taken into account during optimization. It consists of tuples with the 
following parameters:
    - `"name"` - name of the metric to optimize.
    - `"baseline_value"` - (optional parameter) a baseline metric value of the original model. The validations on
    The validation will be initiated entirely in the beginning if nothing specified.
- `"metric_subset_ratio"` - a part of the validation set that is used to compare original full-precision and 
fully quantized models when creating a ranking subset in case of predefined metric values of the original model.
The default value is `0.5`.
- `"tune_hyperparams"` - enables tuning of quantization parameters as a preliminary step before reverting layers back
to the floating-point precision. It can bring an additional boost in performance and accuracy, at the cost of increased overall 
quantization time. The default value is `False`.

## Additional Resources

Example:
 * [Quantization of Object Detection model with control of accuracy](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/object_detection)

 A template and full specification for AccuracyAwareQuantization algorithm for POT command-line interface:
 * [Template](https://github.com/openvinotoolkit/openvino/blob/master/tools/pot/openvino/tools/pot/configs/templates/accuracy_aware_quantization_template.json)
 * [Full specification](https://github.com/openvinotoolkit/openvino/blob/master/tools/pot/configs/accuracy_aware_quantization_spec.json)

  @sphinxdirective

.. dropdown:: Template

   .. code-block:: javascript
      
        /* This configuration file is the fastest way to get started with the accuracy aware
        quantization algorithm. It contains only mandatory options with commonly used
        values. All other options can be considered as an advanced mode and requires
        deep knowledge of the quantization process. An overall description of all possible
        parameters can be found in the accuracy_aware_quantization_spec.json */

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
                        "name": "AccuracyAwareQuantization", // Optimization algorithm name
                        "params": {
                            "preset": "performance", // Preset [performance, mixed, accuracy] which control the quantization
                                                    // mode (symmetric, mixed (weights symmetric and activations asymmetric)
                                                    // and fully asymmetric respectively)

                            "stat_subset_size": 300, // Size of subset to calculate activations statistics that can be used
                                                    // for quantization parameters calculation

                            "maximal_drop": 0.01, // Maximum accuracy drop which has to be achieved after the quantization
                            "tune_hyperparams": false // Whether to search the best quantization parameters for model
                        }
                    }
                ]
            }
        }


@endsphinxdirective
