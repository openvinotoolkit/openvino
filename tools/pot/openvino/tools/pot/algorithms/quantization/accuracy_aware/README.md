# AccuracyAwareQuantization Algorithm {#pot_compression_algorithms_quantization_accuracy_aware_README}

## Overview
AccuracyAware algorithm is designed to perform accurate 8-bit quantization and allows the model to stay in the 
pre-defined range of accuracy drop, for example 1%, defined by the user in the configuration file. This may cause a 
degradation in performance in comparison to [DefaultQuantization](../default/README.md) algorithm because some layers can be reverted back to the original precision.
Generally, the algorithm consists of the following steps:
1. The model gets fully quantized using the DefaultQuantization algorithm.
2. The quantized and full-precision models are compared on a subset of the validation set in order to find mismatches in the target accuracy metric. A ranking subset is extracted based on the mismatches.
3. Optionally, if the accuracy criteria cannot be satisfied with fully symmetric quantization, the quantized model gets converted to mixed mode, and step 2 is repeated.
4. A layer-wise ranking is performed in order to get a contribution of each quantized layer into the accuracy drop. To
get this ranking we revert every layer (one-by-one) back to floating-point precision and measure how it affects accuracy. 
5. Based on the ranking, the most "problematic" layer is reverted back to the original precision. This change is followed by the evaluation of the obtained model on the full validation set in order to get a new accuracy drop.
6. If the accuracy criteria are satisfied for all pre-defined accuracy metrics defined in the configuration file,
 the algorithm finishes. Otherwise, it continues reverting the next "problematic" layer.
7. It may happen that regular reverting does not get any accuracy improvement or even worsen the accuracy. Then the 
re-ranking is triggered as it is described in step 4. However, it is possible to specify the maximum number of reverting
layers using a special parameter. Moreover, the algorithm saves intermediate results (models) that can be used at any time 
without a need to wait until it finishes.

The figure below shows the diagram of the algorithm.

![](../../../../docs/images/aa_quantization_pipeline.png)

## Parameters
Since the [DefaultQuantization](../default/README.md) algorithm is used as an initialization, all its parameters are also valid and can be specified. Here we
describe only AccuracyAware specific parameters:
- `"ranking_subset_size"` - size of a subset that is used to rank layers by their contribution to the accuracy drop. 
Default value is `300`. The more samples it has the better ranking you have, potentially.
- `"max_iter_num"` - maximum number of iterations of the algorithm, in other words maximum number of layers that may
 be reverted back to floating-point precision. By default it is limited by the overall number of quantized layers.
- `"maximal_drop"` - maximum accuracy drop which has to be achieved after the quantization. Default value is `0.01` (1%).
- `"drop_type"` - drop type of the accuracy metric: 
    - `"absolute"` (default) - absolute drop with respect to the results of the full-precision model
    - `"relative"` - relative to the results of the full-precision model
- `"use_prev_if_drop_increase"` - whether to use network snapshot from the previous iteration of in case if drop 
increases. Default value is `True`.
- `"base_algorithm"` - name of the algorithm that is used to quantize model at the beginning. Default value is 
    "DefaultQuantization".
- `"convert_to_mixed_preset"` - whether to convert the model to "mixed" mode if the accuracy criteria for the model
 quantized with "performance" preset are not satisfied. This option can help to reduce number of layers that are reverted
 to floating-point precision. Note: this is an experimental feature.
- `"metrics"` - optional list of metrics that are taken into account during optimization. It consists of tuples with the 
following parameters:
    - `"name"` - name of the metric to optimize
    - `"baseline_value"` - baseline metric value of the original model. This is the optional parameter. The validations on
    the whole validation will be initiated in the beginning if nothing specified.
- `"metric_subset_ratio"` -  part of the validation set that is used to compare original full-precision and 
fully quantized models when creating ranking subset in case of predefined metric values of the original model.
Default value is `0.5`.
- `"tune_hyperparams"` - enables quantization parameters tuning as a preliminary step before reverting layers back
to the floating-point precision. It can bring additional performance and accuracy boost but increase overall 
quantization time. Default value is `False`.

 Below is a fragment of the configuration file that shows overall structure of parameters for this algorithm.

```
"name": "AccuracyAwareQuantization", // compression algorithm name
    "params": {
        "ranking_subset_size": 300, // A size of a subset which is used to rank layers by their contribution to the accuracy drop
        "max_iter_num": 30,    // Maximum number of iterations of the algorithm (maximum of layers that may be reverted back to full-precision)
        "maximal_drop": 0.005,      // Maximum accuracy drop which has to be achieved after the quantization
        "drop_type": "absolute",    // Drop type of the accuracy metric: relative or absolute (default)
        "use_prev_if_drop_increase": false,      // Whether to use NN snapshot from the previous algorithm iteration in case if drop increases
        "base_algorithm": "DefaultQuantization", // Base algorithm that is used to quantize model at the beginning
        "convert_to_mixed_preset": false,  // Whether to convert the model to mixed mode if the accuracy criteria 
                                           // of the symmetrically quantized model are not satisfied
        "metrics": [                    // An optional list of metrics that are taken into account during optimization
                                        // If not specified, all metrics defined in engine config are used
            {
                "name": "accuracy",     // Metric name to optimize
                "baseline_value": 0.72  // Baseline metric value of the original model
            }
        ],
        "metric_subset_ratio": 0.5  // A part of the validation set that is used to compare element-wise full-precision and 
                                    // quantized models in case of predefined metric values of the original model
    }
```
