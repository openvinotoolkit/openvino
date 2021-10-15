# TunableQuantization Algorithm {#pot_compression_algorithms_quantization_tunable_quantization_README}

## Overview

TunableQuantization algorithm is a modified version (to support hyperparameters setting by [Tree-Structured Parzen Estimator (TPE)](../../../optimization/tpe/README.md)) of the vanilla **MinMaxQuantization** quantization method that automatically inserts [FakeQuantize](https://docs.openvinotoolkit.org/latest/_docs_ops_quantization_FakeQuantize_1.html) operations into the model graph based on the specified target hardware and initializes them using statistics collected on the calibration dataset.
It is recommended to be run as a part of an optimization pipeline similar to the one used in [**DefaultQuantization**](../default/README.md):
*  ActivationChannelAlignment - Used as a preliminary step before quantization and allows you to align ranges of output activations of Convolutional layers in order to reduce the quantization error.
*  TunableQuantization - Used instead of **MinMaxQuantization** to allow tuning using [Tree-Structured Parzen Estimator (TPE)](@ref pot_compression_optimization_tpe_README). 
*  FastBiasCorrection - Adjusts biases of Convolutional and Fully-Connected layers based on the quantization error of the layer in order to make the overall error unbiased.

## Parameters
The recommended parameters sets for TunableQuantization are provided in [post-training optimization best practices](@ref pot_docs_BestPractices).
Here we present the complete reference of parameters with their low-level meaning.
It is intended for advanced users who would like to experiment with new [Tree-Structured Parzen Estimator (TPE)](../../../optimization/tpe/README.md)'s search spaces. 

The algorithm accepts the following parameters:
- `"preset"` - preset which controls the quantization mode (symmetric and asymmetric). It can take two values:
    - `"performance"` (default) - stands for symmetric quantization of weights and activations. This is the most 
    performant across all the hardware.
    - `"mixed"` - symmetric quantization of weights and asymmetric quantization of activations. This mode can be useful
    for quantization of neural networks which has both negative and positive input values in quantizing operations.  
- `"stat_subset_size"` - size of subset to calculate activations statistics used for quantization. The whole dataset 
is used if no parameter specified. We recommend using not less than 300 samples.
- `"tuning_scope"` determines which quantization configurations will be returned to [Tree-Structured Parzen Estimator (TPE)](../../../optimization/tpe/README.md) as viable options and can be a list of any combination of the following values:
  - `"bits"` - layer-wise choice of precision (e.g. INT8, INT4) if hardware supports
  - `"mode"` - layer-wise choice of symmetric or asymmetric quantization mode
  - `"range_estimator"` - layer-wise choice of configuration of the algorithm used to estimate min/max FP32 values for the layer
  - `"layer"` - adds to the possible quantization configurations option that specific layer will not be quantized
- `"estimator_tuning_scope"` determines which parameters of the FP32 range estimator will be tuned. This parameter is only necessary when `"range_estimator"` is part of `"tuning_scope"`. Value is a list of any combination of the following values:
  - `"preset"` - choice between two presets: default (using min/max functions) and quantile (removes outlier values out of FP32 range)
  - `"type"` - choice similar to preset, but a bit more granular i.e. separate configuration of min and max functions for every layer
  - `"aggregator"` - choice of function used to aggregate min/max values from all input data samples
  - `"outlier_prob"` - enables tuning of outlier probability value for quantile configurations – outlier probability specify what fraction of FP32 values in input will be considered as out of range and will get saturated min/max value
- `"outlier_prob_choices"` - list of `"outlier_prob"` values to use when tuning `"outlier_prob"` parameter. This parameter is only necessary when `"outlier_prob"` is part of `"estimator_tuning_scope"`.

List of quantization configurations that will be returned to [Tree-Structured Parzen Estimator (TPE)](../../../optimization/tpe/README.md) as viable options is done through derivation process. This derivation is done by creating a list of all available quantization configurations supported by target hardware and then filtering it using base configuration (either from `"preset"` or previous best result) and `"tuning_scope"`. Filtering is done by choosing from all available options only those that differ from base configuration only on values of variables specified in `"tuning_scope"`.

The selection of whether to use `"preset"` or previous best result as base configuration depends on [Tree-Structured Parzen Estimator (TPE)](../../../optimization/tpe/README.md)'s `"trials_load_method"`:
  - `"cold_start - preset"` determines base quantization configuration,
  - `"fine_tune - preset"` option is ignored and quantization configuration used to achieve the best result in previous run is used as base quantization configuration.

Below is a fragment of the configuration file that shows overall structure of parameters for this algorithm.

```
"name": "TunableQuantization",
"params": {
    /* Preset is a collection of optimization algorithm parameters that will specify to the algorithm
    to improve which metric the algorithm needs to concentrate. Each optimization algorithm supports
    [performance, mixed, accuracy] presets which control the quantization mode (symmetric, mixed(weights symmetric and activations asymmetric), and fully asymmetric respectively)*/
    "preset": "performance",
    "stat_subset_size": 300,   // Size of subset to calculate activations statistics that can be used
                               // for quantization parameters calculation.
    "tuning_scope": ["layer"], // List of quantization parameters that will be tuned,
                               // available options: [bits, mode, granularity, layer, range_estimator]
    "estimator_tuning_scope": ["preset", "aggregator", "type", "outlier_prob"], // List of range_estimator parameters that will be tuned,
                                                                                // available options: [preset, aggregator, type, outlier_prob]
    "outlier_prob_choices": [1e-3, 1e-4, 1e-5] // List of outlier_prob values to use when tuning outlier_prob parameter
}
```
