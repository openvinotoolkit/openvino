# TunableQuantization Algorithm

## Overview

TunableQuantization algorithm is a modified version of the vanilla **MinMaxQuantization** quantization method that automatically inserts [FakeQuantize](@ref openvino_docs_ops_quantization_FakeQuantize_1) operations into the model graph based on the specified target hardware and initializes them using statistics collected on the calibration dataset.
It is recommended to be run as a part of an optimization pipeline similar to the one used in [**DefaultQuantization**](../default/README.md):
*  ActivationChannelAlignment - Used as a preliminary step before quantization and allows you to align ranges of output activations of Convolutional layers in order to reduce the quantization error.
*  FastBiasCorrection - Adjusts biases of Convolutional and Fully-Connected layers based on the quantization error of the layer in order to make the overall error unbiased.

## Parameters
The recommended parameters sets for TunableQuantization are provided in [post-training optimization best practices](@ref pot_docs_BestPractices).
Here we present the complete reference of parameters with their low-level meaning.

The algorithm accepts the following parameters:
- `"preset"` - preset which controls the quantization mode (symmetric and asymmetric). It can take two values:
    - `"performance"` (default) - stands for symmetric quantization of weights and activations. This is the most 
    performant across all the hardware.
    - `"mixed"` - symmetric quantization of weights and asymmetric quantization of activations. This mode can be useful
    for quantization of neural networks which has both negative and positive input values in quantizing operations.  
- `"stat_subset_size"` - size of subset to calculate activations statistics used for quantization. The whole dataset 
is used if no parameter specified. We recommend using not less than 300 samples.

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
