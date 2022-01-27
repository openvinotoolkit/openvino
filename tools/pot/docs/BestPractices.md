#  Post-Training Optimization Best Practices {#pot_docs_BestPractices}
This document describes the most common insights about model optimization using the Post-training Optimization Tool (POT). The post-training optimization usually is 
the fastest and easiest way to get a low-precision model because it does not require model fine-tuning and thus, there is no need in the training dataset, pipeline and availability of
the powerful training hardware. In some cases, it may lead to not satisfactory accuracy drop, especially when optimizing the whole model.
However, it can be still helpful for fast performance evaluation in order to understand the possible speed up 
 when applying one or another optimization method. Before going into details
we suggest reading the following [POT documentation](../README.md).

> **NOTE**: POT uses inference on the CPU during model optimization. It means the ability to infer the original
> floating-point model is a prerequisite for model optimization. 
> It is also worth mentioning that in the case of 8-bit quantization it is recommended to run POT on the same CPU
> architecture when optimizing for CPU or VNNI-based CPU when quantizing for a non-CPU device, such as GPU, VPU, or GNA.
> It should help to avoid the impact of the saturation issue that occurs on AVX and SSE based CPU devices. 

## Get Started with Post-Training Quantization

Post-training quantization is a basic feature of the POT and it has lots of knobs that can be used to get an accurate 
quantized model. However, as a starting point we suggest using the `DefaultQuantization` algorithm with default settings.
In many cases it leads to satisfied accuracy and performance speedup. 

A fragment of the configuration file (`config/default_quantization_template.json` in the POT directory) with default settings is shown below:
```
"compression": {
  "target_device": "ANY", // Target device, the specificity of which will be taken into account during optimization.
                          // The default value "ANY" stands for compatible quantization supported by any HW.
  "algorithms": [
    {
      "name": "DefaultQuantization", // Optimization algorithm name
      "params": {
        "preset": "performance", // Preset [performance, mixed] which control the quantization
                                 // mode (symmetric, mixed (weights symmetric and activations asymmetric)
                                 // and fully asymmetric respectively)

        "stat_subset_size": 300  // Size of subset to calculate activations statistics that can be used
                                 // for quantization parameters calculation
      }
    }
  ]
}
```

In the case of substantial accuracy degradation after applying the `DefaultQuantization` algorithm there are two alternatives to use:
1.  Hyperparameters tuning
2.  AccuracyAwareQuantization algorithm

## Tuning Hyperparameters of the DefaultQuantization
The `DefaultQuantization` algorithm provides multiple hyperparameters which can be used in order to improve accuracy results for the fully-quantized model. 
Below is a list of best practices which can be applied to improve accuracy without a substantial performance reduction with respect to default settings:
1.  The first option that we recommend is to change is `preset` from `performance` to `mixed`. This enables asymmetric quantization of 
activations and can be helpful for NNs with non-ReLU activation functions, e.g. YOLO, EfficientNet, etc.
2.  The next option is `use_fast_bias`. Setting this option to `false` enables a different bias correction method which is more accurate, in general,
and applied after model quantization as a part of the `DefaultQuantization` algorithm.
   > **NOTE**: Changing this option can substantially increase quantization time in the POT tool.
3.  Another important option is a `range_estimator`. It defines how to calculate the minimum and maximum of quantization range for weights and activations.
For example, the following `range_estimator` for activations can improve the accuracy for Faster R-CNN based networks:
```
"compression": {
    "target_device": "ANY",        
    "algorithms": [
        {
            "name": "DefaultQuantization", 
            "params": {
                "preset": "performance", 
                "stat_subset_size": 300  
                                         

                "activations": {
                    "range_estimator": {
                        "max": {
                            "aggregator": "max",
                            "type": "abs_max"
                        }
                    }
                }
            }
        }
    ]
}
```

Find the possible options and their description in the `config/default_quantization_spec.json` file in the POT directory.

4.  The next option is `stat_subset_size`. It controls the size of the calibration dataset used by POT to collect statistics for quantization parameters initialization.
It is assumed that this dataset should contain a sufficient number of representative samples. Thus, varying this parameter may affect accuracy (higher is better). 
However, we empirically found that 300 samples are sufficient to get representative statistics in most cases.
5.  The last option is `ignored_scope`. It allows excluding some layers from the quantization process, i.e. their inputs will not be quantized. It may be helpful for some patterns for which it is known in advance that they drop accuracy when executing in low-precision.
For example, `DetectionOutput` layer of SSD model expressed as a subgraph should not be quantized to preserve the accuracy of Object Detection models.
One of the sources for the ignored scope can be the AccuracyAware algorithm which can revert layers back to the original precision (see details below).

## AccuracyAwareQuantization
In case when the steps above do not lead to the accurate quantized model you may use the so-called `AccuracyAwareQuantization` algorithm which leads to mixed-precision models.
The whole idea behind that is to revert quantized layers back to floating-point precision based on their contribution to the accuracy drop until the desired accuracy degradation with respect to
the full-precision model is satisfied.

A fragment of the configuration file with default settings is shown below (`configs/accuracy_aware_quantization_template.json`):
```
"compression": {
        "target_device": "ANY", // Target device, the specificity of which will be taken into account during optimization.
                                // The default value "ANY" stands for compatible quantization supported by any HW.
        "algorithms": [
            {
                "name": "AccuracyAwareQuantization", // Optimization algorithm name
                "params": {
                    "preset": "performance", // Preset [performance, mixed, accuracy] which control the quantization
                                             // mode (symmetric, mixed (weights symmetric and activations asymmetric)
                                             // and fully asymmetric respectively)

                    "stat_subset_size": 300, // Size of subset to calculate activations statistics that can be used
                                             // for quantization parameters calculation

                    "maximal_drop": 0.01 // Maximum accuracy drop which has to be achieved after the quantization
                }
            }
        ]
    }

```

Since the `AccuracyAwareQuantization` calls the `DefaultQuantization` at the first step it means that all the parameters of the latter one are also valid and can be applied to the
accuracy-aware scenario.

> **NOTE**: In general case, possible speedup after applying the `AccuracyAwareQuantization` algorithm is less than after the `DefaultQuantization` when the model gets fully-quantized.

If you do not achieve the desired accuracy and performance after applying the 
`AccuracyAwareQuantization` algorithm or you need an accurate fully-quantized model,
we recommend either using layer-wise hyperparameters tuning with TPE or using 
Quantization-Aware training from [the supported frameworks](LowPrecisionOptimizationGuide.md).
