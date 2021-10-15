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
3.  Layer-wise hyperparameters tuning

## Tuning Hyperparameters of the DefaultQuantization
The `DefaultQuantization` algorithm provides multiple hyperparameters which can be used in order to improve accuracy results for the fully-quantized model. 
Below is a list of best practices which can be applied to improve accuracy without a substantial performance reduction with respect to default settings:
1.  The first option that we recommend to change is `preset` that can be varied from `performance` to `mixed`. It enables asymmetric quantization of 
activations and can be helpful for the NNs with non-ReLU activation functions, e.g. YOLO, EfficientNet, etc.
2.  The next option is `use_fast_bias`. Setting this option for `false` enables a different bias correction method which is more accurate, in general,
and applied after model quantization as a part of `DefaultQuantization` algorithm.
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

Please find the possible options and their description in the `config/default_quantization_spec.json` file in the POT directory.

4.  The next option is `stat_subset_size`. It controls the size of the calibration dataset used by POT to collect statistics for quantization parameters initialization.
It is assumed that this dataset should contain a sufficient number of representative samples. Thus, varying this parameter may affect accuracy (the higher is better). 
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

## Layer-Wise Hyperparameters Tuning Using TPE

As the last step in post-training optimization, you may try layer-wise hyperparameter 
tuning using TPE, which stands for Tree of Parzen Estimators hyperparameter optimizer 
that searches through available configurations trying to find an optimal one. 
For post-training optimization, TPE assigns multiple available configuration 
options to choose from for every layer and by evaluating different sets of parameters, 
it creates a probabilistic model of their impact on accuracy and latency to 
iteratively find an optimal one.

You can run TPE with any combination of parameters in `tuning_scope`, but it is 
recommended to use one of two configurations described below. It is recommended to first try
Range Estimator Configuration. If this configuration will not be able to reach accuracy
target then it is recommended to run Layer Configuration. If for some reason, 
like HW failure or power shutdown, TPE trials stop before completion, you can 
rerun them starting from the last trial by changing `trials_load_method` 
from `cold_start` to `warm_start` as long as logs from the previous execution are available.

> **NOTE**: TPE requires many iterations to converge to an optimal solution, and 
> it is recommended to run it for at least 200 iterations. Because every iteration 
> requires evaluation of a generated model , which means accuracy measurements on a 
> dataset and latency measurements using benchmark, this process may take from 
> 24 hours up to few days to complete, depending on a model. 
> To run this configuration on multiple machines and reduce the execution time,
> see [Multi-node](../openvino/tools/pot/optimization/tpe/multinode.md).

### Range Estimator Configuration

To run TPE with range estimator tuning, use the following configuration:
```json
"optimizer": {
    "name": "Tpe",
    "params": {
        "max_trials": 200,
        "trials_load_method": "cold_start",
        "accuracy_loss": 0.1,
        "latency_reduce": 1.5,
        "accuracy_weight": 1.0,
        "latency_weight": 0.0,
        "benchmark": {
            "performance_count": false,
            "batch_size": 1,
            "nthreads": 8,
            "nstreams": 1,
            "nireq": 1,
            "api_type": "async",
            "niter": 1,
            "duration_seconds": 30,
            "benchmark_app_dir": "<path to benchmark_app>" // Path to benchmark_app If not specified, Python base benchmark will be used. Use benchmark_app to reduce jitter in results.
        }
    }
},
"compression": {
    "target_device": "ANY",
    "algorithms": [
        {
            "name": "ActivationChannelAlignment",
            "params": {
                "stat_subset_size": 300
            }
        },
        {
            "name": "TunableQuantization",
            "params": {
                "stat_subset_size": 300,
                "preset": "performance",
                "tuning_scope": ["range_estimator"],
                "estimator_tuning_scope": ["preset", "outlier_prob"],
                "outlier_prob_choices": [1e-3, 1e-4, 1e-5]
            }
        },
        {
            "name": "FastBiasCorrection",
            "params": {
                "stat_subset_size": 300
            }
        }
    ]
}
```

This configuration searches for optimal preset for `range_estimator` and optimal 
outlier probability for quantiles for every layer. Because this configuration 
only changes final values provided to [FakeQuantize]((https://docs.openvinotoolkit.org/latest/_docs_ops_quantization_FakeQuantize_1.html)) layers, changes in parameters 
do not impact inference latency, thus we set `latency_weight` to 0 to prevent 
jitter in benchmark results to negatively impact model evaluation. Experiments 
show that this configuration can give much better accuracy then the approach of 
just changing `range_estimator` configuration globally.

### Layer Configuration

To run TPE with layer tuning, use the following configuration:
```json
"optimizer": {
    "name": "Tpe",
    "params": {
        "max_trials": 200,
        "trials_load_method": "cold_start",
        "accuracy_loss": 0.1,
        "latency_reduce": 1.5,
        "accuracy_weight": 1.0,
        "latency_weight": 1.0,
        "benchmark": {
            "performance_count": false,
            "batch_size": 1,
            "nthreads": 8,
            "nstreams": 1,
            "nireq": 1,
            "api_type": "async",
            "niter": 1,
            "duration_seconds": 30,
            "benchmark_app_dir": "<path to benchmark_app>" // Path to benchmark_app If not specified, Python base benchmark will be used. Use benchmark_app to reduce jitter in results.
        }
    }
},
"compression": {
    "target_device": "ANY",
    "algorithms": [
        {
            "name": "ActivationChannelAlignment",
            "params": {
                "stat_subset_size": 300
            }
        },
        {
            "name": "TunableQuantization",
            "params": {
                "stat_subset_size": 300,
                "preset": "performance",
                "tuning_scope": ["layer"]
            }
        },
        {
            "name": "FastBiasCorrection",
            "params": {
                "stat_subset_size": 300
            }
        }
    ]
}
```

This configuration is similar to `AccuracyAwareQuantization`, because it also 
tries to revert quantized layers back to floating-point precision, but uses a 
different algorithm to choose layers, which can lead to better results.
