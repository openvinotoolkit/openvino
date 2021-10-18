#  Tree-Structured Parzen Estimator (TPE) {#pot_compression_optimization_tpe_README}

## Overview
Tree-Structured Parzen Estimator (TPE) algorithm is designed to optimize quantization hyperparameters to find quantization configuration that achieve an expected accuracy target and provide best possible latency improvement.
TPE is an iterative process that uses history of evaluated hyperparameters to create probabilistic model, which is used to suggest next set of hyperparameters to evaluate.

Generally, the algorithm consists of the following steps:
1. Define a domain of hyperparameter search space,
2. Create an objective function which takes in hyperparameters and outputs a score (e.g., loss, root mean squared error, cross-entropy) that we want to minimize,
3. Get couple of observations (score) using randomly selected set of hyperparameters,
4. Sort the collected observations by score and divide them into two groups based on some quantile. The first group (x1) contains observations that gave the best scores and the second one (x2)  - all other observations,
5. Two densities l(x1) and g(x2) are modeled using Parzen Estimators (also known as kernel density estimators) which are a simple average of kernels centered on existing data points,
6. Draw sample hyperparameters from l(x1), evaluating them in terms of l(x1)/g(x2), and returning the set that yields the minimum value under l(x1)/g(x1) corresponding to the greatest expected improvement. These hyperparameters are then evaluated on the objective function.
7. Update the observation list from step 3
8. Repeat step 4-7 with a fixed number of trials or until time limit is reached

TPE uses hyperparameters metadata returned by [TunableQuantization algorithm](../../algorithms/quantization/tunable_quantization/README.md) 
to create search space from which it chooses hyperparameters for algorithms. 
The configuration of hyperparameters metadata generation is done at [TunableQuantization algorithm](../../algorithms/quantization/tunable_quantization/README.md) level. 
For details, see description of the [TunableQuantization algorithm](../../algorithms/quantization/tunable_quantization/README.md).

For more information about TPE see [1].

> **NOTE**: TPE requires many iterations to converge to an optimal solution, and 
> it is recommended to run it for at least 200 iterations. Because every iteration 
> requires evaluation of a generated model, which means accuracy measurements on a 
> dataset and latency measurements using benchmark, this process may take from 
> 24 hours up to few days to complete, depending on a model. 
> Due to this, even though the TPE supports all OpenVINO™-supported models, it 
> is being continuously validated only on a subset of models:
> *  SSD MobileNet V1 COCO
> *  Mobilenet V2 1.0 224
> *  Faster R-CNN ResNet 50 COCO
> *  Faster R-CNN Inception V2 COCO
> *  YOLOv3 TF Full COCO

## Progress reporting

After every iteration TPE will output information about current iteration and best result.
Current iteration result may have two forms:
1. `INFO:compression.optimization.tpe.algorithm:Current iteration acc_loss: 0.0054 lat_diff: 1.2787 exec_time: 5.6135`
2. `INFO:compression.optimization.tpe.algorithm:Params already evaluated. Returning previous result: acc_loss: 0.0054 lat_diff: 1.2787`

The first one is used when TPE evaluates new set of hyperparameters.
The second one is used when TPE suggested set of hyperparameters that was already evaluated.
In this case hyperparameters were not reevaluated, instead previous result for these hyperparameters was added to the observations list.

Best result is reported with following line:
`INFO:compression.optimization.tpe.algorithm:Trial iteration end: 1 / 2 acc_loss: 0.0054 lat_diff: 1.2787`

In all of the above logs reported values are following:
- `acc_loss` - relative accuracy loss, computed with formula `(FP32_accuracy - quantized_accuracy) / FP32_accuracy`, lower is better
- `lat_diff` - latency difference compared to FP32 model, computed with formula `FP32_latency / quantized_latency`, higher is better
- `exec_time` - execution time in minutes from start of the current run of the tool

## Parameters
TPE parameters can be divided into two groups: mandatory and optional.

### Mandatory parameters
- `"max_trials"` - maximum number of trails
- `"trials_load_method"` - specifies whether to start from scratch or reuse previous results. It should be used in following manner:
    - `"cold_start"` - start trials from beginning. Logs from previous execution are removed
    - `"warm_start"` - continue execution using logs from previous execution up to the limit set by `"max_trials"` (may be larger than in previous execution). [TunableQuantization algorithm](../../algorithms/quantization/tunable_quantization/README.md) parameters impacting parameters metadata creation are ignored, because search space is retrieved from logs. May be used either after `"cold_start"` or `"fine_tune"`. If no previous logs exist then it behaves like `"cold_start"`
    - `"fine_tune"` - start new trials with new search space derived from best result achieved since last `"cold_start"`. [TunableQuantization algorithm](../../algorithms/quantization/tunable_quantization/README.md) is responsible for modifying parameter metadata to accommodate parameters used to get best result (for more details about parameter metadata generation see [TunableQuantization algorithm](../../algorithms/quantization/tunable_quantization/README.md)). If no previous logs exist then it behaves like `"cold_start"`
    - `"eval"` - load best result to get model
- `"accuracy_loss"` - maximum acceptable relative accuracy loss in percentage
- `"latency_reduce"` - target latency improvement versus original model
- `"accuracy_weight"` and `"latency_weight"` - accuracy and latency weights used in loss function.
These two parameters are intended to be set to 1.0, because accuracy and latency components in the loss function are designed to be balanced equally, so that the algorithm is able to achieve an expected accuracy target and provide best possible latency improvement.
Changing `"accuracy_weight"`, which is left open for experimentation, is discouraged, but it is recommended to change `"latency_weight"` to 0 for configurations that do not change latency result, for example, when tuning parameters that only change numeric values of the parameters, such as quantization ranges, and do not change graph structure or data types
- `"benchmark"` - latency measurement benchmark configuration. For details of configuration options see [Benchmark C++ Tool](https://docs.openvinotoolkit.org/latest/_inference_engine_samples_benchmark_app_README.html)

### Optional parameters
- `"max_minutes"` - trials time limit. When it expires, the last trial is completed and the best result is returned
- `"stop_on_target"` - flag to stop TPE trials when accuracy_loss and latency_reduce targets are reached.
If false or not specified TPE will continue until max_trials or max_minutes is reached even if targets are reached earlier
- `"eval_subset_size"` - subset of test data used to evaluate hyperparameters. The whole dataset is used if no parameter specified.
- `"metrics"` - an optional list of reference metrics values.
If not specified, all metrics will be calculated from the original model.
It consists of tuples with the following parameters:
    - `"name"` - name of the metric to optimize
    - `"baseline_value"` - baseline metric value of the original model

Below is a fragment of the configuration file that shows overall structure of parameters for this algorithm.
 
```json
/* Optimizer used to find "optimal" hyperparameters */
"optimizer": {
    "name": "Tpe", // Optimizer name
    "params": {
        "max_trials": 100, // Maximum number of trails
        "max_minutes": 10, // [Optional] Trials time limit. When it expires, the last trial is completed and the best result is returned.
        "stop_on_target": true, // [Optional] Flag to stop TPE trials when accuracy_loss and latency_reduce targets are reached.
                                // If false or not specified TPE will continue until max_trials or max_minutes is reached even if targets are reached earlier.
        "eval_subset_size": 2000, // [Optional] subset of test data used to evaluate hyperparameters. The whole dataset is used if no parameter specified.
        "trials_load_method": "cold_start", // Start from scratch or reuse previous results, supported options [cold_start, warm_start, fine_tune, eval]
        "accuracy_loss": 0.1, // Accuracy threshold (%)
        "latency_reduce": 1.5, // Target latency improvement versus original model
        "accuracy_weight": 1.0, // Accuracy weight in loss function
        "latency_weight": 1.0, // Latency weight in loss function
        // An optional list of reference metrics values.
        // If not specified, all metrics will be calculated from the original model.
        "metrics": [
            {
                "name": "accuracy", // Metric name
                "baseline_value": 0.72 // Baseline metric value of the original model
            }
        ],
        "benchmark": {
            // Latency measurement benchmark configuration (https://docs.openvinotoolkit.org/latest/_inference_engine_samples_benchmark_app_README.html)
            "performance_count": false,
            "batch_size": 0,
            "nthreads": 4,
            "nstreams": 0,
            "nireq": 0,
            "api_type": "sync",
            "niter": 4,
            "duration_seconds": 30,
            "benchmark_app_dir": "<path to benchmark_app>" // Path to benchmark_app If not specified, Python base benchmark will be used. Use benchmark_app to reduce jitter in results.
        }
    }
}
```

### Multiple node configuration
Domain for hyperparameter search space can be vast. Searching best result is time-consuming process.
The current implementation allow to use multiple machines to work together. For more information go to [Multi-node](multinode.md) description.

## Advantages
1.	TPE supports a wide variety of variables in parameter search space e.g., uniform, log-uniform, quantized log-uniform, normally-distributed real value, categorical. 
2.	Less time to tune. Extremely computationally efficient than conventional methods.
3.	Scope to define the tuning time of quantization.
4.	Scope to define quantization search space and strategies from a wide range of OpenVINO™ quantization algorithms.
5.	Scope to define error tolerance and desired latency improvement. This algorithm guaranteed to get best possible accuracy and latency from optimal parameter combination (quantization algorithms and strategies).

## Drawbacks

TPE does not model interactions between hyperparameters.

## Reference
[1] J. S. Bergstra, R. Bardenet, Y. Bengio, and B. Kégl, “Algorithms for Hyper-Parameter Optimization,” in Advances in Neural Information Processing Systems 24, J. Shawe-Taylor, R. S. Zemel, P. L. Bartlett, F. Pereira, and K. Q. Weinberger, Eds. Curran Associates, Inc., 2011, pp. 2546–2554.