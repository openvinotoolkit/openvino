# Getting Performance Numbers {#openvino_docs_MO_DG_Getting_Performance_Numbers}


## Tip 1. Measure the Proper Set of Operations 

When evaluating performance of your model with the OpenVINO Runtime, you must measure the proper set of operations. To do so, consider the following tips: 

- Avoid including one-time costs like model loading.

- Track separately the operations that happen outside the OpenVINO Runtime, like video decoding. 

> **NOTE**: Some image pre-processing can be baked into the IR and accelerated. For more information, refer to [Embedding Preprocessing Computation](Additional_Optimizations.md)

## Tip 2. Getting Credible Performance Numbers 

You need to build your performance conclusions on reproducible data. Do the performance measurements with a large number of invocations of the same routine. Since the first iteration is almost always significantly slower than the subsequent ones, you can use an aggregated value for the execution time for final projections:

-	If the warm-up run does not help or execution time still varies, you can try running a large number of iterations and then average or find a mean of the results.
-	For time values that range too much, consider geomean.


## Getting performance numbers using OpenVINO's benchmark_app 

To get performance numbers please use the dedicated [Benchmark App](../../../samples/cpp/benchmark_app/README.md) sample that is the best way to produce the performance reference.
It has a lot of device-specific knobs, but the primary usage is as simple as: 
```bash
$ ./benchmark_app –d GPU –m <model> -i <input>
```
to measure the performance of the model on the GPU. 
Or
```bash
$ ./benchmark_app –d CPU –m <model> -i <input>
```
to execute on the CPU instead.

Each of the [OpenVINO supported devices](../../OV_Runtime_UG/supported_plugins/Supported_Devices.md) offers a bunch of performance settings that have a command-line equivalents in the [Benchmark App](../../../samples/cpp/benchmark_app/README.md).
While these settings provide really low-level control and allow to leverage the optimal model performance on the _specific_ device, we suggest to always start the performance evaluation with trying the [OpenVINO High-Level Performance Hints](../../OV_Runtime_UG/performance_hints.md) first:
 - benchmark_app **-hint tput** -d 'device' -m 'path to your favorite model'
 - benchmark_app **-hint latency** -d 'device' -m 'path to your favorite model'

## Comparing Performance with Native/Framework Code 

When comparing the OpenVINO Runtime performance with the framework or another reference code, make sure that both versions are as similar as possible:

-	Wrap exactly the inference execution (refer to the [OpenVINO Samples](../../OV_Runtime_UG/Samples_Overview.md) for examples).
-	Do not include model loading time.
-	Ensure the inputs are identical for the OpenVINO Runtime and the framework. For example, beware of random values that can be used to populate the inputs.
-	Consider [Image Pre-processing and Conversion](../../OV_Runtime_UG/preprocessing_overview.md), while any user-side pre-processing should be tracked separately.
-	If possible, demand the same accuracy. For example, TensorFlow allows `FP16` execution, so when comparing to that, make sure to test the OpenVINO Runtime with the `FP16` as well.