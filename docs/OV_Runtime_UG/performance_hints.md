# High-level Performance Hints {#openvino_docs_OV_UG_Performance_Hints}

Each of OpenVINO's [supported devices](supported_plugins/Device_Plugins.md) offers low-level performance settings. Tweaking this detailed configuration requires deep architecture understanding.
Also, while the performance may be optimal for the specific combination of the device and the inferred model, the resulting configuration is not necessarily optimal for another device or model.
The OpenVINO performance hints are the new way to configure the performance with _portability_ in mind. As the hints are supported by every OpenVINO device, this is a future-proof solution that is fully compatible with the [automatic device selection](./auto_device_selection.md).

The hints also "reverse" the direction of the configuration in the right fashion: rather than map the application needs to the low-level performance settings, and keep an associated application logic to configure each possible device separately, the idea is to express a target scenario with a single config key and let the *device* to configure itself in response.

Previously, a certain level of automatic configuration was coming from the _default_ values of the parameters. For example, the number of CPU streams was deduced from the number of CPU cores, when `ov::streams::AUTO` (`CPU_THROUGHPUT_AUTO` in the pre-OpenVINO 2.0 parlance) is set. However, the resulting number of streams didn't account for actual compute requirements of the model to be inferred.
The hints, in contrast, respect the actual model, so the parameters for optimal throughput are calculated for each model individually (based on its compute versus memory bandwidth requirements and capabilities of the device).

## Performance Hints: Latency and Throughput
As discussed in the [Optimization Guide](../optimization_guide/dldt_optimization_guide.md) there are a few different metrics associated with inference speed.
Throughput and latency are some of the most widely used metrics that measure the overall performance of an application.

This is why, to ease the configuration of the device, OpenVINO offers two dedicated hints, namely `ov::hint::PerformanceMode::THROUGHPUT` and `ov::hint::PerformanceMode::LATENCY`.
A special `ov::hint::PerformanceMode::UNDEFINED` acts the same as specifying no hint.

Please also see the last section in this document on conducting performance measurements with the benchmark_app`.

Note that a typical model may take significantly more time to load with `ov::hint::PerformanceMode::THROUGHPUT` and consume much more memory, compared with `ov::hint::PerformanceMode::LATENCY`.

## Performance Hints: How It Works?
Internally, every device "translates" the value of the hint to the actual performance settings.
For example the `ov::hint::PerformanceMode::THROUGHPUT` selects number of CPU or GPU streams.
For the GPU, additionally the optimal batch size is selected and the [automatic batching](../OV_Runtime_UG/automatic_batching.md) is applied whenever possible (and also if the device supports that [refer to the devices/features support matrix](./supported_plugins/Device_Plugins.md)).

The resulting (device-specific) settings can be queried back from the instance of the `ov:Compiled_Model`.  
Notice that the `benchmark_app`, outputs the actual settings for the THROUGHPUT hint, please the bottom of the output example:

   ```
    $benchmark_app -hint tput -d CPU -m 'path to your favorite model'
    ...
    [Step 8/11] Setting optimal runtime parameters
    [ INFO ] Device: CPU
    [ INFO ]   { PERFORMANCE_HINT , THROUGHPUT }
    ...
    [ INFO ]   { OPTIMAL_NUMBER_OF_INFER_REQUESTS , 4 }
    [ INFO ]   { NUM_STREAMS , 4 }
    ...
   ```

## Using the Performance Hints: Basic API
In the example code-snippet below the  `ov::hint::PerformanceMode::THROUGHPUT` is specified for the `ov::hint::performance_mode` property for the compile_model:
@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_auto_batching.cpp
       :language: cpp
       :fragment: [compile_model]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_auto_batching.py
       :language: python
       :fragment: [compile_model]

@endsphinxdirective

## Additional (Optional) Hints from the App
Let's take an example  of an application that processes 4 video streams.  The most future-proof way to communicate the limitation of the parallel slack is to equip the performance hint with the optional `ov::hint::num_requests` configuration key set to 4. 
As discussed previosly, for the GPU this will limit the batch size, for the CPU - the number of inference streams, so each device uses the `ov::hint::num_requests` while converting the hint to the actual device configuration options:
@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_auto_batching.cpp
       :language: cpp
       :fragment: [hint_num_requests]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_auto_batching.py
       :language: python
       :fragment: [hint_num_requests]

@endsphinxdirective

## Optimal Number of Inference Requests
Using the hints assumes that the application queries the `ov::optimal_number_of_infer_requests` to create and run the returned number of requests simultaneously:
@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_auto_batching.cpp
       :language: cpp
       :fragment: [query_optimal_num_requests]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_auto_batching.py
       :language: python
       :fragment: [query_optimal_num_requests]

@endsphinxdirective

While an application is free to create more requests if needed (for example to support asynchronous inputs population) **it is very important to at least run the `ov::optimal_number_of_infer_requests` of the inference requests in parallel**, for efficiency (device utilization) reasons. 

Also, notice that `ov::hint::PerformanceMode::LATENCY` does not necessarily imply using single inference request. For example, multi-socket CPUs can deliver as high number of requests (at the same minimal latency) as there are NUMA nodes the machine features.
To make your application fully scalable, prefer to query the `ov::optimal_number_of_infer_requests` directly.

## Prefer Async API
The API of the inference requests offers Sync and Async execution. While the `ov::InferRequest::infer()` is inherently synchronous and simple to operate (as it serializes the execution flow in the current application thread), the Async "splits" the `infer()` into `ov::InferRequest::start_async()` and use of the `ov::InferRequest::wait()` (or callbacks). Please consider the [API examples](../OV_Runtime_UG/ov_infer_request.md).
 Although the Synchronous API can be somewhat easier to start with, in the production code always prefer to use the Asynchronous (callbacks-based) API, as it is the most general and scalable way to implement the flow control for any possible number of requests (and hence both latency and throughput scenarios).
 
## Combining the Hints and Individual Low-Level Settings
While sacrificing the portability at a some extent, it is possible to combine the hints with individual device-specific settings. 
For example, you can let the device prepare a configuration `ov::hint::PerformanceMode::THROUGHPUT` while overriding any specific value:  
@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_auto_batching.cpp
       :language: cpp
       :fragment: [hint_plus_low_level]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_auto_batching.py
       :language: python
       :fragment: [hint_plus_low_level]


@endsphinxdirective
## Testing the Performance of The Hints with the Benchmark_App
The `benchmark_app`, that exists in both  [C++](../../samples/cpp/benchmark_app/README.md) and [Python](../../tools/benchmark_tool/README.md) versions, is the best way to evaluate the performance of the performance hints for a particular device:
 - benchmark_app **-hint tput** -d 'device' -m 'path to your model'
 - benchmark_app **-hint latency** -d 'device' -m 'path to your model'
-  Disabling the hints to emulate the pre-hints era (highly recommended before trying the individual low-level settings, such as the number of streams as below, threads, etc):
- - benchmark_app **-hint none -nstreams 1**  -d 'device' -m 'path to your model'
 

### See Also
[Supported Devices](./supported_plugins/Supported_Devices.md)
