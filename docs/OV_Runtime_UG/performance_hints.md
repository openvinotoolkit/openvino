# High-level Performance Hints {#openvino_docs_OV_UG_Performance_Hints}

Each of the OpenVINO's [supported devices](supported_plugins/Supported_Devices.md) offers low-level performance settings. Tweaking this detailed configuration requires deep architecture understanding.
Also, while the performance may be optimal for the specific combination of the device and the model that is inferred, the resulting configuration is not  necessarily optimal for another device or model.
The OpenVINO performance hints is the new way fo configuring the performance with the _portability_ in mind. 

Using the hints also does "reverse" the direction of the configuration in the right fashion: rather than map the application needs to the low-level performance settings, and potentially having associated application logic to configure each possible device separately, the idea is to express a target scenario with a single config key and let the *device* to configure itself in response.
As the hints are supported by every OpenVINO device, this is completely portable and future-proof solution. 
NameSTreams AUTO

## Performance Hints: Latency and Throughput

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

Seeing the results:


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

## (Optional) Additional Hints from the App
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
       :fragment: hint_num_requests]

@endsphinxdirective

Seeing the results:

## Combining the Hints and Individual Low-Level Settings

## Testing the Performance of THe Hints with the Benchmark_App
The `benchmark_app`, that exists in both  [C++](../../samples/cpp/benchmark_app/README.md) and [Python](../../tools/benchmark_tool/README.md) versions, is the best way to evaluate the performance of the performaqnce hints for a particular device:
 - benchmark_app **-hint tput** -d 'device' -m 'path to your favorite model'
 - benchmark_app **-hint latency** -d 'device' -m 'path to your favorite model'
-  Disabling the hints to emulate the pre-hints era (highly recommended before playing the individual low-level settings like number of streams, threads, etc):
- - benchmark_app **-hint none -nstreams 1**  -d 'device' -m 'path to your favorite model'
 

### See Also
[Supported Devices](./supported_plugins/Supported_Devices.md)