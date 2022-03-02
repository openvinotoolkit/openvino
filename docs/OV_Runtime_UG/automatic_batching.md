# Automatic Batching Plugin {#openvino_docs_OV_UG_Automatic_Batching}

## (Automatic) Batching Execution

The Automatic-Batching is a preview of the new functionality in the OpenVINO™ toolkit.  It performs on-the-fly automatic batching (i.e. grouping inference requests together) to improve device utilization, with no programming effort from the user.
Inputs gathering and outputs scattering from the individual inference requests required for the batch happen transparently, without affecting the application code. 
The feature primarily targets existing code written for inferencing many requests (each instance with the batch size 1). To obtain corresponding performance improvements, the application must be *running many inference requests simultaneously*. 
As explained below, the auto-batching functionality can be also used via a special *virtual* device.       

Batching is a straightforward way of leveraging the GPU compute power and saving on communication overheads. The automatic batching is  _implicitly_ triggered on the GPU when the `ov::hint::PerformanceMode::THROUGHPUT` is specified for the `ov::hint::performance_mode` property for the compile_model or set_property calls.
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
> **NOTE**: You can disable the Auto-Batching (for example, for the GPU device) from being triggered by the `ov::hint::PerformanceMode::THROUGHPUT`. To do that, pass the `ov::hint::allow_auto_batching` set to **false** in addition to the `ov::hint::performance_mode`:
@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_auto_batching.cpp
       :language: cpp
       :fragment: [compile_model_no_auto_batching]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_auto_batching.py
       :language: python
       :fragment: [compile_model_no_auto_batching]

@endsphinxdirective


Alternatively, to enable the Auto-Batching in the legacy apps not akin to the notion of the performance hints, you may need to use the **explicit** device notion, such as 'BATCH:GPU'. In both cases (the "throughput" hint or explicit BATCH device), the optimal batch size selection happens automatically. The actual value depends on a model and device specifics (for example, on-device memory for the dGPUs).

This _automatic batch size selection_ assumes that the application queries the `ov::optimal_number_of_infer_requests` to create and run the returned number of requests simultaneously:
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
If not enough inputs were collected, the `timeout` value makes the transparent execution fall back to the execution of individual requests. Configuration-wise, this is the AUTO_BATCH_TIMEOUT property.
The timeout, which adds itself to the execution time of the requests, heavily penalizes the performance. To avoid this, in cases when your parallel slack is bounded, give the OpenVINO an additional hint.

For example, the application processes only 4 video streams, so there is no need to use a batch larger than 4. The most future-proof way to communicate the limitations on the parallelism is to equip the performance hint with the optional `ov::hint::num_requests` configuration key set to *4*. For the GPU this will limit the batch size, for the CPU - the number of inference streams, so each device uses the `ov::hint::num_requests` while converting the hint to the actual device configuration options:
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

For the *explicit* usage, you can limit the batch size using  "BATCH:GPU(4)",  where 4 is the number of requests running in parallel.

### Other Performance Considerations

To achieve the best performance with the Automatic Batching it is strongly recommended that the application
 - Operates the number of inference requests that is multiple of the batch size. In the above example, of the batch size 4, the application best to operate 4, 8, 12, 16, etc requests
 - Uses the requests ("grouped" by the batch size) together, for example the first 4 requests are inferred, while the second group of the requests is being populated, and so on.  

The following are limitations of the current implementations:
 - Although less critical for the throughput-oriented scenarios, the load-time with auto-batching increases by almost 2x.
 - Certain networks are not reshape-able by the "batching" dimension (specified as 'N' in the layouts terms) or if the dimension is not zero-th, the auto-batching is not triggered. 
 - Performance improvements happen at the cost of the memory footprint growth, yet the auto-batching queries the available memory (especially for the dGPUs) and limits the selected batch size accordingly.

 

### Configuring the Automatic Batching
Following the OpenVINO convention for devices names, the *batching* device is named *BATCH*. The configuration options are as follows:

| Parameter name     | Parameter description      | Default            |             Examples                                                      |
| :---               | :---                  | :---               |:-----------------------------------------------------------------------------|
| "AUTO_BATCH_DEVICE" | Device name to apply the automatic batching and optional batch size in brackets | N/A | BATCH:GPU which triggers the automatic batch size selection or explicit batch size BATCH:GPU(4)     |
| "AUTO_BATCH_TIMEOUT" | timeout value, in ms | 1000 |  you can reduce the timeout value (to avoid performance penalty when the data arrives too non-evenly) e.g. pass the "100", or in contrast make it large enough e.g. to accommodate inputs preparation (e.g. when it is serial process)     |

### See Also
[Supported Devices](supported_plugins/Supported_Devices.md)