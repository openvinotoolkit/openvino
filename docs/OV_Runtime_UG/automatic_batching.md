# Automatic Batching Plugin {#openvino_docs_OV_UG_Automatic_Batching}

## (Automatic) Batching Plugin Execution (C++)

@sphinxdirective
.. raw:: html

    <div id="switcher-cpp" class="switcher-anchor">C++</div>
@endsphinxdirective

The BATCH device is a preview of the new, special "virtual" device in the OpenVINO™ toolkit. It performs on-the-fly automatic batching (i.e. grouping inference requests together) to improve device utilization, with no programming effort from the user side.
Inputs gathering and outputs scattering from the individual inference requests required for the batch happen transparently, without affecting the application code. 
The feature primarily targets existing code written for inferencing many requests (each instance with the batch size 1). To obtain corresponding performance improvements, the application must be *running many inference requests simultaneously*. 
  
Batching is a straightforward way of leveraging the GPU compute power and saving on communication overheads. The automatic batching is  _implicitly_ triggered on the GPU when the `ov::hint::PerformanceMode::THROUGHPUT` is specified for the `ov::hint::performance_mode` property for the compile_model or set_property calls. 
> **NOTE**: You can disable the Auto-Batching (for example, for the GPU device) from being triggered by the `ov::hint::PerformanceMode::THROUGHPUT`. To do that, pass the `ov::hint::allow_auto_batching` set to 'false' in addition to the `ov::hint::performance_mode` .
Alternatively, in the legacy apps not akin to the notion of the hints, you may need to use the **explicit** device notion, such as 'BATCH:GPU'. In both cases, the batch size selection depends on a model and device specifics (for example, on-device memory for the dGPUs).

The _automatic batch size selection_ assumes that the application queries the `ov::optimal_number_of_infer_requests` to create and run the returned number of requests simultaneously. If not enough inputs were collected, the `timeout` value makes the transparent execution fall back to the execution of individual requests. Configuration-wise, this is the AUTO_BATCH_TIMEOUT property. 
The timeout, which adds itself to the execution time of the requests, heavily penalizes the performance. To avoid this, in cases when your parallel slack is bounded, (for example, the application processes only 4 video streams, so there is no need to use a batch larger than 4) give the OpenVINO an additional hint.
When the performance hint is equipped with the optional ov::hint::num_requests configuration key set to "4". This is the most future-proof way to communicate the limitations on the parallelism. E.g. for the GPU that will limit the batch size, for the CPU, the number of inference streams <link> accordingly, so each device uses the ov::hint::num_requests when converting the hint to the actual device configuration options.
For the explicit usage i.e. the BATCH device you can limit the batch size via "BATCH:GPU(4)",  when you have only 4 requests running in parallel.

Limitations:
 - Although less critical for the throughput-oriented scenarios, the load-time with auto-batching increases by almost 2x.
 - Certain networks are not reshape-able by the batching dimension (specified as 'N' in the layouts terms), so the auto-batching is not triggered.
- Although performance improvements happen at the cost of the memory footprint growth, the auto-batching queries the available memory (especially for the dGPUs) and limits the selected batch size accordingly.

 

### Configuring the Automatic Batching Plugin
Following the OpenVINO convention for devices names, the  *batching* device is named  *BATCH*. The configuration options are as follows:

| Parameter name     | Parameter description      | Default            |             Examples                                                      |
| :---               | :---                  | :---               |:-----------------------------------------------------------------------------|
| "AUTO_BATCH_DEVICE" | device name to apply the automatic batching and optional batch size in brackets | N/A | BATCH:GPU which triggers the automatic batch size selection or explicit batch size BATCH:GPU(4)     |
| "AUTO_BATCH_TIMEOUT" | timeout value, in ms | 1000 |  you can reduce the timeout value (to avoid performance penalty when the data arrives too non-evenly) e.g. pass the "100", or in contrast make it large enough e.g. to accommodate inputs preparation (e.g. when it is serial process)     |

### See Also
[Supported Devices](supported_plugins/Supported_Devices.md)