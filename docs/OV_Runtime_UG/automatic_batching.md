# Automatic Batching Plugin {#openvino_docs_OV_UG_Automatic_Batching}

## (Automatic) Batching Plugin Execution (C++)

@sphinxdirective
.. raw:: html

    <div id="switcher-cpp" class="switcher-anchor">C++</div>
@endsphinxdirective

The BATCH device is a preview of the new, special "virtual" device in the OpenVINO™ toolkit. It performs on-the-fly automatic batching (i.e. grouping inference requests together) to improve device utilization, with no programming effort from the user side.
Gathering of the inputs (as well as scattering the outputs) from the individual inference requests that is needed to constitute the batch happens in the completely transparent way, so the application code is left intact. The major pre-requisite for the application to enjoy an associated performance improvements is *running many inference requests simultaneously*.
So the feature primarily targets existing codes that are written for inferencing many requests (each instances with batch size of 1).
  
Since the batching is straightforward way of leveraging the GPU compute power (and save on communication overheads) the automatic batching is _implicitly_ triggered on the GPU when the ov::hint::PerformanceMode::THROUGHPUT is specified for the ov::hint::performance_mode property for the compile_model or set_property calls. 
**NOTE**: You can disable the Auto-Batching (e.g. for the GPU device) from being triggered by the ov::hint::PerformanceMode::THROUGHPUT. Just pass the ov::hint::allow_auto_batching (set to the 'false') in addition to the ov::hint::performance_mode .
Alternatively (in the legacy apps that are not akin to the notion of the performance hints) you may want to use the explicit device notion like 'BATCH:GPU'. In both cases the actual batch size selection happens on the per-model basis (also respecting things like on-device memory for the dGPUs).

Notice that _automatic batch size selection_ assumes that the application queries the ov::optimal_number_of_infer_requests to create and runs simultaneously the returned number of requests. This is very important, as there is a "timeout" value so that (when no enough inputs have been collected) the execution transparently falls back to execution the requests individually. Configuration-wise this is AUTO_BATCH_TIMEOUT property, please see below.
The timeout, which adds itself to the execution time of the requests, heavily penalizes the performance. To avoid it, in the cases when your parallel slack is bounded (e.g. the application processes only 4 video-streams, so batch larger than 4 doesn't make sense) make sure to give the OpenVINO an additional hint the OV on this.
When the performance hint is equipped with the optional ov::hint::num_requests configuration key set to "4". This is the most future-proof way to communicate the limitations on the parallelism. E.g. for the GPU that will limit the batch size, for the CPU, the number of inference streams <link> accordingly, so each device uses the ov::hint::num_requests when converting the hint to the actual device configuration options.
For the explicit usage i.e. the BATCH device you can limit the batch size via "BATCH:GPU(4)",  when you have only 4 requests running in parallel.

Limitations:
 - Although less critical for the throughput-oriented scenarios, the load-time is increased almost 2x when auto-batching is used 
 - Certain networks are not reshape-able by the batching (aka 'N', using the layouts parlance) dimension, so the auto-batching will not be triggered 
 - Memory footpint growth is unavoidable price. Yet the auto-batching queries the available memory (especially for the dGPUs) and limits the selected batch size accordingly.

 

### Configuring the Automatic Batching Plugin
Following the OpenVINO convention for devices names, the  "batching" device is named as  "BATCH". The configuration options for that are as follows:

| Parameter name     | Parameter description      | Default            |             Examples                                                      |
| :---               | :---                  | :---               |:-----------------------------------------------------------------------------|
| "AUTO_BATCH_DEVICE" | device name to apply the automatic batching and optional batch size in brackets | N/A | BATCH:GPU which triggers the automatic batch size selection or explicit batch size BATCH:GPU(4)     |
| "AUTO_BATCH_TIMEOUT" | timeout value, in ms | 1000 |  you can reduce the timeout value (to avoid performance penalty when the data arrives too non-evenly) e.g. pass the "100", or in contrast make it large enough e.g. to accommodate inputs preparation (e.g. when it is serial process)     |

### See Also
[Supported Devices](supported_plugins/Supported_Devices.md)