# Automatic Batching {#openvino_docs_OV_UG_Automatic_Batching}

This article covers details in regards of Automatic Batching. Described here are general information alongside configuration and testing performance.

## (Automatic) Batching Execution

The Automatic-Batching is a preview of the new functionality in the OpenVINO™ toolkit. It performs automatic batching on-the-fly (grouping inference requests together) to improve device utilization, with no programming effort from the user.
Gathering the input and scattering the output from the individual inference requests required for the batch happen transparently, without affecting the application code. 

The feature targets primarily the existing code written for inferencing many requests (each instance with the batch size 1). To obtain corresponding performance improvements, the application must be *running many inference requests simultaneously*. 
As explained below, the auto-batching functionality can be also used via a particular *virtual* device.       

Batching is a straightforward way of leveraging the compute power of GPU and saving on communication overheads. The automatic batching is "implicitly" triggered on the GPU when the `ov::hint::PerformanceMode::THROUGHPUT` is specified for the `ov::hint::performance_mode` property for the `compile_model` or `set_property` calls.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_auto_batching.cpp compile_model

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_auto_batching.py compile_model

@endsphinxtab

@endsphinxtabset


> **NOTE**: The Auto-Batching can be disabled (for example, for the GPU device) from being triggered by the `ov::hint::PerformanceMode::THROUGHPUT`. To do that, pass the `ov::hint::allow_auto_batching` set to **false** in addition to the `ov::hint::performance_mode`:


@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_auto_batching.cpp compile_model_no_auto_batching

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_auto_batching.py compile_model_no_auto_batching

@endsphinxtab

@endsphinxtabset


Alternatively, to enable the Auto-Batching in the legacy apps not akin to the notion of the performance hints, you may need to use the **explicit** device notion, such as `BATCH:GPU`. In both cases (the *throughput* hint or explicit BATCH device), the optimal batch size selection happens automatically (the implementation queries the `ov::optimal_batch_size` property from the device, passing the model graph as the parameter). The actual value depends on the model and device specifics, for example, on-device memory for the dGPUs.
Auto-Batching support is not limited to the GPUs, but if a device does not support the `ov::optimal_batch_size` yet, it can work with the auto-batching only when an explicit batch size is specified (i.e., `BATCH:<device>(16)`).

This "automatic batch size selection" works on the presumption that the application queries the `ov::optimal_number_of_infer_requests` to create and run the returned number of requests simultaneously:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_auto_batching.cpp query_optimal_num_requests

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_auto_batching.py query_optimal_num_requests

@endsphinxtab

@endsphinxtabset

If not enough inputs were collected, the `timeout` value makes the transparent execution fall back to the execution of individual requests. Configuration-wise, this is the `AUTO_BATCH_TIMEOUT` property.
The timeout, which adds itself to the execution time of the requests, heavily penalizes the performance. To avoid this, when your parallel slack is bounded, provide OpenVINO with an additional hint.

For example, when the application processes only 4 video streams, there is no need to use a batch larger than 4. The most future-proof way to communicate the limitations on the parallelism is to equip the performance hint with the optional `ov::hint::num_requests` configuration key set to 4. This will limit the batch size for the GPU and the number of inference streams for the CPU. Therefore, each device uses the `ov::hint::num_requests` while converting the hint to the actual device configuration options:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_auto_batching.cpp hint_num_requests

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_auto_batching.py hint_num_requests

@endsphinxtab

@endsphinxtabset


For the *explicit* usage, the batch size can be limited by using `BATCH:GPU(4)`, where 4 is the number of requests running in parallel.

### Other Performance Considerations

To achieve the best performance with the Automatic Batching, the application should:
 - Operate the number of inference requests that represents the multiple of the batch size. In the example above, for batch size 4, the application should operate 4, 8, 12, 16, etc. requests.
 - Use the requests, grouped by the batch size, together. For example, the first 4 requests are inferred, while the second group of the requests is being populated. Essentially, the Automatic Batching shifts the asynchronicity from the individual requests to the groups of requests that constitute the batches.
  - Balance the `timeout` value vs. the batch size. For example, in many cases having a smaller timeout value/ batch size may yield better performance than large batch size. However, the timeout value that is not large enough will not accommodate the full number of the required requests.
  - When the Automatic Batching is enabled, the `timeout` property of the `ov::CompiledModel` can be changed any time, even after loading/compilation of the model. For example, setting the value to 0 disables the auto-batching effectively, as the collection of requests would be omitted.
  - Carefully apply the auto-batching to the pipelines. For example, it is the most beneficial to do auto-batching over the inputs to the detection stage for the conventional "video-sources-> detection-> classification" flow. The resulting number of detections is usually fluent, which makes the auto-batching less applicable for the classification stage.

The following are limitations of the current implementations:
 - Although less critical for the throughput-oriented scenarios, the load-time with auto-batching increases by almost double.
 - Certain networks are not safely reshapable by the "batching" dimension (specified as `N` in the layouts terms). Also, if the batching dimension is not zeroth, the auto-batching is not triggered "implicitly" by the throughput hint.
 -  The "explicit" notion, for example, `BATCH:GPU`, uses the relaxed dimensions tracking, often making the auto-batching possible. For example, this method unlocks most **detection networks**.
 - - When *forcing* the auto-batching via the "explicit" device notion, make sure to validate the results for correctness.   
 - Performance improvements happen at the cost of the memory footprint growth. Yet, the auto-batching queries the available memory (especially for the dGPUs) and limits the selected batch size accordingly.

 
### Configuring the Automatic Batching
Following the OpenVINO naming convention, the *batching* device is assigned the label of *BATCH*. The configuration options are as follows:

| Parameter name     | Parameter description      | Default            |             Examples                                                      |
| :---               | :---                  | :---               |:-----------------------------------------------------------------------------|
| `AUTO_BATCH_DEVICE` | A device name to apply the automatic batching and optional batch size in brackets. | N/A | `BATCH:GPU` which triggers the automatic batch size selection. Another example is the device name (to apply the batching) with directly specified batch size `BATCH:GPU(4)`     |
| `AUTO_BATCH_TIMEOUT` | timeout value, in ms | 1000 |  To reduce the timeout value (to avoid performance penalty when the data arrives too non-evenly), e.g. pass the "100" or in contrast, e.g. make it large enough to accommodate inputs preparation (e.g. when it is a serial process)     |

### Testing Automatic Batching Performance with the Benchmark_App
The `benchmark_app`, that exists in both [C++](../../samples/cpp/benchmark_app/README.md) and [Python](../../tools/benchmark_tool/README.md) versions, is the best way to evaluate the performance of the Automatic Batching:
 -  The most straightforward way is the performance hints:
    - benchmark_app **-hint tput** -d GPU -m 'path to your favorite model'
 -  Overriding the strict rules of implicit reshaping by the batch dimension via the "explicit" device notion:
    - benchmark_app **-hint none -d BATCH:GPU** -m 'path to your favorite model'
 -  Finally, overriding the automatically-deduced batch size as well:
    - $benchmark_app -hint none -d **BATCH:GPU(16)** -m 'path to your favorite model'
    - note that some shell versions (e.g. `bash`) may require adding quotes around complex device names, i.e. -d "BATCH:GPU(16)"

The last example also applies to the CPU or any other device that generally supports the batched execution.  

### See Also
[Supported Devices](supported_plugins/Supported_Devices.md)
