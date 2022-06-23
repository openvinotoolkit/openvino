# Automatic Batching {#openvino_docs_OV_UG_Automatic_Batching}

The Automatic Batching Execution mode (or Auto-batching for short) performs automatic batching on-the-fly to improve device utilization by grouping inference requests together, with no programming effort from the user.
With Automatic Batching, gathering the input and scattering the output from the individual inference requests required for the batch happen transparently, without affecting the application code. 

This article provides a preview of the new Automatic Batching function, including how it works, its configurations, and testing performance.

## Enabling/Disabling Automatic Batching


Auto-batching primarily targets the existing code written for inferencing many requests, each instance with the batch size 1. To obtain corresponding performance improvements, the application **must be running many inference requests simultaneously**. 
Auto-batching can also be used via a particular *virtual* device.       

Batching is a straightforward way of leveraging the compute power of GPU and saving on communication overheads. Automatic Batching is "implicitly" triggered on the GPU when `ov::hint::PerformanceMode::THROUGHPUT` is specified for the `ov::hint::performance_mode` property for the `compile_model` or `set_property` calls.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_auto_batching.cpp compile_model

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_auto_batching.py compile_model

@endsphinxtab

@endsphinxtabset


> **NOTE**: Auto-Batching can be disabled (for example, for the GPU device) to prevent being triggered by `ov::hint::PerformanceMode::THROUGHPUT`. To do that, set `ov::hint::allow_auto_batching` to **false** in addition to the `ov::hint::performance_mode`, as shown below:


@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_auto_batching.cpp compile_model_no_auto_batching

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_auto_batching.py compile_model_no_auto_batching

@endsphinxtab

@endsphinxtabset


To enable Auto-batching in the legacy apps not akin to the notion of performance hints, you need to use the **explicit** device notion, such as `BATCH:GPU`.

## Automatic Batch Size Selection

In both the THROUGHPUT hint and the explicit BATCH device cases, the optimal batch size is selected automatically, as the implementation queries the `ov::optimal_batch_size` property from the device and passes the model graph as the parameter. The actual value depends on the model and device specifics, for example, the on-device memory for dGPUs.
The support for Auto-batching is not limited to GPU. However, if a device does not support `ov::optimal_batch_size` yet, to work with Auto-batching, an explicit batch size must be specified, e.g., `BATCH:<device>(16)`.

This "automatic batch size selection" works on the presumption that the application queries `ov::optimal_number_of_infer_requests` to create the requests of the returned number and run them simultaneously:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_auto_batching.cpp query_optimal_num_requests

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_auto_batching.py query_optimal_num_requests

@endsphinxtab

@endsphinxtabset


### Optimizing Performance by Limiting Batch Size

If not enough inputs were collected, the `timeout` value makes the transparent execution fall back to the execution of individual requests. This value can be configured via the `AUTO_BATCH_TIMEOUT` property.
The timeout, which adds itself to the execution time of the requests, heavily penalizes the performance. To avoid this, when your parallel slack is bounded, provide OpenVINO with an additional hint.

For example, when the application processes only 4 video streams, there is no need to use a batch larger than 4. The most future-proof way to communicate the limitations on the parallelism is to equip the performance hint with the optional `ov::hint::num_requests` configuration key set to 4. This will limit the batch size for the GPU and the number of inference streams for the CPU, hence each device uses `ov::hint::num_requests` while converting the hint to the actual device configuration options:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_auto_batching.cpp hint_num_requests

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_auto_batching.py hint_num_requests

@endsphinxtab

@endsphinxtabset


For the *explicit* usage, you can limit the batch size by using `BATCH:GPU(4)`, where 4 is the number of requests running in parallel.

## Other Performance Considerations

To achieve the best performance with Automatic Batching, the application should:
 - Operate inference requests of the number that represents the multiple of the batch size. In the example above, for batch size 4, the application should operate 4, 8, 12, 16, etc. requests.
 - Use the requests that are grouped by the batch size together. For example, the first 4 requests are inferred, while the second group of the requests is being populated. Essentially, Automatic Batching shifts the asynchronicity from the individual requests to the groups of requests that constitute the batches.
  - Balance the `timeout` value vs. the batch size. For example, in many cases, having a smaller `timeout` value/batch size may yield better performance than having a larger batch size with a `timeout` value that is not large enough to accommodate the full number of the required requests.
  - When Automatic Batching is enabled, the `timeout` property of `ov::CompiledModel` can be changed anytime, even after the loading/compilation of the model. For example, setting the value to 0 disables Auto-batching effectively, as the collection of requests would be omitted.
  - Carefully apply Auto-batching to the pipelines. For example, in the conventional "video-sources -> detection -> classification" flow, it is most beneficial to do Auto-batching over the inputs to the detection stage. The resulting number of detections is usually fluent, which makes Auto-batching less applicable for the classification stage.

The following are limitations of the current implementations:
- Although it is less critical for the throughput-oriented scenarios, the load time with Auto-batching increases by almost double.
 - Certain networks are not safely reshapable by the "batching" dimension (specified as `N` in the layout terms). Besides, if the batching dimension is not zeroth, Auto-batching will not be triggered "implicitly" by the throughput hint.
 -  The "explicit" notion, for example, `BATCH:GPU`, using the relaxed dimensions tracking, often makes Auto-batching possible. For example, this method unlocks most **detection networks**.
 - When *forcing* Auto-batching via the "explicit" device notion, make sure that you validate the results for correctness.   
 - Performance improvements happen at the cost of the growth of memory footprint. However, Auto-batching queries the available memory (especially for dGPU) and limits the selected batch size accordingly.

 
## Configuring Automatic Batching
Following the OpenVINO naming convention, the *batching* device is assigned the label of *BATCH*. The configuration options are as follows:

| Parameter name     | Parameter description      | Default            |             Examples                                                      |
| :---               | :---                  | :---               |:-----------------------------------------------------------------------------|
| `AUTO_BATCH_DEVICE` | The name of the device to apply Automatic batching,  with the optional batch size value in brackets. | N/A | `BATCH:GPU` triggers the automatic batch size selection. `BATCH:GPU(4)` directly specifies the batch size.     |
| `AUTO_BATCH_TIMEOUT` | The timeout value, in ms. | 1000 |  You can reduce the timeout value to avoid performance penalty when the data arrives too unevenly). For example, set it to "100", or the contrary, i.e., make it large enough to accommodate input preparation (e.g. when it is a serial process).     |

## Testing Performance with Benchmark_app
The `benchmark_app` sample, that has both [C++](../../samples/cpp/benchmark_app/README.md) and [Python](../../tools/benchmark_tool/README.md) versions, is the best way to evaluate the performance of Automatic Batching:
 -  The most straightforward way is using the performance hints:
    - benchmark_app **-hint tput** -d GPU -m 'path to your favorite model'
 -  You can also use the "explicit" device notion to override the strict rules of the implicit reshaping by the batch dimension:
    - benchmark_app **-hint none -d BATCH:GPU** -m 'path to your favorite model'
 -  or override the automatically deduced batch size as well:
    - $benchmark_app -hint none -d **BATCH:GPU(16)** -m 'path to your favorite model'
    - This example also applies to CPU or any other device that generally supports batch execution.
    - Note that some shell versions (e.g. `bash`) may require adding quotes around complex device names, i.e. `-d "BATCH:GPU(16)"` in this example.

Note that Benchmark_app performs a warm-up run of a _single_ request. As Auto-Batching requires significantly more requests to execute in batch, this warm-up run hits the default timeout value (1000 ms), as reported in the following example:

### Additional Resources
[Supported Devices](supported_plugins/Supported_Devices.md)
